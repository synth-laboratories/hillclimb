"""Local credentials for the file-backed inference providers."""

from __future__ import annotations

import argparse
import fcntl
from dataclasses import dataclass, field
from getpass import GetPassWarning, getpass
import json
import os
from pathlib import Path
import re
import stat
import sys
import warnings
from tempfile import NamedTemporaryFile
import webbrowser

import httpx


@dataclass(frozen=True)
class ProviderAuth:
    variable: str
    dashboard: str
    check_url: str


PROVIDERS = {
    "reflection": ProviderAuth(
        "REFLECTION_API_KEY",
        "https://platform.reflection.ai/api-keys",
        "https://api.reflection.ai/openai/v1/models",
    ),
    "morph": ProviderAuth(
        "MORPH_API_KEY",
        "https://www.morphllm.com/dashboard/api-keys",
        "https://api.morphllm.com/api/v1/account/settings",
    ),
}
SAVED_ENV = ".env.hillclimb"
_KEY = re.compile(r"[A-Za-z0-9._~+/=-]{1,4096}")


@dataclass(frozen=True)
class Credential:
    key: str = field(repr=False)
    source: str


def _key(value: object) -> str | None:
    if isinstance(value, str):
        value = value.strip()
        if _KEY.fullmatch(value) and value != "workshop-proxy":
            return value
    return None


def _read_private(path: Path) -> str | None:
    try:
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        with os.fdopen(os.open(path, flags), encoding="utf-8") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise ValueError(f"Credential file must be user-owned and private: {path}")
            if info.st_size > 65_536:
                raise ValueError(f"Credential file exceeds 64 KiB: {path}")
            return stream.read()
    except FileNotFoundError:
        return None


def _dotenv_key(text: str, variable: str) -> str | None:
    for line in text.splitlines():
        name, _, value = line.lstrip().removeprefix("export ").partition("=")
        if name.strip() == variable:
            return _key(value.strip().strip("\"'"))
    return None


def resolve_key(
    variable: str, *, env_file: Path | None = None, project: Path | None = None,
) -> Credential | None:
    if key := _key(os.getenv(variable)):
        return Credential(key, "environment")
    project = project or Path.cwd()
    if env_file is not None and env_file.expanduser().is_file():
        path = env_file.expanduser()
        if key := _dotenv_key(path.read_text(encoding="utf-8"), variable):
            return Credential(key, str(path))
    saved = project / SAVED_ENV
    if text := _read_private(saved):
        if key := _dotenv_key(text, variable):
            return Credential(key, str(saved))
    project_env = project / ".env"
    if project_env != env_file and project_env.is_file():
        if key := _dotenv_key(project_env.read_text(encoding="utf-8"), variable):
            return Credential(key, str(project_env))

    if variable == "REFLECTION_API_KEY":
        path = Path.home() / ".mirror" / "reflection_api_key"
        if key := _key(_read_private(path)):
            return Credential(key, "Mirror key file")
    if variable == "MORPH_API_KEY":
        path = Path.home() / ".config" / "morph" / "config.json"
        if text := _read_private(path):
            try:
                config = json.loads(text)
                key = _key(config["profiles"][config["activeProfile"]]["apiKey"])
            except (KeyError, TypeError, ValueError):
                raise ValueError("Morph CLI credential profile is invalid") from None
            if key:
                return Credential(key, "Morph CLI active profile")
    return None


def save_key(project: Path, variable: str, key: str) -> Path:
    if variable not in {provider.variable for provider in PROVIDERS.values()} or _key(key) != key:
        raise ValueError("Invalid provider API key")
    destination = project / SAVED_ENV
    lock_path = project / f"{SAVED_ENV}.lock"
    flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW
    with os.fdopen(os.open(lock_path, flags, 0o600), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.is_symlink():
            raise ValueError(f"Cannot replace a symlinked credential file: {destination}")
        content = _read_private(destination) or ""
        lines = [
            line for line in content.splitlines()
            if line.lstrip().removeprefix("export ").partition("=")[0].strip() != variable
        ]
        lines.append(f"{variable}={key}")
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=project,
                                    prefix=".env.hillclimb.", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write("\n".join(lines) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return destination


def check_key(provider: ProviderAuth, credential: Credential) -> None:
    try:
        with httpx.Client(timeout=15, trust_env=False, follow_redirects=False) as client:
            response = client.get(
                provider.check_url, headers={"Authorization": f"Bearer {credential.key}"}
            )
    except httpx.RequestError:
        raise ValueError("Provider credential check could not reach the service") from None
    if not response.is_success:
        raise ValueError(f"Provider credential check returned HTTP {response.status_code}")


def add_arguments(providers: argparse._SubParsersAction) -> None:
    for name in PROVIDERS:
        parser = providers.add_parser(name, help=f"Manage the {name} API key")
        parser.add_argument("--env-file", type=Path, help="Read keys from this dotenv file first")
        commands = parser.add_subparsers(dest="inference_command", required=True)
        login = commands.add_parser("login", help="Connect using a provider API key")
        login.add_argument("--new", action="store_true", help="Open the dashboard for a replacement key")
        login.add_argument("--no-browser", action="store_true", help="Print the key creation link")
        status = commands.add_parser("status", help="Show the active credential source")
        status.add_argument("--check", action="store_true", help="Verify access with a read-only API request")


def execute(arguments: argparse.Namespace) -> None:
    name = arguments.inference_provider
    provider = PROVIDERS[name]
    credential = resolve_key(provider.variable, env_file=arguments.env_file)
    if arguments.inference_command == "status":
        if credential is None:
            raise ValueError(f"{provider.variable} is missing. Run hillclimb inference {name} login.")
        if arguments.check:
            check_key(provider, credential)
        suffix = " (verified)" if arguments.check else ""
        print(f"{name}: {credential.source}{suffix}")
        return
    if credential is not None and not arguments.new:
        print(f"{name}: already configured via {credential.source}")
        return
    if not sys.stdin.isatty():
        raise ValueError(f"Use a terminal to enter a key from {provider.dashboard}")
    print(f"Create a {name} API key at {provider.dashboard}", file=sys.stderr)
    if not arguments.no_browser:
        webbrowser.open(provider.dashboard)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", GetPassWarning)
            key = _key(getpass(f"{provider.variable}: "))
    except (EOFError, GetPassWarning):
        raise ValueError("A terminal with hidden input is required") from None
    if key is None:
        raise ValueError("Enter a valid provider API key")
    path = save_key(Path.cwd(), provider.variable, key)
    print(f"{name}: saved to {path.name}")
    if credential and credential.source in ("environment", str(arguments.env_file)):
        print("The current environment or explicit dotenv file takes precedence.")
