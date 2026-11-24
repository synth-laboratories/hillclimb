"""Local inference provider status and ChatGPT plan access."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import provider_auth
from .client import HillclimbError


DEVKIT_REPO = "https://github.com/openai/sign-in-with-chatgpt-devkit.git"
DEVKIT_COMMIT = "0a36fefeb913055c8c7a1a29b63d82b96b2e841a"
ELECTRON_VERSION = "42.11.18"
DEFAULT_DEVKIT = Path.home() / "Library/Application Support/Hillclimb/siwc-devkit"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    providers = parser.add_subparsers(dest="inference_provider", required=True)
    provider_auth.add_arguments(providers)
    inventory = providers.add_parser("providers", help="Print provider status as JSON")
    server = providers.add_parser("serve", help="Serve provider status on a local Unix socket")
    server.add_argument("--socket", type=Path, help="Socket in a private directory")
    for command in (inventory, server):
        command.add_argument("--env-file", type=Path, help="Read keys from this dotenv file first")
        command.add_argument("--devkit", type=Path, help="Override the ChatGPT DevKit checkout")
    chatgpt = providers.add_parser("chatgpt", help="Use a ChatGPT plan on this Mac")
    chatgpt.add_argument(
        "--devkit", type=Path, default=os.environ.get("HILLCLIMB_CHATGPT_DEVKIT", DEFAULT_DEVKIT),
        help="built OpenAI Sign in with ChatGPT DevKit checkout",
    )
    commands = chatgpt.add_subparsers(dest="inference_command", required=True)
    commands.add_parser("setup", help="Install the pinned OpenAI DevKit and Electron")
    login = commands.add_parser("login", help="Connect in the browser using macOS Keychain")
    login.add_argument("--reconsent", action="store_true", help="Request plan usage consent again")
    commands.add_parser("status", help="Show the local connection")
    commands.add_parser("models", help="List model slugs for the connected account")
    commands.add_parser("logout", help="Disconnect the local account")
    respond = commands.add_parser("respond", help="Run a text prompt supplied on stdin")
    respond.add_argument("model", help="slug from the models command")
    respond.add_argument("--json", action="store_true", help="Read input and instructions as JSON")
    respond.add_argument("--stream", action="store_true", help="Write text as it arrives")


def provider_status(
    *, project: Path | None = None, env_file: Path | None = None, devkit: Path | None = None,
) -> dict:
    """Return local provider state without exposing credentials or account metadata."""
    if devkit is None:
        devkit = Path(os.environ.get("HILLCLIMB_CHATGPT_DEVKIT", DEFAULT_DEVKIT))
    devkit = devkit.expanduser().resolve()
    statuses = []
    variables = {"openai": "OPENAI_API_KEY"}
    variables.update({name: provider.variable for name, provider in provider_auth.PROVIDERS.items()})
    for name, variable in variables.items():
        try:
            configured = provider_auth.resolve_key(variable, project=project, env_file=env_file) is not None
            state = "configured" if configured else "not_configured"
        except (OSError, ValueError):
            configured = None
            state = "unavailable"
        statuses.append({
            "id": name, "auth": "api_key", "logged_in": configured,
            "ready": configured, "state": state,
        })
    state, logged_in, ready = _chatgpt_status(devkit)
    statuses.append({
        "id": "chatgpt", "auth": "oauth", "logged_in": logged_in,
        "ready": ready, "state": state,
    })
    return {"providers": statuses}


def _chatgpt_status(devkit: Path) -> tuple[str, bool | None, bool | None]:
    if sys.platform != "darwin":
        return "unsupported_platform", None, False
    try:
        command = _chatgpt_command(devkit, "status")
    except HillclimbError:
        return "not_installed", False, False
    try:
        result = subprocess.run(
            command, env=_chatgpt_environment(), stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=5, check=True,
        )
        status = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return "unavailable", None, None
    if not isinstance(status, dict):
        return "unavailable", None, None
    state = status.get("status")
    if state == "connected":
        if status.get("error"):
            return "unavailable", None, None
        if status.get("chatgpt_plan") is True:
            return "connected", True, True
        return "consent_required", True, False
    if state in ("disconnected", "reauth_required", "connecting"):
        return state, False, False
    return "unavailable", None, None


def execute(arguments: argparse.Namespace) -> None:
    if arguments.inference_provider in ("providers", "serve"):
        project = Path.cwd()

        def read_status() -> dict:
            return provider_status(project=project, env_file=arguments.env_file, devkit=arguments.devkit)

        if arguments.inference_provider == "providers":
            print(json.dumps(read_status(), indent=2))
        else:
            from .provider_status_server import serve_provider_status

            serve_provider_status(read_status, arguments.socket)
        return
    if arguments.inference_provider in provider_auth.PROVIDERS:
        provider_auth.execute(arguments)
        return
    if sys.platform != "darwin":
        raise HillclimbError("ChatGPT sign-in requires macOS Keychain")
    devkit = arguments.devkit.expanduser().resolve()
    runtime = devkit / ".hillclimb-electron/node_modules/electron"
    electron = runtime / "dist/Electron.app/Contents/MacOS/Electron"
    if arguments.inference_command == "setup":
        _setup(devkit, runtime, electron)
        return
    command = _chatgpt_command(devkit, arguments.inference_command)
    if arguments.inference_command == "login" and arguments.reconsent:
        command.append("--reconsent")
    if arguments.inference_command == "respond":
        command.append(arguments.model)
        if arguments.json:
            command.append("--json")
        if arguments.stream:
            command.append("--stream")
    status = subprocess.run(command, env=_chatgpt_environment(), check=False).returncode
    if status:
        raise SystemExit(status)


def _chatgpt_command(devkit: Path, action: str) -> list[str]:
    electron = devkit / ".hillclimb-electron/node_modules/electron/dist/Electron.app/Contents/MacOS/Electron"
    if not electron.is_file() or not (devkit / "packages/local/dist/index.js").is_file():
        raise HillclimbError("first run hillclimb inference chatgpt setup")
    return [str(electron), str(Path(__file__).with_name("chatgpt_host.cjs")), str(devkit), action]


def _chatgpt_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "ELECTRON_RUN_AS_NODE", "NODE_OPTIONS", "NODE_PATH", "OPENAI_LOG", "OPENAI_CUSTOM_HEADERS",
        "OPENAI_API_KEY", *(provider.variable for provider in provider_auth.PROVIDERS.values()),
    ):
        environment.pop(name, None)
    return environment


def _setup(devkit: Path, runtime: Path, electron: Path) -> None:
    if not devkit.exists():
        devkit.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", DEVKIT_REPO, str(devkit)], check=True)
        subprocess.run(["git", "checkout", "--detach", DEVKIT_COMMIT], cwd=devkit, check=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel", "HEAD"], cwd=devkit, text=True
    ).splitlines()
    if revision != [str(devkit), DEVKIT_COMMIT]:
        raise HillclimbError("choose an empty setup location or the pinned DevKit checkout")
    subprocess.run(
        ["npm", "ci", "-w", "@siwc/local", "--include-workspace-root", "--ignore-scripts"],
        cwd=devkit, check=True,
    )
    subprocess.run(["npm", "run", "build", "-w", "@siwc/local"], cwd=devkit, check=True)
    # The compiler needs its optional native binary. The runtime uses built-in fetch.
    subprocess.run(
        ["npm", "ci", "-w", "@siwc/local", "--omit=dev", "--omit=optional", "--ignore-scripts"],
        cwd=devkit, check=True,
    )
    runtime_project = runtime.parent.parent
    runtime_project.mkdir(exist_ok=True)
    subprocess.run(
        ["npm", "install", "--prefix", str(runtime_project), "--no-save",
         "--ignore-scripts", f"electron@{ELECTRON_VERSION}"],
        check=True,
    )
    subprocess.run(["node", str(runtime / "install.js")], cwd=runtime, check=True)
    if not electron.is_file():
        raise HillclimbError("Electron setup did not install its macOS executable")
    print(f"ChatGPT DevKit ready at {devkit}")
