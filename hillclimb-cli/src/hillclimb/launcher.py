"""Local CLI updates and campaign selection."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib
from urllib.parse import unquote, urlsplit

from .client import HillclimbError


ALGORITHMS = (
    {"id": "gepa", "name": "GEPA", "aliases": ()},
    {"id": "mipro", "name": "MIPROv2", "aliases": ("miprov2",)},
    {"id": "go-ex", "name": "GELO", "aliases": ("gelo",)},
)
ALGORITHM_CHOICES = tuple(value for entry in ALGORITHMS for value in (entry["id"], *entry["aliases"]))


def _is_checkout(source: Path) -> bool:
    return all((source / path).is_file() for path in (
        "hillclimb", "hillclimb-cli/pyproject.toml", "hillclimb-cli/uv.lock",
    ))


def local_source() -> Path | None:
    """Find the source bound to a checkout launcher or a UV local install."""
    if configured := os.environ.get("HILLCLIMB_SOURCE"):
        return Path(configured).expanduser().resolve()
    source = Path(__file__).resolve().parents[3]
    if _is_checkout(source):
        return source
    try:
        direct_url = distribution("synth-hillclimb").read_text("direct_url.json")
        if direct_url is not None:
            url = urlsplit(json.loads(direct_url)["url"])
            if url.scheme == "file" and url.netloc in {"", "localhost"}:
                candidate = Path(unquote(url.path)).resolve().parent
                if _is_checkout(candidate):
                    return candidate
    except (OSError, PackageNotFoundError, ValueError, KeyError, TypeError):
        pass
    return None


def default_daemon() -> str:
    if configured := os.environ.get("HILLCLIMB_DAEMON"):
        return configured
    source = local_source()
    if source is not None:
        daemon = source / ".hillclimb/bin/hillclimb-daemon"
        if daemon.is_file() and os.access(daemon, os.X_OK):
            return str(daemon)
    bundled = Path(__file__).resolve().parent / "native" / "hillclimb-daemon"
    if bundled.is_file() and os.access(bundled, os.X_OK):
        return str(bundled)
    return "hillclimb-daemon"


def canonical_algorithm(name: str) -> str:
    return next(entry["id"] for entry in ALGORITHMS if name in (entry["id"], *entry["aliases"]))


def read_config(path: Path) -> dict:
    if path.suffix == ".toml":
        with path.open("rb") as stream:
            config = tomllib.load(stream)
    else:
        config = json.loads(path.read_bytes())
    if not isinstance(config, dict):
        raise HillclimbError(f"Expected a campaign object in {path}")
    return config


@dataclass(frozen=True)
class Container:
    id: str
    kind: str
    algorithm: str
    tasks: int | None
    config: Path | None
    entrypoint: Path | None = None


def containers(project: Path, directory: Path | None = None) -> list[Container]:
    home = Path(os.environ.get("HILLCLIMB_HOME", Path.home() / ".hillclimb")).expanduser()
    if directory is not None:
        directory = directory.expanduser()
        if not directory.is_dir():
            raise HillclimbError(f"Container configuration directory does not exist: {directory}")
        paths = sorted(directory.glob("*"))
    else:
        paths = [project / "campaign.toml", project / "campaign.json"]
        for root in (project / ".hillclimb/containers", home / "containers"):
            paths.extend(sorted(root.glob("*")))
    result: dict[str, Container] = {}
    for path in paths:
        if path.suffix not in {".toml", ".json"} or not path.is_file() or path.stem in result:
            continue
        try:
            config = read_config(path)
        except (UnicodeError, ValueError):
            raise HillclimbError(f"Invalid campaign configuration: {path}") from None
        runtime = config.get("runtime")
        if not isinstance(runtime, dict):
            continue
        for kind in ("container", "harbor", "craftax"):
            if kind not in runtime:
                continue
            profile = runtime[kind]
            target = profile.get("target") if isinstance(profile, dict) else None
            row_field = "assignments" if kind == "craftax" else "tasks"
            tasks = target.get(row_field) if isinstance(target, dict) else None
            algorithm = config.get("algorithm_id")
            if isinstance(algorithm, str) and algorithm in ALGORITHM_CHOICES:
                algorithm = canonical_algorithm(algorithm)
            if (
                len(runtime) != 1 or not isinstance(tasks, dict) or not tasks
                or algorithm not in tuple(entry["id"] for entry in ALGORITHMS)
            ):
                raise HillclimbError(f"Invalid optimizer container campaign: {path}")
            result[path.stem] = Container(path.stem, kind, algorithm, len(tasks), path.resolve())
    source = local_source() if directory is None else None
    if source is not None:
        for name, algorithm in (
            ("banking77", "gepa"), ("craftax", "go-ex"),
            ("craftax-rs-gepa", "gepa"), ("healthbench2", "gepa"),
        ):
            entrypoint = source / name
            if name not in result and entrypoint.is_file() and os.access(entrypoint, os.X_OK):
                result[name] = Container(name, "benchmark", algorithm, None, None, entrypoint.resolve())
    return list(result.values())


def pick_campaign(
    targets: list[Container], algorithm: str | None,
) -> tuple[Container | None, str, dict | None] | None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise HillclimbError("Use hillclimb start with a campaign file or --container outside a terminal")
    if sys.version_info < (3, 12):
        raise HillclimbError("The campaign picker requires Python 3.12 or newer")
    from prompt_toolkit.shortcuts import radiolist_dialog, yes_no_dialog

    if algorithm is None:
        algorithm = radiolist_dialog(
            title="Hillclimb", text="Algorithm",
            values=[(entry["id"], entry["name"]) for entry in ALGORITHMS],
            ok_text="Next",
        ).run()
    if algorithm is None:
        return None
    compatible = {target.id: target for target in targets if target.algorithm == algorithm}
    choices = [("", "Synthetic · local")]
    choices.extend(
        (target.id, f"{target.id} · {target.kind}" +
         (f" · {target.tasks} tasks" if target.tasks is not None else ""))
        for target in compatible.values()
    )
    selected = radiolist_dialog(
        title="Hillclimb", text="Target", values=choices, ok_text="Select",
    ).run()
    if selected is None:
        return None
    target = compatible.get(selected)
    config = read_config(target.config) if target and target.config else None
    label = "Synthetic · $0 cap"
    if config is not None:
        cap = config.get("cap_micros")
        if type(cap) is not int or cap < 0:
            raise HillclimbError("Campaign cap must be a nonnegative integer")
        label = f"{target.id} · ${cap / 1_000_000:.6f} cap"
    elif target is not None:
        profile = {"craftax-rs-gepa": "$0 smoke", "healthbench2": "$0 replay"}
        label = f"{target.id} · {profile.get(target.id, 'Live benchmark')}"
    if not yes_no_dialog(
        title="Start campaign", text=f"{algorithm} · {label}", yes_text="Start", no_text="Cancel",
    ).run():
        return None
    return target, algorithm, config


def craftax_daemon(selected: str, source: Path | None) -> str:
    if source is None or os.environ.get("HILLCLIMB_DAEMON"):
        return selected
    generic = source / ".hillclimb/bin/hillclimb-daemon"
    native = generic.with_name("hillclimb-daemon-native-craftax")
    if selected in {"hillclimb-daemon", str(generic)} and native.is_file() and os.access(native, os.X_OK):
        return str(native)
    return selected


def craftax_manifest(target: Container, arguments) -> dict:
    command = [str(target.entrypoint), "--smoke", "--manifest"]
    if arguments.trials is not None:
        command.extend(("--rounds", str(arguments.trials)))
    if arguments.seed is not None:
        command.extend(("--seed", str(arguments.seed)))
    environment = os.environ.copy()
    arguments.daemon = craftax_daemon(arguments.daemon, target.entrypoint.parent)
    if arguments.daemon != "hillclimb-daemon":
        environment["HILLCLIMB_DAEMON"] = arguments.daemon
    manifest = json.loads(subprocess.check_output(command, env=environment))
    if manifest.get("algorithm_id") != "gepa" or "craftax" not in manifest.get("runtime", {}):
        raise HillclimbError("Craftax launcher returned an incompatible manifest")
    return manifest


def start_benchmark(target: Container, arguments) -> None:
    if arguments.detach or arguments.run_id or arguments.seed is not None or arguments.tick_millis is not None:
        raise HillclimbError("This benchmark accepts --headless, --iterations, and --directory")
    command = [str(target.entrypoint)]
    if arguments.headless:
        command.append("--plain")
    if arguments.trials is not None:
        command.extend(("--rounds", str(arguments.trials)))
    if arguments.directory is not None:
        command.extend(("--output", str(arguments.directory.resolve())))
    environment = os.environ.copy()
    if arguments.daemon != "hillclimb-daemon":
        environment["HILLCLIMB_DAEMON"] = arguments.daemon
    subprocess.run(command, env=environment, check=True)


def update(source: Path | None) -> dict:
    source = source.expanduser().resolve() if source is not None else local_source()
    if source is None or not _is_checkout(source):
        raise HillclimbError("Specify a local Hillclimb checkout with --source")
    cli = source / "hillclimb-cli"
    launcher = source / "hillclimb"
    install = source / ".hillclimb/bin/hillclimb"
    if install.exists() or install.is_symlink():
        if not install.is_symlink() or install.resolve() != launcher:
            raise HillclimbError(f"An unrelated launcher already exists at {install}")
    subprocess.run(["uv", "sync", "--locked", "--project", str(cli)], check=True, stdout=sys.stderr)
    install.parent.mkdir(parents=True, exist_ok=True)
    try:
        install.symlink_to(launcher)
    except FileExistsError:
        if not install.is_symlink() or install.resolve() != launcher:
            raise HillclimbError(f"An unrelated launcher already exists at {install}") from None
    return {"source": str(source), "launcher": str(install), "path_entry": str(install.parent)}
