from __future__ import annotations

import argparse
import json
import os
import select
import shlex
import subprocess
import sys
import tempfile
import termios
import time
import tomllib
import tty
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path
from secrets import randbits
from typing import Optional
from uuid import UUID

from rich.console import Console, Group
from rich.live import Live
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from . import inference, launcher
from .client import Client, Command, HillclimbError, Json, Run, launch
from .logs import (
    ExitCode,
    PreflightError,
    exit_code_for_status,
    open_run_client,
    stream_logs,
)
from .mcp_server import collect_capabilities, serve_stdio
from .skill_template import TEMPLATE_NAMES, init_project, install_skill
from .visuals import write_visuals

__all__ = ["main"]

TERMINAL = frozenset({"cancelled", "completed", "failed", "in_doubt"})
VISUAL_INTERVAL = 1.0
STATUS_INTERVAL = 0.20
COMPOSER_LIMIT = 4096


def _is_plain(arguments: argparse.Namespace) -> bool:
    return bool(getattr(arguments, "plain", False)) or not sys.stdout.isatty()


def _is_json(arguments: argparse.Namespace) -> bool:
    return bool(getattr(arguments, "json", False))


def _read_intervention(path: Optional[Path]) -> Optional[Json]:
    if path is None:
        return None
    if not path.is_file():
        raise PreflightError(f"Intervention file not found: {path}")
    try:
        with path.open("rb") as stream:
            arm = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise PreflightError(f"Invalid intervention TOML {path}: {error}") from error
    gepa = arm.get("gepa", {})
    if "guidance_toml" in gepa:
        if "guidance" in gepa:
            raise PreflightError("specify either guidance or guidance_toml")
        guidance_path = path.parent / gepa.pop("guidance_toml")
        try:
            with guidance_path.open("rb") as stream:
                gepa["guidance"] = tomllib.load(stream)
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise PreflightError(f"Invalid guidance TOML {guidance_path}: {error}") from error
    return arm


def _read_campaign_config(path: Path) -> dict:
    if not path.is_file():
        raise PreflightError(f"Campaign configuration file not found: {path}")
    try:
        return launcher.read_config(path)
    except (OSError, ValueError, tomllib.TOMLDecodeError, HillclimbError) as error:
        raise PreflightError(f"Invalid campaign configuration {path}: {error}") from error


@contextmanager
def _terminal_input() -> Iterator[Optional[int]]:
    if not sys.stdin.isatty():
        yield None
        return
    descriptor = sys.stdin.fileno()
    settings = termios.tcgetattr(descriptor)
    tty.setcbreak(descriptor)
    try:
        yield descriptor
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, settings)


class Composer:
    def __init__(self, descriptor: Optional[int]) -> None:
        self._descriptor = descriptor
        self._line = bytearray()

    @property
    def text(self) -> str:
        return self._line.decode("utf-8", errors="replace")

    def ready(self) -> Iterator[str]:
        descriptor = self._descriptor
        if descriptor is None:
            return
        while select.select([descriptor], [], [], 0)[0]:
            fragment = os.read(descriptor, 1)
            if fragment in {b"", b"\x04"}:
                yield "/detach"
                return
            if fragment in {b"\n", b"\r"}:
                line = self.text.strip()
                self._line.clear()
                if line:
                    yield line
            elif fragment in {b"\x7f", b"\x08"}:
                if self._line:
                    previous = self.text[:-1]
                    self._line[:] = previous.encode()
            elif fragment >= b" " and len(self._line) < COMPOSER_LIMIT:
                self._line.extend(fragment)


def _compose(client: Client, line: str) -> str:
    name, _, payload = line.lstrip("/").partition(" ")
    if name == "inject":
        name = "inject_candidate"
    if name in client.status().get("optimizer_commands", {}):
        receipt = client.command(
            Command.ALGO, value={name: json.loads(payload or "{}")}, enqueue=True
        )
        return f"Queued {receipt['op_id']} at event {receipt['admitted_seq']}"
    if name in {"steer_proposer", "inject_candidate", "pin_or_ban_lever"}:
        receipt = client.command(Command(name), value=json.loads(payload), enqueue=True)
        return f"Queued {receipt['op_id']} at event {receipt['admitted_seq']}"
    words = shlex.split(line)
    if not words:
        raise HillclimbError("empty command")
    command = words[0].removeprefix("/")
    if command == "slice":
        proposal = len(words) == 3 and words[1] == "--proposal"
        if len(words) != 2 and not proposal:
            raise HillclimbError("/slice requires a destination")
        child = client.slice(Path(words[-1]), proposal=proposal)
        return f"Slice {child['run_id']}"
    control = Command(command)
    argument = " ".join(words[1:]) if len(words) > 1 else None
    if control is Command.CAP and argument is not None:
        argument = int(argument)
    receipt = client.command(control, value=argument, enqueue=True)
    return f"Queued {receipt['op_id']} at event {receipt['admitted_seq']}"


def _rails(
    status: Json,
    seq: int,
    seconds: float,
    rollout_start: int,
    notice: str,
    composer: str,
) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Run", Text(str(status["run_id"])))
    table.add_row(
        "Campaign",
        Text(
            f"{status['algorithm_id']} · {status['status']} · epoch {status['epoch']}"
        ),
    )
    table.add_row(
        "Progress",
        Text(f"step {status['step']} · event {seq} · attached {seconds:.1f}s"),
    )

    rollouts = int(status["rollouts_completed"])
    rate = (rollouts - rollout_start) / seconds if seconds > 0 else 0
    table.add_row(
        "Rollouts", Text(f"{rollouts} completed · {rate:.1f}/s since attachment")
    )
    spent = int(status["spent_micros"])
    cap = int(status["cap_micros"])
    budget = Text(f"${spent / 1_000_000:.6f} / ${cap / 1_000_000:.6f}")
    if cap > 0:
        table.add_row(
            "Budget", Group(budget, ProgressBar(total=cap, completed=spent, width=28))
        )
    else:
        table.add_row("Budget", budget)
    if notice:
        table.add_row("Command", Text(notice))
    if composer:
        table.add_row("Input", Text(composer))
    if "daemon_error" in status:
        table.add_row("Daemon", Text(str(status["daemon_error"]), style="red"))
    return table


def _follow(run: Run, *, after_seq: int, output: Optional[Path] = None) -> None:
    console = Console()
    with (
        Client(run) as client,
        _terminal_input() as descriptor,
        Live(console=console) as live,
    ):
        composer = Composer(descriptor)
        cursor = after_seq
        status = client.status()
        rollout_start = int(status["rollouts_completed"])
        started = time.monotonic()
        status_at = 0.0
        visual_at = 0.0
        notice = ""
        live.update(
            _rails(status, cursor, 0.0, rollout_start, notice, composer.text),
            refresh=True,
        )
        for event in client.event_frames(after_seq=cursor):
            if event is not None:
                cursor = event.seq
            redraw = False
            for line in composer.ready():
                redraw = True
                if line == "/detach":
                    return
                try:
                    notice = _compose(client, line)
                except (HillclimbError, ValueError) as error:
                    notice = str(error)
            now = time.monotonic()
            if now - status_at >= STATUS_INTERVAL:
                status = client.status()
                status_at = now
                redraw = True
            terminal = str(status["status"]) in TERMINAL or "daemon_error" in status
            drained = terminal and cursor >= int(status["event_seq"])
            if output is not None and (now - visual_at >= VISUAL_INTERVAL or drained):
                write_visuals(
                    output, client.visuals(), algorithm=str(status["algorithm_id"])
                )
                visual_at = now
            if redraw or drained:
                live.update(
                    _rails(
                        status,
                        cursor,
                        now - started,
                        rollout_start,
                        notice,
                        composer.text,
                    ),
                    refresh=True,
                )
            if drained:
                if "daemon_error" in status:
                    raise HillclimbError(str(status["daemon_error"]))
                return


def _headless(
    run: Run,
    after_seq: int,
    *,
    json_mode: bool = False,
    plain_mode: bool = False,
) -> Mapping[str, object]:
    from tqdm import tqdm

    if plain_mode and not json_mode:
        with Client(run) as client:
            return stream_logs(client, follow=True, json_mode=False, after_seq=after_seq)

    cursor = after_seq
    emit_json = json_mode or not sys.stdout.isatty()
    with (
        Client(run) as client,
        tqdm(unit="rollout", disable=emit_json or plain_mode or not sys.stdout.isatty()) as progress,
    ):
        status = client.status()
        for event in client.event_frames(after_seq=after_seq):
            if event is not None:
                cursor = event.seq
                if emit_json:
                    _result(event.envelope)
            if event is None or event.envelope.get("kind") in {
                "settled",
                "cache_hit",
                "completed",
                "command",
                "in_doubt",
                "reservation_breached",
                "budget_refused",
            }:
                status = client.status()
                progress.update(max(0, status["rollouts_completed"] - progress.n))
                progress.set_postfix_str(status["status"], refresh=False)
            if "daemon_error" in status:
                raise HillclimbError(status["daemon_error"])
            if (
                status["status"] in TERMINAL
                or exit_code_for_status(status) == int(ExitCode.CAP_REACHED)
            ) and cursor >= status["event_seq"]:
                return status
        return status


def _attach(
    run: Run,
    after_seq: int,
    headless: bool,
    *,
    json_mode: bool = False,
    plain_mode: bool = False,
) -> Optional[Mapping[str, object]]:
    if not headless and not json_mode and not plain_mode and sys.stdin.isatty() and sys.stdout.isatty():
        if sys.version_info < (3, 12):
            raise HillclimbError(
                "Temper TUI requires Python 3.12 or newer. Use --headless on Python 3.11"
            )
        from .tui import attach

        attach(run, after_seq)
        with Client(run) as client:
            return client.status()
    return _headless(run, after_seq, json_mode=json_mode, plain_mode=plain_mode)


def _output_parent() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="Emit machine-readable JSON")
    parent.add_argument("--plain", action="store_true", default=argparse.SUPPRESS, help="Disable interactive TTY UI")
    return parent


def _parser() -> argparse.ArgumentParser:
    output = _output_parent()
    parser = argparse.ArgumentParser(prog="hillclimb")
    parser.add_argument(
        "--daemon", default=launcher.default_daemon()
    )
    parser.add_argument("--json", action="store_true", default=False, help="Emit machine-readable JSON")
    parser.add_argument("--plain", action="store_true", default=False, help="Disable interactive TTY UI")
    commands = parser.add_subparsers(dest="operation", required=True)

    inference.add_arguments(commands.add_parser("inference", help="Local inference providers"))

    update = commands.add_parser("update", parents=[output], help="Refresh the CLI from a local checkout")
    update.add_argument("--source", type=Path)
    providers = commands.add_parser("providers", parents=[output], help="Show local inference providers")
    providers.add_argument("--env-file", type=Path)
    providers.add_argument("--devkit", type=Path)
    commands.add_parser("algorithms", parents=[output], help="List supported optimizers")
    containers = commands.add_parser("containers", parents=[output], help="List configured container campaigns")
    containers.add_argument("--config-dir", type=Path)

    capabilities_parser = commands.add_parser(
        "capabilities", parents=[output], help="Show supported daemon, adapter, and MCP capabilities"
    )
    capabilities_parser.add_argument("--url", type=str, help="Probe an HTTP daemon URL")
    capabilities_parser.add_argument("--auth-token-env", type=str, help="Bearer token env var for remote URL")

    init_parser = commands.add_parser(
        "init", parents=[output], help="Initialize a starter Hillclimb project and optimizer.toml"
    )
    init_parser.add_argument(
        "template",
        nargs="?",
        default="python",
        help=f"Starter template ({', '.join(TEMPLATE_NAMES)})",
    )
    init_parser.add_argument("--dir", type=Path, default=Path("."), help="Destination directory")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing starter files")

    skill_parser = commands.add_parser("skill", parents=[output], help="Manage agent skill files")
    skill_commands = skill_parser.add_subparsers(dest="skill_action", required=True)
    skill_install = skill_commands.add_parser(
        "install", parents=[output], help="Install SKILL.md for coding agents"
    )
    skill_install.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Target skill directory (default .agents/skills/hillclimb)",
    )
    skill_install.add_argument(
        "--claude",
        action="store_true",
        help="Install under .claude/skills/hillclimb when --dir is omitted",
    )
    skill_install.add_argument("--force", action="store_true", help="Overwrite existing SKILL.md")

    logs_parser = commands.add_parser(
        "logs", parents=[output], help="Stream human-readable or NDJSON campaign logs"
    )
    logs_parser.add_argument("run_id", type=UUID, nargs="?")
    logs_parser.add_argument("-f", "--follow", action="store_true", help="Follow live SSE events until terminal state")
    logs_parser.add_argument("--after-seq", type=int, default=0, help="Replay events strictly after sequence number")
    logs_parser.add_argument("--directory", type=Path)
    logs_parser.add_argument("--url", type=str, help="Daemon HTTP URL for TCP/cloud campaigns")
    logs_parser.add_argument("--auth-token-env", type=str, help="Bearer token env var for remote URL")

    mcp_parser = commands.add_parser(
        "mcp", parents=[output], help="Run the zero-dependency stdio JSON-RPC 2.0 MCP server"
    )
    mcp_parser.add_argument("--run-id", type=UUID, help="Attach default run UUID")
    mcp_parser.add_argument("--directory", type=Path, help="Run directory for --run-id")
    mcp_parser.add_argument("--url", type=str, help="Daemon HTTP URL for TCP/cloud campaigns")
    mcp_parser.add_argument("--auth-token-env", type=str, help="Bearer token env var for remote URL")

    for name in ("start", "run", "eval"):
        options = {"help": "Pick or start a campaign"} if name == "start" else {}
        run = commands.add_parser(name, parents=[output], **options)
        source = run.add_mutually_exclusive_group(required=name != "start")
        source.add_argument("config_file", nargs="?", type=Path)
        source.add_argument("--synthetic", action="store_true")
        source.add_argument("--config", type=Path)
        run.add_argument("--headless", action="store_true")
        if name == "start":
            source.add_argument("--container", help="Start a configured container by ID")
            run.add_argument("--config-dir", type=Path)
        if name != "eval":
            run.add_argument("--algorithm", choices=launcher.ALGORITHM_CHOICES)
        else:
            run.set_defaults(algorithm=None)
        run.add_argument("--iterations", "--trials", dest="trials", type=int)
        run.add_argument("--seed", type=int)
        run.add_argument("--run-id", type=UUID)
        run.add_argument("--directory", type=Path)
        run.add_argument("--detach", action="store_true")
        run.add_argument("--tick-millis", type=int)

    controls = {
        "pause",
        "resume",
        "cancel",
        "steer",
        "cap",
        "edit-runtime",
        "revert-runtime",
        "reconcile",
        "confirm-price",
    }
    for name in (
        "attach",
        "board",
        "send",
        "status",
        "slice",
        "observe",
        *sorted(controls),
    ):
        operation = commands.add_parser(name, parents=[output])
        operation.add_argument(
            "run_id", type=UUID, nargs="?" if name == "attach" else None
        )
        if name == "attach":
            operation.add_argument("--headless", action="store_true")
        if name == "board":
            render = operation.add_mutually_exclusive_group()
            render.add_argument("--watch", action="store_true")
            render.add_argument("--png", type=Path)
        if name == "send":
            operation.add_argument("command")
            operation.add_argument("payload", nargs="?")
            operation.add_argument("--op-id", type=UUID)
            operation.add_argument("--expected-epoch", type=int)
        operation.add_argument("--directory", type=Path)
        if name in {"attach", "observe"}:
            operation.add_argument("--after-seq", type=int, default=0)
        if name == "resume":
            operation.add_argument("--detach", action="store_true")
            operation.add_argument("--tick-millis", type=int)
            operation.add_argument("--live", action="store_true")
        if name == "observe":
            operation.add_argument("--output", type=Path)
        if name == "slice":
            operation.add_argument("--destination", type=Path, required=True)
            operation.add_argument("--proposal", action="store_true")
            operation.add_argument("--intervention", type=Path)
        if name == "status":
            operation.add_argument("--offline", action="store_true")
            operation.add_argument("--url", type=str)
            operation.add_argument("--auth-token-env", type=str)
        if name in controls:
            operation.add_argument("--op-id", type=UUID)
            operation.add_argument("--expected-epoch", type=int)
        if name == "steer":
            operation.add_argument("direction")
        if name == "cap":
            operation.add_argument("micros", type=int)
        if name in {"edit-runtime", "revert-runtime"}:
            operation.add_argument("--expected-revision", required=True)
        if name == "edit-runtime":
            operation.add_argument("--path", required=True)
            operation.add_argument("--content-file", type=Path, required=True)
            operation.add_argument("--rationale", required=True)
        if name == "revert-runtime":
            operation.add_argument("--target-revision", required=True)
        if name in {"reconcile", "confirm-price"}:
            operation.add_argument("--proof", required=True)
        if name == "confirm-price":
            operation.add_argument("--effect-id", required=True)

    return parser


def _result(result: Json) -> None:
    print(json.dumps(result, sort_keys=True), flush=True)


def _new_run_id() -> UUID:
    timestamp = time.time_ns() // 1_000_000
    return UUID(
        int=(timestamp << 80)
        | (7 << 76)
        | (randbits(12) << 64)
        | (2 << 62)
        | randbits(62)
    )


def _control_value(
    command: Command, arguments: argparse.Namespace
) -> Optional[str | int | Json]:
    if command is Command.STEER:
        return arguments.direction
    if command is Command.CAP:
        return arguments.micros
    if command is Command.EDIT_RUNTIME:
        return {
            "expected_revision": arguments.expected_revision,
            "path": arguments.path,
            "content": arguments.content_file.read_text(encoding="utf-8"),
            "rationale": arguments.rationale,
        }
    if command is Command.REVERT_RUNTIME:
        return {
            "expected": arguments.expected_revision,
            "target": arguments.target_revision,
        }
    if command is Command.RECONCILE:
        return {"proof": arguments.proof}
    if command is Command.CONFIRM_PRICE:
        return {"effect_id": arguments.effect_id, "proof": arguments.proof}
    return None


def _discover_single_run() -> UUID:
    sockets = Run.locate(_new_run_id()).socket.parent
    live = []
    for socket in sorted(sockets.glob("*.sock")):
        candidate = Run.locate(UUID(socket.stem))
        try:
            with Client(candidate, timeout=0.5) as client:
                if client.status()["status"] in {"paused", "running", "in_doubt"}:
                    live.append(candidate.run_id)
        except HillclimbError:
            continue
    if len(live) != 1:
        raise PreflightError(f"Specify a run ID ({len(live)} attachable runs)")
    return live[0]


def _select_start(arguments: argparse.Namespace) -> bool:
    algorithm = launcher.canonical_algorithm(arguments.algorithm) if arguments.algorithm else None
    config = arguments.config or arguments.config_file
    contents = None
    target = None
    if config is None and not arguments.synthetic:
        if getattr(arguments, "plain", False) and not arguments.container:
            raise PreflightError(
                "Use hillclimb start with a campaign file or --container in non-interactive mode"
            )
        targets = launcher.containers(Path.cwd(), arguments.config_dir)
        if arguments.container:
            target = next((item for item in targets if item.id == arguments.container), None)
            if target is None:
                raise PreflightError(f"Unknown container {arguments.container!r}")
        else:
            try:
                selection = launcher.pick_campaign(targets, algorithm)
            except HillclimbError as error:
                if "outside a terminal" in str(error):
                    raise PreflightError(str(error)) from error
                raise
            if selection is None:
                return False
            target, algorithm, contents = selection
    if target is not None:
        if algorithm and target.algorithm != algorithm:
            raise PreflightError(f"Container algorithm is {target.algorithm!r}, expected {algorithm!r}")
        if target.id == "craftax-rs-gepa" and target.entrypoint is not None:
            arguments.config_contents = launcher.craftax_manifest(target, arguments)
            arguments.algorithm = None
            arguments.trials = None
            arguments.seed = None
            return True
        if target.entrypoint is not None:
            launcher.start_benchmark(target, arguments)
            return False
        config = target.config
    if config is not None:
        if contents is None:
            contents = _read_campaign_config(config)
        configured = contents.get("algorithm_id")
        if not isinstance(configured, str) or configured not in launcher.ALGORITHM_CHOICES:
            raise PreflightError("Campaign must select GEPA, MIPROv2, or GELO")
        configured = launcher.canonical_algorithm(configured)
        contents["algorithm_id"] = configured
        if algorithm and configured != algorithm:
            raise PreflightError(f"Campaign algorithm is {configured!r}, expected {algorithm!r}")
        if arguments.seed is not None or arguments.trials is not None:
            raise PreflightError("Set the seed and iteration limit in the campaign file")
        arguments.algorithm = None
    else:
        arguments.algorithm = algorithm
    arguments.config = config
    arguments.config_contents = contents
    return True


def _print_capabilities_human(payload: Mapping[str, object]) -> None:
    print(f"Hillclimb CLI {payload['cli_version']} | MCP {payload['protocol_version']}")
    algos = ", ".join(entry["id"] for entry in payload["algorithms"])  # type: ignore[index]
    print(f"Algorithms : {algos}")
    print(f"Templates  : {', '.join(payload['templates'])}")  # type: ignore[arg-type]
    print(f"Targets    : {', '.join(payload['targets'])}")  # type: ignore[arg-type]
    surfaces = payload["surfaces"]
    assert isinstance(surfaces, Mapping)
    surface_list = ", ".join(f"{name}={'yes' if enabled else 'no'}" for name, enabled in surfaces.items())
    print(f"Surfaces   : {surface_list}")
    daemon_info = payload["daemon"]
    assert isinstance(daemon_info, Mapping)
    print(f"Daemon     : {daemon_info['binary']} (executable={daemon_info['executable']})")


def _execute(arguments: argparse.Namespace) -> int:
    if arguments.operation == "inference":
        inference.execute(arguments)
        return int(ExitCode.COMPLETED)

    operation = arguments.operation
    json_mode = _is_json(arguments)
    plain_mode = _is_plain(arguments)

    if operation == "update":
        _result(launcher.update(arguments.source))
        return int(ExitCode.COMPLETED)
    if operation == "providers":
        _result(inference.provider_status(env_file=arguments.env_file, devkit=arguments.devkit))
        return int(ExitCode.COMPLETED)
    if operation == "algorithms":
        _result({"algorithms": launcher.ALGORITHMS})
        return int(ExitCode.COMPLETED)
    if operation == "containers":
        _result({"containers": [
            {"id": item.id, "kind": item.kind, "algorithm": item.algorithm,
             "tasks": item.tasks, "config": str(item.config) if item.config else None,
             "launcher": str(item.entrypoint) if item.entrypoint else None}
            for item in launcher.containers(Path.cwd(), arguments.config_dir)
        ]})
        return int(ExitCode.COMPLETED)
    if operation == "capabilities":
        payload = collect_capabilities(
            daemon=arguments.daemon,
            url=arguments.url,
            auth_token_env=arguments.auth_token_env,
        )
        if json_mode or not sys.stdout.isatty():
            _result(payload)
        else:
            _print_capabilities_human(payload)
        return int(ExitCode.COMPLETED)
    if operation == "init":
        result = init_project(arguments.template, arguments.dir, overwrite=arguments.force)
        if json_mode:
            _result(result.to_dict())
        else:
            print(f"Initialized {result.template.value} project in {result.directory}:")
            for file_path in result.files:
                print(f"  - {file_path}")
        return int(ExitCode.COMPLETED)
    if operation == "skill":
        if arguments.skill_action == "install":
            target_dir = arguments.dir
            if target_dir is None:
                target_dir = Path(".claude/skills/hillclimb" if arguments.claude else ".agents/skills/hillclimb")
            result = install_skill(target_dir, overwrite=arguments.force)
            if json_mode:
                _result(result.to_dict())
            else:
                print(f"Installed Hillclimb agent skill at {result.skill_path}")
            return int(ExitCode.COMPLETED)
        raise PreflightError(f"Unsupported skill action {arguments.skill_action!r}")
    if operation == "mcp":
        serve_stdio(
            daemon=arguments.daemon,
            url=arguments.url,
            auth_token_env=arguments.auth_token_env,
            run_id=arguments.run_id,
            directory=arguments.directory,
        )
        return int(ExitCode.COMPLETED)
    if operation == "logs":
        if arguments.after_seq < 0:
            raise PreflightError("--after-seq must be non-negative")
        run_id = arguments.run_id
        if arguments.url is None and run_id is None:
            run_id = _discover_single_run()
        if run_id is not None and run_id.version != 7:
            raise PreflightError("run ID must be a UUIDv7")
        with open_run_client(
            run_id=run_id,
            directory=arguments.directory,
            url=arguments.url,
            auth_token_env=arguments.auth_token_env,
        ) as client:
            final_status = stream_logs(
                client,
                follow=arguments.follow,
                json_mode=json_mode,
                after_seq=arguments.after_seq,
            )
        return exit_code_for_status(final_status)

    if operation == "start":
        if not _select_start(arguments):
            return int(ExitCode.COMPLETED)
        operation = "run"

    run_id = arguments.run_id if arguments.run_id is not None else _new_run_id()
    if run_id.version != 7:
        raise PreflightError("run ID must be a UUIDv7")
    if operation == "attach" and arguments.run_id is None:
        run_id = _discover_single_run()
    run = Run.locate(run_id, arguments.directory)
    if operation in {"run", "eval"}:
        config = arguments.config or arguments.config_file
        with ExitStack() as files:
            campaign = getattr(arguments, "config_contents", None)
            if campaign is not None or (config and config.suffix == ".toml"):
                run.directory.parent.mkdir(parents=True, exist_ok=True)
                temporary = files.enter_context(
                    tempfile.NamedTemporaryFile(mode="w+", dir=run.directory.parent)
                )
                if campaign is None:
                    assert config is not None
                    campaign = _read_campaign_config(config)
                if isinstance(campaign.get("runtime"), dict) and "craftax" in campaign["runtime"]:
                    arguments.daemon = launcher.craftax_daemon(arguments.daemon, launcher.local_source())
                campaign["run_id"] = str(run_id)
                temporary.write(json.dumps(campaign))
                temporary.flush()
                config = Path(temporary.name)
            elif config is not None and not config.is_file():
                raise PreflightError(f"Campaign configuration file not found: {config}")
            _result(
                launch(
                    run,
                    daemon=arguments.daemon,
                    mode=operation,
                    algorithm=arguments.algorithm,
                    trials=arguments.trials,
                    seed=arguments.seed,
                    config=config,
                    tick_millis=arguments.tick_millis,
                )
            )
        if not arguments.detach:
            final_status = _attach(
                run,
                0,
                getattr(arguments, "headless", False),
                json_mode=json_mode,
                plain_mode=plain_mode,
            )
            if final_status is not None:
                return exit_code_for_status(final_status)
        return int(ExitCode.COMPLETED)
    if operation == "resume" and not arguments.live:
        if (run.directory / "craftax").is_dir():
            arguments.daemon = launcher.craftax_daemon(arguments.daemon, launcher.local_source())
        _result(
            launch(
                run,
                daemon=arguments.daemon,
                mode=operation,
                tick_millis=arguments.tick_millis,
            )
        )
        if not arguments.detach:
            final_status = _attach(
                run,
                0,
                getattr(arguments, "headless", False),
                json_mode=json_mode,
                plain_mode=plain_mode,
            )
            if final_status is not None:
                return exit_code_for_status(final_status)
        return int(ExitCode.COMPLETED)
    if operation == "status" and arguments.offline:
        subprocess.run(
            [
                arguments.daemon,
                "status",
                "--directory",
                str(run.directory),
                "--run-id",
                str(run.run_id),
            ],
            check=True,
        )
        return int(ExitCode.COMPLETED)
    if operation == "attach":
        final_status = _attach(
            run,
            arguments.after_seq,
            arguments.headless,
            json_mode=json_mode,
            plain_mode=plain_mode,
        )
        if final_status is not None:
            return exit_code_for_status(final_status)
        return int(ExitCode.COMPLETED)
    if operation == "board":
        from .tui import board

        board(run, arguments.watch, arguments.png)
        return int(ExitCode.COMPLETED)
    if operation == "send":
        with Client(run) as client:
            name = arguments.command
            value = arguments.payload
            if name in client.status().get("optimizer_commands", {}):
                command, value = Command.ALGO, {name: json.loads(value or "{}")}
            else:
                command = Command(name)
                if value is not None and command != Command.STEER:
                    value = json.loads(value)
            _result(
                client.command(
                    command,
                    value=value,
                    op_id=arguments.op_id,
                    expected_epoch=arguments.expected_epoch,
                    enqueue=True,
                )
            )
        return int(ExitCode.COMPLETED)
    if operation == "observe":
        output = arguments.output
        if output is None:
            with Client(run) as client:
                output = (
                    Path(str(client.status()["directory"])) / "observer" / "visuals"
                )
        _follow(run, after_seq=arguments.after_seq, output=output)
        return int(ExitCode.COMPLETED)
    if operation == "status" and getattr(arguments, "url", None) is not None:
        with open_run_client(
            run_id=run_id,
            directory=arguments.directory,
            url=arguments.url,
            auth_token_env=arguments.auth_token_env,
        ) as client:
            _result(client.status())
        return int(ExitCode.COMPLETED)

    with Client(run) as client:
        if operation == "status":
            _result(client.status())
        elif operation == "slice":
            _result(
                client.slice(
                    arguments.destination,
                    proposal=arguments.proposal,
                    intervention=_read_intervention(arguments.intervention),
                )
            )
        else:
            command = Command(operation.replace("-", "_"))
            _result(
                client.command(
                    command,
                    value=_control_value(command, arguments),
                    op_id=arguments.op_id,
                    expected_epoch=arguments.expected_epoch,
                )
            )
    return int(ExitCode.COMPLETED)


def main() -> None:
    json_requested = "--json" in sys.argv[1:]
    try:
        code = _execute(_parser().parse_args())
        if code:
            raise SystemExit(code)
    except KeyboardInterrupt:
        raise SystemExit(130)
    except PreflightError as error:
        if json_requested:
            print(json.dumps({"error": str(error), "exit_code": int(ExitCode.PREFLIGHT_ERROR)}), file=sys.stderr)
        else:
            print(f"hillclimb: {error}", file=sys.stderr)
        raise SystemExit(int(ExitCode.PREFLIGHT_ERROR)) from error
    except (ValueError, tomllib.TOMLDecodeError) as error:
        if json_requested:
            print(json.dumps({"error": str(error), "exit_code": int(ExitCode.PREFLIGHT_ERROR)}), file=sys.stderr)
        else:
            print(f"hillclimb: {error}", file=sys.stderr)
        raise SystemExit(int(ExitCode.PREFLIGHT_ERROR)) from error
    except (
        HillclimbError,
        OSError,
        subprocess.CalledProcessError,
    ) as error:
        if json_requested:
            print(json.dumps({"error": str(error), "exit_code": 1}), file=sys.stderr)
        else:
            print(f"hillclimb: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
