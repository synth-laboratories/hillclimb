from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Optional, TypeAlias
from uuid import UUID, uuid4

import httpx

__all__ = ["Client", "Command", "Event", "HillclimbError", "Run", "launch"]

Json: TypeAlias = Mapping[str, object]
READ_TIMEOUT_SECONDS = 10.0
START_TIMEOUT_SECONDS = 15.0
START_PROBE_SECONDS = 0.5


class HillclimbError(RuntimeError):
    """The daemon rejected a request or could not be reached."""


class Command(StrEnum):
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"
    CAP = "cap"
    ALGO = "algo"
    STEER = "steer"
    STEER_PROPOSER = "steer_proposer"
    PIN_OR_BAN_LEVER = "pin_or_ban_lever"
    INJECT_CANDIDATE = "inject_candidate"
    EDIT_RUNTIME = "edit_runtime"
    REVERT_RUNTIME = "revert_runtime"
    RECONCILE = "reconcile"
    CONFIRM_PRICE = "confirm_price"


@dataclass(frozen=True, slots=True)
class Run:
    run_id: UUID
    directory: Path
    socket: Path

    @classmethod
    def locate(cls, run_id: UUID, directory: Optional[Path] = None) -> Run:
        home = Path(os.environ.get("HILLCLIMB_HOME", Path.home() / ".hillclimb"))
        run_directory = (
            directory if directory is not None else home / "runs" / str(run_id)
        )
        return cls(
            run_id,
            run_directory.resolve(),
            (home / "sockets" / f"{run_id}.sock").resolve(),
        )


@dataclass(frozen=True, slots=True)
class Event:
    seq: int
    envelope: Json


class Client:
    """An HTTP/1.1 connection to an existing local Hillclimb run."""

    def __init__(self, run: Run, *, timeout: float = READ_TIMEOUT_SECONDS) -> None:
        self._http = httpx.Client(
            base_url="http://localhost",
            transport=httpx.HTTPTransport(uds=str(run.socket)),
            timeout=timeout,
            trust_env=False,
        )

    def __enter__(self) -> Client:
        return self

    def __exit__(
        self,
        exception_type: Optional[type[BaseException]],
        exception: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self._http.close()

    def status(self) -> Json:
        return self._json("GET", "/v1/status")

    def visuals(self) -> Json:
        return self._json("GET", "/v1/visuals")

    def command(
        self,
        command: Command,
        *,
        value: Optional[str | int | Json] = None,
        op_id: Optional[UUID] = None,
        expected_epoch: Optional[int] = None,
        enqueue: bool = False,
    ) -> Json:
        """Omit the epoch to resolve the request at the writer's next barrier."""
        commands_with_value = {
            Command.ALGO,
            Command.CAP,
            Command.STEER,
            Command.STEER_PROPOSER,
            Command.INJECT_CANDIDATE,
            Command.PIN_OR_BAN_LEVER,
            Command.EDIT_RUNTIME,
            Command.REVERT_RUNTIME,
            Command.RECONCILE,
            Command.CONFIRM_PRICE,
        }
        if command in commands_with_value and value is None:
            raise HillclimbError(f"{command} requires a value")
        if command not in commands_with_value and value is not None:
            raise HillclimbError(f"{command} does not take a value")
        identifier = op_id if op_id is not None else uuid4()
        try:
            return self._json(
                "POST",
                "/v1/commands?enqueue=true" if enqueue else "/v1/commands",
                {
                    "schema": "hillclimb.command.v1",
                    "op_id": str(identifier),
                    "actor": "human",
                    "expected_epoch": expected_epoch,
                    "command": command.value
                    if value is None
                    else {command.value: value},
                },
            )
        except HillclimbError as error:
            scope = f" at epoch {expected_epoch}" if expected_epoch is not None else ""
            raise HillclimbError(f"operation {identifier}{scope}: {error}") from error

    def slice(self, directory: Path, *, proposal: bool = False, intervention: Optional[Json] = None) -> Json:
        return self._json(
            "POST",
            "/v1/slice",
            {"directory": str(directory.resolve()), "proposal": proposal, **({"intervention": intervention} if intervention is not None else {})},
        )

    def events(self, *, after_seq: int = 0) -> Iterator[Event]:
        for event in self.event_frames(after_seq=after_seq):
            if event is not None:
                yield event

    def event_frames(self, *, after_seq: int = 0) -> Iterator[Optional[Event]]:
        """Yield durable events and heartbeat frames, suitable for a terminal UI."""
        try:
            with self._http.stream(
                "GET", "/v1/events", headers={"Last-Event-ID": str(after_seq)}
            ) as response:
                if response.is_error:
                    response.read()
                    self._ensure_success(response)
                yield from _event_frames(response.iter_lines(), after_seq)
        except httpx.RequestError as error:
            raise HillclimbError(f"event stream disconnected: {error}") from error

    def _json(self, method: str, route: str, body: Optional[Json] = None) -> Json:
        try:
            response = self._http.request(method, route, json=body)
        except httpx.RequestError as error:
            raise HillclimbError(f"{method} {route}: {error}") from error
        self._ensure_success(response)
        return response.json()

    @staticmethod
    def _ensure_success(response: httpx.Response) -> None:
        if response.is_error:
            raise HillclimbError(f"HTTP {response.status_code}: {response.text}")


def _event_frames(lines: Iterator[str], after_seq: int) -> Iterator[Optional[Event]]:
    cursor = after_seq
    identifier = None
    data: list[str] = []
    for line in lines:
        if not line:
            if data and identifier is not None:
                if identifier <= cursor:
                    raise HillclimbError(
                        f"SSE sequence regressed from {cursor} to {identifier}"
                    )
                envelope = json.loads("\n".join(data))
                if envelope["seq"] != identifier:
                    raise HillclimbError(
                        f"SSE id {identifier} differs from event envelope"
                    )
                yield Event(identifier, envelope)
                cursor = identifier
            identifier = None
            data.clear()
        elif line.startswith(":"):
            yield None
        else:
            field, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if field == "id":
                identifier = int(value)
            elif field == "data":
                data.append(value)


def launch(
    run: Run,
    *,
    daemon: str,
    mode: str,
    algorithm: Optional[str] = None,
    trials: Optional[int] = None,
    seed: Optional[int] = None,
    config: Optional[Path] = None,
    tick_millis: Optional[int] = None,
) -> Json:
    """Launch a detached Rust daemon and check server readiness before returning."""
    run.socket.parent.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    run.socket.parent.mkdir(mode=0o700, exist_ok=True)
    arguments = [
        daemon,
        mode,
        "--run-id",
        str(run.run_id),
        "--directory",
        str(run.directory),
        "--socket",
        str(run.socket),
    ]
    if algorithm is not None:
        arguments.extend(("--algorithm", algorithm))
    if trials is not None:
        arguments.extend(("--trials", str(trials)))
    if seed is not None:
        arguments.extend(("--seed", str(seed)))
    if config is not None:
        arguments.extend(("--config", str(config.resolve())))
    if tick_millis is not None:
        arguments.extend(("--tick-millis", str(tick_millis)))
    log_path = run.socket.with_suffix(".log")
    try:
        descriptor = os.open(
            log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600
        )
        with os.fdopen(descriptor, "ab") as log:
            process = subprocess.Popen(
                arguments,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
                close_fds=True,
            )
    except OSError as error:
        raise HillclimbError(f"could not execute {daemon}: {error}") from error

    deadline = time.monotonic() + START_TIMEOUT_SECONDS
    with Client(run, timeout=START_PROBE_SECONDS) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise HillclimbError(
                    f"daemon exited with status {process.returncode}; see {log_path}"
                )
            try:
                status = client.status()
            except HillclimbError as error:
                if not isinstance(
                    error.__cause__, (httpx.ConnectError, httpx.TimeoutException)
                ):
                    raise
                time.sleep(0.05)
                continue
            if status["pid"] == process.pid:
                return status
            raise HillclimbError(
                f"run already has an active daemon (pid {status['pid']})"
            )
    raise HillclimbError(
        f"daemon pid {process.pid} readiness unconfirmed; see {log_path}"
    )
