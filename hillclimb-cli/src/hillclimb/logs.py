"""Human-readable live log formatter, SSE follower, and HTTP/UDS client helpers."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Optional, TextIO, Union
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx

from .client import (
    READ_TIMEOUT_SECONDS,
    Client,
    Command,
    Event,
    HillclimbError,
    Json,
    Run,
    _event_frames,
)

__all__ = [
    "ExitCode",
    "HttpClient",
    "PreflightError",
    "RunStatusKind",
    "exit_code_for_status",
    "format_event_line",
    "open_run_client",
    "stream_logs",
]

_TERMINAL_STATUSES = frozenset({"cancelled", "completed", "failed", "in_doubt", "cap_reached"})
_TERMINAL_EVENT_KINDS = frozenset({"completed", "cancelled", "in_doubt", "reservation_breached", "budget_refused", "cap_reached"})
_MILESTONE_STATUS_REFRESH_KINDS = frozenset(
    {
        "created",
        "generation_start",
        "frontier_accepted",
        "frontier_rejected",
        "minibatch_rejected",
        "budget_warning",
        "budget_refused",
        "paused",
        "in_doubt",
        "reservation_breached",
        "completed",
        "cancelled",
        "cap_reached",
    }
)
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class ExitCode(IntEnum):
    """Semantic process exit codes for CLI runs and log followers."""

    COMPLETED = 0
    CAP_REACHED = 2
    IN_DOUBT = 3
    PREFLIGHT_ERROR = 4


class RunStatusKind(StrEnum):
    """Canonical daemon campaign lifecycle statuses."""

    RUNNING = "running"
    PAUSED = "paused"
    IN_DOUBT = "in_doubt"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    CAP_REACHED = "cap_reached"


class PreflightError(HillclimbError):
    """Raised when CLI configuration, arguments, or preflight checks fail."""


@dataclass(frozen=True, slots=True)
class _EventEnvelope:
    seq: Optional[int]
    kind: str
    raw: Mapping[str, object]


def _parse_event(event: Union[Event, Mapping[str, object]]) -> _EventEnvelope:
    if isinstance(event, Event):
        raw = event.envelope
        assert isinstance(raw, Mapping), "Event.envelope must be a mapping"
        return _EventEnvelope(seq=event.seq, kind=str(raw.get("kind", "event")), raw=raw)
    assert isinstance(event, Mapping), "event must be an Event or mapping"
    raw_seq = event.get("seq")
    seq = int(raw_seq) if isinstance(raw_seq, int) and not isinstance(raw_seq, bool) else None
    return _EventEnvelope(seq=seq, kind=str(event.get("kind", "event")), raw=event)


def _format_usd(raw: Mapping[str, object], usd_key: str, micros_key: str) -> Optional[str]:
    usd = raw.get(usd_key)
    if isinstance(usd, (int, float)) and not isinstance(usd, bool):
        return f"${float(usd):.4f}"
    micros = raw.get(micros_key)
    if isinstance(micros, (int, float)) and not isinstance(micros, bool):
        return f"${float(micros) / 1_000_000:.4f}"
    return None


def _format_spend(raw: Mapping[str, object]) -> Optional[str]:
    spent = _format_usd(raw, "spent_usd", "spent_micros")
    cap = _format_usd(raw, "cap_usd", "cap_micros")
    if spent is not None and cap is not None:
        return f"{spent}/{cap}"
    return spent


def _format_score(value: object) -> Optional[str]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.4f}"
    return None


def _format_delta(value: object) -> Optional[str]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):+.4f}"
    return None


def _format_diff(raw: Mapping[str, object]) -> Optional[str]:
    diff = raw.get("prompt_diff")
    if diff is None:
        diff = raw.get("diff")
    if isinstance(diff, str) and diff.strip():
        lines = [line.rstrip() for line in diff.strip().splitlines()[:6]]
        return "  diff: " + " | ".join(lines)
    patch = raw.get("patch")
    if isinstance(patch, Mapping) and patch:
        parts = []
        for key, value in list(patch.items())[:3]:
            preview = str(value).replace("\n", " ")[:100]
            parts.append(f"{key}={preview!r}")
        return "  patch: " + ", ".join(parts)
    return None


def _short_id(value: object) -> str:
    text = str(value)
    if len(text) > 16:
        return text[:12] + "..."
    return text


def format_event_line(event: Union[Event, Mapping[str, object]]) -> str:
    """Format a durable campaign event as a concise human-readable log line."""
    parsed = _parse_event(event)
    raw = parsed.raw
    kind = parsed.kind
    seq_tag = f"#{parsed.seq}" if parsed.seq is not None else "#?"
    step = raw.get("step")
    epoch = raw.get("epoch")
    policy = raw.get("child") or raw.get("policy_id") or raw.get("champion") or raw.get("selected_policy")
    spend = _format_spend(raw)

    iteration = raw.get("iteration")
    if iteration is None:
        iteration = raw.get("round")
    if iteration is None:
        iteration = raw.get("generation")

    if kind == "created":
        parts = [f"=== Run {_short_id(raw.get('run_id', 'unknown'))}"]
        if raw.get("algorithm_id") is not None:
            parts.append(f"algo={raw['algorithm_id']}")
        if policy is not None:
            parts.append(f"seed={_short_id(policy)}")
        cap = _format_usd(raw, "cap_usd", "cap_micros")
        if cap is not None:
            parts.append(f"cap={cap}")
        return " | ".join(parts) + " ==="

    if kind == "generation_start":
        iter_label = f"{int(iteration):02d}" if isinstance(iteration, int) else "?"
        parts = [f"--- Iter {iter_label}"]
        if step is not None:
            parts.append(f"step={step}")
        parent = raw.get("parent") or policy
        if parent is not None:
            parts.append(f"parent={_short_id(parent)}")
        if spend is not None:
            parts.append(f"spend={spend}")
        return " | ".join(parts) + " ---"

    if kind in {"frontier_accepted", "frontier_rejected", "minibatch_rejected", "perfect_parent", "decision"}:
        decision = str(raw.get("decision") or kind)
        iter_label = f"Iter {int(iteration):02d}" if isinstance(iteration, int) else f"[{seq_tag}]"
        parts = [iter_label, decision]
        if policy is not None:
            parts.append(f"policy={_short_id(policy)}")
        parent_mb = _format_score(raw.get("parent_score"))
        child_mb = _format_score(raw.get("child_score"))
        mb = _format_score(raw.get("minibatch_score"))
        if parent_mb is not None and child_mb is not None:
            parts.append(f"minibatch={parent_mb}->{child_mb}")
        elif mb is not None:
            parts.append(f"minibatch={mb}")
        val = _format_score(raw.get("val_score") if raw.get("val_score") is not None else raw.get("validation_score"))
        if val is not None:
            parts.append(f"val={val}")
        delta = _format_delta(raw.get("delta") if raw.get("delta") is not None else raw.get("val_delta"))
        if delta is not None:
            parts.append(f"delta={delta}")
        if spend is not None:
            parts.append(f"spend={spend}")
        line = " | ".join(parts)
        diff_text = _format_diff(raw)
        return f"{line}\n{diff_text}" if diff_text else line

    if kind == "budget_warning":
        parts = [f"[{seq_tag}] WARN budget_warning"]
        if spend is not None:
            parts.append(f"spend={spend}")
        if raw.get("reason"):
            parts.append(str(raw["reason"]))
        return " | ".join(parts)

    if kind in {"budget_refused", "cap_reached"}:
        parts = [f"[{seq_tag}] CAP_REACHED {kind}"]
        if step is not None:
            parts.append(f"step={step}")
        if spend is not None:
            parts.append(f"spend={spend}")
        if raw.get("reason"):
            parts.append(f"reason={raw['reason']}")
        return " | ".join(parts)

    if kind in {"in_doubt", "reservation_breached"}:
        parts = [f"[{seq_tag}] IN_DOUBT {kind}"]
        if step is not None:
            parts.append(f"step={step}")
        if spend is not None:
            parts.append(f"spend={spend}")
        if raw.get("reason"):
            parts.append(f"reason={raw['reason']}")
        return " | ".join(parts)

    if kind == "paused":
        parts = [f"[{seq_tag}] PAUSED"]
        if step is not None:
            parts.append(f"step={step}")
        if raw.get("reason"):
            parts.append(f"reason={raw['reason']}")
        if spend is not None:
            parts.append(f"spend={spend}")
        return " | ".join(parts)

    if kind == "completed":
        parts = ["=== Completed"]
        if policy is not None:
            parts.append(f"champion={_short_id(policy)}")
        val = _format_score(raw.get("val_score") if raw.get("val_score") is not None else raw.get("dev"))
        if val is not None:
            parts.append(f"val={val}")
        heldout = _format_score(raw.get("heldout_score") if raw.get("heldout_score") is not None else raw.get("heldout"))
        if heldout is not None:
            parts.append(f"heldout={heldout}")
        if spend is not None:
            parts.append(f"spend={spend}")
        return " | ".join(parts) + " ==="

    if kind == "cancelled":
        parts = ["=== Cancelled"]
        if step is not None:
            parts.append(f"step={step}")
        if raw.get("reason"):
            parts.append(f"reason={raw['reason']}")
        if spend is not None:
            parts.append(f"spend={spend}")
        return " | ".join(parts) + " ==="

    if kind in {"command", "command_admitted", "command_refused"}:
        parts = [f"[{seq_tag}] {kind}"]
        if raw.get("causal_op_id"):
            parts.append(f"op={_short_id(raw['causal_op_id'])}")
        if epoch is not None:
            parts.append(f"epoch={epoch}")
        if raw.get("reason"):
            parts.append(f"reason={raw['reason']}")
        return " | ".join(parts)

    parts = [f"[{seq_tag}] {kind}"]
    if iteration is not None:
        parts.append(f"iter={iteration}")
    if step is not None:
        parts.append(f"step={step}")
    if policy is not None:
        parts.append(f"policy={_short_id(policy)}")
    mb = _format_score(raw.get("minibatch_score"))
    if mb is not None:
        parts.append(f"minibatch={mb}")
    val = _format_score(raw.get("val_score") if raw.get("val_score") is not None else raw.get("score"))
    if val is not None:
        parts.append(f"score={val}")
    delta = _format_delta(raw.get("delta") if raw.get("delta") is not None else raw.get("val_delta"))
    if delta is not None:
        parts.append(f"delta={delta}")
    if raw.get("decision"):
        parts.append(f"decision={raw['decision']}")
    if raw.get("rollouts_completed") is not None:
        parts.append(f"rollouts={raw['rollouts_completed']}")
    if spend is not None:
        parts.append(f"spend={spend}")
    if raw.get("status") is not None and kind not in {"settled", "cache_hit"}:
        parts.append(f"status={raw['status']}")
    line = " | ".join(parts)
    diff_text = _format_diff(raw)
    return f"{line}\n{diff_text}" if diff_text else line


def exit_code_for_status(status: Mapping[str, object]) -> int:
    """Map daemon run status metadata to a semantic CLI exit code."""
    raw_status = str(status.get("status", ""))
    reason = str(status.get("reason") or "").lower()
    unknown_cost = status.get("unknown_cost_ops")

    if raw_status == RunStatusKind.IN_DOUBT:
        return int(ExitCode.IN_DOUBT)
    if isinstance(unknown_cost, int) and unknown_cost > 0 and raw_status in _TERMINAL_STATUSES:
        return int(ExitCode.IN_DOUBT)
    if raw_status == RunStatusKind.CAP_REACHED:
        return int(ExitCode.CAP_REACHED)
    if raw_status == RunStatusKind.PAUSED and any(
        token in reason for token in ("budget", "cap", "request limit exhausted")
    ):
        return int(ExitCode.CAP_REACHED)
    if raw_status in {RunStatusKind.COMPLETED, RunStatusKind.RUNNING, RunStatusKind.PAUSED}:
        return int(ExitCode.COMPLETED)
    if raw_status in {RunStatusKind.CANCELLED, RunStatusKind.FAILED}:
        return 1
    return int(ExitCode.COMPLETED)


def _merge_status_context(
    envelope: Mapping[str, object],
    status: Mapping[str, object],
) -> dict[str, object]:
    merged = dict(envelope)
    for key in ("spent_micros", "cap_micros", "rollouts_completed"):
        if key not in merged and key in status:
            merged[key] = status[key]
    if merged.get("kind") == "completed" and not merged.get("policy_id") and status.get("frozen_policy"):
        merged["policy_id"] = status["frozen_policy"]
    return merged


def _is_cap_pause_status(status: Mapping[str, object]) -> bool:
    return exit_code_for_status(status) == int(ExitCode.CAP_REACHED)


def stream_logs(
    client: object,
    *,
    follow: bool = True,
    json_mode: bool = False,
    after_seq: int = 0,
    out: TextIO = sys.stdout,
) -> Mapping[str, object]:
    """Stream SSE events from `client.events()` as human-readable logs or NDJSON."""
    assert after_seq >= 0, "after_seq must be non-negative"
    initial_status = dict(client.status())  # type: ignore[attr-defined]
    latest_status = dict(initial_status)
    raw_target = initial_status.get("event_seq")
    target_seq = int(raw_target) if isinstance(raw_target, int) and not isinstance(raw_target, bool) else None

    if not follow and target_seq is not None and after_seq >= target_seq:
        return latest_status
    if (
        follow
        and target_seq is not None
        and after_seq >= target_seq
        and (
            str(latest_status.get("status", "")) in _TERMINAL_STATUSES
            or _is_cap_pause_status(latest_status)
        )
    ):
        return latest_status

    cursor = after_seq
    for event in client.events(after_seq=after_seq):  # type: ignore[attr-defined]
        parsed = _parse_event(event)
        if parsed.seq is not None:
            cursor = parsed.seq

        envelope = dict(parsed.raw)
        if parsed.seq is not None and "seq" not in envelope:
            envelope["seq"] = parsed.seq

        if parsed.kind in _MILESTONE_STATUS_REFRESH_KINDS and (
            "spent_micros" not in envelope and "spent_usd" not in envelope
        ):
            latest_status = dict(client.status())  # type: ignore[attr-defined]
        else:
            for key in ("status", "step", "epoch", "reason", "spent_micros", "cap_micros", "rollouts_completed"):
                if key in envelope:
                    latest_status[key] = envelope[key]

        enriched = _merge_status_context(envelope, latest_status)
        if json_mode:
            out.write(json.dumps(enriched, sort_keys=True, ensure_ascii=False) + "\n")
        else:
            out.write(format_event_line(enriched) + "\n")
        out.flush()

        if not follow and target_seq is not None and target_seq > 0 and cursor >= target_seq:
            break

        if follow and (
            parsed.kind in _TERMINAL_EVENT_KINDS
            or str(envelope.get("status", "")) in _TERMINAL_STATUSES
        ):
            latest_status = dict(client.status())  # type: ignore[attr-defined]
            latest_seq = latest_status.get("event_seq")
            drained = not isinstance(latest_seq, int) or cursor >= latest_seq
            if drained and (
                str(latest_status.get("status", "")) in _TERMINAL_STATUSES
                or _is_cap_pause_status(latest_status)
            ):
                break

    if cursor > after_seq and str(latest_status.get("status", "")) not in _TERMINAL_STATUSES:
        latest_status = dict(client.status())  # type: ignore[attr-defined]
    return latest_status


def validate_daemon_url(url: str, *, auth_token_env: Optional[str] = None) -> tuple[str, dict[str, str]]:
    """Validate an HTTP daemon URL and enforce bearer auth on non-loopback hosts."""
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise PreflightError(f"Invalid daemon URL: {url!r}")
    is_loopback = parsed.hostname in _LOOPBACK_HOSTS
    headers: dict[str, str] = {}
    if auth_token_env is not None:
        token = os.environ.get(auth_token_env, "").strip()
        if not token:
            raise PreflightError(
                f"Bearer auth environment variable {auth_token_env!r} is missing or empty"
            )
        headers["Authorization"] = f"Bearer {token}"
    elif not is_loopback:
        raise PreflightError(
            "Non-loopback daemon URLs require --auth-token-env for bearer authentication"
        )
    return url.rstrip("/"), headers


class HttpClient:
    """HTTP/SSE client compatible with `hillclimb.client.Client` for TCP loopback or remote daemons."""

    def __init__(
        self,
        base_url: str,
        *,
        headers: Optional[Mapping[str, str]] = None,
        timeout: float = READ_TIMEOUT_SECONDS,
    ) -> None:
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=dict(headers or {}),
            timeout=timeout,
            trust_env=False,
        )

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def status(self) -> Json:
        return self._json("GET", "/v1/status")

    def visuals(self) -> Json:
        return self._json("GET", "/v1/visuals")

    def capabilities(self) -> Json:
        return self._json("GET", "/v1/capabilities")

    def command(
        self,
        command: Command,
        *,
        value: Optional[Union[str, int, Json]] = None,
        op_id: Optional[UUID] = None,
        expected_epoch: Optional[int] = None,
        enqueue: bool = False,
    ) -> Json:
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
                    "command": command.value if value is None else {command.value: value},
                },
            )
        except HillclimbError as error:
            scope = f" at epoch {expected_epoch}" if expected_epoch is not None else ""
            raise HillclimbError(f"operation {identifier}{scope}: {error}") from error

    def slice(
        self,
        directory: Path,
        *,
        proposal: bool = False,
        intervention: Optional[Json] = None,
        at_seq: Optional[int] = None,
    ) -> Json:
        payload: dict[str, object] = {
            "directory": str(directory.resolve()),
            "proposal": proposal,
        }
        if intervention is not None:
            payload["intervention"] = intervention
        if at_seq is not None:
            payload["at_seq"] = at_seq
        return self._json("POST", "/v1/slice", payload)

    def events(self, *, after_seq: int = 0) -> Iterator[Event]:
        for event in self.event_frames(after_seq=after_seq):
            if event is not None:
                yield event

    def event_frames(self, *, after_seq: int = 0) -> Iterator[Optional[Event]]:
        try:
            with self._http.stream(
                "GET", "/v1/events", headers={"Last-Event-ID": str(after_seq)}
            ) as response:
                if response.is_error:
                    response.read()
                    raise HillclimbError(f"HTTP {response.status_code}: {response.text}")
                yield from _event_frames(response.iter_lines(), after_seq)
        except httpx.RequestError as error:
            raise HillclimbError(f"event stream disconnected: {error}") from error

    def _json(self, method: str, route: str, body: Optional[Json] = None) -> Json:
        try:
            response = self._http.request(method, route, json=body)
        except httpx.RequestError as error:
            raise HillclimbError(f"{method} {route}: {error}") from error
        if response.is_error:
            raise HillclimbError(f"HTTP {response.status_code}: {response.text}")
        return response.json()


def open_run_client(
    *,
    run_id: Optional[UUID] = None,
    directory: Optional[Path] = None,
    url: Optional[str] = None,
    auth_token_env: Optional[str] = None,
    timeout: float = READ_TIMEOUT_SECONDS,
) -> Union[Client, HttpClient]:
    """Open either an HTTP daemon client (`--url`) or a local UDS `Client(Run)`."""
    if url is not None:
        clean_url, headers = validate_daemon_url(url, auth_token_env=auth_token_env)
        return HttpClient(clean_url, headers=headers, timeout=timeout)
    if run_id is None:
        raise PreflightError("Specify either run_id or --url")
    return Client(Run.locate(run_id, directory), timeout=timeout)
