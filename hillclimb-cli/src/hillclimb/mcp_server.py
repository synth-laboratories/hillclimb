"""Zero-dependency stdio JSON-RPC 2.0 MCP server with SSE push events for Hillclimb."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import threading
import tomllib
from collections import deque
from collections.abc import Callable, Mapping
from contextlib import ExitStack, closing
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Optional, TextIO, Union
from uuid import UUID, uuid4

import httpx
import zstandard

from . import launcher
from .client import Client, Command, Event, HillclimbError, Json, Run, _event_frames, launch
from .logs import (
    HttpClient,
    format_event_line,
    validate_daemon_url,
)
from .skill_template import TEMPLATE_NAMES

__all__ = [
    "MCP_PROTOCOL_VERSION",
    "MCP_TOOL_NAMES",
    "McpServer",
    "collect_capabilities",
    "serve_stdio",
]

MCP_PROTOCOL_VERSION = "2025-03-26"
_SUPPORTED_PROTOCOL_VERSIONS = frozenset({"2024-11-05", "2025-03-26", "2025-06-18"})
_MAX_BUFFER_EVENTS = 500
_MAX_TAIL_LIMIT = 200
_MAX_FAILURES_LIMIT = 50
_MAX_TRACE_BYTES = 512 * 1024
_MAX_PREVIEW_CHARS = 2048
_SEARCH_SPLITS = frozenset({"train_reflect", "val_pareto"})
_PROGRESS_MILESTONE_KINDS = frozenset(
    {
        "created",
        "generation_start",
        "settled",
        "cache_hit",
        "frontier_accepted",
        "frontier_rejected",
        "minibatch_rejected",
        "perfect_parent",
        "decision",
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
_MESSAGE_EVENT_KINDS = frozenset(
    {
        "frontier_accepted",
        "budget_warning",
        "budget_refused",
        "in_doubt",
        "reservation_breached",
        "completed",
        "cancelled",
        "cap_reached",
    }
)
_TERMINAL_STATUSES = frozenset({"cancelled", "completed", "failed", "in_doubt", "cap_reached"})
_RESOURCE_URI_RE = re.compile(r"^hillclimb://runs/([0-9a-fA-F-]{36})/(status|events)$")


class McpToolName(StrEnum):
    """Canonical names of the 8 bounded Hillclimb MCP tools."""

    CAPABILITIES = "hillclimb_capabilities"
    START = "hillclimb_start"
    STATUS = "hillclimb_status"
    TAIL = "hillclimb_tail"
    INSPECT_FAILURES = "hillclimb_inspect_failures"
    CONTROL = "hillclimb_control"
    SLICE = "hillclimb_slice"
    CHAMPION = "hillclimb_champion"


MCP_TOOL_NAMES: tuple[str, ...] = tuple(item.value for item in McpToolName)


class _JsonRpcError(Exception):
    def __init__(self, code: int, message: str, *, data: Optional[object] = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


@dataclass
class _RunSession:
    """Mutable per-run tracking state guarded by `lock` inside `McpServer`."""

    run_id: str
    run: Optional[Run]
    base_url: Optional[str]
    headers: dict[str, str]
    directory: Optional[Path]
    events: deque[dict[str, object]] = field(default_factory=lambda: deque(maxlen=_MAX_BUFFER_EVENTS))
    cursor: int = 0
    progress_tokens: set[Union[str, int]] = field(default_factory=set)
    latest_status: dict[str, object] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    condition: threading.Condition = field(init=False)
    stop_event: threading.Event = field(default_factory=threading.Event)
    watcher_thread: Optional[threading.Thread] = None
    stream_http: Optional[httpx.Client] = None

    def __post_init__(self) -> None:
        self.condition = threading.Condition(self.lock)

    def open_client(self, *, timeout: float = 10.0) -> Union[Client, HttpClient]:
        if self.base_url is not None:
            return HttpClient(self.base_url, headers=self.headers, timeout=timeout)
        assert self.run is not None, "local run session requires a Run handle"
        return Client(self.run, timeout=timeout)


def _new_run_id() -> UUID:
    from .cli import _new_run_id as cli_new_run_id

    return cli_new_run_id()


def _parse_uuid(value: object, *, field_name: str, require_v7: bool = False) -> UUID:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a UUID string")
    try:
        parsed = UUID(value.strip())
    except ValueError as error:
        raise ValueError(f"{field_name} must be a valid UUID: {error}") from error
    if require_v7 and parsed.version != 7:
        raise ValueError(f"{field_name} must be a UUIDv7")
    return parsed


def _parse_resource_uri(uri: object) -> tuple[UUID, str]:
    if not isinstance(uri, str):
        raise _JsonRpcError(-32602, "resource uri must be a string")
    match = _RESOURCE_URI_RE.fullmatch(uri.strip())
    if match is None:
        raise _JsonRpcError(
            -32602,
            f"Unsupported resource URI {uri!r}; expected hillclimb://runs/{{run_id}}/(status|events)",
        )
    run_id = _parse_uuid(match.group(1), field_name="run_id")
    kind = match.group(2)
    return run_id, kind


def _probe_daemon_binary(daemon: str) -> dict[str, object]:
    daemon_path = Path(daemon)
    exists = daemon_path.is_file() and os.access(daemon_path, os.X_OK)
    if not exists:
        import shutil

        resolved = shutil.which(daemon)
        exists = resolved is not None
        if resolved is not None:
            daemon = resolved
    info: dict[str, object] = {
        "binary": daemon,
        "executable": exists,
        "capabilities": None,
    }
    if not exists:
        return info
    try:
        proc = subprocess.run(
            [daemon, "capabilities"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        info["probe_error"] = str(error)
        return info
    if proc.returncode == 0 and proc.stdout.strip():
        try:
            info["capabilities"] = json.loads(proc.stdout)
        except ValueError as error:
            info["probe_error"] = f"invalid JSON from daemon capabilities: {error}"
    else:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        info["probe_error"] = detail[0] if detail else f"exit code {proc.returncode}"
    return info


def collect_capabilities(
    *,
    daemon: Optional[str] = None,
    url: Optional[str] = None,
    auth_token_env: Optional[str] = None,
) -> dict[str, object]:
    """Assemble live daemon, CLI adapter, template, and MCP capability metadata."""
    try:
        cli_version = importlib.metadata.version("synth-hillclimb")
    except importlib.metadata.PackageNotFoundError:
        cli_version = "0.0.1"

    source = launcher.local_source()
    registry: Optional[dict[str, object]] = None
    if source is not None and (source / "capabilities.toml").is_file():
        with (source / "capabilities.toml").open("rb") as stream:
            registry = tomllib.load(stream)

    daemon_binary = daemon or launcher.default_daemon()
    daemon_info = _probe_daemon_binary(daemon_binary)

    server_capabilities: Optional[object] = None
    server_probe_error: Optional[str] = None
    if url is not None:
        clean_url, headers = validate_daemon_url(url, auth_token_env=auth_token_env)
        with HttpClient(clean_url, headers=headers, timeout=3.0) as client:
            try:
                server_capabilities = dict(client.capabilities())
            except HillclimbError as error:
                server_probe_error = str(error)

    surfaces = {
        "cli": True,
        "logs": True,
        "mcp": True,
        "sdk": importlib.util.find_spec("hillclimb.sdk") is not None,
        "config_toml": importlib.util.find_spec("hillclimb.config_toml") is not None,
        "compat_dspy": importlib.util.find_spec("hillclimb.compat.dspy") is not None,
        "compat_gepa_ai": importlib.util.find_spec("hillclimb.compat.gepa_ai") is not None,
    }

    payload: dict[str, object] = {
        "schema": "hillclimb.capabilities.v1",
        "cli_version": cli_version,
        "protocol_version": MCP_PROTOCOL_VERSION,
        "algorithms": [dict(entry) for entry in launcher.ALGORITHMS],
        "templates": list(TEMPLATE_NAMES),
        "targets": ["synthetic", "python_callback", "container", "harbor", "craftax"],
        "endpoints": [
            "/v1/status",
            "/v1/visuals",
            "/v1/commands",
            "/v1/slice",
            "/v1/events",
        ],
        "surfaces": surfaces,
        "exit_codes": {
            "completed": 0,
            "cap_reached": 2,
            "in_doubt": 3,
            "preflight_error": 4,
        },
        "mcp": {
            "transport": "stdio",
            "tools": list(MCP_TOOL_NAMES),
            "resources": [
                "hillclimb://runs/{run_id}/status",
                "hillclimb://runs/{run_id}/events",
            ],
            "push_notifications": [
                "notifications/progress",
                "notifications/message",
                "notifications/resources/updated",
            ],
        },
        "daemon": daemon_info,
    }
    if url is not None:
        payload["server_url"] = url
        payload["server_capabilities"] = server_capabilities
        if server_probe_error is not None:
            payload["server_probe_error"] = server_probe_error
    if registry is not None:
        payload["registry"] = registry
    return payload


def _read_cas_bytes(directory: Path, digest: str, *, limit: int = _MAX_TRACE_BYTES) -> bytes:
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"invalid CAS digest: {digest!r}")
    blob = directory / "cas" / "sha256" / digest[:2] / f"{digest}.zst"
    with (
        blob.open("rb") as raw,
        zstandard.ZstdDecompressor().stream_reader(raw) as reader,
    ):
        data = reader.read(limit + 1)
    if len(data) > limit:
        raise ValueError("CAS artifact exceeds read limit")
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"CAS digest mismatch for {digest}")
    return data


def _preview_json_or_text(data: bytes) -> object:
    text = data.decode("utf-8", errors="replace")
    if len(text) <= _MAX_PREVIEW_CHARS:
        try:
            return json.loads(text)
        except ValueError:
            return text
    return text[:_MAX_PREVIEW_CHARS] + "...(truncated)"


def _toml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_scalar(item) for item in value) + "]"
    raise ValueError(f"Unsupported TOML scalar value: {type(value).__name__}")


def _mapping_to_toml(table: Mapping[str, object], *, prefix: str = "") -> str:
    lines: list[str] = []
    nested: list[tuple[str, Mapping[str, object]]] = []
    for key, value in table.items():
        if isinstance(value, Mapping):
            nested.append((key, value))
        elif value is not None:
            lines.append(f"{key} = {_toml_scalar(value)}")
    for key, subtable in nested:
        section = f"{prefix}.{key}" if prefix else key
        if lines:
            lines.append("")
        lines.append(f"[{section}]")
        lines.append(_mapping_to_toml(subtable, prefix=section).rstrip())
    return "\n".join(lines) + "\n"


def _inspect_ledger_failures(
    directory: Path,
    *,
    policy_id: Optional[str],
    limit: int,
) -> list[dict[str, object]]:
    ledger = directory / "ledger.sqlite"
    if not ledger.is_file():
        return []
    uri = ledger.as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        db.row_factory = sqlite3.Row
        tables = {
            row["name"]
            for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        if "trials" not in tables:
            return []
        has_assessments = "assessments" in tables
        where_parts = ["t.split IN ('train_reflect', 'val_pareto')"]
        params: list[object] = []
        if policy_id is not None:
            where_parts.append("t.policy_id = ?")
            params.append(policy_id)
        where_sql = " AND ".join(where_parts)
        if has_assessments:
            sql = (
                "SELECT t.trial_id, t.rollout_id, t.policy_id, t.split, t.task_id, "
                "t.trace_cas_sha256, a.primary_score, a.metrics_json "
                "FROM trials t "
                "LEFT JOIN assessments a ON a.effect_id = t.effect_id "
                f"WHERE {where_sql} "
                "ORDER BY COALESCE(a.primary_score, 0.0) ASC, t.rowid DESC LIMIT ?"
            )
        else:
            sql = (
                "SELECT t.trial_id, t.rollout_id, t.policy_id, t.split, t.task_id, "
                "t.trace_cas_sha256, NULL AS primary_score, NULL AS metrics_json "
                "FROM trials t "
                f"WHERE {where_sql} "
                "ORDER BY t.rowid DESC LIMIT ?"
            )
        params.append(limit)
        rows = db.execute(sql, params).fetchall()
        results: list[dict[str, object]] = []
        for row in rows:
            entry: dict[str, object] = {
                "trial_id": row["trial_id"],
                "rollout_id": row["rollout_id"],
                "policy_id": row["policy_id"],
                "split": row["split"],
                "task_id": row["task_id"],
                "score": row["primary_score"],
            }
            if row["metrics_json"]:
                try:
                    entry["metrics"] = json.loads(row["metrics_json"])
                except ValueError:
                    entry["metrics"] = str(row["metrics_json"])[:512]
            digest = row["trace_cas_sha256"]
            if isinstance(digest, str):
                try:
                    entry["trace"] = _preview_json_or_text(_read_cas_bytes(directory, digest))
                except (OSError, ValueError, zstandard.ZstdError) as error:
                    entry["trace_error"] = str(error)
            results.append(entry)
        return results


def _inspect_visual_failures(
    visuals: Mapping[str, object],
    *,
    policy_id: Optional[str],
    limit: int,
) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []

    raw_failures = visuals.get("failures")
    if isinstance(raw_failures, list):
        for item in raw_failures:
            if not isinstance(item, Mapping):
                continue
            if item.get("split") == "final_test":
                continue
            if policy_id is not None and item.get("policy_id") != policy_id:
                continue
            failures.append(dict(item))

    target = visuals.get("target")
    if isinstance(target, Mapping) and isinstance(target.get("rows"), list):
        search_rows = [
            row
            for row in target["rows"]
            if isinstance(row, Mapping)
            and row.get("split") in _SEARCH_SPLITS
            and (policy_id is None or row.get("policy_id") == policy_id)
        ]

        def _row_sort_key(row: Mapping[str, object]) -> float:
            metrics = row.get("metrics")
            if isinstance(metrics, Mapping):
                for key in ("score", "primary_score", "reward", "mean_clipped"):
                    val = metrics.get(key)
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        return float(val)
            score = row.get("score")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                return float(score)
            return 0.0

        for row in sorted(search_rows, key=_row_sort_key):
            failures.append(
                {
                    "trial_id": row.get("trial_id"),
                    "rollout_id": row.get("rollout_id"),
                    "policy_id": row.get("policy_id"),
                    "split": row.get("split"),
                    "task_id": row.get("task_id"),
                    "score": _row_sort_key(row),
                    "metrics": row.get("metrics"),
                }
            )

    algo = visuals.get("algo")
    if isinstance(algo, Mapping) and isinstance(algo.get("decisions"), list):
        for decision in algo["decisions"]:
            if not isinstance(decision, Mapping):
                continue
            kind = str(decision.get("decision") or "")
            child = decision.get("child")
            parent = decision.get("parent")
            if policy_id is not None and policy_id not in {child, parent}:
                continue
            deltas = decision.get("deltas")
            has_breaks = isinstance(deltas, list) and "Break" in deltas
            if kind != "frontier_accepted" or has_breaks or decision.get("rejection_reason"):
                failures.append(
                    {
                        "kind": "decision",
                        "parent": parent,
                        "policy_id": child or parent,
                        "decision": kind,
                        "rejection_reason": decision.get("rejection_reason"),
                        "breaks": deltas.count("Break") if isinstance(deltas, list) else 0,
                        "fixes": deltas.count("Fix") if isinstance(deltas, list) else 0,
                    }
                )

    return failures[:limit]


def _tool_definitions() -> list[dict[str, object]]:
    return [
        {
            "name": McpToolName.CAPABILITIES.value,
            "description": "Report supported Hillclimb algorithms, targets, endpoints, adapters, and daemon capabilities.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": McpToolName.START.value,
            "description": "Start a new Hillclimb campaign or attach to an explicit daemon URL and begin SSE progress streaming.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "config_path": {"type": "string"},
                    "config": {"type": "object"},
                    "synthetic": {"type": "boolean"},
                    "algorithm": {"type": "string", "enum": ["gepa", "mipro", "go-ex", "miprov2", "gelo"]},
                    "iterations": {"type": "integer", "minimum": 1, "maximum": 10000},
                    "seed": {"type": "integer"},
                    "run_id": {"type": "string"},
                    "directory": {"type": "string"},
                    "tick_millis": {"type": "integer", "minimum": 1, "maximum": 60000},
                    "url": {"type": "string"},
                    "progress_token": {"type": ["string", "integer"]},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": McpToolName.STATUS.value,
            "description": "Read current bounded campaign status, progress counters, and budget metrics.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "string"},
                    "directory": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": McpToolName.TAIL.value,
            "description": "Read bounded durable campaign events after a given SSE sequence cursor.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "string"},
                    "after_seq": {"type": "integer", "minimum": 0, "default": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": _MAX_TAIL_LIMIT, "default": 20},
                    "directory": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": McpToolName.INSPECT_FAILURES.value,
            "description": "Inspect bounded search-split failures and trace previews (final_test remains sealed).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "string"},
                    "policy_id": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": _MAX_FAILURES_LIMIT, "default": 10},
                    "directory": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": McpToolName.CONTROL.value,
            "description": "Submit an idempotent control command (pause, resume, steer_proposer, inject_candidate).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "string"},
                    "action": {
                        "type": "string",
                        "enum": ["pause", "resume", "steer_proposer", "inject_candidate"],
                    },
                    "op_id": {"type": "string"},
                    "expected_epoch": {"type": "integer", "minimum": 0},
                    "note": {"type": "string"},
                    "next_proposals": {"type": "integer", "minimum": 1},
                    "parent_id": {"type": "string"},
                    "intervention": {"type": "object"},
                    "directory": {"type": "string"},
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        },
        {
            "name": McpToolName.SLICE.value,
            "description": "Fork a checkpoint slice at the current barrier or specified event sequence with an optional proposer intervention.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "string"},
                    "at_seq": {"type": "integer", "minimum": 0},
                    "intervention": {"type": ["object", "string", "null"]},
                    "destination": {"type": "string"},
                    "proposal": {"type": "boolean"},
                    "directory": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": McpToolName.CHAMPION.value,
            "description": "Return the current champion policy summary, validation score, and sealed held-out score when completed.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "string"},
                    "directory": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    ]


class McpServer:
    """Thread-safe JSON-RPC 2.0 MCP server bridging Hillclimb daemon HTTP/UDS+SSE."""

    def __init__(
        self,
        *,
        out: TextIO = sys.stdout,
        daemon: Optional[str] = None,
        default_url: Optional[str] = None,
        auth_token_env: Optional[str] = None,
        default_run_id: Optional[UUID] = None,
        default_directory: Optional[Path] = None,
    ) -> None:
        self._out = out
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._daemon = daemon or launcher.default_daemon()
        self._default_url: Optional[str] = None
        self._default_headers: dict[str, str] = {}
        if default_url is not None:
            self._default_url, self._default_headers = validate_daemon_url(
                default_url, auth_token_env=auth_token_env
            )
        self._auth_token_env = auth_token_env
        self._runs: dict[str, _RunSession] = {}
        self._subscriptions: set[str] = set()
        self._closed = False
        self._initialized = False

        if default_run_id is not None:
            run_id_str = str(default_run_id)
            run = None if self._default_url is not None else Run.locate(default_run_id, default_directory)
            self._runs[run_id_str] = _RunSession(
                run_id=run_id_str,
                run=run,
                base_url=self._default_url,
                headers=dict(self._default_headers),
                directory=default_directory or (run.directory if run else None),
            )

    def _emit_json(self, payload: Mapping[str, object]) -> None:
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        with self._write_lock:
            if self._closed:
                return
            self._out.write(encoded + "\n")
            self._out.flush()

    def _emit_notification(self, method: str, params: Mapping[str, object]) -> None:
        self._emit_json({"jsonrpc": "2.0", "method": method, "params": dict(params)})

    def _resolve_session(
        self,
        *,
        run_id_value: Optional[object] = None,
        directory_value: Optional[object] = None,
    ) -> _RunSession:
        directory = Path(str(directory_value)).expanduser().resolve() if directory_value is not None else None
        if run_id_value is not None:
            parsed = _parse_uuid(run_id_value, field_name="run_id")
            key = str(parsed)
            with self._state_lock:
                session = self._runs.get(key)
                if session is not None:
                    return session
                run = None if self._default_url is not None else Run.locate(parsed, directory)
                session = _RunSession(
                    run_id=key,
                    run=run,
                    base_url=self._default_url,
                    headers=dict(self._default_headers),
                    directory=directory or (run.directory if run else None),
                )
                self._runs[key] = session
                return session

        with self._state_lock:
            if len(self._runs) == 1:
                return next(iter(self._runs.values()))
            if len(self._runs) > 1:
                raise ValueError("Multiple runs are active; specify run_id")

        if self._default_url is not None:
            with HttpClient(self._default_url, headers=self._default_headers) as client:
                status = dict(client.status())
            parsed = _parse_uuid(status.get("run_id"), field_name="run_id")
            key = str(parsed)
            with self._state_lock:
                session = _RunSession(
                    run_id=key,
                    run=None,
                    base_url=self._default_url,
                    headers=dict(self._default_headers),
                    directory=Path(str(status["directory"])) if status.get("directory") else directory,
                    latest_status=status,
                )
                self._runs[key] = session
                return session

        raise ValueError("No active run; call hillclimb_start or specify run_id")

    def _ensure_watcher(self, session: _RunSession) -> None:
        with session.lock:
            if session.watcher_thread is not None and session.watcher_thread.is_alive():
                return
            if session.stop_event.is_set() or self._closed:
                return
            thread = threading.Thread(
                target=self._watch_sse,
                args=(session,),
                name=f"hillclimb-mcp-sse-{session.run_id[:8]}",
                daemon=True,
            )
            session.watcher_thread = thread
            thread.start()

    def _watch_sse(self, session: _RunSession) -> None:
        while not session.stop_event.is_set() and not self._closed:
            with session.lock:
                after_seq = session.cursor
            try:
                if session.base_url is not None:
                    http = httpx.Client(
                        base_url=session.base_url.rstrip("/"),
                        headers=session.headers,
                        timeout=None,
                        trust_env=False,
                    )
                else:
                    assert session.run is not None
                    http = httpx.Client(
                        base_url="http://localhost",
                        transport=httpx.HTTPTransport(uds=str(session.run.socket)),
                        timeout=None,
                        trust_env=False,
                    )
                with session.lock:
                    session.stream_http = http
                with http:
                    with http.stream(
                        "GET", "/v1/events", headers={"Last-Event-ID": str(after_seq)}
                    ) as response:
                        if response.is_error:
                            response.read()
                            raise HillclimbError(f"HTTP {response.status_code}: {response.text}")
                        for frame in _event_frames(response.iter_lines(), after_seq):
                            if session.stop_event.is_set() or self._closed:
                                return
                            if frame is None:
                                continue
                            self._handle_sse_event(session, frame)
            except (HillclimbError, httpx.RequestError, OSError):
                if session.stop_event.is_set() or self._closed:
                    return
                with session.lock:
                    latest_state = str(session.latest_status.get("status", ""))
                if latest_state in _TERMINAL_STATUSES:
                    return
                session.stop_event.wait(0.2)
            finally:
                with session.lock:
                    session.stream_http = None

            with session.lock:
                latest_state = str(session.latest_status.get("status", ""))
            if latest_state in _TERMINAL_STATUSES:
                return
            session.stop_event.wait(0.1)

    def _handle_sse_event(self, session: _RunSession, frame: Event) -> None:
        envelope = dict(frame.envelope)
        envelope["seq"] = frame.seq
        kind = str(envelope.get("kind", "event"))
        status_uri = f"hillclimb://runs/{session.run_id}/status"
        events_uri = f"hillclimb://runs/{session.run_id}/events"

        with session.lock:
            if frame.seq <= session.cursor:
                return
            session.cursor = frame.seq
            session.events.append(envelope)
            for key in (
                "status",
                "step",
                "epoch",
                "reason",
                "spent_micros",
                "cap_micros",
                "rollouts_completed",
                "policy_id",
            ):
                if key in envelope:
                    session.latest_status[key] = envelope[key]
            session.latest_status["event_seq"] = frame.seq
            tokens = list(session.progress_tokens)
            latest_snapshot = dict(session.latest_status)
            session.condition.notify_all()

        with self._state_lock:
            subscribed_status = status_uri in self._subscriptions
            subscribed_events = events_uri in self._subscriptions

        if tokens and kind in _PROGRESS_MILESTONE_KINDS:
            progress = envelope.get("rollouts_completed")
            if not isinstance(progress, int):
                progress = envelope.get("step")
            if not isinstance(progress, int):
                progress = frame.seq
            total = envelope.get("total_steps") or latest_snapshot.get("iterations")
            progress_params: dict[str, object] = {
                "progress": progress,
                "message": format_event_line(envelope),
            }
            if isinstance(total, int) and total > 0:
                progress_params["total"] = total
            for token in tokens:
                self._emit_notification(
                    "notifications/progress",
                    {"progressToken": token, **progress_params},
                )

        decision = str(envelope.get("decision") or "")
        if kind in _MESSAGE_EVENT_KINDS or decision == "frontier_accepted":
            level = "info"
            if kind in {"in_doubt", "reservation_breached"}:
                level = "error"
            elif kind in {"budget_warning", "budget_refused", "cancelled", "cap_reached"}:
                level = "warning"
            bounded_data = {
                key: envelope[key]
                for key in (
                    "kind",
                    "seq",
                    "run_id",
                    "step",
                    "epoch",
                    "status",
                    "policy_id",
                    "child",
                    "decision",
                    "val_score",
                    "delta",
                    "spent_micros",
                    "cap_micros",
                    "reason",
                )
                if key in envelope
            }
            bounded_data["summary"] = format_event_line(envelope)
            self._emit_notification(
                "notifications/message",
                {"level": level, "logger": "hillclimb", "data": bounded_data},
            )

        if subscribed_events:
            self._emit_notification("notifications/resources/updated", {"uri": events_uri})
        if subscribed_status:
            self._emit_notification("notifications/resources/updated", {"uri": status_uri})

    def _tool_capabilities(self, arguments: Mapping[str, object]) -> dict[str, object]:
        if arguments:
            raise ValueError("hillclimb_capabilities takes no arguments")
        return collect_capabilities(
            daemon=self._daemon,
            url=self._default_url,
            auth_token_env=self._auth_token_env,
        )

    def _tool_start(
        self,
        arguments: Mapping[str, object],
        *,
        meta: Mapping[str, object],
    ) -> dict[str, object]:
        progress_token = arguments.get("progress_token")
        if progress_token is None:
            progress_token = meta.get("progressToken")
        if progress_token is not None and not isinstance(progress_token, (str, int)):
            raise ValueError("progressToken must be a string or integer")

        explicit_url = arguments.get("url")
        if explicit_url is not None and not isinstance(explicit_url, str):
            raise ValueError("url must be a string")

        config_path = arguments.get("config_path")
        config_obj = arguments.get("config")
        synthetic = bool(arguments.get("synthetic", False))
        directory_raw = arguments.get("directory")
        directory = Path(str(directory_raw)).expanduser().resolve() if directory_raw is not None else None

        target_url = explicit_url
        if target_url is None and self._default_url is not None and config_path is None and config_obj is None:
            target_url = self._default_url

        if target_url is not None:
            clean_url, headers = validate_daemon_url(target_url, auth_token_env=self._auth_token_env)
            with HttpClient(clean_url, headers=headers) as client:
                status = dict(client.status())
            run_uuid = _parse_uuid(
                arguments.get("run_id") or status.get("run_id"),
                field_name="run_id",
            )
            run_id_str = str(run_uuid)
            session = _RunSession(
                run_id=run_id_str,
                run=None,
                base_url=clean_url,
                headers=headers,
                directory=Path(str(status["directory"])) if status.get("directory") else directory,
                latest_status=status,
            )
            if progress_token is not None:
                session.progress_tokens.add(progress_token)
            with self._state_lock:
                existing = self._runs.get(run_id_str)
                if existing is not None:
                    session = existing
                    if progress_token is not None:
                        with session.lock:
                            session.progress_tokens.add(progress_token)
                else:
                    self._runs[run_id_str] = session
            self._ensure_watcher(session)
            return {
                "run_id": run_id_str,
                "status": status,
                "resources": [
                    f"hillclimb://runs/{run_id_str}/status",
                    f"hillclimb://runs/{run_id_str}/events",
                ],
            }

        if config_path is None and config_obj is None and not synthetic:
            raise ValueError("Specify config_path, config, synthetic=true, or url")

        run_uuid = (
            _parse_uuid(arguments["run_id"], field_name="run_id", require_v7=True)
            if "run_id" in arguments
            else _new_run_id()
        )
        run = Run.locate(run_uuid, directory)
        algorithm = arguments.get("algorithm")
        if algorithm is not None:
            if not isinstance(algorithm, str) or algorithm not in launcher.ALGORITHM_CHOICES:
                raise ValueError(f"Unsupported algorithm {algorithm!r}")
            algorithm = launcher.canonical_algorithm(algorithm)

        iterations = arguments.get("iterations")
        if iterations is not None and (type(iterations) is not int or not 1 <= iterations <= 10_000):
            raise ValueError("iterations must be an integer between 1 and 10000")
        seed = arguments.get("seed")
        if seed is not None and type(seed) is not int:
            raise ValueError("seed must be an integer")
        tick_millis = arguments.get("tick_millis")
        if tick_millis is not None and (type(tick_millis) is not int or not 1 <= tick_millis <= 60_000):
            raise ValueError("tick_millis must be an integer between 1 and 60000")

        with ExitStack() as stack:
            resolved_config_path: Optional[Path] = None
            if config_path is not None or config_obj is not None:
                if config_path is not None and config_obj is not None:
                    raise ValueError("Specify either config_path or config, not both")
                if config_path is not None:
                    campaign = launcher.read_config(Path(str(config_path)).expanduser().resolve())
                else:
                    assert isinstance(config_obj, Mapping)
                    campaign = dict(config_obj)
                campaign["run_id"] = str(run_uuid)
                run.directory.parent.mkdir(parents=True, exist_ok=True)
                temp = stack.enter_context(
                    tempfile.NamedTemporaryFile(mode="w+", dir=run.directory.parent, suffix=".json")
                )
                temp.write(json.dumps(campaign))
                temp.flush()
                resolved_config_path = Path(temp.name)
                algorithm = None
            status = dict(
                launch(
                    run,
                    daemon=self._daemon,
                    mode="run",
                    algorithm=algorithm,
                    trials=iterations,
                    seed=seed,
                    config=resolved_config_path,
                    tick_millis=tick_millis,
                )
            )

        run_id_str = str(run_uuid)
        session = _RunSession(
            run_id=run_id_str,
            run=run,
            base_url=None,
            headers={},
            directory=run.directory,
            latest_status=status,
        )
        if progress_token is not None:
            session.progress_tokens.add(progress_token)
        with self._state_lock:
            self._runs[run_id_str] = session
        self._ensure_watcher(session)
        return {
            "run_id": run_id_str,
            "status": status,
            "resources": [
                f"hillclimb://runs/{run_id_str}/status",
                f"hillclimb://runs/{run_id_str}/events",
            ],
        }

    def _tool_status(
        self,
        arguments: Mapping[str, object],
        *,
        meta: Mapping[str, object],
    ) -> dict[str, object]:
        session = self._resolve_session(
            run_id_value=arguments.get("run_id"),
            directory_value=arguments.get("directory"),
        )
        progress_token = meta.get("progressToken")
        if isinstance(progress_token, (str, int)):
            with session.lock:
                session.progress_tokens.add(progress_token)
        with session.open_client() as client:
            status = dict(client.status())
        with session.lock:
            session.latest_status = status
        return status

    def _tool_tail(
        self,
        arguments: Mapping[str, object],
        *,
        meta: Mapping[str, object],
    ) -> dict[str, object]:
        session = self._resolve_session(
            run_id_value=arguments.get("run_id"),
            directory_value=arguments.get("directory"),
        )
        after_seq = arguments.get("after_seq", 0)
        limit = arguments.get("limit", 20)
        if type(after_seq) is not int or after_seq < 0:
            raise ValueError("after_seq must be a non-negative integer")
        if type(limit) is not int or not 1 <= limit <= _MAX_TAIL_LIMIT:
            raise ValueError(f"limit must be an integer between 1 and {_MAX_TAIL_LIMIT}")

        progress_token = meta.get("progressToken")
        if isinstance(progress_token, (str, int)):
            with session.lock:
                session.progress_tokens.add(progress_token)

        with session.open_client() as client:
            status = dict(client.status())
            server_seq = status.get("event_seq")
            target_seq = int(server_seq) if isinstance(server_seq, int) else None

            if target_seq is not None and after_seq >= target_seq:
                return {
                    "run_id": session.run_id,
                    "after_seq": after_seq,
                    "events": [],
                    "next_seq": after_seq,
                    "has_more": False,
                }

            with session.condition:
                watcher_alive = session.watcher_thread is not None and session.watcher_thread.is_alive()
                if watcher_alive and target_seq is not None:
                    desired_seq = min(target_seq, after_seq + limit)
                    if session.cursor < desired_seq:
                        session.condition.wait(timeout=0.5)
                buffered = [dict(ev) for ev in session.events if int(ev.get("seq", 0)) > after_seq]
                earliest_buffered = int(session.events[0]["seq"]) if session.events else None

            if buffered and (earliest_buffered is None or after_seq + 1 >= earliest_buffered):
                page = buffered[:limit]
            else:
                page = []
                for frame in client.event_frames(after_seq=after_seq):
                    if frame is None:
                        if target_seq is None or len(page) > 0:
                            break
                        continue
                    item = dict(frame.envelope)
                    item["seq"] = frame.seq
                    page.append(item)
                    if len(page) >= limit:
                        break
                    if target_seq is not None and target_seq > 0 and frame.seq >= target_seq:
                        break

        formatted = [
            {**item, "formatted": format_event_line(item)}
            for item in page
        ]
        next_seq = int(formatted[-1]["seq"]) if formatted else after_seq
        has_more = target_seq is not None and next_seq < target_seq
        return {
            "run_id": session.run_id,
            "after_seq": after_seq,
            "events": formatted,
            "next_seq": next_seq,
            "has_more": has_more,
        }

    def _tool_inspect_failures(self, arguments: Mapping[str, object]) -> dict[str, object]:
        session = self._resolve_session(
            run_id_value=arguments.get("run_id"),
            directory_value=arguments.get("directory"),
        )
        policy_id = arguments.get("policy_id")
        if policy_id is not None and (not isinstance(policy_id, str) or not policy_id.strip()):
            raise ValueError("policy_id must be a non-empty string")
        limit = arguments.get("limit", 10)
        if type(limit) is not int or not 1 <= limit <= _MAX_FAILURES_LIMIT:
            raise ValueError(f"limit must be an integer between 1 and {_MAX_FAILURES_LIMIT}")

        with session.open_client() as client:
            status = dict(client.status())
            run_dir = session.directory or (
                Path(str(status["directory"])) if status.get("directory") else None
            )
            if run_dir is not None and (run_dir / "ledger.sqlite").is_file():
                failures = _inspect_ledger_failures(run_dir, policy_id=policy_id, limit=limit)
            else:
                try:
                    visuals = client.visuals()
                except HillclimbError as error:
                    if "HTTP 404" not in str(error):
                        raise
                    visuals = {}
                failures = _inspect_visual_failures(visuals, policy_id=policy_id, limit=limit)

        return {
            "run_id": session.run_id,
            "policy_id": policy_id,
            "splits": sorted(_SEARCH_SPLITS),
            "failures": failures[:limit],
            "count": min(len(failures), limit),
        }

    def _tool_control(self, arguments: Mapping[str, object]) -> dict[str, object]:
        session = self._resolve_session(
            run_id_value=arguments.get("run_id"),
            directory_value=arguments.get("directory"),
        )
        action = arguments.get("action")
        if action not in {"pause", "resume", "steer_proposer", "inject_candidate"}:
            raise ValueError(f"Unsupported control action {action!r}")

        op_id_raw = arguments.get("op_id")
        op_id = _parse_uuid(op_id_raw, field_name="op_id") if op_id_raw is not None else uuid4()
        expected_epoch = arguments.get("expected_epoch")
        if expected_epoch is not None and (type(expected_epoch) is not int or expected_epoch < 0):
            raise ValueError("expected_epoch must be a non-negative integer")

        value: Optional[Union[str, int, Json]] = None
        if action == "pause":
            command = Command.PAUSE
        elif action == "resume":
            command = Command.RESUME
        elif action == "steer_proposer":
            note = arguments.get("note")
            if not isinstance(note, str) or not note.strip() or len(note) > 4096:
                raise ValueError("steer_proposer requires a non-empty note <= 4096 chars")
            next_proposals = arguments.get("next_proposals")
            if next_proposals is not None and (type(next_proposals) is not int or next_proposals < 1):
                raise ValueError("next_proposals must be a positive integer")
            command = Command.STEER_PROPOSER
            value = {"note": note, "next_proposals": next_proposals}
        else:
            parent_id = arguments.get("parent_id")
            intervention = arguments.get("intervention")
            if not isinstance(parent_id, str) or not parent_id.strip():
                raise ValueError("inject_candidate requires a non-empty parent_id")
            if not isinstance(intervention, Mapping) or not intervention:
                raise ValueError("inject_candidate requires a non-empty intervention object")
            command = Command.INJECT_CANDIDATE
            value = {"parent_id": parent_id, "intervention": dict(intervention)}

        with session.open_client() as client:
            receipt = dict(
                client.command(
                    command,
                    value=value,
                    op_id=op_id,
                    expected_epoch=expected_epoch,
                    enqueue=True,
                )
            )
        return {
            "run_id": session.run_id,
            "action": action,
            "op_id": str(op_id),
            "receipt": receipt,
        }

    def _tool_slice(self, arguments: Mapping[str, object]) -> dict[str, object]:
        session = self._resolve_session(
            run_id_value=arguments.get("run_id"),
            directory_value=arguments.get("directory"),
        )
        at_seq = arguments.get("at_seq")
        if at_seq is not None and (type(at_seq) is not int or at_seq < 0):
            raise ValueError("at_seq must be a non-negative integer")

        raw_intervention = arguments.get("intervention")
        intervention_obj: Optional[dict[str, object]] = None
        intervention_path: Optional[Path] = None
        if isinstance(raw_intervention, str):
            intervention_path = Path(raw_intervention).expanduser().resolve()
            with intervention_path.open("rb") as stream:
                intervention_obj = tomllib.load(stream)
        elif isinstance(raw_intervention, Mapping):
            intervention_obj = dict(raw_intervention)
        elif raw_intervention is not None:
            raise ValueError("intervention must be an object or TOML file path")

        proposal = bool(arguments.get("proposal", intervention_obj is not None or at_seq is not None))
        dest_raw = arguments.get("destination")

        with session.open_client() as client:
            status = dict(client.status())
            run_dir = session.directory or (
                Path(str(status["directory"])) if status.get("directory") else None
            )
            if dest_raw is not None:
                dest_dir = Path(str(dest_raw)).expanduser().resolve()
            else:
                home = Path(os.environ.get("HILLCLIMB_HOME", Path.home() / ".hillclimb"))
                dest_dir = (home / "runs" / f"slice-{_new_run_id()}").resolve()

            if (
                at_seq is not None
                and run_dir is not None
                and (run_dir / "ledger.sqlite").is_file()
                and session.base_url is None
            ):
                with ExitStack() as stack:
                    args = [
                        self._daemon,
                        "slice-proposal" if proposal else "slice",
                        "--from",
                        str(run_dir),
                        "--to",
                        str(dest_dir),
                    ]
                    if proposal:
                        args.extend(("--at-seq", str(at_seq)))
                    if intervention_path is not None:
                        args.extend(("--intervention", str(intervention_path)))
                    elif intervention_obj is not None:
                        tmp_dir = stack.enter_context(tempfile.TemporaryDirectory())
                        tmp_toml = Path(tmp_dir) / "intervention.toml"
                        tmp_toml.write_text(_mapping_to_toml(intervention_obj), encoding="utf-8")
                        args.extend(("--intervention", str(tmp_toml)))
                    completed = subprocess.run(args, capture_output=True, text=True, check=True)
                    result = json.loads(completed.stdout)
                    return dict(result)

            if isinstance(client, HttpClient):
                result = dict(
                    client.slice(
                        dest_dir,
                        proposal=proposal,
                        intervention=intervention_obj,
                        at_seq=at_seq,
                    )
                )
            else:
                result = dict(
                    client.slice(
                        dest_dir,
                        proposal=proposal,
                        intervention=intervention_obj,
                    )
                )
            return result

    def _tool_champion(self, arguments: Mapping[str, object]) -> dict[str, object]:
        session = self._resolve_session(
            run_id_value=arguments.get("run_id"),
            directory_value=arguments.get("directory"),
        )
        with session.open_client() as client:
            status = dict(client.status())
            try:
                visuals = dict(client.visuals())
            except HillclimbError as error:
                if "HTTP 404" not in str(error):
                    raise
                visuals = {}

        core = visuals.get("core") if isinstance(visuals.get("core"), Mapping) else {}
        summary = core.get("summary_metrics") if isinstance(core.get("summary_metrics"), Mapping) else {}
        algo = visuals.get("algo") if isinstance(visuals.get("algo"), Mapping) else {}

        champion_id = (
            status.get("frozen_policy")
            or algo.get("selected_policy")
            or algo.get("champion")
            or visuals.get("champion_policy_id")
        )
        if champion_id is None and isinstance(algo.get("frontier"), list) and algo["frontier"]:
            best = max(
                (pt for pt in algo["frontier"] if isinstance(pt, Mapping) and pt.get("scores")),
                key=lambda pt: sum(pt["scores"]) / len(pt["scores"]),
                default=None,
            )
            if best is not None:
                champion_id = best.get("policy") or best.get("policy_id")

        is_completed = str(status.get("status", "")) == "completed"
        heldout_score = summary.get("heldout") if is_completed else None
        spent_micros = int(status.get("spent_micros", 0))

        policy_preview: Optional[object] = visuals.get("champion_policy")
        run_dir = session.directory or (
            Path(str(status["directory"])) if status.get("directory") else None
        )
        if policy_preview is None and run_dir is not None and isinstance(champion_id, str):
            digest = champion_id.removeprefix("sha256:")
            if len(digest) == 64 and all(c in "0123456789abcdef" for c in digest):
                try:
                    policy_preview = _preview_json_or_text(_read_cas_bytes(run_dir, digest))
                except (OSError, ValueError, zstandard.ZstdError):
                    policy_preview = None

        return {
            "run_id": session.run_id,
            "algorithm_id": status.get("algorithm_id"),
            "status": status.get("status"),
            "champion_policy_id": champion_id,
            "is_frozen": status.get("frozen_policy") is not None,
            "val_score": summary.get("dev"),
            "heldout_score": heldout_score,
            "spent_usd": spent_micros / 1_000_000,
            "policy_preview": policy_preview,
        }

    def _call_tool(
        self,
        name: str,
        arguments: Mapping[str, object],
        *,
        meta: Mapping[str, object],
    ) -> dict[str, object]:
        dispatch: dict[str, Callable[[], dict[str, object]]] = {
            McpToolName.CAPABILITIES.value: lambda: self._tool_capabilities(arguments),
            McpToolName.START.value: lambda: self._tool_start(arguments, meta=meta),
            McpToolName.STATUS.value: lambda: self._tool_status(arguments, meta=meta),
            McpToolName.TAIL.value: lambda: self._tool_tail(arguments, meta=meta),
            McpToolName.INSPECT_FAILURES.value: lambda: self._tool_inspect_failures(arguments),
            McpToolName.CONTROL.value: lambda: self._tool_control(arguments),
            McpToolName.SLICE.value: lambda: self._tool_slice(arguments),
            McpToolName.CHAMPION.value: lambda: self._tool_champion(arguments),
        }
        handler = dispatch.get(name)
        if handler is None:
            raise _JsonRpcError(-32602, f"Unknown tool: {name}")
        return handler()

    def _list_resources(self) -> list[dict[str, object]]:
        resources: list[dict[str, object]] = []
        with self._state_lock:
            run_ids = sorted(self._runs)
        for run_id in run_ids:
            resources.extend(
                [
                    {
                        "uri": f"hillclimb://runs/{run_id}/status",
                        "name": f"Run {run_id[:8]} status",
                        "mimeType": "application/json",
                    },
                    {
                        "uri": f"hillclimb://runs/{run_id}/events",
                        "name": f"Run {run_id[:8]} events",
                        "mimeType": "application/json",
                    },
                ]
            )
        return resources

    def _read_resource(self, uri: str) -> dict[str, object]:
        run_uuid, kind = _parse_resource_uri(uri)
        session = self._resolve_session(run_id_value=str(run_uuid))
        if kind == "status":
            with session.open_client() as client:
                payload = dict(client.status())
            with session.lock:
                session.latest_status = payload
        else:
            payload = self._tool_tail({"run_id": str(run_uuid), "after_seq": 0, "limit": 50}, meta={})
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps(payload, ensure_ascii=False),
                }
            ]
        }

    def _subscribe_resource(self, uri: str, *, meta: Mapping[str, object]) -> dict[str, object]:
        run_uuid, _ = _parse_resource_uri(uri)
        session = self._resolve_session(run_id_value=str(run_uuid))
        progress_token = meta.get("progressToken")
        if isinstance(progress_token, (str, int)):
            with session.lock:
                session.progress_tokens.add(progress_token)
        with self._state_lock:
            self._subscriptions.add(uri)
        self._ensure_watcher(session)
        return {}

    def _unsubscribe_resource(self, uri: str) -> dict[str, object]:
        _parse_resource_uri(uri)
        with self._state_lock:
            self._subscriptions.discard(uri)
        return {}

    def handle_message(self, message: object) -> Optional[dict[str, object]]:
        """Process one decoded JSON-RPC 2.0 request/notification and return response if `id` is set."""
        if not isinstance(message, Mapping):
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Invalid Request: expected JSON object"},
            }
        has_id = "id" in message
        req_id = message.get("id")
        if message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
            if not has_id:
                return None
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32600, "message": "Invalid Request"},
            }

        method = str(message["method"])
        raw_params = message.get("params")
        if raw_params is None:
            params: Mapping[str, object] = {}
        elif isinstance(raw_params, Mapping):
            params = raw_params
        else:
            if not has_id:
                return None
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": "Invalid params: expected object"},
            }

        if not has_id:
            if method == "notifications/initialized":
                self._initialized = True
            return None

        try:
            result = self._dispatch_request(method, params)
        except _JsonRpcError as error:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": error.code,
                    "message": error.message,
                    **({"data": error.data} if error.data is not None else {}),
                },
            }
        except (HillclimbError, ValueError, OSError, sqlite3.Error, subprocess.SubprocessError) as error:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32603, "message": str(error)},
            }
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def _dispatch_request(self, method: str, params: Mapping[str, object]) -> dict[str, object]:
        if method == "initialize":
            client_version = params.get("protocolVersion")
            negotiated = (
                str(client_version)
                if isinstance(client_version, str) and client_version in _SUPPORTED_PROTOCOL_VERSIONS
                else MCP_PROTOCOL_VERSION
            )
            return {
                "protocolVersion": negotiated,
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"subscribe": True, "listChanged": True},
                    "logging": {},
                },
                "serverInfo": {"name": "hillclimb-mcp", "version": "0.1.0"},
            }
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": _tool_definitions()}
        if method == "tools/call":
            name = params.get("name")
            if not isinstance(name, str) or not name:
                raise _JsonRpcError(-32602, "tools/call requires a string tool name")
            raw_args = params.get("arguments")
            if raw_args is None:
                arguments: Mapping[str, object] = {}
            elif isinstance(raw_args, Mapping):
                arguments = raw_args
            else:
                raise _JsonRpcError(-32602, "tools/call arguments must be an object")
            raw_meta = params.get("_meta")
            meta = raw_meta if isinstance(raw_meta, Mapping) else {}
            try:
                payload = self._call_tool(name, arguments, meta=meta)
            except _JsonRpcError:
                raise
            except (HillclimbError, ValueError, OSError, sqlite3.Error, subprocess.SubprocessError) as error:
                return {
                    "content": [{"type": "text", "text": json.dumps({"error": str(error)})}],
                    "isError": True,
                }
            return {
                "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
                "structuredContent": payload,
                "isError": False,
            }
        if method == "resources/list":
            return {"resources": self._list_resources()}
        if method == "resources/templates/list":
            return {
                "resourceTemplates": [
                    {
                        "uriTemplate": "hillclimb://runs/{run_id}/status",
                        "name": "Hillclimb run status",
                        "mimeType": "application/json",
                    },
                    {
                        "uriTemplate": "hillclimb://runs/{run_id}/events",
                        "name": "Hillclimb run event tail",
                        "mimeType": "application/json",
                    },
                ]
            }
        if method == "resources/read":
            uri = params.get("uri")
            if not isinstance(uri, str):
                raise _JsonRpcError(-32602, "resources/read requires uri")
            return self._read_resource(uri)
        if method == "resources/subscribe":
            uri = params.get("uri")
            if not isinstance(uri, str):
                raise _JsonRpcError(-32602, "resources/subscribe requires uri")
            raw_meta = params.get("_meta")
            meta = raw_meta if isinstance(raw_meta, Mapping) else {}
            return self._subscribe_resource(uri, meta=meta)
        if method == "resources/unsubscribe":
            uri = params.get("uri")
            if not isinstance(uri, str):
                raise _JsonRpcError(-32602, "resources/unsubscribe requires uri")
            return self._unsubscribe_resource(uri)
        raise _JsonRpcError(-32601, f"Method not found: {method}")

    def close(self) -> None:
        with self._write_lock:
            self._closed = True
        with self._state_lock:
            sessions = list(self._runs.values())
        for session in sessions:
            session.stop_event.set()
            with session.lock:
                stream_http = session.stream_http
                session.condition.notify_all()
            if stream_http is not None:
                stream_http.close()
        for session in sessions:
            thread = session.watcher_thread
            if thread is not None and thread.is_alive():
                thread.join(timeout=1.0)

    def serve(self, *, inp: TextIO = sys.stdin) -> None:
        """Read newline-delimited JSON-RPC 2.0 frames from `inp` until EOF."""
        try:
            while True:
                line = inp.readline()
                if not line:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    decoded = json.loads(stripped)
                except ValueError as error:
                    self._emit_json(
                        {
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {"code": -32700, "message": f"Parse error: {error}"},
                        }
                    )
                    continue
                response = self.handle_message(decoded)
                if response is not None:
                    self._emit_json(response)
        finally:
            self.close()


def serve_stdio(
    *,
    daemon: Optional[str] = None,
    url: Optional[str] = None,
    auth_token_env: Optional[str] = None,
    run_id: Optional[UUID] = None,
    directory: Optional[Path] = None,
    inp: TextIO = sys.stdin,
    out: TextIO = sys.stdout,
) -> None:
    """Start the zero-dependency Hillclimb MCP server over stdio."""
    server = McpServer(
        out=out,
        daemon=daemon,
        default_url=url,
        auth_token_env=auth_token_env,
        default_run_id=run_id,
        default_directory=directory,
    )
    server.serve(inp=inp)
