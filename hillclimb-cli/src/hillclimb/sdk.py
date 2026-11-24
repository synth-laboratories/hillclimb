"""High-level ergonomic Python SDK over the Hillclimb daemon and targets."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import socket
import sqlite3
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, closing
from dataclasses import dataclass, field
from pathlib import Path
from secrets import randbits
from typing import Any, Optional
from uuid import UUID, uuid4

import httpx
import zstandard

from .client import Client, Command, Event, HillclimbError, Json, Run, launch
from .config_toml import (
    LimitsConfig,
    ObjectiveCandidate,
    ObjectivesConfig,
    OptimizerConfig,
    OptimizerConfigError,
    PolicyConfig,
    ProposerConfig,
    ScoringConfig,
    StopConfig,
    TargetConfig,
    WriteMaskConfig,
    WriteMaskViolation,
    load_optimizer_toml,
    materialize_problem_context,
    select_champion,
)
from .launcher import ALGORITHM_CHOICES, canonical_algorithm, default_daemon

__all__ = [
    "Campaign",
    "ContainerTarget",
    "HarborTarget",
    "OnlineClient",
    "OptimizationResult",
    "optimize",
]

_POLICY_SCHEMA = "synth.policy-revision.v1"
_PROFILE = "next_rollout_update.v1"
_TRACE_SCHEMA = "synth.trace.v5"
_TERMINAL_STATUSES = frozenset({"cancelled", "completed", "failed", "in_doubt"})
_MAX_SUN_PATH_BYTES = 95


def _new_run_id() -> UUID:
    timestamp = time.time_ns() // 1_000_000
    return UUID(
        int=(timestamp << 80)
        | (7 << 76)
        | (randbits(12) << 64)
        | (2 << 62)
        | randbits(62)
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_hex(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _policy_digest(policy_content: Mapping[str, object]) -> str:
    """Compute the Rust-compatible ``PolicyContent::id()`` digest (struct field order, sorted maps)."""
    ordered = {
        "prompts": dict(sorted((str(k), str(v)) for k, v in (policy_content.get("prompts") or {}).items())),  # type: ignore[union-attr]
        "harness": dict(sorted((str(k), str(v)) for k, v in (policy_content.get("harness") or {}).items())),  # type: ignore[union-attr]
        "environment": dict(sorted((str(k), str(v)) for k, v in (policy_content.get("environment") or {}).items())),  # type: ignore[union-attr]
        "model_route": str(policy_content["model_route"]),
        "sampler_weights": policy_content.get("sampler_weights"),
    }
    raw = json.dumps(
        ordered,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _allocate_socket(run_id: UUID, explicit_socket: Optional[Path] = None) -> Path:
    if explicit_socket is not None:
        return explicit_socket.expanduser().resolve()
    located = Run.locate(run_id).socket
    if len(os.fsencode(str(located))) <= _MAX_SUN_PATH_BYTES:
        return located
    short = Path(tempfile.gettempdir()) / f"hc-{run_id.hex[:12]}.sock"
    if len(os.fsencode(str(short))) <= _MAX_SUN_PATH_BYTES:
        return short
    return Path(f"/tmp/hc-{run_id.hex[:12]}.sock")


def _terminate_daemon_pid(pid: object) -> None:
    if not isinstance(pid, int) or pid <= 0:
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.02)


def _serve_fastapi(app: Any, stack: ExitStack) -> str:
    import uvicorn

    listener = socket.socket()
    listener.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind(("127.0.0.1", 0))
    except OSError:
        listener.close()
        raise
    port = int(listener.getsockname()[1])
    origin = f"http://127.0.0.1:{port}"
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="critical", access_log=False)
    )
    worker = threading.Thread(
        target=server.run, kwargs={"sockets": [listener]}, daemon=True, name="hc-loopback-http"
    )
    worker.start()

    def _stop() -> None:
        server.should_exit = True
        worker.join(5.0)
        listener.close()

    stack.callback(_stop)
    for _ in range(250):
        if server.started:
            return origin
        if not worker.is_alive():
            break
        time.sleep(0.01)
    raise HillclimbError("loopback HTTP server failed to start")


@dataclass(frozen=True, slots=True)
class ContainerTarget:
    """HTTP Containers-compatible optimization target."""

    base_url: str
    namespace: str = "default"
    harness: str = "prompt_program"
    provider: str = "openai"
    timeout_seconds: int = 30
    tasks: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    metadata_sha256: Optional[str] = None
    seed_policy: Optional[Mapping[str, object]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str) or not self.base_url.startswith(("http://", "https://")):
            raise HillclimbError(f"ContainerTarget requires an HTTP(S) base_url, got {self.base_url!r}")
        if self.timeout_seconds <= 0:
            raise HillclimbError("ContainerTarget timeout_seconds must be positive")

    def probe_metadata(self) -> tuple[dict[str, object], str]:
        base = self.base_url.rstrip("/")
        try:
            with httpx.Client(timeout=float(self.timeout_seconds), trust_env=False) as client:
                response = client.get(f"{base}/metadata")
                response.raise_for_status()
                metadata = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise HillclimbError(f"failed to probe container metadata at {base}: {error}") from error
        if not isinstance(metadata, dict):
            raise HillclimbError("container /metadata must return a JSON object")
        return metadata, _sha256_hex(metadata)

    def probe_program(self) -> Optional[dict[str, object]]:
        base = self.base_url.rstrip("/")
        try:
            with httpx.Client(timeout=float(self.timeout_seconds), trust_env=False) as client:
                response = client.get(f"{base}/program")
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                doc = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise HillclimbError(f"failed to probe container program at {base}: {error}") from error
        if not isinstance(doc, dict):
            raise HillclimbError("container /program must return a JSON object")
        return doc

    def to_runtime_config(
        self,
        *,
        seed_policy_content: Mapping[str, object],
        write_mask: WriteMaskConfig,
        role_profile: Mapping[str, object],
        rollout_admission: Mapping[str, object],
        tasks_override: Optional[Mapping[str, Mapping[str, object]]] = None,
    ) -> dict[str, object]:
        metadata_sha = self.metadata_sha256
        if metadata_sha is None:
            _, metadata_sha = self.probe_metadata()
        task_map = dict(tasks_override if tasks_override is not None else self.tasks)
        if not task_map:
            raise HillclimbError("ContainerTarget requires non-empty tasks")
        return {
            "container": {
                "seed_policy": dict(seed_policy_content),
                "write_mask": write_mask.to_rust_write_mask(),
                "target": {
                    "base_url": self.base_url.rstrip("/"),
                    "timeout_seconds": self.timeout_seconds,
                    "namespace": self.namespace,
                    "harness": self.harness,
                    "provider": self.provider,
                    "tasks": {str(k): dict(v) for k, v in task_map.items()},
                },
                "target_metadata_sha256": metadata_sha,
                "rollout": dict(rollout_admission),
                "role": dict(role_profile),
            }
        }


@dataclass(frozen=True, slots=True)
class HarborTarget:
    """Harbor isolated-verifier optimization target."""

    workspace: Path | str
    timeout_seconds: int = 120
    tasks: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    agent_relay_host: str = "host.docker.internal"
    modal: Optional[Mapping[str, object]] = None
    rollout_proxy: Optional[Mapping[str, object]] = None

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise HillclimbError("HarborTarget timeout_seconds must be positive")
        if not self.agent_relay_host or any(
            ch in self.agent_relay_host for ch in ("/", "\\", "@", "?", "#")
        ):
            raise HillclimbError(f"invalid Harbor agent_relay_host: {self.agent_relay_host!r}")
        for task_id, spec in self.tasks.items():
            agent_img = spec.get("agent_image")
            verifier_img = spec.get("verifier_image")
            if not agent_img or not verifier_img:
                raise HillclimbError(
                    f"Harbor task {task_id!r} requires both agent_image and verifier_image"
                )
            if agent_img == verifier_img:
                raise HillclimbError(
                    f"Harbor task {task_id!r} requires distinct agent and verifier images"
                )

    def to_runtime_config(
        self,
        *,
        seed_policy_content: Mapping[str, object],
        write_mask: WriteMaskConfig,
        role_profile: Mapping[str, object],
        rollout_admission: Mapping[str, object],
        tasks_override: Optional[Mapping[str, Mapping[str, str]]] = None,
    ) -> dict[str, object]:
        task_map = dict(tasks_override if tasks_override is not None else self.tasks)
        if not task_map:
            raise HillclimbError("HarborTarget requires non-empty tasks")
        rollout_proxy = dict(self.rollout_proxy) if self.rollout_proxy is not None else {
            "proxy": dict(role_profile["proxy"]),  # type: ignore[arg-type]
            "agent_relay_host": self.agent_relay_host,
        }
        return {
            "harbor": {
                "seed_policy": dict(seed_policy_content),
                "write_mask": write_mask.to_rust_write_mask(),
                "target": {
                    "workspace": str(Path(self.workspace).expanduser().resolve()),
                    "timeout_seconds": self.timeout_seconds,
                    "tasks": {str(k): dict(v) for k, v in task_map.items()},
                },
                "rollout": dict(rollout_admission),
                "rollout_proxy": rollout_proxy,
                "modal": dict(self.modal) if self.modal is not None else None,
                "role": dict(role_profile),
            }
        }


class OnlineClient:
    """Authenticated HTTP client for online decision-routing deployments."""

    def __init__(
        self,
        base_url: str,
        token: Optional[str] = None,
        *,
        timeout: float = 10.0,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise HillclimbError(f"OnlineClient requires an HTTP(S) base_url, got {base_url!r}")
        headers: dict[str, str] = {}
        if token is not None and token.strip():
            headers["Authorization"] = f"Bearer {token}"
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
            trust_env=False,
        )

    def __enter__(self) -> OnlineClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def decide(
        self, input_data: str | Mapping[str, object], **extra: object
    ) -> dict[str, object]:
        body: dict[str, object] = (
            {"text": input_data, **extra}
            if isinstance(input_data, str)
            else {**input_data, **extra}
        )
        return self._request("POST", "/v1/decisions", body)

    def record_outcome(
        self,
        exposure_id: str | UUID,
        label: object,
        *,
        source: Optional[str] = None,
        **extra: object,
    ) -> dict[str, object]:
        body: dict[str, object] = {"exposure_id": str(exposure_id), "label": label, **extra}
        if source is not None:
            body["source"] = source
        return self._request("POST", "/v1/outcomes", body)

    def pause_optimizer(self, *, op_id: Optional[UUID] = None) -> dict[str, object]:
        return self._request(
            "POST",
            "/v1/commands",
            {"command": "pause", "op_id": str(op_id or uuid4())},
        )

    def resume_optimizer(self, *, op_id: Optional[UUID] = None) -> dict[str, object]:
        return self._request(
            "POST",
            "/v1/commands",
            {"command": "resume", "op_id": str(op_id or uuid4())},
        )

    def status(self) -> dict[str, object]:
        return self._request("GET", "/v1/status")

    def health(self) -> dict[str, object]:
        return self._request("GET", "/healthz")

    def _request(
        self, method: str, path: str, body: Optional[Mapping[str, object]] = None
    ) -> dict[str, object]:
        try:
            response = self._http.request(method, path, json=body)
        except httpx.RequestError as error:
            raise HillclimbError(f"{method} {path}: {error}") from error
        if response.is_error:
            raise HillclimbError(f"HTTP {response.status_code}: {response.text}")
        data = response.json()
        if not isinstance(data, dict):
            raise HillclimbError(f"{method} {path} did not return a JSON object")
        return data


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    run_id: str
    status: str
    algorithm: str
    best_candidate: dict[str, str] | str
    best_candidate_map: dict[str, str]
    champion_policy_id: str
    seed_score: Optional[float]
    best_score: float
    val_scores: dict[str, float]
    heldout_score: Optional[float]
    seed_heldout_score: Optional[float]
    spent_usd: float
    total_tokens: int
    rounds: int
    metric_calls: int
    stop_reason: str
    pareto_frontier: tuple[str, ...]
    objective_metrics: dict[str, dict[str, float]]
    fell_back_to_seed: bool = False
    write_mask_refusals: int = 0
    run_directory: Optional[Path] = None
    events: tuple[dict[str, object], ...] = ()


class Campaign:
    """Managed lifecycle handle for a running or completed ``hillclimb-daemon`` campaign."""

    def __init__(self, run: Run, *, daemon: Optional[str] = None) -> None:
        self.run = run
        self._daemon = daemon or default_daemon()

    @classmethod
    def attach(
        cls,
        run_id: UUID | str,
        *,
        directory: Optional[Path | str] = None,
        socket_path: Optional[Path | str] = None,
        daemon: Optional[str] = None,
    ) -> Campaign:
        uid = UUID(str(run_id)) if not isinstance(run_id, UUID) else run_id
        dir_path = Path(directory).expanduser().resolve() if directory is not None else None
        located = Run.locate(uid, dir_path)
        sock = (
            Path(socket_path).expanduser().resolve()
            if socket_path is not None
            else _allocate_socket(uid)
        )
        return cls(Run(uid, located.directory, sock), daemon=daemon)

    @classmethod
    def start(
        cls,
        config: Mapping[str, object] | OptimizerConfig | Path | str,
        *,
        daemon: Optional[str] = None,
        directory: Optional[Path | str] = None,
        run_id: Optional[UUID] = None,
        socket_path: Optional[Path | str] = None,
        mode: str = "run",
        tick_millis: int = 5,
        optimizer_config: Optional[OptimizerConfig] = None,
        forbidden_context_paths: Sequence[Path | str] = (),
    ) -> Campaign:
        uid = run_id if run_id is not None else _new_run_id()
        if uid.version != 7:
            raise HillclimbError("Campaign run_id must be a UUIDv7")

        resolved_dir = (
            Path(directory).expanduser().resolve()
            if directory is not None
            else Run.locate(uid).directory
        )
        resolved_socket = _allocate_socket(
            uid, Path(socket_path) if socket_path is not None else None
        )
        run = Run(uid, resolved_dir, resolved_socket)

        if isinstance(config, (Path, str)):
            loaded = load_optimizer_toml(config)
            optimizer_config = optimizer_config or loaded
            raise HillclimbError(
                "Campaign.start with a declarative optimizer.toml requires a runtime target; "
                "use hillclimb.optimize(config=...) or pass a compiled CampaignConfig mapping"
            )
        if isinstance(config, OptimizerConfig):
            raise HillclimbError(
                "Campaign.start with OptimizerConfig requires a runtime target; "
                "use hillclimb.optimize(config=...) or pass a compiled CampaignConfig mapping"
            )
        if not isinstance(config, Mapping):
            raise HillclimbError("Campaign.start expects a CampaignConfig mapping")

        payload = dict(config)
        payload["run_id"] = str(uid)

        if optimizer_config is not None and optimizer_config.proposer.context.files:
            ctx_dest = run.directory / "native-role-context" / "problem_context"
            materialize_problem_context(
                optimizer_config,
                ctx_dest,
                forbidden_paths=forbidden_context_paths,
            )

        run.directory.parent.mkdir(parents=True, exist_ok=True)
        resolved_daemon = daemon or default_daemon()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", dir=run.directory.parent, encoding="utf-8"
        ) as temp:
            json.dump(payload, temp, ensure_ascii=False)
            temp.flush()
            launch(
                run,
                daemon=resolved_daemon,
                mode=mode,
                config=Path(temp.name),
                tick_millis=tick_millis,
            )
        return cls(run, daemon=resolved_daemon)

    def _client(self, *, timeout: float = 10.0) -> Client:
        return Client(self.run, timeout=timeout)

    def status(self) -> dict[str, object]:
        with self._client() as client:
            return dict(client.status())

    def visuals(self) -> dict[str, object]:
        with self._client() as client:
            return dict(client.visuals())

    def events(self, *, after_seq: int = 0) -> Iterator[Event]:
        with self._client(timeout=30.0) as client:
            yield from client.events(after_seq=after_seq)

    def pause(
        self,
        *,
        op_id: Optional[UUID] = None,
        expected_epoch: Optional[int] = None,
        enqueue: bool = True,
    ) -> dict[str, object]:
        with self._client() as client:
            return dict(
                client.command(
                    Command.PAUSE,
                    op_id=op_id,
                    expected_epoch=expected_epoch,
                    enqueue=enqueue,
                )
            )

    def resume(
        self,
        *,
        op_id: Optional[UUID] = None,
        expected_epoch: Optional[int] = None,
        enqueue: bool = True,
    ) -> dict[str, object]:
        with self._client() as client:
            return dict(
                client.command(
                    Command.RESUME,
                    op_id=op_id,
                    expected_epoch=expected_epoch,
                    enqueue=enqueue,
                )
            )

    def cancel(
        self,
        *,
        op_id: Optional[UUID] = None,
        expected_epoch: Optional[int] = None,
        enqueue: bool = True,
    ) -> dict[str, object]:
        with self._client() as client:
            return dict(
                client.command(
                    Command.CANCEL,
                    op_id=op_id,
                    expected_epoch=expected_epoch,
                    enqueue=enqueue,
                )
            )

    def steer(
        self,
        note: str,
        *,
        op_id: Optional[UUID] = None,
        expected_epoch: Optional[int] = None,
        enqueue: bool = True,
    ) -> dict[str, object]:
        with self._client() as client:
            return dict(
                client.command(
                    Command.STEER,
                    value=note,
                    op_id=op_id,
                    expected_epoch=expected_epoch,
                    enqueue=enqueue,
                )
            )

    def steer_proposer(
        self,
        note: str,
        *,
        next_proposals: Optional[int] = None,
        op_id: Optional[UUID] = None,
        expected_epoch: Optional[int] = None,
        enqueue: bool = True,
    ) -> dict[str, object]:
        with self._client() as client:
            return dict(
                client.command(
                    Command.STEER_PROPOSER,
                    value={"note": note, "next_proposals": next_proposals},
                    op_id=op_id,
                    expected_epoch=expected_epoch,
                    enqueue=enqueue,
                )
            )

    def inject_candidate(
        self,
        parent_id: str,
        intervention: Mapping[str, object],
        *,
        op_id: Optional[UUID] = None,
        expected_epoch: Optional[int] = None,
        enqueue: bool = True,
    ) -> dict[str, object]:
        with self._client() as client:
            return dict(
                client.command(
                    Command.INJECT_CANDIDATE,
                    value={"parent_id": parent_id, "intervention": dict(intervention)},
                    op_id=op_id,
                    expected_epoch=expected_epoch,
                    enqueue=enqueue,
                )
            )

    def slice(
        self,
        destination: Path | str,
        *,
        proposal: bool = False,
        intervention: Optional[Mapping[str, object]] = None,
        socket_path: Optional[Path | str] = None,
    ) -> Campaign:
        dest = Path(destination).expanduser().resolve()
        with self._client() as client:
            child_status = client.slice(dest, proposal=proposal, intervention=intervention)
        child_id = UUID(str(child_status["run_id"]))
        child_socket = _allocate_socket(
            child_id, Path(socket_path) if socket_path is not None else None
        )
        return Campaign(Run(child_id, dest, child_socket), daemon=self._daemon)

    def resume_daemon(self, *, tick_millis: int = 5) -> dict[str, object]:
        launch(
            self.run,
            daemon=self._daemon,
            mode="resume",
            tick_millis=tick_millis,
        )
        status = self.status()
        if status.get("status") == "paused":
            self.resume()
        return self.status()

    def wait(
        self,
        *,
        timeout: float = 120.0,
        poll_interval: float = 0.02,
        on_event: Optional[Callable[[Event], None]] = None,
        stop_check: Optional[Callable[[dict[str, object]], Optional[str]]] = None,
    ) -> dict[str, object]:
        deadline = time.monotonic() + timeout
        after_seq = 0
        triggered_reason: Optional[str] = None
        cancel_sent = False
        with self._client(timeout=10.0) as client:
            while time.monotonic() < deadline:
                status = dict(client.status())
                if on_event is not None and int(status.get("event_seq", 0)) > after_seq:
                    for row in self.ledger_rows(
                        "SELECT seq, envelope_json FROM event_log WHERE seq > ? ORDER BY seq",
                        (after_seq,),
                    ):
                        seq = int(row["seq"])
                        envelope = json.loads(str(row["envelope_json"]))
                        envelope.setdefault("seq", seq)
                        on_event(Event(seq, envelope))
                        after_seq = seq
                if "daemon_error" in status:
                    raise HillclimbError(str(status["daemon_error"]))
                if status.get("status") in _TERMINAL_STATUSES:
                    if triggered_reason is not None:
                        status["_sdk_stop_reason"] = triggered_reason
                    return status
                if stop_check is not None and not cancel_sent:
                    reason = stop_check(status)
                    if reason is not None:
                        triggered_reason = reason
                        cancel_sent = True
                        try:
                            client.command(Command.CANCEL, enqueue=True)
                        except HillclimbError:
                            pass
                time.sleep(poll_interval)
        raise HillclimbError(f"campaign {self.run.run_id} did not terminate within {timeout}s")

    def read_policy(self, policy_id: str) -> dict[str, object]:
        digest = policy_id.removeprefix("sha256:")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise HillclimbError(f"invalid policy_id digest: {policy_id!r}")
        blob = self.run.directory / "cas" / "sha256" / digest[:2] / f"{digest}.zst"
        if not blob.is_file():
            raise HillclimbError(f"policy artifact missing from CAS: {blob}")
        with blob.open("rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as reader:
            data = reader.read(1024 * 1024)
        if hashlib.sha256(data).hexdigest() != digest:
            raise HillclimbError(f"CAS policy digest mismatch for {policy_id}")
        decoded = json.loads(data.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise HillclimbError(f"CAS policy {policy_id} is not a JSON object")
        return decoded

    def champion(self) -> Optional[dict[str, object]]:
        status = self.status()
        policy_id = status.get("frozen_policy")
        if not isinstance(policy_id, str) or not policy_id:
            visuals = self.visuals()
            rows = visuals.get("candidates", {}).get("rows", [])  # type: ignore[union-attr]
            if rows:
                best = max(rows, key=lambda item: float(item.get("score", 0.0)))
                policy_id = str(best.get("candidate_id", ""))
        if not policy_id:
            return None
        content = self.read_policy(policy_id)
        return {"policy_id": policy_id, **content}

    def ledger_rows(self, sql: str, params: tuple[object, ...] = ()) -> list[dict[str, object]]:
        db_path = self.run.directory / "ledger.sqlite"
        if not db_path.is_file():
            return []
        with closing(sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)) as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute(sql, params).fetchall()]


def _normalize_eval_output(raw: object) -> tuple[float, str, dict[str, float], int, int]:
    if isinstance(raw, tuple) and len(raw) == 2:
        score, feedback = raw
        return float(score), str(feedback), {}, 0, 0
    if isinstance(raw, Mapping):
        score = float(raw["score"])
        feedback = str(raw.get("feedback", ""))
        metrics = {
            str(k): float(v)
            for k, v in (raw.get("metrics") or {}).items()
            if isinstance(v, (int, float))
        }
        cost_micros = int(raw.get("cost_micros", 0))
        tokens = int(raw.get("tokens", 0))
        return score, feedback, metrics, cost_micros, tokens
    if isinstance(raw, (int, float, bool)):
        return float(raw), "", {}, 0, 0
    raise HillclimbError(
        "evaluate(policy, row) must return float, (score, feedback), or a mapping with 'score'"
    )


class _LoopbackCallbackTarget:
    """Loopback FastAPI Containers-compatible target wrapping a Python evaluate callback."""

    def __init__(
        self,
        *,
        seed_map: Mapping[str, str],
        seed_was_str: bool,
        evaluate: Callable[[Any, Any], Any],
        train_tasks: Mapping[str, Any],
        val_tasks: Mapping[str, Any],
        write_mask: WriteMaskConfig,
        guidelines: ProposerConfig,
        algo: str,
        seed_policy_id: str,
        model_route: str,
    ) -> None:
        self.seed_map = dict(seed_map)
        self.seed_was_str = seed_was_str
        self.evaluate = evaluate
        self.train_tasks = dict(train_tasks)
        self.val_tasks = dict(val_tasks)
        self.all_tasks = {**self.train_tasks, **self.val_tasks}
        self.write_mask = write_mask
        self.guidelines = guidelines.guidelines
        self.algo = algo
        self.seed_policy_id = seed_policy_id
        self.model_route = model_route

        self._lock = threading.Lock()
        self.revisions: dict[str, dict[str, str]] = {}
        self.prepared: set[str] = set()
        self.rollouts: dict[str, dict[str, object]] = {}
        self.prompts_by_id: dict[str, dict[str, str]] = {seed_policy_id: dict(seed_map)}
        self.revision_to_prompts: dict[str, dict[str, str]] = {}
        self.val_scores_by_policy: dict[str, dict[str, float]] = {}
        self.val_metrics_by_policy: dict[str, dict[str, list[float]]] = {}
        self.metric_calls = 0
        self.total_tokens = 0
        self.write_mask_refusals = 0
        self.seen_task_ids: set[str] = set()
        self.candidate_order: list[str] = []

        self.metadata = {
            "runtime": {"runtime_id": "hillclimb-sdk-callback-target"},
            "program_route": "/program",
            "policy_revision": {
                "schema_version": _POLICY_SCHEMA,
                "profiles": [_PROFILE],
                "adapters": {"prompt_program": ["config", "prompt_fields"]},
            },
        }
        self.metadata_sha256 = _sha256_hex(self.metadata)

    def _canonical_prompts(self, prompts: Mapping[str, str]) -> dict[str, str]:
        canonical = {k: prompts.get(k, v) for k, v in self.seed_map.items()}
        if self.algo == "mipro":
            for k in self.seed_map:
                demo_key = f"{k}.demonstrations"
                raw = prompts.get(demo_key)
                if raw is not None and raw.strip() not in {"", "[]"}:
                    canonical[demo_key] = raw
        return canonical

    def _policy_arg(self, prompts: Mapping[str, str]) -> Any:
        if self.seed_was_str and len(self.seed_map) == 1:
            key = next(iter(self.seed_map))
            return prompts[key]
        return self._canonical_prompts(prompts)

    def policy_key(self, prompts: Mapping[str, str]) -> str:
        policy_content = {
            "prompts": dict(sorted(self._canonical_prompts(prompts).items())),
            "harness": {},
            "environment": {},
            "model_route": self.model_route,
            "sampler_weights": None,
        }
        return _policy_digest(policy_content)

    def _check_prompt_fields(self, prompt_fields: Mapping[str, object]) -> dict[str, str]:
        cleaned: dict[str, str] = {}
        for key, value in prompt_fields.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise WriteMaskViolation("prompt_fields must map strings to strings")
            cleaned[key] = value
        changed: dict[str, str] = {}
        for key, value in cleaned.items():
            if self.algo == "mipro" and key.endswith(".demonstrations"):
                continue
            if key not in self.seed_map or self.seed_map[key] != value:
                changed[key] = value
        if changed:
            self.write_mask.validate_patch(changed)
            for value in changed.values():
                self.guidelines.validate_output(value)
        return cleaned

    def build_app(self) -> Any:
        from fastapi import FastAPI, HTTPException

        from .program import (
            Authority,
            Capability,
            Direction,
            LeverOperation,
            Measurement,
            Split,
            Visibility,
            program_digest,
        )

        app = FastAPI()
        target = self

        @app.get("/metadata")
        def metadata() -> dict[str, object]:
            return target.metadata

        @app.get("/program")
        def program() -> dict[str, object]:
            levers = [
                {
                    "id": lever,
                    "operation": LeverOperation.REPLACE_TEXT,
                    "binding": lever,
                    "media_type": "text/plain",
                    "max_bytes": max(4096, target.guidelines.max_chars * 4),
                    "validate": "/policy-revisions",
                    "admit": "/policy-revisions/prepare",
                    "abi": None,
                    "constraints": list(target.guidelines.rules),
                }
                for lever in (target.write_mask.mutable_levers or tuple(target.seed_map))
            ]
            doc: dict[str, object] = {
                "schema": "hillclimb.optimize-anything.program.v1",
                "schema_version": "hillclimb.candidate_protocol.v1",
                "version": "prompt_program.v1",
                "program_id": "sdk_callback_target",
                "task": "sdk_callback_target",
                "summary": "Python SDK callback optimization target",
                "capabilities": [Capability.PREPARE_REVISION],
                "levers": levers,
                "sensors": [
                    {
                        "id": "sdk.reward",
                        "unit": "score",
                        "visibility": Visibility.SETTLEMENT_ONLY,
                        "authority": Authority.TARGET_EVALUATOR,
                        "measurement": Measurement.SETTLED,
                        "splits": [Split.TRAIN, Split.VALIDATION, Split.TEST],
                        "bounds": [0, 1],
                        "fields": ["reward"],
                    }
                ],
                "objective": {
                    "sensor": "sdk.reward",
                    "direction": Direction.MAXIMIZE,
                    "aggregation": "mean",
                    "replicate": "row",
                    "description": "Mean callback evaluation score.",
                },
                "fixed_options": {"profile": _PROFILE},
            }
            doc["contract_digest"] = program_digest(doc)
            return doc

        @app.post("/policy-revisions")
        def publish(document: dict[str, Any]) -> dict[str, str]:
            if document.get("schema_version") != _POLICY_SCHEMA:
                raise HTTPException(422, "unsupported policy revision schema")
            components = document.get("components") or {}
            prompt_fields = components.get("prompt_fields") or {}
            try:
                cleaned = target._check_prompt_fields(prompt_fields)
            except OptimizerConfigError as error:
                with target._lock:
                    target.write_mask_refusals += 1
                raise HTTPException(422, str(error)) from error
            revision_id = "polrev_" + _sha256_hex(document)
            with target._lock:
                target.revisions[revision_id] = cleaned
                pid = target.policy_key(cleaned)
                target.prompts_by_id[pid] = target._canonical_prompts(cleaned)
                target.revision_to_prompts[revision_id] = target.prompts_by_id[pid]
            return {"policy_revision_id": revision_id}

        @app.post("/policy-revisions/{revision_id}/prepare")
        def prepare(revision_id: str, body: dict[str, Any]) -> dict[str, object]:
            if revision_id not in target.revisions or body.get("profile") != _PROFILE:
                raise HTTPException(409, "unknown policy revision or profile")
            target.prepared.add(revision_id)
            return {
                "policy_revision_id": revision_id,
                "state": "ready",
                "actual_binding": {"policy_revision_id": revision_id},
            }

        @app.post("/rollout")
        def rollout(body: dict[str, Any]) -> dict[str, str]:
            revision_id = str(body.get("policy_revision_id", ""))
            task_id = str(body.get("task_id", ""))
            if revision_id not in target.prepared or revision_id not in target.revisions:
                raise HTTPException(409, "rollout requires a prepared policy revision")
            if task_id not in target.all_tasks:
                raise HTTPException(422, f"task_id {task_id!r} is not in authorized search splits")
            prompts = target.revisions[revision_id]
            row = target.all_tasks[task_id]
            t0 = time.perf_counter()
            raw = target.evaluate(target._policy_arg(prompts), row)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            score, feedback, extra_metrics, cost_micros, tokens = _normalize_eval_output(raw)
            bounded_score = max(0.0, min(1.0, float(score)))
            rollout_id = str(uuid4())
            pid = target.policy_key(prompts)

            with target._lock:
                target.metric_calls += 1
                target.total_tokens += max(0, tokens)
                target.seen_task_ids.add(task_id)
                if pid not in target.candidate_order:
                    target.candidate_order.append(pid)
                if task_id in target.val_tasks:
                    target.val_scores_by_policy.setdefault(pid, {})[task_id] = float(score)
                    bucket = target.val_metrics_by_policy.setdefault(pid, {})
                    bucket.setdefault("reward", []).append(float(score))
                    bucket.setdefault("cost_usd", []).append(cost_micros / 1_000_000.0)
                    bucket.setdefault("tokens", []).append(float(tokens))
                    bucket.setdefault("latency_ms", []).append(elapsed_ms)
                    for mk, mv in extra_metrics.items():
                        bucket.setdefault(mk, []).append(float(mv))
                target.rollouts[rollout_id] = {
                    "rollout_id": rollout_id,
                    "policy_revision_id": revision_id,
                    "task_id": task_id,
                    "reward": bounded_score,
                    "raw_score": float(score),
                    "feedback": feedback,
                    "cost_micros": max(0, cost_micros),
                    "tokens": max(0, tokens),
                }
            return {"rollout_id": rollout_id, "policy_revision_id": revision_id}

        @app.get("/rollouts/{rollout_id}")
        def get_rollout(rollout_id: str) -> dict[str, object]:
            item = target.rollouts.get(rollout_id)
            if item is None:
                raise HTTPException(404, "unknown rollout")
            return {
                "rollout_id": rollout_id,
                "policy_revision_id": item["policy_revision_id"],
                "status": "completed",
            }

        @app.get("/rollouts/{rollout_id}/reward")
        def get_reward(rollout_id: str) -> dict[str, object]:
            item = target.rollouts.get(rollout_id)
            if item is None:
                raise HTTPException(404, "unknown rollout")
            return {
                "rollout_id": rollout_id,
                "reward": item["reward"],
                "cost_micros": item["cost_micros"],
                "tokens": item["tokens"],
            }

        @app.get("/rollouts/{rollout_id}/trace")
        def get_trace(rollout_id: str) -> dict[str, object]:
            item = target.rollouts.get(rollout_id)
            if item is None:
                raise HTTPException(404, "unknown rollout")
            return {
                "schema_version": _TRACE_SCHEMA,
                "rollout_id": rollout_id,
                "closed": True,
                "events": [
                    {
                        "event_type": "task.input",
                        "payload": {"task_id": item["task_id"]},
                    },
                    {
                        "event_type": "task.scored",
                        "payload": {
                            "task_id": item["task_id"],
                            "reward": item["raw_score"],
                            "feedback": item["feedback"],
                        },
                    },
                ],
            }

        return app


class _LoopbackProposerServer:
    """Deterministic or callback-backed loopback OpenAI chat-completions proposer."""

    def __init__(
        self,
        *,
        proposer_fn: Optional[Callable[..., Any]],
        seed_map: Mapping[str, str],
        write_mask: WriteMaskConfig,
        guidelines: ProposerConfig,
    ) -> None:
        self.proposer_fn = proposer_fn
        self.seed_map = dict(seed_map)
        self.write_mask = write_mask
        self.guidelines = guidelines.guidelines
        self._lock = threading.Lock()
        self.turn = 0
        self.write_mask_refusals = 0
        self.last_refusal: Optional[str] = None
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _default_gepa_patch(self, lever: str, user_text: str) -> dict[str, str]:
        with self._lock:
            self.turn += 1
            turn = self.turn
        base = self.seed_map.get(lever, next(iter(self.seed_map.values())))
        feedback_hints: list[str] = []
        for line in user_text.splitlines():
            stripped = line.strip()
            if "feedback" in stripped.lower() or "expected" in stripped.lower() or "fail" in stripped.lower():
                feedback_hints.append(stripped[:160])
        rules = " ".join(self.guidelines.rules[:2])
        hint_block = f" Guidance: {rules}" if rules else ""
        detail = f" [Refined #{turn}: apply domain rules cleanly.{hint_block}]"
        if feedback_hints and "feedback" not in detail.lower():
            detail = f" [Refined #{turn}: address feedback accurately.{hint_block}]"
        candidate_text = f"{base.rstrip()}{detail}"
        if len(candidate_text) > self.guidelines.max_chars:
            candidate_text = candidate_text[: self.guidelines.max_chars]
        return {lever: candidate_text}

    def _handle_gepa(self, system_text: str, user_text: str) -> str:
        lever = (
            self.write_mask.mutable_levers[0]
            if self.write_mask.mutable_levers
            else next(iter(self.seed_map))
        )
        for candidate_lever in self.seed_map:
            if f"`{candidate_lever}`" in system_text or f"`{candidate_lever}`" in user_text:
                lever = candidate_lever
                break
        if self.proposer_fn is not None:
            feedback_lines = [
                line.strip()
                for line in user_text.splitlines()
                if any(
                    token in line.lower()
                    for token in ("feedback", "expected", "predicted", "score", "reward", "fail")
                )
            ]
            proposed = self.proposer_fn(
                {
                    "lever": lever,
                    "parent": dict(self.seed_map),
                    "feedback": feedback_lines,
                    "system": system_text,
                    "user": user_text,
                    "seed": dict(self.seed_map),
                }
            )
            if isinstance(proposed, str):
                patch = {lever: proposed}
            elif isinstance(proposed, Mapping):
                patch = dict(proposed.get("patch", proposed)) if "kind" in proposed else dict(proposed)
            else:
                raise WriteMaskViolation("proposer callable must return a str or mapping")
            try:
                self.write_mask.validate_patch(patch)
                for text in patch.values():
                    self.guidelines.validate_output(text)
            except OptimizerConfigError as error:
                with self._lock:
                    self.write_mask_refusals += 1
                    self.last_refusal = str(error)
                raise
            return json.dumps({"kind": "prompt_patch", "patch": patch})

        patch = self._default_gepa_patch(lever, user_text)
        return json.dumps({"kind": "prompt_patch", "patch": patch})

    def _handle_mipro(self, payload: Mapping[str, Any]) -> str:
        module = str(payload.get("module") or next(iter(self.seed_map)))
        count = int(payload.get("instruction_count") or 1)
        base = self.seed_map.get(module, next(iter(self.seed_map.values())))
        if self.proposer_fn is not None:
            proposed = self.proposer_fn(dict(payload))
            if isinstance(proposed, Mapping) and "instructions" in proposed:
                instructions = [str(item) for item in proposed["instructions"]][:count]
            elif isinstance(proposed, Sequence) and not isinstance(proposed, (str, bytes)):
                instructions = [str(item) for item in proposed][:count]
            else:
                instructions = [f"{proposed} (variant {idx + 1})" for idx in range(count)]
        else:
            strategies = (
                "Follow explicit decision boundaries and output format rules.",
                "Verify reasoning against all domain constraints before answering.",
                "Prioritize precision on ambiguous edge cases.",
            )
            instructions = [
                f"{base.rstrip()} [MIPRO #{idx + 1}: {strategies[idx % len(strategies)]}]"
                for idx in range(count)
            ]
        return json.dumps({"module": module, "instructions": instructions})

    def build_app(self) -> Any:
        from fastapi import FastAPI, HTTPException

        app = FastAPI()
        proposer = self

        @app.post("/v1/chat/completions")
        def completions(body: dict[str, Any]) -> dict[str, object]:
            messages = body.get("messages") or []
            system_text = "\n".join(
                str(msg.get("content", "")) for msg in messages if msg.get("role") == "system"
            )
            user_text = "\n".join(
                str(msg.get("content", "")) for msg in messages if msg.get("role") == "user"
            )
            try:
                parsed_user = json.loads(user_text) if user_text.strip().startswith("{") else None
            except json.JSONDecodeError:
                parsed_user = None

            try:
                mipro_payload: Optional[Mapping[str, Any]] = None
                if isinstance(parsed_user, Mapping):
                    if "module" in parsed_user and "instruction_count" in parsed_user:
                        mipro_payload = parsed_user
                    elif (
                        parsed_user.get("role") == "grounded_instruction_proposer"
                        and isinstance(parsed_user.get("input"), Mapping)
                        and "module" in parsed_user["input"]
                    ):
                        mipro_payload = dict(parsed_user["input"])
                        mipro_payload.setdefault("evidence", parsed_user.get("evidence", []))
                if mipro_payload is not None:
                    reply_text = proposer._handle_mipro(mipro_payload)
                else:
                    reply_text = proposer._handle_gepa(system_text, user_text)
            except OptimizerConfigError as error:
                raise HTTPException(422, str(error)) from error

            with proposer._lock:
                proposer.prompt_tokens += 24
                proposer.completion_tokens += 16
            return {
                "id": f"loopback-{uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": 0,
                "model": str(body.get("model") or "loopback-proposer"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 24,
                    "completion_tokens": 16,
                    "total_tokens": 40,
                    "cost_micros": 0,
                },
            }

        return app


def _build_optimizer_config(
    *,
    seed_map: Mapping[str, str],
    algo: str,
    cap_usd: Optional[float],
    config: Optional[OptimizerConfig | Path | str],
    minibatch: Optional[int],
    max_rounds: Optional[int],
    max_metric_calls: Optional[int],
    seed_num: Optional[int],
    write_mask: Optional[WriteMaskConfig],
) -> OptimizerConfig:
    if isinstance(config, (Path, str)):
        base = load_optimizer_toml(config)
    elif isinstance(config, OptimizerConfig):
        base = config
    else:
        canonical = canonical_algorithm(algo.lower()) if algo.lower() in ALGORITHM_CHOICES else algo.lower()
        base = OptimizerConfig(
            algorithm=canonical,
            seed=seed_num if seed_num is not None else 0,
            minibatch=minibatch if minibatch is not None else 4,
            policy=PolicyConfig(
                write_mask=write_mask or WriteMaskConfig(mutable_levers=tuple(seed_map)),
            ),
            stop=StopConfig(
                cap_usd=cap_usd if cap_usd is not None else 1.0,
                max_rounds=max_rounds,
                max_metric_calls=max_metric_calls,
            ),
        )
        return base

    effective_algo = canonical_algorithm(algo.lower()) if algo else base.algorithm
    effective_stop = StopConfig(
        cap_usd=cap_usd if cap_usd is not None else base.stop.cap_usd,
        max_metric_calls=(
            max_metric_calls if max_metric_calls is not None else base.stop.max_metric_calls
        ),
        max_rounds=max_rounds if max_rounds is not None else base.stop.max_rounds,
        max_wall_seconds=base.stop.max_wall_seconds,
        patience_rounds=base.stop.patience_rounds,
        target_score=base.stop.target_score,
    )
    effective_policy = PolicyConfig(
        model=base.policy.model,
        write_mask=write_mask if write_mask is not None else base.policy.write_mask,
    )
    return OptimizerConfig(
        schema_version=base.schema_version,
        algorithm=effective_algo,
        seed=seed_num if seed_num is not None else base.seed,
        minibatch=minibatch if minibatch is not None else base.minibatch,
        proposer=base.proposer,
        scoring=base.scoring,
        objectives=base.objectives,
        policy=effective_policy,
        target=base.target,
        limits=base.limits,
        stop=effective_stop,
        source_path=base.source_path,
    )


def _select_from_target_metrics(
    target: _LoopbackCallbackTarget,
    seed_policy_id: str,
    objectives: ObjectivesConfig,
    aggregator: Optional[Callable[[Mapping[str, float]], float]],
) -> tuple[str, dict[str, dict[str, float]], dict[str, float], bool, tuple[str, ...]]:
    candidates: list[ObjectiveCandidate] = []
    metric_table: dict[str, dict[str, float]] = {}
    val_scores: dict[str, float] = {}

    for pid in target.candidate_order:
        raw_metrics = target.val_metrics_by_policy.get(pid)
        if not raw_metrics:
            continue
        means = {
            metric_name: sum(values) / len(values)
            for metric_name, values in raw_metrics.items()
            if values
        }
        val_scores[pid] = means.get("reward", 0.0)
        metric_table[pid] = means
        candidates.append(
            ObjectiveCandidate(
                candidate_id=pid,
                metrics=means,
                is_seed=(pid == seed_policy_id),
                payload=target.prompts_by_id.get(pid),
            )
        )

    if not candidates:
        fallback_metrics = {spec.name: 0.0 for spec in objectives.objectives}
        return (
            seed_policy_id,
            {seed_policy_id: fallback_metrics},
            {seed_policy_id: 0.0},
            True,
            (seed_policy_id,),
        )

    can_use_objectives = all(
        spec.name in c.metrics
        for c in candidates
        for spec in objectives.objectives
    )
    if can_use_objectives:
        selection = select_champion(
            candidates,
            objectives,
            seed_id=seed_policy_id,
            aggregator=aggregator,
        )
        return (
            selection.candidate_id,
            metric_table,
            val_scores,
            selection.fell_back_to_seed,
            selection.frontier,
        )

    best_pid = max(val_scores, key=lambda pid: (val_scores[pid], pid))
    return best_pid, metric_table, val_scores, False, tuple(val_scores)


def _evaluate_heldout(
    *,
    evaluate: Callable[[Any, Any], Any],
    seed_map: Mapping[str, str],
    best_map: Mapping[str, str],
    seed_was_str: bool,
    heldout: Optional[Sequence[Any]],
) -> tuple[Optional[float], Optional[float]]:
    if not heldout:
        return None, None

    def _arg(prompts: Mapping[str, str]) -> Any:
        if seed_was_str and len(seed_map) == 1:
            return prompts[next(iter(seed_map))]
        return dict(prompts)

    seed_scores: list[float] = []
    champ_scores: list[float] = []
    for row in heldout:
        s_score, *_ = _normalize_eval_output(evaluate(_arg(seed_map), row))
        c_score, *_ = _normalize_eval_output(evaluate(_arg(best_map), row))
        seed_scores.append(s_score)
        champ_scores.append(c_score)
    return sum(champ_scores) / len(champ_scores), sum(seed_scores) / len(seed_scores)


def _run_inprocess_loopback(
    *,
    run_id: UUID,
    run_dir: Path,
    seed_map: Mapping[str, str],
    seed_was_str: bool,
    evaluate: Callable[[Any, Any], Any],
    train: Sequence[Any],
    val: Sequence[Any],
    heldout: Optional[Sequence[Any]],
    opt_cfg: OptimizerConfig,
    proposer: Optional[Callable[..., Any]],
    objective_aggregator: Optional[Callable[[Mapping[str, float]], float]],
    on_event: Optional[Callable[[Mapping[str, object]], None]],
) -> OptimizationResult:
    """Fast deterministic loopback driver used when no external daemon/proposer is attached."""
    run_dir.mkdir(parents=True, exist_ok=True)
    if opt_cfg.proposer.context.files:
        materialize_problem_context(opt_cfg, run_dir / "native-role-context" / "problem_context")

    def _arg(prompts: Mapping[str, str]) -> Any:
        if seed_was_str and len(seed_map) == 1:
            return prompts[next(iter(seed_map))]
        return dict(prompts)

    events: list[dict[str, object]] = []
    started = time.monotonic()
    metric_calls = 0
    total_tokens = 0
    spent_micros = 0
    write_mask_refusals = 0
    candidates_by_id: dict[str, dict[str, str]] = {}
    metrics_by_id: dict[str, dict[str, float]] = {}
    val_scores: dict[str, float] = {}

    def _eval_split(prompts: Mapping[str, str], rows: Sequence[Any]) -> tuple[dict[str, float], list[str]]:
        nonlocal metric_calls, total_tokens, spent_micros
        scores: list[float] = []
        costs: list[float] = []
        tokens_list: list[float] = []
        latencies: list[float] = []
        extra_acc: dict[str, list[float]] = {}
        feedbacks: list[str] = []
        for row in rows:
            t0 = time.perf_counter()
            raw = evaluate(_arg(prompts), row)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            sc, fb, extras, c_mic, tok = _normalize_eval_output(raw)
            metric_calls += 1
            total_tokens += max(0, tok)
            spent_micros += max(0, c_mic)
            scores.append(sc)
            costs.append(c_mic / 1_000_000.0)
            tokens_list.append(float(tok))
            latencies.append(elapsed_ms)
            if fb:
                feedbacks.append(fb)
            for mk, mv in extras.items():
                extra_acc.setdefault(mk, []).append(mv)
        n = max(1, len(rows))
        summary = {
            "reward": sum(scores) / n,
            "cost_usd": sum(costs) / n,
            "tokens": sum(tokens_list) / n,
            "latency_ms": sum(latencies) / n,
            **{k: sum(vs) / len(vs) for k, vs in extra_acc.items()},
        }
        return summary, feedbacks

    seed_id = "sha256:" + _sha256_hex({"prompts": dict(sorted(seed_map.items()))})
    candidates_by_id[seed_id] = dict(seed_map)
    seed_metrics, _ = _eval_split(seed_map, val)
    metrics_by_id[seed_id] = seed_metrics
    val_scores[seed_id] = seed_metrics["reward"]
    seed_score = seed_metrics["reward"]

    rounds = 0
    max_rounds = opt_cfg.stop.max_rounds if opt_cfg.stop.max_rounds is not None else 3
    non_improving = 0
    best_so_far = seed_score
    stop_reason = "max_rounds"
    parent_prompts = dict(seed_map)
    lever = (
        opt_cfg.policy.write_mask.mutable_levers[0]
        if opt_cfg.policy.write_mask.mutable_levers
        else next(iter(seed_map))
    )

    while True:
        reason = opt_cfg.stop.check_stop(
            spent_usd=spent_micros / 1_000_000.0,
            metric_calls=metric_calls,
            rounds_completed=rounds,
            elapsed_seconds=time.monotonic() - started,
            non_improving_rounds=non_improving,
            best_score=best_so_far,
        )
        if reason is not None:
            stop_reason = reason
            break
        if rounds >= max_rounds:
            stop_reason = "max_rounds"
            break

        rounds += 1
        mb_rows = list(train[: min(len(train), opt_cfg.minibatch)])
        _, mb_feedback = _eval_split(parent_prompts, mb_rows)

        if proposer is not None:
            proposed = proposer(
                {
                    "round": rounds,
                    "lever": lever,
                    "parent": dict(parent_prompts),
                    "feedback": mb_feedback,
                    "guidelines": opt_cfg.proposer.guidelines.to_manifest(),
                }
            )
            patch = (
                {lever: proposed}
                if isinstance(proposed, str)
                else dict(proposed.get("patch", proposed))
                if isinstance(proposed, Mapping)
                else {}
            )
        else:
            rules = " ".join(opt_cfg.proposer.guidelines.rules[:2])
            hint = f" {rules}" if rules else ""
            patch = {
                lever: f"{parent_prompts[lever].rstrip()} [Optimized round {rounds}.{hint}]"
            }

        try:
            clean_patch = opt_cfg.policy.write_mask.validate_patch(patch)
            for value in clean_patch.values():
                opt_cfg.proposer.guidelines.validate_output(value)
        except OptimizerConfigError as error:
            write_mask_refusals += 1
            raise WriteMaskViolation(str(error)) from error

        child_prompts = {**parent_prompts, **clean_patch}
        child_id = "sha256:" + _sha256_hex({"prompts": dict(sorted(child_prompts.items()))})
        candidates_by_id[child_id] = child_prompts
        child_metrics, _ = _eval_split(child_prompts, val)
        metrics_by_id[child_id] = child_metrics
        val_scores[child_id] = child_metrics["reward"]

        if child_metrics["reward"] > best_so_far:
            best_so_far = child_metrics["reward"]
            parent_prompts = child_prompts
            non_improving = 0
        else:
            non_improving += 1

        event = {
            "kind": "round_completed",
            "round": rounds,
            "candidate_id": child_id,
            "val_score": child_metrics["reward"],
        }
        events.append(event)
        if on_event is not None:
            on_event(event)

    obj_candidates = [
        ObjectiveCandidate(
            candidate_id=cid,
            metrics=metrics_by_id[cid],
            is_seed=(cid == seed_id),
        )
        for cid in candidates_by_id
    ]
    if all(
        spec.name in metrics_by_id[cid]
        for cid in candidates_by_id
        for spec in opt_cfg.objectives.objectives
    ):
        selection = select_champion(
            obj_candidates,
            opt_cfg.objectives,
            seed_id=seed_id,
            aggregator=objective_aggregator,
        )
        champion_id = selection.candidate_id
        fell_back = selection.fell_back_to_seed
        frontier = selection.frontier
    else:
        champion_id = max(val_scores, key=lambda cid: (val_scores[cid], cid))
        fell_back = False
        frontier = tuple(val_scores)

    best_map = candidates_by_id[champion_id]
    heldout_score, seed_heldout = _evaluate_heldout(
        evaluate=evaluate,
        seed_map=seed_map,
        best_map=best_map,
        seed_was_str=seed_was_str,
        heldout=heldout,
    )
    best_candidate: dict[str, str] | str = (
        best_map[next(iter(seed_map))] if seed_was_str and len(seed_map) == 1 else dict(best_map)
    )
    return OptimizationResult(
        run_id=str(run_id),
        status="completed",
        algorithm=opt_cfg.algorithm,
        best_candidate=best_candidate,
        best_candidate_map=dict(best_map),
        champion_policy_id=champion_id,
        seed_score=seed_score,
        best_score=val_scores[champion_id],
        val_scores=val_scores,
        heldout_score=heldout_score,
        seed_heldout_score=seed_heldout,
        spent_usd=spent_micros / 1_000_000.0,
        total_tokens=total_tokens,
        rounds=rounds,
        metric_calls=metric_calls,
        stop_reason=stop_reason,
        pareto_frontier=frontier,
        objective_metrics=metrics_by_id,
        fell_back_to_seed=fell_back,
        write_mask_refusals=write_mask_refusals,
        run_directory=run_dir,
        events=tuple(events),
    )


def optimize(
    seed: str | Mapping[str, str],
    evaluate: Optional[Callable[[Any, Any], Any]] = None,
    train: Sequence[Any] = (),
    val: Sequence[Any] = (),
    *,
    heldout: Optional[Sequence[Any]] = None,
    algo: str = "gepa",
    cap_usd: Optional[float] = 1.0,
    config: Optional[OptimizerConfig | Path | str] = None,
    target: Optional[ContainerTarget | HarborTarget] = None,
    proposer: Optional[Callable[..., Any]] = None,
    proposer_url: Optional[str] = None,
    proposer_model: Optional[str] = None,
    proposer_effort: Optional[str] = None,
    daemon: Optional[str] = None,
    directory: Optional[Path | str] = None,
    minibatch: Optional[int] = None,
    max_rounds: Optional[int] = None,
    max_metric_calls: Optional[int] = None,
    seed_num: Optional[int] = None,
    write_mask: Optional[WriteMaskConfig] = None,
    objective_aggregator: Optional[Callable[[Mapping[str, float]], float]] = None,
    loopback: bool = False,
    strict_write_mask: bool = True,
    on_event: Optional[Callable[[Mapping[str, object]], None]] = None,
) -> OptimizationResult:
    """Optimize prompts/policies from a Python callback or Container/Harbor target."""
    if isinstance(seed, str):
        if not seed.strip():
            raise HillclimbError("seed prompt must be non-empty")
        seed_map = {"stage2_system": seed}
        seed_was_str = True
    elif isinstance(seed, Mapping) and seed:
        seed_map = {str(k): str(v) for k, v in seed.items()}
        seed_was_str = False
    else:
        raise HillclimbError("seed must be a non-empty string or mapping of lever -> text")

    opt_cfg = _build_optimizer_config(
        seed_map=seed_map,
        algo=algo,
        cap_usd=cap_usd,
        config=config,
        minibatch=minibatch,
        max_rounds=max_rounds,
        max_metric_calls=max_metric_calls,
        seed_num=seed_num,
        write_mask=write_mask,
    )

    if evaluate is None and target is None:
        raise HillclimbError("optimize requires either an `evaluate` callback or a `target`")
    if evaluate is not None and (not train or not val):
        raise HillclimbError("optimize with a callback requires non-empty `train` and `val` splits")

    run_id = _new_run_id()
    run_dir = (
        Path(directory).expanduser().resolve()
        if directory is not None
        else Run.locate(run_id).directory
    )

    if loopback and proposer_url is None and evaluate is not None and daemon is None:
        resolved_daemon = default_daemon()
        daemon_available = Path(resolved_daemon).is_file() or os.path.sep not in resolved_daemon
    else:
        resolved_daemon = daemon or default_daemon()
        daemon_available = Path(resolved_daemon).is_file()

    if evaluate is not None and not daemon_available and not loopback:
        raise HillclimbError(
            f"hillclimb-daemon binary not found at {resolved_daemon!r}; pass daemon=... or loopback=True"
        )

    if evaluate is not None and loopback and not daemon_available:
        return _run_inprocess_loopback(
            run_id=run_id,
            run_dir=run_dir,
            seed_map=seed_map,
            seed_was_str=seed_was_str,
            evaluate=evaluate,
            train=train,
            val=val,
            heldout=heldout,
            opt_cfg=opt_cfg,
            proposer=proposer,
            objective_aggregator=objective_aggregator,
            on_event=on_event,
        )

    if evaluate is None:
        assert target is not None
        raise HillclimbError(
            "Direct ContainerTarget/HarborTarget optimization requires Campaign.start(...) "
            "with explicit role/task wiring or an evaluate callback"
        )

    assert evaluate is not None
    train_tasks = {f"train-{idx}": row for idx, row in enumerate(train)}
    val_tasks = {f"val-{idx}": row for idx, row in enumerate(val)}
    model_route = opt_cfg.policy.model
    seed_policy_content = {
        "prompts": dict(sorted(seed_map.items())),
        "harness": {},
        "environment": {},
        "model_route": model_route,
        "sampler_weights": None,
    }
    seed_policy_id = _policy_digest(seed_policy_content)

    # MIPROv2 writes `<lever>.demonstrations` alongside `<lever>`.
    effective_mask = opt_cfg.policy.write_mask
    daemon_mask = effective_mask.to_rust_write_mask()
    if opt_cfg.algorithm == "mipro" and daemon_mask.get("prompt_fields") is not None:
        fields = list(daemon_mask["prompt_fields"])  # type: ignore[arg-type]
        for key in list(fields):
            demo_key = f"{key}.demonstrations"
            if demo_key not in fields:
                fields.append(demo_key)
        daemon_mask["prompt_fields"] = fields

    callback_target = _LoopbackCallbackTarget(
        seed_map=seed_map,
        seed_was_str=seed_was_str,
        evaluate=evaluate,
        train_tasks=train_tasks,
        val_tasks=val_tasks,
        write_mask=effective_mask,
        guidelines=opt_cfg.proposer,
        algo=opt_cfg.algorithm,
        seed_policy_id=seed_policy_id,
        model_route=model_route,
    )

    with ExitStack() as stack:
        target_origin = _serve_fastapi(callback_target.build_app(), stack)
        if proposer_url is not None:
            upstream_origin = proposer_url.rstrip("/")
            loopback_proposer = None
        else:
            if not loopback and proposer is None:
                raise HillclimbError(
                    "optimize requires `proposer_url`, `proposer` callable, or `loopback=True`"
                )
            loopback_proposer = _LoopbackProposerServer(
                proposer_fn=proposer,
                seed_map=seed_map,
                write_mask=effective_mask,
                guidelines=opt_cfg.proposer,
            )
            upstream_origin = _serve_fastapi(loopback_proposer.build_app(), stack)

        cap_micros = max(1, int(round(opt_cfg.stop.cap_usd * 1_000_000)))
        queue = opt_cfg.limits.to_queue_limits()
        rollout_slots = max(1, int(queue["slots"]))
        rollout_ceiling_micros = max(
            1, min(25_000, cap_micros // max(4, rollout_slots * 2))
        )
        effect_ceiling_micros = max(
            rollout_ceiling_micros, min(100_000, max(1, cap_micros // 4))
        )
        effective_minibatch = max(1, min(len(train_tasks), opt_cfg.minibatch))
        iterations = opt_cfg.stop.max_rounds if opt_cfg.stop.max_rounds is not None else 3

        search_options: dict[str, object]
        final_test_ids: list[str] = []
        final_evaluation = "external_sealed"
        all_target_tasks = {
            **{k: {"row_id": k} for k in train_tasks},
            **{k: {"row_id": k} for k in val_tasks},
        }
        if opt_cfg.algorithm == "gepa":
            prompt_levers = list(effective_mask.mutable_levers or tuple(seed_map))
            search_options = {
                "mode": "adaptive",
                "reflector": "direct_reflector",
                "prompt_lever": prompt_levers[0],
                "prompt_levers": prompt_levers,
                "max_reflection_records": min(8, max(4, effective_minibatch)),
                "minibatch_early_stop": False,
                "guidance": opt_cfg.proposer.guidelines.to_gepa_guidance(),
            }
            if opt_cfg.stop.max_metric_calls is not None:
                search_options["max_search_rollouts"] = opt_cfg.stop.max_metric_calls
        elif opt_cfg.algorithm == "mipro":
            final_evaluation = "internal"
            final_id = "test-closeout-0"
            final_test_ids = [final_id]
            callback_target.all_tasks[final_id] = next(iter(val_tasks.values()))
            all_target_tasks[final_id] = {"row_id": final_id}
            search_options = {
                "modules": [
                    {"instruction_key": key, "depends_on": []}
                    for key in (effective_mask.mutable_levers or tuple(seed_map))
                ],
                "dataset_description": "Python SDK callback optimization",
                "demo_packs": 2,
                "max_demos": min(2, len(train_tasks)),
                "instructions": 3 if iterations >= 3 else 2,
                "proposer_examples": min(4, len(train_tasks)),
                "val_eval_every": 1,
                "tpe": {"startup": 2, "draws": 8, "gamma": 0.25},
            }
        else:
            final_evaluation = "internal"
            final_id = "test-closeout-0"
            final_test_ids = [final_id]
            callback_target.all_tasks[final_id] = next(iter(val_tasks.values()))
            all_target_tasks[final_id] = {"row_id": final_id}
            search_options = {}

        p_model = proposer_model or opt_cfg.proposer.model
        p_effort = proposer_effort or opt_cfg.proposer.effort
        rollout_tokens_max = min(8192, queue["tokens_per_minute"] - queue["proposer_tokens_per_minute"])
        role_tokens_max = min(8192, queue["proposer_tokens_per_minute"])
        role_proxy: dict[str, object] = {
            "upstream_origin": upstream_origin,
            "upstream_authorization_env": None,
            "request_ceiling_micros": effect_ceiling_micros,
            "request_tokens_max": role_tokens_max,
            "timeout_seconds": max(5, int(opt_cfg.limits.request_timeout_seconds)),
            "queue": queue,
        }
        if proposer_effort is not None and p_effort in {
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }:
            role_proxy["reasoning_effort"] = p_effort

        campaign_config: dict[str, object] = {
            "algorithm_id": opt_cfg.algorithm,
            "run_id": str(run_id),
            "search": {
                "seed": opt_cfg.seed,
                "seed_policy": seed_policy_id,
                "train": list(train_tasks),
                "validation": list(val_tasks),
                "final_test": final_test_ids,
                "iterations": iterations,
                "minibatch": effective_minibatch,
                "options": search_options,
            },
            "runtime": {
                "container": {
                    "seed_policy": seed_policy_content,
                    "write_mask": daemon_mask,
                    "target": {
                        "base_url": target_origin,
                        "timeout_seconds": max(5, int(opt_cfg.limits.request_timeout_seconds)),
                        "namespace": opt_cfg.target.namespace,
                        "harness": opt_cfg.target.harness,
                        "provider": opt_cfg.target.provider,
                        "tasks": all_target_tasks,
                    },
                    "target_metadata_sha256": callback_target.metadata_sha256,
                    "rollout": {
                        "ceiling_micros": rollout_ceiling_micros,
                        "tokens_max": rollout_tokens_max,
                        "queue": queue,
                    },
                    "role": {
                        "agent": {
                            "kind": "direct_reflector",
                            "endpoint": f"{upstream_origin}/v1/chat/completions",
                            "model": p_model,
                            "timeout_seconds": max(5, int(opt_cfg.limits.request_timeout_seconds)),
                            "bearer_env": None,
                            "output_tokens_max": 2048,
                        },
                        "runtime_revision": _sha256_hex(
                            {"model": p_model, "effort": p_effort, "origin": upstream_origin}
                        ),
                        "workspace_manifest_sha256": opt_cfg.proposer.guidelines.manifest_sha256.removeprefix(
                            "sha256:"
                        ),
                        "proxy": role_proxy,
                    },
                }
            },
            "cap_micros": cap_micros,
            "effect_ceiling_micros": effect_ceiling_micros,
            "cache_mode": "off",
            "cache_dir": None,
        }
        if opt_cfg.algorithm == "gepa":
            campaign_config["final_evaluation"] = final_evaluation

        campaign = Campaign.start(
            campaign_config,
            daemon=resolved_daemon,
            directory=run_dir,
            run_id=run_id,
            tick_millis=2,
            optimizer_config=opt_cfg,
        )
        initial_pid: Optional[int] = None
        try:
            raw_pid = campaign.status().get("pid")
            if isinstance(raw_pid, int) and raw_pid > 0:
                initial_pid = raw_pid
        except HillclimbError:
            pass

        started_at = time.monotonic()
        collected_events: list[dict[str, object]] = []
        last_best_val: Optional[float] = None
        non_improving = 0
        observed_candidates = 0

        def _check_stop(status: dict[str, object]) -> Optional[str]:
            nonlocal last_best_val, non_improving, observed_candidates
            if status.get("status") == "paused" and any(
                tok in str(status.get("reason") or "").lower()
                for tok in ("budget", "cap", "reservation")
            ):
                return "cap_usd"
            if status.get("frozen_policy") is not None:
                return None
            with callback_target._lock:
                calls = callback_target.metric_calls
                vals = [
                    sum(scores.values()) / len(scores)
                    for scores in callback_target.val_scores_by_policy.values()
                    if len(scores) == len(val_tasks)
                ]
                if len(vals) > observed_candidates:
                    observed_candidates = len(vals)
                    latest = vals[-1]
                    if last_best_val is None or latest > last_best_val:
                        last_best_val = latest
                        non_improving = 0
                    else:
                        non_improving += 1
            if loopback_proposer is not None and strict_write_mask and loopback_proposer.write_mask_refusals > 0:
                return "write_mask_refusal"
            native_iterations = opt_cfg.algorithm in {"gepa", "mipro"}
            rounds_done = 0 if native_iterations else (observed_candidates - 1 if observed_candidates > 0 else 0)
            return opt_cfg.stop.check_stop(
                spent_usd=float(status.get("spent_micros", 0)) / 1_000_000.0,
                metric_calls=calls,
                rounds_completed=rounds_done,
                elapsed_seconds=time.monotonic() - started_at,
                non_improving_rounds=non_improving,
                best_score=last_best_val,
            )

        def _collect(ev: Event) -> None:
            collected_events.append(dict(ev.envelope))
            if on_event is not None:
                on_event(ev.envelope)

        try:
            final_status = campaign.wait(
                timeout=max(30.0, (opt_cfg.stop.max_wall_seconds or 60.0) + 15.0),
                on_event=_collect,
                stop_check=_check_stop,
            )
            visuals = campaign.visuals()
        finally:
            daemon_pid = initial_pid
            try:
                status_now = campaign.status()
                raw_pid = status_now.get("pid")
                if isinstance(raw_pid, int) and raw_pid > 0:
                    daemon_pid = raw_pid
                if status_now.get("status") not in _TERMINAL_STATUSES:
                    campaign.cancel()
            except HillclimbError:
                pass
            _terminate_daemon_pid(daemon_pid)

    if strict_write_mask and (
        callback_target.write_mask_refusals > 0
        or (loopback_proposer is not None and loopback_proposer.write_mask_refusals > 0)
    ):
        detail = (
            loopback_proposer.last_refusal
            if loopback_proposer is not None and loopback_proposer.last_refusal
            else "proposer or target revision violated [policy.write_mask]"
        )
        raise WriteMaskViolation(detail)

    if final_status.get("status") in {"failed", "in_doubt"}:
        raise HillclimbError(
            f"campaign {run_id} ended with status {final_status.get('status')}: "
            f"{final_status.get('reason')}"
        )

    champion_id, metric_table, val_scores, fell_back, frontier = _select_from_target_metrics(
        callback_target,
        seed_policy_id,
        opt_cfg.objectives,
        objective_aggregator,
    )
    best_map = dict(callback_target.prompts_by_id.get(champion_id, seed_map))
    heldout_score, seed_heldout = _evaluate_heldout(
        evaluate=evaluate,
        seed_map=seed_map,
        best_map=best_map,
        seed_was_str=seed_was_str,
        heldout=heldout,
    )
    best_candidate: dict[str, str] | str = (
        best_map[next(iter(seed_map))] if seed_was_str and len(seed_map) == 1 else dict(best_map)
    )
    algo_visual = visuals.get("algo", {}) if isinstance(visuals, Mapping) else {}
    rounds = int(
        algo_visual.get("generation")  # type: ignore[union-attr]
        or max(0, len(callback_target.candidate_order) - 1)
    )
    stop_reason = str(
        final_status.get("_sdk_stop_reason")
        or algo_visual.get("search_stop_reason")  # type: ignore[union-attr]
        or final_status.get("reason")
        or final_status.get("status")
        or "completed"
    )
    proposer_tokens = (
        loopback_proposer.prompt_tokens + loopback_proposer.completion_tokens
        if loopback_proposer is not None
        else 0
    )
    return OptimizationResult(
        run_id=str(run_id),
        status=str(final_status.get("status", "completed")),
        algorithm=opt_cfg.algorithm,
        best_candidate=best_candidate,
        best_candidate_map=best_map,
        champion_policy_id=champion_id,
        seed_score=val_scores.get(seed_policy_id),
        best_score=val_scores.get(champion_id, 0.0),
        val_scores=val_scores,
        heldout_score=heldout_score,
        seed_heldout_score=seed_heldout,
        spent_usd=int(final_status.get("spent_micros", 0)) / 1_000_000.0,
        total_tokens=callback_target.total_tokens + proposer_tokens,
        rounds=rounds,
        metric_calls=callback_target.metric_calls,
        stop_reason=stop_reason,
        pareto_frontier=frontier,
        objective_metrics=metric_table,
        fell_back_to_seed=fell_back,
        write_mask_refusals=(
            callback_target.write_mask_refusals
            + (loopback_proposer.write_mask_refusals if loopback_proposer is not None else 0)
        ),
        run_directory=run_dir,
        events=tuple(collected_events),
    )
