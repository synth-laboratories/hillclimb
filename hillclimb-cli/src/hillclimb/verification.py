"""Bounded per-criterion verification for target evaluators.

The queue schedules and records verifier work. Targets own rubric prompts,
verdict decoding, and reward aggregation. Transport errors are propagated
without resending a call whose charge or outcome may be unknown.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from enum import StrEnum
from threading import BoundedSemaphore, Lock
from typing import Protocol

from .pipeline import VerificationIntent

__all__ = [
    "VerificationAttempt",
    "VerificationDecision",
    "VerificationQueue",
    "VerificationReceipt",
    "VerificationRequest",
    "VerificationStatus",
]

_RECEIPT_SCHEMA = "hillclimb.verification-receipt.v1"
_MAX_ATTEMPTS = 8
_THROUGHPUT_WINDOW_SECONDS = 30


class VerificationStatus(StrEnum):
    VERIFIED = "verified"
    INVALID_RESPONSE = "invalid_response"


class _Verifier(Protocol):
    def __call__(
        self, system_prompt: str, user_prompt: str
    ) -> tuple[str, Mapping[str, object]]: ...


@dataclass(frozen=True, slots=True)
class VerificationRequest:
    criterion_id: str
    system_prompt: str
    user_prompt: str

    def digest(self) -> str:
        """Identify the exact criterion and messages evaluated, without exporting them."""
        content = json.dumps(
            [self.criterion_id, self.system_prompt, self.user_prompt],
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        return "sha256:" + hashlib.sha256(content).hexdigest()


@dataclass(frozen=True, slots=True)
class VerificationDecision:
    met: bool
    explanation: str


@dataclass(frozen=True, slots=True)
class VerificationAttempt:
    attempt: int
    status: VerificationStatus
    elapsed_ms: float
    total_tokens: int | None
    cost_micros: int | None


@dataclass(frozen=True, slots=True)
class VerificationReceipt:
    criterion_id: str
    protocol: str
    status: VerificationStatus
    decision: VerificationDecision | None
    queued_ms: float
    elapsed_ms: float
    attempts: tuple[VerificationAttempt, ...]
    intent: VerificationIntent | None = None
    request_digest: str | None = None

    def to_payload(self) -> dict[str, object]:
        payload = {
            "schema": _RECEIPT_SCHEMA,
            "criterion_id": self.criterion_id,
            "protocol": self.protocol,
            "status": self.status.value,
            "decision": asdict(self.decision) if self.decision is not None else None,
            "request_digest": self.request_digest,
            "queued_ms": self.queued_ms,
            "elapsed_ms": self.elapsed_ms,
            "retries": len(self.attempts) - 1,
            "total_tokens": _total(
                tuple(attempt.total_tokens for attempt in self.attempts)
            ),
            "cost_micros": _total(
                tuple(attempt.cost_micros for attempt in self.attempts)
            ),
            "attempts": [
                {**asdict(attempt), "status": attempt.status.value}
                for attempt in self.attempts
            ],
        }
        if self.intent is not None:
            payload["binding"] = {
                **asdict(self.intent),
                "operation_id": str(self.intent.operation_id),
            }
        return payload


def _total(values: Sequence[int | None]) -> int | None:
    if None in values:
        return None
    return sum(value for value in values if value is not None)


def _usage(usage: Mapping[str, object], name: str) -> int | None:
    value = usage.get(name)
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f"verification usage {name} must be a nonnegative integer")
    return value


_Decoder = Callable[[str], VerificationDecision | None]


class VerificationQueue:
    """Bound worker concurrency and pending jobs across concurrent rollouts.

    Callers may block waiting to submit when the queue is full. Each verifier
    call must enforce its own transport deadline. Retry only decoded failures
    up to the configured attempt limit. Each retry retains the earlier usage.
    """

    def __init__(
        self,
        *,
        max_workers: int = 12,
        max_pending: int | None = None,
        max_attempts: int = 2,
    ) -> None:
        if type(max_workers) is not int or max_workers < 1:
            raise ValueError("verification max_workers must be a positive integer")
        if max_pending is None:
            max_pending = max_workers
        if type(max_pending) is not int or max_pending < 0:
            raise ValueError("verification max_pending must be a nonnegative integer")
        if type(max_attempts) is not int or not 1 <= max_attempts <= _MAX_ATTEMPTS:
            raise ValueError(
                f"verification max_attempts must be between 1 and {_MAX_ATTEMPTS}"
            )
        self.slots = max_workers
        self.max_pending = max_pending
        self.max_attempts = max_attempts
        self._admission = BoundedSemaphore(max_workers + max_pending)
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="hillclimb-verify"
        )
        self._lock = Lock()
        self._closed = False
        self._queued = 0
        self._in_flight = 0
        self._completed = 0
        self._retries = 0
        self._parse_failures = 0
        self._verification_failures = 0
        self._unknown_usage_attempts = 0
        self._rollouts_verified = 0
        self._elapsed_ms_sum = 0.0
        self._recent_done: deque[tuple[int, int]] = deque()
        self._started_at = time.monotonic()

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._pool.shutdown(wait=True, cancel_futures=False)

    def snapshot(self) -> dict[str, int | float]:
        now = time.monotonic()
        with self._lock:
            self._prune_recent(int(now))
            recent = sum(count for _, count in self._recent_done)
            window = max(1.0, min(_THROUGHPUT_WINDOW_SECONDS, now - self._started_at))
            return {
                "slots": self.slots,
                "max_pending": self.max_pending,
                "max_attempts": self.max_attempts,
                "queued": self._queued,
                "in_flight": self._in_flight,
                "completed_criteria": self._completed,
                "rollouts_verified": self._rollouts_verified,
                "retries": self._retries,
                "parse_failures": self._parse_failures,
                "verification_failures": self._verification_failures,
                "unknown_usage_attempts": self._unknown_usage_attempts,
                "crit_per_s": round(recent / window, 2),
                "overall_crit_per_s": round(
                    self._completed / max(0.1, now - self._started_at), 2
                ),
                "mean_ms": round(self._elapsed_ms_sum / max(1, self._completed), 1),
            }

    def verify_rollout(
        self,
        requests: Sequence[VerificationRequest],
        *,
        protocol: str,
        verifier: _Verifier,
        decode: _Decoder,
        intent: VerificationIntent | None = None,
    ) -> list[VerificationReceipt]:
        """Return receipts in request order even if calls complete out of order."""
        if not isinstance(protocol, str) or not protocol.strip():
            raise ValueError("verification protocol identity is required")
        if intent is not None and (
            not isinstance(intent, VerificationIntent) or protocol != intent.verifier.id
        ):
            raise ValueError(
                "verification protocol does not match the pinned verifier intent"
            )
        criteria = [request.criterion_id for request in requests]
        if any(not isinstance(key, str) or not key.strip() for key in criteria) or len(
            set(criteria)
        ) != len(criteria):
            raise ValueError(
                "verification criterion IDs must be nonblank and unique within a rollout"
            )
        with self._lock:
            if self._closed:
                raise RuntimeError("verification queue is closed")
        futures: list[Future[VerificationReceipt]] = []
        try:
            for request in requests:
                futures.append(
                    self._submit(request, protocol, verifier, decode, intent)
                )
        finally:
            wait(futures)
        receipts = [future.result() for future in futures]
        with self._lock:
            self._rollouts_verified += 1
        return receipts

    def _submit(
        self,
        request: VerificationRequest,
        protocol: str,
        verifier: _Verifier,
        decode: _Decoder,
        intent: VerificationIntent | None,
    ) -> Future[VerificationReceipt]:
        submitted_at = time.monotonic()
        self._admission.acquire()
        try:
            with self._lock:
                if self._closed:
                    raise RuntimeError("verification queue is closed")
                future = self._pool.submit(
                    self._verify_one,
                    request,
                    protocol,
                    verifier,
                    decode,
                    submitted_at,
                    intent,
                )
                self._queued += 1
                future.add_done_callback(lambda _: self._admission.release())
                return future
        except BaseException:
            self._admission.release()
            raise

    def _verify_one(
        self,
        request: VerificationRequest,
        protocol: str,
        verifier: _Verifier,
        decode: _Decoder,
        submitted_at: float,
        intent: VerificationIntent | None,
    ) -> VerificationReceipt:
        started_at = time.monotonic()
        with self._lock:
            self._queued -= 1
            self._in_flight += 1
        attempts: list[VerificationAttempt] = []
        decision = None
        failed = True
        started_attempts = 0
        try:
            for attempt in range(1, self.max_attempts + 1):
                attempt_started = time.monotonic()
                started_attempts = attempt
                raw, usage = verifier(request.system_prompt, request.user_prompt)
                decision = decode(raw)
                status = (
                    VerificationStatus.VERIFIED
                    if decision is not None
                    else VerificationStatus.INVALID_RESPONSE
                )
                attempts.append(
                    VerificationAttempt(
                        attempt,
                        status,
                        (time.monotonic() - attempt_started) * 1000,
                        _usage(usage, "total_tokens"),
                        _usage(usage, "cost_micros"),
                    )
                )
                if decision is not None:
                    break
            failed = False
            return VerificationReceipt(
                request.criterion_id,
                protocol,
                status,
                decision,
                (started_at - submitted_at) * 1000,
                (time.monotonic() - started_at) * 1000,
                tuple(attempts),
                intent,
                request.digest(),
            )
        finally:
            with self._lock:
                self._record_completion(
                    started_at, started_attempts, attempts, failed, decision
                )

    def _record_completion(
        self,
        started_at: float,
        started_attempts: int,
        attempts: Sequence[VerificationAttempt],
        failed: bool,
        decision: VerificationDecision | None,
    ) -> None:
        now = time.monotonic()
        second = int(now)
        self._in_flight -= 1
        self._completed += 1
        self._retries += max(0, started_attempts - 1)
        self._unknown_usage_attempts += started_attempts - len(attempts)
        self._unknown_usage_attempts += sum(
            attempt.total_tokens is None or attempt.cost_micros is None
            for attempt in attempts
        )
        self._verification_failures += failed
        self._parse_failures += not failed and decision is None
        self._elapsed_ms_sum += (now - started_at) * 1000
        self._prune_recent(second)
        if self._recent_done and self._recent_done[-1][0] == second:
            _, count = self._recent_done.pop()
            self._recent_done.append((second, count + 1))
        else:
            self._recent_done.append((second, 1))

    def _prune_recent(self, second: int) -> None:
        while (
            self._recent_done
            and second - self._recent_done[0][0] >= _THROUGHPUT_WINDOW_SECONDS
        ):
            self._recent_done.popleft()
