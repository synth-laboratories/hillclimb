"""Zero-import structural drop-in compatibility shim for ``gepa-ai`` adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

import httpx

from ..client import HillclimbError
from ..config_toml import OptimizerConfig, WriteMaskConfig
from ..sdk import OptimizationResult
from ..sdk import optimize as sdk_optimize

__all__ = [
    "AdapterProtocol",
    "EvaluationBatchProtocol",
    "GepaResult",
    "optimize",
]


@runtime_checkable
class EvaluationBatchProtocol(Protocol):
    outputs: Sequence[Any]
    scores: Sequence[float]


@runtime_checkable
class AdapterProtocol(Protocol):
    def evaluate(
        self,
        batch: Sequence[Any],
        candidate: Mapping[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatchProtocol: ...


@dataclass(frozen=True, slots=True)
class GepaResult:
    """Drop-in result object matching ``gepa-ai`` result accessors plus Hillclimb receipts."""

    best_candidate: dict[str, str]
    candidates: tuple[dict[str, str], ...]
    val_aggregate_scores: tuple[float, ...]
    best_idx: int
    spent_usd: float
    total_metric_calls: int
    run_id: str
    stop_reason: str
    heldout_score: Optional[float]
    optimization_result: OptimizationResult


def _extract_feedback(output: Any, trajectory: Any = None) -> str:
    if isinstance(output, Mapping):
        for key in ("Feedback", "feedback", "reason", "message"):
            if key in output and output[key]:
                return str(output[key])
    for attr in ("feedback", "Feedback", "reason"):
        val = getattr(output, attr, None)
        if val:
            return str(val)
    if trajectory is not None:
        return str(trajectory)
    return str(output) if output is not None else ""


def _build_target_url_evaluator(
    target_url: str, *, seed_candidate: Mapping[str, str], timeout: float = 30.0
) -> Callable[[Mapping[str, str], Any], tuple[float, str]]:
    base = target_url.rstrip("/")

    def evaluate(candidate: Mapping[str, str], row: Any) -> tuple[float, str]:
        task_id = str(row["task_id"]) if isinstance(row, Mapping) and "task_id" in row else str(row)
        task_payload = dict(row) if isinstance(row, Mapping) else {"row_id": task_id}
        revision_body = {
            "schema_version": "synth.policy-revision.v1",
            "namespace": "gepa_compat",
            "name": "hillclimb",
            "harness": "prompt_program",
            "components": {
                "prompt_fields": dict(candidate),
                "config": {"policy": {"provider": "openai", "model": "gpt-6-luna"}},
            },
        }
        with httpx.Client(base_url=base, timeout=timeout, trust_env=False) as client:
            pub = client.post("/policy-revisions", json=revision_body)
            pub.raise_for_status()
            revision_id = pub.json()["policy_revision_id"]
            prep = client.post(
                f"/policy-revisions/{revision_id}/prepare",
                json={
                    "operation_id": f"compat-{revision_id}",
                    "profile": "next_rollout_update.v1",
                    "timeout_s": int(timeout),
                },
            )
            prep.raise_for_status()
            roll = client.post(
                "/rollout",
                json={
                    "policy_revision_id": revision_id,
                    "seed": 0,
                    "task_id": task_id,
                    "task": task_payload,
                    "submission_mode": "sync",
                },
            )
            roll.raise_for_status()
            rollout_id = roll.json()["rollout_id"]
            rew = client.get(f"/rollouts/{rollout_id}/reward")
            rew.raise_for_status()
            rew_payload = rew.json()
            score = float(rew_payload["reward"])
            cost_micros = int(rew_payload.get("cost_micros") or 0)
            tokens = int(rew_payload.get("tokens") or 0)
            tr = client.get(f"/rollouts/{rollout_id}/trace")
            feedback = ""
            if tr.is_success:
                events = tr.json().get("events") or []
                if events:
                    feedback = str(events[-1].get("payload", {}))
            if cost_micros or tokens:
                return {
                    "score": score,
                    "feedback": feedback,
                    "cost_micros": cost_micros,
                    "tokens": tokens,
                }
            return score, feedback

    return evaluate


def optimize(
    seed_candidate: Mapping[str, str],
    trainset: Sequence[Any],
    valset: Optional[Sequence[Any]] = None,
    *,
    adapter: Optional[AdapterProtocol] = None,
    task_lm: Optional[Callable[..., Any]] = None,
    evaluator: Optional[Callable[[Any, Any], Any]] = None,
    reflection_lm: Optional[Callable[..., Any] | str] = None,
    max_metric_calls: Optional[int] = None,
    max_rounds: Optional[int] = None,
    cap_usd: float = 1.0,
    candidate_selection_strategy: str = "pareto",
    reflection_minibatch_size: int = 3,
    seed: int = 0,
    config: Optional[OptimizerConfig | Path | str] = None,
    heldout: Optional[Sequence[Any]] = None,
    proposer_url: Optional[str] = None,
    proposer_effort: Optional[str] = None,
    daemon: Optional[str] = None,
    directory: Optional[Path | str] = None,
    write_mask: Optional[WriteMaskConfig] = None,
    loopback: bool = False,
    target_url: Optional[str] = None,
    on_event: Optional[Callable[[Mapping[str, object]], None]] = None,
    **extra_options: Any,
) -> GepaResult:
    """Drop-in ``gepa-ai``-compatible ``optimize(...)`` shim routed through Hillclimb."""
    del task_lm, candidate_selection_strategy, extra_options
    if not isinstance(seed_candidate, Mapping) or not seed_candidate:
        raise HillclimbError("seed_candidate must be a non-empty mapping of component -> text")
    seed_map = {str(k): str(v) for k, v in seed_candidate.items()}

    if not trainset:
        raise HillclimbError("trainset must be non-empty")
    if valset is None:
        if len(trainset) < 2:
            raise HillclimbError("valset is required when trainset has fewer than 2 examples")
        mid = max(1, len(trainset) // 2)
        effective_train = list(trainset[:mid])
        effective_val = list(trainset[mid:])
    else:
        effective_train = list(trainset)
        effective_val = list(valset)

    custom_proposer: Optional[Callable[..., Any]] = (
        reflection_lm if callable(reflection_lm) else None
    )
    proposer_model = reflection_lm if isinstance(reflection_lm, str) else None

    if adapter is not None:
        if not callable(getattr(adapter, "evaluate", None)):
            raise HillclimbError("adapter must implement evaluate(batch, candidate, capture_traces=...)")

        def evaluate_row(candidate: Any, row: Any) -> Any:
            cand_dict = (
                {next(iter(seed_map)): str(candidate)}
                if isinstance(candidate, str)
                else {str(k): str(v) for k, v in candidate.items()}
            )
            batch_result = adapter.evaluate([row], cand_dict, capture_traces=True)
            scores = list(batch_result.scores)
            outputs = list(batch_result.outputs)
            trajectories = getattr(batch_result, "trajectories", None)
            if not scores:
                raise HillclimbError("adapter.evaluate returned empty scores")
            output = outputs[0] if outputs else None
            trajectory = trajectories[0] if trajectories else None
            feedback = _extract_feedback(output, trajectory)
            if isinstance(output, Mapping) and ("cost_micros" in output or "tokens" in output):
                return {
                    "score": float(scores[0]),
                    "feedback": feedback,
                    "cost_micros": int(output.get("cost_micros") or 0),
                    "tokens": int(output.get("tokens") or 0),
                }
            return float(scores[0]), feedback

        propose_hook = getattr(adapter, "propose_new_texts", None)
        if custom_proposer is None and callable(propose_hook):
            def _adapter_proposer(context: Mapping[str, Any]) -> Mapping[str, str]:
                lever = str(context.get("lever") or next(iter(seed_map)))
                parent = dict(context.get("parent") or seed_map)
                feedback_items = list(context.get("feedback") or ())
                reflective_dataset = {
                    lever: [{"Feedback": fb} for fb in feedback_items]
                }
                return dict(propose_hook(parent, reflective_dataset, [lever]))

            custom_proposer = _adapter_proposer
    elif evaluator is not None:
        evaluate_row = evaluator
    elif target_url is not None:
        evaluate_row = _build_target_url_evaluator(
            target_url, seed_candidate=seed_map
        )
    else:
        raise HillclimbError(
            "compat.gepa_ai.optimize requires `adapter`, `evaluator`, or `target_url`"
        )

    result = sdk_optimize(
        seed=seed_map,
        evaluate=evaluate_row,
        train=effective_train,
        val=effective_val,
        heldout=heldout,
        algo="gepa",
        cap_usd=cap_usd,
        config=config,
        proposer=custom_proposer,
        proposer_url=proposer_url,
        proposer_model=proposer_model,
        proposer_effort=proposer_effort,
        daemon=daemon,
        directory=directory,
        minibatch=max(1, reflection_minibatch_size),
        max_rounds=max_rounds,
        max_metric_calls=max_metric_calls,
        seed_num=seed,
        write_mask=write_mask,
        loopback=loopback,
        on_event=on_event,
    )

    ordered_ids = list(result.val_scores)
    scores_tuple = tuple(float(result.val_scores[cid]) for cid in ordered_ids)
    if not scores_tuple:
        scores_tuple = (result.best_score,)
    best_idx = (
        ordered_ids.index(result.champion_policy_id)
        if result.champion_policy_id in ordered_ids
        else 0
    )
    candidates: list[dict[str, str]] = [dict(seed_map)]
    if result.best_candidate_map != seed_map:
        candidates.append(dict(result.best_candidate_map))

    return GepaResult(
        best_candidate=dict(result.best_candidate_map),
        candidates=tuple(candidates),
        val_aggregate_scores=scores_tuple,
        best_idx=best_idx,
        spent_usd=result.spent_usd,
        total_metric_calls=result.metric_calls,
        run_id=result.run_id,
        stop_reason=result.stop_reason,
        heldout_score=result.heldout_score,
        optimization_result=result,
    )
