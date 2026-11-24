"""Zero-import structural teleprompter adapters for DSPy-shaped modules."""

from __future__ import annotations

import copy
import dataclasses
import inspect
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

from ..client import HillclimbError
from ..config_toml import OptimizerConfig, WriteMaskConfig
from ..sdk import OptimizationResult
from ..sdk import optimize as sdk_optimize

__all__ = [
    "GEPA",
    "HillclimbGEPA",
    "HillclimbMIPRO",
    "MIPROv2",
    "ModuleProtocol",
    "PredictorProtocol",
    "SignatureProtocol",
]


@runtime_checkable
class SignatureProtocol(Protocol):
    instructions: str


@runtime_checkable
class PredictorProtocol(Protocol):
    signature: Any


@runtime_checkable
class ModuleProtocol(Protocol):
    def named_predictors(self) -> Iterable[tuple[str, Any]]: ...


def _extract_predictors(student: Any) -> list[tuple[str, Any]]:
    named = getattr(student, "named_predictors", None)
    if callable(named):
        predictors = [(str(name), pred) for name, pred in named()]
        if predictors:
            return predictors
    if hasattr(student, "signature") or hasattr(student, "extended_signature"):
        return [("predictor", student)]
    raise HillclimbError(
        "student module must expose named_predictors() or a signature attribute"
    )


def _read_instruction(predictor: Any) -> str:
    for attr in ("signature", "extended_signature"):
        sig = getattr(predictor, attr, None)
        if sig is not None and hasattr(sig, "instructions"):
            return str(sig.instructions)
    if hasattr(predictor, "instructions"):
        return str(predictor.instructions)
    raise HillclimbError("predictor has no signature.instructions or instructions attribute")


def _write_instruction(predictor: Any, new_instruction: str) -> None:
    for attr in ("signature", "extended_signature"):
        sig = getattr(predictor, attr, None)
        if sig is None or not hasattr(sig, "instructions"):
            continue
        with_instr = getattr(sig, "with_instructions", None)
        if callable(with_instr):
            setattr(predictor, attr, with_instr(new_instruction))
            return
        if dataclasses.is_dataclass(sig):
            setattr(predictor, attr, dataclasses.replace(sig, instructions=new_instruction))
            return
        try:
            sig.instructions = new_instruction
            return
        except AttributeError as error:
            raise HillclimbError(
                f"cannot update frozen signature attribute {attr!r}: {error}"
            ) from error
    if hasattr(predictor, "instructions"):
        predictor.instructions = new_instruction
        return
    raise HillclimbError("predictor has no writable instruction field")


def _extract_input_keys(predictors: Sequence[tuple[str, Any]]) -> Optional[tuple[str, ...]]:
    first_sig = getattr(predictors[0][1], "signature", None) if predictors else None
    input_fields = getattr(first_sig, "input_fields", None)
    if isinstance(input_fields, Mapping):
        return tuple(str(k) for k in input_fields)
    if isinstance(input_fields, Sequence) and not isinstance(input_fields, (str, bytes)):
        keys: list[str] = []
        for field_item in input_fields:
            if isinstance(field_item, str):
                keys.append(field_item)
            elif hasattr(field_item, "name"):
                keys.append(str(field_item.name))
        if keys:
            return tuple(keys)
    return None


def _invoke_module(
    module: Any, example: Any, input_keys: Optional[tuple[str, ...]]
) -> Any:
    inputs_fn = getattr(example, "inputs", None)
    if callable(inputs_fn):
        raw_inputs = inputs_fn()
        if isinstance(raw_inputs, Mapping):
            return module(**dict(raw_inputs))
    if isinstance(example, Mapping):
        if input_keys:
            selected = {key: example[key] for key in input_keys if key in example}
            return module(**selected)
        try:
            sig = inspect.signature(module)
            params = list(sig.parameters.values())
        except (TypeError, ValueError):
            params = []
        if len(params) == 1 and params[0].kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ) and params[0].name in {"example", "row", "item", "sample", "x"}:
            return module(example)
        filtered = {
            k: v
            for k, v in example.items()
            if k not in {"answer", "label", "expected", "gold", "target", "output"}
        }
        return module(**filtered)
    return module(example)


def _normalize_metric_output(raw: object) -> Any:
    if isinstance(raw, tuple) and len(raw) == 2:
        return float(raw[0]), str(raw[1])
    if isinstance(raw, Mapping) and "score" in raw:
        if "cost_micros" in raw or "tokens" in raw or "metrics" in raw:
            return dict(raw)
        return float(raw["score"]), str(raw.get("feedback", ""))
    if hasattr(raw, "score"):
        score = float(getattr(raw, "score"))
        feedback = str(getattr(raw, "feedback", ""))
        cost_micros = getattr(raw, "cost_micros", None)
        tokens = getattr(raw, "tokens", None)
        if cost_micros is not None or tokens is not None:
            return {
                "score": score,
                "feedback": feedback,
                "cost_micros": int(cost_micros or 0),
                "tokens": int(tokens or 0),
            }
        return score, feedback
    if isinstance(raw, (int, float, bool)):
        return float(raw), ""
    raise HillclimbError(
        "DSPy-style metric must return bool, float, (score, feedback), or {score, feedback}"
    )


class _BaseTeleprompter:
    """Shared teleprompter base routing DSPy-shaped modules through ``hillclimb.sdk.optimize``."""

    _algorithm: str = "gepa"

    def __init__(
        self,
        *,
        metric: Optional[Callable[..., Any]] = None,
        max_metric_calls: Optional[int] = None,
        cap_usd: Optional[float] = 1.0,
        config: Optional[OptimizerConfig | Path | str] = None,
        proposer: Optional[Callable[..., Any]] = None,
        proposer_url: Optional[str] = None,
        proposer_model: Optional[str] = None,
        proposer_effort: Optional[str] = None,
        daemon: Optional[str] = None,
        directory: Optional[Path | str] = None,
        minibatch: Optional[int] = None,
        max_rounds: Optional[int] = None,
        seed: int = 0,
        write_mask: Optional[WriteMaskConfig] = None,
        loopback: bool = False,
        on_event: Optional[Callable[[Mapping[str, object]], None]] = None,
    ) -> None:
        self.metric = metric
        self.max_metric_calls = max_metric_calls
        self.cap_usd = cap_usd
        self.config = config
        self.proposer = proposer
        self.proposer_url = proposer_url
        self.proposer_model = proposer_model
        self.proposer_effort = proposer_effort
        self.daemon = daemon
        self.directory = directory
        self.minibatch = minibatch
        self.max_rounds = max_rounds
        self.seed = seed
        self.write_mask = write_mask
        self.loopback = loopback
        self.on_event = on_event
        self.last_result: Optional[OptimizationResult] = None

    def compile(
        self,
        student: ModuleProtocol | Any,
        *,
        trainset: Sequence[Any],
        valset: Optional[Sequence[Any]] = None,
        metric: Optional[Callable[..., Any]] = None,
        heldout: Optional[Sequence[Any]] = None,
        **overrides: Any,
    ) -> Any:
        active_metric = metric or self.metric
        if active_metric is None:
            raise HillclimbError(f"{type(self).__name__}.compile requires a metric callable")
        if not trainset:
            raise HillclimbError("compile requires a non-empty trainset")
        if valset is None:
            if len(trainset) < 2:
                raise HillclimbError("compile requires valset or at least 2 trainset examples")
            split_idx = max(1, len(trainset) // 2)
            effective_train = list(trainset[:split_idx])
            effective_val = list(trainset[split_idx:])
        else:
            effective_train = list(trainset)
            effective_val = list(valset)

        predictors = _extract_predictors(student)
        seed_map = {name: _read_instruction(pred) for name, pred in predictors}
        input_keys = _extract_input_keys(predictors)
        copy_lock = threading.Lock()

        def evaluate(policy: Any, row: Any) -> tuple[float, str]:
            policy_map = (
                {next(iter(seed_map)): str(policy)}
                if isinstance(policy, str)
                else {str(k): str(v) for k, v in policy.items()}
            )
            with copy_lock:
                candidate_module = copy.deepcopy(student)
            for name, pred in _extract_predictors(candidate_module):
                if name in policy_map:
                    _write_instruction(pred, policy_map[name])
            prediction = _invoke_module(candidate_module, row, input_keys)
            return _normalize_metric_output(active_metric(row, prediction))

        result = sdk_optimize(
            seed=seed_map,
            evaluate=evaluate,
            train=effective_train,
            val=effective_val,
            heldout=heldout,
            algo=self._algorithm,
            cap_usd=overrides.get("cap_usd", self.cap_usd),
            config=overrides.get("config", self.config),
            proposer=overrides.get("proposer", self.proposer),
            proposer_url=overrides.get("proposer_url", self.proposer_url),
            proposer_model=overrides.get("proposer_model", self.proposer_model),
            proposer_effort=overrides.get("proposer_effort", self.proposer_effort),
            daemon=overrides.get("daemon", self.daemon),
            directory=overrides.get("directory", self.directory),
            minibatch=overrides.get("minibatch", self.minibatch),
            max_rounds=overrides.get("max_rounds", self.max_rounds),
            max_metric_calls=overrides.get("max_metric_calls", self.max_metric_calls),
            seed_num=overrides.get("seed", self.seed),
            write_mask=overrides.get("write_mask", self.write_mask),
            loopback=overrides.get("loopback", self.loopback),
            on_event=overrides.get("on_event", self.on_event),
        )
        self.last_result = result

        compiled = copy.deepcopy(student)
        for name, pred in _extract_predictors(compiled):
            if name in result.best_candidate_map:
                _write_instruction(pred, result.best_candidate_map[name])
        try:
            compiled.hillclimb_result = result
        except AttributeError:
            pass
        return compiled


class HillclimbGEPA(_BaseTeleprompter):
    """DSPy-compatible GEPA teleprompter backed by Hillclimb."""

    _algorithm = "gepa"


class HillclimbMIPRO(_BaseTeleprompter):
    """DSPy-compatible MIPROv2 teleprompter backed by Hillclimb."""

    _algorithm = "mipro"


GEPA = HillclimbGEPA
MIPROv2 = HillclimbMIPRO
