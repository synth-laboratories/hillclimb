"""Schema-validated declarative ``optimizer.toml`` configuration and runtime guards."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import stat
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Optional

from .client import HillclimbError
from .launcher import ALGORITHM_CHOICES, canonical_algorithm
from .verification import VerificationDecision, VerificationQueue, VerificationRequest

__all__ = [
    "AggregationMode",
    "ChampionSelection",
    "ContextFileEntry",
    "ContextManifest",
    "CriterionKind",
    "CriterionResult",
    "HybridScoreResult",
    "LimitsConfig",
    "ObjectiveCandidate",
    "ObjectiveDirection",
    "ObjectivesConfig",
    "ObjectiveSpec",
    "OptimizerConfig",
    "OptimizerConfigError",
    "PathSecurityError",
    "PolicyConfig",
    "ProposerConfig",
    "ProposerContextConfig",
    "ProposerGuidelinesConfig",
    "RubricCriterion",
    "RubricParseError",
    "RubricScoringConfig",
    "ScoringConfig",
    "ScoringMode",
    "StopConfig",
    "StyleExampleDigest",
    "TargetConfig",
    "VerifierKind",
    "VerifierScoringConfig",
    "WriteMaskConfig",
    "WriteMaskViolation",
    "load_optimizer_toml",
    "materialize_problem_context",
    "pareto_frontier",
    "parse_optimizer_toml",
    "score_hybrid",
    "select_champion",
    "validate_patch",
]

_SCHEMA_VERSION = "hillclimb.optimizer.v1"
_CONTEXT_MANIFEST_SCHEMA = "hillclimb.problem-context-manifest.v1"
_GUIDELINES_MANIFEST_SCHEMA = "hillclimb.proposer-guidelines.v1"
_HELD_OUT_STEMS = frozenset({"heldout", "held_out", "final_test"})


class OptimizerConfigError(HillclimbError, ValueError):
    """Raised when ``optimizer.toml`` or a derived runtime guard fails validation."""


class WriteMaskViolation(OptimizerConfigError):
    """Raised when a proposal or patch attempts to write a locked or undeclared lever."""


class PathSecurityError(OptimizerConfigError):
    """Raised on path traversal, symlink escape, missing input, or held-out split leaks."""


class RubricParseError(OptimizerConfigError):
    """Raised when rubric judge parse failures exceed the configured threshold."""


class ScoringMode(StrEnum):
    VERIFIER = "verifier"
    RUBRIC = "rubric"
    HYBRID = "hybrid"


class VerifierKind(StrEnum):
    PYTHON_CALLBACK = "python_callback"
    HTTP = "http"
    CONTAINER = "container"
    HARBOR = "harbor"


class CriterionKind(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    GUARDRAIL = "guardrail"


class ObjectiveDirection(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class AggregationMode(StrEnum):
    WEIGHTED_SUM = "weighted_sum"
    CONSTRAINED_ARGMAX = "constrained_argmax"
    LEXICOGRAPHIC = "lexicographic"
    PYTHON_CALLABLE = "python_callable"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _require_mapping(value: object, section: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise OptimizerConfigError(f"[{section}] must be a TOML table")
    return value


def _reject_unknown_keys(
    table: Mapping[str, object], allowed: frozenset[str], section: str
) -> None:
    unknown = set(table) - allowed
    if unknown:
        joined = ", ".join(sorted(unknown))
        raise OptimizerConfigError(f"[{section}] contains unknown key(s): {joined}")


def _require_nonempty_str(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise OptimizerConfigError(f"{label} must be a non-empty string")
    return value


def _require_str_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise OptimizerConfigError(f"{label} must be an array of strings")
    return tuple(_require_nonempty_str(item, label) for item in value)


def _require_int(
    value: object, label: str, *, minimum: int = 1, maximum: Optional[int] = None
) -> int:
    if type(value) is not int or value < minimum:
        raise OptimizerConfigError(f"{label} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise OptimizerConfigError(f"{label} must be <= {maximum}")
    return value


def _require_float(
    value: object, label: str, *, minimum: Optional[float] = None, allow_zero: bool = True
) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise OptimizerConfigError(f"{label} must be a finite number")
    result = float(value)
    if minimum is not None:
        if allow_zero and result < minimum:
            raise OptimizerConfigError(f"{label} must be >= {minimum}")
        if not allow_zero and result <= minimum:
            raise OptimizerConfigError(f"{label} must be > {minimum}")
    return result


def _validate_relative_pattern(raw_path: str, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip() or "\x00" in raw_path:
        raise PathSecurityError(f"{label} must be a non-empty relative path")
    if raw_path.startswith("~") or "\\" in raw_path:
        raise PathSecurityError(f"{label} cannot use '~' or backslashes: {raw_path!r}")
    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise PathSecurityError(f"{label} must be relative, not absolute: {raw_path!r}")
    if ".." in candidate.parts:
        raise PathSecurityError(f"{label} rejects path traversal ('..'): {raw_path!r}")
    return candidate


def _ensure_within_base(path: Path, base_dir: Path, label: str) -> Path:
    resolved_base = base_dir.resolve()
    if path.is_symlink():
        raise PathSecurityError(f"{label} refuses symbolic link: {path}")
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_base)
    except ValueError as error:
        raise PathSecurityError(
            f"{label} escapes base directory {resolved_base}: {resolved}"
        ) from error
    return resolved


def _is_heldout_relative_path(relative: Path) -> bool:
    for part in relative.parts:
        lower = part.lower()
        stem = Path(lower).stem
        if lower in _HELD_OUT_STEMS or stem in _HELD_OUT_STEMS:
            return True
        if any(
            stem.startswith(f"{token}_") or stem.startswith(f"{token}-")
            for token in _HELD_OUT_STEMS
        ):
            return True
    return False


@dataclass(frozen=True, slots=True)
class StyleExampleDigest:
    path: str
    sha256: str
    size_bytes: int
    content: str


@dataclass(frozen=True, slots=True)
class ProposerGuidelinesConfig:
    format: str = "structured_markdown"
    max_chars: int = 4000
    required_placeholders: tuple[str, ...] = ()
    rules: tuple[str, ...] = ()
    style_examples: tuple[StyleExampleDigest, ...] = ()
    manifest_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.manifest_sha256:
            object.__setattr__(
                self, "manifest_sha256", _sha256_json(self.to_manifest_body())
            )

    def to_manifest_body(self) -> dict[str, object]:
        return {
            "schema": _GUIDELINES_MANIFEST_SCHEMA,
            "format": self.format,
            "max_chars": self.max_chars,
            "required_placeholders": list(self.required_placeholders),
            "rules": list(self.rules),
            "style_examples": [
                {
                    "path": item.path,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                }
                for item in self.style_examples
            ],
        }

    def to_manifest(self) -> dict[str, object]:
        return {
            **self.to_manifest_body(),
            "manifest_sha256": self.manifest_sha256,
        }

    def to_gepa_guidance(self) -> dict[str, object]:
        guidelines = list(self.rules)
        if self.format:
            guidelines.append(f"Output format contract: {self.format} (max {self.max_chars} chars).")
        for placeholder in self.required_placeholders:
            guidelines.append(f"Preserve required placeholder `{placeholder}`.")
        return {
            "schema_version": "hillclimb.gepa.guidance.v1",
            "reflective_proposer": {
                "summary": (
                    "Improve the target lever instruction while strictly obeying the declared "
                    "guidelines, length limits, and required placeholders."
                ),
                "guidelines": guidelines
                or ["Keep instructions concise, deterministic, and grounded in feedback."],
            },
            "system_aware_merger": {
                "summary": "Merge complementary frontier instructions without violating guidelines.",
                "guidelines": guidelines
                or ["Combine complementary strengths concisely without duplication."],
            },
        }

    def validate_output(self, text: str) -> None:
        if not isinstance(text, str) or not text.strip():
            raise OptimizerConfigError("proposed instruction must be non-empty text")
        if len(text) > self.max_chars:
            raise OptimizerConfigError(
                f"proposed instruction exceeds max_chars ({len(text)} > {self.max_chars})"
            )
        for placeholder in self.required_placeholders:
            if placeholder not in text:
                raise OptimizerConfigError(
                    f"proposed instruction is missing required placeholder {placeholder!r}"
                )


@dataclass(frozen=True, slots=True)
class ContextFileEntry:
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ContextManifest:
    dest_dir: Path
    manifest_path: Path
    manifest_sha256: str
    total_bytes: int
    allow_search: bool
    files: tuple[ContextFileEntry, ...]


@dataclass(frozen=True, slots=True)
class ProposerContextConfig:
    files: tuple[str, ...] = ()
    max_total_bytes: int = 262_144
    allow_search: bool = True
    base_dir: Optional[Path] = None


def _parse_guidelines(
    raw: Mapping[str, object], base_dir: Path
) -> ProposerGuidelinesConfig:
    _reject_unknown_keys(
        raw,
        frozenset({"format", "max_chars", "required_placeholders", "rules", "style_examples"}),
        "proposer.guidelines",
    )
    fmt = _require_nonempty_str(
        raw.get("format", "structured_markdown"), "proposer.guidelines.format"
    )
    max_chars = _require_int(
        raw.get("max_chars", 4000), "proposer.guidelines.max_chars", minimum=1, maximum=1_000_000
    )
    placeholders = _require_str_tuple(
        raw.get("required_placeholders", ()),
        "proposer.guidelines.required_placeholders",
    )
    rules = _require_str_tuple(raw.get("rules", ()), "proposer.guidelines.rules")
    raw_examples = _require_str_tuple(
        raw.get("style_examples", ()), "proposer.guidelines.style_examples"
    )

    digests: list[StyleExampleDigest] = []
    seen_paths: set[str] = set()
    for rel_text in raw_examples:
        rel_path = _validate_relative_pattern(rel_text, "proposer.guidelines.style_examples")
        resolved = _ensure_within_base(base_dir / rel_path, base_dir, "style_example")
        if not resolved.is_file():
            raise PathSecurityError(f"missing style example file: {rel_text}")
        raw_bytes = resolved.read_bytes()
        try:
            content = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise OptimizerConfigError(
                f"style example {rel_text!r} must be valid UTF-8"
            ) from error
        posix_rel = rel_path.as_posix()
        if posix_rel in seen_paths:
            raise OptimizerConfigError(f"duplicate style example path: {posix_rel!r}")
        seen_paths.add(posix_rel)
        digests.append(
            StyleExampleDigest(
                path=posix_rel,
                sha256=_sha256_bytes(raw_bytes),
                size_bytes=len(raw_bytes),
                content=content,
            )
        )
    return ProposerGuidelinesConfig(
        format=fmt,
        max_chars=max_chars,
        required_placeholders=placeholders,
        rules=rules,
        style_examples=tuple(digests),
    )


def _parse_context(raw: Mapping[str, object], base_dir: Path) -> ProposerContextConfig:
    _reject_unknown_keys(
        raw, frozenset({"files", "max_total_bytes", "allow_search"}), "proposer.context"
    )
    files = _require_str_tuple(raw.get("files", ()), "proposer.context.files")
    for pattern in files:
        rel = _validate_relative_pattern(pattern, "proposer.context.files")
        if _is_heldout_relative_path(rel):
            raise PathSecurityError(
                f"proposer.context cannot reference held-out split path {pattern!r}"
            )
    max_total_bytes = _require_int(
        raw.get("max_total_bytes", 262_144),
        "proposer.context.max_total_bytes",
        minimum=1,
        maximum=64 * 1024 * 1024,
    )
    allow_search = raw.get("allow_search", True)
    if type(allow_search) is not bool:
        raise OptimizerConfigError("proposer.context.allow_search must be boolean")
    return ProposerContextConfig(
        files=files,
        max_total_bytes=max_total_bytes,
        allow_search=allow_search,
        base_dir=base_dir.resolve(),
    )


def _matches_forbidden(
    resolved_file: Path,
    relative_file: Path,
    resolved_base: Path,
    forbidden_paths: Sequence[Path | str],
) -> bool:
    for item in forbidden_paths:
        raw_str = str(item)
        if any(ch in raw_str for ch in ("*", "?", "[")):
            if relative_file.match(raw_str) or resolved_file.match(raw_str):
                return True
            continue
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = resolved_base / candidate
        resolved_forbidden = candidate.resolve()
        if resolved_file == resolved_forbidden:
            return True
        try:
            resolved_file.relative_to(resolved_forbidden)
            return True
        except ValueError:
            pass
    return False


def _set_readonly_permissions(dest_dir: Path, created_files: Sequence[Path]) -> None:
    for file_path in created_files:
        os.chmod(file_path, stat.S_IRUSR)
    subdirs: list[Path] = []
    for root, dirnames, _ in os.walk(dest_dir):
        for name in dirnames:
            subdirs.append(Path(root) / name)
    for subdir in sorted(subdirs, key=lambda p: len(p.parts), reverse=True):
        os.chmod(subdir, stat.S_IRUSR | stat.S_IXUSR)
    os.chmod(dest_dir, stat.S_IRUSR | stat.S_IXUSR)


def materialize_problem_context(
    cfg: OptimizerConfig | ProposerContextConfig,
    dest_dir: Path | str,
    *,
    forbidden_paths: Optional[Sequence[Path | str]] = None,
    base_dir: Optional[Path] = None,
) -> ContextManifest:
    """Materialize declared ``[proposer.context]`` files read-only (0500/0400) with manifest.json."""
    context_cfg = cfg.proposer.context if isinstance(cfg, OptimizerConfig) else cfg
    root_base = (base_dir or context_cfg.base_dir or Path.cwd()).resolve()
    dest = Path(dest_dir).resolve()
    forbidden = tuple(forbidden_paths or ())

    matched_files: dict[str, tuple[Path, bytes]] = {}
    for pattern in context_cfg.files:
        rel_pattern = _validate_relative_pattern(pattern, "proposer.context.files")
        if _is_heldout_relative_path(rel_pattern):
            raise PathSecurityError(
                f"proposer.context refuses held-out split pattern {pattern!r}"
            )
        if any(ch in pattern for ch in ("*", "?", "[")):
            hits = sorted(root_base.glob(pattern))
        else:
            hits = [root_base / rel_pattern]
        regular_hits = [path for path in hits if path.exists() or path.is_symlink()]
        if not regular_hits:
            raise PathSecurityError(
                f"proposer.context pattern matched no files: {pattern!r}"
            )
        for hit in regular_hits:
            resolved = _ensure_within_base(hit, root_base, "proposer.context file")
            if not resolved.is_file():
                raise PathSecurityError(
                    f"proposer.context entry is not a regular file: {hit}"
                )
            rel = resolved.relative_to(root_base)
            if _is_heldout_relative_path(rel) or _matches_forbidden(
                resolved, rel, root_base, forbidden
            ):
                raise PathSecurityError(
                    f"proposer.context refuses held-out or forbidden path: {rel.as_posix()!r}"
                )
            key = rel.as_posix()
            if key not in matched_files:
                matched_files[key] = (resolved, resolved.read_bytes())

    total_bytes = sum(len(data) for _, data in matched_files.values())
    if total_bytes > context_cfg.max_total_bytes:
        raise PathSecurityError(
            f"proposer.context total bytes ({total_bytes}) exceeds max_total_bytes "
            f"({context_cfg.max_total_bytes})"
        )

    if dest.exists():
        if dest.is_symlink() or not dest.is_dir():
            raise PathSecurityError(f"context destination must be a directory: {dest}")
        if any(dest.iterdir()):
            raise PathSecurityError(
                f"context destination directory must be empty: {dest}"
            )
    else:
        dest.mkdir(parents=True, mode=0o700)

    created_files: list[Path] = []
    entries: list[ContextFileEntry] = []
    for rel_posix in sorted(matched_files):
        _, payload = matched_files[rel_posix]
        target_path = dest / rel_posix
        target_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        target_path.write_bytes(payload)
        created_files.append(target_path)
        entries.append(
            ContextFileEntry(
                relative_path=rel_posix,
                sha256=_sha256_bytes(payload),
                size_bytes=len(payload),
            )
        )

    manifest_body = {
        "schema": _CONTEXT_MANIFEST_SCHEMA,
        "allow_search": context_cfg.allow_search,
        "max_total_bytes": context_cfg.max_total_bytes,
        "total_bytes": total_bytes,
        "files": [
            {
                "path": entry.relative_path,
                "sha256": entry.sha256,
                "size_bytes": entry.size_bytes,
            }
            for entry in entries
        ],
    }
    manifest_sha256 = _sha256_json(manifest_body)
    manifest_doc = {**manifest_body, "manifest_sha256": manifest_sha256}
    manifest_path = dest / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    created_files.append(manifest_path)

    _set_readonly_permissions(dest, created_files)

    return ContextManifest(
        dest_dir=dest,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        total_bytes=total_bytes,
        allow_search=context_cfg.allow_search,
        files=tuple(entries),
    )


@dataclass(frozen=True, slots=True)
class VerifierScoringConfig:
    kind: VerifierKind = VerifierKind.PYTHON_CALLBACK
    weight: float = 1.0
    url: Optional[str] = None


@dataclass(frozen=True, slots=True)
class RubricCriterion:
    id: str
    description: str
    points: float = 1.0
    kind: CriterionKind = CriterionKind.POSITIVE
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class RubricScoringConfig:
    rubric_file: Optional[str] = None
    criteria: tuple[RubricCriterion, ...] = ()
    rubric_sha256: Optional[str] = None
    judge_model: str = "gpt-6-luna"
    judge_concurrency: int = 16
    max_parse_fail_rate: float = 0.01
    weight: float = 0.5


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    mode: ScoringMode = ScoringMode.VERIFIER
    verifier: VerifierScoringConfig = field(default_factory=VerifierScoringConfig)
    rubric: RubricScoringConfig = field(default_factory=RubricScoringConfig)


@dataclass(frozen=True, slots=True)
class CriterionResult:
    criterion_id: str
    met: bool
    points: float
    explanation: str
    status: str
    cost_micros: Optional[int]
    total_tokens: Optional[int]


@dataclass(frozen=True, slots=True)
class HybridScoreResult:
    score: float
    verifier_score: Optional[float]
    rubric_score: Optional[float]
    guardrail_failed: bool
    criteria_results: tuple[CriterionResult, ...]
    parse_failures: int
    parse_fail_rate: float
    cost_micros: int
    total_tokens: int
    feedback: str


def _parse_criterion_entry(raw: object, fallback_id: Optional[str] = None) -> RubricCriterion:
    table = _require_mapping(raw, "scoring.rubric.criteria")
    _reject_unknown_keys(
        table,
        frozenset({"id", "description", "points", "kind", "weight"}),
        "scoring.rubric.criteria",
    )
    cid = _require_nonempty_str(table.get("id", fallback_id), "criterion.id")
    description = _require_nonempty_str(table.get("description"), f"criterion[{cid}].description")
    points = _require_float(table.get("points", 1.0), f"criterion[{cid}].points")
    kind = CriterionKind(_require_nonempty_str(table.get("kind", "positive"), f"criterion[{cid}].kind"))
    weight = _require_float(
        table.get("weight", 1.0), f"criterion[{cid}].weight", minimum=0.0, allow_zero=False
    )
    return RubricCriterion(id=cid, description=description, points=points, kind=kind, weight=weight)


def _load_rubric_criteria(path: Path) -> tuple[tuple[RubricCriterion, ...], str]:
    raw_bytes = path.read_bytes()
    digest = _sha256_bytes(raw_bytes)
    if path.suffix == ".toml":
        parsed = tomllib.loads(raw_bytes.decode("utf-8"))
    elif path.suffix == ".json":
        parsed = json.loads(raw_bytes.decode("utf-8"))
    else:
        raise OptimizerConfigError(f"unsupported rubric file format: {path.suffix!r}")
    doc = _require_mapping(parsed, "rubric_file")
    items = doc.get("criteria")
    criteria: list[RubricCriterion] = []
    if isinstance(items, Sequence) and not isinstance(items, (str, bytes)):
        for item in items:
            criteria.append(_parse_criterion_entry(item))
    elif isinstance(items, Mapping):
        for key, item in items.items():
            criteria.append(_parse_criterion_entry(item, fallback_id=str(key)))
    else:
        raise OptimizerConfigError("rubric_file must define a `criteria` array or table")
    if not criteria or len({c.id for c in criteria}) != len(criteria):
        raise OptimizerConfigError("rubric_file criteria must be non-empty with unique ids")
    return tuple(criteria), digest


def _parse_scoring(raw: Mapping[str, object], base_dir: Path) -> ScoringConfig:
    _reject_unknown_keys(raw, frozenset({"mode", "verifier", "rubric"}), "scoring")
    mode = ScoringMode(_require_nonempty_str(raw.get("mode", "verifier"), "scoring.mode"))

    raw_verifier = _require_mapping(raw.get("verifier", {}), "scoring.verifier")
    _reject_unknown_keys(raw_verifier, frozenset({"kind", "weight", "url"}), "scoring.verifier")
    v_kind = VerifierKind(
        _require_nonempty_str(raw_verifier.get("kind", "python_callback"), "scoring.verifier.kind")
    )
    default_v_weight = 0.5 if mode is ScoringMode.HYBRID else 1.0
    v_weight = _require_float(
        raw_verifier.get("weight", default_v_weight),
        "scoring.verifier.weight",
        minimum=0.0,
    )
    v_url = raw_verifier.get("url")
    if v_url is not None:
        v_url = _require_nonempty_str(v_url, "scoring.verifier.url")
    verifier = VerifierScoringConfig(kind=v_kind, weight=v_weight, url=v_url)

    raw_rubric = _require_mapping(raw.get("rubric", {}), "scoring.rubric")
    _reject_unknown_keys(
        raw_rubric,
        frozenset(
            {
                "rubric_file",
                "criteria",
                "judge_model",
                "judge_concurrency",
                "max_parse_fail_rate",
                "weight",
            }
        ),
        "scoring.rubric",
    )
    rubric_file = raw_rubric.get("rubric_file")
    criteria: tuple[RubricCriterion, ...] = ()
    rubric_sha256: Optional[str] = None
    if rubric_file is not None:
        rel = _validate_relative_pattern(str(rubric_file), "scoring.rubric.rubric_file")
        resolved = _ensure_within_base(base_dir / rel, base_dir, "rubric_file")
        if not resolved.is_file():
            raise PathSecurityError(f"missing rubric file: {rubric_file}")
        rubric_file = rel.as_posix()
        criteria, rubric_sha256 = _load_rubric_criteria(resolved)
    if "criteria" in raw_rubric:
        if criteria:
            raise OptimizerConfigError(
                "specify either scoring.rubric.rubric_file or inline criteria, not both"
            )
        raw_criteria = raw_rubric["criteria"]
        if not isinstance(raw_criteria, Sequence) or isinstance(raw_criteria, (str, bytes)):
            raise OptimizerConfigError("scoring.rubric.criteria must be a list")
        criteria = tuple(_parse_criterion_entry(item) for item in raw_criteria)
        if not criteria or len({c.id for c in criteria}) != len(criteria):
            raise OptimizerConfigError("inline rubric criteria must be non-empty with unique ids")
        rubric_sha256 = _sha256_json([c.id for c in criteria])

    judge_model = _require_nonempty_str(
        raw_rubric.get("judge_model", "gpt-6-luna"), "scoring.rubric.judge_model"
    )
    judge_concurrency = _require_int(
        raw_rubric.get("judge_concurrency", 16),
        "scoring.rubric.judge_concurrency",
        minimum=1,
        maximum=128,
    )
    max_parse_fail_rate = _require_float(
        raw_rubric.get("max_parse_fail_rate", 0.01),
        "scoring.rubric.max_parse_fail_rate",
        minimum=0.0,
    )
    if max_parse_fail_rate > 1.0:
        raise OptimizerConfigError("scoring.rubric.max_parse_fail_rate must be <= 1.0")
    r_weight = _require_float(
        raw_rubric.get("weight", 0.5),
        "scoring.rubric.weight",
        minimum=0.0,
    )
    rubric = RubricScoringConfig(
        rubric_file=rubric_file,
        criteria=criteria,
        rubric_sha256=rubric_sha256,
        judge_model=judge_model,
        judge_concurrency=judge_concurrency,
        max_parse_fail_rate=max_parse_fail_rate,
        weight=r_weight,
    )

    if mode in (ScoringMode.RUBRIC, ScoringMode.HYBRID) and not rubric.criteria:
        raise OptimizerConfigError(
            f"scoring.mode={mode.value!r} requires rubric criteria or rubric_file"
        )
    if mode is ScoringMode.HYBRID and (verifier.weight + rubric.weight) <= 0.0:
        raise OptimizerConfigError("hybrid scoring weights must sum to a positive number")

    return ScoringConfig(mode=mode, verifier=verifier, rubric=rubric)


def _default_rubric_decode(raw: str) -> Optional[VerificationDecision]:
    cleaned = raw.strip()
    if cleaned.startswith("{"):
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, Mapping) and type(payload.get("met")) is bool:
            return VerificationDecision(
                met=payload["met"],
                explanation=str(payload.get("explanation", "")),
            )
        return None
    lower = cleaned.lower()
    if lower in {"yes", "true", "pass", "met"}:
        return VerificationDecision(True, cleaned)
    if lower in {"no", "false", "fail", "unmet"}:
        return VerificationDecision(False, cleaned)
    return None


def score_hybrid(
    scoring: ScoringConfig,
    *,
    verifier_score: Optional[float] = None,
    verifier_feedback: str = "",
    response_text: str = "",
    task_context: str = "",
    judge: Optional[Callable[[str, str], tuple[str, Mapping[str, object]]]] = None,
    decode: Optional[Callable[[str], Optional[VerificationDecision]]] = None,
    queue: Optional[VerificationQueue] = None,
    protocol: str = "hillclimb.rubric.v1",
) -> HybridScoreResult:
    """Evaluate verifier, rubric (via VerificationQueue), or hybrid weighted score."""
    if scoring.mode in (ScoringMode.VERIFIER, ScoringMode.HYBRID):
        if verifier_score is None or not math.isfinite(verifier_score):
            raise OptimizerConfigError(
                f"scoring.mode={scoring.mode.value!r} requires a finite verifier_score"
            )
    if scoring.mode is ScoringMode.VERIFIER:
        score = float(verifier_score)  # type: ignore[arg-type]
        return HybridScoreResult(
            score=score,
            verifier_score=score,
            rubric_score=None,
            guardrail_failed=False,
            criteria_results=(),
            parse_failures=0,
            parse_fail_rate=0.0,
            cost_micros=0,
            total_tokens=0,
            feedback=verifier_feedback,
        )

    if judge is None:
        raise OptimizerConfigError("rubric or hybrid scoring requires a judge callable")
    criteria = scoring.rubric.criteria
    if not criteria:
        raise OptimizerConfigError("rubric scoring requires at least one criterion")

    decoder = decode or _default_rubric_decode
    requests = [
        VerificationRequest(
            criterion_id=criterion.id,
            system_prompt=(
                f"Evaluate criterion `{criterion.id}` ({criterion.kind.value}, "
                f"points={criterion.points}) with model {scoring.rubric.judge_model}: "
                f"{criterion.description}\nReturn JSON {{\"met\": true|false, \"explanation\": \"...\"}}."
            ),
            user_prompt=(
                f"Task context:\n{task_context}\n\nCandidate response:\n{response_text}"
                if task_context
                else response_text
            ),
        )
        for criterion in criteria
    ]

    owns_queue = queue is None
    active_queue = queue or VerificationQueue(
        max_workers=scoring.rubric.judge_concurrency,
        max_attempts=2,
    )
    try:
        receipts = active_queue.verify_rollout(
            requests,
            protocol=protocol,
            verifier=judge,
            decode=decoder,
        )
    finally:
        if owns_queue:
            active_queue.close()

    by_id = {c.id: c for c in criteria}
    results: list[CriterionResult] = []
    parse_failures = 0
    total_cost = 0
    total_tokens = 0
    earned_points = 0.0
    max_positive_points = 0.0
    guardrail_failed = False
    feedback_lines: list[str] = []
    if verifier_feedback:
        feedback_lines.append(f"verifier: {verifier_feedback}")

    for receipt in receipts:
        criterion = by_id[receipt.criterion_id]
        payload = receipt.to_payload()
        attempt_cost = payload.get("cost_micros")
        attempt_tokens = payload.get("total_tokens")
        if isinstance(attempt_cost, int):
            total_cost += attempt_cost
        if isinstance(attempt_tokens, int):
            total_tokens += attempt_tokens

        if receipt.decision is None:
            parse_failures += 1
            results.append(
                CriterionResult(
                    criterion_id=criterion.id,
                    met=False,
                    points=0.0,
                    explanation="judge output failed schema parse",
                    status=receipt.status.value,
                    cost_micros=attempt_cost if isinstance(attempt_cost, int) else None,
                    total_tokens=attempt_tokens if isinstance(attempt_tokens, int) else None,
                )
            )
            feedback_lines.append(f"[{criterion.id}:PARSE_ERROR] invalid judge response")
            continue

        met = receipt.decision.met
        delta_points = 0.0
        if criterion.kind is CriterionKind.POSITIVE:
            max_positive_points += abs(criterion.points) * criterion.weight
            if met:
                delta_points = criterion.points * criterion.weight
                earned_points += delta_points
        elif criterion.kind is CriterionKind.NEGATIVE:
            if met:
                delta_points = -abs(criterion.points) * criterion.weight
                earned_points += delta_points
        else:  # CriterionKind.GUARDRAIL
            if not met:
                guardrail_failed = True
        results.append(
            CriterionResult(
                criterion_id=criterion.id,
                met=met,
                points=delta_points,
                explanation=receipt.decision.explanation,
                status=receipt.status.value,
                cost_micros=attempt_cost if isinstance(attempt_cost, int) else None,
                total_tokens=attempt_tokens if isinstance(attempt_tokens, int) else None,
            )
        )
        tag = "PASS" if met else "FAIL"
        feedback_lines.append(
            f"[{criterion.id}:{tag}] {receipt.decision.explanation}".strip()
        )

    parse_fail_rate = parse_failures / len(criteria)
    if parse_fail_rate > scoring.rubric.max_parse_fail_rate:
        raise RubricParseError(
            f"rubric parse fail rate {parse_fail_rate:.4f} exceeds "
            f"max_parse_fail_rate={scoring.rubric.max_parse_fail_rate:.4f}"
        )

    if guardrail_failed:
        rubric_score = 0.0
    elif max_positive_points > 0.0:
        rubric_score = max(0.0, min(1.0, earned_points / max_positive_points))
    else:
        rubric_score = 1.0 if earned_points >= 0.0 else 0.0

    if scoring.mode is ScoringMode.RUBRIC:
        final_score = rubric_score
    else:
        v_score = float(verifier_score)  # type: ignore[arg-type]
        if guardrail_failed:
            final_score = 0.0
        else:
            denom = scoring.verifier.weight + scoring.rubric.weight
            final_score = (
                scoring.verifier.weight * v_score + scoring.rubric.weight * rubric_score
            ) / denom

    return HybridScoreResult(
        score=final_score,
        verifier_score=float(verifier_score) if verifier_score is not None else None,
        rubric_score=rubric_score,
        guardrail_failed=guardrail_failed,
        criteria_results=tuple(results),
        parse_failures=parse_failures,
        parse_fail_rate=parse_fail_rate,
        cost_micros=total_cost,
        total_tokens=total_tokens,
        feedback=" | ".join(feedback_lines),
    )


@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    name: str
    direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE
    weight: float = 1.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass(frozen=True, slots=True)
class ObjectiveCandidate:
    candidate_id: str
    metrics: Mapping[str, float]
    is_seed: bool = False
    payload: object = None


@dataclass(frozen=True, slots=True)
class ObjectivesConfig:
    aggregation: AggregationMode = AggregationMode.WEIGHTED_SUM
    primary: str = "reward"
    objectives: tuple[ObjectiveSpec, ...] = field(
        default_factory=lambda: (ObjectiveSpec(name="reward"),)
    )
    callable_ref: Optional[str] = None
    callable_fn: Optional[Callable[[Mapping[str, float]], float]] = None


@dataclass(frozen=True, slots=True)
class ChampionSelection:
    candidate_id: str
    metrics: Mapping[str, float]
    score: float
    fell_back_to_seed: bool
    reason: str
    frontier: tuple[str, ...]


def _parse_objective_spec(raw: object) -> ObjectiveSpec:
    table = _require_mapping(raw, "objectives.metrics")
    _reject_unknown_keys(
        table,
        frozenset({"name", "direction", "weight", "min", "max", "min_value", "max_value"}),
        "objectives.metrics",
    )
    name = _require_nonempty_str(table.get("name"), "objective.name")
    direction = ObjectiveDirection(
        _require_nonempty_str(table.get("direction", "maximize"), f"objective[{name}].direction")
    )
    weight = _require_float(table.get("weight", 1.0), f"objective[{name}].weight", minimum=0.0)
    raw_min = table.get("min", table.get("min_value"))
    raw_max = table.get("max", table.get("max_value"))
    min_value = _require_float(raw_min, f"objective[{name}].min") if raw_min is not None else None
    max_value = _require_float(raw_max, f"objective[{name}].max") if raw_max is not None else None
    if min_value is not None and max_value is not None and min_value > max_value:
        raise OptimizerConfigError(f"objective[{name}] min > max")
    return ObjectiveSpec(
        name=name,
        direction=direction,
        weight=weight,
        min_value=min_value,
        max_value=max_value,
    )


def _load_python_callable(ref: str) -> Callable[[Mapping[str, float]], float]:
    module_name, sep, attr_name = ref.partition(":")
    if not sep or not module_name.strip() or not attr_name.strip():
        raise OptimizerConfigError(
            f"objectives.python_callable must use 'module.path:callable_name', got {ref!r}"
        )
    module = importlib.import_module(module_name)
    target = getattr(module, attr_name, None)
    if not callable(target):
        raise OptimizerConfigError(f"objectives callable {ref!r} is not callable")
    return target


def _parse_objectives(raw: Mapping[str, object]) -> ObjectivesConfig:
    _reject_unknown_keys(
        raw,
        frozenset(
            {
                "aggregation",
                "primary",
                "metrics",
                "items",
                "objectives",
                "directions",
                "weights",
                "constraints",
                "python_callable",
                "callable",
            }
        ),
        "objectives",
    )
    aggregation = AggregationMode(
        _require_nonempty_str(raw.get("aggregation", "weighted_sum"), "objectives.aggregation")
    )
    callable_ref_raw = raw.get("python_callable", raw.get("callable"))
    callable_ref = (
        _require_nonempty_str(callable_ref_raw, "objectives.python_callable")
        if callable_ref_raw is not None
        else None
    )

    specs: list[ObjectiveSpec] = []
    list_key = next(
        (key for key in ("metrics", "items", "objectives") if key in raw), None
    )
    if list_key is not None:
        raw_list = raw[list_key]
        if not isinstance(raw_list, Sequence) or isinstance(raw_list, (str, bytes)):
            raise OptimizerConfigError(f"objectives.{list_key} must be an array of tables")
        specs = [_parse_objective_spec(item) for item in raw_list]
    else:
        directions = _require_mapping(raw.get("directions", {}), "objectives.directions")
        weights = _require_mapping(raw.get("weights", {}), "objectives.weights")
        constraints = _require_mapping(raw.get("constraints", {}), "objectives.constraints")
        names: list[str] = []
        primary_hint = raw.get("primary")
        if isinstance(primary_hint, str) and primary_hint.strip():
            names.append(primary_hint)
        for source in (directions, weights, constraints):
            for key in source:
                if key not in names:
                    names.append(str(key))
        if not names:
            names.append("reward")
        for name in names:
            direction = ObjectiveDirection(
                _require_nonempty_str(directions.get(name, "maximize"), f"directions[{name}]")
            )
            weight = _require_float(weights.get(name, 1.0), f"weights[{name}]", minimum=0.0)
            bound_raw = constraints.get(name)
            min_value = None
            max_value = None
            if bound_raw is not None:
                bound_map = _require_mapping(bound_raw, f"objectives.constraints.{name}")
                _reject_unknown_keys(
                    bound_map, frozenset({"min", "max", "eq"}), f"objectives.constraints.{name}"
                )
                if "eq" in bound_map:
                    exact = _require_float(bound_map["eq"], f"constraints[{name}].eq")
                    min_value = exact
                    max_value = exact
                if "min" in bound_map:
                    min_value = _require_float(bound_map["min"], f"constraints[{name}].min")
                if "max" in bound_map:
                    max_value = _require_float(bound_map["max"], f"constraints[{name}].max")
            specs.append(
                ObjectiveSpec(
                    name=name,
                    direction=direction,
                    weight=weight,
                    min_value=min_value,
                    max_value=max_value,
                )
            )

    if not specs or len({spec.name for spec in specs}) != len(specs):
        raise OptimizerConfigError("objectives must be non-empty with unique metric names")

    primary = _require_nonempty_str(raw.get("primary", specs[0].name), "objectives.primary")
    if primary not in {spec.name for spec in specs}:
        raise OptimizerConfigError(
            f"objectives.primary={primary!r} is not listed in objectives"
        )
    if aggregation is AggregationMode.WEIGHTED_SUM and all(spec.weight == 0.0 for spec in specs):
        raise OptimizerConfigError("weighted_sum objectives require at least one positive weight")
    return ObjectivesConfig(
        aggregation=aggregation,
        primary=primary,
        objectives=tuple(specs),
        callable_ref=callable_ref,
    )


def _normalize_candidate(item: ObjectiveCandidate | Mapping[str, object]) -> ObjectiveCandidate:
    if isinstance(item, ObjectiveCandidate):
        return item
    if isinstance(item, Mapping):
        cid = _require_nonempty_str(
            item.get("candidate_id", item.get("policy_id", item.get("id"))),
            "candidate_id",
        )
        metrics = item.get("metrics")
        if not isinstance(metrics, Mapping):
            raise OptimizerConfigError(f"candidate {cid!r} requires a metrics mapping")
        return ObjectiveCandidate(
            candidate_id=cid,
            metrics={str(k): _require_float(v, f"{cid}.{k}") for k, v in metrics.items()},
            is_seed=bool(item.get("is_seed", False)),
            payload=item.get("payload"),
        )
    raise OptimizerConfigError("candidate must be ObjectiveCandidate or Mapping")


def _extract_metrics(
    candidate: ObjectiveCandidate, objectives: Sequence[ObjectiveSpec]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for spec in objectives:
        if spec.name not in candidate.metrics:
            raise OptimizerConfigError(
                f"candidate {candidate.candidate_id!r} missing objective {spec.name!r}"
            )
        out[spec.name] = _require_float(
            candidate.metrics[spec.name], f"{candidate.candidate_id}.{spec.name}"
        )
    return out


def _oriented_value(value: float, direction: ObjectiveDirection) -> float:
    return value if direction is ObjectiveDirection.MAXIMIZE else -value


def _dominates(
    left: Mapping[str, float],
    right: Mapping[str, float],
    objectives: Sequence[ObjectiveSpec],
) -> bool:
    strictly_better = False
    for spec in objectives:
        lv = _oriented_value(left[spec.name], spec.direction)
        rv = _oriented_value(right[spec.name], spec.direction)
        if lv < rv:
            return False
        if lv > rv:
            strictly_better = True
    return strictly_better


def pareto_frontier(
    candidates: Sequence[ObjectiveCandidate | Mapping[str, object]],
    objectives: ObjectivesConfig | Sequence[ObjectiveSpec],
) -> tuple[ObjectiveCandidate, ...]:
    """Return non-dominated candidates preserving deterministic Pareto dominance."""
    specs = objectives.objectives if isinstance(objectives, ObjectivesConfig) else tuple(objectives)
    if not specs:
        raise OptimizerConfigError("pareto_frontier requires at least one objective")
    normalized = [_normalize_candidate(item) for item in candidates]
    if not normalized:
        return ()
    vectors = [_extract_metrics(item, specs) for item in normalized]
    frontier: list[ObjectiveCandidate] = []
    for index, candidate in enumerate(normalized):
        vec = vectors[index]
        if not any(
            i != index and _dominates(vectors[i], vec, specs)
            for i in range(len(normalized))
        ):
            frontier.append(candidate)
    return tuple(frontier)


def _satisfies_constraints(
    metrics: Mapping[str, float], objectives: Sequence[ObjectiveSpec]
) -> bool:
    for spec in objectives:
        value = metrics[spec.name]
        if spec.min_value is not None and value < spec.min_value:
            return False
        if spec.max_value is not None and value > spec.max_value:
            return False
    return True


def select_champion(
    candidates: Sequence[ObjectiveCandidate | Mapping[str, object]],
    objectives_cfg: ObjectivesConfig,
    *,
    seed_id: Optional[str] = None,
    aggregator: Optional[Callable[[Mapping[str, float]], float]] = None,
) -> ChampionSelection:
    """Deterministically select the champion from candidate objective vectors."""
    normalized = [_normalize_candidate(item) for item in candidates]
    if not normalized:
        raise OptimizerConfigError("select_champion requires at least one candidate")

    specs = objectives_cfg.objectives
    vectors = {c.candidate_id: _extract_metrics(c, specs) for c in normalized}
    frontier_ids = tuple(c.candidate_id for c in pareto_frontier(normalized, objectives_cfg))

    feasible = [
        c for c in normalized if _satisfies_constraints(vectors[c.candidate_id], specs)
    ]
    if not feasible:
        seed_candidate = None
        if seed_id is not None:
            seed_candidate = next((c for c in normalized if c.candidate_id == seed_id), None)
        if seed_candidate is None:
            seed_candidate = next((c for c in normalized if c.is_seed), None)
        if seed_candidate is None:
            raise OptimizerConfigError(
                "all candidates violated hard constraints and no seed fallback was found"
            )
        seed_metrics = vectors[seed_candidate.candidate_id]
        primary_spec = next(s for s in specs if s.name == objectives_cfg.primary)
        return ChampionSelection(
            candidate_id=seed_candidate.candidate_id,
            metrics=seed_metrics,
            score=float(seed_metrics[primary_spec.name]),
            fell_back_to_seed=True,
            reason="all candidates violated hard constraints; fell back to seed",
            frontier=frontier_ids,
        )

    mode = objectives_cfg.aggregation
    if mode is AggregationMode.CONSTRAINED_ARGMAX:
        primary_spec = next(s for s in specs if s.name == objectives_cfg.primary)
        ordered = [primary_spec, *(s for s in specs if s.name != primary_spec.name)]

        def key_fn(c: ObjectiveCandidate) -> tuple[object, ...]:
            vec = vectors[c.candidate_id]
            return (
                tuple(_oriented_value(vec[s.name], s.direction) for s in ordered),
                c.candidate_id,
            )

        best = max(feasible, key=key_fn)
        score = float(vectors[best.candidate_id][primary_spec.name])
    elif mode is AggregationMode.LEXICOGRAPHIC:

        def lex_key(c: ObjectiveCandidate) -> tuple[object, ...]:
            vec = vectors[c.candidate_id]
            return (
                tuple(_oriented_value(vec[s.name], s.direction) for s in specs),
                c.candidate_id,
            )

        best = max(feasible, key=lex_key)
        score = float(vectors[best.candidate_id][objectives_cfg.primary])
    elif mode is AggregationMode.PYTHON_CALLABLE:
        fn = aggregator or objectives_cfg.callable_fn
        if fn is None and objectives_cfg.callable_ref is not None:
            fn = _load_python_callable(objectives_cfg.callable_ref)
        if fn is None:
            raise OptimizerConfigError(
                "aggregation='python_callable' requires an aggregator callable"
            )
        scored = {
            c.candidate_id: _require_float(
                fn(vectors[c.candidate_id]), f"aggregator({c.candidate_id})"
            )
            for c in feasible
        }
        best = max(feasible, key=lambda c: (scored[c.candidate_id], c.candidate_id))
        score = scored[best.candidate_id]
    else:  # AggregationMode.WEIGHTED_SUM
        bounds: dict[str, tuple[float, float]] = {}
        for spec in specs:
            vals = [vectors[c.candidate_id][spec.name] for c in feasible]
            bounds[spec.name] = (min(vals), max(vals))

        def weighted_score(c: ObjectiveCandidate) -> float:
            vec = vectors[c.candidate_id]
            total = 0.0
            for spec in specs:
                lo, hi = bounds[spec.name]
                if hi == lo:
                    norm = 1.0
                elif spec.direction is ObjectiveDirection.MAXIMIZE:
                    norm = (vec[spec.name] - lo) / (hi - lo)
                else:
                    norm = (hi - vec[spec.name]) / (hi - lo)
                total += spec.weight * norm
            return total

        scored = {c.candidate_id: weighted_score(c) for c in feasible}
        best = max(feasible, key=lambda c: (scored[c.candidate_id], c.candidate_id))
        score = scored[best.candidate_id]

    return ChampionSelection(
        candidate_id=best.candidate_id,
        metrics=vectors[best.candidate_id],
        score=score,
        fell_back_to_seed=False,
        reason=f"selected by {mode.value}",
        frontier=frontier_ids,
    )


@dataclass(frozen=True, slots=True)
class WriteMaskConfig:
    mutable_levers: tuple[str, ...] = ()
    locked_levers: tuple[str, ...] = ()
    restrict_to_mutable: bool = True
    allow_harness: bool = False
    allow_environment: bool = False
    allow_model_route: bool = False
    allow_weights: bool = False

    def validate_patch(self, patch: object) -> Mapping[str, str]:
        """Fail closed if a patch touches any locked, undeclared, or disallowed lever."""
        if not isinstance(patch, Mapping) or not patch:
            raise WriteMaskViolation("patch must be a non-empty mapping")

        kind = "prompt_patch"
        fields_obj: object = patch
        if "kind" in patch or "patch" in patch:
            if set(patch) - {"kind", "patch"}:
                raise WriteMaskViolation("intervention envelope may only contain `kind` and `patch`")
            raw_kind = patch.get("kind")
            if not isinstance(raw_kind, str) or not raw_kind.strip():
                raise WriteMaskViolation("intervention envelope requires a string `kind`")
            kind = raw_kind
            fields_obj = patch.get("patch")

        if kind in {"harness_patch", "source_edit"} and not self.allow_harness:
            raise WriteMaskViolation(f"write_mask forbids {kind}")
        if kind == "environment_patch" and not self.allow_environment:
            raise WriteMaskViolation("write_mask forbids environment patches")
        if kind == "model_route_patch" and not self.allow_model_route:
            raise WriteMaskViolation("write_mask forbids model_route patches")
        if kind in {"sft", "fbc", "opsd", "rlvr", "weights"} and not self.allow_weights:
            raise WriteMaskViolation(f"write_mask forbids {kind} updates")
        if kind not in {
            "prompt_patch",
            "harness_patch",
            "source_edit",
            "environment_patch",
            "model_route_patch",
            "sft",
            "fbc",
            "opsd",
            "rlvr",
            "weights",
        }:
            raise WriteMaskViolation(f"unknown intervention kind: {kind!r}")

        if not isinstance(fields_obj, Mapping) or not fields_obj:
            raise WriteMaskViolation("patch payload must be a non-empty mapping of lever -> value")

        locked = set(self.locked_levers)
        mutable = set(self.mutable_levers)
        validated: dict[str, str] = {}
        for key, value in fields_obj.items():
            if not isinstance(key, str) or not key.strip():
                raise WriteMaskViolation("patch lever key must be a non-empty string")
            if key in locked:
                raise WriteMaskViolation(f"lever {key!r} is locked by [policy.write_mask]")
            if self.restrict_to_mutable and key not in mutable:
                raise WriteMaskViolation(
                    f"lever {key!r} is not in mutable_levers={sorted(mutable)!r}"
                )
            if not self.restrict_to_mutable and not mutable and not locked:
                raise WriteMaskViolation("no mutable levers declared in [policy.write_mask]")
            if not isinstance(value, str):
                raise WriteMaskViolation(f"patch value for lever {key!r} must be a string")
            validated[key] = value
        return validated

    def to_rust_write_mask(self) -> dict[str, object]:
        return {
            "prompts": bool(self.mutable_levers) or not self.restrict_to_mutable,
            "prompt_fields": list(self.mutable_levers) if self.restrict_to_mutable else None,
            "harness": self.allow_harness,
            "harness_fields": None,
            "environment": self.allow_environment,
            "environment_fields": None,
            "model_route": self.allow_model_route,
            "weights": self.allow_weights,
        }


def validate_patch(
    mask_or_config: WriteMaskConfig | OptimizerConfig, patch: object
) -> Mapping[str, str]:
    mask = (
        mask_or_config.policy.write_mask
        if isinstance(mask_or_config, OptimizerConfig)
        else mask_or_config
    )
    return mask.validate_patch(patch)


def _parse_write_mask(raw: Mapping[str, object]) -> WriteMaskConfig:
    _reject_unknown_keys(
        raw,
        frozenset(
            {
                "mutable_levers",
                "locked_levers",
                "allow_harness",
                "allow_environment",
                "allow_model_route",
                "allow_weights",
            }
        ),
        "policy.write_mask",
    )
    has_mutable = "mutable_levers" in raw
    mutable = _require_str_tuple(raw.get("mutable_levers", ()), "policy.write_mask.mutable_levers")
    locked = _require_str_tuple(raw.get("locked_levers", ()), "policy.write_mask.locked_levers")
    overlap = set(mutable) & set(locked)
    if overlap:
        raise OptimizerConfigError(
            f"[policy.write_mask] mutable_levers and locked_levers overlap: {sorted(overlap)}"
        )
    flags: dict[str, bool] = {}
    for flag in ("allow_harness", "allow_environment", "allow_model_route", "allow_weights"):
        val = raw.get(flag, False)
        if type(val) is not bool:
            raise OptimizerConfigError(f"policy.write_mask.{flag} must be boolean")
        flags[flag] = val
    restrict_to_mutable = has_mutable or not bool(locked)
    return WriteMaskConfig(
        mutable_levers=mutable,
        locked_levers=locked,
        restrict_to_mutable=restrict_to_mutable,
        **flags,
    )


@dataclass(frozen=True, slots=True)
class LimitsConfig:
    max_rpm: int = 600
    max_tpm: int = 4_000_000
    rollout_parallelism: int = 8
    proposer_parallelism: int = 1
    request_timeout_seconds: float = 120.0
    max_retries_429: int = 3
    backoff_base_seconds: float = 0.5

    def to_queue_limits(self) -> dict[str, int]:
        proposer_slots = min(self.proposer_parallelism, 64)
        total_slots = min(1024, max(proposer_slots + 1, self.rollout_parallelism + proposer_slots))
        proposer_rpm = max(1, min(self.max_rpm - 1, max(10, self.max_rpm // 10)))
        total_rpm = min(100_000, max(proposer_rpm + 1, self.max_rpm))
        proposer_tpm = max(1, min(self.max_tpm - 1024, max(65_536, self.max_tpm // 4)))
        total_tpm = min(1_000_000_000, max(proposer_tpm + 32_768, self.max_tpm))
        return {
            "slots": total_slots,
            "requests_per_minute": total_rpm,
            "tokens_per_minute": total_tpm,
            "proposer_slots": proposer_slots,
            "proposer_requests_per_minute": proposer_rpm,
            "proposer_tokens_per_minute": proposer_tpm,
        }


def _parse_limits(raw: Mapping[str, object]) -> LimitsConfig:
    _reject_unknown_keys(
        raw,
        frozenset(
            {
                "max_rpm",
                "max_tpm",
                "rollout_parallelism",
                "proposer_parallelism",
                "request_timeout_seconds",
                "max_retries_429",
                "backoff_base_seconds",
            }
        ),
        "limits",
    )
    return LimitsConfig(
        max_rpm=_require_int(raw.get("max_rpm", 600), "limits.max_rpm", minimum=2, maximum=100_000),
        max_tpm=_require_int(
            raw.get("max_tpm", 4_000_000), "limits.max_tpm", minimum=2048, maximum=1_000_000_000
        ),
        rollout_parallelism=_require_int(
            raw.get("rollout_parallelism", 8), "limits.rollout_parallelism", minimum=1, maximum=960
        ),
        proposer_parallelism=_require_int(
            raw.get("proposer_parallelism", 1), "limits.proposer_parallelism", minimum=1, maximum=64
        ),
        request_timeout_seconds=_require_float(
            raw.get("request_timeout_seconds", 120.0),
            "limits.request_timeout_seconds",
            minimum=0.0,
            allow_zero=False,
        ),
        max_retries_429=_require_int(
            raw.get("max_retries_429", 3), "limits.max_retries_429", minimum=0, maximum=16
        ),
        backoff_base_seconds=_require_float(
            raw.get("backoff_base_seconds", 0.5),
            "limits.backoff_base_seconds",
            minimum=0.0,
            allow_zero=False,
        ),
    )


@dataclass(frozen=True, slots=True)
class StopConfig:
    cap_usd: float = 1.0
    max_metric_calls: Optional[int] = None
    max_rounds: Optional[int] = None
    max_wall_seconds: Optional[float] = None
    patience_rounds: Optional[int] = None
    target_score: Optional[float] = None

    def check_stop(
        self,
        *,
        spent_usd: float,
        metric_calls: int,
        rounds_completed: int,
        elapsed_seconds: float,
        non_improving_rounds: int,
        best_score: Optional[float],
    ) -> Optional[str]:
        if self.target_score is not None and best_score is not None and best_score >= self.target_score:
            return "target_score"
        if spent_usd >= self.cap_usd and self.cap_usd > 0.0:
            return "cap_usd"
        if self.max_metric_calls is not None and metric_calls >= self.max_metric_calls:
            return "max_metric_calls"
        if self.max_rounds is not None and rounds_completed >= self.max_rounds:
            return "max_rounds"
        if self.max_wall_seconds is not None and elapsed_seconds >= self.max_wall_seconds:
            return "max_wall_seconds"
        if self.patience_rounds is not None and non_improving_rounds >= self.patience_rounds:
            return "patience_rounds"
        return None


def _parse_stop(raw: Mapping[str, object]) -> StopConfig:
    _reject_unknown_keys(
        raw,
        frozenset(
            {
                "cap_usd",
                "max_metric_calls",
                "max_rounds",
                "max_wall_seconds",
                "patience_rounds",
                "target_score",
            }
        ),
        "stop",
    )
    cap_usd = _require_float(raw.get("cap_usd", 1.0), "stop.cap_usd", minimum=0.0)
    max_metric_calls = (
        _require_int(raw["max_metric_calls"], "stop.max_metric_calls", minimum=1)
        if "max_metric_calls" in raw
        else None
    )
    max_rounds = (
        _require_int(raw["max_rounds"], "stop.max_rounds", minimum=1, maximum=10_000)
        if "max_rounds" in raw
        else None
    )
    max_wall_seconds = (
        _require_float(
            raw["max_wall_seconds"], "stop.max_wall_seconds", minimum=0.0, allow_zero=False
        )
        if "max_wall_seconds" in raw
        else None
    )
    patience_rounds = (
        _require_int(raw["patience_rounds"], "stop.patience_rounds", minimum=1)
        if "patience_rounds" in raw
        else None
    )
    target_score = (
        _require_float(raw["target_score"], "stop.target_score")
        if "target_score" in raw
        else None
    )
    return StopConfig(
        cap_usd=cap_usd,
        max_metric_calls=max_metric_calls,
        max_rounds=max_rounds,
        max_wall_seconds=max_wall_seconds,
        patience_rounds=patience_rounds,
        target_score=target_score,
    )


@dataclass(frozen=True, slots=True)
class ProposerConfig:
    model: str = "gpt-6-luna"
    effort: str = "medium"
    guidelines: ProposerGuidelinesConfig = field(default_factory=ProposerGuidelinesConfig)
    context: ProposerContextConfig = field(default_factory=ProposerContextConfig)


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    model: str = "gpt-6-luna"
    write_mask: WriteMaskConfig = field(
        default_factory=lambda: WriteMaskConfig(mutable_levers=("stage2_system",))
    )


@dataclass(frozen=True, slots=True)
class TargetConfig:
    kind: str = "python_callback"
    base_url: Optional[str] = None
    namespace: str = "default"
    harness: str = "prompt_program"
    provider: str = "openai"
    timeout_seconds: int = 30


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    schema_version: str = _SCHEMA_VERSION
    algorithm: str = "gepa"
    seed: int = 0
    minibatch: int = 4
    proposer: ProposerConfig = field(default_factory=ProposerConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    objectives: ObjectivesConfig = field(default_factory=ObjectivesConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    stop: StopConfig = field(default_factory=StopConfig)
    source_path: Optional[Path] = None

    @classmethod
    def from_toml(
        cls, path: Path | str, *, base_dir: Optional[Path] = None
    ) -> OptimizerConfig:
        return load_optimizer_toml(path, base_dir=base_dir)

    @classmethod
    def from_mapping(
        cls,
        document: Mapping[str, object],
        *,
        base_dir: Optional[Path] = None,
        source_path: Optional[Path] = None,
    ) -> OptimizerConfig:
        root_dir = (base_dir or (source_path.parent if source_path else Path.cwd())).resolve()
        _reject_unknown_keys(
            document,
            frozenset(
                {
                    "schema_version",
                    "algorithm",
                    "seed",
                    "minibatch",
                    "proposer",
                    "scoring",
                    "objectives",
                    "policy",
                    "target",
                    "limits",
                    "stop",
                }
            ),
            "optimizer",
        )
        schema_version = _require_nonempty_str(
            document.get("schema_version", _SCHEMA_VERSION), "schema_version"
        )
        if schema_version != _SCHEMA_VERSION:
            raise OptimizerConfigError(
                f"unsupported optimizer.toml schema_version {schema_version!r}"
            )
        raw_algo = _require_nonempty_str(document.get("algorithm", "gepa"), "algorithm").lower()
        if raw_algo not in ALGORITHM_CHOICES:
            raise OptimizerConfigError(
                f"unsupported algorithm {raw_algo!r}; expected one of {ALGORITHM_CHOICES}"
            )
        algorithm = canonical_algorithm(raw_algo)
        seed = _require_int(document.get("seed", 0), "seed", minimum=0)
        minibatch = _require_int(document.get("minibatch", 4), "minibatch", minimum=1, maximum=1024)

        raw_proposer = _require_mapping(document.get("proposer", {}), "proposer")
        _reject_unknown_keys(
            raw_proposer, frozenset({"model", "effort", "guidelines", "context"}), "proposer"
        )
        proposer = ProposerConfig(
            model=_require_nonempty_str(raw_proposer.get("model", "gpt-6-luna"), "proposer.model"),
            effort=_require_nonempty_str(raw_proposer.get("effort", "medium"), "proposer.effort"),
            guidelines=_parse_guidelines(
                _require_mapping(raw_proposer.get("guidelines", {}), "proposer.guidelines"),
                root_dir,
            ),
            context=_parse_context(
                _require_mapping(raw_proposer.get("context", {}), "proposer.context"),
                root_dir,
            ),
        )

        scoring = _parse_scoring(
            _require_mapping(document.get("scoring", {}), "scoring"), root_dir
        )
        objectives = _parse_objectives(
            _require_mapping(document.get("objectives", {}), "objectives")
        )

        raw_policy = _require_mapping(document.get("policy", {}), "policy")
        _reject_unknown_keys(raw_policy, frozenset({"model", "write_mask"}), "policy")
        write_mask = (
            _parse_write_mask(_require_mapping(raw_policy["write_mask"], "policy.write_mask"))
            if "write_mask" in raw_policy
            else WriteMaskConfig(mutable_levers=("stage2_system",))
        )
        policy = PolicyConfig(
            model=_require_nonempty_str(raw_policy.get("model", "gpt-6-luna"), "policy.model"),
            write_mask=write_mask,
        )

        raw_target = _require_mapping(document.get("target", {}), "target")
        _reject_unknown_keys(
            raw_target,
            frozenset({"kind", "base_url", "namespace", "harness", "provider", "timeout_seconds"}),
            "target",
        )
        target = TargetConfig(
            kind=_require_nonempty_str(raw_target.get("kind", "python_callback"), "target.kind"),
            base_url=(
                _require_nonempty_str(raw_target["base_url"], "target.base_url")
                if "base_url" in raw_target
                else None
            ),
            namespace=_require_nonempty_str(raw_target.get("namespace", "default"), "target.namespace"),
            harness=_require_nonempty_str(raw_target.get("harness", "prompt_program"), "target.harness"),
            provider=_require_nonempty_str(raw_target.get("provider", "openai"), "target.provider"),
            timeout_seconds=_require_int(
                raw_target.get("timeout_seconds", 30), "target.timeout_seconds", minimum=1, maximum=3600
            ),
        )

        limits = _parse_limits(_require_mapping(document.get("limits", {}), "limits"))
        stop = _parse_stop(_require_mapping(document.get("stop", {}), "stop"))

        return cls(
            schema_version=schema_version,
            algorithm=algorithm,
            seed=seed,
            minibatch=minibatch,
            proposer=proposer,
            scoring=scoring,
            objectives=objectives,
            policy=policy,
            target=target,
            limits=limits,
            stop=stop,
            source_path=source_path.resolve() if source_path is not None else None,
        )


def parse_optimizer_toml(
    raw: str, *, base_dir: Optional[Path] = None
) -> OptimizerConfig:
    """Parse and validate an ``optimizer.toml`` string."""
    if not isinstance(raw, str):
        raise OptimizerConfigError("optimizer.toml content must be text")
    try:
        parsed = tomllib.loads(raw)
    except tomllib.TOMLDecodeError as error:
        raise OptimizerConfigError(f"invalid optimizer.toml syntax: {error}") from error
    return OptimizerConfig.from_mapping(parsed, base_dir=base_dir)


def load_optimizer_toml(
    path: Path | str, *, base_dir: Optional[Path] = None
) -> OptimizerConfig:
    """Load and validate an ``optimizer.toml`` file from disk."""
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise PathSecurityError(f"optimizer config file does not exist: {file_path}")
    try:
        raw = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise OptimizerConfigError(f"failed to read {file_path}: {error}") from error
    try:
        parsed = tomllib.loads(raw)
    except tomllib.TOMLDecodeError as error:
        raise OptimizerConfigError(f"invalid TOML in {file_path}: {error}") from error
    return OptimizerConfig.from_mapping(
        parsed,
        base_dir=base_dir or file_path.parent,
        source_path=file_path,
    )
