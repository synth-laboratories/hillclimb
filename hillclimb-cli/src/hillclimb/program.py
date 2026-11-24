"""Negotiation for Hillclimb's typed Optimize Anything program declaration.

This module verifies declarations before an optimizer accepts a lever or
interprets measured feedback. The target remains the authority for validating
and admitting a prepared revision.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from string import hexdigits

from .pipeline import MaterializedPipeline, StageProgram, VerifierPin

__all__ = [
    "AnnotationPin",
    "Authority",
    "Capability",
    "Direction",
    "Lever",
    "LeverOperation",
    "Measurement",
    "Objective",
    "Program",
    "Sensor",
    "Split",
    "StageProgram",
    "VerifierPin",
    "Visibility",
    "program_digest",
]

_PROGRAM_SCHEMA = "hillclimb.optimize-anything.program.v1"
_ANNOTATION_SCHEMA = "synth.live-annotation-protocol.v1"
_MAX_DECLARATIONS = 128
_MAX_STRING_BYTES = 4096
_MAX_LEVER_BYTES = 16 * 1024 * 1024
_MAX_PROGRAM_BYTES = 1024 * 1024
_LOWERCASE_HEX = frozenset(hexdigits.lower())


class LeverOperation(StrEnum):
    REPLACE_TEXT = "replace_text"
    REPLACE_SOURCE_FILE = "replace_source_file"
    APPLY_SOURCE_DIFF = "apply_source_diff"


class Capability(StrEnum):
    PREPARE_REVISION = "revision.prepare"
    LIVE_ANNOTATION = "annotation.live"
    PAIRED_HELD_OUT = "evaluation.paired_heldout"
    STAGED_SIGNATURES = "program.staged_signatures"
    QUEUED_VERIFIER = "evaluation.queued_verifier"


class Authority(StrEnum):
    OBSERVER = "observer"
    TARGET_EVALUATOR = "target_evaluator"


class Measurement(StrEnum):
    PROVISIONAL = "provisional"
    SETTLED = "settled"


class Visibility(StrEnum):
    TRAIN_ONLY = "train_only"
    SETTLEMENT_ONLY = "settlement_only"


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class Direction(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


def _document(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return value


def _text(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > _MAX_STRING_BYTES
    ):
        raise ValueError(f"{name} must be nonempty bounded text")
    return value


def _items(value: object, name: str) -> list[object]:
    if not isinstance(value, list) or not 0 < len(value) <= _MAX_DECLARATIONS:
        raise ValueError(f"{name} must be a nonempty bounded array")
    return value


def _names(value: object, name: str) -> tuple[str, ...]:
    items = tuple(_text(item, name) for item in _items(value, name))
    if len(set(items)) != len(items):
        raise ValueError(f"{name} contains duplicate values")
    return items


def _exact_fields(
    document: Mapping[str, object], fields: AbstractSet[str], name: str
) -> None:
    if document.keys() != fields:
        raise ValueError(f"{name} fields differ from the declared wire schema")


def _hex(value: object, name: str, *, prefix: str = "", length: int = 64) -> str:
    text = _text(value, name)
    if (
        not text.startswith(prefix)
        or len(text) != len(prefix) + length
        or not set(text[len(prefix) :]).issubset(_LOWERCASE_HEX)
    ):
        raise ValueError(f"{name} has an invalid digest or revision identity")
    return text


def _route(value: object, name: str) -> str:
    text = _text(value, name)
    segments = text.split("/")[1:]
    if not text.startswith("/") or any(
        not part
        or part in {".", ".."}
        or any(
            not character.isascii() or not (character.isalnum() or character in "-_.")
            for character in part
        )
        for part in segments
    ):
        raise ValueError(f"{name} must be a target-relative route")
    return text


def program_digest(document: Mapping[str, object]) -> str:
    """Hash canonical v1 program contents with the digest field excluded."""
    body = dict(document)
    body.pop("contract_digest", None)
    raw = json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(raw) > _MAX_PROGRAM_BYTES:
        raise ValueError("program declaration exceeds its byte limit")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class Lever:
    id: str
    operation: LeverOperation
    binding: str
    media_type: str
    max_bytes: int
    validate: str
    admit: str
    abi: str | None
    constraints: tuple[str, ...]

    @classmethod
    def from_wire(cls, raw: object) -> Lever:
        row = _document(raw, "lever")
        _exact_fields(
            row,
            {
                "id",
                "operation",
                "binding",
                "media_type",
                "max_bytes",
                "validate",
                "admit",
                "abi",
                "constraints",
            },
            "lever",
        )
        operation = LeverOperation(_text(row["operation"], "lever operation"))
        binding = _text(row["binding"], "lever binding")
        media_type = _text(row["media_type"], "lever media type")
        maximum = row["max_bytes"]
        if type(maximum) is not int or not 0 < maximum <= _MAX_LEVER_BYTES:
            raise ValueError("lever max_bytes must be a positive bounded integer")
        abi = row["abi"]
        if operation in {
            LeverOperation.REPLACE_SOURCE_FILE,
            LeverOperation.APPLY_SOURCE_DIFF,
        }:
            path = PurePosixPath(binding)
            if (
                path.is_absolute()
                or str(path) != binding
                or not path.parts
                or ".." in path.parts
                or "\\" in binding
            ):
                raise ValueError("source lever must bind one normalized relative file")
            abi = _text(abi, "source lever ABI")
        elif abi is not None or "/" in binding or "\\" in binding:
            raise ValueError("text lever must bind one named field without an ABI")
        return cls(
            _text(row["id"], "lever id"),
            operation,
            binding,
            media_type,
            maximum,
            _route(row["validate"], "lever validator"),
            _route(row["admit"], "lever admission"),
            abi,
            _names(row["constraints"], "lever constraints"),
        )

    def require_payload(self, source: str) -> None:
        if not isinstance(source, str):
            raise TypeError("lever input must be text")
        if not 0 < len(source.encode("utf-8")) <= self.max_bytes:
            raise ValueError(f"lever {self.id} payload exceeds its size bounds")

    def proposer_constraints(self) -> str:
        interface = (f"Preserve the ABI {self.abi}.",) if self.abi is not None else ()
        return " ".join((*interface, *self.constraints))


@dataclass(frozen=True, slots=True)
class Sensor:
    id: str
    unit: str
    visibility: Visibility
    authority: Authority
    measurement: Measurement
    splits: frozenset[Split]
    bounds: tuple[float, float] | None
    fields: tuple[str, ...]

    @classmethod
    def from_wire(cls, raw: object) -> Sensor:
        row = _document(raw, "sensor")
        _exact_fields(
            row,
            {
                "id",
                "unit",
                "visibility",
                "authority",
                "measurement",
                "splits",
                "bounds",
                "fields",
            },
            "sensor",
        )
        visibility = Visibility(_text(row["visibility"], "sensor visibility"))
        splits = frozenset(
            Split(split) for split in _names(row["splits"], "sensor splits")
        )
        if visibility is Visibility.TRAIN_ONLY and splits != {Split.TRAIN}:
            raise ValueError("training-only sensor must bind the training split")
        bounds = None
        if row["bounds"] is not None:
            interval = _items(row["bounds"], "sensor bounds")
            if (
                len(interval) != 2
                or any(
                    type(value) not in (int, float) or not math.isfinite(value)
                    for value in interval
                )
                or interval[0] >= interval[1]
            ):
                raise ValueError("sensor bounds must be a finite increasing interval")
            bounds = (float(interval[0]), float(interval[1]))
        return cls(
            _text(row["id"], "sensor id"),
            _text(row["unit"], "sensor unit"),
            visibility,
            Authority(_text(row["authority"], "sensor authority")),
            Measurement(_text(row["measurement"], "sensor measurement")),
            splits,
            bounds,
            _names(row["fields"], "sensor fields"),
        )

    def reading(self, value: object) -> float:
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError(f"sensor {self.id} requires a finite numeric reading")
        if self.bounds is not None and not self.bounds[0] <= value <= self.bounds[1]:
            raise ValueError(f"sensor {self.id} reading is outside declared bounds")
        return float(value)


@dataclass(frozen=True, slots=True)
class Objective:
    sensor: str
    direction: Direction
    aggregation: str
    replicate: str
    description: str

    @classmethod
    def from_wire(cls, raw: object) -> Objective:
        row = _document(raw, "objective")
        _exact_fields(
            row,
            {"sensor", "direction", "aggregation", "replicate", "description"},
            "objective",
        )
        return cls(
            _text(row["sensor"], "objective sensor"),
            Direction(_text(row["direction"], "objective direction")),
            _text(row["aggregation"], "objective aggregation"),
            _text(row["replicate"], "objective replicate"),
            _text(row["description"], "objective description"),
        )


@dataclass(frozen=True, slots=True)
class AnnotationPin:
    schema: str
    protocol_id: str
    revision_id: str
    code_sha256: str
    configuration_digest: str
    runner_digest: str
    source_revision: str | None

    @classmethod
    def from_wire(cls, raw: object) -> AnnotationPin:
        row = _document(raw, "annotation pin")
        schema = row.get("schema")
        if schema != _ANNOTATION_SCHEMA:
            raise ValueError("unsupported annotation protocol schema")
        source = row.get("source_revision")
        return cls(
            schema,
            _text(row.get("protocol_id"), "annotation protocol"),
            _hex(
                row.get("protocol_revision_id"),
                "annotation revision",
                prefix="anprev_",
                length=16,
            ),
            _hex(row.get("code_sha256"), "annotation code"),
            _hex(
                row.get("configuration_digest"),
                "annotation configuration",
                prefix="sha256:",
            ),
            _hex(row.get("runner_digest"), "annotation runner", prefix="sha256:"),
            None if source is None else _text(source, "annotation source revision"),
        )


@dataclass(frozen=True, slots=True)
class Program:
    task: str
    digest: str
    capabilities: frozenset[str]
    levers: tuple[Lever, ...]
    sensors: tuple[Sensor, ...]
    objective: Objective
    annotation: AnnotationPin | None
    pipeline: StageProgram | None = None
    verifier: VerifierPin | None = None

    @classmethod
    def from_wire(cls, raw: object, *, expected_digest: str | None = None) -> Program:
        row = _document(raw, "program")
        if row.get("schema") != _PROGRAM_SCHEMA:
            raise ValueError("unsupported Hillclimb program schema")
        digest = _hex(row.get("contract_digest"), "program contract", prefix="sha256:")
        if digest != program_digest(row) or (
            expected_digest is not None and digest != expected_digest
        ):
            raise ValueError("Hillclimb program digest does not match the contract")
        capabilities = frozenset(
            _names(row.get("capabilities"), "program capabilities")
        )
        levers = tuple(
            Lever.from_wire(item)
            for item in _items(row.get("levers"), "program levers")
        )
        if len({lever.id for lever in levers}) != len(levers):
            raise ValueError("program contains duplicate lever IDs")
        if len({lever.binding for lever in levers}) != len(levers):
            raise ValueError("program contains conflicting lever bindings")
        sensors = tuple(
            Sensor.from_wire(item)
            for item in _items(row.get("sensors"), "program sensors")
        )
        if len({sensor.id for sensor in sensors}) != len(sensors):
            raise ValueError("program contains duplicate sensor IDs")
        objective = Objective.from_wire(row.get("objective"))
        trusted = next(
            (sensor for sensor in sensors if sensor.id == objective.sensor), None
        )
        if (
            trusted is None
            or trusted.authority is not Authority.TARGET_EVALUATOR
            or trusted.measurement is not Measurement.SETTLED
        ):
            raise ValueError(
                "objective requires a declared settled target evaluator sensor"
            )
        annotations = row.get("annotations")
        annotation = (
            None if annotations is None else AnnotationPin.from_wire(annotations)
        )
        if Capability.LIVE_ANNOTATION in capabilities and annotation is None:
            raise ValueError("live annotation capability requires a pinned protocol")
        pipeline_document = row.get("pipeline")
        pipeline = (
            None
            if pipeline_document is None
            else StageProgram.from_wire(pipeline_document)
        )
        if Capability.STAGED_SIGNATURES in capabilities and pipeline is None:
            raise ValueError("staged signatures require a pipeline declaration")
        verifier_document = row.get("verifier")
        verifier = None
        if verifier_document is not None:
            verifier = VerifierPin.from_wire(verifier_document)
            if verifier.stage is not None or pipeline is not None:
                raise ValueError("top-level verifier requires a flat actor output")
        if pipeline is not None:
            signature_levers = {
                lever.binding
                for lever in levers
                if lever.operation is LeverOperation.REPLACE_TEXT
                and lever.media_type == "application/json"
            }
            if not pipeline.signature_bindings <= signature_levers:
                raise ValueError("pipeline signatures require named JSON text levers")
            verifier = pipeline.verifier
        if Capability.QUEUED_VERIFIER in capabilities and verifier is None:
            raise ValueError("queued verifier capability requires a verifier pin")
        if verifier is not None and verifier.output_sensor != objective.sensor:
            raise ValueError(
                "queued verifier must produce the trusted objective sensor"
            )
        return cls(
            _text(row.get("task"), "program task"),
            digest,
            capabilities,
            levers,
            sensors,
            objective,
            annotation,
            pipeline,
            verifier,
        )

    def materialize_pipeline(
        self,
        parent: Mapping[str, str],
        changes: Mapping[str, str],
        *,
        joint_id: str | None = None,
    ) -> MaterializedPipeline:
        if self.pipeline is None:
            raise ValueError("target does not declare a signature pipeline")
        for lever in self.levers:
            if lever.binding in self.pipeline.signature_bindings:
                if lever.binding in parent:
                    lever.require_payload(parent[lever.binding])
                if lever.binding in changes:
                    lever.require_payload(changes[lever.binding])
        return self.pipeline.materialize(parent, changes, joint_id=joint_id)

    def require_capabilities(self, required: AbstractSet[Capability]) -> None:
        missing = sorted(required.difference(self.capabilities))
        if missing:
            raise ValueError(
                f"target does not advertise required capabilities {', '.join(missing)}"
            )

    def require_lever(self, identity: str, *, operation: LeverOperation) -> Lever:
        for lever in self.levers:
            if lever.id == identity and lever.operation is operation:
                return lever
        raise ValueError(
            f"target does not advertise lever {identity} with operation {operation}"
        )

    def sensor(self, identity: str) -> Sensor:
        for sensor in self.sensors:
            if sensor.id == identity:
                return sensor
        raise ValueError(f"target does not advertise sensor {identity}")
