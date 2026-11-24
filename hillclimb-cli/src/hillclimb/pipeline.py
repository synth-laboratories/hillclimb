"""Typed signature pipelines, joint materialization, and verifier bindings."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

__all__ = [
    "FieldSpec",
    "MaterializedPipeline",
    "SignatureSpec",
    "StageProgram",
    "VerificationIntent",
    "VerifierPin",
]

_MAX_STAGES = 32
_MAX_FIELDS = 32
_MAX_OUTPUT_FIELDS = 10
_MAX_SIGNATURE_BYTES = 64 * 1024
_IDENTIFIER = re.compile(r"[a-zA-Z][a-zA-Z0-9_.-]{0,79}")
_FIELD = re.compile(r"[a-z][a-z0-9_]{0,47}")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_SCALARS = frozenset({"string", "list[string]", "boolean", "number"})


def _object(value: object, keys: set[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or value.keys() != keys:
        raise ValueError(f"{label} fields differ from the declared wire schema")
    return value


def _list(value: object, maximum: int, label: str) -> list[object]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(f"{label} must be a bounded array")
    return value


def _text(value: object, maximum: int, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise ValueError(f"{label} must be nonempty bounded text")
    return value


def _id(value: object, label: str, *, field: bool = False) -> str:
    if not isinstance(value, str) or not (_FIELD if field else _IDENTIFIER).fullmatch(
        value
    ):
        raise ValueError(f"invalid {label} identity")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} requires a SHA-256 digest")
    return value


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode()).hexdigest()


def _unique_json(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key}")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class FieldSpec:
    name: str
    field_type: str
    description: str
    required: bool = True

    @classmethod
    def from_wire(cls, value: object) -> FieldSpec:
        row = _object(
            value, {"name", "type", "description", "required"}, "signature field"
        )
        name = _id(row["name"], "signature field", field=True)
        field_type = _text(row["type"], 256, "field type")
        if field_type not in _SCALARS:
            options = field_type[5:-1].split("|")
            if (
                not field_type.startswith("enum[")
                or not field_type.endswith("]")
                or len(options) < 2
                or len(set(options)) != len(options)
                or any(
                    not option
                    or option.strip() != option
                    or "[" in option
                    or "]" in option
                    for option in options
                )
            ):
                raise ValueError(f"unsupported field type {field_type}")
        required = row["required"]
        if type(required) is not bool:
            raise ValueError("field required must be boolean")
        return cls(
            name,
            field_type,
            _text(row["description"], 500, "field description"),
            required,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "type": self.field_type,
            "description": self.description,
            "required": self.required,
        }

    def accepts(self, value: object) -> bool:
        match self.field_type:
            case "string":
                return isinstance(value, str)
            case "list[string]":
                return isinstance(value, list) and all(
                    isinstance(item, str) for item in value
                )
            case "boolean":
                return type(value) is bool
            case "number":
                return type(value) is int or (
                    type(value) is float and math.isfinite(value)
                )
            case _:
                return isinstance(value, str) and value in self.field_type[5:-1].split(
                    "|"
                )


def _fields(value: object, maximum: int, label: str) -> tuple[FieldSpec, ...]:
    fields = tuple(FieldSpec.from_wire(item) for item in _list(value, maximum, label))
    if len({field.name for field in fields}) != len(fields):
        raise ValueError(f"{label} contains duplicate field names")
    return fields


def _values(
    value: Mapping[str, object], fields: tuple[FieldSpec, ...], label: str
) -> None:
    if not isinstance(value, Mapping) or not value.keys() <= {
        field.name for field in fields
    }:
        raise ValueError(f"{label} contains undeclared fields")
    for field in fields:
        if field.name not in value:
            if field.required:
                raise ValueError(f"{label} requires field {field.name}")
        elif not field.accepts(value[field.name]):
            raise ValueError(f"{label} has wrong type for {field.name}")


@dataclass(frozen=True, slots=True)
class SignatureSpec:
    name: str
    instructions: str
    input_fields: tuple[FieldSpec, ...]
    output_fields: tuple[FieldSpec, ...]

    @classmethod
    def from_json(cls, raw: str) -> SignatureSpec:
        if not isinstance(raw, str) or len(raw.encode()) > _MAX_SIGNATURE_BYTES:
            raise ValueError("signature exceeds its UTF-8 byte limit")
        try:
            document = json.loads(raw, object_pairs_hook=_unique_json)
        except (json.JSONDecodeError, RecursionError) as error:
            raise ValueError("signature must be a valid JSON object") from error
        row = _object(
            document,
            {"name", "instructions", "input_fields", "output_fields"},
            "signature",
        )
        output = _fields(row["output_fields"], _MAX_OUTPUT_FIELDS, "signature outputs")
        if not output:
            raise ValueError("signature must declare output fields")
        return cls(
            _id(row["name"], "signature name"),
            _text(row["instructions"], 8000, "signature instructions"),
            _fields(row["input_fields"], _MAX_FIELDS, "signature inputs"),
            output,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "instructions": self.instructions,
            "input_fields": [field.to_dict() for field in self.input_fields],
            "output_fields": [field.to_dict() for field in self.output_fields],
        }

    def to_json(self) -> str:
        return _canonical(self.to_dict())

    def validate_output(self, value: Mapping[str, object]) -> None:
        _values(value, self.output_fields, f"{self.name} output")


class _Source(StrEnum):
    REQUEST = "request"
    STAGE = "stage_output"


class _Codec(StrEnum):
    IDENTITY = "identity"
    JSON_OBJECT = "json_object"


@dataclass(frozen=True, slots=True)
class _Input:
    field: str
    source: _Source
    source_stage: str | None
    source_field: str | None
    codec: _Codec

    @classmethod
    def parse(cls, raw: object) -> _Input:
        row = _object(
            raw,
            {"field", "source", "source_stage", "source_field", "codec"},
            "stage input",
        )
        source = _Source(row["source"])
        codec = _Codec(row["codec"])
        stage = row["source_stage"]
        field = row["source_field"]
        if source is _Source.REQUEST:
            if stage is not None or codec is not _Codec.IDENTITY:
                raise ValueError("request input requires an identity field connection")
        else:
            stage = _id(stage, "source stage")
        if codec is _Codec.JSON_OBJECT:
            if field is not None:
                raise ValueError(
                    "JSON object connection binds the complete source stage output"
                )
        else:
            field = _id(field, "source field", field=True)
        return cls(
            _id(row["field"], "input field", field=True), source, stage, field, codec
        )


@dataclass(frozen=True, slots=True)
class _Stage:
    id: str
    signature: str
    inputs: tuple[_Input, ...]


@dataclass(frozen=True, slots=True)
class _Joint:
    id: str
    bindings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VerifierPin:
    id: str
    version_digest: str
    output_sensor: str
    stage: str | None
    field: str

    @classmethod
    def from_wire(cls, raw: object) -> VerifierPin:
        row = _object(
            raw,
            {"id", "version_digest", "mode", "output_sensor", "stage", "field"},
            "verifier",
        )
        if row["mode"] != "queued":
            raise ValueError("verifier mode must be queued")
        return cls(
            _id(row["id"], "verifier"),
            _sha256(row["version_digest"], "verifier revision"),
            _id(row["output_sensor"], "verifier output sensor"),
            None
            if row["stage"] is None
            else _id(row["stage"], "verifier source stage"),
            _id(row["field"], "verifier source field", field=True),
        )

    def queue(
        self,
        *,
        operation_id: UUID,
        contract_digest: str,
        candidate_digest: str,
        rollout_id: str,
        split: str,
        terminal: str,
    ) -> VerificationIntent:
        """Bind verifier work to exact candidate and terminal artifact digests."""
        if not isinstance(operation_id, UUID):
            raise TypeError("verifier operation requires a UUID")
        if split not in {"train", "validation", "test"}:
            raise ValueError("invalid verifier split binding")
        if not isinstance(terminal, str) or not terminal.strip():
            raise ValueError("verifier requires a complete terminal response")
        if not isinstance(rollout_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", rollout_id
        ):
            raise ValueError("invalid verifier rollout identity")
        return VerificationIntent(
            operation_id,
            _sha256(contract_digest, "program contract"),
            _sha256(candidate_digest, "verifier candidate"),
            rollout_id,
            split,
            "sha256:" + hashlib.sha256(terminal.encode()).hexdigest(),
            self,
        )


@dataclass(frozen=True, slots=True)
class VerificationIntent:
    operation_id: UUID
    contract_digest: str
    candidate_digest: str
    rollout_id: str
    split: str
    terminal_digest: str
    verifier: VerifierPin


@dataclass(frozen=True, slots=True)
class MaterializedPipeline:
    digest: str
    pipeline_digest: str
    prompts: tuple[tuple[str, str], ...]
    signatures: tuple[tuple[str, SignatureSpec], ...]
    verifier: VerifierPin

    def signature(self, stage_id: str) -> SignatureSpec:
        for identity, signature in self.signatures:
            if identity == stage_id:
                return signature
        raise ValueError(f"unknown materialized stage {stage_id}")


@dataclass(frozen=True, slots=True)
class StageProgram:
    digest: str
    inputs: tuple[FieldSpec, ...]
    stages: tuple[_Stage, ...]
    terminal_stage: str
    terminal_field: FieldSpec
    terminal_description_immutable: bool
    joints: tuple[_Joint, ...]
    verifier: VerifierPin

    @classmethod
    def from_wire(cls, raw: object) -> StageProgram:
        row = _object(
            raw,
            {"inputs", "stages", "terminal", "joints", "verifier"},
            "stage pipeline",
        )
        inputs = _fields(row["inputs"], _MAX_FIELDS, "program inputs")
        stage_rows = _list(row["stages"], _MAX_STAGES, "pipeline stages")
        stages: dict[str, _Stage] = {}
        bindings: set[str] = set()
        for raw_stage in stage_rows:
            doc = _object(raw_stage, {"id", "signature", "inputs"}, "pipeline stage")
            identity = _id(doc["id"], "stage")
            binding = _id(doc["signature"], "stage signature binding")
            if identity in stages or binding in bindings:
                raise ValueError("duplicate stage identity or signature binding")
            connections = tuple(
                _Input.parse(item)
                for item in _list(doc["inputs"], _MAX_FIELDS, "stage inputs")
            )
            if len({connection.field for connection in connections}) != len(
                connections
            ):
                raise ValueError("duplicate input connection for a stage field")
            stages[identity] = _Stage(identity, binding, connections)
            bindings.add(binding)
        names = {field.name for field in inputs}
        for stage in stages.values():
            for edge in stage.inputs:
                if edge.source is _Source.REQUEST and edge.source_field not in names:
                    raise ValueError(f"stage {stage.id} binds an unknown program input")
                if edge.source is _Source.STAGE and edge.source_stage not in stages:
                    raise ValueError(f"stage {stage.id} binds an unknown source stage")
        ordered: list[_Stage] = []
        visited: set[str] = set()
        while len(ordered) < len(stages):
            ready = [
                stage
                for stage in stages.values()
                if stage.id not in visited
                and all(
                    edge.source_stage is None or edge.source_stage in visited
                    for edge in stage.inputs
                )
            ]
            if not ready:
                raise ValueError("pipeline stages contain a dependency cycle")
            ordered.extend(ready)
            visited.update(stage.id for stage in ready)
        terminal = row["terminal"]
        if not isinstance(terminal, Mapping) or terminal.keys() not in (
            {"stage", "field"},
            {"stage", "field", "immutable_description"},
        ):
            raise ValueError(
                "terminal output fields differ from the declared wire schema"
            )
        immutable_description = terminal.get("immutable_description", False)
        if type(immutable_description) is not bool:
            raise ValueError("terminal immutable_description must be boolean")
        terminal_stage = _id(terminal["stage"], "terminal stage")
        terminal_field = FieldSpec.from_wire(terminal["field"])
        if (
            terminal_stage not in stages
            or terminal_field.field_type != "string"
            or not terminal_field.required
        ):
            raise ValueError(
                "terminal output must be a required string in one declared stage"
            )
        if any(
            edge.source_stage == terminal_stage
            for stage in stages.values()
            for edge in stage.inputs
        ):
            raise ValueError("terminal output stage must be a pipeline sink")
        verifier = VerifierPin.from_wire(row["verifier"])
        if (verifier.stage, verifier.field) != (terminal_stage, terminal_field.name):
            raise ValueError("queued verifier must bind the protected terminal output")
        joints: dict[str, _Joint] = {}
        for raw_joint in _list(row["joints"], _MAX_STAGES, "joint levers"):
            doc = _object(raw_joint, {"id", "bindings"}, "joint lever")
            identity = _id(doc["id"], "joint lever")
            members = tuple(
                _id(item, "joint binding")
                for item in _list(doc["bindings"], _MAX_STAGES, "joint bindings")
            )
            if (
                identity in joints
                or not members
                or len(set(members)) != len(members)
                or not set(members) <= bindings
            ):
                raise ValueError(
                    "joint lever must use unique declared stage signature bindings"
                )
            joints[identity] = _Joint(identity, members)
        return cls(
            _digest({"schema": "hillclimb.stage-pipeline.v1", **row}),
            inputs,
            tuple(ordered),
            terminal_stage,
            terminal_field,
            immutable_description,
            tuple(joints.values()),
            verifier,
        )

    @property
    def signature_bindings(self) -> frozenset[str]:
        return frozenset(stage.signature for stage in self.stages)

    def joint_bindings(self, joint_id: str) -> tuple[str, ...]:
        for joint in self.joints:
            if joint.id == joint_id:
                return joint.bindings
        raise ValueError(f"unknown pipeline joint lever {joint_id}")

    def materialize(
        self,
        parent: Mapping[str, str],
        changes: Mapping[str, str],
        *,
        joint_id: str | None = None,
    ) -> MaterializedPipeline:
        """Validate the complete child before returning any candidate bytes."""
        if joint_id is None:
            if len(changes) > 1 or not changes.keys() <= self.signature_bindings:
                raise ValueError("multiple edits require a declared joint lever")
        elif changes.keys() != set(self.joint_bindings(joint_id)):
            raise ValueError("joint edit must provide every declared signature binding")
        original = self._signatures(parent)
        child = {**parent, **changes}
        candidate = self._signatures(child)
        for stage, signature in candidate.items():
            previous = original[stage]
            if (
                signature.name != previous.name
                or signature.input_fields != previous.input_fields
            ):
                raise ValueError(f"stage {stage} changed its inherited input signature")
        return MaterializedPipeline(
            _digest({"schema": "hillclimb.materialized-pipeline.v1", "prompts": child}),
            self.digest,
            tuple(sorted(child.items())),
            tuple(candidate.items()),
            self.verifier,
        )

    def _signatures(self, prompts: Mapping[str, str]) -> dict[str, SignatureSpec]:
        if not self.signature_bindings <= prompts.keys() or any(
            not isinstance(name, str) or not isinstance(value, str)
            for name, value in prompts.items()
        ):
            raise ValueError("pipeline prompts require every bound signature as text")
        signatures: dict[str, SignatureSpec] = {}
        request = {field.name: field for field in self.inputs}
        for stage in self.stages:
            signature = SignatureSpec.from_json(prompts[stage.signature])
            declared = {field.name: field for field in signature.input_fields}
            if declared.keys() != {edge.field for edge in stage.inputs}:
                raise ValueError(
                    f"stage {stage.id} input fields do not match the pipeline"
                )
            for edge in stage.inputs:
                destination = declared[edge.field]
                if edge.codec is _Codec.JSON_OBJECT:
                    if destination.field_type != "string":
                        raise ValueError(
                            f"stage {stage.id} JSON object input must be a string"
                        )
                    continue
                assert edge.source_field is not None
                if edge.source is _Source.REQUEST:
                    source = request.get(edge.source_field)
                else:
                    assert edge.source_stage is not None
                    source = next(
                        (
                            field
                            for field in signatures[edge.source_stage].output_fields
                            if field.name == edge.source_field
                        ),
                        None,
                    )
                if (
                    source is None
                    or source.field_type != destination.field_type
                    or (destination.required and not source.required)
                ):
                    raise ValueError(
                        f"stage {stage.id} cannot inherit input {edge.field} from its source"
                    )
            if stage.id == self.terminal_stage:
                self._require_terminal(signature)
            signatures[stage.id] = signature
        return signatures

    def _require_terminal(self, signature: SignatureSpec) -> None:
        protected = self.terminal_field
        output = next(
            (
                field
                for field in signature.output_fields
                if field.name == protected.name
            ),
            None,
        )
        if (
            output is None
            or output.field_type != protected.field_type
            or output.required != protected.required
            or (
                self.terminal_description_immutable
                and output.description != protected.description
            )
        ):
            raise ValueError("candidate changed the protected terminal output")

    def bind_inputs(
        self,
        stage_id: str,
        request: Mapping[str, object],
        stage_outputs: Mapping[str, Mapping[str, object]],
        *,
        revision: MaterializedPipeline,
    ) -> dict[str, object]:
        """Build one stage's typed inputs, including validated inherited artifacts."""
        self._require_revision(revision)
        _values(request, self.inputs, "pipeline request")
        stage = next(
            (candidate for candidate in self.stages if candidate.id == stage_id), None
        )
        if stage is None:
            raise ValueError(f"unknown pipeline stage {stage_id}")
        result: dict[str, object] = {}
        for edge in stage.inputs:
            if edge.source is _Source.REQUEST:
                if edge.source_field in request:
                    result[edge.field] = request[edge.source_field]
                continue
            assert edge.source_stage is not None
            upstream = stage_outputs.get(edge.source_stage)
            if upstream is None:
                raise ValueError(
                    f"stage {stage_id} requires output from {edge.source_stage}"
                )
            revision.signature(edge.source_stage).validate_output(upstream)
            if edge.codec is _Codec.JSON_OBJECT:
                result[edge.field] = _canonical(dict(upstream))
            elif edge.source_field in upstream:
                result[edge.field] = upstream[edge.source_field]
        _values(result, revision.signature(stage_id).input_fields, f"{stage_id} inputs")
        return result

    def terminal_output(
        self, output: Mapping[str, object], *, revision: MaterializedPipeline
    ) -> str:
        self._require_revision(revision)
        revision.signature(self.terminal_stage).validate_output(output)
        terminal = output[self.terminal_field.name]
        assert isinstance(terminal, str)
        if not terminal.strip():
            raise ValueError("protected terminal output cannot be empty")
        return terminal

    def queue_verifier(
        self,
        *,
        operation_id: UUID,
        contract_digest: str,
        revision: MaterializedPipeline,
        rollout_id: str,
        split: str,
        terminal: str,
    ) -> VerificationIntent:
        """Bind a verifier intent to settled actor output without granting a score."""
        self._require_revision(revision)
        return self.verifier.queue(
            operation_id=operation_id,
            contract_digest=contract_digest,
            candidate_digest=revision.digest,
            rollout_id=rollout_id,
            split=split,
            terminal=terminal,
        )

    def _require_revision(self, revision: MaterializedPipeline) -> None:
        if (
            revision.pipeline_digest != self.digest
            or revision.verifier != self.verifier
        ):
            raise ValueError("materialized revision belongs to a different pipeline")
