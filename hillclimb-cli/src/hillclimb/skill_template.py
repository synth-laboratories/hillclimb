"""Project scaffolding templates (`hillclimb init`) and agent skill installer (`hillclimb skill install`)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Mapping

from .logs import PreflightError

__all__ = [
    "InitResult",
    "SKILL_MARKDOWN",
    "SkillInstallResult",
    "TEMPLATE_NAMES",
    "TemplateKind",
    "init_project",
    "install_skill",
]


class TemplateKind(StrEnum):
    """Supported starter templates for `hillclimb init`."""

    PYTHON = "python"
    DSPY = "dspy"
    GEPA_AI = "gepa-ai"
    CONTAINER = "container"
    HARBOR = "harbor"


TEMPLATE_NAMES: tuple[str, ...] = tuple(item.value for item in TemplateKind)


@dataclass(frozen=True, slots=True)
class InitResult:
    """Receipt returned by `init_project`."""

    template: TemplateKind
    directory: Path
    files: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "template": self.template.value,
            "directory": str(self.directory),
            "files": list(self.files),
        }


@dataclass(frozen=True, slots=True)
class SkillInstallResult:
    """Receipt returned by `install_skill`."""

    directory: Path
    skill_path: Path
    bytes_written: int

    def to_dict(self) -> dict[str, object]:
        return {
            "directory": str(self.directory),
            "skill_path": str(self.skill_path),
            "bytes_written": self.bytes_written,
        }


_PYTHON_TOML = """schema_version = "hillclimb.optimizer.v1"
algorithm = "gepa"
seed = 1
minibatch = 4

[proposer.guidelines]
format = "structured_markdown"
max_chars = 4000
rules = [
  "Keep instructions deterministic and directly actionable.",
  "Preserve the required output format.",
]

[scoring]
mode = "verifier"

[scoring.verifier]
kind = "python_callback"
weight = 1.0

[stop]
cap_usd = 5.0
max_rounds = 8
patience_rounds = 4
"""

_PYTHON_STARTER = '''"""Minimal Python Hillclimb optimization starter (native proxy metering)."""

from __future__ import annotations


def evaluate(policy: dict[str, str], row: dict[str, str]) -> tuple[float, str]:
    """Score one candidate policy against one training or validation example."""
    instructions = policy.get("stage2_system", "")
    expected = row["expected"]
    prediction = row["expected"] if expected.lower() in instructions.lower() else ""
    score = 1.0 if prediction == expected else 0.0
    feedback = "matched expected answer" if score == 1.0 else f"missed expected={expected!r}"
    return score, feedback


def dataset() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train = [{"id": f"train-{i}", "expected": "approve"} for i in range(4)]
    val = [{"id": f"val-{i}", "expected": "approve"} for i in range(4)]
    return train, val


def main() -> None:
    import hillclimb

    train, val = dataset()
    result = hillclimb.optimize(
        seed={"stage2_system": "Return a concise decision."},
        evaluate=evaluate,
        train=train,
        val=val,
        config="optimizer.toml",
    )
    print("Champion:", result)


if __name__ == "__main__":
    main()
'''

_DSPY_TOML = """schema_version = "hillclimb.optimizer.v1"
algorithm = "gepa"
seed = 1
minibatch = 4

[proposer.guidelines]
rules = [
  "Keep signature instructions clear and grounded in the training feedback.",
]

[policy.write_mask]
mutable_levers = ["predictor"]

[stop]
cap_usd = 5.0
max_rounds = 8
patience_rounds = 4
"""

_DSPY_STARTER = '''"""Minimal DSPy-compatible Hillclimb starter (native metering, zero external proxy)."""

from __future__ import annotations

from dataclasses import dataclass

from hillclimb.compat.dspy import GEPA


@dataclass
class ClassifierSignature:
    instructions: str


class ClassifierModule:
    """Drop-in student module whose `signature.instructions` is optimized by Hillclimb."""

    def __init__(self, instructions: str) -> None:
        self.signature = ClassifierSignature(instructions=instructions)

    def __call__(self, text: str) -> str:
        instr = self.signature.instructions.lower()
        return "yes" if "priority" in instr and "urgent" in text.lower() else "no"


def metric(example: dict[str, str], prediction: str) -> float:
    return 1.0 if prediction == example["label"] else 0.0


def main() -> None:
    trainset = [
        {"id": "t0", "text": "Urgent outage report", "label": "yes"},
        {"id": "t1", "text": "Routine weekly update", "label": "no"},
    ]
    valset = [
        {"id": "v0", "text": "Urgent escalation", "label": "yes"},
        {"id": "v1", "text": "General announcement", "label": "no"},
    ]
    optimizer = GEPA(metric=metric, config="optimizer.toml")
    compiled = optimizer.compile(
        ClassifierModule("Classify incoming messages."),
        trainset=trainset,
        valset=valset,
    )
    print("Compiled instructions:", compiled.signature.instructions)


if __name__ == "__main__":
    main()
'''

_GEPA_AI_TOML = """schema_version = "hillclimb.optimizer.v1"
algorithm = "gepa"
seed = 1
minibatch = 4

[proposer.guidelines]
rules = [
  "Address concrete failures surfaced by the minibatch evaluator.",
]

[policy.write_mask]
mutable_levers = ["system_prompt"]

[stop]
cap_usd = 5.0
max_rounds = 8
patience_rounds = 4
"""

_GEPA_AI_STARTER = '''"""Minimal gepa-ai-compatible Hillclimb starter (native proxy metering)."""

from __future__ import annotations

from hillclimb.compat.gepa_ai import optimize


def evaluate(candidate: dict[str, str], row: dict[str, str]) -> tuple[float, str]:
    prompt = candidate["system_prompt"]
    score = 1.0 if row["keyword"] in prompt.lower() else 0.0
    return score, f"expected keyword {row['keyword']!r} in system_prompt"


def main() -> None:
    result = optimize(
        seed_candidate={"system_prompt": "Answer accurately and concisely."},
        trainset=[{"id": "t0", "keyword": "concise"}, {"id": "t1", "keyword": "verified"}],
        valset=[{"id": "v0", "keyword": "verified"}],
        evaluator=evaluate,
        config="optimizer.toml",
    )
    print("Best candidate:", result.best_candidate)


if __name__ == "__main__":
    main()
'''

_CONTAINER_TOML = """algorithm_id = "gepa"
cap_micros = 5_000_000
effect_ceiling_micros = 100_000

[search]
seed = 1
iterations = 8
minibatch = 4
train = ["train-0", "train-1", "train-2", "train-3"]
validation = ["val-0", "val-1"]
final_test = ["test-0", "test-1"]

[runtime.container.target]
url = "http://127.0.0.1:8090"

[runtime.container.target.tasks.train-0]
split = "train_reflect"

[runtime.container.target.tasks.val-0]
split = "val_pareto"

[runtime.container.target.tasks.test-0]
split = "final_test"
"""

_CONTAINER_STARTER = '''"""Minimal local HTTP container target for `hillclimb run --config optimizer.toml`."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class TargetHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._send(200, {"status": "ok"})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/evaluate":
            prompt = str(payload.get("policy", {}).get("instructions", ""))
            score = 1.0 if "accurate" in prompt.lower() else 0.25
            self._send(200, {"score": score, "trace": {"prompt_len": len(prompt)}})
            return
        self._send(404, {"error": "not found"})

    def _send(self, code: int, body: dict[str, object]) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 8090), TargetHandler).serve_forever()
'''

_HARBOR_TOML = """algorithm_id = "gepa"
cap_micros = 5_000_000
effect_ceiling_micros = 100_000

[search]
seed = 1
iterations = 4
minibatch = 2
train = ["task-train-0", "task-train-1"]
validation = ["task-val-0"]
final_test = ["task-test-0"]

[runtime.harbor]
mode = "local"
network_disabled_verifier = true

[runtime.harbor.target.tasks.task-train-0]
split = "train_reflect"

[runtime.harbor.target.tasks.task-val-0]
split = "val_pareto"

[runtime.harbor.target.tasks.task-test-0]
split = "final_test"
"""

_HARBOR_TASK_SPEC = """[task]
name = "sample-harbor-task"
timeout_seconds = 60

[verifier]
command = ["sh", "-c", "test -s /workspace/output.txt"]
network = "none"
"""

_TEMPLATES: Mapping[TemplateKind, Mapping[str, str]] = {
    TemplateKind.PYTHON: {
        "optimizer.toml": _PYTHON_TOML,
        "optimize.py": _PYTHON_STARTER,
    },
    TemplateKind.DSPY: {
        "optimizer.toml": _DSPY_TOML,
        "optimize.py": _DSPY_STARTER,
    },
    TemplateKind.GEPA_AI: {
        "optimizer.toml": _GEPA_AI_TOML,
        "optimize.py": _GEPA_AI_STARTER,
    },
    TemplateKind.CONTAINER: {
        "optimizer.toml": _CONTAINER_TOML,
        "target_server.py": _CONTAINER_STARTER,
    },
    TemplateKind.HARBOR: {
        "optimizer.toml": _HARBOR_TOML,
        "harbor_task.toml": _HARBOR_TASK_SPEC,
    },
}


SKILL_MARKDOWN = """---
name: hillclimb
description: Operate Hillclimb optimization campaigns via the agent-first CLI and MCP server.
---

# Hillclimb Agent Skill

Use `hillclimb` CLI commands or `hillclimb mcp` tools to launch, inspect, steer, fork, and verify optimization runs safely.

## Quick CLI Reference

- Discover runtime capabilities and installed adapters:
  `hillclimb capabilities --json`
- Scaffold a starter project (`python`, `dspy`, `gepa-ai`, `container`, `harbor`):
  `hillclimb init python --dir .`
- Launch a campaign detached and stream events:
  `hillclimb run --config optimizer.toml --detach --json`
  `hillclimb logs -f <run_id> --json`
- Inspect status, steer, pause, resume, or fork a checkpoint slice:
  `hillclimb status <run_id> --json`
  `hillclimb pause <run_id> --op-id <uuid>`
  `hillclimb steer <run_id> "Focus on edge-case parse accuracy" --op-id <uuid>`
  `hillclimb resume <run_id> --live --op-id <uuid>`
  `hillclimb slice <run_id> --destination /abs/path/to/child --proposal`

## Semantic Exit Codes

- `0`: Campaign completed cleanly (or inspect command succeeded)
- `2`: Budget or request cap reached (`paused` on cap / `cap_reached`)
- `3`: Outcome `in_doubt` (pending price confirmation or effect reconciliation)
- `4`: Preflight or configuration error

## MCP Server (`hillclimb mcp`)

Run `hillclimb mcp` over stdio JSON-RPC 2.0 (or `--url http://127.0.0.1:<port>` for an HTTP daemon).

- **Bounded tools:**
  - `hillclimb_capabilities`
  - `hillclimb_start`
  - `hillclimb_status`
  - `hillclimb_tail(after_seq, limit)`
  - `hillclimb_inspect_failures(policy_id, limit)` (search splits only; `final_test` stays sealed)
  - `hillclimb_control(action, op_id, ...)` (`pause` | `resume` | `steer_proposer` | `inject_candidate`)
  - `hillclimb_slice(at_seq, intervention)`
  - `hillclimb_champion`
- **Subscribable resources & push events:**
  - Subscribe to `hillclimb://runs/{run_id}/status` and `hillclimb://runs/{run_id}/events` for `notifications/resources/updated`.
  - Supply `_meta.progressToken` to receive `notifications/progress` on generation/rollout milestones.
  - Listen for `notifications/message` on `frontier_accepted`, `budget_warning`, `in_doubt`, and `completed`.
"""


def init_project(
    template: str,
    dest_dir: Path,
    *,
    overwrite: bool = False,
) -> InitResult:
    """Scaffold a starter Hillclimb project and `optimizer.toml` in `dest_dir`."""
    try:
        kind = TemplateKind(template)
    except ValueError as error:
        supported = ", ".join(TEMPLATE_NAMES)
        raise PreflightError(
            f"Unknown init template {template!r}; expected one of: {supported}"
        ) from error

    target_dir = Path(dest_dir).expanduser().resolve()
    files = _TEMPLATES[kind]
    assert files, "template file map must not be empty"

    if not overwrite:
        existing = [name for name in files if (target_dir / name).exists()]
        if existing:
            joined = ", ".join(existing)
            raise PreflightError(
                f"Refusing to overwrite existing files in {target_dir}: {joined} (pass --force)"
            )

    target_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for filename, content in files.items():
        file_path = target_dir / filename
        file_path.write_text(content, encoding="utf-8")
        written.append(str(file_path))

    return InitResult(template=kind, directory=target_dir, files=tuple(written))


def install_skill(
    target_dir: Path,
    *,
    overwrite: bool = False,
) -> SkillInstallResult:
    """Install `SKILL.md` for coding agents into `target_dir`."""
    directory = Path(target_dir).expanduser().resolve()
    skill_path = directory / "SKILL.md"
    if skill_path.exists() and not overwrite:
        raise PreflightError(
            f"Skill file already exists at {skill_path} (pass --force to replace)"
        )
    directory.mkdir(parents=True, exist_ok=True)
    encoded = SKILL_MARKDOWN.encode("utf-8")
    skill_path.write_bytes(encoded)
    assert skill_path.is_file(), "SKILL.md must exist after install_skill"
    return SkillInstallResult(
        directory=directory,
        skill_path=skill_path,
        bytes_written=len(encoded),
    )
