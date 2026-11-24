from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias
from contextlib import contextmanager

from .canvas_spec import table as _table, plot as _plot
from .lenses import project
from .lenses.panels import lens_panels as project_panels

__all__ = ["write_visuals"]

Document: TypeAlias = Mapping[str, object]
INPUTS = ("core", "candidates", "algo", "edits")
MAX_INPUT_BYTES = 512 * 1024
MAX_ROWS = 20_000


def _bounded(value: object, rows: list[int], *, array: bool = False) -> object:
    if isinstance(value, Mapping):
        return {name: _bounded(child, rows) for name, child in value.items()}
    if isinstance(value, list):
        result = []
        for child in value:
            if rows[0] == 0:
                break
            rows[0] -= 1
            result.append(_bounded(child, rows, array=True))
        return result
    if isinstance(value, str) and array:
        return value[:1024]
    return value


def _input(document: Document) -> bytes:
    budget = MAX_ROWS
    while budget > 0:
        bounded = _bounded(document, [budget])
        payload = json.dumps(bounded, separators=(",", ":"), allow_nan=False).encode()
        if len(payload) <= MAX_INPUT_BYTES:
            return payload
        budget //= 2
    raise ValueError("visual metadata exceeds the Canvas input limit")


def _replace(
    directory: Path, name: str, data: bytes, *, preserve: bool = False
) -> None:
    with tempfile.NamedTemporaryFile(
        dir=directory, prefix=f".{name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            if preserve:
                try:
                    os.link(temporary, directory / name)
                except FileExistsError:
                    pass
            else:
                temporary.replace(directory / name)
        finally:
            temporary.unlink(missing_ok=True)


@contextmanager
def view_lock(directory):
    with (directory / ".canvas.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def lens_registry():
    home = Path(os.environ.get("HILLCLIMB_HOME", Path.home() / ".hillclimb"))
    return home / "lenses"


def validate_view(payload):
    from mlok_temper.ui.canvas.model import parse

    if len(payload) > MAX_INPUT_BYTES:
        raise ValueError("Canvas spec exceeds 512 KiB")
    spec = parse(payload.decode())
    allowed = {f"{name}.json" for name in (*INPUTS, "target", "joint")}
    if any(source["path"] not in allowed for source in spec.get("inputs", {}).values()):
        raise ValueError("Canvas input outside observer projections")
    return spec


def lens_name(algorithm, target):
    name = f"{algorithm}__{target}"
    if len(name) > 160 or not all(
        c.isascii() and (c.isalnum() or c in "_-") for c in name
    ):
        raise ValueError("Invalid lens identifier")
    return name


def _optimizer_tables(algorithm, algo):
    panels = []
    if algorithm == "go-ex":
        algo["rulers"] = [
            {"metric": label, "value": algo.get(key) or "—"}
            for label, key in (
                ("Best base", "best_base"),
                ("Global leader", "global_leader"),
                ("Champion", "champion"),
            )
        ]
        algo["theme_rows"] = [
            dict(
                theme=ident,
                state="saturated" if t["saturated"] else t["status"],
                seeds=len(t["checkpoints"]),
                goal=t["goal"],
                attempts=len(t.get("attempts", [])),
                winner=t.get("winner") or "—",
            )
            for ident, t in algo.get("themes", {}).items()
        ]
        panels.extend(
            [
                _table(
                    "GELO rulers",
                    "algo",
                    "rulers",
                    [("Ruler", "metric"), ("Policy", "value")],
                ),
                _table(
                    "Themes",
                    "algo",
                    "theme_rows",
                    [
                        ("Theme", "theme"),
                        ("State", "state"),
                        ("Seeds", "seeds"),
                        ("Attempts", "attempts"),
                    ],
                ),
            ]
        )
    elif algorithm == "gepa":
        coverage = algo.get("instance_coverage", [])
        algo["pareto"] = [
            dict(
                policy=p["policy"], mean=sum(p["scores"]) / len(p["scores"]), coverage=c
            )
            for p, c in zip(algo.get("frontier", []), coverage)
            if p["scores"]
        ]
        algo["changes"] = [
            dict(
                child=d.get("child") or "—",
                fixes=d.get("deltas", []).count("Fix"),
                breaks=d.get("deltas", []).count("Break"),
            )
            for d in algo.get("decisions", [])
        ]
        panels.extend(
            [
                _table(
                    "Pareto archive",
                    "algo",
                    "pareto",
                    [("Policy", "policy"), ("Mean", "mean"), ("Coverage", "coverage")],
                ),
                _table(
                    "Matched examples",
                    "algo",
                    "changes",
                    [("Child", "child"), ("Fix", "fixes"), ("Break", "breaks")],
                ),
            ]
        )
    elif algorithm == "mipro":
        algo["surface"] = [
            dict(
                program=p["policy_id"],
                module=i // 2,
                instruction=values[i],
                demos=values[i + 1],
                delta=p.get("paired_delta"),
            )
            for p in algo.get("programs", [])
            for values in [p.get("parameters", [])]
            for i in range(0, len(values) - 1, 2)
        ]
        panels.append(
            _table(
                "Instruction × demo pack",
                "algo",
                "surface",
                [
                    ("Module", "module"),
                    ("Instruction", "instruction"),
                    ("Pack", "demos"),
                    ("Paired Δ", "delta"),
                ],
            )
        )
    if algorithm == "mipro":
        cells = {}
        for row in algo["surface"]:
            if row["delta"] is not None:
                # One latest observed value per categorical coordinate for each module.
                cells[(row["module"], row["instruction"], row["demos"])] = dict(
                    instruction=f"m{row['module']}:i{row['instruction']}",
                    pack=f"d{row['demos']}",
                    delta=row["delta"],
                )
        algo["categories"] = list(cells.values())[-60:]
        if algo["categories"]:
            panels.append(
                _plot(
                    "heatmap",
                    "Observed TPE deltas",
                    "algo",
                    "categories",
                    row="instruction",
                    column="pack",
                    value="delta",
                )
            )
    if algorithm == "gepa" and algo["pareto"]:
        panels.append(
            _plot(
                "scatter",
                "Validation × coverage",
                "algo",
                "pareto",
                x="coverage",
                y="mean",
                label="policy",
            )
        )
    return panels


def _algorithm_panel(algorithm: str) -> Document:
    if algorithm == "gepa":
        title, path, columns = (
            "GEPA generations",
            "decisions",
            (
                ("Parent", "parent"),
                ("Child", "child"),
                ("Decision", "decision"),
            ),
        )
    elif algorithm == "mipro":
        title, path, columns = (
            "MIPRO programs",
            "programs",
            (
                ("Policy", "policy_id"),
                ("Examples", "examples"),
                ("Paired delta", "paired_delta"),
            ),
        )
    elif algorithm == "go-ex":
        title, path, columns = (
            "GELO promotions",
            "promotions",
            (
                ("Candidate", "candidate"),
                ("Paired seeds", "paired_seeds"),
                ("Mean delta", "mean_delta"),
            ),
        )
    elif algorithm == "eval":
        title, path, columns = (
            "Baseline scores",
            "baseline_scores",
            (
                ("Split", "split"),
                ("Episode", "episode"),
                ("Score", "score"),
            ),
        )
    else:
        title, path, columns = (
            algorithm,
            "rows",
            (("Metric", "metric"), ("Value", "value")),
        )
    return {
        "kind": "table",
        "title": title,
        "from": {
            "source": {
                "input": "algo",
                "path": path,
                "transform": [{"op": "limit", "count": 400}],
            },
            "columns": [
                {"header": header, "field": field} for header, field in columns
            ],
        },
    }


def write_visuals(
    directory: Path, bundle: Mapping[str, Document], *, algorithm: str
) -> None:
    """Write bounded Canvas inputs and preserve the operator's current view."""
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    decks = project(bundle, algorithm)
    lens_panels = project_panels(decks)
    data = {name: bundle[name] for name in INPUTS}
    data.update(decks)
    data["algo"] = dict(data["algo"])
    primary_panel = _algorithm_panel(algorithm)
    data["algo"].setdefault(primary_panel["from"]["source"]["path"], [])
    panels = _optimizer_tables(algorithm, data["algo"])
    edits = dict(data["edits"])
    inbox = [dict(r) for r in edits.get("inbox", [])]
    admitted = {receipt["op_id"] for receipt in inbox}
    edits["queue"] = [
        *inbox,
        *(
            dict(row, status="applied")
            for row in reversed(edits.get("commands", []))
            if row.get("op_id") not in admitted
        ),
    ]
    for row in edits["queue"]:
        cmd = row["command"]
        row["label"] = (
            next(iter(cmd.get("algo", cmd)), "") if isinstance(cmd, dict) else cmd
        )
    data["edits"] = edits
    # Temper Canvas v1 binds four files. Fold shared tables into the core input when lenses mount.
    if decks:
        data["core"] = dict(
            data["core"], candidates=data["candidates"], edits=data["edits"]
        )
    documents = {name: _input(doc) for name, doc in data.items()}
    for name in {"target", "joint"} - decks.keys():
        (directory / f"{name}.json").unlink(missing_ok=True)
    for name, payload in documents.items():
        _replace(directory, f"{name}.json", payload)

    spec = {
        "version": 1,
        "title": f"{algorithm.upper()} campaign",
        "layout": {"columns": 2, "minColumnWidth": 36},
        "inputs": {name: {"path": f"{name}.json"} for name in INPUTS},
        "panels": [
            {
                "kind": "metrics",
                "title": "Campaign",
                "from": {
                    "source": {"input": "core", "path": "summary_metrics"},
                    "items": [
                        {"label": "Dev", "value": "dev"},
                        {"label": "Held out", "value": "heldout"},
                        {"label": "Spent USD", "value": "spent_usd"},
                        {"label": "Reserved USD", "value": "reserved_usd"},
                        {"label": "In doubt USD", "value": "in_doubt_usd"},
                        {"label": "Cap USD", "value": "cap_usd"},
                        {"label": "Rollouts", "value": "rollouts_completed"},
                        {"label": "Rollouts/s", "value": "rollouts_per_second"},
                        {"label": "Active rollouts", "value": "active_rollouts"},
                        {"label": "Cache hits", "value": "cache_hits"},
                    ],
                },
            },
            {
                "kind": "series",
                "title": "Cost",
                "from": {
                    "source": {"input": "core", "path": "cost_series"},
                    "x": "step",
                    "y": "spent_usd",
                },
            },
            {
                "kind": "table",
                "title": "Candidates",
                "from": {
                    "source": {
                        "input": "candidates",
                        "path": "rows",
                        "transform": [{"op": "limit", "count": 400}],
                    },
                    "columns": [
                        {"header": "Candidate", "field": "candidate_id"},
                        {"header": "Score", "field": "score"},
                    ],
                },
                "cellStyles": [
                    {"column": "Score", "op": "gt", "value": 0, "color": "#45B78D"},
                    {"column": "Score", "op": "lt", "value": 0, "color": "#CF655F"},
                ],
            },
            {
                "kind": "table",
                "title": "Commands",
                "from": {
                    "source": {
                        "input": "edits",
                        "path": "queue",
                        "transform": [{"op": "limit", "count": 400}],
                    },
                    "columns": [
                        {"header": "Operation", "field": "op_id"},
                        {"header": "Command", "field": "label"},
                        {"header": "State", "field": "status"},
                    ],
                },
            },
        ],
    }
    spec["panels"].extend([primary_panel, *panels])
    if decks:
        spec["inputs"] = {k: {"path": f"{k}.json"} for k in ("core", "algo", *decks)}
        for panel in spec["panels"]:
            source = panel["from"]["source"]
            if source["input"] in {"candidates", "edits"}:
                source["path"] = source["input"] + "." + source["path"]
                source["input"] = "core"
        for name, doc in decks.items():
            title = doc["target_id"] if name == "target" else doc["lens"]
            spec["panels"].append(
                {
                    "kind": "table",
                    "title": title,
                    "from": {
                        "source": {
                            "input": name,
                            "path": "summary",
                            "transform": [{"op": "limit", "count": 400}],
                        },
                        "columns": [
                            {"header": k.title(), "field": k}
                            for k in ("policy", "metric", "value")
                        ],
                    },
                }
            )
    if data["core"].get("active_agents"):
        spec["panels"].insert(
            1,
            _table(
                "Active roles",
                "core",
                "active_agents",
                [("Role", "role"), ("Count", "count")],
            ),
        )
    spec["panels"].extend(lens_panels)
    # Producers may supply bounded declarative panels using their own input.
    # Validate each candidate with Temper before replacing the portable default.
    for name, hints in (
        ("algo", data["algo"].get("board")),
        (
            "target",
            next(
                (
                    r.get("view")
                    for r in reversed(decks.get("target", {}).get("rows", []))
                    if r.get("view")
                ),
                None,
            ),
        ),
    ):
        if not isinstance(hints, dict) or not isinstance(hints.get("panels"), list):
            continue
        try:
            from mlok_temper.ui.canvas.model import parse
        except ImportError:
            continue
        for panel in hints["panels"][:4]:
            if (
                not isinstance(panel, dict)
                or panel.get("from", {}).get("source", {}).get("input") != name
            ):
                continue
            candidate = dict(spec, panels=[*spec["panels"], panel])
            try:
                parse(json.dumps(candidate))
            except (ValueError, TypeError):
                continue
            spec = candidate
    payload = json.dumps(spec, separators=(",", ":")).encode()
    base, active = directory / "view.base.json", directory / "view.active.json"
    with view_lock(directory):
        owned_default = (
            base.exists()
            and active.exists()
            and active.read_bytes() == base.read_bytes()
        )
        # A new pair can first appear after several telemetry-free revisions.
        previous = json.loads(base.read_bytes()) if base.exists() else {}
        previous_target = previous.get("title")
        target_id = decks.get("target", {}).get("target_id")
        if target_id and (
            not active.exists()
            or owned_default
            and previous_target != f"{algorithm.upper()} · {target_id}"
        ):
            try:
                saved = lens_registry() / (
                    lens_name(algorithm, target_id) + ".canvas.json"
                )
            except ValueError:
                saved = None
            if saved and saved.is_file() and saved.stat().st_size <= MAX_INPUT_BYTES:
                cached = saved.read_bytes()
                try:
                    configured = validate_view(cached)
                    if configured.get("inputs", {}).keys() <= spec["inputs"].keys():
                        _replace(directory, "view.active.json", cached)
                        owned_default = False
                except (ValueError, TypeError, UnicodeError):
                    pass
        if target_id:
            spec["title"] = f"{algorithm.upper()} · {target_id}"
            payload = json.dumps(spec, separators=(",", ":")).encode()
        _replace(directory, "view.base.json", payload)
        _replace(directory, "view.active.json", payload, preserve=not owned_default)
