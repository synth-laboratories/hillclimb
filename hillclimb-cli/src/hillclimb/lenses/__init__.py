"""Observer-only reducers over the daemon's bounded window of settled trial receipts."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import fmean
from math import isfinite

__all__ = ["project"]

CLASSIC = (
    "collect_wood",
    "place_table",
    "eat_cow",
    "collect_sapling",
    "collect_drink",
    "make_wood_pickaxe",
    "make_wood_sword",
    "place_plant",
    "defeat_zombie",
    "collect_stone",
    "place_stone",
    "eat_plant",
    "defeat_skeleton",
    "make_stone_pickaxe",
    "make_stone_sword",
    "wake_up",
    "place_furnace",
    "collect_coal",
    "collect_iron",
    "collect_diamond",
    "make_iron_pickaxe",
    "make_iron_sword",
)


def _ladder(metrics):
    names = metrics.get("achievement_names", CLASSIC)
    if not isinstance(names, (tuple, list)) or len(names) not in (22, 66):
        raise ValueError("expected a Classic or Full achievement ladder")
    if not all(isinstance(name, str) and 0 < len(name) <= 100 for name in names) or len(
        set(names)
    ) != len(names):
        raise ValueError("invalid achievement names")
    return names


def _achievements(metrics):
    mask = metrics["achievements_bitmask"]
    names = _ladder(metrics)
    # Hex strings retain all 66 bits on wire transports with 64-bit JSON integers.
    if isinstance(mask, str) and mask.startswith("0x") and len(mask) <= 19:
        mask = int(mask, 16)
    if type(mask) is not int or not 0 <= mask < 1 << len(names):
        raise ValueError("invalid achievement mask")
    return [name for i, name in enumerate(names) if mask & (1 << i)]


def _craftax(rows):
    summary = []
    for policy, split, trials in _groups(rows):
        # Checkpoint suffixes have selected world starts. The fresh-world funnel uses only fresh starts.
        fresh = [r for r in trials if r["origin"] == "fresh"]
        counts = Counter(name for r in fresh for name in _achievements(r["metrics"]))
        for goal in dict.fromkeys(
            goal for row in fresh for goal in _ladder(row["metrics"])
        ):
            summary.append(
                dict(
                    policy=policy,
                    split=split,
                    metric=goal,
                    value=counts[goal] / len(fresh),
                )
            )
        deaths = Counter(
            r["metrics"]["death_cause"]
            for r in trials
            if r["metrics"].get("death_cause")
        )
        summary.extend(
            dict(policy=policy, split=split, metric=cause, value=count)
            for cause, count in deaths.items()
        )
        latest = trials[-1]
        summary.extend(
            dict(policy=policy, split=split, metric=k, value=v)
            for k, v in _world_summary(latest)
        )
        summary.append(
            dict(
                policy=policy,
                split=split,
                metric="Rejected plans",
                value=sum(r["metrics"].get("rejected_plans", 0) for r in trials),
            )
        )
    return summary


def _world_summary(row):
    m = row["metrics"]
    result = []
    for key in ("vitals", "inventory"):
        if isinstance(m.get(key), dict):
            result.append(
                (key.title(), ", ".join(f"{k}={v}" for k, v in sorted(m[key].items())))
            )
    for index, line in enumerate(m.get("local_map", [])[:9]):
        result.append((f"Viewport {index + 1}", str(line)[:32]))
    return result


def _groups(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["policy_id"], row["split"]].append(row)
    return [(policy, split, group) for (policy, split), group in sorted(groups.items())]


def _banking(rows):
    summary = []
    for policy, split, trials in _groups(rows):
        metrics = [r["metrics"] for r in trials]
        pairs = Counter(
            (m["gold_intent"], m["predicted_intent"])
            for m in metrics
            if m["parse_ok"] and m["gold_intent"] != m["predicted_intent"]
        )
        for (gold, predicted), count in pairs.most_common():
            summary.append(
                dict(
                    policy=policy,
                    split=split,
                    gold=gold,
                    predicted=predicted,
                    metric=f"{gold} → {predicted}",
                    value=count,
                )
            )
        summary.append(
            dict(
                policy=policy,
                split=split,
                metric="Parse validity",
                value=fmean(m["parse_ok"] for m in metrics),
            )
        )
    return summary


def _health(rows):
    summary = []
    for policy, split, trials in _groups(rows):
        values = defaultdict(list)
        for r in trials:
            m = r["metrics"]
            for key in (
                "mean_clipped",
                "unclipped_rubric_sum",
                "negative_penalties_triggered",
                "actor_cost_micros",
                "judge_cost_micros",
            ):
                values[key].append(m[key])
            for key in ("positive_criteria_hit", "positive_criteria_total"):
                if key in m:
                    values[key].append(m[key])
            for axis, score in m["axes"].items():
                values[axis].append(score)
        for key, series in values.items():
            total = key.endswith("_micros") or key in (
                "negative_penalties_triggered",
                "positive_criteria_hit",
                "positive_criteria_total",
            )
            summary.append(
                dict(
                    policy=policy,
                    split=split,
                    metric=key,
                    value=None
                    if None in series
                    else sum(series)
                    if total
                    else fmean(series),
                )
            )
    return summary


# Classic crafting requirements shared by the mining and tool branches.
REQUIRES = {
    "place_table": ("collect_wood",),
    "make_wood_pickaxe": ("place_table", "collect_wood"),
    "make_wood_sword": ("place_table", "collect_wood"),
    "collect_stone": ("make_wood_pickaxe",),
    "place_stone": ("collect_stone",),
    "place_furnace": ("collect_stone",),
    "make_stone_pickaxe": ("place_table", "collect_stone", "collect_wood"),
    "make_stone_sword": ("place_table", "collect_stone", "collect_wood"),
    "collect_coal": ("make_wood_pickaxe",),
    "collect_iron": ("make_stone_pickaxe",),
    "make_iron_pickaxe": (
        "collect_iron",
        "collect_coal",
        "place_table",
        "place_furnace",
        "collect_wood",
    ),
    "make_iron_sword": (
        "collect_iron",
        "collect_coal",
        "place_table",
        "place_furnace",
        "collect_wood",
    ),
    "collect_diamond": ("make_iron_pickaxe",),
    "eat_plant": ("place_plant",),
    "place_plant": ("collect_sapling",),
}


def _gelo_worlds(algo, rows, summary):
    result = []
    telemetry = {row["rollout_id"]: row for row in rows}
    base_goals = set().union(
        *(
            _achievements(r["metrics"])
            for r in rows
            if r["policy_id"] == algo.get("best_base") and r["origin"] == "fresh"
        )
    )
    themes = algo.get("themes", {})
    for goal in dict.fromkeys(goal for row in rows for goal in _ladder(row["metrics"])):
        states = [
            (key, "saturated" if t["saturated"] else t["status"])
            for key, t in themes.items()
            if t["goal"] == goal
        ]
        result.append(
            dict(
                policy=", ".join(t for t, _ in states) or "—",
                metric=goal,
                value=", ".join(s for _, s in states)
                or ("base solved" if goal in base_goals else "unreached"),
                requires=REQUIRES.get(goal, ()),
            )
        )
    for name, theme in themes.items():
        result.append(
            dict(
                policy=name,
                metric=theme["goal"],
                value="saturated" if theme["saturated"] else theme["status"],
            )
        )
        for policy, suffixes in theme.get("paired_suffixes", {}).items():
            for suffix in suffixes:
                key = suffix["checkpoint"]
                baseline = next(
                    (
                        s
                        for s in theme.get("paired_bases", {}).get(
                            policy, theme.get("baseline", [])
                        )
                        if s["checkpoint"] == key
                    ),
                    None,
                )
                if baseline is None:
                    continue
                after = telemetry.get(suffix["rollout"])
                if after is None or after["policy_id"] != policy:
                    continue
                if (
                    not isinstance(after["origin"], dict)
                    or "checkpoint" not in after["origin"]
                ):
                    continue
                origin = after["origin"]["checkpoint"]
                if any(origin[k] != v for k, v in key.items()):
                    continue
                gained = sorted(
                    set(suffix["achievements"]) - set(baseline["achievements"])
                )
                before = telemetry.get(baseline["rollout"])
                if before and (
                    before["origin"] != after["origin"]
                    or before["seed"] != after["seed"]
                    or before["task_id"] != after["task_id"]
                ):
                    continue
                result.append(
                    dict(
                        policy=policy,
                        metric=f"{key['parent_rollout_id']} @ {key['policy_call_index']}",
                        value=suffix["score"] - baseline["score"],
                        gained=gained,
                        base_world=before["metrics"] if before else None,
                        child_world=after["metrics"],
                        base_rollout=baseline["rollout"],
                        child_rollout=suffix["rollout"],
                    )
                )
    return result


def _frontier(algo, rows, summary):
    policies = {p["policy"] for p in algo["frontier"]}
    return [
        r for r in summary if r["policy"] in policies and r["split"] == "val_pareto"
    ]


def _clinical_frontier(algo, rows, summary):
    frontier = _frontier(algo, rows, summary)
    seed = algo.get("seed_policy")
    reference = {
        (r["task_id"], r["seed"]): r["metrics"]
        for r in rows
        if r["split"] == "val_pareto"
        and r["policy_id"] == seed
        and r["origin"] == "fresh"
    }
    for policy in (p["policy"] for p in algo["frontier"] if p["policy"] != seed):
        for row in rows:
            key = (row["task_id"], row["seed"])
            if (
                row["policy_id"] != policy
                or row["split"] != "val_pareto"
                or row["origin"] != "fresh"
                or key not in reference
            ):
                continue
            base, m = reference[key], row["metrics"]
            for tag in sorted(
                set(m.get("penalty_tags") or []) - set(base.get("penalty_tags") or [])
            ):
                frontier.append(
                    dict(
                        policy=policy,
                        metric=f"New penalty · {tag}",
                        value=m["mean_clipped"] - base["mean_clipped"],
                        task=row["task_id"],
                    )
                )
            for axis in m["axes"].keys() & base["axes"].keys():
                frontier.append(
                    dict(
                        policy=policy,
                        metric=f"Δ {axis}",
                        value=m["axes"][axis] - base["axes"][axis],
                        task=row["task_id"],
                    )
                )
    return frontier


def _demo_confusion(algo, rows, summary):
    coverage = algo.get("demo_coverage", [])
    tasks = algo.get("demo_task_ids", [])
    observed = {
        r["task_id"]: r["metrics"]["gold_intent"]
        for r in rows
        if r["split"] == "train_reflect"
    }
    programs = {p["policy_id"]: p["parameters"] for p in algo["programs"]}
    result = []
    for row in summary:
        if (
            "gold" not in row
            or row["policy"] not in programs
            or row["split"] != "val_pareto"
        ):
            continue
        for module, pack in enumerate(programs[row["policy"]][1::2]):
            if module >= len(coverage) or pack >= len(coverage[module]):
                continue
            labels = list(coverage[module][pack])
            if module < len(tasks) and pack < len(tasks[module]):
                labels = [
                    observed.get(task, label)
                    for task, label in zip(tasks[module][pack], labels, strict=True)
                ]
            counts = (
                "unknown"
                if None in labels
                else f"{labels.count(row['gold'])} / {labels.count(row['predicted'])}"
            )
            result.append(
                dict(
                    policy=row["policy"],
                    metric=row["metric"],
                    module=module,
                    pack=pack,
                    value=counts,
                    errors=row["value"],
                )
            )
    return result


TARGETS = {"craftax": _craftax, "banking77": _banking, "healthbench2": _health}
JOINT = {
    ("go-ex", "craftax"): _gelo_worlds,
    ("gepa", "craftax"): _frontier,
    ("gepa", "healthbench2"): _clinical_frontier,
    ("mipro", "banking77"): _demo_confusion,
}


def project(bundle, algorithm):
    """Return independent optional decks from eligible receipts only."""
    target = bundle.get("target")
    if not target:
        return {}
    target_id = target["target_id"]
    rows = [
        row
        for row in target["rows"]
        if row["split"] in ("train_reflect", "val_pareto")
        and row["target_id"] == target_id
    ]
    if not rows:
        return {}
    reducer = TARGETS.get(target_id)
    valid = []
    rejected = 0
    for row in rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            rejected += 1
            continue
        try:
            if reducer:
                reducer([row])
            if target_id == "banking77" and (
                type(metrics["parse_ok"]) is not bool
                or not isinstance(metrics["gold_intent"], str)
            ):
                raise ValueError("invalid intent fields")
            if target_id == "healthbench2" and not all(
                isinstance(v, (int, float)) and isfinite(v)
                for v in metrics["axes"].values()
            ):
                raise ValueError("invalid clinical axes")
        except (ValueError, KeyError, TypeError, AttributeError):
            rejected += 1
            continue
        valid.append(row)
    rows = valid
    summary = (
        reducer(rows)
        if reducer
        else [
            dict(policy=row["policy_id"], split=row["split"], metric=key, value=value)
            for row in rows
            for key, value in row["metrics"].items()
            if isinstance(value, (str, float, int))
        ]
    )
    if rejected:
        summary.append(
            dict(policy="—", metric="Invalid telemetry rows", value=rejected)
        )
    decks = {"target": dict(target_id=target_id, rows=rows, summary=summary)}
    joint = JOINT.get((algorithm, target_id))
    if joint and rows:
        decks["joint"] = dict(
            lens=f"{algorithm}__{target_id}",
            summary=joint(bundle["algo"], rows, summary),
        )
    return decks
