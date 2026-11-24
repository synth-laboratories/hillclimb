"""Optional producer hooks for scored search trials. Return values belong in reward.domain_visuals."""

from collections import defaultdict
from math import isfinite

SEARCH_SPLITS = {"train_reflect", "val_pareto"}


def banking77(split, *, predicted_intent, gold_intent, parse_ok):
    if split not in SEARCH_SPLITS:
        return None
    if (
        type(parse_ok) is not bool
        or not isinstance(gold_intent, str)
        or parse_ok
        and not isinstance(predicted_intent, str)
    ):
        raise ValueError("invalid scored intent")
    return {
        "target_id": "banking77",
        "metrics": {
            "predicted_intent": predicted_intent,
            "gold_intent": gold_intent,
            "parse_ok": parse_ok,
        },
    }


def healthbench2(split, criteria, *, actor_cost_micros, judge_cost_micros):
    """Summarize signed criterion grades. Unknown accounting stays null. A missing grade is unscored."""
    if split not in SEARCH_SPLITS:
        return None
    if any(
        cost is not None and (type(cost) is not int or cost < 0)
        for cost in (actor_cost_micros, judge_cost_micros)
    ):
        raise ValueError("invalid wire cost")
    possible = achieved = hits = positive = 0
    negative = []
    axes = defaultdict(lambda: [0, 0])
    for criterion in criteria:
        points, met = criterion["points"], criterion["met"]
        if (
            type(met) is not bool
            or type(points) not in (int, float)
            or not isfinite(points)
        ):
            raise ValueError("incomplete rubric grade")
        possible += max(0, points)
        achieved += points * met
        positive += int(points > 0)
        hits += int(points > 0 and met)
        if met and points < 0:
            negative.append(criterion["tag"])
        for axis in set(criterion.get("axes", ())):
            axes[axis][0] += points * met
            axes[axis][1] += max(points, 0)
    if not possible:
        raise ValueError("rubric requires positive available points")
    return {
        "target_id": "healthbench2",
        "metrics": {
            "mean_clipped": max(0, min(1, achieved / possible)),
            "unclipped_rubric_sum": achieved,
            "rubric_points_possible": possible,
            "axes": {axis: total / den for axis, (total, den) in axes.items() if den},
            "positive_criteria_hit": hits,
            "positive_criteria_total": positive,
            "negative_penalties_triggered": len(negative),
            "penalty_tags": negative,
            "actor_cost_micros": actor_cost_micros,
            "judge_cost_micros": judge_cost_micros,
        },
    }
