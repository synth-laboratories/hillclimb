"""Target-specific figures over bounded semantic projections."""

from ..canvas_spec import plot, table
from . import CLASSIC


def lens_panels(decks):
    panels = []
    if not decks:
        return panels
    target = decks["target"]
    joint = decks.get("joint", {})
    pair = joint.get("lens")
    rows = joint.get("summary", [])
    if target["target_id"] == "craftax" and target["rows"]:
        latest = target["rows"][-1]["metrics"]
        target["viewport"] = [
            {"line": line} for line in latest.get("local_map", [])[:9]
        ]
        inventory = latest.get("inventory")
        target["inventory"] = (
            [{"item": k, "count": v} for k, v in inventory.items()]
            if isinstance(inventory, dict)
            else []
        )
        if target["viewport"]:
            panels.append(
                table(
                    "Latest Craftax viewport", "target", "viewport", [("Map", "line")]
                )
            )
        if target["inventory"]:
            panels.append(
                table(
                    "Latest inventory",
                    "target",
                    "inventory",
                    [("Item", "item"), ("Count", "count")],
                )
            )
    if pair == "go-ex__craftax":
        joint["milestones"] = [
            dict(r, prerequisites=", ".join(r["requires"]))
            for r in rows
            if "requires" in r
        ]
        joint["forks"] = [
            dict(
                r,
                gained_names=", ".join(r["gained"]) or "—",
                checkpoint=f"{r['metric'].split(' @ ')[0][:8]} @ {r['metric'].split(' @ ')[-1]}",
                base_hp=(r.get("base_world") or {}).get("vitals", {}).get("health"),
                child_hp=r["child_world"].get("vitals", {}).get("health"),
            )
            for r in rows
            if "child_rollout" in r
        ]
        panels.extend(
            [
                table(
                    "Theme prerequisites",
                    "joint",
                    "milestones",
                    [
                        ("Milestone", "metric"),
                        ("Requires", "prerequisites"),
                        ("Theme", "value"),
                    ],
                ),
                table(
                    "Paired checkpoint forks",
                    "joint",
                    "forks",
                    [
                        ("Parent @ call", "checkpoint"),
                        ("Gain", "gained_names"),
                        ("Δ", "value"),
                        ("P0 HP", "base_hp"),
                        ("P1 HP", "child_hp"),
                    ],
                ),
            ]
        )
    elif pair == "gepa__healthbench2":
        joint["penalties"] = [r for r in rows if r["metric"].startswith("New penalty")]
        axes = {(r["policy"], r["metric"]): r["value"] for r in rows if "task" not in r}
        joint["clinical_points"] = [
            dict(
                policy=p,
                referral=axes[p, "emergency_referrals"],
                context=axes[p, "context_seeking"],
            )
            for p in sorted({p for p, _ in axes})
            if (p, "emergency_referrals") in axes and (p, "context_seeking") in axes
        ]
        if joint["clinical_points"]:
            panels.append(
                plot(
                    "scatter",
                    "Referral × context seeking",
                    "joint",
                    "clinical_points",
                    x="referral",
                    y="context",
                    label="policy",
                )
            )
        panels.append(
            table(
                "New validation penalties",
                "joint",
                "penalties",
                [("Criterion", "metric"), ("Task", "task"), ("Δ", "value")],
            )
        )
    elif pair == "mipro__banking77":
        panels.append(
            table(
                "Demo coverage by confusion",
                "joint",
                "summary",
                [
                    ("Confusion", "metric"),
                    ("Pack", "pack"),
                    ("Gold / predicted", "value"),
                    ("Errors", "errors"),
                ],
            )
        )
    elif pair == "gepa__craftax":
        joint["specialists"] = [
            r
            for r in rows
            if r["metric"]
            in {
                goal
                for row in target["rows"]
                for goal in row["metrics"].get("achievement_names", CLASSIC)
            }
        ]
        if joint["specialists"]:
            panels.append(
                plot(
                    "heatmap",
                    "Achievement specialists",
                    "joint",
                    "specialists",
                    row="policy",
                    column="metric",
                    value="value",
                )
            )
    return panels
