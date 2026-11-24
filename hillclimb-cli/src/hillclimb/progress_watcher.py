"""A replayable progress correspondent for an optimizer run."""

from .event_consumers import Consumer
from .limit_forecast import FORECAST, OBSERVATIONS

WATCH = "progress.update"
NARRATION = "progress.narration"


def milestone(ob):
    if ob["status"] != "running":
        if ob["status"] == "completed" and ob["campaign"] == "gepa":
            return "champion evaluated"
        return ob["status"]
    if ob["campaign"] == "seed":
        return "seed held-out" if ob["phase"] == "heldout" else "seed baseline"
    if ob["phase"] == "seed_validation":
        return "seed validation"
    if ob["phase"] in ("freeze", "final_test", "complete"):
        return "champion held-out"
    return "GEPA search"


def describe(ob, phase, previous, forecast):
    sentences = []
    if phase == "seed validation":
        scored = min(ob["rollouts"], ob["validation_rows"])
        sentences.append(f"Scoring the starting policy. Validation is {scored}/{ob['validation_rows']} complete.")
    elif phase == "GEPA search":
        step = ob["phase"].replace("_", " ")
        sentences.append(f"GEPA is at {step} with {ob['rounds']}/{ob['max_rounds']} rounds complete.")
        if ob["recent_decision"] and (
            previous is None or ob["rounds"] != previous["rounds"]
        ):
            decision = ob["recent_decision"].replace("_", " ")
            sentences.append(f"Round {ob['rounds']} ended with {decision}.")
        if ob["champion_correct"] is not None:
            best, total = ob["champion_correct"], ob["validation_rows"]
            seed = ob["seed_correct"]
            comparison = f" from {seed}/{total} at the start" if seed is not None and best != seed else ""
            sentences.append(f"Best validation score is {best}/{total}{comparison}.")
    elif phase == "champion held-out":
        sentences.append(f"Search has ended. The frozen champion has finished {ob['heldout']}/{ob['heldout_rows']} held-out examples.")
    elif phase == "champion evaluated":
        sentences.append("The champion finished its held-out pass. The seed baseline is next.")
    elif phase == "seed baseline":
        scored = min(ob["rollouts"], ob["validation_rows"])
        sentences.append(f"Scoring the seed baseline. Validation is {scored}/{ob['validation_rows']} complete.")
    elif phase == "seed held-out":
        sentences.append(f"The seed baseline has finished {ob['heldout']}/{ob['heldout_rows']} held-out examples.")
    elif phase == "completed":
        sentences.append("Both held-out passes are complete. The final report will include the paired results.")
    else:
        sentences.append(f"The {ob['campaign'].upper()} campaign is {phase.replace('_', ' ')}.")
    queue = ob.get("queue") or {}
    if phase in ("GEPA search", "champion held-out", "seed held-out") and queue.get("in_flight"):
        sentences.append(f"{queue['in_flight']} rollout requests are in flight.")
    if (phase in ("GEPA search", "seed held-out") and forecast
            and forecast["campaign"] == ob["campaign"]
            and forecast.get("observed_at_ms", 0) >= ob["observed_at_ms"] - 3500
            and (estimate := forecast.get("finish" if phase == "seed held-out" else "search_stop"))
            and estimate["seconds"] > 0):
        prefix = "Finish" if phase == "seed held-out" else "Search stop"
        low, high = round(estimate["low"]), round(estimate["high"])
        window = f"{low}–{high} seconds" if low != high else f"{low} seconds"
        sentences.append(f"{prefix} estimate is {window}.")
    return " ".join(sentences)


def watcher(interval_seconds: int) -> Consumer:
    if interval_seconds < 1:
        raise ValueError("The progress watch interval must be positive")

    def reduce(state, event):
        ob = event["payload"]
        if event["producer"] == FORECAST:
            return {**state, "forecast": ob}, None
        phase = milestone(ob)
        previous = state.get("reported")
        elapsed = ob["clock_ms"] - state.get("last_ms", 0)
        if previous is not None and phase == state["phase"]:
            if ob["status"] != "running" or elapsed < interval_seconds * 1000:
                return state, None
        text = describe(ob, phase, previous, state.get("forecast"))
        return {**state, "last_ms": ob["clock_ms"], "reported": {"rounds": ob["rounds"]}, "phase": phase}, {
            "text": text, "campaign": ob["campaign"], "phase": phase,
            "status": ob["status"], "observed_at_ms": ob["observed_at_ms"],
        }

    return Consumer(WATCH, (OBSERVATIONS, FORECAST), "hillclimb.progress-update.v1", reduce)
