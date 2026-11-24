"""Phase-aware limit and queue reducers shared by Hillclimb event consumers."""

from statistics import median

from .event_consumers import Consumer

OBSERVATIONS = "run.observations"
DAEMON_EVENTS = "run.events"
LIMIT_PROGRESS = "limit.progress"
FORECAST = "limit.forecast"
QUEUE = "queue.progress"

DONE_SEARCH = {"freeze", "final_test", "complete"}
PANEL_PHASES = {"seed_validation", "child_validation", "final_test", "validation", "heldout"}


def track_limits(state, event):
    if event["producer"] == DAEMON_EVENTS:
        # The runner's observations carry the actual counters associated with this cursor.
        # Raw lifecycle events provide an early pause/stop signal while the next poll catches up.
        raw = event["payload"]
        previous = state.get("last")
        if (previous and raw["campaign"] == previous["campaign"]
                and raw["daemon_seq"] > previous["daemon_seq"]
                and raw.get("status") in ("paused", "cancelled", "in_doubt", "completed")):
            observation = {**previous, "status": raw["status"], "daemon_seq": raw["daemon_seq"]}
            return {**state, "last": observation}, observation
        return state, None
    observation = event["payload"]
    previous = state.get("last")
    active = state.get("active_ms", 0)
    if previous and previous["campaign"] == observation["campaign"]:
        if previous["status"] == "running":
            active += max(0, observation["elapsed_ms"] - previous["elapsed_ms"])
    else:
        active = 0
    observation = {**observation, "active_ms": active}
    return {"last": observation, "active_ms": active}, observation


def estimate(seconds, samples, *, exact=False):
    if exact:
        return {"seconds": seconds, "low": seconds, "high": seconds,
                "confidence": "exact", "samples": samples}
    return {"seconds": seconds, "low": seconds * 0.75, "high": seconds * 1.5,
            "confidence": "medium" if samples >= 5 else "low", "samples": samples}


def observed_rate(samples, key):
    changes = []
    for earlier, later in zip(samples, samples[1:]):
        elapsed = (later["active_ms"] - earlier["active_ms"]) / 1000
        change = later[key] - earlier[key]
        if elapsed > 0 and change > 0:
            changes.append((earlier, later))
    if len(changes) < 2:
        return None, len(changes)
    first, _ = changes[0]
    _, last = changes[-1]
    seconds = (last["active_ms"] - first["active_ms"]) / 1000
    return (last[key] - first[key]) / seconds, len(changes)


def counter_forecast(samples, key, remaining):
    if remaining <= 0:
        return estimate(0, 0, exact=True)
    rate, count = observed_rate(samples, key)
    return estimate(remaining / rate, count) if rate else None


def track_forecast(state, event):
    ob = event["payload"]
    previous = state.get("latest")
    new_campaign = previous is None or previous["campaign"] != ob["campaign"]
    recent = [] if new_campaign else state["recent"]
    if previous is None or ob != previous:
        recent = [*recent[-63:], ob]
    first = ob if new_campaign else state["first"]
    round_durations = [] if new_campaign else state["round_durations"]
    round_anchor = ob["active_ms"] if new_campaign else state["round_anchor"]
    if previous and not new_campaign:
        if previous["phase"] == "seed_validation" and ob["phase"] == "parent_minibatch":
            round_anchor = ob["active_ms"]
        elif ob["rounds"] > previous["rounds"]:
            duration = (ob["active_ms"] - round_anchor) / (1000 * (ob["rounds"] - previous["rounds"]))
            if duration > 0:
                round_durations = [*round_durations[-7:], duration]
            round_anchor = ob["active_ms"]
    round_remaining = None
    if round_durations:
        round_remaining = max(0, median(round_durations) - (ob["active_ms"] - round_anchor) / 1000)
    completed = (ob["campaign"] == "seed" and ob["status"] == "completed")
    searching = ob["campaign"] == "gepa" and ob["phase"] not in DONE_SEARCH and ob["status"] != "completed"
    forecasts = {}
    finish = None
    if ob["status"] == "running":
        forecasts["deadline"] = estimate(max(0, ob["deadline_ms"] - ob["elapsed_ms"]) / 1000, 0, exact=True)
        for name, key, ceiling, used in (
            ("rounds", "rounds", ob["max_rounds"], ob["rounds"]),
            ("search rollouts", "search_rollouts", ob["max_search_rollouts"], ob["search_rollouts"]),
            ("budget", "spent_micros", ob["cap_micros"], ob["spent_micros"] + ob["reserved_micros"] + ob["in_doubt_micros"]),
            ("rollout requests", "rollout_requests", ob["max_rollout_requests"], ob["rollout_requests"]),
            ("proposer requests", "proposer_requests", ob["max_proposer_requests"], ob["proposer_requests"]),
        ):
            if ceiling is None or (not searching and name in ("rounds", "search rollouts", "proposer requests")):
                continue
            # A new GEPA round needs room for the parent and child minibatch.
            remaining = ceiling - used
            if name == "rounds" and round_remaining is not None and remaining > 0:
                forecast = estimate(round_remaining + (remaining - 1) * median(round_durations), len(round_durations))
            elif name == "search rollouts" and 0 <= remaining < ob["next_round_min_rollouts"]:
                forecast = estimate(round_remaining, len(round_durations)) if round_remaining is not None else None
            else:
                if name == "search rollouts" and remaining > 0:
                    remaining -= ob["next_round_min_rollouts"] - 1
                forecast = counter_forecast(recent, key, remaining)
            if forecast is not None:
                forecasts[name] = forecast
        if ob["phase"] in ("final_test", "heldout"):
            samples = [item for item in recent if item["phase"] == ob["phase"]]
            remaining = ob["heldout_rows"] - ob["heldout"]
            if ob["campaign"] == "gepa":
                remaining += ob["validation_rows"] + ob["heldout_rows"]
            finish = counter_forecast(samples, "heldout", remaining)
        elif searching:
            # Estimate the finishing stretch only with a pace observed on validation panels.
            samples = [item for item in recent if item["phase"] in PANEL_PHASES
                       and item["phase"] == first["phase"]]
            validation_rate, count = observed_rate(samples, "rollouts")
            candidate = min(forecasts.values(), key=lambda item: item["seconds"], default=None)
            if validation_rate and candidate and candidate["confidence"] != "exact":
                finish = estimate(candidate["seconds"] +
                                  (2 * ob["heldout_rows"] + ob["validation_rows"]) / validation_rate, count)
    nearest = min(forecasts, key=lambda name: forecasts[name]["seconds"], default=None)
    search_stop = forecasts[nearest] if searching and nearest and len(forecasts) > 1 else None
    if not searching and ob["status"] in ("running", "completed"):
        search_stop = estimate(0, 0, exact=True)
    if completed:
        finish = estimate(0, 0, exact=True)
    payload = {"campaign": ob["campaign"], "phase": ob["phase"], "status": ob["status"],
               "observed_at_ms": ob["observed_at_ms"], "daemon_seq": ob["daemon_seq"],
               "binding_limit": nearest, "limits": forecasts, "search_stop": search_stop, "finish": finish}
    return {"recent": recent, "latest": ob, "first": first,
            "round_durations": round_durations, "round_anchor": round_anchor}, payload


def track_queue(state, event):
    ob = event["payload"]
    recent = [item for item in state.get("recent", []) if item[0] >= ob["clock_ms"] - 30_000]
    recent.append([ob["clock_ms"], ob["total_rollouts"]])
    elapsed = (recent[-1][0] - recent[0][0]) / 1000
    rate = (recent[-1][1] - recent[0][1]) / elapsed if elapsed > 0 else None
    return {"recent": recent}, {"campaign": ob["campaign"], "daemon_seq": ob["daemon_seq"],
                                 "observed_at_ms": ob["observed_at_ms"], "queue": ob["queue"],
                                 "rollouts_per_second": rate}


CONSUMERS = (
    Consumer(LIMIT_PROGRESS, (OBSERVATIONS, DAEMON_EVENTS), "hillclimb.limit-progress.v1", track_limits),
    Consumer(FORECAST, (LIMIT_PROGRESS,), "hillclimb.limit-forecast.v1", track_forecast),
    Consumer(QUEUE, (OBSERVATIONS,), "hillclimb.queue-progress.v1", track_queue),
)
