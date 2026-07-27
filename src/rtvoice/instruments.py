"""Metrics derived from an event log. Pure functions over Event lists, so they
work identically on a live session and a replayed one.
"""
from __future__ import annotations

import statistics

from .events import Event


def latency_report(events: list[Event]) -> dict:
    """Time from a dispatching turn (user_complete) to the concierge's reply."""
    gaps: list[int] = []
    awaiting: int | None = None
    for e in events:
        if e.kind == "turn_event" and e.data.get("state") == "user_complete":
            awaiting = e.t_ms
        elif e.kind == "concierge_act" and awaiting is not None:
            gaps.append(e.t_ms - awaiting)
            awaiting = None

    if not gaps:
        return {"n": 0, "mean_ms": 0, "p50_ms": 0, "p95_ms": 0, "max_ms": 0}

    gaps.sort()
    pick = lambda q: gaps[min(int(len(gaps) * q), len(gaps) - 1)]
    return {
        "n": len(gaps),
        "mean_ms": int(statistics.mean(gaps)),
        "p50_ms": pick(0.50),
        "p95_ms": pick(0.95),
        "max_ms": gaps[-1],
    }


def violation_rate(events: list[Event]) -> float:
    """Mean invalid speech-act generations per emitted concierge act.

    `concierge_act.data["violations"]` is *not* a per-act flag. It is the
    Concierge's session-lifetime counter of invalid generations (Concierge.
    violations, see concierge.py), snapshotted onto every act it emits. That
    counter is cumulative and monotonically non-decreasing across the
    session, and a single turn can bump it by up to 2 (a failed first
    attempt, a failed re-prompt, then a fallback acknowledge).

    Because the counter is cumulative, the last (equivalently, since it is
    non-decreasing, the max) value observed across the acts *is* the total
    count of invalid generations produced during the whole session, no
    summing required — summing the per-event snapshots would wildly
    over-count. Dividing that total by the number of acts gives the mean
    number of invalid generations behind each act the concierge actually
    emitted: a measure of how much correction work the session needed, not
    a bounded pass/fail fraction. Because up to 2 violations can precede a
    single act, this value can exceed 1.0 in a session with heavy
    correction traffic — that is expected, not a bug.

    Use `max()` rather than indexing the last element so this stays correct
    even if a caller hands in events out of strict append order.
    """
    acts = [e for e in events if e.kind == "concierge_act"]
    if not acts:
        return 0.0
    return max(e.data.get("violations", 0) for e in acts) / len(acts)
