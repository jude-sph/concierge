import pytest

from rtvoice.events import Event
from rtvoice.instruments import latency_report, violation_rate


def ev(kind, t_ms, **data):
    return Event(kind=kind, t_ms=t_ms, data=data)


def test_latency_measures_turn_to_first_speech():
    events = [
        ev("turn_event", 1000, state="user_complete", transcript="hi"),
        ev("concierge_act", 1400, act="acknowledge", text="on it"),
        ev("turn_event", 5000, state="user_complete", transcript="again"),
        ev("concierge_act", 5600, act="acknowledge", text="sure"),
    ]
    r = latency_report(events)
    assert r["n"] == 2
    assert r["mean_ms"] == 500
    assert r["max_ms"] == 600


def test_latency_ignores_non_dispatching_turns():
    events = [
        ev("turn_event", 0, state="user_incomplete", transcript="and"),
        ev("turn_event", 1000, state="user_complete", transcript="hi"),
        ev("concierge_act", 1200, act="acknowledge"),
    ]
    assert latency_report(events)["n"] == 1


def test_empty_log_reports_zero_not_a_crash():
    assert latency_report([])["n"] == 0


def test_violation_rate_is_violations_over_acts():
    events = [
        ev("concierge_act", 0, act="acknowledge", violations=0),
        ev("concierge_act", 1, act="relay", violations=1),
        ev("concierge_act", 2, act="chat", violations=1),
    ]
    assert violation_rate(events) == pytest.approx(1 / 3)
