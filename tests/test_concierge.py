from rtvoice.concierge import SpeechAct, validate_act
from rtvoice.protocol import ReasonerMessage
from rtvoice.registry import TaskRegistry


def registry_with_done_task():
    r = TaskRegistry()
    r.apply(ReasonerMessage(kind="ack", task_id="t1", understood_as="rename contacts"))
    r.apply(ReasonerMessage(kind="done", task_id="t1", result="renamed 47 contacts"))
    return r


def test_acknowledge_needs_no_citation():
    ok, _ = validate_act(SpeechAct(act="acknowledge", text="on it"), TaskRegistry())
    assert ok


def test_chat_needs_no_citation():
    ok, _ = validate_act(SpeechAct(act="chat", text="sure, what else?"), TaskRegistry())
    assert ok


def test_relay_without_citation_is_rejected():
    ok, err = validate_act(
        SpeechAct(act="relay", text="I renamed all your contacts"), registry_with_done_task()
    )
    assert not ok
    assert "cites" in err


def test_relay_citing_unknown_task_is_rejected():
    ok, err = validate_act(
        SpeechAct(act="relay", text="done", cites="nope"), registry_with_done_task()
    )
    assert not ok
    assert "unknown" in err


def test_relay_missing_the_verbatim_span_is_rejected():
    """The whole point: the concierge may not restate a fact in its own words."""
    ok, err = validate_act(
        SpeechAct(act="relay", text="all done with your contacts!", cites="t1"),
        registry_with_done_task(),
    )
    assert not ok
    assert "verbatim" in err


def test_relay_containing_the_verbatim_span_is_accepted():
    ok, _ = validate_act(
        SpeechAct(act="relay", text="Okay — renamed 47 contacts. Anything else?", cites="t1"),
        registry_with_done_task(),
    )
    assert ok


def test_abort_requires_a_live_task():
    r = TaskRegistry()
    r.apply(ReasonerMessage(kind="ack", task_id="t1", understood_as="rename"))
    ok, _ = validate_act(SpeechAct(act="abort", cites="t1"), r)
    assert ok
    ok, err = validate_act(SpeechAct(act="abort", cites="ghost"), r)
    assert not ok
