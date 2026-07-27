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


def test_relay_citing_pending_task_requires_fact():
    """A relay citing a task that is only ack'd (PENDING) has no fact to relay yet."""
    r = TaskRegistry()
    r.apply(ReasonerMessage(kind="ack", task_id="t1", understood_as="rename contacts"))
    # t1 is pending; no done/confirm_required/failed message
    ok, err = validate_act(
        SpeechAct(act="relay", text="Renaming your contacts now", cites="t1"),
        r,
    )
    assert not ok
    assert "fact recorded yet" in err or "no fact" in err


def test_bare_placeholder_text_is_rejected():
    """The live-session failure: the model copied the '{"text": "..."}'
    template slot straight out of the format spec, and it was spoken aloud
    verbatim as literal dots. A bare placeholder must never pass validation,
    for any act."""
    ok, err = validate_act(SpeechAct(act="acknowledge", text="..."), TaskRegistry())
    assert not ok
    assert "placeholder" in err


def test_placeholder_variants_are_rejected():
    for text in ["...", "…", "....", "[...]", "<...>", "  ...  "]:
        ok, _ = validate_act(SpeechAct(act="chat", text=text), TaskRegistry())
        assert not ok, f"{text!r} should have been rejected as a placeholder"


def test_a_real_sentence_that_happens_to_end_in_an_ellipsis_is_fine():
    """Only a BARE placeholder is rejected -- an ellipsis used as ordinary
    punctuation inside a real sentence must not be caught by the same net."""
    ok, _ = validate_act(SpeechAct(act="chat", text="hold on, let me check..."),
                         TaskRegistry())
    assert ok


def test_relay_citing_running_task_requires_fact():
    """A relay citing a task that is only progress'd (RUNNING) has no fact to relay yet."""
    r = TaskRegistry()
    r.apply(ReasonerMessage(kind="ack", task_id="t1", understood_as="rename contacts"))
    r.apply(ReasonerMessage(kind="progress", task_id="t1", update="5 renamed so far"))
    # t1 is running; no done/confirm_required/failed message
    ok, err = validate_act(
        SpeechAct(act="relay", text="Renaming your contacts now", cites="t1"),
        r,
    )
    assert not ok
    assert "fact recorded yet" in err or "no fact" in err
