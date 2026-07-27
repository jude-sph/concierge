import json

import pytest

from rtvoice.concierge import Concierge, SpeechAct, validate_act
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


# --- Concierge._complete / respond: the unparseable-reply fallback ----------
#
# Empirically, against a real 7B model, six consecutive `user_turn` calls all
# returned bare text ("On it.", "Sure, what else?") instead of the required
# JSON envelope -- every one failed json.loads and the turn was lost to a
# JSONDecodeError. These tests fake the HTTP layer to reproduce that body
# verbatim and check the defensive fallback in concierge.py, not just the
# prompt wording (a prompt rule is not a guarantee).


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._content}}]}


def _concierge_replying(content: str) -> Concierge:
    """A Concierge whose HTTP POST always returns `content` verbatim as the
    model's raw completion body -- the only thing faked, same technique used
    for LlmReasoner in test_llm_reasoner.py."""
    c = Concierge()

    async def fake_post(url, **kw):
        return _FakeResponse(content)

    c._client.post = fake_post
    return c


@pytest.mark.asyncio
async def test_bare_text_reply_becomes_a_safe_spoken_act_not_a_crash():
    """The live-session failure this fix targets: a bare sentence instead of
    the JSON envelope must not raise JSONDecodeError out of respond() and
    lose the turn -- it must surface as a safe spoken act, and it must be
    counted as a violation so the regression is visible in the metrics."""
    c = _concierge_replying("On it.")
    act = await c.respond(TaskRegistry(), [], trigger="user_turn")
    assert act.act in ("chat", "acknowledge")
    assert act.text == "On it."
    assert c.violations == 1


@pytest.mark.asyncio
async def test_bare_text_reply_can_never_become_a_relay():
    """An unparsed reply carries no citation and no verified verbatim span,
    so it must never be used to state a task fact -- even when the bare text
    happens to look exactly like a citation and its verbatim span."""
    act = await _concierge_replying("t1: renamed 47 contacts").respond(
        registry_with_done_task(), [], trigger="reasoner_update"
    )
    assert act.act != "relay"


@pytest.mark.asyncio
async def test_malformed_json_is_not_swallowed_as_if_it_were_prose():
    """Truncated/garbled JSON (still containing brace punctuation) is a
    different failure from bare spoken text and must not be salvaged -- only
    genuine bare-text replies are. Re-prompting repeats the same malformed
    body here, so the error still propagates out of respond()."""
    c = _concierge_replying('{"act": "chat", "text": "On it.')
    with pytest.raises(json.JSONDecodeError):
        await c.respond(TaskRegistry(), [], trigger="user_turn")
