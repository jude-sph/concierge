import json
import pytest
from rtvoice.device import DeviceState
from rtvoice.protocol import OrchestratorMessage
from rtvoice.reasoner_stub import ReasonerStub


@pytest.fixture
def stub(tmp_path):
    state = tmp_path / "device_state.json"
    state.write_text(json.dumps({
        "contacts": [{"id": 1, "first_name": "Sarah", "group": "work"},
                     {"id": 2, "first_name": "Marcus", "group": "work"}]
    }))
    return ReasonerStub(DeviceState(state, tmp_path / "j.jsonl"), latency_ms=0)


@pytest.mark.asyncio
async def test_chat_returns_noop(stub):
    out = await stub.handle(OrchestratorMessage(kind="utterance", text="hello there"))
    assert [m.kind for m in out] == ["noop"]


@pytest.mark.asyncio
async def test_destructive_command_requires_confirmation_before_writing(stub):
    out = await stub.handle(
        OrchestratorMessage(kind="utterance", text="set all my contacts to Hans")
    )
    kinds = [m.kind for m in out]
    assert "ack" in kinds
    assert "confirm_required" in kinds
    # nothing written yet
    assert [c["first_name"] for c in stub.device.query("contacts")] == ["Sarah", "Marcus"]


@pytest.mark.asyncio
async def test_confirmation_commits_the_write(stub):
    out = await stub.handle(
        OrchestratorMessage(kind="utterance", text="set all my contacts to Hans")
    )
    tid = next(m.task_id for m in out if m.kind == "confirm_required")
    done = await stub.handle(
        OrchestratorMessage(kind="clarification_answer", task_id=tid, text="yes do it")
    )
    assert any(m.kind == "done" for m in done)
    assert [c["first_name"] for c in stub.device.query("contacts")] == ["Hans", "Hans"]


@pytest.mark.asyncio
async def test_cancel_before_confirm_leaves_state_untouched(stub):
    out = await stub.handle(
        OrchestratorMessage(kind="utterance", text="set all my contacts to Hans")
    )
    tid = next(m.task_id for m in out if m.kind == "confirm_required")
    await stub.handle(OrchestratorMessage(kind="cancel", task_id=tid))
    assert [c["first_name"] for c in stub.device.query("contacts")] == ["Sarah", "Marcus"]


@pytest.mark.asyncio
async def test_compound_command_creates_multiple_tasks(stub):
    out = await stub.handle(OrchestratorMessage(
        kind="utterance",
        text="find chinese restaurants in soho and set all my contacts to Hans",
    ))
    acks = [m for m in out if m.kind == "ack"]
    assert len(acks) == 2
    assert len({m.task_id for m in acks}) == 2


@pytest.mark.asyncio
async def test_rejection_with_dont_do_it(stub):
    """Negation-blind regex would match 'do it' in 'don't do it'"""
    out = await stub.handle(
        OrchestratorMessage(kind="utterance", text="set all my contacts to Hans")
    )
    tid = next(m.task_id for m in out if m.kind == "confirm_required")
    result = await stub.handle(
        OrchestratorMessage(kind="clarification_answer", task_id=tid, text="don't do it")
    )
    assert any(m.kind == "failed" for m in result)
    assert [c["first_name"] for c in stub.device.query("contacts")] == ["Sarah", "Marcus"]


@pytest.mark.asyncio
async def test_rejection_with_please_dont_confirm(stub):
    """Negation-blind regex would match 'confirm' in 'please don't confirm this'"""
    out = await stub.handle(
        OrchestratorMessage(kind="utterance", text="set all my contacts to Hans")
    )
    tid = next(m.task_id for m in out if m.kind == "confirm_required")
    result = await stub.handle(
        OrchestratorMessage(kind="clarification_answer", task_id=tid, text="please don't confirm this")
    )
    assert any(m.kind == "failed" for m in result)
    assert [c["first_name"] for c in stub.device.query("contacts")] == ["Sarah", "Marcus"]


@pytest.mark.asyncio
async def test_rejection_with_not_okay(stub):
    """Negation-blind regex would match 'okay' in 'that is not okay'"""
    out = await stub.handle(
        OrchestratorMessage(kind="utterance", text="set all my contacts to Hans")
    )
    tid = next(m.task_id for m in out if m.kind == "confirm_required")
    result = await stub.handle(
        OrchestratorMessage(kind="clarification_answer", task_id=tid, text="that is not okay")
    )
    assert any(m.kind == "failed" for m in result)
    assert [c["first_name"] for c in stub.device.query("contacts")] == ["Sarah", "Marcus"]


@pytest.mark.asyncio
async def test_rejection_with_no_dont_confirm(stub):
    """Negation-blind regex would match 'confirm' in 'no, don't confirm it'"""
    out = await stub.handle(
        OrchestratorMessage(kind="utterance", text="set all my contacts to Hans")
    )
    tid = next(m.task_id for m in out if m.kind == "confirm_required")
    result = await stub.handle(
        OrchestratorMessage(kind="clarification_answer", task_id=tid, text="no, don't confirm it")
    )
    assert any(m.kind == "failed" for m in result)
    assert [c["first_name"] for c in stub.device.query("contacts")] == ["Sarah", "Marcus"]


@pytest.mark.asyncio
async def test_rejection_with_bare_no(stub):
    """Bare 'no' should reject"""
    out = await stub.handle(
        OrchestratorMessage(kind="utterance", text="set all my contacts to Hans")
    )
    tid = next(m.task_id for m in out if m.kind == "confirm_required")
    result = await stub.handle(
        OrchestratorMessage(kind="clarification_answer", task_id=tid, text="no")
    )
    assert any(m.kind == "failed" for m in result)
    assert [c["first_name"] for c in stub.device.query("contacts")] == ["Sarah", "Marcus"]


@pytest.mark.asyncio
async def test_rejection_with_ambiguous_hmm(stub):
    """Ambiguous input should reject"""
    out = await stub.handle(
        OrchestratorMessage(kind="utterance", text="set all my contacts to Hans")
    )
    tid = next(m.task_id for m in out if m.kind == "confirm_required")
    result = await stub.handle(
        OrchestratorMessage(kind="clarification_answer", task_id=tid, text="hmm")
    )
    assert any(m.kind == "failed" for m in result)
    assert [c["first_name"] for c in stub.device.query("contacts")] == ["Sarah", "Marcus"]


# --- Final-review regressions ------------------------------------------------
#
# CRITICAL 1(a): the confirmation gate must first decide the utterance IS
# answer-shaped -- short, and essentially only a yes/no rather than a sentence
# carrying its own new request. The negation-first/default-deny affirmative
# test applies only after that. Anything not answer-shaped is NOT an answer,
# so it must neither commit the write nor cancel the task: the question stays
# open.


async def arm(stub):
    out = await stub.handle(
        OrchestratorMessage(kind="utterance", text="set all my contacts to Hans")
    )
    return next(m.task_id for m in out if m.kind == "confirm_required")


@pytest.mark.parametrize("text", [
    "okay so what's on my calendar tomorrow",
    "yes I was talking to my colleague, ignore that",
    "sorry, I was on the phone, ok anyway",
    "set all my contacts to Hans okay and find chinese restaurants in soho",
    "yeah I'll call him back later about the restaurant booking",
])
@pytest.mark.asyncio
async def test_non_answer_shaped_reply_neither_commits_nor_cancels(stub, text):
    tid = await arm(stub)
    result = await stub.handle(
        OrchestratorMessage(kind="clarification_answer", task_id=tid, text=text))

    assert [m.kind for m in result] == ["noop"]
    assert [c["first_name"] for c in stub.device.query("contacts")] == ["Sarah", "Marcus"]

    # the question is still open, so a real answer still works
    done = await stub.handle(
        OrchestratorMessage(kind="clarification_answer", task_id=tid, text="yes"))
    assert any(m.kind == "done" for m in done)


@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
@pytest.mark.asyncio
async def test_empty_answer_is_never_an_answer(stub, blank):
    """IMPORTANT 5: an empty transcript reached _confirm and default-denied,
    silently cancelling a destructive task the user never spoke about."""
    tid = await arm(stub)
    result = await stub.handle(
        OrchestratorMessage(kind="clarification_answer", task_id=tid, text=blank))

    assert [m.kind for m in result] == ["noop"]
    assert [c["first_name"] for c in stub.device.query("contacts")] == ["Sarah", "Marcus"]

    done = await stub.handle(
        OrchestratorMessage(kind="clarification_answer", task_id=tid, text="yes"))
    assert any(m.kind == "done" for m in done)


@pytest.mark.parametrize("word", ["yes", "yeah", "yep", "ok", "okay", "sure", "right"])
@pytest.mark.asyncio
async def test_conversational_affirmatives_all_confirm(stub, word):
    """IMPORTANT 6: states.py treats "sure"/"right" as backchannels (so the
    policy promotes them to answers when a question is pending) but the
    reasoner's affirmative set did not accept them, so answering "sure"
    produced FAILED "cancelled by user"."""
    tid = await arm(stub)
    result = await stub.handle(
        OrchestratorMessage(kind="clarification_answer", task_id=tid, text=word))
    assert any(m.kind == "done" for m in result), [m.kind for m in result]
    assert [c["first_name"] for c in stub.device.query("contacts")] == ["Hans", "Hans"]


@pytest.mark.parametrize("word", ["not sure", "no, sure thing", "that's not right"])
@pytest.mark.asyncio
async def test_negation_still_beats_the_widened_affirmative_set(stub, word):
    """Widening the affirmative vocabulary must not weaken negation-first
    default-deny."""
    tid = await arm(stub)
    result = await stub.handle(
        OrchestratorMessage(kind="clarification_answer", task_id=tid, text=word))
    assert not any(m.kind == "done" for m in result)
    assert [c["first_name"] for c in stub.device.query("contacts")] == ["Sarah", "Marcus"]
