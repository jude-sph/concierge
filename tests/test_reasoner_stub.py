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
