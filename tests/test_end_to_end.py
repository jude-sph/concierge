import json

import pytest
from fakes import FakeConcierge, FakeVoice

from rtvoice.device import DeviceState
from rtvoice.events import EventLog
from rtvoice.orchestrator import Orchestrator
from rtvoice.reasoner_stub import ReasonerStub
from rtvoice.registry import TaskStatus
from rtvoice.states import TurnEvent, UserState


@pytest.fixture
def orch(tmp_path):
    state = tmp_path / "device_state.json"
    state.write_text(json.dumps({
        "contacts": [{"id": 1, "first_name": "Sarah", "group": "work"},
                     {"id": 2, "first_name": "Marcus", "group": "work"}]
    }))
    device = DeviceState(state, tmp_path / "journal.jsonl")
    return Orchestrator(
        reasoner=ReasonerStub(device, latency_ms=0),
        concierge=FakeConcierge(),
        voice=FakeVoice(),
        log=EventLog(tmp_path / "events.jsonl"),
        device=device,
    )


@pytest.mark.asyncio
async def test_incomplete_does_not_dispatch(orch):
    await orch.on_turn_event(TurnEvent(UserState.INCOMPLETE, "find food and", 0))
    assert orch.registry.live_ids() == []


@pytest.mark.asyncio
async def test_compound_command_creates_two_tasks(orch):
    await orch.on_turn_event(TurnEvent(
        UserState.COMPLETE,
        "find chinese restaurants in soho and set all my contacts to Hans", 0))
    assert len(orch.registry.all()) == 2


@pytest.mark.asyncio
async def test_confirmed_destructive_command_mutates_device_state(orch):
    await orch.on_turn_event(
        TurnEvent(UserState.COMPLETE, "set all my contacts to Hans", 0))
    tid = next(t.task_id for t in orch.registry.all()
               if t.status == TaskStatus.AWAITING_CONFIRM)
    await orch.on_turn_event(TurnEvent(UserState.COMPLETE, "yes do it", 1000))
    assert orch.registry.get(tid).status == TaskStatus.DONE
    assert [c["first_name"] for c in orch.device.query("contacts")] == ["Hans", "Hans"]


@pytest.mark.asyncio
async def test_barge_in_stops_speech(orch):
    orch.policy_state.speaking = True
    await orch.on_turn_event(TurnEvent(UserState.NONIDLE, "wait", 0))
    assert orch.voice.stops == 1


@pytest.mark.asyncio
async def test_backchannel_does_not_stop_speech(orch):
    orch.policy_state.speaking = True
    await orch.on_turn_event(TurnEvent(UserState.BACKCHANNEL, "mm hm", 0))
    assert orch.voice.stops == 0


@pytest.mark.asyncio
async def test_every_turn_is_logged(orch):
    await orch.on_turn_event(TurnEvent(UserState.COMPLETE, "hello", 0))
    kinds = [e.kind for e in EventLog.read(orch.log.path)]
    assert "turn_event" in kinds
    assert "concierge_act" in kinds


@pytest.mark.asyncio
async def test_bare_yes_backchannel_while_speaking_confirms_destructive_write(orch):
    """Highest-risk turn-policy combination (flagged in Task 7's review, carried
    forward to Task 12): the user answers a destructive-write confirmation with
    a bare "yes" WHILE the system is still speaking the confirmation prompt.
    The adapter classifies "yes" as a backchannel by wording alone; the policy
    layer must recognise it as the answer to the pending question (because
    policy_state.pending_question is set) rather than silently dropping it as
    filler. If this regresses, a destructive write is confirmed by the user but
    never actually committed.
    """
    await orch.on_turn_event(
        TurnEvent(UserState.COMPLETE, "set all my contacts to Hans", 0))
    tid = next(t.task_id for t in orch.registry.all()
               if t.status == TaskStatus.AWAITING_CONFIRM)
    assert orch.policy_state.pending_question is not None

    # The system is still voicing "This will rename 2 contacts to Hans. Confirm?"
    orch.policy_state.speaking = True

    await orch.on_turn_event(TurnEvent(UserState.BACKCHANNEL, "yes", 1000))

    assert orch.registry.get(tid).status == TaskStatus.DONE
    assert [c["first_name"] for c in orch.device.query("contacts")] == ["Hans", "Hans"]
