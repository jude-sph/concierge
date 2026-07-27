import json
import pytest
from rtvoice.device import DeviceState
from rtvoice.events import EventLog
from rtvoice.orchestrator import Orchestrator
from rtvoice.protocol import ReasonerMessage
from rtvoice.reasoner_stub import ReasonerStub
from rtvoice.states import TurnEvent, UserState
from fakes import FakeConcierge, FakeVoice


def make(tmp_path, **kw):
    state = tmp_path / "device_state.json"
    state.write_text(json.dumps({"contacts": [{"id": 1, "first_name": "Sarah"}]}))
    device = DeviceState(state, tmp_path / "j.jsonl")
    return Orchestrator(
        reasoner=ReasonerStub(device, latency_ms=0),
        concierge=FakeConcierge(), voice=FakeVoice(),
        log=EventLog(tmp_path / "e.jsonl"), device=device, **kw,
    )


@pytest.mark.asyncio
async def test_silence_timeout_forces_a_turn(tmp_path):
    """If SoulX-Duplug never emits 'speak', the system must not hang forever."""
    orch = make(tmp_path, silence_timeout_ms=2000)
    # NOTE: deviates from the brief's literal text ("rename my contacts") --
    # that string never matches ReasonerStub's rename regex (it requires a
    # trailing "to <name>"), so the reasoner would always answer "noop" and
    # no task would ever be registered, regardless of orchestrator behaviour.
    # Using a command the stub can actually parse keeps the test's real
    # intent: a forced dispatch after silence must produce a registered task.
    await orch.on_turn_event(TurnEvent(UserState.INCOMPLETE, "rename my contacts to Hans", 0))
    await orch.on_tick(now_ms=1000)
    assert orch.registry.all() == []       # not yet
    await orch.on_tick(now_ms=2500)
    assert len(orch.registry.all()) >= 1   # forced dispatch


@pytest.mark.asyncio
async def test_completed_turn_clears_the_silence_timer(tmp_path):
    orch = make(tmp_path, silence_timeout_ms=2000)
    await orch.on_turn_event(TurnEvent(UserState.INCOMPLETE, "partial", 0))
    await orch.on_turn_event(TurnEvent(UserState.COMPLETE, "rename my contacts to Hans", 500))
    n = len(orch.registry.all())
    await orch.on_tick(now_ms=5000)
    assert len(orch.registry.all()) == n    # no duplicate dispatch


@pytest.mark.asyncio
async def test_hung_reasoner_produces_a_failed_task(tmp_path):
    class HungReasoner:
        async def handle(self, msg):
            import asyncio
            await asyncio.sleep(10)
            return []

    orch = make(tmp_path, reasoner_timeout_s=0.05)
    orch.reasoner = HungReasoner()
    await orch.on_turn_event(TurnEvent(UserState.COMPLETE, "do a thing", 0))
    kinds = [e.kind for e in EventLog.read(orch.log.path)]
    assert "reasoner_timeout" in kinds


@pytest.mark.asyncio
async def test_concierge_bypass_speaks_reasoner_text_directly(tmp_path):
    """With use_concierge=False, reasoner verbatim text goes straight to TTS."""
    orch = make(tmp_path, use_concierge=False)
    await orch.on_reasoner_messages([
        ReasonerMessage(kind="ack", task_id="t1", understood_as="rename"),
        ReasonerMessage(kind="done", task_id="t1", result="renamed 47 contacts"),
    ])
    assert orch.voice.spoken == ["renamed 47 contacts"]
    assert orch.concierge.calls == []
