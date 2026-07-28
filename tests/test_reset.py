"""POST /reset: restoring a clean slate between demo runs.

The project owner re-runs the same demo repeatedly and needs a way to get
back to a known-good starting point without restarting the process: device
data back to the pristine fixture, every task cleared, conversation history
cleared (so neither model carries stale context into the next run), and any
pending confirmation cleared. The session's event log is deliberately left
alone -- it is the recording of what happened, not scratch state.
"""
import json

import pytest
from fastapi.testclient import TestClient
from fakes import FakeConcierge, FakeVoice

from rtvoice.device import DeviceState
from rtvoice.events import EventLog
from rtvoice.orchestrator import Orchestrator, create_app
from rtvoice.protocol import OrchestratorMessage
from rtvoice.reasoner_stub import ReasonerStub
from rtvoice.registry import TaskStatus
from rtvoice.states import TurnEvent, UserState

PRISTINE = {
    "contacts": [{"id": 1, "first_name": "Sarah", "group": "work"},
                 {"id": 2, "first_name": "Marcus", "group": "work"}],
}


def make_orch(tmp_path, state_path, **kw):
    device = DeviceState(state_path, tmp_path / "journal.jsonl")
    return Orchestrator(
        reasoner=kw.pop("reasoner", ReasonerStub(device, latency_ms=0)),
        concierge=FakeConcierge(), voice=FakeVoice(),
        log=EventLog(tmp_path / "events.jsonl"), device=device, **kw,
    )


@pytest.mark.asyncio
async def test_reset_restores_the_fixture_clears_tasks_history_and_pending_confirmation(tmp_path):
    state_path = tmp_path / "device_state.json"
    state_path.write_text(json.dumps(PRISTINE))
    orch = make_orch(tmp_path, state_path)

    # Leave a dirtied device (a real commit, so state_path itself is
    # rewritten -- reload() must still recover the pristine content), a live
    # task, an outstanding confirmation, and something in history.
    await orch.on_turn_event(TurnEvent(UserState.COMPLETE, "set all my contacts to Hans", 0))
    tid = next(t.task_id for t in orch.registry.all()
               if t.status == TaskStatus.AWAITING_CONFIRM)
    await orch.on_turn_event(TurnEvent(UserState.COMPLETE, "yes do it", 1000))
    assert orch.registry.get(tid).status == TaskStatus.DONE
    assert [c["first_name"] for c in orch.device.query("contacts")] == ["Hans", "Hans"]
    assert json.loads(state_path.read_text())["contacts"][0]["first_name"] == "Hans"
    assert orch.history

    # A second task left genuinely unanswered, so there is a live pending
    # confirmation at the moment of reset, not just a finished one.
    await orch.on_turn_event(TurnEvent(UserState.COMPLETE, "set all my contacts to Priya", 2000))
    pending_tid = next(t.task_id for t in orch.registry.all()
                       if t.status == TaskStatus.AWAITING_CONFIRM)
    assert orch.policy_state.pending_question is not None

    # The demo operator's between-runs step: the pristine fixture is copied
    # back over the configured path (on the real demo machine this happens
    # at launch; here it stands in for that external step).
    state_path.write_text(json.dumps(PRISTINE))

    await orch.reset()

    assert [c["first_name"] for c in orch.device.query("contacts")] == ["Sarah", "Marcus"]
    assert orch.registry.all() == []
    assert orch.history == []
    assert orch.policy_state.pending_question is None
    assert orch._pending_questions == {}

    # The stale confirmation is truly gone, not merely hidden behind a
    # cleared registry: the reasoner's own bookkeeping for it is cleared
    # too, so it cannot be resurrected and committed later.
    assert orch.reasoner._pending == {}
    result = await orch.reasoner.handle(
        OrchestratorMessage(kind="clarification_answer", task_id=pending_tid, text="yes"))
    assert [m.kind for m in result] == ["noop"]
    assert [c["first_name"] for c in orch.device.query("contacts")] == ["Sarah", "Marcus"]


@pytest.mark.asyncio
async def test_reset_does_not_touch_the_session_event_log(tmp_path):
    """The event log is the recording of what happened in the session; a
    reset restores the DEMO's state, not the log of it. It survives, with
    the reset itself becoming one more entry in it."""
    state_path = tmp_path / "device_state.json"
    state_path.write_text(json.dumps(PRISTINE))
    orch = make_orch(tmp_path, state_path)

    await orch.on_turn_event(TurnEvent(UserState.COMPLETE, "hello", 0))
    before = EventLog.read(orch.log.path)
    assert before  # something was already logged

    await orch.reset()

    after = EventLog.read(orch.log.path)
    assert len(after) > len(before)  # append-only: reset adds, never truncates
    assert [e.kind for e in after[:len(before)]] == [e.kind for e in before]
    assert "reset" in [e.kind for e in after]


def test_post_reset_endpoint_restores_a_clean_slate(tmp_path):
    state_path = tmp_path / "device_state.json"
    state_path.write_text(json.dumps(PRISTINE))
    orch = make_orch(tmp_path, state_path)
    app = create_app(orch)
    client = TestClient(app)

    resp = client.post("/inject", json={"text": "set all my contacts to Hans"})
    assert resp.status_code == 200
    assert resp.json()["tasks"]  # a task now exists, awaiting confirmation

    resp = client.post("/reset")
    assert resp.status_code == 200
    body = resp.json()
    assert body["tasks"] == []
    assert [c["first_name"] for c in body["device"]["contacts"]] == ["Sarah", "Marcus"]

    state = client.get("/state").json()
    assert state["tasks"] == []
    assert state["pending_question"] is None
    assert orch.history == []


# --- the microphone buffer -----------------------------------------------
#
# It holds the last 30 seconds regardless of what the conversation is doing,
# and the read mark sits where the previous turn ended. Observed live: a fresh
# run opened with "Change my work contacts. Can you get me a booking at a
# Chinese restaurant?" -- the first half said in the PREVIOUS run -- and that
# whole thing was dispatched to the reasoner as one command.

@pytest.mark.asyncio
async def test_reset_discards_audio_from_the_previous_run(tmp_path):
    import numpy as np
    from rtvoice.voice_service import VoiceService

    state_path = tmp_path / "device_state.json"
    state_path.write_text(json.dumps(PRISTINE))
    orch = make_orch(tmp_path, state_path)
    orch.voice = VoiceService(tmp_path / "session")
    orch.voice.audio_log.append(np.ones(16000, dtype=np.float32) * 0.3)
    orch.voice.last_speech_ms = 1234

    await orch.reset()

    assert orch.voice.audio_log.take().size == 0
    assert orch.voice.last_speech_ms is None


@pytest.mark.asyncio
async def test_reset_clears_the_merge_hold_clock(tmp_path):
    """Otherwise the first utterance after a reset is measured against a hold
    that started in the run before it."""
    state_path = tmp_path / "device_state.json"
    state_path.write_text(json.dumps(PRISTINE))
    orch = make_orch(tmp_path, state_path)
    orch._merge_first_ms = 500
    orch._in_flight = "something from before"

    await orch.reset()

    assert orch._merge_first_ms is None
    assert orch._in_flight is None


@pytest.mark.asyncio
async def test_reset_survives_a_voice_service_with_no_microphone(tmp_path):
    """FakeVoice, POST /inject, and every existing test have no audio_log."""
    state_path = tmp_path / "device_state.json"
    state_path.write_text(json.dumps(PRISTINE))
    orch = make_orch(tmp_path, state_path)
    await orch.reset()          # must not raise


@pytest.mark.asyncio
async def test_reset_clears_the_planner_own_conversation(tmp_path):
    """The planner keeps its own history, separate from orchestrator.history.

    Left standing across a reset it plans a fresh run's first utterance
    against turns from a run that no longer exists. Observed: a vague request
    came back as the previous session's plan, staged against records nobody
    had mentioned; and a plain delete request came back as `noop` after a long
    demo, read against a dozen unrelated earlier turns.
    """
    class Planner:
        def __init__(self):
            self._history = [{"role": "user", "content": "rename Bao House"}]
            self.tokens = {}

        async def handle(self, msg):
            return []

    state_path = tmp_path / "device_state.json"
    state_path.write_text(json.dumps(PRISTINE))
    planner = Planner()
    orch = make_orch(tmp_path, state_path, reasoner=planner)

    await orch.reset()

    assert planner._history == []
