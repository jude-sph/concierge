import json
import pytest
from rtvoice.device import DeviceState
from rtvoice.events import EventLog
from rtvoice.orchestrator import Orchestrator
from rtvoice.protocol import ReasonerMessage
from rtvoice.reasoner_stub import ReasonerStub
from rtvoice.registry import TaskStatus
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


@pytest.mark.asyncio
async def test_concierge_bypass_skips_concierge_on_a_full_turn(tmp_path):
    """CRITICAL 1 regression: decide() emits AskConcierge(trigger="user_turn")
    for every COMPLETE turn, independent of the reasoner. use_concierge=False
    must gate that action too, not just the reasoner_update ask, or the flag
    can't measure whether the concierge earns its place -- it would still
    speak filler on every ordinary turn."""
    orch = make(tmp_path, use_concierge=False)
    await orch.on_turn_event(TurnEvent(UserState.COMPLETE, "find pizza places", 0))
    assert orch.concierge.calls == []
    assert orch.voice.spoken == ["found 12 results for pizza places"]


@pytest.mark.asyncio
async def test_silence_timeout_reconciled_confirm_required_not_falsely_cancelled(tmp_path):
    """CRITICAL 2(a) regression: a forced dispatch that lands on a confirm-
    required command must not have the later genuine COMPLETE for the SAME
    text routed into _dispatch's "awaiting confirm -> clarification_answer"
    branch, where it contains no yes/no wording and gets misread as a
    decline, silently flipping the task to FAILED "cancelled by user" when
    the user cancelled nothing."""
    orch = make(tmp_path, silence_timeout_ms=2000)
    text = "rename my contacts to Hans"
    await orch.on_turn_event(TurnEvent(UserState.INCOMPLETE, text, 0))
    await orch.on_tick(now_ms=2500)  # forces the dispatch -> AWAITING_CONFIRM task

    tasks = orch.registry.all()
    assert len(tasks) == 1
    assert tasks[0].status == TaskStatus.AWAITING_CONFIRM

    # The turn-taking model catches up and emits the genuine COMPLETE for the
    # identical text.
    await orch.on_turn_event(TurnEvent(UserState.COMPLETE, text, 3000))

    tasks = orch.registry.all()
    assert len(tasks) == 1
    assert tasks[0].status == TaskStatus.AWAITING_CONFIRM   # not FAILED


@pytest.mark.asyncio
async def test_silence_timeout_reconciled_non_confirm_command_not_duplicated(tmp_path):
    """CRITICAL 2(b) regression: a forced dispatch of a non-confirm command
    completes; the later genuine COMPLETE for the SAME text must not
    dispatch it again as a brand-new task (duplicate work, spoken twice)."""
    orch = make(tmp_path, silence_timeout_ms=2000)
    text = "find pizza places"
    await orch.on_turn_event(TurnEvent(UserState.INCOMPLETE, text, 0))
    await orch.on_tick(now_ms=2500)  # forces the dispatch -> one DONE task
    assert len(orch.registry.all()) == 1

    await orch.on_turn_event(TurnEvent(UserState.COMPLETE, text, 3000))
    assert len(orch.registry.all()) == 1   # no duplicate task


@pytest.mark.asyncio
async def test_silence_timeout_reconciliation_does_not_swallow_genuinely_new_content(tmp_path):
    """The exact-match memo must not suppress dispatch of a COMPLETE that
    carries MORE text than what was force-dispatched -- that is genuinely
    new content (the user kept talking) and must still go to the reasoner."""
    orch = make(tmp_path, silence_timeout_ms=2000)
    await orch.on_turn_event(TurnEvent(UserState.INCOMPLETE, "find pizza", 0))
    await orch.on_tick(now_ms=2500)  # forces dispatch of "find pizza"
    assert len(orch.registry.all()) == 1

    await orch.on_turn_event(
        TurnEvent(UserState.COMPLETE, "find pizza places nearby", 3000)
    )
    assert len(orch.registry.all()) == 2   # genuinely new text, dispatched


@pytest.mark.asyncio
async def test_nonidle_resuming_speech_clears_the_silence_timer(tmp_path):
    """IMPORTANT 3 regression: a NONIDLE after an INCOMPLETE (the user
    resumed speaking) must reset the pending-partial timer. Otherwise
    on_tick force-dispatches a stale partial while the user is still
    mid-sentence."""
    orch = make(tmp_path, silence_timeout_ms=2000)
    await orch.on_turn_event(TurnEvent(UserState.INCOMPLETE, "rename my contacts", 0))
    await orch.on_turn_event(TurnEvent(UserState.NONIDLE, "rename my contacts to", 1500))
    await orch.on_tick(now_ms=2100)
    assert orch.registry.all() == []   # timer was reset by NONIDLE, no forced dispatch


@pytest.mark.asyncio
async def test_reasoner_timeout_produces_distinct_failed_tasks(tmp_path):
    """IMPORTANT 4 regression: falling back to a fixed task_id ("unknown")
    for every timed-out fresh utterance collapses them onto one task; the
    registry's terminal-state guard then silently swallows every timeout
    after the first. Two separate timeouts must produce two distinct,
    visible failed tasks."""
    class HungReasoner:
        async def handle(self, msg):
            import asyncio
            await asyncio.sleep(10)
            return []

    orch = make(tmp_path, reasoner_timeout_s=0.05)
    orch.reasoner = HungReasoner()
    await orch.on_turn_event(TurnEvent(UserState.COMPLETE, "do a thing", 0))
    await orch.on_turn_event(TurnEvent(UserState.COMPLETE, "do another thing", 1000))

    failed = [t for t in orch.registry.all() if t.status == TaskStatus.FAILED]
    assert len(failed) == 2
    assert len({t.task_id for t in failed}) == 2   # distinct ids, both visible
