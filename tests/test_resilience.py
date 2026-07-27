import asyncio
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
async def test_done_result_bypasses_the_concierge_even_when_it_is_enabled(tmp_path):
    """The design: the reasoner writes facts, verbatim; the concierge writes
    only conversational framing. A `done` fact must never be routed through
    the concierge's ask/validate/relay round trip -- not just when
    use_concierge=False (that was already true before this change), but
    unconditionally, because a second model asked to relay a fact is a
    second chance for it to invent a question instead of stating the
    answer (the live-model failure this change removes)."""
    orch = make(tmp_path)  # use_concierge defaults to True
    await orch.on_reasoner_messages([
        ReasonerMessage(kind="ack", task_id="t1", understood_as="rename"),
        ReasonerMessage(kind="done", task_id="t1", result="renamed 47 contacts"),
    ])
    assert orch.voice.spoken == ["renamed 47 contacts"]
    assert orch.concierge.calls == []


@pytest.mark.asyncio
async def test_confirm_required_is_spoken_exactly_as_authored(tmp_path):
    """The other live-model failure this change removes: a concierge asked
    to relay a destructive-write confirmation could reword it, silently
    changing what the user believes they are agreeing to. The fact now
    reaches the speaker unmediated, character for character."""
    orch = make(tmp_path)
    verbatim = "This will delete 3 messages from Marcus Webb. Confirm?"
    await orch.on_reasoner_messages([
        ReasonerMessage(kind="ack", task_id="t1", understood_as="delete"),
        ReasonerMessage(kind="confirm_required", task_id="t1", verbatim_text=verbatim),
    ])
    assert orch.voice.spoken == [verbatim]
    assert orch.concierge.calls == []


@pytest.mark.asyncio
async def test_concierge_still_called_for_a_conversational_turn_with_no_task(tmp_path):
    """Facts bypass the concierge entirely, but its own job is untouched: a
    turn that produces no task at all (ReasonerStub's noop path, for plain
    chit-chat) must still reach the concierge via the ordinary "user_turn"
    ask -- there is simply nothing for a fact-bypass to have skipped."""
    orch = make(tmp_path)
    await orch.on_turn_event(TurnEvent(UserState.COMPLETE, "hello, how are you", 0))
    assert orch.registry.all() == []
    assert orch.concierge.calls and orch.concierge.calls[0][1] == "user_turn"
    assert orch.voice.spoken == ["on it"]  # the concierge's own ack, unaffected


@pytest.mark.asyncio
async def test_direct_speak_path_is_cancelled_by_stop_via_generation_counter(tmp_path):
    """_speak_facts (the direct-speak path for reasoner facts) must be
    guarded by the same speech-generation counter as _ask_concierge, not a
    separate, unguarded route: a Stop() (barge-in) landing while a fact is
    still queued behind the speak lock must prevent it from ever reaching
    voice.spoken or history, exactly as proven for the concierge path by
    test_stop_during_inflight_concierge_ask_prevents_stale_speech."""
    orch = make(tmp_path)

    # Hold the speak lock ourselves, standing in for another in-flight
    # speech act (a concierge reply, or a sibling fact) still being spoken
    # when the Stop() below fires.
    await orch._speak_lock.acquire()
    try:
        speak_task = asyncio.create_task(orch.on_reasoner_messages([
            ReasonerMessage(kind="done", task_id="t1", result="renamed 47 contacts"),
        ]))
        await asyncio.sleep(0)  # let it snapshot its generation and block on the lock

        orch.policy_state.speaking = True
        await orch.on_turn_event(TurnEvent(UserState.NONIDLE, "wait", 0))
        assert orch.voice.stops == 1
    finally:
        orch._speak_lock.release()

    await speak_task

    assert orch.voice.spoken == []
    assert all(h.get("content") != "renamed 47 contacts" for h in orch.history)


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


class FailingVoice:
    """Stands in for a VoiceService whose TTS is unavailable (e.g. `kokoro`
    is not installed): speak() always raises, exactly like
    VoiceService.speak() does when `self.tts` fails to construct."""

    def __init__(self):
        self.spoken = []
        self.stops = 0

    async def speak(self, text, utterance_id):
        raise RuntimeError("kokoro not installed")

    async def stop(self):
        self.stops += 1


@pytest.mark.asyncio
async def test_tts_failure_is_caught_and_logged_on_the_bypass_path(tmp_path):
    """Graceful degradation, bypass path (use_concierge=False, the v1
    default): voice.speak() raising must not propagate -- the reply text is
    already in history before speak() is attempted, so the transcript stays
    correct even with no audio, and a clear event is logged instead of a
    crash."""
    orch = make(tmp_path, use_concierge=False)
    orch.voice = FailingVoice()
    await orch.on_reasoner_messages([
        ReasonerMessage(kind="ack", task_id="t1", understood_as="rename"),
        ReasonerMessage(kind="done", task_id="t1", result="renamed 47 contacts"),
    ])
    assert orch.history[-1] == {"role": "assistant", "content": "renamed 47 contacts"}
    kinds = [e.kind for e in EventLog.read(orch.log.path)]
    assert "tts_failed" in kinds


@pytest.mark.asyncio
async def test_tts_failure_is_caught_and_logged_via_the_concierge(tmp_path):
    orch = make(tmp_path)
    orch.voice = FailingVoice()
    await orch._ask_concierge("user_turn")
    assert orch.history[-1] == {"role": "assistant", "content": "on it"}
    kinds = [e.kind for e in EventLog.read(orch.log.path)]
    assert "tts_failed" in kinds


@pytest.mark.asyncio
async def test_tts_failure_does_not_crash_a_full_turn(tmp_path):
    """Before this fix, voice.speak() had no try/except at all: the only
    reason a TTS failure didn't crash the process was that _ask_concierge
    happened to run inside on_turn_event's gather(return_exceptions=True) --
    which does not cover every call path (on_tick's silence-timeout and
    merge-window flushes call straight into _dispatch, outside any gather).
    This exercises the ordinary gather-covered path and pins the new,
    explicit behaviour: a distinct tts_failed event, not a turn_task_error."""
    orch = make(tmp_path)
    orch.voice = FailingVoice()
    await orch.on_turn_event(TurnEvent(UserState.COMPLETE, "find pizza places", 0))
    kinds = [e.kind for e in EventLog.read(orch.log.path)]
    assert "turn_task_error" not in kinds
    assert "tts_failed" in kinds


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
