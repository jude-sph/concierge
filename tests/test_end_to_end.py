import asyncio
import json

import pytest
from fakes import FakeConcierge, FakeVoice

from rtvoice.cancellation import CancellationToken
from rtvoice.concierge import SpeechAct
from rtvoice.device import DeviceState
from rtvoice.events import EventLog
from rtvoice.llm_reasoner import LlmReasoner
from rtvoice.orchestrator import Orchestrator
from rtvoice.protocol import ReasonerMessage
from rtvoice.reasoner_stub import ReasonerStub
from rtvoice.registry import TaskStatus
from rtvoice.states import TurnEvent, UserState


def make_orch(tmp_path, **kw):
    state = tmp_path / "device_state.json"
    state.write_text(json.dumps({
        "contacts": [{"id": 1, "first_name": "Sarah", "group": "work"},
                     {"id": 2, "first_name": "Marcus", "group": "work"}]
    }))
    device = DeviceState(state, tmp_path / "journal.jsonl")
    return Orchestrator(
        reasoner=kw.pop("reasoner", ReasonerStub(device, latency_ms=0)),
        concierge=kw.pop("concierge", FakeConcierge()),
        voice=kw.pop("voice", FakeVoice()),
        log=EventLog(tmp_path / "events.jsonl"), device=device, **kw,
    )


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

    # The system is still voicing "Rename all 2 contacts to Hans?"
    orch.policy_state.speaking = True

    await orch.on_turn_event(TurnEvent(UserState.BACKCHANNEL, "yes", 1000))

    assert orch.registry.get(tid).status == TaskStatus.DONE
    assert [c["first_name"] for c in orch.device.query("contacts")] == ["Hans", "Hans"]


# --- Coordinator fix-round tests -------------------------------------------
#
# Round 1 review found two real defects in the Task 12 implementation:
#
# 1. CRITICAL: Orchestrator.tokens was declared but never populated, so
#    _abort() could never actually fire a live CancellationToken -- it was
#    silently a no-op, and cancellation depended entirely on the reasoner
#    receiving and processing a fire-and-forget "cancel" message. Fixed by
#    sharing a single tokens dict between the orchestrator and any reasoner
#    that exposes one (ReasonerStub now does), with ReasonerStub._confirm()
#    registering its live token there.
#
# 2. IMPORTANT: making SendUtterance/AskConcierge run concurrently (the fix
#    for the pipeline bug, above) opened a race: two _ask_concierge() calls
#    can be in flight at once, so (a) a Stop() from a later turn didn't
#    prevent an already-in-flight, now-stale reply from being spoken, and
#    (b) two concurrent replies could interleave audio. Fixed with a speech
#    generation counter (checked right before speaking) plus a speak lock.


class SlowReasoner:
    """Test double: mimics a reasoner in the middle of a slow destructive
    write. It registers a live CancellationToken in the shared `tokens` dict
    (exactly as ReasonerStub._confirm does) and then blocks indefinitely --
    standing in for a device.update() loop that has not returned yet. It also
    stalls on "cancel" messages, so a test can prove that Orchestrator._abort
    cancels the live token immediately, before this reasoner has any chance
    to process that message.
    """

    def __init__(self, *, tokens=None):
        self.tokens = tokens if tokens is not None else {}
        self.write_started = asyncio.Event()
        self.cancel_processed = False

    async def handle(self, msg):
        if msg.kind == "utterance":
            return [
                ReasonerMessage(kind="ack", task_id="slow-task",
                                understood_as="slow destructive op"),
                ReasonerMessage(kind="confirm_required", task_id="slow-task",
                                verbatim_text="This will do something destructive. Confirm?"),
            ]
        if msg.kind == "clarification_answer":
            token = CancellationToken()
            self.tokens[msg.task_id] = token
            self.write_started.set()
            try:
                await asyncio.sleep(10)  # stands in for a write that never finishes
            except asyncio.CancelledError:
                pass
            return [ReasonerMessage(kind="noop")]
        if msg.kind == "cancel":
            await asyncio.sleep(10)  # the reasoner never gets around to this
            self.cancel_processed = True
            return [ReasonerMessage(kind="failed", task_id=msg.task_id, reason="cancelled")]
        return [ReasonerMessage(kind="noop")]


@pytest.mark.asyncio
async def test_abort_cancels_a_live_token_without_the_reasoner_processing_cancel(tmp_path):
    state = tmp_path / "device_state.json"
    state.write_text(json.dumps({"contacts": [{"id": 1, "first_name": "Sarah"}]}))
    device = DeviceState(state, tmp_path / "journal.jsonl")
    reasoner = SlowReasoner()
    orch = Orchestrator(
        reasoner=reasoner, concierge=FakeConcierge(), voice=FakeVoice(),
        log=EventLog(tmp_path / "events.jsonl"), device=device,
    )
    # The orchestrator must have bound reasoner.tokens to its own dict.
    assert orch.tokens is reasoner.tokens

    await orch.on_turn_event(TurnEvent(UserState.COMPLETE, "do something destructive", 0))
    tid = "slow-task"
    assert orch.registry.get(tid).status == TaskStatus.AWAITING_CONFIRM

    # Confirm it: the (fake) reasoner registers a live token, then blocks --
    # standing in for a real, still-running destructive write.
    confirm_task = asyncio.create_task(
        orch.on_turn_event(TurnEvent(UserState.COMPLETE, "yes do it", 1)))
    await asyncio.wait_for(reasoner.write_started.wait(), timeout=1)

    token = orch.tokens[tid]
    assert token.cancelled is False

    await orch._abort(tid)

    # Cancelled synchronously, by direct reference to the shared token --
    # with no dependency on the fire-and-forget "cancel" OrchestratorMessage
    # ever being delivered or processed.
    assert token.cancelled is True
    assert reasoner.cancel_processed is False

    # Cleanup: cancel the still-pending background tasks this test spawned.
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)


class SlowConcierge:
    """Concierge double whose respond() takes a controllable amount of time,
    so a test can trigger a Stop() while a reply is still in flight."""

    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self.calls = 0

    async def respond(self, registry, history, trigger):
        self.calls += 1
        await asyncio.sleep(self.delay)
        return SpeechAct(act="acknowledge", text="on it")


@pytest.mark.asyncio
async def test_stop_during_inflight_concierge_ask_prevents_stale_speech(tmp_path):
    state = tmp_path / "device_state.json"
    state.write_text(json.dumps({"contacts": []}))
    device = DeviceState(state, tmp_path / "journal.jsonl")
    orch = Orchestrator(
        reasoner=ReasonerStub(device, latency_ms=0),
        concierge=SlowConcierge(delay=0.05),
        voice=FakeVoice(),
        log=EventLog(tmp_path / "events.jsonl"),
        device=device,
    )

    ask_task = asyncio.create_task(orch._ask_concierge("reasoner_update"))
    await asyncio.sleep(0)  # let _ask_concierge capture its generation and start respond()

    # A barge-in on a later turn stops speech while the concierge is still
    # composing an earlier reply.
    orch.policy_state.speaking = True
    await orch.on_turn_event(TurnEvent(UserState.NONIDLE, "wait", 0))
    assert orch.voice.stops == 1

    await ask_task

    # The stale reply must never have been spoken (nor recorded in history).
    assert orch.voice.spoken == []
    assert all(h.get("content") != "on it" for h in orch.history)


class RecordingVoice:
    """Voice double that records the peak number of concurrent speak() calls,
    to prove the speak lock actually serialises audio."""

    def __init__(self, delay: float = 0.02):
        self.delay = delay
        self._active = 0
        self.max_concurrent = 0
        self.spoken = []
        self.stops = 0

    async def speak(self, text, utterance_id):
        self._active += 1
        self.max_concurrent = max(self.max_concurrent, self._active)
        await asyncio.sleep(self.delay)
        self.spoken.append(text)
        self._active -= 1

    async def stop(self):
        self.stops += 1


@pytest.mark.asyncio
async def test_concurrent_concierge_asks_do_not_interleave_speech(tmp_path):
    state = tmp_path / "device_state.json"
    state.write_text(json.dumps({"contacts": []}))
    device = DeviceState(state, tmp_path / "journal.jsonl")
    voice = RecordingVoice(delay=0.02)
    orch = Orchestrator(
        reasoner=ReasonerStub(device, latency_ms=0),
        concierge=FakeConcierge(),
        voice=voice,
        log=EventLog(tmp_path / "events.jsonl"),
        device=device,
    )

    await asyncio.gather(
        orch._ask_concierge("user_turn"),
        orch._ask_concierge("reasoner_update"),
    )

    assert voice.max_concurrent == 1
    assert len(voice.spoken) == 2


@pytest.mark.asyncio
async def test_one_branch_raising_does_not_orphan_its_sibling_and_is_logged(tmp_path):
    state = tmp_path / "device_state.json"
    state.write_text(json.dumps({"contacts": []}))
    device = DeviceState(state, tmp_path / "journal.jsonl")

    class ExplodingReasoner:
        async def handle(self, msg):
            raise RuntimeError("boom")

    concierge = FakeConcierge()
    orch = Orchestrator(
        reasoner=ExplodingReasoner(), concierge=concierge, voice=FakeVoice(),
        log=EventLog(tmp_path / "events.jsonl"), device=device,
    )

    # decide() returns [SendUtterance, AskConcierge] together for a COMPLETE
    # turn; the reasoner branch raises, but the concierge branch must still
    # run to completion rather than being orphaned.
    await orch.on_turn_event(TurnEvent(UserState.COMPLETE, "hello", 0))

    assert len(concierge.calls) == 1  # the sibling branch was not orphaned
    kinds = [e.kind for e in EventLog.read(orch.log.path)]
    assert "turn_task_error" in kinds


# --- conversation ordering: the user's turn must never appear to follow
#     the assistant's reply to it --------------------------------------------


class SlowDispatchReasoner:
    """A reasoner whose handle() takes a while, so the reasoner-dispatch side
    of on_turn_event's gather is still in flight well after the concierge
    (which never waits on it) has already replied."""

    async def handle(self, msg):
        await asyncio.sleep(0.05)
        return [ReasonerMessage(kind="noop")]


@pytest.mark.asyncio
async def test_the_users_line_is_recorded_in_history_before_the_concierges_reply(tmp_path):
    """orch.history is what both models read as conversation context, and
    what a transcript rendered from it would show. However long the reasoner
    takes, and regardless of how fast the concierge answers, the user's own
    turn must already be in history before the concierge's reply can be
    appended to it -- never the other way round."""
    orch = make_orch(tmp_path, reasoner=SlowDispatchReasoner())

    await orch.on_turn_event(TurnEvent(UserState.COMPLETE, "Hello.", 0))

    assert orch.history[0] == {"role": "user", "content": "Hello."}
    roles = [h["role"] for h in orch.history]
    assert roles.index("user") < roles.index("assistant")


@pytest.mark.asyncio
async def test_user_utterance_event_precedes_concierge_act_even_when_the_merge_window_holds_the_reasoner_dispatch(tmp_path):
    """The live-session bug: the UI transcript is built from the event
    stream, in arrival order, and used to key the user's line off
    "to_reasoner" -- an event only emitted once _dispatch actually calls the
    reasoner. An ordinary spoken utterance sits in the fragment-merge window
    for up to merge_window_ms before that happens (see orchestrator.py's
    merge-window comments), while the concierge answers immediately, so
    "to_reasoner" arrives AFTER "concierge_act" and the assistant's reply
    rendered first. "user_utterance" is logged synchronously, in the same
    place self.history is appended to, independent of the merge window, so
    it must always precede "concierge_act" -- and "to_reasoner" must not
    even have happened yet.
    """
    orch = make_orch(tmp_path, merge_window_ms=1200)

    # Deliver the turn event the way AudioDriver does: the audio clock has
    # already reached this point, which is what makes the utterance eligible
    # to be held in the merge window rather than bypassing it (see
    # Orchestrator._clocked / _bypasses_merge).
    await orch.on_tick(1000)
    await orch.on_turn_event(TurnEvent(UserState.COMPLETE, "Hello.", 1000))

    kinds = [e.kind for e in EventLog.read(orch.log.path)]
    assert "user_utterance" in kinds
    assert "concierge_act" in kinds
    # The merge window is still open -- the reasoner has not been dispatched
    # to yet, so "to_reasoner" cannot have fired at all. If the transcript
    # were still keyed off it, at this point it would show nothing at all
    # for the user's turn while already showing the assistant's reply.
    assert "to_reasoner" not in kinds
    assert kinds.index("user_utterance") < kinds.index("concierge_act")


# --- not-actionable vs. genuine failure, through the full orchestrator -----
#
# The live-session defect: "Don't you have access to the calendar?" is a
# rhetorical question about the SYSTEM, not a device lookup. The reasoner
# and the concierge both receive every finalised turn in parallel (see
# policy.decide), so if the reasoner cannot cleanly classify a question like
# this, its own internal fallback wording ("I couldn't work out what to do
# with that") gets spoken as a `failed` fact, stepping on whatever the
# concierge says for the same turn. Correctly classified as operation="none"
# (see llm_reasoner.SYSTEM_PROMPT's worked example for exactly this
# phrasing), nothing is spoken on the reasoner's behalf at all -- the
# concierge alone carries the turn. A genuinely concrete request that
# actually cannot be done must still be audible; the fix here must not
# blur into silencing that too.


class _FakeCompletion:
    """Minimal stand-in for the httpx.Response the reasoner's HTTP client
    normally returns -- same shape as FakeResponse in test_llm_reasoner.py."""

    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._content}}]}


def _llm_reasoner_replying(tmp_path, plan: dict) -> LlmReasoner:
    state_path = tmp_path / "reasoner_device.json"
    state_path.write_text(json.dumps({}))
    device = DeviceState(state_path, tmp_path / "reasoner_journal.jsonl")
    reasoner = LlmReasoner(device, base_url="http://fake/v1", model="fake-model")

    async def fake_post(url, **kw):
        return _FakeCompletion(json.dumps(plan))

    reasoner._client.post = fake_post
    return reasoner


@pytest.mark.asyncio
async def test_not_actionable_utterance_produces_no_spoken_failure(tmp_path):
    reasoner = _llm_reasoner_replying(tmp_path, {"intents": [
        {"operation": "none",
         "understood_as": "asked whether the system can read the calendar"},
    ]})
    orch = make_orch(tmp_path, reasoner=reasoner)

    await orch.on_turn_event(
        TurnEvent(UserState.COMPLETE, "Don't you have access to my calendar?", 0))

    # No task was ever registered, and nothing failure-shaped was spoken --
    # the reasoner contributed silence, exactly like ordinary chit-chat.
    assert orch.registry.all() == []
    assert not any("couldn't work out" in s for s in orch.voice.spoken)
    # The concierge's own reply for this turn is unaffected -- it is the
    # only thing that spoke.
    assert orch.voice.spoken == ["on it"]


@pytest.mark.asyncio
async def test_genuine_operation_failure_is_still_spoken(tmp_path):
    """Contrast case: a real request for something the device genuinely has
    no way to do must remain audible -- the fix above must not swallow this
    too."""
    reasoner = _llm_reasoner_replying(tmp_path, {"intents": [
        {"operation": "unsupported", "understood_as": "get an uber to the station"},
    ]})
    orch = make_orch(tmp_path, reasoner=reasoner)

    await orch.on_turn_event(
        TurnEvent(UserState.COMPLETE, "get me an uber to the station", 0))

    assert "this phone can't do that yet" in orch.voice.spoken
