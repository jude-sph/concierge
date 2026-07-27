"""The fragment merge window.

Measured on real hardware, one 21s compound instruction arrived from
SoulX-Duplug as SIX separate `user_complete` events:

     6.08s  "Can you get me booking at a Chinese restaurant?"
     9.76s  "Tn in Soho and please change."
    12.16s  "nge all of my contact."
    15.20s  "HA hands."
    18.08s  "And see if you can get me an Uber."
    20.96s  "To central Sahob."

The speaker paused at a clause boundary and "can you get me a booking at a
chinese restaurant" genuinely IS a complete sentence, so their semantic
detector is not malfunctioning -- the fact that the speaker meant to continue
is simply not in the text. Sweeping their `max_wait_num` from 8 to 32 moved it
6 -> 4 turns, non-monotonically, and never to 1. So it is mitigated here, by
holding a finalised utterance for merge_window_ms and concatenating whatever
follows before the reasoner ever sees it.

Note the fragments are cut mid-word ("chan-"/"-nge"), because their ASR buffer
resets at each false boundary. Nothing here may assume any single fragment is
well-formed.
"""
import asyncio
import json

import numpy as np
import pytest
from fakes import FakeConcierge, FakeVoice

from rtvoice.audio_driver import AudioDriver
from rtvoice.device import DeviceState
from rtvoice.events import EventLog
from rtvoice.orchestrator import Orchestrator
from rtvoice.protocol import ReasonerMessage
from rtvoice.reasoner_stub import ReasonerStub
from rtvoice.registry import TaskStatus
from rtvoice.soulx_client import CHUNK_SAMPLES
from rtvoice.states import TurnEvent, UserState


class RecordingReasoner:
    """Records every protocol message and does nothing else."""

    def __init__(self):
        self.seen = []
        self.tokens = {}

    async def handle(self, msg):
        self.seen.append(msg)
        return [ReasonerMessage(kind="noop")]

    @property
    def utterances(self) -> list[str]:
        return [m.text for m in self.seen if m.kind == "utterance"]


def make(tmp_path, reasoner=None, **kw):
    state = tmp_path / "device_state.json"
    state.write_text(json.dumps({
        "contacts": [{"id": 1, "first_name": "Sarah", "group": "work"},
                     {"id": 2, "first_name": "Marcus", "group": "work"}],
    }))
    device = DeviceState(state, tmp_path / "journal.jsonl")
    return Orchestrator(
        reasoner=reasoner if reasoner is not None else ReasonerStub(device, latency_ms=0),
        concierge=FakeConcierge(), voice=FakeVoice(),
        log=EventLog(tmp_path / "events.jsonl"), device=device, **kw,
    )


async def say(orch, text, t_ms, state=UserState.COMPLETE):
    """Deliver a turn event the way AudioDriver does: the audio clock has
    already reached this point in the stream when the event is handed over."""
    await orch.on_tick(t_ms)
    await orch.on_turn_event(TurnEvent(state, text, t_ms))


def kinds(orch):
    return [e.kind for e in EventLog.read(orch.log.path)]


# --- the merge itself --------------------------------------------------------

@pytest.mark.asyncio
async def test_three_fragments_inside_the_window_reach_the_reasoner_as_one(tmp_path):
    """The headline case. Three `user_complete` events for one spoken command
    must produce exactly ONE utterance, not three."""
    reasoner = RecordingReasoner()
    orch = make(tmp_path, reasoner=reasoner, merge_window_ms=1200)

    await say(orch, "book a chinese restaurant in soho", 1000)
    await say(orch, "and change all my contacts", 1800)
    await say(orch, "to hans", 2600)
    assert reasoner.utterances == []          # still accumulating

    await orch.on_tick(3900)                  # window closes

    assert reasoner.utterances == [
        "book a chinese restaurant in soho and change all my contacts to hans"]


@pytest.mark.asyncio
async def test_a_fragment_after_the_window_expires_starts_a_new_utterance(tmp_path):
    """The window must close. A fragment arriving after it is a new command,
    and merging it into the previous one would fabricate an instruction the
    user never gave."""
    reasoner = RecordingReasoner()
    orch = make(tmp_path, reasoner=reasoner, merge_window_ms=1200)

    await say(orch, "find pizza places", 1000)
    await say(orch, "near london bridge", 1600)      # inside the window
    await say(orch, "what is on my calendar", 9000)  # long after it

    await orch.on_tick(11000)

    assert reasoner.utterances == [
        "find pizza places near london bridge",
        "what is on my calendar",
    ]


@pytest.mark.asyncio
async def test_nonidle_during_a_pending_merge_prevents_premature_dispatch(tmp_path):
    """The measured fragments were 2.4-3.7s apart -- far wider than the
    window. What keeps them together is that the user is audibly still
    speaking in between, which arrives as `user_nonidle`. If NONIDLE did not
    hold the buffer open, the window would close mid-command and the merge
    would never fire on real audio at all."""
    reasoner = RecordingReasoner()
    orch = make(tmp_path, reasoner=reasoner, merge_window_ms=1200)

    await say(orch, "book a chinese restaurant", 1000)
    await say(orch, "book a chinese", 1900, state=UserState.NONIDLE)
    await orch.on_tick(2800)          # 1800ms after the fragment: expired but
    assert reasoner.utterances == []  # for the NONIDLE holding it open

    await say(orch, "and get me an uber", 3000)
    await orch.on_tick(4300)

    assert reasoner.utterances == ["book a chinese restaurant and get me an uber"]


@pytest.mark.asyncio
async def test_a_merge_is_logged_as_a_distinct_event(tmp_path):
    """Everything renders from the event log: a merge that leaves no trace
    makes the reasoner appear to receive text nobody said in one breath, with
    no way to see why -- in the live UI or in replay."""
    reasoner = RecordingReasoner()
    orch = make(tmp_path, reasoner=reasoner, merge_window_ms=1200)

    await say(orch, "find pizza", 1000)
    await say(orch, "in soho", 1600)
    await orch.on_tick(3000)

    merges = [e for e in EventLog.read(orch.log.path) if e.kind == "utterance_merged"]
    assert len(merges) == 1
    assert merges[0].data["parts"] == ["find pizza", "in soho"]
    assert merges[0].data["transcript"] == "find pizza in soho"


@pytest.mark.asyncio
async def test_the_concierge_still_answers_every_fragment_immediately(tmp_path):
    """Why the delay is affordable HERE: the concierge and the reasoner are
    two heads on the same stream. The user is acknowledged the instant they
    stop speaking, exactly as before; only the reasoner dispatch moves. If the
    concierge were held back too, this would be a latency regression on every
    single turn."""
    reasoner = RecordingReasoner()
    orch = make(tmp_path, reasoner=reasoner, merge_window_ms=1200)

    await say(orch, "book a restaurant", 1000)
    assert len(orch.concierge.calls) == 1     # answered without waiting
    await say(orch, "in soho", 1600)
    assert len(orch.concierge.calls) == 2

    await orch.on_tick(3000)
    assert reasoner.utterances == ["book a restaurant in soho"]


# --- the confirmation exception ---------------------------------------------

@pytest.mark.asyncio
async def test_an_answer_to_a_pending_confirmation_bypasses_the_window(tmp_path):
    """A "yes" answering a destructive-write confirmation is time-critical and
    must never be merged: concatenated with the next thing the user says it is
    no longer answer-shaped, so the write can never be confirmed at all, and
    the merged text risks being read as a fresh command instead."""
    orch = make(tmp_path, merge_window_ms=1200)

    await say(orch, "set all my contacts to Hans", 1000)
    await orch.on_tick(2400)                  # let that command's window close
    tid = next(t.task_id for t in orch.registry.all()
               if t.status == TaskStatus.AWAITING_CONFIRM)

    # The answer, immediately followed by a fresh (fragmented) command.
    await say(orch, "yes do it", 3000)
    assert orch.registry.get(tid).status == TaskStatus.DONE      # not delayed
    assert [c["first_name"] for c in orch.device.query("contacts")] == ["Hans", "Hans"]

    await say(orch, "now find chinese restaurants", 3400)
    await say(orch, "in soho", 4000)
    await orch.on_tick(5300)

    assert any(t.understood_as == "search: chinese restaurants in soho"
               for t in orch.registry.all())


@pytest.mark.asyncio
async def test_a_non_answer_during_a_pending_confirmation_is_still_merged(tmp_path):
    """The exception is narrow: only an answer-SHAPED utterance skips the
    window. A new command spoken while a confirmation is pending is ordinary
    speech and is just as likely to be fragmented, so it still merges -- and
    it still must not commit the pending write."""
    orch = make(tmp_path, merge_window_ms=1200)
    await say(orch, "set all my contacts to Hans", 1000)
    await orch.on_tick(2400)
    tid = next(t.task_id for t in orch.registry.all()
               if t.status == TaskStatus.AWAITING_CONFIRM)

    await say(orch, "find chinese restaurants", 3000)
    await say(orch, "in soho", 3600)
    await orch.on_tick(5000)

    assert any(t.understood_as == "search: chinese restaurants in soho"
               for t in orch.registry.all())
    assert orch.registry.get(tid).status == TaskStatus.AWAITING_CONFIRM
    assert [c["first_name"] for c in orch.device.query("contacts")] == ["Sarah", "Marcus"]


# --- interaction with the silence timeout -----------------------------------

@pytest.mark.asyncio
async def test_merging_does_not_break_the_silence_timeout_dispatch(tmp_path):
    """The two timers handle OPPOSITE failures of the same model -- one for
    `user_complete` never arriving, one for it arriving too often -- and both
    now hang off on_tick. Each must still fire, exactly once."""
    reasoner = RecordingReasoner()
    orch = make(tmp_path, reasoner=reasoner,
                merge_window_ms=1200, silence_timeout_ms=2000)

    await say(orch, "find pizza", 1000)
    await say(orch, "in soho", 1600)
    await orch.on_tick(3000)                       # merge dispatch
    assert reasoner.utterances == ["find pizza in soho"]

    # Now the opposite failure: the model declines the turn and never returns.
    await orch.on_turn_event(
        TurnEvent(UserState.INCOMPLETE, "rename my contacts to Hans", 4000))
    await orch.on_tick(5000)
    assert reasoner.utterances == ["find pizza in soho"]          # not yet
    await orch.on_tick(6100)

    assert reasoner.utterances == ["find pizza in soho",
                                   "rename my contacts to Hans"]


@pytest.mark.asyncio
async def test_a_forced_dispatch_is_not_double_dispatched_by_the_merge(tmp_path):
    """The silence timeout dispatches directly; it must not also leave the
    text sitting in the merge buffer for a later tick to send again."""
    reasoner = RecordingReasoner()
    orch = make(tmp_path, reasoner=reasoner,
                merge_window_ms=1200, silence_timeout_ms=2000)

    await orch.on_turn_event(TurnEvent(UserState.INCOMPLETE, "find pizza", 0))
    await orch.on_tick(2500)
    assert reasoner.utterances == ["find pizza"]
    for t in (2700, 4000, 9000):
        await orch.on_tick(t)
    assert reasoner.utterances == ["find pizza"]


@pytest.mark.asyncio
async def test_merging_preserves_force_dispatch_reconciliation(tmp_path):
    """A genuine COMPLETE that exactly repeats a force-dispatched text is the
    turn-taking model catching up, not new content, and must stay suppressed
    -- including now that it would otherwise land in the merge buffer, where
    the duplicate would resurface a window later and be glued onto whatever
    the user said next."""
    reasoner = RecordingReasoner()
    orch = make(tmp_path, reasoner=reasoner,
                merge_window_ms=1200, silence_timeout_ms=2000)
    text = "rename my contacts to Hans"

    await orch.on_turn_event(TurnEvent(UserState.INCOMPLETE, text, 0))
    await orch.on_tick(2500)                 # forced dispatch
    await say(orch, text, 3000)              # the model catches up
    await orch.on_tick(4300)
    assert reasoner.utterances == [text]     # not dispatched twice

    # ...and a genuinely new command afterwards still merges normally.
    await say(orch, "find pizza", 5000)
    await say(orch, "in soho", 5600)
    await orch.on_tick(7000)
    assert reasoner.utterances == [text, "find pizza in soho"]


@pytest.mark.asyncio
async def test_a_superset_of_a_force_dispatched_text_still_reaches_the_reasoner(tmp_path):
    """The exact-match memo must not swallow a COMPLETE carrying MORE than was
    force-dispatched: the user kept talking, that is real content -- and under
    merging it is exactly the kind of content the window exists to collect."""
    reasoner = RecordingReasoner()
    orch = make(tmp_path, reasoner=reasoner,
                merge_window_ms=1200, silence_timeout_ms=2000)

    await orch.on_turn_event(TurnEvent(UserState.INCOMPLETE, "find pizza", 0))
    await orch.on_tick(2500)                          # forced dispatch
    await say(orch, "find pizza places nearby", 3000)  # strict superset
    await say(orch, "and italian ones", 3600)
    await orch.on_tick(5000)

    assert reasoner.utterances == ["find pizza",
                                   "find pizza places nearby and italian ones"]


# --- concurrency -------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_merge_bypassing_dispatch_still_runs_beside_the_concierge(tmp_path):
    """The bypass must not re-serialise the two heads. A confirmation answer
    is the one utterance that still dispatches from inside on_turn_event, so
    it is where an `await self._dispatch(...)` could most easily creep back in
    -- and it is the worst place for it, since the user gets no acknowledgment
    until a destructive write has finished.

    The reasoner here refuses to finish the write until the concierge has been
    asked, so a serialised orchestrator deadlocks rather than merely being
    slow.
    """
    gate: dict = {"event": None}

    class GatedReasoner:
        def __init__(self):
            self.tokens = {}

        async def handle(self, msg):
            if msg.kind == "utterance":
                return [
                    ReasonerMessage(kind="ack", task_id="t1",
                                    understood_as="something destructive"),
                    ReasonerMessage(kind="confirm_required", task_id="t1",
                                    verbatim_text="Do the destructive thing?"),
                ]
            if msg.kind == "clarification_answer":
                if gate["event"] is not None:
                    await asyncio.wait_for(gate["event"].wait(), timeout=1)
                return [ReasonerMessage(kind="done", task_id=msg.task_id,
                                        result="did it")]
            return [ReasonerMessage(kind="noop")]

    class SignallingConcierge(FakeConcierge):
        async def respond(self, registry, history, trigger):
            if gate["event"] is not None:
                gate["event"].set()
            return await super().respond(registry, history, trigger)

    orch = make(tmp_path, reasoner=GatedReasoner(), merge_window_ms=1200)
    orch.concierge = SignallingConcierge()

    await say(orch, "do something destructive", 1000)
    await orch.on_tick(2400)
    assert orch.registry.get("t1").status == TaskStatus.AWAITING_CONFIRM

    gate["event"] = asyncio.Event()
    await say(orch, "yes do it", 3000)

    assert orch.registry.get("t1").status == TaskStatus.DONE
    assert "turn_task_error" not in kinds(orch)


# --- the measured recording, through the real driver ------------------------

MEASURED_FRAGMENTS = [
    (6080, "Can you get me booking at a Chinese restaurant?"),
    (9760, "Tn in Soho and please change."),
    (12160, "nge all of my contact."),
    (15200, "HA hands."),
    (18080, "And see if you can get me an Uber."),
    (20960, "To central Sahob."),
]


class ScriptedVoice:
    """Voice service double: replays a per-chunk script and advances an audio
    clock exactly as the real one does (events are stamped before the clock
    advances)."""

    CHUNK_MS = 160

    def __init__(self, script):
        self.script = script
        self._t_ms = 0
        self.spoken = []
        self.stops = 0

    @property
    def stream_ms(self) -> int:
        return self._t_ms

    async def feed_audio(self, chunk):
        pairs = self.script.get(self._t_ms, [])
        events = [TurnEvent(s, t, self._t_ms) for s, t in pairs]
        self._t_ms += self.CHUNK_MS
        return events

    async def speak(self, text, utterance_id):
        self.spoken.append(text)

    async def stop(self):
        self.stops += 1


@pytest.mark.asyncio
async def test_the_measured_six_fragment_recording_becomes_one_utterance(tmp_path):
    """End to end through AudioDriver, on the sequence actually measured on
    hardware: continuous speech from 0.5s to 21s, broken into six
    `user_complete` events at the timestamps observed. One spoken instruction
    must reach the reasoner as one utterance."""
    fragments = dict(MEASURED_FRAGMENTS)
    script: dict[int, list] = {}
    for t in range(480, 21120, 160):
        if t in fragments:
            script[t] = [(UserState.COMPLETE, fragments[t])]
        else:
            script[t] = [(UserState.NONIDLE, "")]

    reasoner = RecordingReasoner()
    orch = make(tmp_path, reasoner=reasoner, merge_window_ms=1200)
    voice = ScriptedVoice(script)
    driver = AudioDriver(voice, orch)

    # 24s of audio: the recording plus enough trailing silence for the window
    # to close.
    await driver.run([np.zeros(CHUNK_SAMPLES, dtype=np.float32)] * 150)

    assert reasoner.utterances == [" ".join(t for _, t in MEASURED_FRAGMENTS)]
