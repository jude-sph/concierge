"""Wires turn events, the reasoner and the concierge together.

The concierge and reasoner run in PARALLEL, not as a pipeline: the concierge
answers every turn at conversational latency while the reasoner acts only on
what is actionable. Neither waits for the other.
"""
from __future__ import annotations

import asyncio
import uuid

# Imported at module level, not lazily inside create_app: `from __future__
# import annotations` (above) turns every annotation in this file into a
# string, resolved later via typing.get_type_hints() against the function's
# __globals__. FastAPI relies on exactly that resolution to recognise a
# `ws: WebSocket` parameter as "inject the connection", not a query param.
# A name that only exists in create_app's *local* scope (as it did when this
# was `from fastapi import WebSocket` inside the function body) is invisible
# to get_type_hints, which resolves against module globals -- so every
# websocket route silently degraded to expecting `ws` as a query parameter
# and closed the connection with code 1008 before the handler ever ran. This
# had no test covering it (nothing here previously drove a websocket through
# real FastAPI request handling) until the /audio route below did.
# FastAPI itself has no import-time side effects (no filesystem or network
# access), so hoisting this above create_app does not reintroduce what
# test_importing_the_orchestrator_module_has_no_side_effects guards against --
# that only forbids *constructing an app* (or a device, or a log) at import
# time, which still happens nowhere but inside create_app / create_default_app.
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .cancellation import CancellationToken
from .events import EventLog
from .policy import AskConcierge, PolicyState, SendUtterance, Stop, decide
from .protocol import OrchestratorMessage, ReasonerMessage
from .registry import TaskRegistry, TaskStatus
from .states import TurnEvent, UserState, is_answer_shaped

SPEAK_ON = {"done", "failed", "need_clarification", "confirm_required"}


class Orchestrator:
    def __init__(
        self, reasoner, concierge, voice, log: EventLog, device,
        silence_timeout_ms: int = 2000,
        reasoner_timeout_s: float = 20.0,
        use_concierge: bool = True,
        merge_window_ms: int = 1200,
    ) -> None:
        self.reasoner = reasoner
        self.concierge = concierge
        self.voice = voice
        self.log = log
        self.device = device
        self.registry = TaskRegistry()
        self.policy_state = PolicyState()
        self.history: list[dict] = []
        self.tokens: dict[str, CancellationToken] = {}
        # Pending yes/no questions, keyed by the task that asked. PolicyState
        # carries a single `pending_question` slot but AWAITING_CONFIRM is
        # per-task: setting that slot on ANY confirm_required and clearing it
        # on ANY done/failed meant that for a compound command, the second
        # clause finishing wiped the first clause's question, and clause ORDER
        # decided whether a bare "yes" was heard as an answer at all. The slot
        # is now DERIVED from this dict (see _sync_pending_question).
        self._pending_questions: dict[str, str] = {}
        self._seq = 0
        self.silence_timeout_ms = silence_timeout_ms
        self.reasoner_timeout_s = reasoner_timeout_s
        self.use_concierge = use_concierge
        self._pending_partial: str | None = None
        self._pending_since_ms: int | None = None
        # Set by on_tick() when the silence timer forces a dispatch. If the
        # turn-taking model later catches up and emits a genuine COMPLETE for
        # the exact same text, on_turn_event must recognise the utterance as
        # already handled instead of dispatching it a second time (or, worse,
        # routing it into _dispatch's "awaiting confirm" branch where it would
        # be misread as a yes/no answer to the task the forced dispatch just
        # created). A COMPLETE carrying MORE text than this is genuinely new
        # content and is dispatched normally.
        self._force_dispatched_text: str | None = None

        # --- fragment merge window ------------------------------------------
        #
        # Measured on real hardware: SoulX-Duplug emits MULTIPLE `user_complete`
        # events for ONE spoken command. A 21s compound instruction arrived as
        # six turns, because the speaker paused at a clause boundary and "can
        # you get me a booking at a chinese restaurant" genuinely IS a complete
        # sentence. Their semantic detector is not malfunctioning: the fact
        # that the speaker intended to continue is not present in the text.
        # Sweeping their `max_wait_num` from 8 to 32 moved it 6 -> 4 turns,
        # non-monotonically, and never to 1. So it is mitigated here.
        #
        # A finalised utterance is held for merge_window_ms and concatenated
        # with anything that follows it, then dispatched to the reasoner ONCE.
        # This is affordable *here specifically* because the concierge and the
        # reasoner run in parallel: the concierge still answers immediately, so
        # the user is acknowledged at the same latency as before and only the
        # (already slow, already async) reasoner dispatch moves.
        self.merge_window_ms = merge_window_ms
        self._merge_parts: list[str] = []
        self._merge_last_ms: int | None = None
        # The audio clock's last position, as reported by on_tick. Holding an
        # utterance is only safe when something will later flush it, and
        # on_tick is that something: AudioDriver pumps it after every 160ms
        # chunk, so a live audio session is always clocked past the timestamp
        # of the turn event it just delivered. A turn event that arrives
        # AHEAD of the clock did not come off the audio path at all -- POST
        # /inject types text straight in, and its caller reads the resulting
        # tasks from the response -- so it was never fragmented by the
        # turn-taking model and must not be delayed by a window nothing is
        # driving. See _clocked.
        self._last_tick_ms: int | None = None

        # Speech generation counter (same pattern as tts.py's KokoroTTS): every
        # Stop() bumps it, and an in-flight _ask_concierge that captured an
        # older value knows its reply is stale and must not speak it.
        self._speech_gen = 0
        # Serialises the "actually call voice.speak()" critical section so a
        # concierge reply (the top-level user_turn ask) and a directly-spoken
        # reasoner fact (_speak_facts, below) can never have overlapping
        # audio.
        self._speak_lock = asyncio.Lock()

        # Some reasoners run in-process and hold live CancellationTokens for
        # work they're doing (e.g. ReasonerStub mid-write). If the reasoner
        # exposes a `tokens` dict, bind it to our own so _abort can fire a
        # token by direct reference -- correctness must not depend on the
        # reasoner ever receiving or processing a "cancel" message.
        if hasattr(reasoner, "tokens"):
            reasoner.tokens = self.tokens

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    async def on_tick(self, now_ms: int) -> None:
        """Drive both timers. Called ~every 160ms on the audio clock.

        The two handle OPPOSITE failures of the same upstream model and are
        deliberately kept as separate buffers so they can never dispatch the
        same words twice:

        * the silence timeout covers `user_complete` never arriving -- it
          holds an INCOMPLETE partial, and a COMPLETE clears it;
        * the merge window covers `user_complete` arriving too OFTEN -- it
          holds finalised COMPLETE texts, and nothing else writes to it.
        """
        self._last_tick_ms = now_ms
        await self._check_silence_timeout(now_ms)
        await self._flush_merge_if_expired(now_ms)

    async def _check_silence_timeout(self, now_ms: int) -> None:
        """SoulX-Duplug declining to take the turn is normally correct, but if
        it never fires the system would hang. After silence_timeout_ms of
        holding an incomplete utterance, dispatch it anyway."""
        if self._pending_since_ms is None or self._pending_partial is None:
            return
        if now_ms - self._pending_since_ms < self.silence_timeout_ms:
            return
        text = self._pending_partial
        self._pending_partial = None
        self._pending_since_ms = None
        self._force_dispatched_text = text
        self.log.append("silence_timeout", transcript=text)
        self.history.append({"role": "user", "content": text})
        await self._dispatch(text)

    # --- merge window --------------------------------------------------------

    def _merge_append(self, text: str, t_ms: int) -> None:
        self._merge_parts.append(text)
        self._merge_last_ms = t_ms
        self.log.append("merge_pending", transcript=text,
                        parts=len(self._merge_parts))

    def _merge_take(self) -> str | None:
        """Empty the buffer and return the concatenated utterance."""
        if not self._merge_parts:
            return None
        parts, self._merge_parts = self._merge_parts, []
        self._merge_last_ms = None
        text = " ".join(parts)
        if len(parts) > 1:
            # A distinct event kind so a merge is visible in the session log
            # and in replay -- otherwise the reasoner appears to receive text
            # nobody said in one breath, with no record of why.
            self.log.append("utterance_merged", parts=parts, transcript=text,
                            count=len(parts))
        return text

    def _merge_expired(self, now_ms: int) -> bool:
        return (self._merge_last_ms is not None
                and now_ms - self._merge_last_ms >= self.merge_window_ms)

    async def _flush_merge_if_expired(self, now_ms: int) -> None:
        if not self._merge_expired(now_ms):
            return
        text = self._merge_take()
        if text is not None:
            await self._dispatch(text)

    def _clocked(self, t_ms: int) -> bool:
        """Is a clock running that has already reached this turn event?

        Deferring an utterance is only safe if something will flush the
        buffer. on_tick is that something, and AudioDriver pumps it after
        every audio chunk, so during a live session the clock is always at or
        past the timestamp of the event just delivered. An event arriving
        ahead of the clock was not produced by the audio path -- POST /inject
        synthesises a COMPLETE at t=0 and reads the resulting tasks straight
        out of the response -- so it cannot be a turn-taking fragment and
        must not be held for a window nothing is going to close.
        """
        return self._last_tick_ms is not None and self._last_tick_ms >= t_ms

    def _bypasses_merge(self, text: str, t_ms: int) -> bool:
        """Must this utterance go to the reasoner right now?

        Answering a pending confirmation is never delayed. A "yes" resolving a
        destructive-write confirmation is time-critical, and concatenating it
        with whatever the user says next would both destroy its answer shape
        (making the write impossible to confirm at all) and risk the merged
        text being read as a fresh command. The AWAITING_CONFIRM +
        answer-shaped test is exactly the gate _answer_target applies, through
        the one is_answer_shaped in states.py.
        """
        if self.merge_window_ms <= 0:
            return True
        if is_answer_shaped(text) and any(t.status == TaskStatus.AWAITING_CONFIRM
                                          for t in self.registry.all()):
            return True
        return not self._clocked(t_ms)

    async def on_turn_event(self, ev: TurnEvent) -> None:
        self.log.append("turn_event", state=ev.state.value,
                        transcript=ev.transcript, t_ms=ev.t_ms)

        if ev.state is UserState.INCOMPLETE and ev.transcript.strip():
            self._pending_partial = ev.transcript
            self._pending_since_ms = ev.t_ms
        elif ev.state in (UserState.COMPLETE, UserState.NONIDLE):
            # NONIDLE means the user resumed speaking: the timer that was
            # armed for the PREVIOUS incomplete utterance must not keep
            # counting down against fresh speech, or on_tick will force-
            # dispatch a stale partial while the user is still mid-sentence.
            # A later INCOMPLETE (if the model declines the turn again)
            # re-arms it naturally, from the correct new t_ms.
            self._pending_partial = None
            self._pending_since_ms = None

        if (ev.state is UserState.NONIDLE
                and self._merge_last_ms is not None
                and not self._merge_expired(ev.t_ms)):
            # The user resumed speaking inside the merge window: whatever they
            # are saying now belongs with what is already buffered, so hold it
            # and keep accumulating rather than dispatching a fragment. This is
            # what makes the window work at all on real audio -- the six
            # measured fragments were 2-4s apart, far wider than the window,
            # but the user is audibly speaking throughout the gaps.
            self._merge_last_ms = ev.t_ms

        concurrent: list[asyncio.Task] = []

        # A fragment arriving after the window has elapsed is a NEW utterance,
        # not a continuation. Flush first, using this event's own clock, so
        # on_tick is not the only thing that can close a merge.
        if self._merge_expired(ev.t_ms):
            expired = self._merge_take()
            if expired is not None:
                concurrent.append(asyncio.create_task(self._dispatch(expired)))

        # SendUtterance (-> reasoner) and AskConcierge (-> concierge) are
        # launched concurrently, never one awaited before the other starts.
        # The reasoner is the gatekeeper for device-touching work and may be
        # slow (a real LLM call); the concierge must still answer at
        # conversational latency. Awaiting them sequentially here would turn
        # this into a pipeline and make every spoken reply wait on the
        # reasoner's round trip, which is exactly the failure mode this
        # module exists to avoid.
        for action in decide(ev, self.policy_state):
            if isinstance(action, Stop):
                await self.voice.stop()
                self.policy_state.speaking = False
                self._speech_gen += 1
                self.log.append("tts_stopped")

            elif isinstance(action, SendUtterance):
                if (self._force_dispatched_text is not None
                        and action.text == self._force_dispatched_text):
                    # The silence timer already force-dispatched this exact
                    # text (see on_tick). The turn-taking model has now
                    # caught up and emitted the genuine COMPLETE for the same
                    # utterance -- it is not new content, so do not dispatch
                    # it again (that would either duplicate the work, or, if
                    # a task is AWAITING_CONFIRM, get misread by _dispatch as
                    # a yes/no answer to it).
                    self._force_dispatched_text = None
                    self.log.append("silence_timeout_reconciled", transcript=action.text)
                else:
                    self._force_dispatched_text = None
                    # History is per-fragment and immediate: the concierge
                    # answers this fragment now, so it must see the words that
                    # were actually just spoken, not wait for the merge.
                    self.history.append({"role": "user", "content": action.text})
                    if self._bypasses_merge(action.text, ev.t_ms):
                        concurrent.append(
                            asyncio.create_task(self._dispatch(action.text)))
                    else:
                        self._merge_append(action.text, ev.t_ms)

            elif isinstance(action, AskConcierge):
                # This is the ONLY trigger that ever reaches the concierge --
                # reasoner facts are spoken directly by _speak_facts and never
                # ask the concierge for anything (see on_reasoner_messages).
                # use_concierge=False bypasses this call so the flag can be
                # used to measure whether the concierge earns its place at
                # all, rather than it still speaking filler on every ordinary
                # turn.
                if self.use_concierge:
                    concurrent.append(asyncio.create_task(self._ask_concierge(action.trigger)))

        if concurrent:
            # return_exceptions=True: one branch raising must not leave its
            # sibling running orphaned (gather would otherwise still wait for
            # it, but a plain gather re-raises and abandons bookkeeping around
            # it). Log failures instead of swallowing them silently.
            results = await asyncio.gather(*concurrent, return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    self.log.append("turn_task_error",
                                    error=repr(result), error_type=type(result).__name__)

    def _sync_pending_question(self) -> None:
        """Re-derive PolicyState's single pending_question slot from the
        per-task questions, dropping any whose task is no longer awaiting.

        Deriving it (rather than setting and clearing it at each call site)
        is what makes it impossible for one task's completion to clear
        another task's question, and impossible for an abort to leave a
        question set forever -- which disabled reflexive barge-in for the
        rest of the session, since policy.decide() then read every NONIDLE
        as an answer instead of an interruption.
        """
        for tid in list(self._pending_questions):
            task = self.registry.get(tid)
            if task is None or task.status != TaskStatus.AWAITING_CONFIRM:
                del self._pending_questions[tid]
        questions = list(self._pending_questions.values())
        # The most recent question is the one the user is being asked.
        self.policy_state.pending_question = questions[-1] if questions else None

    def _answer_target(self, text: str) -> str | None:
        """The task this utterance ANSWERS, if any.

        Routing every utterance to the pending task while one is awaiting
        confirmation is how a non-answer became a destructive commit. A
        pending yes/no question only captures an utterance that is actually
        shaped like a yes/no answer; anything else -- a new command, an aside
        to someone else, the user's own continuation of a sentence the
        silence timer force-dispatched -- is dispatched as a fresh utterance
        and leaves the question pending, resolved neither way.

        An open clarification ("which contacts?") is different: it asks for
        free-form content, so any non-empty utterance answers it.
        """
        if not text.strip():
            return None
        awaiting = [t for t in self.registry.all()
                    if t.status == TaskStatus.AWAITING_CONFIRM]
        if not awaiting:
            return None
        target = awaiting[-1]
        if target.task_id in self._pending_questions and not is_answer_shaped(text):
            return None
        return target.task_id

    async def _dispatch(self, text: str) -> None:
        target = self._answer_target(text)
        if target is not None:
            msg = OrchestratorMessage(kind="clarification_answer",
                                      task_id=target,
                                      text=text, seq=self._next_seq())
        else:
            if self._pending_questions:
                self.log.append("confirm_left_pending",
                                task_ids=list(self._pending_questions),
                                transcript=text)
            msg = OrchestratorMessage(kind="utterance", text=text,
                                      raw_transcript=text, seq=self._next_seq())

        data = msg.model_dump()
        self.log.append("to_reasoner", msg_kind=data.pop("kind"), **data)
        try:
            replies = await asyncio.wait_for(
                self.reasoner.handle(msg), timeout=self.reasoner_timeout_s
            )
        except asyncio.TimeoutError:
            self.log.append("reasoner_timeout", seq=msg.seq)
            # A fresh utterance has no task_id yet (only clarification_answer
            # messages carry the id of the task they're answering). Falling
            # back to a fixed literal here would collapse every timed-out
            # fresh utterance onto the same task id; the registry's
            # terminal-state guard then silently swallows every timeout after
            # the first, since that id is already FAILED. Derive a unique id
            # from the message's own seq instead.
            replies = [ReasonerMessage(
                kind="failed", task_id=msg.task_id or f"timeout-{msg.seq}",
                reason="timed out waiting for the reasoner",
            )]
        await self.on_reasoner_messages(replies)

    async def on_reasoner_messages(self, msgs: list[ReasonerMessage]) -> None:
        should_speak = False
        for m in msgs:
            data = m.model_dump()
            self.log.append("from_reasoner", msg_kind=data.pop("kind"), **data)
            self.registry.apply(m)
            if m.kind in SPEAK_ON:
                should_speak = True
            if m.kind == "confirm_required" and m.task_id:
                self._pending_questions[m.task_id] = m.verbatim_text
        # Derived, not assigned: a question belongs to the task that asked it,
        # and is cleared only when THAT task stops awaiting an answer.
        self._sync_pending_question()
        if not should_speak:
            return
        await self._speak_facts(msgs)

    async def _speak_facts(self, msgs: list[ReasonerMessage]) -> None:
        """Speak the reasoner's own authored text directly -- no concierge in
        the loop at all.

        The design is: the reasoner writes facts, verbatim; the concierge
        writes only conversational framing. Routing facts through a concierge
        "relay" act asked a *second* model to reproduce the first model's
        words, and against real models that failed both ways that matter --
        a correct answer got replaced by a parroted-back question, and a
        destructive-write confirmation got reworded (silently changing what
        the user was agreeing to). Speaking `m.result`/`m.reason`/
        `m.verbatim_text`/`m.question` straight from the ReasonerMessage means
        nothing is generated here, so nothing can be invented -- this is the
        only path a `done`, `failed`, `need_clarification` or
        `confirm_required` is ever spoken on, unconditionally, whether or not
        a concierge is configured at all. The concierge keeps its own job
        (conversational turns, acknowledgements, its own clarifying
        questions) via the separate "user_turn" ask in on_turn_event, which
        this does not touch.

        Still routed through the same generation-check + speak lock as
        _ask_concierge, so a Stop() during a spoken fact can't leave stale
        audio playing or overlap with a barge-in -- there is no unguarded
        speech path.
        """
        gen = self._speech_gen
        for m in msgs:
            if m.kind not in SPEAK_ON:
                continue
            text = m.result or m.reason or m.verbatim_text or m.question
            if not text:
                continue
            async with self._speak_lock:
                if gen != self._speech_gen:
                    self.log.append("speak_skipped_stale",
                                    trigger="reasoner_fact", text=text)
                    continue
                self.history.append({"role": "assistant", "content": text})
                self.policy_state.speaking = True
                try:
                    await self.voice.speak(text, uuid.uuid4().hex)
                except Exception as exc:
                    # kokoro may not be installed, or TTS may fail for any
                    # other reason. The text is already in history above,
                    # so the transcript is correct even with no audio --
                    # a silent-but-correct demo beats a crashed one. This
                    # must never propagate: on_tick (silence timeout, merge
                    # flush) calls into here directly, outside any
                    # gather(return_exceptions=True), so an uncaught
                    # exception here would kill the audio socket's loop.
                    self.log.append("tts_failed", text=text, error=repr(exc),
                                    error_type=type(exc).__name__)
                finally:
                    self.policy_state.speaking = False

    async def _ask_concierge(self, trigger: str) -> None:
        # Snapshot the speech generation before doing anything async. If a
        # Stop() (barge-in from a later turn) bumps it while we're waiting on
        # the concierge or on the speak lock, this reply is stale by the time
        # we'd speak it and must be dropped rather than played over -- or
        # after -- whatever superseded it.
        gen = self._speech_gen
        act = await self.concierge.respond(self.registry, self.history, trigger)
        self.log.append("concierge_act", act=act.act, cites=act.cites, text=act.text,
                        violations=getattr(self.concierge, "violations", 0))

        if act.act == "abort" and act.cites:
            await self._abort(act.cites)
            return

        if not act.text:
            return

        async with self._speak_lock:
            if gen != self._speech_gen:
                self.log.append("speak_skipped_stale", trigger=trigger, text=act.text)
                return
            self.history.append({"role": "assistant", "content": act.text})
            self.policy_state.speaking = True
            try:
                await self.voice.speak(act.text, uuid.uuid4().hex)
            except Exception as exc:
                # See the matching comment in on_reasoner_messages: TTS being
                # unavailable (no kokoro, no GPU) must not crash the caller.
                # The reply text is already in history, so the transcript is
                # right even when nothing is heard.
                self.log.append("tts_failed", text=act.text, error=repr(exc),
                                error_type=type(exc).__name__)
            finally:
                self.policy_state.speaking = False

    async def _abort(self, task_id: str) -> None:
        """Fire the token FIRST; notifying the reasoner is secondary and
        correctness never depends on it arriving."""
        token = self.tokens.get(task_id)
        if token is not None:
            token.cancel()
        self.registry.mark_cancelled(task_id)
        # An aborted task's question is gone with it. Left set, PolicyState's
        # pending_question makes decide() read every subsequent NONIDLE as an
        # answer rather than a barge-in, so "Stop!" stops halting speech for
        # the rest of the session.
        self._pending_questions.pop(task_id, None)
        self._sync_pending_question()
        self.log.append("aborted", task_id=task_id)
        msg = OrchestratorMessage(kind="cancel", task_id=task_id, seq=self._next_seq())
        asyncio.create_task(self.reasoner.handle(msg))


class Inject(BaseModel):
    """Request body for POST /inject.

    Defined at module scope, not inside create_app: this file has `from
    __future__ import annotations`, which makes every annotation a string
    resolved lazily via the function's *global* namespace. A Pydantic model
    defined inside create_app is a local variable, invisible to that lookup,
    so FastAPI silently fails to recognise it as the request body and treats
    `text` as a missing query parameter instead (every call 422s). Module
    scope keeps it resolvable.
    """

    text: str


def create_app(orch: Orchestrator) -> "FastAPI":
    """HTTP/WS surface. /inject is the development affordance that lets the
    whole loop be exercised without a microphone."""
    from .audio_ws import install_audio_route

    app = FastAPI()
    app.state.orch = orch

    install_audio_route(app, orch)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        from pathlib import Path

        return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")

    @app.post("/inject")
    async def inject(body: Inject) -> dict:
        o: Orchestrator = app.state.orch
        await o.on_turn_event(TurnEvent(UserState.COMPLETE, body.text, 0))
        return {
            "tasks": [
                {"task_id": t.task_id, "understood_as": t.understood_as,
                 "status": t.status.value, "detail": t.detail}
                for t in o.registry.all()
            ]
        }

    @app.get("/state")
    async def state() -> dict:
        o: Orchestrator = app.state.orch
        return {
            "tasks": [
                {"task_id": t.task_id, "understood_as": t.understood_as,
                 "status": t.status.value, "detail": t.detail}
                for t in o.registry.all()
            ],
            "device": o.device.snapshot(),
            "speaking": o.policy_state.speaking,
            "pending_question": o.policy_state.pending_question,
        }

    @app.websocket("/events")
    async def events(ws: WebSocket) -> None:
        await ws.accept()
        o: Orchestrator = app.state.orch
        try:
            async for ev in o.log.subscribe():
                await ws.send_text(ev.to_json())
        except WebSocketDisconnect:
            pass

    return app


def _build_reasoner(device):
    """Which reasoner this process runs, from the environment.

    REASONER=llm selects the model-driven one, which needs an OpenAI-compatible
    endpoint (REASONER_URL, falling back to the concierge's). Anything else --
    including unset -- keeps the rule-based stub, so offline use and the test
    suite are unaffected by this switch existing.
    """
    import os

    latency_ms = int(os.environ.get("REASONER_LATENCY_MS", "0"))
    if os.environ.get("REASONER", "").strip().lower() == "llm":
        from .llm_reasoner import LlmReasoner

        return LlmReasoner(
            device,
            base_url=os.environ.get(
                "REASONER_URL",
                os.environ.get("CONCIERGE_URL", "http://localhost:8001/v1")),
            model=os.environ.get("REASONER_MODEL", "Qwen/Qwen3-4B"),
            latency_ms=latency_ms,
        )

    from .reasoner_stub import ReasonerStub

    return ReasonerStub(device, latency_ms=latency_ms)


def build_default_orchestrator(session_dir=None) -> Orchestrator:
    """The standard wiring: journaled device, a reasoner, real concierge.

    Shared by the HTTP app and by tools/audio_client.py so there is one
    definition of "the system", not two that drift.
    """
    import os
    from pathlib import Path

    from .concierge import Concierge
    from .device import DeviceState
    from .voice_service import VoiceService

    session = Path(session_dir) if session_dir is not None else (
        Path("sessions") / os.environ.get("SESSION_ID", "dev"))
    device = DeviceState(
        os.environ.get("DEVICE_STATE", "fixtures/device_state.json"),
        session / "device_journal.jsonl",
    )
    log = EventLog(session / "events.jsonl")
    voice = VoiceService(session, os.environ.get("SOULX_URL", "ws://localhost:8000/turn"))
    concierge = Concierge(base_url=os.environ.get("CONCIERGE_URL", "http://localhost:8001/v1"))
    # Off by default: the concierge needs a vLLM server, which the v1 demo
    # does not require. Set USE_CONCIERGE=1 to turn it on once that's running.
    use_concierge = os.environ.get("USE_CONCIERGE", "0").strip().lower() in ("1", "true", "yes")
    return Orchestrator(
        reasoner=_build_reasoner(device),
        concierge=concierge, voice=voice, log=log, device=device,
        use_concierge=use_concierge,
    )


def create_default_app() -> "FastAPI":
    """ASGI factory.

    This used to run at import time as `app = _default_app()`, so merely
    importing this module built a device, opened an event log and created
    session directories under whatever the current working directory happened
    to be -- including during test collection. Behind a factory, importing the
    module has no filesystem or network side effects.

    Serve it with:  uvicorn rtvoice.orchestrator:create_default_app --factory
    """
    return create_app(build_default_orchestrator())
