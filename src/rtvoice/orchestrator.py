"""Wires turn events, the reasoner and the concierge together.

The concierge and reasoner run in PARALLEL, not as a pipeline: the concierge
answers every turn at conversational latency while the reasoner acts only on
what is actionable. Neither waits for the other.
"""
from __future__ import annotations

import asyncio
import uuid

from pydantic import BaseModel

from .cancellation import CancellationToken
from .events import EventLog
from .policy import AskConcierge, PolicyState, SendUtterance, Stop, decide
from .protocol import OrchestratorMessage, ReasonerMessage
from .registry import TaskRegistry, TaskStatus
from .states import TurnEvent, UserState

SPEAK_ON = {"done", "failed", "need_clarification", "confirm_required"}


class Orchestrator:
    def __init__(
        self, reasoner, concierge, voice, log: EventLog, device,
        silence_timeout_ms: int = 2000,
        reasoner_timeout_s: float = 20.0,
        use_concierge: bool = True,
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

        # Speech generation counter (same pattern as tts.py's KokoroTTS): every
        # Stop() bumps it, and an in-flight _ask_concierge that captured an
        # older value knows its reply is stale and must not speak it.
        self._speech_gen = 0
        # Serialises the "actually call voice.speak()" critical section so two
        # concurrent concierge replies (the top-level user_turn ask and a
        # nested reasoner_update ask) can never have overlapping audio.
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
        """Drive the silence timeout. Called ~every 160ms by voice-service.

        SoulX-Duplug declining to take the turn is normally correct, but if it
        never fires the system would hang. After silence_timeout_ms of holding
        an incomplete utterance, dispatch it anyway.
        """
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

        # SendUtterance (-> reasoner) and AskConcierge (-> concierge) are
        # launched concurrently, never one awaited before the other starts.
        # The reasoner is the gatekeeper for device-touching work and may be
        # slow (a real LLM call); the concierge must still answer at
        # conversational latency. Awaiting them sequentially here would turn
        # this into a pipeline and make every spoken reply wait on the
        # reasoner's round trip, which is exactly the failure mode this
        # module exists to avoid.
        concurrent: list[asyncio.Task] = []
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
                    self.history.append({"role": "user", "content": action.text})
                    concurrent.append(asyncio.create_task(self._dispatch(action.text)))

            elif isinstance(action, AskConcierge):
                # use_concierge=False must genuinely bypass the concierge on
                # every path, not just the reasoner_update ask -- otherwise
                # the flag can't be used to measure whether the concierge
                # earns its place, since it would still speak filler on
                # every ordinary turn.
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

    async def _dispatch(self, text: str) -> None:
        awaiting = [t for t in self.registry.all()
                    if t.status == TaskStatus.AWAITING_CONFIRM]
        if awaiting:
            msg = OrchestratorMessage(kind="clarification_answer",
                                      task_id=awaiting[0].task_id,
                                      text=text, seq=self._next_seq())
        else:
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
            if m.kind == "confirm_required":
                self.policy_state.pending_question = m.verbatim_text
            if m.kind in ("done", "failed"):
                self.policy_state.pending_question = None
        if not should_speak:
            return
        if self.use_concierge:
            await self._ask_concierge("reasoner_update")
            return
        # Bypass: speak the reasoner's verbatim text directly, without the
        # concierge in the loop at all. Nothing can be invented because
        # nothing is generated -- this exists so "does the concierge earn
        # its keep" can be measured against a real baseline. Still routed
        # through the same generation-check + speak lock as _ask_concierge
        # so a Stop() during a bypassed reply can't leave stale audio
        # playing or overlap with a barge-in.
        gen = self._speech_gen
        for m in msgs:
            if m.kind in SPEAK_ON:
                text = m.result or m.reason or m.verbatim_text or m.question
                if not text:
                    continue
                async with self._speak_lock:
                    if gen != self._speech_gen:
                        self.log.append("speak_skipped_stale",
                                        trigger="reasoner_update_bypass", text=text)
                        continue
                    self.history.append({"role": "assistant", "content": text})
                    self.policy_state.speaking = True
                    await self.voice.speak(text, uuid.uuid4().hex)
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
            await self.voice.speak(act.text, uuid.uuid4().hex)
            self.policy_state.speaking = False

    async def _abort(self, task_id: str) -> None:
        """Fire the token FIRST; notifying the reasoner is secondary and
        correctness never depends on it arriving."""
        token = self.tokens.get(task_id)
        if token is not None:
            token.cancel()
        self.registry.mark_cancelled(task_id)
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
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect

    from .states import TurnEvent, UserState

    app = FastAPI()
    app.state.orch = orch

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


def _default_app():
    import os
    from pathlib import Path

    from .device import DeviceState
    from .concierge import Concierge
    from .reasoner_stub import ReasonerStub
    from .voice_service import VoiceService

    session = Path("sessions") / os.environ.get("SESSION_ID", "dev")
    device = DeviceState("fixtures/device_state.json", session / "device_journal.jsonl")
    log = EventLog(session / "events.jsonl")
    voice = VoiceService(session)
    concierge = Concierge(base_url=os.environ.get("CONCIERGE_URL", "http://localhost:8001/v1"))
    orch = Orchestrator(
        reasoner=ReasonerStub(device, latency_ms=int(os.environ.get("REASONER_LATENCY_MS", "0"))),
        concierge=concierge, voice=voice, log=log, device=device,
    )
    return create_app(orch)


app = _default_app()
