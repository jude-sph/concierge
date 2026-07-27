"""Stand-in for the memory-enabled reasoner.

Rule-based intent extraction keeps the tests deterministic; an LLM roleplay
path is available behind `use_llm` for demos. Either way it mutates a real
journaled DeviceState, so writes are verifiable by diffing a file.

`latency_ms` exists so long-wait behaviour can be exercised on demand - the
stub's latency is chosen rather than discovered, and that is the point.
"""
from __future__ import annotations

import asyncio
import itertools
import re

from fastapi import FastAPI

from .cancellation import CancellationToken
from .device import DeviceState
from .protocol import OrchestratorMessage, ReasonerMessage

_ids = itertools.count(1)

# Splitting on " and " is enough for the compound commands we demo; the real
# reasoner does this properly.
_SPLIT = re.compile(r"\s+and\s+(?=(?:find|get|set|change|rename|book|text|call|delete|add|search)\b)")
_RENAME = re.compile(r"(?:set|change|rename)\s+(?:all\s+)?(?:my\s+)?contacts?.*?to\s+(\w+)", re.I)
_SEARCH = re.compile(r"(?:find|search for)\s+(.+)", re.I)


class ReasonerStub:
    def __init__(
        self,
        device: DeviceState,
        latency_ms: int = 0,
        *,
        tokens: dict[str, CancellationToken] | None = None,
    ) -> None:
        self.device = device
        self.latency_ms = latency_ms
        self._pending: dict[str, dict] = {}
        self._tokens: dict[str, CancellationToken] = {}
        # Shared with the orchestrator (which binds its own dict here after
        # construction) so a live token can be fired by direct reference,
        # with no dependency on any message being sent or received. Kept
        # separate from `_tokens` (still used for this stub's own bookkeeping,
        # e.g. `_cancel`'s lookup) so a caller-supplied dict is never
        # unexpectedly mutated with entries the caller doesn't care about
        # beyond what's explicitly published here.
        self.tokens: dict[str, CancellationToken] = tokens if tokens is not None else {}

    async def handle(self, msg: OrchestratorMessage) -> list[ReasonerMessage]:
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000)

        if msg.kind == "cancel":
            return self._cancel(msg.task_id)
        if msg.kind == "clarification_answer":
            return await self._confirm(msg.task_id, msg.text)
        if msg.kind == "nudge":
            task = self._pending.get(msg.task_id or "")
            return [ReasonerMessage(kind="progress", task_id=msg.task_id,
                                    status="still working" if task else "no such task")]
        if msg.kind != "utterance":
            return [ReasonerMessage(kind="noop")]

        out: list[ReasonerMessage] = []
        for clause in _SPLIT.split(msg.text):
            out.extend(self._plan(clause.strip()))
        return out or [ReasonerMessage(kind="noop")]

    def _plan(self, clause: str) -> list[ReasonerMessage]:
        rename = _RENAME.search(clause)
        if rename:
            name = rename.group(1)
            tid = f"t{next(_ids)}"
            n = len(self.device.query("contacts"))
            self._pending[tid] = {"op": "rename", "name": name}
            return [
                ReasonerMessage(kind="ack", task_id=tid,
                                understood_as=f"rename all contacts to {name}"),
                ReasonerMessage(
                    kind="confirm_required", task_id=tid,
                    verbatim_text=f"This will rename {n} contacts to {name}. Confirm?",
                ),
            ]

        search = _SEARCH.search(clause)
        if search:
            tid = f"t{next(_ids)}"
            query = search.group(1)
            hits = 12  # the stub does not really search
            return [
                ReasonerMessage(kind="ack", task_id=tid, understood_as=f"search: {query}"),
                ReasonerMessage(kind="done", task_id=tid, result=f"found {hits} results for {query}"),
            ]

        return []

    async def _confirm(self, task_id: str | None, answer: str) -> list[ReasonerMessage]:
        plan = self._pending.pop(task_id or "", None)
        if plan is None:
            return [ReasonerMessage(kind="noop")]

        # Default-deny: check for negations first, regardless of any affirmative words present
        if re.search(r"\b(no|nope|don't|dont|do not|not|never mind|nevermind|cancel|stop|wait|hold on|forget it)\b", answer, re.I):
            self.device.rollback()
            self._tokens.pop(task_id, None)
            self.tokens.pop(task_id, None)
            return [ReasonerMessage(kind="failed", task_id=task_id, reason="cancelled by user")]

        # Only commit if affirmative word is present
        if not re.search(r"\b(yes|yeah|yep|do it|go ahead|confirm|ok|okay)\b", answer, re.I):
            self.device.rollback()
            self._tokens.pop(task_id, None)
            self.tokens.pop(task_id, None)
            return [ReasonerMessage(kind="failed", task_id=task_id, reason="cancelled by user")]

        token = CancellationToken()
        self._tokens[task_id] = token
        # Published to the shared registry the orchestrator holds a reference
        # to, so a concurrent _abort() can fire this exact token in-process,
        # synchronously -- without waiting for (or requiring) a "cancel"
        # message to ever be sent, delivered, or processed.
        self.tokens[task_id] = token
        try:
            n = self.device.update("contacts", {"first_name": plan["name"]}, token=token)
            self.device.commit()
        except Exception:
            self.device.rollback()
            self._tokens.pop(task_id, None)
            self.tokens.pop(task_id, None)
            return [ReasonerMessage(kind="failed", task_id=task_id, reason="stopped partway")]

        self._tokens.pop(task_id, None)
        self.tokens.pop(task_id, None)
        return [ReasonerMessage(kind="done", task_id=task_id,
                                result=f"renamed {n} contacts to {plan['name']}")]

    def _cancel(self, task_id: str | None) -> list[ReasonerMessage]:
        # Only claim cancellation if there's actually a pending task
        had_pending = task_id in self._pending or (task_id or "") in self._pending
        if task_id in self._tokens:
            self._tokens[task_id].cancel()
            self._tokens.pop(task_id, None)
        self.tokens.pop(task_id, None)
        was_removed = self._pending.pop(task_id or "", None) is not None

        # Only rollback if there was an uncommitted task
        if was_removed:
            self.device.rollback()
            return [ReasonerMessage(kind="failed", task_id=task_id, reason="cancelled")]

        # Unknown or already-completed task
        return [ReasonerMessage(kind="noop")]


def create_app(device: DeviceState, latency_ms: int = 0) -> FastAPI:
    app = FastAPI()
    stub = ReasonerStub(device, latency_ms=latency_ms)
    app.state.stub = stub

    @app.post("/message")
    async def message(msg: OrchestratorMessage) -> list[ReasonerMessage]:
        return await app.state.stub.handle(msg)

    @app.post("/latency/{ms}")
    async def set_latency(ms: int) -> dict:
        app.state.stub.latency_ms = ms
        return {"latency_ms": ms}

    return app
