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
from typing import Literal

from fastapi import FastAPI

from .cancellation import CancellationToken
from .device import DeviceState
from .protocol import OrchestratorMessage, ReasonerMessage
from .states import (
    AFFIRMATIVE_PHRASES,
    AFFIRMATIVE_WORDS,
    NEGATION_PHRASES,
    NEGATION_WORDS,
    is_answer_shaped,
)

_ids = itertools.count(1)


def _vocab_re(words: set[str], phrases: set[str]) -> re.Pattern:
    """Word-boundary alternation over a vocabulary, longest phrase first."""
    parts = sorted(phrases, key=len, reverse=True) + sorted(words)
    return re.compile(r"\b(?:" + "|".join(re.escape(p) for p in parts) + r")\b", re.I)


# Both vocabularies come from states.py, which also owns the backchannel set,
# so the words the turn policy promotes to answers and the words the reasoner
# accepts as answers cannot drift apart again.
_NEGATION_RE = _vocab_re(NEGATION_WORDS, NEGATION_PHRASES)
_AFFIRMATIVE_RE = _vocab_re(AFFIRMATIVE_WORDS, AFFIRMATIVE_PHRASES)

# Splitting on " and " is enough for the compound commands we demo; the real
# reasoner does this properly.
_SPLIT = re.compile(r"\s+and\s+(?=(?:find|get|set|change|rename|book|text|call|delete|add|search)\b)")
_RENAME = re.compile(r"(?:set|change|rename)\s+(?:all\s+)?(?:my\s+)?contacts?.*?to\s+(\w+)", re.I)
_SEARCH = re.compile(r"(?:find|search for)\s+(.+)", re.I)

ConfirmVerdict = Literal["not_an_answer", "declined", "confirmed"]


def confirmation_decision(answer: str) -> ConfirmVerdict:
    """May this utterance resolve a pending destructive write, and how?

    Three gates, in this order. This function is the ONLY implementation of
    them in the system; every reasoner imports it rather than restating it,
    because the ordering *is* the safety property and two copies of it would
    eventually disagree.

    Gate 1 -- is this an ANSWER at all? Only an utterance that is essentially
    just a yes/no may resolve a pending destructive write. A sentence carrying
    its own new request is not an answer no matter what words it happens to
    contain, and must leave the question open rather than being read as either
    assent or a decline. (Before this gate, "okay so what's on my calendar
    tomorrow" renamed every contact on the device.) The caller must leave its
    pending task untouched on "not_an_answer": the task stays AWAITING_CONFIRM
    and remains answerable.

    Gate 2 -- default-deny: negations are checked first, regardless of any
    affirmative words also present ("don't do it", "that is not okay",
    "no, don't confirm it").

    Gate 3 -- only an explicit affirmative commits. Anything ambiguous ("hmm")
    declines.
    """
    if not is_answer_shaped(answer):
        return "not_an_answer"
    if _NEGATION_RE.search(answer):
        return "declined"
    if not _AFFIRMATIVE_RE.search(answer):
        return "declined"
    return "confirmed"


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
                    # One short spoken question, not "This will X. Confirm?"
                    # -- the exact count still survives verbatim.
                    verbatim_text=f"Rename all {n} contacts to {name}?",
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
        key = task_id or ""
        plan = self._pending.get(key)
        if plan is None:
            return [ReasonerMessage(kind="noop")]

        # The three gates live in confirmation_decision() above, so this
        # reasoner and the LLM one cannot drift apart on the one question that
        # matters. "not_an_answer" leaves `_pending` untouched on purpose: the
        # task stays AWAITING_CONFIRM and remains answerable.
        verdict = confirmation_decision(answer)
        if verdict == "not_an_answer":
            return [ReasonerMessage(kind="noop")]

        self._pending.pop(key, None)

        if verdict != "confirmed":
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
