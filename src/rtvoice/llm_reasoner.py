"""The memory-enabled reasoner, driven by a real model instead of regexes.

`ReasonerStub` understands three regexes. This one asks a model for a
*structured plan* over the device's actual tables, then executes that plan
itself. The split matters:

    the model decides WHAT was meant;
    this module decides what is ALLOWED, and computes every fact it states.

Nothing the model says is ever spoken to the user, and nothing it claims is
ever believed. It cannot invent an operation (the plan is validated against
what `DeviceState` can actually do), it cannot name a table or field that does
not exist, it cannot commit a write (only an affirmative answer to a
confirmation does that), and above all it cannot tell the user how many
records a write will touch -- the blast radius in `confirm_required` is
counted on the device, from the same filter the write will use.

The confirmation gates are not reimplemented here. `confirmation_decision`
is imported from `reasoner_stub`, so the answer-shape test and the
negation-first default-deny that were hard-won there cannot drift.

Failure is always safe: an unreachable endpoint, a timeout, unparseable JSON,
a plan referring to something that does not exist -- all end in `failed` with
nothing written. The system never guesses at an intent it could not read.
"""
from __future__ import annotations

import asyncio
import itertools
import json
from typing import Any, Literal, Optional

import httpx
from pydantic import BaseModel, ValidationError

from .cancellation import CancellationToken
from .device import DeviceState
from .protocol import OrchestratorMessage, ReasonerMessage
from .reasoner_stub import confirmation_decision

_ids = itertools.count(1)

# What DeviceState can actually do, and nothing more. `query` and `update` map
# one-to-one onto its two methods; the other two are outcomes, not operations,
# and exist so the model has somewhere honest to put a request rather than
# forcing it into a write.
OPERATIONS = ("query", "update", "unsupported", "none")

# Scalars only. A list or a nested object as a filter value can never match
# anything DeviceState stores, and accepting one would only produce a
# confidently wrong row count.
_SCALARS = (str, int, float, bool)

SYSTEM_PROMPT = """You are the reasoning system inside a voice assistant that \
operates the memory on a phone.

You never speak to the user - a separate voice system does that. Your only job
is to turn what the user said into a PLAN over the device's tables.

OPERATIONS - the device supports these and nothing else:
  query        read rows from a table. Optional "where" filter. Read-only.
  update       change field values on rows. "values" is what to set; "where"
               scopes which rows. NO "where" MEANS EVERY ROW IN THE TABLE.
  unsupported  the user asked for a real-world action this device cannot
               perform (booking a table, calling a car, sending a message).
  none         chit-chat, or nothing actionable.

RULES:
- One intent per action, in the order the user said them. A compound request
  ("do X and do Y") is two intents.
- "table" must be one of the tables below. Every key in "where" and "values"
  must be a field that exists on that table, and every value must be a plain
  string, number or boolean, copied in the exact form the schema shows.
- Scope a change as narrowly as the user actually asked. If they named one
  person, filter to that person. Only omit "where" when they really did mean
  every row.
- NEVER count anything, and never say how many records are affected. You do
  not know, and the system computes it itself from the device.
- If you cannot tell what was meant, use "none". Do not guess at a write.

Reply with a single JSON object and nothing else:
{"intents": [{"operation": "query|update|unsupported|none",
              "table": "<table or null>",
              "where": {"<field>": "<value>"} or null,
              "values": {"<field>": "<value>"} or null,
              "understood_as": "<short paraphrase of this one action>"}]}
"""

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "intents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": list(OPERATIONS)},
                    "table": {"type": ["string", "null"]},
                    "where": {"type": ["object", "null"]},
                    "values": {"type": ["object", "null"]},
                    "understood_as": {"type": "string"},
                },
                "required": ["operation"],
            },
        }
    },
    "required": ["intents"],
}


class Intent(BaseModel):
    """One planned action. Unknown keys are ignored, so a model that helpfully
    volunteers e.g. an "affected_rows" field cannot smuggle a fact in."""

    operation: Literal["query", "update", "unsupported", "none"]
    table: Optional[str] = None
    where: Optional[dict[str, Any]] = None
    values: Optional[dict[str, Any]] = None
    understood_as: str = ""


class Plan(BaseModel):
    intents: list[Intent] = []


# --- rendering ---------------------------------------------------------------
#
# Everything the user will hear about a task is built here, from device data.
# None of it is generated text.


def _fmt(value: Any) -> str:
    return f'"{value}"' if isinstance(value, str) else str(value)


def _describe_values(values: dict) -> str:
    return "set " + " and ".join(f"{k} to {_fmt(v)}" for k, v in values.items())


def _describe_where(where: dict | None) -> str:
    if not where:
        return "everything"
    return " and ".join(f"{k} = {_fmt(v)}" for k, v in where.items())


def _rows_word(n: int) -> str:
    return "row" if n == 1 else "rows"


def _row_summary(row: dict) -> str:
    return " ".join(
        str(v) for k, v in row.items()
        if k != "id" and v not in (None, "", [])
    )


def _describe_hits(table: str, rows: list[dict]) -> str:
    n = len(rows)
    if n == 0:
        return f"no matches in {table}"
    head = "; ".join(_row_summary(r) for r in rows[:3])
    word = "match" if n == 1 else "matches"
    more = f" and {n - 3} more" if n > 3 else ""
    return f"{n} {word} in {table}: {head}{more}"


class LlmReasoner:
    """Same interface as ReasonerStub; the intent extraction is a model call.

    Attributes:
        misreads: Count of model responses that could not be used -- transport
            failures, unparseable JSON, schema violations, and plans naming
            tables or fields the device does not have. Every one of them
            mutated nothing. Tracked for the same reason Concierge.violations
            is: generation quality is a measurable property, not a vibe.
    """

    def __init__(
        self,
        device: DeviceState,
        *,
        base_url: str = "http://localhost:8001/v1",
        model: str = "Qwen/Qwen3-4B",
        latency_ms: int = 0,
        tokens: dict[str, CancellationToken] | None = None,
        timeout: float = 15.0,
        history_turns: int = 8,
    ) -> None:
        self.device = device
        self.base_url = base_url
        self.model = model
        self.latency_ms = latency_ms
        self.history_turns = history_turns
        self.misreads = 0
        self._pending: dict[str, dict] = {}
        self._tokens: dict[str, CancellationToken] = {}
        # Shared with the orchestrator, which rebinds this attribute to its own
        # dict after construction, so a live token can be fired by direct
        # reference without any "cancel" message needing to arrive. Kept
        # separate from `_tokens` for the same reason ReasonerStub keeps them
        # separate: a caller-supplied dict is never mutated with bookkeeping
        # the caller did not ask for.
        self.tokens: dict[str, CancellationToken] = tokens if tokens is not None else {}
        self._history: list[dict] = []
        self._client = httpx.AsyncClient(timeout=timeout)

    # --- device schema -------------------------------------------------------

    def schema_block(self) -> str:
        """The tables, the fields actually present on them, and the distinct
        values of small enumerated fields.

        The last part is what makes "change my WORK contacts" resolvable at
        all: `group` as a bare field name does not tell the model that "work"
        is one of its values, and a filter that matches nothing is a silently
        wrong answer rather than a loud one.
        """
        lines = []
        for table, rows in self.device.snapshot().items():
            if not isinstance(rows, list):
                continue
            fields: list[str] = []
            for row in rows:
                if isinstance(row, dict):
                    fields.extend(k for k in row if k not in fields)
            lines.append(f"{table} ({len(rows)} rows): {', '.join(fields) or '(empty)'}")
            for field in fields:
                seen = [r.get(field) for r in rows if isinstance(r, dict)]
                values = sorted({v for v in seen if isinstance(v, str) and len(v) <= 24})
                if 0 < len(values) <= 6 and len(values) < len(seen):
                    lines.append(f"    {field} values: " + ", ".join(_fmt(v) for v in values))
        return "\n".join(lines) or "(the device has no tables)"

    def _table_fields(self, table: str) -> list[str] | None:
        rows = self.device.snapshot().get(table)
        if not isinstance(rows, list):
            return None
        fields: list[str] = []
        for row in rows:
            if isinstance(row, dict):
                fields.extend(k for k in row if k not in fields)
        return fields

    # --- the model call ------------------------------------------------------

    async def _complete(self, text: str) -> Plan:
        """One plan, or an exception. Never a partial or repaired plan."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"DEVICE SCHEMA:\n{self.schema_block()}"},
            *self._history[-self.history_turns:],
            {"role": "user", "content": text},
        ]
        resp = await self._client.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": 400,
                "temperature": 0.2,
                "guided_json": PLAN_SCHEMA,
            },
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return Plan(**json.loads(content))

    # --- protocol ------------------------------------------------------------

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
        return await self._plan(msg.text)

    async def _plan(self, text: str) -> list[ReasonerMessage]:
        if not text.strip():
            return [ReasonerMessage(kind="noop")]

        try:
            plan = await self._complete(text)
        except Exception as exc:
            # Unreachable, timed out, non-2xx, not JSON, or not plan-shaped.
            # There is no safe fallback interpretation of an utterance we could
            # not read, so nothing is attempted and nothing is written -- but
            # the user is told, because silence is indistinguishable from the
            # system having ignored them.
            self.misreads += 1
            return [ReasonerMessage(
                kind="failed", task_id=f"t{next(_ids)}",
                reason="I couldn't work out what to do with that",
                understood_as=f"understand: {text}",
                status=type(exc).__name__,
            )]

        self._history.append({"role": "user", "content": text})

        out: list[ReasonerMessage] = []
        for intent in plan.intents:
            out.extend(self._apply(intent))
        if out:
            self._history.append({
                "role": "assistant",
                "content": "planned: " + "; ".join(
                    i.operation + (f" {i.table}" if i.table else "") for i in plan.intents
                ),
            })
        return out or [ReasonerMessage(kind="noop")]

    def _validate(self, intent: Intent) -> str:
        """Why this intent cannot be executed, or "" if it can.

        A model that names a table or a field the device does not have has
        misunderstood the device, not just the phrasing -- and an `update`
        with an unknown key in `values` would silently ADD that field to every
        matched row. Rejecting is the only safe reading.
        """
        if not intent.table:
            return "no table"
        fields = self._table_fields(intent.table)
        if fields is None:
            return f"there's no {intent.table} on this device"
        for source in (intent.where or {}, intent.values or {}):
            for key, value in source.items():
                if key not in fields:
                    return f"{intent.table} has no {key}"
                if not isinstance(value, _SCALARS) and value is not None:
                    return f"{key} can't be matched on that"
        if intent.operation == "update" and not intent.values:
            return "nothing to change"
        return ""

    def _apply(self, intent: Intent) -> list[ReasonerMessage]:
        if intent.operation == "none":
            return []

        tid = f"t{next(_ids)}"

        if intent.operation == "unsupported":
            # The paraphrase is the model's, but it is the only thing here that
            # is: it restates the user's own request and asserts no device
            # fact. The outcome is fixed and true regardless of what it says.
            return [
                ReasonerMessage(kind="ack", task_id=tid,
                                understood_as=intent.understood_as[:120] or "that"),
                ReasonerMessage(kind="failed", task_id=tid,
                                reason="this phone can't do that yet"),
            ]

        problem = self._validate(intent)
        if problem:
            self.misreads += 1
            return [
                ReasonerMessage(kind="ack", task_id=tid,
                                understood_as=intent.understood_as[:120] or "that"),
                ReasonerMessage(kind="failed", task_id=tid, reason=problem),
            ]

        table = intent.table or ""
        where = intent.where or None

        if intent.operation == "query":
            # Read-only: no confirmation, and the answer is the device's, row
            # for row. Nothing is staged, so there is nothing to roll back.
            rows = self.device.query(table, where)
            return [
                ReasonerMessage(kind="ack", task_id=tid,
                                understood_as=f"look up {table} where {_describe_where(where)}"),
                ReasonerMessage(kind="done", task_id=tid,
                                result=f"found {_describe_hits(table, rows)}"),
            ]

        # --- update: destructive, so it stops here ---------------------------
        values = intent.values or {}
        understood = f"{_describe_values(values)} in {table} where {_describe_where(where)}"

        # THE count. Taken from the device, using the same filter the write
        # will use, at the moment of asking -- never from the model, which is
        # told not to count and is not believed if it does anyway. This number
        # is the entire content of the user's consent.
        n = len(self.device.query(table, where))

        if n == 0:
            # Nothing matches, so there is no blast radius to consent to and
            # no write to make. Asking "shall I change 0 rows?" is noise.
            return [
                ReasonerMessage(kind="ack", task_id=tid, understood_as=understood),
                ReasonerMessage(kind="done", task_id=tid,
                                result=f"nothing in {table} matches "
                                       f"{_describe_where(where)}, so nothing changed"),
            ]

        self._pending[tid] = {"table": table, "where": where, "values": values}
        return [
            ReasonerMessage(kind="ack", task_id=tid, understood_as=understood),
            ReasonerMessage(
                kind="confirm_required", task_id=tid,
                verbatim_text=f"This will update {n} {_rows_word(n)} in {table}: "
                              f"{_describe_values(values)}. Confirm?",
            ),
        ]

    async def _confirm(self, task_id: str | None, answer: str) -> list[ReasonerMessage]:
        key = task_id or ""
        plan = self._pending.get(key)
        if plan is None:
            return [ReasonerMessage(kind="noop")]

        # The three gates, imported wholesale from reasoner_stub so there is
        # exactly one implementation of them in the system: answer-shape first
        # (a sentence carrying its own request is not an answer either way and
        # leaves the question open), then negation-first default-deny.
        verdict = confirmation_decision(answer)
        if verdict == "not_an_answer":
            # `_pending` is deliberately untouched: the task stays awaiting
            # confirmation and remains answerable.
            return [ReasonerMessage(kind="noop")]

        self._pending.pop(key, None)

        if verdict != "confirmed":
            self.device.rollback()
            self._forget(task_id)
            return [ReasonerMessage(kind="failed", task_id=task_id, reason="cancelled by user")]

        token = CancellationToken()
        self._tokens[task_id] = token
        # Published to the shared registry the orchestrator holds, so a
        # concurrent abort can fire this exact token in-process, synchronously.
        self.tokens[task_id] = token
        try:
            n = self.device.update(plan["table"], plan["values"],
                                   plan["where"], token=token)
            self.device.commit()
        except Exception:
            self.device.rollback()
            self._forget(task_id)
            return [ReasonerMessage(kind="failed", task_id=task_id, reason="stopped partway")]

        self._forget(task_id)
        return [ReasonerMessage(
            kind="done", task_id=task_id,
            result=f"updated {n} {_rows_word(n)} in {plan['table']}: "
                   f"{_describe_values(plan['values'])}",
        )]

    def _forget(self, task_id: str | None) -> None:
        self._tokens.pop(task_id, None)
        self.tokens.pop(task_id, None)

    def _cancel(self, task_id: str | None) -> list[ReasonerMessage]:
        if task_id in self._tokens:
            self._tokens[task_id].cancel()
        self._forget(task_id)
        was_pending = self._pending.pop(task_id or "", None) is not None
        if was_pending:
            self.device.rollback()
            return [ReasonerMessage(kind="failed", task_id=task_id, reason="cancelled")]
        return [ReasonerMessage(kind="noop")]

    async def aclose(self) -> None:
        await self._client.aclose()
