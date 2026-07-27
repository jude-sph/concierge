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
import datetime as dt
import itertools
import json
from typing import Any, Callable, Literal, Optional

import httpx
from pydantic import BaseModel

from .cancellation import CancellationToken
from .device import DeviceState
from .protocol import OrchestratorMessage, ReasonerMessage
from .reasoner_stub import confirmation_decision

_ids = itertools.count(1)

# The four things DeviceState can do, plus two honest outcomes for requests
# that are not device operations at all. Nothing else is representable, so
# nothing else can be planned.
DEVICE_OPERATIONS = ("query", "update", "delete", "insert")
OPERATIONS = DEVICE_OPERATIONS + ("unsupported", "none")

# Everything that mutates. Each one stops at confirm_required with a true
# count, and writes only after an affirmative answer.
DESTRUCTIVE = ("update", "delete", "insert")

# Scalars only. DeviceState matches by equality, so a list or a nested object
# as a filter value can never match anything it stores -- accepting one would
# only produce a confidently wrong row count.
_SCALARS = (str, int, float, bool)

SYSTEM_PROMPT = """You are the reasoning system inside a voice assistant that \
operates the memory on a phone.

You never speak to the user - a separate voice system does that. Your only job
is to turn what the user said into a PLAN over the device's tables.

OPERATIONS - the device supports these and nothing else:
  query        read rows from a table. Optional "where" filter. Read-only.
  update       change fields on existing rows. "values" is what to set,
               "where" scopes which rows. NO "where" MEANS EVERY ROW.
  delete       remove rows. "where" scopes which rows. NO "where" EMPTIES THE
               ENTIRE TABLE - only ever do that if the user unmistakably asked
               for exactly that.
  insert       add ONE new row. "values" is the whole record. Never set "id";
               the device assigns it.
  unsupported  a real-world action with nothing on the device behind it
               (ordering a car, placing a call).
  none         chit-chat, or nothing actionable.

RULES:
- One intent per action, in the order the user said them. A compound request
  ("do X and do Y") is two intents.
- Sending a message or making a booking IS an insert into the table that
  records it. Only use "unsupported" when no table could hold the result.
- "table" must be one of the tables below. Every key in "where" and "values"
  must be a field that exists on that table, and every value must be a plain
  string, number or boolean, copied in the exact form the schema shows.
- Resolve relative dates ("tomorrow", "Friday") into the exact date format the
  schema shows, using the dates given below. Never write a relative word into
  a filter.
- Scope a change as narrowly as the user actually asked. If they named one
  person, filter to that person. If a later clause narrows an earlier one
  ("delete yesterday's messages, just the ones from Marcus"), plan ONE intent
  with BOTH filters.
- In an update, "where" and "values" answer different questions: "where" is
  WHICH rows to find, described by their CURRENT values; "values" is what to
  set on them. A new value must NEVER appear in "where" - if it does, the
  update will match nothing, because the row does not have the new value yet.
- NEVER count anything, and never say how many records are affected. You do
  not know, and the system computes it itself from the device.
- If you cannot tell what was meant, use "none". Do not guess at a write.

EXAMPLES (table names and dates below are illustrative, not the real device):

  "rename Priya to Jude Hawrani" ->
  {"intents": [{"operation": "update", "table": "contacts",
    "where": {"first_name": "Priya"},
    "values": {"first_name": "Jude", "last_name": "Hawrani"},
    "understood_as": "rename Priya to Jude Hawrani"}]}
  Note: "where" is Priya's CURRENT name. "Jude" and "Hawrani" are the NEW
  values and appear ONLY in "values", never in "where".

  "change my work contacts to Hans" ->
  {"intents": [{"operation": "update", "table": "contacts",
    "where": {"group": "work"}, "values": {"first_name": "Hans"},
    "understood_as": "rename work contacts to Hans"}]}

  "what's in my calendar tomorrow" ->
  {"intents": [{"operation": "query", "table": "calendar",
    "where": {"day": "2026-07-28"},
    "understood_as": "look up tomorrow's calendar"}]}

  "delete yesterday's messages from Marcus" ->
  {"intents": [{"operation": "delete", "table": "messages",
    "where": {"sent": "2026-07-26", "contact": "Marcus Webb"},
    "understood_as": "delete yesterday's messages from Marcus"}]}

  "add a dentist appointment tomorrow at 4:30pm" ->
  {"intents": [{"operation": "insert", "table": "calendar",
    "values": {"title": "dentist", "day": "2026-07-28", "when": "2026-07-28T16:30"},
    "understood_as": "add a dentist appointment tomorrow"}]}

  "book a table for four and text Sarah about it" ->
  {"intents": [
    {"operation": "insert", "table": "calendar",
     "values": {"title": "table for four", "day": "2026-07-28"},
     "understood_as": "book a table for four"},
    {"operation": "insert", "table": "messages",
     "values": {"contact": "Sarah Chen", "body": "table booked for four"},
     "understood_as": "text Sarah about the table"}]}

Reply with a single JSON object and nothing else, shaped exactly like the
examples above:
{"intents": [{"operation": "query|update|delete|insert|unsupported|none",
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

    operation: Literal["query", "update", "delete", "insert", "unsupported", "none"]
    table: Optional[str] = None
    where: Optional[dict[str, Any]] = None
    values: Optional[dict[str, Any]] = None
    understood_as: str = ""


class Plan(BaseModel):
    intents: list[Intent] = []


def _extract_json(content: str) -> Any:
    """Parse `content` as JSON, tolerating prose or markdown fences around the
    object it contains.

    A real model does not reliably emit a bare JSON object even when told
    to -- "Sure, here's the plan:\\n```json\\n{...}\\n```" is a JSONDecodeError
    under a plain `json.loads`, even though the object itself is well-formed.
    So on a first-pass failure, this scans for the first balanced `{...}`
    (quote-aware, so a brace inside a string value cannot desync the count)
    and parses that instead. If no such object exists, the original
    JSONDecodeError propagates -- there is nothing here worth guessing at.
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    start = content.find("{")
    if start == -1:
        return json.loads(content)  # re-raise the original error

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(content)):
        ch = content[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(content[start:i + 1])
    return json.loads(content)  # unbalanced -- re-raise the original error


# --- rendering ---------------------------------------------------------------
#
# Everything the user will hear about a task is built here, from device data.
# None of it is generated text.


def _fmt(value: Any) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool):
        # These get spoken aloud; "read True" is a Python repr, not English.
        return "true" if value else "false"
    return str(value)


def _describe_values(values: dict) -> str:
    return "set " + " and ".join(f"{k} to {_fmt(v)}" for k, v in values.items())


def _describe_record(values: dict) -> str:
    return ", ".join(f"{k} {_fmt(v)}" for k, v in values.items())


def _describe_where(where: dict | None) -> str:
    if not where:
        return "everything"
    return " and ".join(f"{k} = {_fmt(v)}" for k, v in where.items())


def _rows_word(n: int) -> str:
    return "row" if n == 1 else "rows"


def _scope(table: str, where: dict | None, n: int) -> str:
    """The blast radius, in words. `n` is always counted on the device.

    An unfiltered write says "all", because "9 rows in messages" and "every
    message you have" are the same fact but only one of them is audibly a
    whole-table operation.
    """
    if not where:
        return f"all {n} {_rows_word(n)} in {table}"
    return f"{n} {_rows_word(n)} in {table} where {_describe_where(where)}"


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


def _date_block(today: dt.date) -> str:
    """Relative dates, pre-resolved.

    "tomorrow" and "yesterday" are the two most common filters a person speaks
    and the two a model has no way to compute -- it does not know what day it
    is, and asking it to do date arithmetic in its head is exactly the kind of
    silent wrongness that ends in the wrong rows being deleted. So the
    arithmetic is done here and handed over as literals.
    """
    day = dt.timedelta(days=1)
    week = "; ".join(
        f"{(today + dt.timedelta(days=i)):%A} {(today + dt.timedelta(days=i)).isoformat()}"
        for i in range(7)
    )
    return (
        f"TODAY IS {today.isoformat()} ({today:%A}). "
        f"Tomorrow is {(today + day).isoformat()}. "
        f"Yesterday is {(today - day).isoformat()}.\n"
        f"The next seven days are: {week}."
    )


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
        now: Callable[[], dt.datetime] | None = None,
    ) -> None:
        self.device = device
        self.base_url = base_url
        self.model = model
        self.latency_ms = latency_ms
        self.history_turns = history_turns
        self.misreads = 0
        # Injectable so "tomorrow" is testable without waiting for tomorrow.
        self.now = now or dt.datetime.now
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
        wrong answer rather than a loud one. Free-text fields (every value
        distinct, or long) are deliberately not enumerated -- listing message
        bodies would be a privacy leak and a token sink for no gain.
        """
        lines = []
        for table, rows in self.device.snapshot().items():
            if not isinstance(rows, list):
                continue
            fields = self._fields_of(rows)
            lines.append(f"{table} ({len(rows)} rows): {', '.join(fields) or '(empty)'}")
            for field in fields:
                seen = [r.get(field) for r in rows if isinstance(r, dict)]
                values = {v for v in seen if isinstance(v, str)}
                if (0 < len(values) <= 6 and len(values) < len(seen)
                        and all(len(v) <= 24 for v in values)):
                    lines.append(f"    {field} is one of: "
                                 + ", ".join(_fmt(v) for v in sorted(values)))
        return "\n".join(lines) or "(the device has no tables)"

    @staticmethod
    def _fields_of(rows: list) -> list[str]:
        fields: list[str] = []
        for row in rows:
            if isinstance(row, dict):
                fields.extend(k for k in row if k not in fields)
        return fields

    def _table_fields(self, table: str) -> list[str] | None:
        rows = self.device.snapshot().get(table)
        if not isinstance(rows, list):
            return None
        return self._fields_of(rows)

    # --- the model call ------------------------------------------------------

    async def _post(self, messages: list[dict]) -> str:
        """One HTTP round trip. Transport failures (unreachable, timeout,
        non-2xx) propagate uncaught -- those are not retried; see `_complete`.
        """
        resp = await self._client.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": 400,
                # Planning is structured extraction, not conversation -- there
                # is one right answer per utterance, so less sampling noise
                # is strictly better here than in Concierge's spoken replies.
                "temperature": 0.0,
                "guided_json": PLAN_SCHEMA,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _to_plan(content: str) -> tuple[Optional[Plan], str]:
        """The plan, or "" as an error message if `content` could not be
        read as one -- not JSON at all, or JSON that does not fit the
        schema (Plan(**data) also rejects e.g. a bare list or a string)."""
        try:
            return Plan(**_extract_json(content)), ""
        except Exception as exc:
            return None, str(exc)

    async def _complete(self, text: str) -> Plan:
        """One plan. Retries once on a parse or schema failure -- following
        the same shape as `Concierge.respond`: the error is appended to the
        conversation so the model can see what it did wrong and try again.
        A transport failure (see `_post`) is not caught here and is never
        retried; only the model's own malformed output is.

        If both attempts fail, the second error is raised, and the caller
        (`_plan`) treats it exactly like an unreachable model: nothing
        written, a `failed` message, `misreads` counted once.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"DEVICE SCHEMA:\n{self.schema_block()}"},
            {"role": "system", "content": _date_block(self.now().date())},
            *self._history[-self.history_turns:],
            {"role": "user", "content": text},
        ]

        content = await self._post(messages)
        plan, err = self._to_plan(content)
        if plan is not None:
            return plan

        messages.append({"role": "system", "content":
                          f"That reply could not be read as a plan ({err}). "
                          "Reply with a single JSON object and nothing else."})
        content = await self._post(messages)
        plan, err = self._to_plan(content)
        if plan is not None:
            return plan
        raise ValueError(f"invalid plan after retry: {err}")

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

    # --- plan validation -----------------------------------------------------

    def _validate(self, intent: Intent) -> str:
        """Why this intent cannot be executed, or "" if it can.

        A model that names a table or a field the device does not have has
        misunderstood the device, not just the phrasing -- and an `update`
        with an unknown key in `values` would silently ADD that field to every
        matched row. Rejecting is the only safe reading; there is no repair
        that is not a guess.
        """
        if not intent.table:
            return "no table"
        fields = self._table_fields(intent.table)
        if fields is None:
            return f"there's no {intent.table} on this device"

        for source in (intent.where or {}, intent.values or {}):
            for key, value in source.items():
                # An insert into a table with no rows yet has no known fields,
                # so anything scalar is allowed; otherwise rows stay uniform.
                if fields and key not in fields:
                    return f"{intent.table} has no {key}"
                if value is not None and not isinstance(value, _SCALARS):
                    return f"{key} can't be matched on that"

        if intent.operation == "update" and not intent.values:
            return "nothing to change"
        if intent.operation == "insert" and not (intent.values or {}):
            return "nothing to add"
        if intent.operation == "insert" and intent.where:
            # An insert scoped by a filter is incoherent. It usually means the
            # model meant `update`, and guessing which would be a write nobody
            # asked for.
            return "can't add a row and filter at the same time"
        if intent.operation == "delete" and intent.values:
            # Likewise: a delete carrying new field values is almost certainly
            # a botched update, and the two differ by everything.
            return "can't delete and set values at the same time"
        return ""

    # --- planning ------------------------------------------------------------

    def _apply(self, intent: Intent) -> list[ReasonerMessage]:
        if intent.operation == "none":
            return []

        tid = f"t{next(_ids)}"
        said = intent.understood_as[:120] or "that"

        if intent.operation == "unsupported":
            # The paraphrase is the model's, but it is the only thing here that
            # is: it restates the user's own request and asserts no device
            # fact. The outcome is fixed and true regardless of what it says.
            return [
                ReasonerMessage(kind="ack", task_id=tid, understood_as=said),
                ReasonerMessage(kind="failed", task_id=tid,
                                reason="this phone can't do that yet"),
            ]

        problem = self._validate(intent)
        if problem:
            self.misreads += 1
            return [
                ReasonerMessage(kind="ack", task_id=tid, understood_as=said),
                ReasonerMessage(kind="failed", task_id=tid, reason=problem),
            ]

        table = intent.table or ""
        where = intent.where or None
        values = intent.values or {}

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

        if intent.operation == "insert":
            # The id is the device's to assign, never the model's.
            values = {k: v for k, v in values.items() if k != "id"}
            self._pending[tid] = {"op": "insert", "table": table,
                                  "where": None, "values": values}
            return [
                ReasonerMessage(kind="ack", task_id=tid,
                                understood_as=f"add a row to {table}: {_describe_record(values)}"),
                ReasonerMessage(
                    kind="confirm_required", task_id=tid,
                    verbatim_text=f"This will add 1 row to {table}: "
                                  f"{_describe_record(values)}. Confirm?",
                ),
            ]

        # --- update and delete: scoped, destructive, so they stop here -------
        #
        # THE count. Taken from the device, using the same filter the write
        # will use, at the moment of asking -- never from the model, which is
        # told not to count and is not believed if it does anyway. This number
        # is the entire content of the user's consent.
        n = len(self.device.query(table, where))

        if intent.operation == "update":
            understood = f"{_describe_values(values)} in {table} where {_describe_where(where)}"
            nothing = f"nothing in {table} matches {_describe_where(where)}, so nothing changed"
            question = f"This will update {_scope(table, where, n)}: {_describe_values(values)}. Confirm?"
        else:
            understood = f"delete from {table} where {_describe_where(where)}"
            nothing = f"nothing in {table} matches {_describe_where(where)}, so nothing was deleted"
            # An unfiltered delete is the most destructive thing this device
            # can do, and must not be describable as anything vaguer than what
            # it is.
            emptied = ", leaving the table empty" if not where else ""
            question = f"This will delete {_scope(table, where, n)}{emptied}. Confirm?"

        if n == 0:
            # Nothing matches, so there is no blast radius to consent to and no
            # write to make. Asking "shall I change 0 rows?" is noise.
            return [
                ReasonerMessage(kind="ack", task_id=tid, understood_as=understood),
                ReasonerMessage(kind="done", task_id=tid, result=nothing),
            ]

        self._pending[tid] = {"op": intent.operation, "table": table,
                              "where": where, "values": values}
        return [
            ReasonerMessage(kind="ack", task_id=tid, understood_as=understood),
            ReasonerMessage(kind="confirm_required", task_id=tid, verbatim_text=question),
        ]

    # --- confirmation and execution ------------------------------------------

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
            result = self._execute(plan, token)
            self.device.commit()
        except Exception:
            # Includes Cancelled: the write is staged, so rolling back undoes
            # every record already touched, not just the ones after the abort.
            self.device.rollback()
            self._forget(task_id)
            return [ReasonerMessage(kind="failed", task_id=task_id, reason="stopped partway")]

        self._forget(task_id)
        return [ReasonerMessage(kind="done", task_id=task_id, result=result)]

    def _execute(self, plan: dict, token: CancellationToken) -> str:
        """Perform the staged write and report what it actually did.

        The counts here are the device's return values, not the numbers quoted
        at confirmation time and not anything the model said: between the
        question and the answer, a cancelled sibling task could have rolled the
        working copy back underneath us.
        """
        table, where, values = plan["table"], plan["where"], plan["values"]

        if plan["op"] == "insert":
            created = self.device.insert(table, values, token=token)
            return f"added 1 row to {table}: {_row_summary(created)}"

        if plan["op"] == "delete":
            n = self.device.delete(table, where, token=token)
            return f"deleted {n} {_rows_word(n)} from {table}"

        n = self.device.update(table, values, where, token=token)
        return f"updated {n} {_rows_word(n)} in {table}: {_describe_values(values)}"

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
