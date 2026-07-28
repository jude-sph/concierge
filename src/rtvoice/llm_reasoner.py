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

# DeviceState matches by exact equality; there is no pattern syntax. A model
# that writes one of these into "where" anyway (instead of omitting "where"
# to mean "every row") would otherwise match nothing and be read as a normal,
# honest "no matches" -- indistinguishable from a filter that just happens to
# be wrong. Rejecting it outright surfaces the real problem instead.
_WILDCARD_VALUES = ("*", "%")

SYSTEM_PROMPT = """You are the reasoning system inside a voice assistant that \
operates the memory on a phone.

You never speak to the user - a separate voice system does that. Your only job
is to turn what the user said into a PLAN over the device's tables.

OPERATIONS - the device supports these and nothing else:
  query        read rows from a table. Optional "where" filter. Read-only.
               May also take "order_by" (a field name), "descending" (true for
               newest/largest first) and "limit" (how many rows to keep).
  update       change fields on existing rows. "values" is what to set,
               "where" scopes which rows. NO "where" MEANS EVERY ROW.
  delete       remove rows. "where" scopes which rows. NO "where" EMPTIES THE
               ENTIRE TABLE - only ever do that if the user unmistakably asked
               for exactly that.
  insert       add ONE new row. "values" is the whole record. Never set "id";
               the device assigns it.
  unsupported  a real-world action with nothing on the device behind it
               (ordering a car, placing a call). NOT for something the device
               can plainly do but you lack the details for -- adding a place,
               renaming a contact and deleting a message are all supported, so
               "unsupported" for any of them states a falsehood about this
               phone. When the details are missing, use "none".
  none         chit-chat, nothing actionable, or a question ABOUT the
               system itself rather than a request to look something up.

RULES:
- One intent per action, in the order the user said them. A compound request
  ("do X and do Y") is two intents.
- A question about what the system can do, or whether it can do it - "what
  can you do?", "what are you capable of?", "how does this work?", "don't you
  have access to my calendar?", "can you see my messages?", "do you know my
  contacts?" - is a question about the SYSTEM, not a lookup of any row. It is
  "none", never "unsupported" and never a guessed table: the system genuinely
  can read and change every table below, so there is no honest "unsupported"
  answer to give, and there is no filter to invent either, because nothing was
  actually asked to be found. Only an actual request for a fact ("what's on my
  calendar", "who's Priya") is a "query".
  This matters more than it looks. A question like this is ANSWERED BY THE
  CONVERSATION, and it only gets there if you return "none" - anything you
  return instead is spoken to the person INSTEAD of an answer. Asked "what are
  you capable of?", a guessed contacts lookup made the system reply "there are
  11 contacts: Ju, Marcus, Aisha, and 8 more", which answers nothing and is a
  non sequitur. When a sentence is about YOU rather than about the data,
  return "none" and let the conversation handle it.
- Sending a message or making a booking IS an insert into the table that
  records it. Only use "unsupported" when no table could hold the result.
- "table" must be one of the tables below. Every key in "where" and "values"
  must be a field that exists on that table, and every value must be a plain
  string, number or boolean, copied in the exact form the schema shows.
- A FIELD BELONGS TO ONE TABLE ONLY. Check the field you are about to use
  against the table you chose, in the schema below, every time. Two tables
  naming a person do not name them the same way, and the schema lists a
  person's full name as a value of "contact" - which is a field on MESSAGES.
  Writing that onto "contacts" (where people are stored as "first_name" and
  "last_name") matches nothing, and the person is then told their contact
  does not exist while they are looking at it.
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
  This includes "set/change ALL my X to Y": Y is the new value, so it goes in
  "values" only - never copy it into "where" too.
- Filters match by EXACT equality only. There is no *, %, LIKE, or range
  syntax on this device - never invent one. To act on every row in a table,
  OMIT "where" entirely (or use {}); that is the only correct way to mean
  "all of them".
- A "where" value is always a LITERAL value that could appear in the data. A
  word like "latest", "recent", "newest", "last" or "any" is NOT such a value
  and must never be written into a filter - it matches nothing, and the user
  is then told there are no records when there are. "The latest X" is said
  with "order_by" + "descending" + "limit", and "order_by"/"limit" work on
  "query" only. If a superlative is asked for on an update or delete, narrow
  it with "where" instead, or use "none".
- NEVER count anything, and never say how many records are affected. You do
  not know, and the system computes it itself from the device.
- If you cannot tell what was meant, use "none". Do not guess at a write.
- NEVER invent a value the user did not say. Every string in "values" and
  "where" has to have come out of their mouth or be resolvable from the
  schema/dates given here. Asked "can you add a place?" - which names no
  place - the only correct answers are "none"; filling in a plausible
  {"name": "New York Deli", "cuisine": "american"} writes a record that
  describes nothing real and that the person never asked for, and they will
  be told it was added. An offer to act is not an instruction to act: "can
  you add a place", "can you delete?", "could you rename someone" are asking
  whether the system CAN, and the answer is a conversation, not a write. Use
  "none" for those - NOT "unsupported", which would say the phone cannot do
  it, and the phone plainly can.
- Earlier turns are there to resolve REFERENCES ("do it for her too", "the
  same for Marcus"), and for nothing else. A write must name its target and
  its new value in THIS turn, or inherit them from a turn that plainly
  continues it. "Please change the contact" on its own names neither which
  contact nor what to change, so it is "none" - never the previous turn's
  plan repeated with its old table and old values. Repeating a plan the user
  did not ask for again is the worst thing you can do: it stages a write
  against records they were not talking about.

EXAMPLES (table names and dates below are illustrative, not the real device):

  "rename Priya to Jude Hawrani" ->
  {"intents": [{"operation": "update", "table": "contacts",
    "where": {"first_name": "Priya"},
    "values": {"first_name": "Jude", "last_name": "Hawrani"},
    "understood_as": "rename Priya to Jude Hawrani"}]}
  Note: "where" is Priya's CURRENT name. "Jude" and "Hawrani" are the NEW
  values and appear ONLY in "values", never in "where".

  "delete the contact Sarah Chen" ->
  {"intents": [{"operation": "delete", "table": "contacts",
    "where": {"first_name": "Sarah", "last_name": "Chen"},
    "understood_as": "delete the contact Sarah Chen"}]}
  Note: a full name on the CONTACTS table is TWO fields. {"contact": "Sarah
  Chen"} is the messages table's way of naming a person and matches no
  contact at all.

  "delete Sarah Chen's messages" ->
  {"intents": [{"operation": "delete", "table": "messages",
    "where": {"contact": "Sarah Chen"},
    "understood_as": "delete Sarah Chen's messages"}]}
  Note: the same person, a different table, therefore a different field.

  "change my work contacts to Hans" ->
  {"intents": [{"operation": "update", "table": "contacts",
    "where": {"group": "work"}, "values": {"first_name": "Hans"},
    "understood_as": "rename work contacts to Hans"}]}

  "set all my contacts to Hans" ->
  {"intents": [{"operation": "update", "table": "contacts",
    "values": {"first_name": "Hans"},
    "understood_as": "rename all contacts to Hans"}]}
  Note: EVERY contact, not some of them, so "where" is OMITTED entirely -
  not filled with "Hans" (that is the new value, it belongs only in
  "values") and not filled with a wildcard (there is no such syntax).
  Omitting "where" IS how "all rows" is said.

  "delete all my messages" ->
  {"intents": [{"operation": "delete", "table": "messages",
    "understood_as": "delete all messages"}]}
  Note: same pattern for the most destructive operation there is - "where"
  omitted, never invented.

  "what's in my calendar tomorrow" ->
  {"intents": [{"operation": "query", "table": "calendar",
    "where": {"day": "2026-07-28"},
    "understood_as": "look up tomorrow's calendar"}]}

  "what's my latest message?" ->
  {"intents": [{"operation": "query", "table": "messages",
    "order_by": "sent", "descending": true, "limit": 1,
    "understood_as": "look up the most recent message"}]}
  Note: "latest" describes an ORDER, not a value any row holds. Writing
  {"where": {"sent": "latest"}} matches nothing and reports "no messages",
  which is a false statement about the device. Note also there is NO "where"
  at all: the ordering already picks the row out, and adding {"sent": today}
  on top of it would find nothing on any day the person happened not to be
  messaged. Only filter when the user actually named something to filter BY.

  "read me the last three messages from Marcus" ->
  {"intents": [{"operation": "query", "table": "messages",
    "where": {"contact": "Marcus Webb"},
    "order_by": "sent", "descending": true, "limit": 3,
    "understood_as": "look up Marcus's three most recent messages"}]}

  "what are you capable of?" ->
  {"intents": [{"operation": "none",
    "understood_as": "asked what the system can do"}]}
  Note: about the SYSTEM, not about any row. No table, no filter. Returning a
  contacts query here made the system answer "there are 11 contacts", which is
  not an answer to the question that was asked.

  "hello" / "thanks, that's great" ->
  {"intents": [{"operation": "none", "understood_as": "chit-chat"}]}

  "don't you have access to my calendar?" ->
  {"intents": [{"operation": "none",
    "understood_as": "asked whether the system can read the calendar"}]}
  Note: nothing was actually asked to be found, so there is no table and no
  filter to invent - this asks ABOUT the system, not for a fact from it.
  "none" here is not "the system can't do this"; it just means this
  particular utterance is not itself a lookup or a write.

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
              "order_by": "<field or null>",   // query only
              "descending": true|false,         // query only
              "limit": <number or null>,        // query only
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
                    "order_by": {"type": ["string", "null"]},
                    "descending": {"type": "boolean"},
                    "limit": {"type": ["integer", "null"]},
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
    # Ordering and truncation, for QUERY ONLY. Without these there is no way
    # to say "my latest message", and the planner did the only thing left open
    # to it: it invented a filter value, emitting
    # `where={"sent": "latest"}` -- which matches no row, and is then reported
    # as an honest "no messages matched" that is indistinguishable from the
    # user simply having no messages.
    #
    # Deliberately NOT available on update/delete. `delete ... limit 1` reads
    # as a safe, narrow operation while actually meaning "delete whichever row
    # happened to sort first", and which row that is depends on data the user
    # never saw. Narrowing a destructive write stays the job of `where`, whose
    # blast radius is stated back to the person before anything happens.
    order_by: Optional[str] = None
    descending: bool = False
    limit: Optional[int] = None


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


def _sort_key(value):
    """Order mixed-typed column values without raising.

    Rows are model-editable JSON, so one column can end up holding a string in
    one row and a number in another, and a bare `sorted(key=...)` on that
    raises TypeError mid-request. Missing values sort first (ascending), which
    puts them last for the "newest" queries this exists to serve.
    """
    if value is None:
        return (0, 0.0, "")
    if isinstance(value, bool):
        return (1, float(value), "")
    if isinstance(value, (int, float)):
        return (1, float(value), "")
    return (2, 0.0, str(value))


def _ordered(rows: list[dict], order_by: str | None, descending: bool,
             limit: int | None) -> list[dict]:
    """Sort and truncate query results.

    ISO-8601 dates and timestamps -- the format every date field on this
    device uses -- sort correctly as plain strings, which is why this needs no
    date parsing and cannot misparse one.
    """
    if order_by:
        rows = sorted(rows, key=lambda r: _sort_key(r.get(order_by)),
                      reverse=descending)
    if limit is not None and limit > 0:
        rows = rows[:limit]
    return rows


def _describe_scope(where: dict | None, order_by: str | None,
                    descending: bool, limit: int | None) -> str:
    """The `understood_as` tail for a query, in the panel's own shorthand."""
    parts = [f"where {_describe_where(where)}"]
    if order_by:
        parts.append(f"{'newest' if descending else 'oldest'} by {order_by}")
    if limit:
        parts.append(f"top {limit}")
    return ", ".join(parts)


# Tables whose name is not itself a countable noun. "calendar" was previously
# left unchanged on the grounds that it reads for one entry or many, which
# produced "there are 6 calendar" out loud. A table absent from here falls
# back to the plural-name heuristic below.
_TABLE_NOUNS = {"calendar": ("calendar entry", "calendar entries")}


def _noun(table: str, n: int) -> str:
    """`table` as a countable noun, singular for exactly one row.

    Most tables here are named for their plural ("contacts", "messages",
    "places") and just lose the trailing s. A table this gets wrong only
    reads a little oddly -- it never touches n, the one number that must
    always be exactly right.
    """
    if table in _TABLE_NOUNS:
        singular, plural = _TABLE_NOUNS[table]
        return singular if n == 1 else plural
    if n == 1 and table.endswith("s") and not table.endswith("ss"):
        return table[:-1]
    return table


# Fields that read better as a preposition ("from Marcus Webb", "sent
# 2026-07-26") than as a bare adjective in front of the noun. Anything not
# listed here (e.g. "group", "cuisine") is spoken as a plain adjective
# instead ("2 work contacts", "3 chinese places") -- that fallback is safe
# for an unrecognised field name; it just reads a little flatter.
_FIELD_PREPOSITIONS = {
    "contact": "from", "sender": "from", "recipient": "to",
    "area": "in", "location": "in",
    "day": "on", "date": "on", "sent": "sent",
}

# Identity-shaped clauses ("from Marcus Webb") read more naturally right
# after the noun than a trailing date does, so they are surfaced first when
# a filter carries more than one prepositional clause.
_PREPOSITION_PRIORITY = ("contact", "sender", "recipient", "name",
                        "area", "location", "day", "date", "sent")


def _scope_phrase(table: str, where: dict | None, n: int) -> str:
    """The blast radius, read as an English noun phrase -- no `where`, no
    `=`, no quoted field names, no "rows in <table>". `n` is always counted
    on the device and is never allowed to go unsaid: "all 11 contacts" is
    fine, "several contacts" is not.
    """
    noun = _noun(table, n)
    if not where:
        return f"all {n} {noun}"

    adjectives: list[str] = []
    clauses: list[tuple[str, str]] = []
    for key, value in where.items():
        sv = _spoken(value)
        prep = _FIELD_PREPOSITIONS.get(key)
        if prep:
            clauses.append((key, f"{prep} {sv}"))
        else:
            adjectives.append(sv)
    clauses.sort(key=lambda kc: _PREPOSITION_PRIORITY.index(kc[0])
                 if kc[0] in _PREPOSITION_PRIORITY else len(_PREPOSITION_PRIORITY))

    prefix = " ".join(adjectives) + " " if adjectives else ""
    phrase = f"{n} {prefix}{noun}"
    if clauses:
        phrase += " " + " ".join(clause for _, clause in clauses)
    return phrase


def _rendered_update(table: str, where: dict | None, values: dict, n: int,
                     *, past: bool) -> tuple[str, str]:
    """The spoken update, as (action, count_clause).

    Renaming (the field a user is scoping BY is also the one whose new
    value they're setting -- "rename Priya to Jude") is the shape almost
    every real update takes on a phone, so it gets a dedicated phrasing.
    When that also happens to identify a single record, the filter's own
    value ("Priya") reads better than a bare count -- but the count would
    then be nowhere in the sentence at all, and it must never simply go
    missing, so it is stated once more as its own short clause. Every other
    shape states the count inline, via `_scope_phrase`, and needs no
    separate clause.
    """
    shared = [k for k in values if where and k in where]
    rename = "renamed" if past else "rename"

    if shared and n == 1:
        # The filter's own value ("Priya") IS the target -- no scope phrase
        # needed, but its count then appears nowhere else, so it is stated
        # once more as its own short clause.
        target = _spoken(where[shared[0]])
        new = " ".join(_spoken(v) for v in values.values())
        return f"{rename} {target} to {new}", f" {n} {_noun(table, n)}."
    if shared or len(values) == 1:
        new = " ".join(_spoken(v) for v in values.values())
        return f"{rename} {_scope_phrase(table, where, n)} to {new}", ""

    update = "updated" if past else "update"
    new = ", ".join(f"{k.replace('_', ' ')} {_spoken(v)}" for k, v in values.items())
    return f"{update} {_scope_phrase(table, where, n)}: {new}", ""


def _capitalized(s: str) -> str:
    """`s` with its first character upper-cased -- for turning an action
    description (always written lower-case, since it is also spoken mid
    sentence elsewhere) into the start of a standalone question."""
    return s[:1].upper() + s[1:] if s else s


def _confirm_question(action: str, count_clause: str = "") -> str:
    """A destructive write's confirmation, phrased as one short question a
    person would actually ask -- never "This will X. Confirm?", which reads
    like a log line, not speech. A live session had a user hear exactly that
    read aloud, internals-adjacent phrasing and all:
    "This will rename Sarah to Michael. 1 contact. Confirm?"

    `action` already states the write and, via `_scope_phrase`, its blast
    radius -- except in the one shape that can't: renaming a single record
    identified by name reads best as "rename Sarah to Michael", which has no
    room left for the count the user's consent depends on. That count is
    carried separately in `count_clause` (see `_rendered_update`) and, when
    present, becomes its own short trailing sentence ("That's 1 contact.")
    rather than a second clause dangling off the same one. Either way, the
    exact count is never dropped, softened, or reworded here -- only how the
    words around it are arranged changes.
    """
    question = f"{_capitalized(action)}?"
    if count_clause:
        question += f" That's {count_clause.strip().rstrip('.')}."
    return question


def _delete_confirm_question(table: str, where: dict | None, n: int) -> str:
    """Same idea as `_confirm_question`, for delete -- which must also never
    let an unfiltered delete (the most destructive thing this device can do)
    read as anything vaguer than "empties it completely", in plain words,
    not a technical aside.
    """
    emptied = ", leaving nothing" if not where else ""
    return f"{_capitalized(f'delete {_scope_phrase(table, where, n)}{emptied}')}?"


def _spoken(value: Any) -> str:
    """Like `_fmt`, but for text that is read aloud rather than echoed as a
    quoted literal: a spoken sentence has no use for JSON-style quote marks
    or Python's capitalised bool repr.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _speakable_fields(row: dict) -> dict:
    """The fields of `row` worth saying aloud, in the device's own order.

    Three kinds of field are dropped, because saying them adds noise, not
    information: the internal id (never meaningful spoken - "id 4" identifies
    nothing a listener can use); anything empty; and a *false* boolean, which
    is a field's silent default state, not an event worth reporting ("saved
    false" reads like a fourth fact about the place when it is really the
    absence of one). A *true* boolean is kept - "read true" / "saved true" is
    exactly the positive fact a person would ask about.
    """
    out = {}
    for k, v in row.items():
        if k == "id" or v is None or v == "" or v == []:
            continue
        if isinstance(v, bool) and not v:
            continue
        out[k] = v
    return out


def _row_summary(row: dict) -> str:
    """One row, read as a short spoken clause: field names in words ("cuisine
    chinese"), not a silent positional dump ("chinese") that only makes sense
    if you already know the column order.

    Used for update/delete/insert results, which are framed by their own
    sentence ("added to messages: ...") that this clause slots into. Query
    results use `_row_sentence` below instead -- a query is not framed by any
    surrounding sentence of its own, so the row itself has to read as one.
    """
    fields = _speakable_fields(row)
    return ", ".join(f"{k} {_spoken(v)}" for k, v in fields.items())


def _row_headline(row: dict) -> str:
    """Just enough to pick this row out of a list of several - its first
    speakable value (typically a name or title), not every column. Reading
    every field of every match is a table read aloud one cell at a time;
    naming the first few by their headline value and saying how many there
    are is what a person would actually say.
    """
    fields = _speakable_fields(row)
    if not fields:
        return "a row"
    return _spoken(next(iter(fields.values())))


# Fields that read as a predicate ("rated 4.5") rather than a bare adjective
# or a prepositional clause. Anything not listed here, and not in
# _FIELD_PREPOSITIONS, falls back to a plain "key value" clause in
# _row_sentence -- flatter, but never a silent positional dump.
_PREDICATE_FIELDS = {"rating": "rated"}


def _row_sentence(table: str, row: dict) -> str:
    """One row, read as a full spoken SENTENCE naming it -- "Golden Lotus is
    a chinese place in Soho, rated 4.5." -- never a field-by-field dump
    ("name Golden Lotus, cuisine chinese, area Soho, rating 4.5") that only
    makes sense read off a table. This is the single-match rendering for a
    query result; `_row_summary` above (a bare clause, not a sentence) is for
    update/delete/insert, which already state their own sentence around it.

    The row's first speakable field is treated as its subject (a name or
    title); everything else becomes either a short adjective, a
    prepositional clause (reusing the same _FIELD_PREPOSITIONS table
    `_scope_phrase` uses, so "in Soho"/"from Marcus" read the same way here
    as they do in a confirmation), or a predicate clause, so the description
    is always a sentence and never a bare list of "key value" pairs.
    """
    fields = _speakable_fields(row)
    if not fields:
        return f"a {_noun(table, 1)} with nothing recorded."

    items = list(fields.items())
    # "first_name" + "last_name" are one identity, not two facts -- kept as
    # a single subject ("Sarah Chen") rather than turning the surname into a
    # nonsensical adjective ("a Chen contact").
    if len(items) >= 2 and items[0][0] == "first_name" and items[1][0] == "last_name":
        subject = f"{_spoken(items[0][1])} {_spoken(items[1][1])}"
        rest = items[2:]
    else:
        subject = _spoken(items[0][1])
        rest = items[1:]

    if not rest:
        return f"{subject}."

    adjectives: list[str] = []
    clauses: list[str] = []
    predicates: list[str] = []
    for key, value in rest:
        sv = _spoken(value)
        if key in _PREDICATE_FIELDS:
            predicates.append(f"{_PREDICATE_FIELDS[key]} {sv}")
        elif key in _FIELD_PREPOSITIONS:
            clauses.append(f"{_FIELD_PREPOSITIONS[key]} {sv}")
        elif isinstance(value, bool):
            # A true boolean not otherwise named -- state it as a flag.
            predicates.append(key.replace("_", " "))
        elif (isinstance(value, str) and len(value) <= 20
              and "@" not in value and not any(c.isdigit() for c in value)):
            # Short, plain words ("chinese", "work") read naturally as a
            # bare adjective right in front of the noun.
            adjectives.append(sv)
        else:
            # Anything identifier-shaped (a phone number, an email, a long
            # free-text field) does not read as an adjective -- state it as
            # its own named clause instead.
            predicates.append(f"{key.replace('_', ' ')} {sv}")

    prefix = " ".join(adjectives) + " " if adjectives else ""
    sentence = f"{subject} is a {prefix}{_noun(table, 1)}"
    if clauses:
        sentence += " " + " ".join(clauses)
    if predicates:
        sentence += ", " + " and ".join(predicates)
    return sentence + "."


def _describe_hits(table: str, rows: list[dict]) -> str:
    """The query result, read as something a person would actually say --
    never the SQL-shaped "found N match in <table>: field, field, field" a
    live session produced. Zero rows says so plainly; one row is rendered as
    a full sentence about that row; several rows are named (the first few,
    by headline) plus an exact count -- never every field of every row.
    """
    n = len(rows)
    noun = _noun(table, n)
    if n == 0:
        return f"no {noun} matched."
    if n == 1:
        return _row_sentence(table, rows[0])
    head = ", ".join(_row_headline(r) for r in rows[:3])
    more = f", and {n - 3} more" if n > 3 else ""
    # "there are N <noun>" is the same idiom _scope_phrase already uses for a
    # write's blast radius ("all N contacts") -- a real sentence, and one
    # that reads correctly for every table name this device has, including
    # "calendar" (see _noun's docstring: it is deliberately left unchanged
    # for both one entry and many).
    return f"there are {n} {noun}: {head}{more}."


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
        # How often a plan named a field its table does not have and had to be
        # sent back. Worth watching: it is the single most common way this
        # model misreads the device, and each one costs a second round trip.
        self.field_corrections = 0
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
                # A plan is compact JSON, not prose: even a compound plan of
                # several intents (each with a short "where"/"values" and a
                # one-line "understood_as") comfortably fits in a fraction of
                # the old 400 -- and a lower ceiling caps how much latency one
                # slow generation can spend before the reasoner even starts.
                "max_tokens": 250,
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
            complaint = self._field_complaint(plan)
            if complaint is None:
                return plan
            # The plan parsed but names a field its table does not have, which
            # this model does repeatedly and which no amount of prompting has
            # fixed: told plainly that a person on `contacts` is
            # first_name + last_name, and shown two worked examples, a 3B
            # still writes {"contact": "Sarah Chen"} -- because the schema
            # lists exactly that string as a value of `messages.contact`.
            #
            # So it is corrected the same way malformed JSON already is: hand
            # back the specific mistake and let it try again. That is general
            # (any wrong field on any table, not a rule about names) and it
            # gives the model the one fact it demonstrably lacks at the moment
            # of generating -- WHICH field it got wrong, and what the real
            # ones are.
            self.field_corrections += 1
            messages.append({"role": "system", "content": complaint})
        else:
            messages.append({"role": "system", "content":
                              f"That reply could not be read as a plan ({err}). "
                              "Reply with a single JSON object and nothing else."})

        content = await self._post(messages)
        retried, err = self._to_plan(content)
        if retried is not None:
            # Second attempt wins only if it is actually better; a retry that
            # reintroduces the same bad field would otherwise replace a plan
            # that at least parsed.
            if plan is None or self._field_complaint(retried) is None:
                return retried
        if plan is not None:
            return plan          # corrected attempt was no better; report the original
        raise ValueError(f"invalid plan after retry: {err}")

    def _field_complaint(self, plan: Plan) -> Optional[str]:
        """The specific "that field is not on that table" message, or None.

        Deliberately names the offending key AND lists the table's real
        fields: the schema block is already in the prompt and was ignored, so
        what is added here is not more information but a pointed correction at
        the moment it is needed.
        """
        for intent in plan.intents:
            if intent.operation not in DEVICE_OPERATIONS or not intent.table:
                continue
            fields = self._table_fields(intent.table)
            if not fields:
                continue
            for source in (intent.where or {}, intent.values or {}):
                for key in source:
                    if key not in fields:
                        return (
                            f'"{intent.table}" has no field "{key}". Its fields '
                            f'are: {", ".join(sorted(fields))}. A person on '
                            f'"contacts" is first_name plus last_name. Redo the '
                            f"plan using only fields that exist on the table you "
                            f"chose, and reply with a single JSON object."
                        )
        return None

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
            #
            # This is reserved for genuine breakdowns (the model unreachable,
            # or producing nothing plan-shaped after a retry) -- not for
            # ordinary conversation. An utterance that is actually just
            # chit-chat, or a question about the system itself, is expected
            # to come back as operation="none" (see SYSTEM_PROMPT's worked
            # example) and never reaches this branch at all: `_apply` returns
            # nothing for it, so nothing is ever spoken on the reasoner's
            # behalf and the concierge alone carries that turn. Phrased as a
            # person would say it, not as an internal error dump -- this is
            # the one message that has no fact of any kind behind it.
            self.misreads += 1
            # Start the conversation over. History is only appended to on
            # SUCCESS, so a turn that fails to plan leaves it frozen exactly
            # as it was -- and if that state is what caused the failure, every
            # subsequent turn is planned against it and fails identically,
            # forever. Measured: a session took one bad turn and then failed
            # on "Can you tell me about my calendar?", which had worked
            # minutes earlier and works from a clean history. Continuity is
            # worth a lot, but not the ability to ever answer again.
            self._history.clear()
            return [ReasonerMessage(
                kind="failed", task_id=f"t{next(_ids)}",
                reason="Sorry, I didn't catch what you wanted there.",
                understood_as=f"understand: {text}",
                status=type(exc).__name__,
            )]

        self._history.append({"role": "user", "content": text})

        out: list[ReasonerMessage] = []
        for intent in plan.intents:
            out.extend(self._apply(intent))
        if out:
            # The assistant turn is stored as the PLAN ITSELF, in the exact
            # JSON the model is asked to produce -- not as a readable summary
            # of it.
            #
            # It used to be "planned: query contacts". That put text in the
            # assistant role that did not look like the required output, and
            # a small model imitates the conversation it can see in preference
            # to an instruction further up the prompt. One or two such turns
            # were survivable; by the third the pattern won, and the model
            # replied `planned: insert into places` -- prose, unparseable --
            # to a request it handles correctly from a clean history. Because
            # history only grows on success, that then froze permanently and
            # every later turn failed the same way.
            #
            # Storing the real plan makes the history reinforce the format
            # instead of fighting it, and gives the model something it can
            # actually use: the previous filters and values, which is what
            # "do the same for Marcus" has to resolve against.
            self._history.append({
                "role": "assistant",
                "content": json.dumps(
                    {"intents": [i.model_dump(exclude_none=True) for i in plan.intents]},
                    separators=(",", ":"),
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

        Every string returned here is spoken to the user verbatim (see
        Orchestrator._speak_facts) and shown as the task's on-screen detail --
        there is no second, separate "spoken version". A live session once
        had a user hear "contacts has no contact": accurate to a developer,
        meaningless (and ungrammatical) to a listener, and it named an
        internal field ("contact") the user never typed. Every message below
        is a plain sentence built from words a person actually said or would
        say back -- table/field identifiers are never interpolated raw,
        except the table's own name, which is an ordinary English noun here
        ("contacts", "messages", "calendar", "places").
        """
        if not intent.table:
            return "I'm not sure what that should apply to."
        fields = self._table_fields(intent.table)
        if fields is None:
            return f"there's no {intent.table} on this device"

        noun = _noun(intent.table, 1)
        for source, is_where in ((intent.where or {}, True), (intent.values or {}, False)):
            for key, value in source.items():
                # An insert into a table with no rows yet has no known fields,
                # so anything scalar is allowed; otherwise rows stay uniform.
                if fields and key not in fields:
                    if is_where:
                        # NOT "I couldn't find a contact matching that". The
                        # device was never searched -- the filter named a
                        # field this table does not have, which is a mistake
                        # on our side. Reporting it as absence is a false
                        # statement about the phone, and the person hears it
                        # while looking at the record on screen.
                        return (f"I couldn't work out how to look that {noun} "
                                "up -- try saying it another way?")
                    return f"that's not something I can set on a {noun}."
                if value is not None and not isinstance(value, _SCALARS):
                    if is_where:
                        return f"I can't match a {noun} on that kind of value."
                    return f"that's not a value I can set on a {noun}."
                if is_where and value in _WILDCARD_VALUES:
                    return (f"a wildcard can't be used to filter {noun}s -- "
                            "leaving the filter out matches everyone instead")

        if intent.order_by is not None:
            if intent.operation != "query":
                # See the note on Intent.order_by: ordering a destructive write
                # would let "delete the oldest one" pick a row by criteria the
                # person never saw.
                return f"I can only sort {noun}s when looking them up."
            if fields and intent.order_by not in fields:
                return f"{noun}s aren't sorted by that."
        if intent.limit is not None:
            if intent.operation != "query":
                return f"I can only take the first few {noun}s when looking them up."
            if not isinstance(intent.limit, int) or intent.limit < 1:
                return "that's not a number of results I can take."

        if intent.operation == "update" and not intent.values:
            return "there's nothing to change."
        if intent.operation == "insert" and not (intent.values or {}):
            return "there's nothing to add."
        if intent.operation == "insert" and intent.where:
            # An insert scoped by a filter is incoherent. It usually means the
            # model meant `update`, and guessing which would be a write nobody
            # asked for.
            return f"I can't add a new {noun} while also filtering for existing ones."
        if intent.operation == "delete" and intent.values:
            # Likewise: a delete carrying new field values is almost certainly
            # a botched update, and the two differ by everything.
            return f"I can't delete a {noun} and change it at the same time."
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
            rows = _ordered(self.device.query(table, where),
                            intent.order_by, intent.descending, intent.limit)
            scope = _describe_scope(where, intent.order_by, intent.descending,
                                    intent.limit)
            return [
                ReasonerMessage(kind="ack", task_id=tid,
                                understood_as=f"look up {table} {scope}"),
                ReasonerMessage(kind="done", task_id=tid,
                                result=_describe_hits(table, rows)),
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
                    # Spoken, so no quotes, no "row to <table>": the same
                    # unquoted "field value" style already used for reading a
                    # matched row aloud (_row_summary). An insert never has a
                    # model-claimed count to distrust -- it is always exactly
                    # one record -- so there is no count to state here. One
                    # short question, not "This will X. Confirm?".
                    verbatim_text=f"Add to {table}: {_row_summary(values)}?",
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
            nothing = f"no {_noun(table, 0)} matched, so nothing changed"
            action, count_clause = _rendered_update(table, where, values, n, past=False)
            question = _confirm_question(action, count_clause)
        else:
            understood = f"delete from {table} where {_describe_where(where)}"
            nothing = f"no {_noun(table, 0)} matched, so nothing was deleted"
            # An unfiltered delete is the most destructive thing this device
            # can do, and must not be describable as anything vaguer than what
            # it is -- "leaving nothing" alongside "all N <noun>" says so
            # twice over, in plain words, with no SQL-ish syntax.
            question = _delete_confirm_question(table, where, n)

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
            return f"added to {table}: {_row_summary(created)}"

        if plan["op"] == "delete":
            n = self.device.delete(table, where, token=token)
            return f"deleted {n} {_noun(table, n)}"

        n = self.device.update(table, values, where, token=token)
        action, count_clause = _rendered_update(table, where, values, n, past=True)
        # count_clause, when present, is its own trailing sentence ("
        # 1 contact.") and needs a full stop after `action` to separate the
        # two; when absent, `action` alone is the whole (unpunctuated) result.
        return f"{action}.{count_clause}" if count_clause else action

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
