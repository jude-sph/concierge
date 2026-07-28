"""LlmReasoner against a fake model endpoint.

Every test here monkeypatches the HTTP POST, so the suite stays offline and
fast -- no vLLM, no GPU, no network. What is NOT faked is everything that
matters: the JSON parsing, the schema validation, the plan validation against
the real device schema, the confirmation gates, and the device writes.

The safety tests are the point of the file. A model that lies about the blast
radius, returns garbage, or cannot be reached must never result in a mutation.
"""
import copy
import datetime as dt
import json

import httpx
import pytest

from rtvoice import llm_reasoner as llm_mod
from rtvoice.cancellation import CancellationToken
from rtvoice.device import DeviceState
from rtvoice.llm_reasoner import LlmReasoner
from rtvoice.protocol import OrchestratorMessage

TODAY = dt.datetime(2026, 7, 27, 9, 30)      # a Monday
TOMORROW = "2026-07-28"
YESTERDAY = "2026-07-26"

# A trimmed fixtures/device_state.json: same shape, small enough to assert on.
# Every filter used below selects a subset and leaves records alone.
DEVICE = {
    "contacts": [
        {"id": 1, "first_name": "Sarah", "last_name": "Chen", "group": "work"},
        {"id": 2, "first_name": "Marcus", "last_name": "Webb", "group": "work"},
        {"id": 3, "first_name": "Priya", "last_name": "Nair", "group": "family"},
        {"id": 4, "first_name": "Tom", "last_name": "Okafor", "group": "friends"},
    ],
    "messages": [
        {"id": 1, "contact": "Tom Okafor", "body": "cinema on friday?",
         "sent": "2026-07-25", "read": True},
        {"id": 2, "contact": "Marcus Webb", "body": "running late",
         "sent": YESTERDAY, "read": True},
        {"id": 3, "contact": "Marcus Webb", "body": "did you see the deck?",
         "sent": YESTERDAY, "read": False},
        {"id": 4, "contact": "Sarah Chen", "body": "dinner still on?",
         "sent": YESTERDAY, "read": True},
        {"id": 5, "contact": "Sarah Chen", "body": "sent you the address",
         "sent": "2026-07-27", "read": False},
    ],
    "calendar": [
        {"id": 1, "title": "standup", "day": TOMORROW, "when": "2026-07-28T09:00"},
        {"id": 2, "title": "dentist", "day": TOMORROW, "when": "2026-07-28T16:30"},
        {"id": 3, "title": "cinema", "day": "2026-07-31", "when": "2026-07-31T19:30"},
    ],
    "places": [
        {"id": 1, "name": "Golden Lotus", "cuisine": "chinese", "area": "Soho",
         "rating": 4.5, "saved": False},
        {"id": 2, "name": "Jade Garden", "cuisine": "chinese", "area": "Soho",
         "rating": 4.2, "saved": True},
        {"id": 3, "name": "Red Lantern", "cuisine": "chinese", "area": "Chinatown",
         "rating": 4.7, "saved": False},
        {"id": 4, "name": "Trattoria Verde", "cuisine": "italian",
         "area": "Covent Garden", "rating": 4.3, "saved": True},
    ],
}


# --- the fake endpoint -------------------------------------------------------


class FakeResponse:
    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._content}}]}


class FakeLlm:
    """Stands in for httpx.AsyncClient.post.

    Replies may be dicts (serialised to JSON for the reasoner to parse back,
    so the real parse path is exercised), raw strings (for malformed output),
    or exceptions (for transport failures). The last reply is reused once the
    scripted ones run out.
    """

    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    async def __call__(self, url, **kw):
        # `json` is taken by the keyword httpx is called with, hence **kw
        self.calls.append(kw.get("json") or {})
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, BaseException):
            raise reply
        # dicts are serialised so the reasoner parses them back itself
        return FakeResponse(reply if isinstance(reply, str) else json.dumps(reply))


def build(tmp_path, *replies, state=None):
    path = tmp_path / "device_state.json"
    path.write_text(json.dumps(state if state is not None else DEVICE))
    device = DeviceState(path, tmp_path / "journal.jsonl")
    reasoner = LlmReasoner(device, base_url="http://fake/v1", model="fake-model",
                           now=lambda: TODAY)
    fake = FakeLlm(*replies)
    reasoner._client.post = fake  # the only thing faked
    return reasoner, fake


def names(reasoner) -> list[str]:
    return [c["first_name"] for c in reasoner.device.query("contacts")]


def ids(reasoner, table) -> list[int]:
    return [r["id"] for r in reasoner.device.query(table)]


async def say(reasoner, text):
    return await reasoner.handle(OrchestratorMessage(kind="utterance", text=text))


async def answer(reasoner, tid, text):
    return await reasoner.handle(
        OrchestratorMessage(kind="clarification_answer", task_id=tid, text=text))


def kinds(msgs):
    return [m.kind for m in msgs]


def only(msgs, kind):
    return next(m for m in msgs if m.kind == kind)


def every(msgs, kind):
    return [m for m in msgs if m.kind == kind]


def plan(*intents):
    return {"intents": list(intents)}


def journal(reasoner) -> list[dict]:
    text = reasoner.device.journal_path.read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# --- interface ---------------------------------------------------------------


def test_exposes_the_same_surface_the_orchestrator_uses(tmp_path):
    reasoner, _ = build(tmp_path)
    assert reasoner.device is not None
    assert reasoner.latency_ms == 0
    assert reasoner.tokens == {}


def test_tokens_dict_can_be_shared_by_the_orchestrator(tmp_path):
    shared: dict[str, CancellationToken] = {}
    path = tmp_path / "d.json"
    path.write_text(json.dumps(DEVICE))
    r = LlmReasoner(DeviceState(path, tmp_path / "j.jsonl"),
                    base_url="http://fake/v1", model="m", tokens=shared)
    assert r.tokens is shared


def test_schema_block_lists_real_tables_fields_and_enumerated_values(tmp_path):
    reasoner, _ = build(tmp_path)
    block = reasoner.schema_block()

    assert "contacts (4 rows)" in block
    assert "first_name" in block and "group" in block
    # low-cardinality values are surfaced, which is what makes "my work
    # contacts" and "chinese in soho" resolvable into filters at all
    assert '"work"' in block and '"family"' in block
    assert '"chinese"' in block and '"Soho"' in block
    # free text is not: enumerating message bodies leaks and buys nothing
    assert "dinner still on?" not in block


def test_relative_dates_are_resolved_before_the_model_sees_them(tmp_path):
    reasoner, _ = build(tmp_path)
    block = llm_mod._date_block(TODAY.date())
    assert "TODAY IS 2026-07-27 (Monday)" in block
    assert f"Tomorrow is {TOMORROW}" in block
    assert f"Yesterday is {YESTERDAY}" in block
    assert "Friday 2026-07-31" in block


@pytest.mark.asyncio
async def test_the_prompt_carries_the_schema_and_the_dates(tmp_path):
    reasoner, fake = build(tmp_path, plan({"operation": "none"}))
    await say(reasoner, "hello")
    system = " ".join(m["content"] for m in fake.calls[0]["messages"]
                      if m["role"] == "system")
    assert "contacts (4 rows)" in system
    assert f"Yesterday is {YESTERDAY}" in system
    for op in ("query", "update", "delete", "insert"):
        assert op in system


# --- phrasings the regexes could never handle --------------------------------


@pytest.mark.asyncio
async def test_renames_a_single_record_by_name(tmp_path):
    """'please rename Priya as Jude Hawrani' -- the live-session failure."""
    reasoner, fake = build(tmp_path, plan({
        "operation": "update", "table": "contacts",
        "where": {"first_name": "Priya"},
        "values": {"first_name": "Jude", "last_name": "Hawrani"},
        "understood_as": "rename Priya",
    }))

    out = await say(reasoner, "please rename Priya as Jude Hawrani")

    assert kinds(out) == ["ack", "confirm_required"]
    assert fake.calls[0]["messages"][-1]["content"] == "please rename Priya as Jude Hawrani"
    # nothing written before confirmation
    assert names(reasoner) == ["Sarah", "Marcus", "Priya", "Tom"]

    # Fluent, not "This will rename Priya to Jude Hawrani. 1 contact.
    # Confirm?" -- the live-session wording this fix replaces -- while the
    # exact affected count still survives, verbatim.
    assert only(out, "confirm_required").verbatim_text == (
        "Rename Priya to Jude Hawrani? That's 1 contact.")

    tid = only(out, "confirm_required").task_id
    done = await answer(reasoner, tid, "yes")
    assert only(done, "done").result == "renamed Priya to Jude Hawrani. 1 contact."
    assert names(reasoner) == ["Sarah", "Marcus", "Jude", "Tom"]
    assert reasoner.device.query("contacts", {"first_name": "Jude"})[0]["last_name"] == "Hawrani"


@pytest.mark.asyncio
async def test_scoped_update_touches_only_the_filtered_rows(tmp_path):
    """'change my work contacts to Hans' -- scoped by a field value, not all."""
    reasoner, _ = build(tmp_path, plan({
        "operation": "update", "table": "contacts",
        "where": {"group": "work"}, "values": {"first_name": "Hans"},
    }))

    out = await say(reasoner, "change my work contacts to Hans")
    verbatim = only(out, "confirm_required").verbatim_text
    assert verbatim == "Rename 2 work contacts to Hans?"

    await answer(reasoner, only(out, "confirm_required").task_id, "go ahead")
    assert names(reasoner) == ["Hans", "Hans", "Priya", "Tom"]


@pytest.mark.asyncio
async def test_read_only_query_completes_without_confirmation(tmp_path):
    """'what's in my calendar tomorrow' -- a read, so no consent is needed."""
    reasoner, _ = build(tmp_path, plan({
        "operation": "query", "table": "calendar", "where": {"day": TOMORROW},
    }))

    out = await say(reasoner, "what's in my calendar tomorrow")

    assert kinds(out) == ["ack", "done"]
    assert "confirm_required" not in kinds(out)
    result = only(out, "done").result
    # the answer is the device's rows, not the model's prose, and the filter
    # genuinely excludes something
    assert "there are 2 calendar" in result
    assert "standup" in result and "dentist" in result
    assert "cinema" not in result


@pytest.mark.asyncio
async def test_scoped_search_over_places(tmp_path):
    """'find the chinese restaurants in soho' -- two filters, 2 of 4 places."""
    reasoner, _ = build(tmp_path, plan({
        "operation": "query", "table": "places",
        "where": {"cuisine": "chinese", "area": "Soho"},
    }))
    out = await say(reasoner, "find all the chinese restaurants in soho")
    result = only(out, "done").result
    assert "there are 2 places" in result
    assert "Golden Lotus" in result and "Jade Garden" in result
    assert "Red Lantern" not in result  # chinese, but not in Soho


@pytest.mark.asyncio
async def test_a_single_match_reads_as_a_sentence_not_a_record_dump(tmp_path):
    """The live-session failure: 'found 1 match in places: name Golden Lotus,
    cuisine chinese, area Soho, rating 4.5' is a raw field-by-field dump read
    aloud -- SQL-shaped, not speech. It must instead read as an actual
    sentence: 'Golden Lotus is a chinese place in Soho, rated 4.5.' The
    internal id and a false boolean (the "saved" flag) must not be spoken at
    all."""
    reasoner, _ = build(tmp_path, plan({
        "operation": "query", "table": "places",
        "where": {"name": "Golden Lotus"},
        "understood_as": "look up Golden Lotus",
    }))

    out = await say(reasoner, 'what type of cuisine is golden lotus')
    result = only(out, "done").result

    assert result == "Golden Lotus is a chinese place in Soho, rated 4.5."
    # no SQL-shaped "found N match in <table>:" framing, no field-listing
    assert "found" not in result and "match" not in result
    assert "cuisine" not in result and "area" not in result and "rating" not in result
    # the internal id and the false "saved" flag carry nothing spoken aloud
    assert "id" not in result
    assert "False" not in result and "false" not in result and "saved" not in result


@pytest.mark.asyncio
async def test_conversational_input_produces_no_task(tmp_path):
    reasoner, _ = build(tmp_path, plan({"operation": "none",
                                        "understood_as": "small talk"}))
    out = await say(reasoner, "hey, how's your day going")
    assert kinds(out) == ["noop"]
    assert names(reasoner) == ["Sarah", "Marcus", "Priya", "Tom"]


@pytest.mark.asyncio
async def test_empty_utterance_never_reaches_the_model(tmp_path):
    reasoner, fake = build(tmp_path, plan({"operation": "none"}))
    out = await say(reasoner, "   ")
    assert kinds(out) == ["noop"]
    assert fake.calls == []


@pytest.mark.asyncio
async def test_update_matching_nothing_changes_nothing_and_asks_nothing(tmp_path):
    reasoner, _ = build(tmp_path, plan({
        "operation": "update", "table": "contacts",
        "where": {"first_name": "Nobody"}, "values": {"first_name": "Hans"},
    }))
    out = await say(reasoner, "rename Nobody to Hans")
    assert kinds(out) == ["ack", "done"]
    assert "nothing changed" in only(out, "done").result
    assert names(reasoner) == ["Sarah", "Marcus", "Priya", "Tom"]


# --- delete ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scoped_delete(tmp_path):
    """'delete the messages from yesterday' -- 3 of 5, and only after a yes."""
    reasoner, _ = build(tmp_path, plan({
        "operation": "delete", "table": "messages", "where": {"sent": YESTERDAY},
        "understood_as": "delete yesterday's messages",
    }))

    out = await say(reasoner, "delete the messages from yesterday")

    assert kinds(out) == ["ack", "confirm_required"]
    assert only(out, "confirm_required").verbatim_text == (
        "Delete 3 messages sent 2026-07-26?")
    assert ids(reasoner, "messages") == [1, 2, 3, 4, 5]  # nothing gone yet

    done = await answer(reasoner, only(out, "confirm_required").task_id, "yes")
    assert only(done, "done").result == "deleted 3 messages"
    assert ids(reasoner, "messages") == [1, 5]


@pytest.mark.asyncio
async def test_a_narrower_scope_deletes_strictly_less(tmp_path):
    """'just the ones from Marcus' -- the trailing qualifier from clip07,
    which changes the scope of an instruction that was already complete."""
    reasoner, _ = build(tmp_path, plan({
        "operation": "delete", "table": "messages",
        "where": {"sent": YESTERDAY, "contact": "Marcus Webb"},
    }))

    out = await say(reasoner,
                    "delete the messages from yesterday, just the ones from Marcus")
    verbatim = only(out, "confirm_required").verbatim_text
    assert "2 messages" in verbatim
    assert "from Marcus Webb" in verbatim
    assert "sent 2026-07-26" in verbatim

    await answer(reasoner, only(out, "confirm_required").task_id, "yes")
    assert ids(reasoner, "messages") == [1, 4, 5]


@pytest.mark.asyncio
async def test_an_unfiltered_delete_says_it_empties_the_table(tmp_path):
    """The most destructive thing this device can do must be unmistakable."""
    reasoner, _ = build(tmp_path, plan({
        "operation": "delete", "table": "messages",
        "understood_as": "clear out some old messages",
    }))

    out = await say(reasoner, "get rid of my messages")

    assert only(out, "confirm_required").verbatim_text == (
        "Delete all 5 messages, leaving nothing?")
    assert ids(reasoner, "messages") == [1, 2, 3, 4, 5]

    done = await answer(reasoner, only(out, "confirm_required").task_id, "yes")
    assert only(done, "done").result == "deleted 5 messages"
    assert reasoner.device.query("messages") == []


@pytest.mark.asyncio
async def test_delete_matching_nothing_deletes_nothing_and_asks_nothing(tmp_path):
    reasoner, _ = build(tmp_path, plan({
        "operation": "delete", "table": "messages", "where": {"sent": "1999-01-01"},
    }))
    out = await say(reasoner, "delete the messages from 1999")
    assert kinds(out) == ["ack", "done"]
    assert "nothing was deleted" in only(out, "done").result
    assert ids(reasoner, "messages") == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_delete_is_journaled(tmp_path):
    reasoner, _ = build(tmp_path, plan({
        "operation": "delete", "table": "messages", "where": {"sent": YESTERDAY},
    }))
    out = await say(reasoner, "delete yesterday's messages")
    await answer(reasoner, only(out, "confirm_required").task_id, "yes")

    entries = journal(reasoner)
    rec = next(e for e in entries if e["op"] == "delete")
    assert rec["table"] == "messages" and rec["rows"] == 3
    assert rec["where"] == {"sent": YESTERDAY}
    assert any(e["op"] == "commit" for e in entries)


# --- insert ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert(tmp_path):
    """'add a reminder for Tuesday' -- Tuesday is resolved to a real date
    before the plan is made, and nothing is added until a yes."""
    reasoner, _ = build(tmp_path, plan({
        "operation": "insert", "table": "calendar",
        "values": {"title": "reminder", "day": TOMORROW, "when": "2026-07-28T09:00"},
        "understood_as": "add a reminder",
    }))

    out = await say(reasoner, "add a reminder for Tuesday")

    assert kinds(out) == ["ack", "confirm_required"]
    assert only(out, "confirm_required").verbatim_text == (
        "Add to calendar: title reminder, day 2026-07-28, "
        "when 2026-07-28T09:00?")
    assert ids(reasoner, "calendar") == [1, 2, 3]

    done = await answer(reasoner, only(out, "confirm_required").task_id, "yes")
    assert only(done, "done").result == (
        "added to calendar: title reminder, day 2026-07-28, "
        "when 2026-07-28T09:00")
    assert ids(reasoner, "calendar") == [1, 2, 3, 4]
    assert reasoner.device.query("calendar", {"id": 4})[0]["title"] == "reminder"


@pytest.mark.asyncio
async def test_insert_ignores_a_model_supplied_id(tmp_path):
    reasoner, _ = build(tmp_path, plan({
        "operation": "insert", "table": "calendar",
        "values": {"id": 1, "title": "reminder", "day": TOMORROW},
    }))
    out = await say(reasoner, "add a reminder for Tuesday")
    assert "id" not in only(out, "confirm_required").verbatim_text
    await answer(reasoner, only(out, "confirm_required").task_id, "yes")
    assert ids(reasoner, "calendar") == [1, 2, 3, 4]
    assert reasoner.device.query("calendar", {"id": 1})[0]["title"] == "standup"


@pytest.mark.asyncio
async def test_insert_is_journaled(tmp_path):
    reasoner, _ = build(tmp_path, plan({
        "operation": "insert", "table": "calendar",
        "values": {"title": "reminder", "day": TOMORROW},
    }))
    out = await say(reasoner, "add a reminder for Tuesday")
    await answer(reasoner, only(out, "confirm_required").task_id, "yes")

    rec = next(e for e in journal(reasoner) if e["op"] == "insert")
    assert rec["table"] == "calendar"
    assert rec["record"] == {"id": 4, "title": "reminder", "day": TOMORROW}


@pytest.mark.asyncio
async def test_compound_utterance_produces_multiple_tasks(tmp_path):
    """'book me a table for four and text Sarah' -- two inserts, in two
    different tables, each needing its own confirmation."""
    reasoner, _ = build(tmp_path, plan(
        {"operation": "insert", "table": "calendar",
         "values": {"title": "table for four", "day": TOMORROW,
                    "when": "2026-07-28T19:00"},
         "understood_as": "book a table for four"},
        {"operation": "insert", "table": "messages",
         "values": {"contact": "Sarah Chen", "body": "table booked for four",
                    "sent": "2026-07-27", "read": True},
         "understood_as": "text Sarah"},
    ))

    out = await say(reasoner, "book me a table for four and text Sarah")

    acks = every(out, "ack")
    confirms = every(out, "confirm_required")
    assert len(acks) == 2 and len(confirms) == 2
    assert len({m.task_id for m in acks}) == 2
    # neither wrote anything on the strength of being asked
    assert ids(reasoner, "calendar") == [1, 2, 3]
    assert ids(reasoner, "messages") == [1, 2, 3, 4, 5]

    for m in confirms:
        await answer(reasoner, m.task_id, "yes")
    assert ids(reasoner, "calendar") == [1, 2, 3, 4]
    assert ids(reasoner, "messages") == [1, 2, 3, 4, 5, 6]
    assert reasoner.device.query("messages", {"id": 6})[0]["contact"] == "Sarah Chen"


@pytest.mark.asyncio
async def test_confirming_one_of_two_leaves_the_other_untouched(tmp_path):
    reasoner, _ = build(tmp_path, plan(
        {"operation": "insert", "table": "calendar", "values": {"title": "dinner"}},
        {"operation": "delete", "table": "messages", "where": {"sent": YESTERDAY}},
    ))
    out = await say(reasoner, "add dinner to my calendar and delete yesterday's texts")
    first, second = every(out, "confirm_required")

    await answer(reasoner, first.task_id, "yes")
    assert ids(reasoner, "calendar") == [1, 2, 3, 4]
    assert ids(reasoner, "messages") == [1, 2, 3, 4, 5]  # still all there

    await answer(reasoner, second.task_id, "no")
    assert ids(reasoner, "messages") == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_compound_mixing_a_read_and_a_write(tmp_path):
    reasoner, _ = build(tmp_path, plan(
        {"operation": "query", "table": "contacts", "where": {"group": "work"}},
        {"operation": "update", "table": "contacts",
         "where": {"group": "work"}, "values": {"first_name": "Hans"}},
    ))

    out = await say(reasoner, "find all my work contacts and change them all to Hans")

    assert kinds(out) == ["ack", "done", "ack", "confirm_required"]
    # the read completed, the write did not
    assert names(reasoner) == ["Sarah", "Marcus", "Priya", "Tom"]


@pytest.mark.asyncio
async def test_an_action_with_no_table_behind_it_is_refused_honestly(tmp_path):
    reasoner, _ = build(tmp_path, plan({"operation": "unsupported",
                                        "understood_as": "get an uber to the station"}))
    out = await say(reasoner, "get me an uber to the station")
    assert kinds(out) == ["ack", "failed"]
    assert only(out, "ack").understood_as == "get an uber to the station"
    assert only(out, "failed").reason == "this phone can't do that yet"


# --- SAFETY: the blast radius is counted, never quoted ------------------------


@pytest.mark.asyncio
async def test_confirm_states_the_real_count_from_the_device_not_the_model(tmp_path):
    """The model claims 99 rows and names its own count in the paraphrase. The
    confirmation the user actually hears is counted on the device."""
    reasoner, _ = build(tmp_path, plan({
        "operation": "update", "table": "contacts",
        "where": {"group": "work"}, "values": {"first_name": "Hans"},
        "understood_as": "change all 99 contacts to Hans",
        "affected_rows": 99, "row_count": 99, "result": "renamed 99 contacts",
    }))

    out = await say(reasoner, "change my work contacts to Hans")
    verbatim = only(out, "confirm_required").verbatim_text

    assert "2 work contacts" in verbatim
    assert "99" not in verbatim

    # and the completion report is counted at write time, from the write
    done = await answer(reasoner, only(out, "confirm_required").task_id, "yes")
    assert only(done, "done").result == "renamed 2 work contacts to Hans"
    assert "99" not in only(done, "done").result


@pytest.mark.asyncio
async def test_delete_confirm_states_the_real_count_not_the_model_s(tmp_path):
    """The model says it is tidying up 3 old messages; the filter it actually
    planned takes the whole table. The user hears 5, and hears that it empties
    the table."""
    reasoner, _ = build(tmp_path, plan({
        "operation": "delete", "table": "messages",
        "understood_as": "delete 3 old messages",
        "affected_rows": 3, "result": "deleted 3 messages",
    }))

    out = await say(reasoner, "clear out the old messages")
    verbatim = only(out, "confirm_required").verbatim_text

    assert verbatim == (
        "Delete all 5 messages, leaving nothing?")
    assert "3" not in verbatim
    assert ids(reasoner, "messages") == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_unfiltered_update_states_the_whole_table_count(tmp_path):
    reasoner, _ = build(tmp_path, plan({
        "operation": "update", "table": "contacts",
        "values": {"first_name": "Hans"},
        "understood_as": "change 3 contacts", "affected_rows": 3,
    }))
    out = await say(reasoner, "change all my contacts to Hans")
    assert "all 4 contacts" in only(out, "confirm_required").verbatim_text


@pytest.mark.asyncio
async def test_unfiltered_update_confirms_true_count_and_changes_every_record(tmp_path):
    """'set all my contacts to Hans' with "where" correctly omitted -- the
    fixed prompt's own worked example. The count the user hears must be the
    device's true row count (4, from this fixture), and confirming must
    actually change every one of them, not just the ones a filter happened
    to select."""
    reasoner, _ = build(tmp_path, plan({
        "operation": "update", "table": "contacts",
        "values": {"first_name": "Hans"},
        "understood_as": "rename all contacts to Hans",
    }))

    out = await say(reasoner, "set all my contacts to Hans")
    verbatim = only(out, "confirm_required").verbatim_text
    assert "all 4 contacts" in verbatim
    assert names(reasoner) == ["Sarah", "Marcus", "Priya", "Tom"]  # nothing yet

    done = await answer(reasoner, only(out, "confirm_required").task_id, "yes")
    assert only(done, "done").result == "renamed all 4 contacts to Hans"
    assert names(reasoner) == ["Hans", "Hans", "Hans", "Hans"]


@pytest.mark.asyncio
async def test_the_model_cannot_state_a_fact_the_user_hears(tmp_path):
    """Whatever the model puts in fact-shaped fields is dropped on the floor:
    only `understood_as` (a paraphrase of the user's own request) survives,
    and only for outcomes that assert nothing about the device."""
    reasoner, _ = build(tmp_path, plan({
        "operation": "query", "table": "calendar", "where": {"day": TOMORROW},
        "understood_as": "read the calendar", "result": "you have 40 meetings",
        "verbatim_text": "I already deleted them", "reason": "trust me",
    }))
    out = await say(reasoner, "what's on tomorrow")
    text = " ".join(m.result + m.verbatim_text + m.reason for m in out)
    assert "40 meetings" not in text
    assert "already deleted" not in text
    assert "there are 2 calendar" in text


# --- SAFETY: bad model output mutates nothing --------------------------------


MALFORMED = [
    pytest.param("this is not json at all", id="not-json"),
    pytest.param("```json\n{\"intents\": []}\n```", id="fenced-markdown"),
    pytest.param('{"intents": [{"operation": "drop_table", "table": "contacts"}]}',
                 id="operation-off-the-menu"),
    pytest.param('{"intents": [{"operation": "update"}]}', id="no-table"),
    pytest.param('{"intents": [{"operation": "delete"}]}', id="delete-with-no-table"),
    pytest.param('{"intents": [{"operation": "update", "table": "passwords",'
                 ' "values": {"x": 1}}]}', id="table-that-does-not-exist"),
    pytest.param('{"intents": [{"operation": "delete", "table": "photos"}]}',
                 id="delete-a-table-that-does-not-exist"),
    pytest.param('{"intents": [{"operation": "update", "table": "contacts",'
                 ' "values": {"salary": 0}}]}', id="field-that-does-not-exist"),
    pytest.param('{"intents": [{"operation": "update", "table": "contacts",'
                 ' "where": {"nickname": "P"}, "values": {"first_name": "Hans"}}]}',
                 id="filter-field-that-does-not-exist"),
    pytest.param('{"intents": [{"operation": "delete", "table": "messages",'
                 ' "where": {"nickname": "P"}}]}', id="delete-on-an-unknown-field"),
    pytest.param('{"intents": [{"operation": "update", "table": "contacts",'
                 ' "values": {}}]}', id="nothing-to-set"),
    pytest.param('{"intents": [{"operation": "insert", "table": "calendar"}]}',
                 id="nothing-to-add"),
    pytest.param('{"intents": [{"operation": "insert", "table": "messages",'
                 ' "where": {"contact": "Sarah Chen"}, "values": {"body": "hi"}}]}',
                 id="insert-with-a-filter"),
    pytest.param('{"intents": [{"operation": "delete", "table": "messages",'
                 ' "where": {"sent": "2026-07-26"}, "values": {"read": true}}]}',
                 id="delete-carrying-values"),
    pytest.param('{"intents": [{"operation": "update", "table": "contacts",'
                 ' "where": {"group": ["work", "family"]},'
                 ' "values": {"first_name": "Hans"}}]}', id="non-scalar-filter"),
    pytest.param('{"intents": [{"operation": "update", "table": "contacts",'
                 ' "where": {"first_name": "*"}, "values": {"first_name": "Hans"}}]}',
                 id="asterisk-wildcard-filter"),
    pytest.param('{"intents": [{"operation": "update", "table": "contacts",'
                 ' "where": {"first_name": "*", "last_name": "*"},'
                 ' "values": {"first_name": "Hans"}}]}',
                 id="asterisk-wildcard-filter-on-every-field"),
    pytest.param('{"intents": [{"operation": "delete", "table": "messages",'
                 ' "where": {"contact": "%"}}]}', id="percent-wildcard-filter"),
    pytest.param('{"intents": "rename everything"}', id="intents-not-a-list"),
    pytest.param("", id="empty-body"),
    pytest.param("null", id="null"),
]


@pytest.mark.parametrize("payload", MALFORMED)
@pytest.mark.asyncio
async def test_malformed_llm_output_mutates_nothing(tmp_path, payload):
    reasoner, _ = build(tmp_path, payload)
    before = copy.deepcopy(reasoner.device.snapshot())

    out = await say(reasoner, "change all my contacts to Hans")

    assert "confirm_required" not in kinds(out)
    assert reasoner.device.snapshot() == before
    # and there is nothing a later "yes" could commit
    for m in out:
        if m.task_id:
            assert kinds(await answer(reasoner, m.task_id, "yes")) == ["noop"]
    assert reasoner.device.snapshot() == before


@pytest.mark.parametrize("payload", MALFORMED)
@pytest.mark.asyncio
async def test_malformed_llm_output_is_always_audible(tmp_path, payload):
    """Failing safe must not mean failing silently: every malformed plan ends
    in a `failed` or a `noop`, never in the system quietly doing nothing while
    appearing to work."""
    reasoner, _ = build(tmp_path, payload)
    out = await say(reasoner, "change all my contacts to Hans")
    assert "failed" in kinds(out) or kinds(out) == ["noop"]


@pytest.mark.asyncio
async def test_wildcard_filter_is_rejected_explicitly_not_read_as_zero_matches(tmp_path):
    """'set all my contacts to hands' -- the second live-model failure. Asked
    for every contact, the model invented a "*" filter instead of omitting
    "where". That filter matches nothing on a real device (equality only), so
    without this check it would be indistinguishable from an honest "no
    matches" -- an ack + done reporting "nothing changed", which is wrong: the
    plan was malformed, not empty. It must instead be refused outright, the
    same safe path as any other unreadable plan, and audibly so."""
    reasoner, _ = build(tmp_path, plan({
        "operation": "update", "table": "contacts",
        "where": {"first_name": "*", "last_name": "*"},
        "values": {"first_name": "Hans"},
        "understood_as": "set all my contacts to hands",
    }))

    out = await say(reasoner, "set all my contacts to hands")

    assert kinds(out) == ["ack", "failed"]
    assert "wildcard" in only(out, "failed").reason
    # not the "0 matches" phrasing an empty filter would have produced
    assert "nothing changed" not in only(out, "failed").reason
    assert names(reasoner) == ["Sarah", "Marcus", "Priya", "Tom"]


# --- parsing resilience: prose/fences tolerated, one retry, then fail safe ---


@pytest.mark.asyncio
async def test_prose_wrapped_json_is_parsed_and_executed(tmp_path):
    """A real model narrating around its own JSON ("Sure, here you go:\n```json
    \n{...}\n```\nHope that helps!") is not unreadable -- the object inside is
    well-formed and must be extracted and executed on the FIRST attempt, with
    no retry needed."""
    payload = (
        "Sure, here is the plan:\n```json\n"
        + json.dumps(plan({
            "operation": "query", "table": "calendar", "where": {"day": TOMORROW},
        }))
        + "\n```\nHope that helps!"
    )
    reasoner, fake = build(tmp_path, payload)

    out = await say(reasoner, "what's in my calendar tomorrow")

    assert kinds(out) == ["ack", "done"]
    assert "there are 2 calendar" in only(out, "done").result
    assert len(fake.calls) == 1  # parsed clean the first time -- no retry needed


@pytest.mark.asyncio
async def test_bad_json_then_valid_json_retries_once_and_executes(tmp_path):
    """The model stumbles once, then gets it right when told what went wrong.
    That must not be indistinguishable from an unreadable utterance."""
    good = plan({"operation": "query", "table": "calendar", "where": {"day": TOMORROW}})
    reasoner, fake = build(tmp_path, "this is not json at all", good)

    out = await say(reasoner, "what's in my calendar tomorrow")

    assert len(fake.calls) == 2  # the retry happened
    retry_messages = fake.calls[1]["messages"]
    assert "could not be read as a plan" in retry_messages[-1]["content"]
    assert kinds(out) == ["ack", "done"]
    assert "there are 2 calendar" in only(out, "done").result


@pytest.mark.asyncio
async def test_bad_json_on_both_attempts_fails_safe(tmp_path):
    """When the retry ALSO fails, the current safe behaviour holds exactly:
    a `failed` message, nothing written -- not a crash, not a silent noop."""
    reasoner, fake = build(tmp_path, "not json", "still not json either")
    before = copy.deepcopy(reasoner.device.snapshot())

    out = await say(reasoner, "delete all my messages")

    assert len(fake.calls) == 2  # both attempts were made
    assert kinds(out) == ["failed"]
    assert reasoner.device.snapshot() == before
    assert reasoner.misreads == 1


@pytest.mark.asyncio
async def test_a_bad_intent_does_not_kill_its_siblings(tmp_path):
    reasoner, _ = build(tmp_path, plan(
        {"operation": "delete", "table": "nowhere"},
        {"operation": "query", "table": "calendar", "where": {"day": TOMORROW}},
    ))
    out = await say(reasoner, "do the impossible and check my calendar")
    assert kinds(out) == ["ack", "failed", "ack", "done"]
    assert only(out, "failed").reason == "there's no nowhere on this device"
    assert "there are 2 calendar" in only(out, "done").result
    assert reasoner.misreads == 1


@pytest.mark.asyncio
async def test_unknown_where_field_is_spoken_as_plain_english(tmp_path):
    """The second live-session failure: asked to look up a contact, the model
    filtered on "contact" -- a field that exists on `messages`, not
    `contacts`. The device correctly refuses (that key isn't on this table),
    but what got spoken was "contacts has no contact": ungrammatical, and it
    named an internal field the user never said. The refusal must instead
    read as something a person would say, with no raw table.field jargon."""
    reasoner, _ = build(tmp_path, plan({
        "operation": "query", "table": "contacts", "where": {"contact": "Tom"},
    }))

    out = await say(reasoner, "look up the contact for Tom")

    assert kinds(out) == ["ack", "failed"]
    reason = only(out, "failed").reason
    assert reason == "I couldn't find a contact matching that."
    assert "has no" not in reason
    assert "contacts.contact" not in reason
    assert reasoner.misreads == 1


@pytest.mark.asyncio
async def test_unknown_values_field_is_spoken_as_plain_english(tmp_path):
    """Same defect class, but on the write side: an update naming a field the
    table does not have must not echo the raw table/field pairing back."""
    reasoner, _ = build(tmp_path, plan({
        "operation": "update", "table": "contacts",
        "where": {"first_name": "Tom"}, "values": {"nickname": "Tommy"},
    }))

    out = await say(reasoner, "set Tom's nickname to Tommy")

    assert kinds(out) == ["ack", "failed"]
    reason = only(out, "failed").reason
    assert reason == "that's not something I can set on a contact."
    assert "has no" not in reason
    assert "nickname" not in reason


# --- SAFETY: an unreachable or slow model mutates nothing --------------------


@pytest.mark.parametrize("boom", [
    pytest.param(httpx.ConnectError("connection refused"), id="unreachable"),
    pytest.param(httpx.ReadTimeout("timed out"), id="timeout"),
    pytest.param(httpx.HTTPStatusError("500", request=None, response=None), id="server-error"),
])
@pytest.mark.asyncio
async def test_llm_transport_failure_mutates_nothing(tmp_path, boom):
    reasoner, _ = build(tmp_path, boom)
    before = copy.deepcopy(reasoner.device.snapshot())

    out = await say(reasoner, "delete all my messages")

    assert kinds(out) == ["failed"]
    assert out[0].task_id  # addressable, so the user can be told about it
    assert reasoner.device.snapshot() == before
    assert reasoner.misreads == 1
    # a following "yes" has nothing to confirm
    assert kinds(await answer(reasoner, out[0].task_id, "yes")) == ["noop"]
    assert reasoner.device.snapshot() == before


@pytest.mark.asyncio
async def test_each_timed_out_utterance_gets_its_own_task_id(tmp_path):
    """Collapsing them onto one id makes the registry's terminal-state guard
    swallow every failure after the first."""
    reasoner, _ = build(tmp_path, httpx.ConnectError("nope"))
    first = await say(reasoner, "change all my contacts to Hans")
    second = await say(reasoner, "change all my contacts to Greta")
    assert first[0].task_id != second[0].task_id


# --- SAFETY: the confirmation gates, imported not reimplemented --------------


def test_confirmation_gates_are_the_stub_s_own():
    from rtvoice.reasoner_stub import confirmation_decision

    assert llm_mod.confirmation_decision is confirmation_decision


async def arm(tmp_path, *, op="update"):
    """A destructive task awaiting confirmation. Parametrised by operation so
    the gates are proved for each of them, not just for update."""
    intent = {"operation": op, "table": "contacts"}
    if op != "delete":
        intent["values"] = {"first_name": "Hans"}
    reasoner, _ = build(tmp_path, plan(intent))
    out = await say(reasoner, "change all my contacts to Hans")
    return reasoner, only(out, "confirm_required").task_id


@pytest.mark.parametrize("text", [
    "okay so what's on my calendar tomorrow",
    "yes I was talking to my colleague, ignore that",
    "sorry, I was on the phone, ok anyway",
    "yeah I'll call him back later about the restaurant booking",
])
@pytest.mark.asyncio
async def test_non_answer_shaped_reply_does_not_resolve_the_confirmation(tmp_path, text):
    reasoner, tid = await arm(tmp_path)

    result = await answer(reasoner, tid, text)

    assert kinds(result) == ["noop"]
    assert names(reasoner) == ["Sarah", "Marcus", "Priya", "Tom"]
    # neither committed nor cancelled: the question is still open
    done = await answer(reasoner, tid, "yes")
    assert "done" in kinds(done)
    assert names(reasoner) == ["Hans", "Hans", "Hans", "Hans"]


@pytest.mark.parametrize("op", ["update", "delete", "insert"])
@pytest.mark.parametrize("text", ["no", "don't do it", "that is not okay",
                                  "no, don't confirm it", "hmm", "not sure"])
@pytest.mark.asyncio
async def test_negation_and_ambiguity_default_deny_every_destructive_op(tmp_path, op, text):
    reasoner, tid = await arm(tmp_path, op=op)
    before = copy.deepcopy(reasoner.device.snapshot())

    result = await answer(reasoner, tid, text)

    assert only(result, "failed").reason == "cancelled by user"
    assert reasoner.device.snapshot() == before


@pytest.mark.parametrize("op", ["update", "delete", "insert"])
@pytest.mark.asyncio
async def test_a_non_answer_leaves_every_destructive_op_pending(tmp_path, op):
    reasoner, tid = await arm(tmp_path, op=op)
    before = copy.deepcopy(reasoner.device.snapshot())
    assert kinds(await answer(reasoner, tid, "what's the weather like")) == ["noop"]
    assert reasoner.device.snapshot() == before
    assert "done" in kinds(await answer(reasoner, tid, "yes"))


@pytest.mark.parametrize("word", ["yes", "yeah", "sure", "right", "go ahead", "do it"])
@pytest.mark.asyncio
async def test_affirmatives_commit(tmp_path, word):
    reasoner, tid = await arm(tmp_path)
    assert "done" in kinds(await answer(reasoner, tid, word))
    assert names(reasoner) == ["Hans", "Hans", "Hans", "Hans"]


@pytest.mark.asyncio
async def test_answer_to_an_unknown_task_is_a_noop(tmp_path):
    reasoner, _ = build(tmp_path, plan({"operation": "none"}))
    assert kinds(await answer(reasoner, "ghost", "yes")) == ["noop"]


# --- SAFETY: cancellation ----------------------------------------------------


@pytest.mark.parametrize("op", ["update", "delete", "insert"])
@pytest.mark.asyncio
async def test_cancel_before_confirm_leaves_state_untouched(tmp_path, op):
    reasoner, tid = await arm(tmp_path, op=op)
    before = copy.deepcopy(reasoner.device.snapshot())

    out = await reasoner.handle(OrchestratorMessage(kind="cancel", task_id=tid))

    assert only(out, "failed").reason == "cancelled"
    assert reasoner.device.snapshot() == before
    # the task is gone, so a late "yes" cannot resurrect it
    assert kinds(await answer(reasoner, tid, "yes")) == ["noop"]
    assert reasoner.device.snapshot() == before


class CancelAfterOneRow(CancellationToken):
    def __init__(self) -> None:
        super().__init__()
        self._checks = 0

    def check(self) -> None:
        self._checks += 1
        if self._checks > 1:
            self.cancel()
        super().check()


@pytest.mark.asyncio
async def test_a_token_firing_mid_update_rolls_the_whole_thing_back(tmp_path, monkeypatch):
    """The write runs against a CancellationToken and is staged, so an abort
    partway through leaves NO rows changed -- not the ones already touched."""
    monkeypatch.setattr(llm_mod, "CancellationToken", CancelAfterOneRow)

    reasoner, tid = await arm(tmp_path)
    out = await answer(reasoner, tid, "yes")

    assert only(out, "failed").reason == "stopped partway"
    assert names(reasoner) == ["Sarah", "Marcus", "Priya", "Tom"]
    assert reasoner.tokens == {}
    entries = journal(reasoner)
    assert any(e["op"] == "update" and e.get("cancelled") for e in entries)
    assert any(e["op"] == "rollback" for e in entries)


@pytest.mark.asyncio
async def test_a_token_firing_mid_delete_rolls_the_whole_thing_back(tmp_path, monkeypatch):
    """The dangerous one: a delete interrupted halfway must not leave the
    device holding half a table."""
    monkeypatch.setattr(llm_mod, "CancellationToken", CancelAfterOneRow)

    reasoner, _ = build(tmp_path, plan({"operation": "delete", "table": "messages"}))
    out = await say(reasoner, "delete all my messages")
    result = await answer(reasoner, only(out, "confirm_required").task_id, "yes")

    assert only(result, "failed").reason == "stopped partway"
    assert ids(reasoner, "messages") == [1, 2, 3, 4, 5]
    entries = journal(reasoner)
    cancelled = next(e for e in entries if e["op"] == "delete" and e.get("cancelled"))
    assert cancelled["rows"] == 1  # one was already gone from the working copy
    assert any(e["op"] == "rollback" for e in entries)
    # and nothing reached disk
    assert len(json.loads(reasoner.device.state_path.read_text())["messages"]) == 5


@pytest.mark.asyncio
async def test_the_live_token_is_published_for_the_orchestrator(tmp_path, monkeypatch):
    """Orchestrator._abort fires the token by direct reference, so it must be
    in the shared dict while the write is in flight."""
    seen: list[str] = []

    class Watcher(CancellationToken):
        def check(self) -> None:
            seen.append("tokens=" + ",".join(reasoner.tokens))
            super().check()

    monkeypatch.setattr(llm_mod, "CancellationToken", Watcher)
    reasoner, tid = await arm(tmp_path)
    await answer(reasoner, tid, "yes")
    assert seen and all(s == f"tokens={tid}" for s in seen)
    assert reasoner.tokens == {}  # cleaned up afterwards


# --- other protocol messages -------------------------------------------------


@pytest.mark.asyncio
async def test_nudge_reports_on_a_pending_task(tmp_path):
    reasoner, tid = await arm(tmp_path)
    out = await reasoner.handle(OrchestratorMessage(kind="nudge", task_id=tid))
    assert only(out, "progress").status == "still working"
    out = await reasoner.handle(OrchestratorMessage(kind="nudge", task_id="ghost"))
    assert only(out, "progress").status == "no such task"


@pytest.mark.asyncio
async def test_latency_is_chosen_not_discovered(tmp_path):
    reasoner, _ = build(tmp_path, plan({"operation": "none"}))
    reasoner.latency_ms = 20
    import time

    t0 = time.monotonic()
    await say(reasoner, "hello")
    assert time.monotonic() - t0 >= 0.02


@pytest.mark.asyncio
async def test_recent_conversation_is_carried_into_the_next_plan(tmp_path):
    reasoner, fake = build(tmp_path, plan({"operation": "query", "table": "calendar"}))
    await say(reasoner, "what's on tomorrow")
    await say(reasoner, "and the day after")
    contents = [m["content"] for m in fake.calls[1]["messages"]]
    assert "what's on tomorrow" in contents


# --- wiring ------------------------------------------------------------------


def test_default_wiring_is_the_stub(tmp_path, monkeypatch):
    from rtvoice.orchestrator import build_default_orchestrator
    from rtvoice.reasoner_stub import ReasonerStub

    monkeypatch.delenv("REASONER", raising=False)
    monkeypatch.setenv("DEVICE_STATE", str(tmp_path / "d.json"))
    (tmp_path / "d.json").write_text(json.dumps(DEVICE))
    orch = build_default_orchestrator(tmp_path / "session")
    assert isinstance(orch.reasoner, ReasonerStub)


def test_reasoner_env_var_selects_the_llm_reasoner(tmp_path, monkeypatch):
    from rtvoice.orchestrator import build_default_orchestrator

    monkeypatch.setenv("REASONER", "llm")
    monkeypatch.setenv("REASONER_URL", "http://elsewhere:9000/v1")
    monkeypatch.setenv("REASONER_MODEL", "some/model")
    monkeypatch.setenv("DEVICE_STATE", str(tmp_path / "d.json"))
    (tmp_path / "d.json").write_text(json.dumps(DEVICE))

    orch = build_default_orchestrator(tmp_path / "session")

    assert isinstance(orch.reasoner, LlmReasoner)
    assert orch.reasoner.base_url == "http://elsewhere:9000/v1"
    assert orch.reasoner.model == "some/model"
    # the orchestrator binds its own token registry, as it does for the stub
    assert orch.reasoner.tokens is orch.tokens
