"""LlmReasoner against a fake model endpoint.

Every test here monkeypatches the HTTP POST, so the suite stays offline and
fast -- no vLLM, no GPU, no network. What is NOT faked is everything that
matters: the JSON parsing, the schema validation, the plan validation against
the real device schema, the confirmation gates, and the device writes.

The safety tests are the point of the file. A model that lies about the blast
radius, returns garbage, or cannot be reached must never result in a mutation.
"""
import copy
import json

import httpx
import pytest

from rtvoice import llm_reasoner as llm_mod
from rtvoice.cancellation import CancellationToken
from rtvoice.device import DeviceState
from rtvoice.llm_reasoner import LlmReasoner
from rtvoice.protocol import OrchestratorMessage

DEVICE = {
    "contacts": [
        {"id": 1, "first_name": "Sarah", "last_name": "Chen", "group": "work"},
        {"id": 2, "first_name": "Marcus", "last_name": "Webb", "group": "work"},
        {"id": 3, "first_name": "Priya", "last_name": "Nair", "group": "family"},
        {"id": 4, "first_name": "Tom", "last_name": "Okafor", "group": "friends"},
    ],
    "calendar": [
        {"id": 1, "title": "standup", "when": "2026-07-28T09:00"},
        {"id": 2, "title": "dentist", "when": "2026-07-28T16:30"},
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
    reasoner = LlmReasoner(device, base_url="http://fake/v1", model="fake-model")
    fake = FakeLlm(*replies)
    reasoner._client.post = fake  # the only thing faked
    return reasoner, fake


def names(reasoner) -> list[str]:
    return [c["first_name"] for c in reasoner.device.query("contacts")]


async def say(reasoner, text):
    return await reasoner.handle(OrchestratorMessage(kind="utterance", text=text))


async def answer(reasoner, tid, text):
    return await reasoner.handle(
        OrchestratorMessage(kind="clarification_answer", task_id=tid, text=text))


def kinds(msgs):
    return [m.kind for m in msgs]


def only(msgs, kind):
    return next(m for m in msgs if m.kind == kind)


def plan(*intents):
    return {"intents": list(intents)}


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
    # contacts" resolvable into a filter at all
    assert '"work"' in block and '"family"' in block


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
    # the user's words reached the model, along with the device schema
    assert fake.calls[0]["messages"][-1]["content"] == "please rename Priya as Jude Hawrani"
    assert "contacts" in fake.calls[0]["messages"][1]["content"]
    # nothing written before confirmation
    assert names(reasoner) == ["Sarah", "Marcus", "Priya", "Tom"]

    tid = only(out, "confirm_required").task_id
    done = await answer(reasoner, tid, "yes")
    assert only(done, "done").result == 'updated 1 row in contacts: set first_name to "Jude" and last_name to "Hawrani"'
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
    tid = only(out, "confirm_required").task_id
    assert "2 rows" in only(out, "confirm_required").verbatim_text

    await answer(reasoner, tid, "go ahead")
    assert names(reasoner) == ["Hans", "Hans", "Priya", "Tom"]


@pytest.mark.asyncio
async def test_read_only_query_completes_without_confirmation(tmp_path):
    """'what's in my calendar tomorrow' -- a read, so no consent is needed."""
    reasoner, _ = build(tmp_path, plan({"operation": "query", "table": "calendar"}))

    out = await say(reasoner, "what's in my calendar tomorrow")

    assert kinds(out) == ["ack", "done"]
    assert "confirm_required" not in kinds(out)
    result = only(out, "done").result
    # the answer is the device's rows, not the model's prose
    assert "2 matches in calendar" in result
    assert "standup" in result and "dentist" in result


@pytest.mark.asyncio
async def test_compound_utterance_produces_multiple_tasks(tmp_path):
    """'book me a table for four and text Sarah' -- two actions, two tasks."""
    reasoner, _ = build(tmp_path, plan(
        {"operation": "unsupported", "understood_as": "book a table for four"},
        {"operation": "unsupported", "understood_as": "text Sarah"},
    ))

    out = await say(reasoner, "book me a table for four and text Sarah")

    acks = [m for m in out if m.kind == "ack"]
    assert len(acks) == 2
    assert len({m.task_id for m in acks}) == 2
    assert [m.understood_as for m in acks] == ["book a table for four", "text Sarah"]
    # honest about what the device cannot do, rather than silently doing nothing
    assert all(m.reason == "this phone can't do that yet"
               for m in out if m.kind == "failed")
    assert names(reasoner) == ["Sarah", "Marcus", "Priya", "Tom"]


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

    assert verbatim == 'This will update 2 rows in contacts: set first_name to "Hans". Confirm?'
    assert "99" not in verbatim

    # and the completion report is counted at write time, from the write
    done = await answer(reasoner, only(out, "confirm_required").task_id, "yes")
    assert only(done, "done").result == 'updated 2 rows in contacts: set first_name to "Hans"'
    assert "99" not in only(done, "done").result


@pytest.mark.asyncio
async def test_unfiltered_update_states_the_whole_table_count(tmp_path):
    reasoner, _ = build(tmp_path, plan({
        "operation": "update", "table": "contacts",
        "values": {"first_name": "Hans"},
        "understood_as": "change 3 contacts", "affected_rows": 3,
    }))
    out = await say(reasoner, "change all my contacts to Hans")
    assert "4 rows" in only(out, "confirm_required").verbatim_text


@pytest.mark.asyncio
async def test_the_model_cannot_state_a_fact_the_user_hears(tmp_path):
    """Whatever the model puts in fact-shaped fields is dropped on the floor:
    only `understood_as` (a paraphrase of the user's own request) survives,
    and only for outcomes that assert nothing about the device."""
    reasoner, _ = build(tmp_path, plan({
        "operation": "query", "table": "calendar",
        "understood_as": "read the calendar", "result": "you have 40 meetings",
        "verbatim_text": "I already deleted them", "reason": "trust me",
    }))
    out = await say(reasoner, "what's on tomorrow")
    text = " ".join(m.result + m.verbatim_text + m.reason for m in out)
    assert "40 meetings" not in text
    assert "already deleted" not in text
    assert "2 matches in calendar" in text


# --- SAFETY: bad model output mutates nothing --------------------------------


MALFORMED = [
    pytest.param("this is not json at all", id="not-json"),
    pytest.param("```json\n{\"intents\": []}\n```", id="fenced-markdown"),
    pytest.param('{"intents": [{"operation": "delete", "table": "contacts"}]}',
                 id="operation-off-the-menu"),
    pytest.param('{"intents": [{"operation": "update"}]}', id="no-table"),
    pytest.param('{"intents": [{"operation": "update", "table": "passwords",'
                 ' "values": {"x": 1}}]}', id="table-that-does-not-exist"),
    pytest.param('{"intents": [{"operation": "update", "table": "contacts",'
                 ' "values": {"salary": 0}}]}', id="field-that-does-not-exist"),
    pytest.param('{"intents": [{"operation": "update", "table": "contacts",'
                 ' "where": {"nickname": "P"}, "values": {"first_name": "Hans"}}]}',
                 id="filter-field-that-does-not-exist"),
    pytest.param('{"intents": [{"operation": "update", "table": "contacts",'
                 ' "values": {}}]}', id="nothing-to-set"),
    pytest.param('{"intents": [{"operation": "update", "table": "contacts",'
                 ' "where": {"group": ["work", "family"]},'
                 ' "values": {"first_name": "Hans"}}]}', id="non-scalar-filter"),
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
    assert names(reasoner) == ["Sarah", "Marcus", "Priya", "Tom"]
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
async def test_a_bad_intent_does_not_kill_its_siblings(tmp_path):
    reasoner, _ = build(tmp_path, plan(
        {"operation": "update", "table": "nowhere", "values": {"a": 1}},
        {"operation": "query", "table": "calendar"},
    ))
    out = await say(reasoner, "do the impossible and check my calendar")
    assert kinds(out) == ["ack", "failed", "ack", "done"]
    assert only(out, "failed").reason == "there's no nowhere on this device"
    assert "2 matches in calendar" in only(out, "done").result
    assert reasoner.misreads == 1


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

    out = await say(reasoner, "change all my contacts to Hans")

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


@pytest.mark.asyncio
async def test_confirmation_gates_are_the_stub_s_own(tmp_path):
    from rtvoice.reasoner_stub import confirmation_decision

    assert llm_mod.confirmation_decision is confirmation_decision


async def arm(tmp_path):
    reasoner, _ = build(tmp_path, plan({
        "operation": "update", "table": "contacts",
        "values": {"first_name": "Hans"},
    }))
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


@pytest.mark.parametrize("text", ["no", "don't do it", "that is not okay",
                                  "no, don't confirm it", "hmm", "not sure"])
@pytest.mark.asyncio
async def test_negation_and_ambiguity_default_deny(tmp_path, text):
    reasoner, tid = await arm(tmp_path)
    result = await answer(reasoner, tid, text)
    assert only(result, "failed").reason == "cancelled by user"
    assert names(reasoner) == ["Sarah", "Marcus", "Priya", "Tom"]


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


@pytest.mark.asyncio
async def test_cancel_before_confirm_leaves_state_untouched(tmp_path):
    reasoner, tid = await arm(tmp_path)
    out = await reasoner.handle(OrchestratorMessage(kind="cancel", task_id=tid))
    assert only(out, "failed").reason == "cancelled"
    assert names(reasoner) == ["Sarah", "Marcus", "Priya", "Tom"]
    # and the task is gone, so a late "yes" cannot resurrect it
    assert kinds(await answer(reasoner, tid, "yes")) == ["noop"]
    assert names(reasoner) == ["Sarah", "Marcus", "Priya", "Tom"]


@pytest.mark.asyncio
async def test_a_token_firing_mid_write_rolls_the_whole_update_back(tmp_path, monkeypatch):
    """The write runs against a CancellationToken and is staged, so an abort
    partway through leaves NO rows changed -- not the ones already touched."""

    class CancelAfterOneRow(CancellationToken):
        def __init__(self) -> None:
            super().__init__()
            self._checks = 0

        def check(self) -> None:
            self._checks += 1
            if self._checks > 1:
                self.cancel()
            super().check()

    monkeypatch.setattr(llm_mod, "CancellationToken", CancelAfterOneRow)

    reasoner, tid = await arm(tmp_path)
    out = await answer(reasoner, tid, "yes")

    assert only(out, "failed").reason == "stopped partway"
    assert names(reasoner) == ["Sarah", "Marcus", "Priya", "Tom"]
    assert reasoner.tokens == {}
    journal = (tmp_path / "journal.jsonl").read_text()
    assert '"cancelled": true' in journal
    assert '"op": "rollback"' in journal


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
