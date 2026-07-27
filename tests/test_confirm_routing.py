"""Orchestrator-level regressions for the final whole-branch review.

THE core property of this system: a misheard, truncated or merely
adjacent utterance must never mutate device data without an explicit,
answer-shaped confirmation. Every test in this file pins one way that
property was violated by shipped code.
"""
import asyncio
import json

import pytest
from fakes import FakeConcierge, FakeVoice

from rtvoice.device import DeviceState
from rtvoice.events import EventLog
from rtvoice.orchestrator import Orchestrator
from rtvoice.reasoner_stub import ReasonerStub
from rtvoice.registry import TaskStatus
from rtvoice.states import TurnEvent, UserState


def make(tmp_path, **kw):
    state = tmp_path / "device_state.json"
    state.write_text(json.dumps({
        "contacts": [{"id": 1, "first_name": "Sarah", "group": "work"},
                     {"id": 2, "first_name": "Marcus", "group": "work"}],
    }))
    device = DeviceState(state, tmp_path / "journal.jsonl")
    return Orchestrator(
        reasoner=ReasonerStub(device, latency_ms=0),
        concierge=FakeConcierge(), voice=FakeVoice(),
        log=EventLog(tmp_path / "events.jsonl"), device=device, **kw,
    )


@pytest.fixture
def orch(tmp_path):
    return make(tmp_path)


def names(orch):
    return [c["first_name"] for c in orch.device.query("contacts")]


async def arm_confirm(orch, t_ms=0):
    """Get a destructive rename to AWAITING_CONFIRM and return its task id."""
    await orch.on_turn_event(
        TurnEvent(UserState.COMPLETE, "set all my contacts to Hans", t_ms))
    return next(t.task_id for t in orch.registry.all()
                if t.status == TaskStatus.AWAITING_CONFIRM)


# --- CRITICAL 1: a non-answer carrying a stray affirmative -------------------

@pytest.mark.parametrize("utterance", [
    "okay so what's on my calendar tomorrow",
    "yes I was talking to my colleague, ignore that",
    "sorry, I was on the phone, ok anyway",
])
@pytest.mark.asyncio
async def test_non_answer_with_a_stray_affirmative_never_commits(orch, utterance):
    """CRITICAL 1 regression. _dispatch used to route EVERY utterance as a
    clarification_answer while any task was AWAITING_CONFIRM, and _confirm
    only asked whether an affirmative WORD appeared anywhere in it. Each of
    these three utterances renamed every contact on the device."""
    tid = await arm_confirm(orch)

    await orch.on_turn_event(TurnEvent(UserState.COMPLETE, utterance, 1000))

    assert names(orch) == ["Sarah", "Marcus"]          # nothing mutated
    # and the confirmation is neither committed nor silently declined
    assert orch.registry.get(tid).status == TaskStatus.AWAITING_CONFIRM


@pytest.mark.asyncio
async def test_a_pending_confirm_survives_a_non_answer_and_can_still_be_answered(orch):
    """The pending question must remain answerable: it is left pending, not
    resolved either way, so a following "yes" still commits."""
    tid = await arm_confirm(orch)
    await orch.on_turn_event(
        TurnEvent(UserState.COMPLETE, "okay so what's on my calendar tomorrow", 1000))
    assert orch.registry.get(tid).status == TaskStatus.AWAITING_CONFIRM

    await orch.on_turn_event(TurnEvent(UserState.COMPLETE, "yes do it", 2000))
    assert orch.registry.get(tid).status == TaskStatus.DONE
    assert names(orch) == ["Hans", "Hans"]


@pytest.mark.asyncio
async def test_a_new_command_during_a_pending_confirm_is_dispatched_as_a_command(orch):
    """CRITICAL 1(b): _dispatch must stop swallowing plainly-new commands."""
    tid = await arm_confirm(orch)

    await orch.on_turn_event(
        TurnEvent(UserState.COMPLETE, "find chinese restaurants in soho", 1000))

    assert orch.registry.get(tid).status == TaskStatus.AWAITING_CONFIRM
    done = [t for t in orch.registry.all() if t.status == TaskStatus.DONE]
    assert [t.understood_as for t in done] == ["search: chinese restaurants in soho"]


# --- CRITICAL 2: the user's own continuation of a truncated sentence ---------

@pytest.mark.asyncio
async def test_continuation_of_a_force_dispatched_sentence_does_not_commit_it(tmp_path):
    """CRITICAL 2 regression, exact reproduced sequence. The silence timer
    force-dispatches a truncated command; the user was in fact still talking,
    and the genuine COMPLETE is a strict superset of the truncated text. Task
    13's reconciliation only matched EXACT repeats, so the superset fell into
    the "awaiting confirm" branch, its stray "okay" committed the rename, and
    the second clause was swallowed entirely."""
    orch = make(tmp_path, silence_timeout_ms=2000)

    await orch.on_turn_event(
        TurnEvent(UserState.INCOMPLETE, "set all my contacts to Hans", 0))
    await orch.on_tick(now_ms=2500)          # forced dispatch

    tasks = orch.registry.all()
    assert len(tasks) == 1
    t1 = tasks[0]
    assert t1.status == TaskStatus.AWAITING_CONFIRM

    await orch.on_turn_event(TurnEvent(
        UserState.COMPLETE,
        "set all my contacts to Hans okay and find chinese restaurants in soho",
        3000))

    # the truncated command is NOT committed by the user's own continuation
    assert orch.registry.get(t1.task_id).status == TaskStatus.AWAITING_CONFIRM
    assert names(orch) == ["Sarah", "Marcus"]
    # and the second clause was not swallowed
    assert any(t.understood_as == "search: chinese restaurants in soho"
               for t in orch.registry.all())


# --- IMPORTANT 1: pending_question is one slot, AWAITING_CONFIRM is per-task -

@pytest.mark.asyncio
async def test_bare_yes_works_when_a_later_clause_completes_first(orch):
    """IMPORTANT 1 regression. on_reasoner_messages set pending_question on
    ANY confirm_required and cleared it on ANY done/failed, so for a compound
    command the SECOND clause's done wiped the FIRST clause's question and a
    bare "yes" was silently dropped as a backchannel. Clause order decided
    whether confirming worked at all."""
    await orch.on_turn_event(TurnEvent(
        UserState.COMPLETE,
        "set all my contacts to Hans and find chinese restaurants in soho", 0))

    tid = next(t.task_id for t in orch.registry.all()
               if t.status == TaskStatus.AWAITING_CONFIRM)
    assert orch.policy_state.pending_question is not None

    await orch.on_turn_event(TurnEvent(UserState.BACKCHANNEL, "yes", 1000))

    assert orch.registry.get(tid).status == TaskStatus.DONE
    assert names(orch) == ["Hans", "Hans"]


@pytest.mark.asyncio
async def test_abort_clears_the_pending_question_so_barge_in_still_works(orch):
    """IMPORTANT 1 regression. _abort never cleared pending_question, so
    policy.decide() treated every later NONIDLE as an answer rather than a
    barge-in and reflexive "Stop!" was dead for the rest of the session."""
    tid = await arm_confirm(orch)
    await orch._abort(tid)
    await asyncio.sleep(0)   # let the fire-and-forget cancel message run

    assert orch.policy_state.pending_question is None

    orch.policy_state.speaking = True
    await orch.on_turn_event(TurnEvent(UserState.NONIDLE, "stop", 2000))
    assert orch.voice.stops == 1


@pytest.mark.asyncio
async def test_pending_question_is_cleared_when_its_own_task_finishes(orch):
    tid = await arm_confirm(orch)
    assert orch.policy_state.pending_question is not None
    await orch.on_turn_event(TurnEvent(UserState.COMPLETE, "yes do it", 1000))
    assert orch.registry.get(tid).status == TaskStatus.DONE
    assert orch.policy_state.pending_question is None


# --- IMPORTANT 5: an empty speak frame -------------------------------------

@pytest.mark.parametrize("blank", ["", "   "])
@pytest.mark.asyncio
async def test_an_empty_speak_frame_never_cancels_a_pending_task(orch, blank):
    """IMPORTANT 5 regression. "" is in _BACKCHANNELS; with a question pending
    the policy promoted it to a real answer, it reached _confirm as text=""
    and default-denied -- so a destructive task was cancelled with the user
    having said nothing at all."""
    tid = await arm_confirm(orch)

    await orch.on_turn_event(TurnEvent(UserState.BACKCHANNEL, blank, 1000))

    assert orch.registry.get(tid).status == TaskStatus.AWAITING_CONFIRM
    assert names(orch) == ["Sarah", "Marcus"]
