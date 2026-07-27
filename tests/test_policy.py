# tests/test_policy.py
from rtvoice.policy import AskConcierge, PolicyState, SendUtterance, Stop, decide
from rtvoice.states import TurnEvent, UserState


def ev(state, transcript=""):
    return TurnEvent(state=state, transcript=transcript, t_ms=0)


def test_incomplete_does_nothing():
    """The whole reason SoulX-Duplug is here: hold through mid-sentence pauses."""
    assert decide(ev(UserState.INCOMPLETE, "find food and"), PolicyState()) == []


def test_backchannel_does_not_stop_speech():
    assert decide(ev(UserState.BACKCHANNEL, "mm hm"), PolicyState(speaking=True)) == []


def test_bare_yes_answering_a_pending_question_is_a_real_turn():
    """A bare "yes" is a backchannel by wording but an ANSWER when we are holding
    a question. Without this, confirming a destructive write with "yes" is dropped."""
    state = PolicyState(pending_question="This will rename 47 contacts. Confirm?")
    actions = decide(ev(UserState.BACKCHANNEL, "yes"), state)
    assert actions == [
        SendUtterance(text="yes"),
        AskConcierge(trigger="user_turn"),
    ]


def test_backchannel_with_no_pending_question_is_still_ignored():
    assert decide(ev(UserState.BACKCHANNEL, "mm hm"), PolicyState()) == []


def test_nonidle_while_speaking_stops_reflexively():
    actions = decide(ev(UserState.NONIDLE, "wait"), PolicyState(speaking=True))
    assert actions == [Stop()]


def test_nonidle_while_silent_does_nothing():
    assert decide(ev(UserState.NONIDLE, "hi"), PolicyState(speaking=False)) == []


def test_nonidle_during_pending_question_is_an_answer_not_a_bargein():
    state = PolicyState(speaking=True, pending_question="which contacts?")
    assert decide(ev(UserState.NONIDLE, "the work ones"), state) == []


def test_complete_dispatches_and_asks_concierge():
    actions = decide(ev(UserState.COMPLETE, "rename my contacts"), PolicyState())
    assert actions == [
        SendUtterance(text="rename my contacts"),
        AskConcierge(trigger="user_turn"),
    ]


def test_complete_while_speaking_stops_first():
    actions = decide(ev(UserState.COMPLETE, "stop"), PolicyState(speaking=True))
    assert actions[0] == Stop()
    assert SendUtterance(text="stop") in actions


def test_idle_does_nothing():
    assert decide(ev(UserState.IDLE), PolicyState()) == []


def test_empty_backchannel_is_never_an_answer_to_a_pending_question():
    """IMPORTANT 5: an empty `speak` frame classifies as a backchannel ("" is
    in the backchannel vocabulary). Promoting it to a real answer sends an
    empty clarification_answer to the reasoner, which default-denies -- so a
    pending destructive task is cancelled with the user having said nothing."""
    state = PolicyState(pending_question="This will rename 47 contacts. Confirm?")
    assert decide(ev(UserState.BACKCHANNEL, ""), state) == []
    assert decide(ev(UserState.BACKCHANNEL, "   "), state) == []
