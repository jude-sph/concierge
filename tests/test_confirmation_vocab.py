"""IMPORTANT 6: the backchannel vocabulary and the confirmation vocabulary
used to live in two modules and disagree. `states.py` classified "sure" and
"right" as backchannels -- which the policy promotes to real answers when a
question is pending -- while `reasoner_stub.py`'s affirmative regex did not
accept them, so answering a destructive-write confirmation with "sure" came
back FAILED "cancelled by user". Fail-safe, but wrong.

These tests pin the two sets to a single source of truth so they cannot drift
apart again.
"""
import pytest

from rtvoice.states import (
    AFFIRMATIVE_BACKCHANNELS,
    AFFIRMATIVE_WORDS,
    ANSWER_VOCABULARY,
    is_answer_shaped,
    is_backchannel,
)


def test_every_affirmative_backchannel_is_also_an_affirmative_answer():
    assert AFFIRMATIVE_BACKCHANNELS <= AFFIRMATIVE_WORDS


def test_every_affirmative_backchannel_is_still_a_backchannel():
    """If a word stops being a backchannel it stops being promoted to an
    answer at all, which would break confirmation from the other side."""
    for word in AFFIRMATIVE_BACKCHANNELS:
        assert is_backchannel(word), word


def test_every_affirmative_backchannel_is_answer_shaped_on_its_own():
    for word in AFFIRMATIVE_BACKCHANNELS:
        assert is_answer_shaped(word), word


def test_affirmative_words_are_all_in_the_answer_vocabulary():
    assert AFFIRMATIVE_WORDS <= ANSWER_VOCABULARY


@pytest.mark.parametrize("text", [
    "yes", "no", "yes do it", "don't do it", "no, don't confirm it",
    "that is not okay", "sure", "right", "hmm", "go ahead", "please don't",
    "yeah ok", "nope", "cancel that",
])
def test_answer_shaped_accepts_real_answers(text):
    assert is_answer_shaped(text)


@pytest.mark.parametrize("text", [
    "",
    "   ",
    "okay so what's on my calendar tomorrow",
    "yes I was talking to my colleague, ignore that",
    "sorry, I was on the phone, ok anyway",
    "set all my contacts to Hans okay and find chinese restaurants in soho",
    "find chinese restaurants in soho",
    "yes rename them to Hans",
    "ok text Marcus and tell him I'm running late",
    "no idea, ask my wife about the dinner reservation",
])
def test_answer_shaped_rejects_anything_carrying_its_own_request(text):
    assert not is_answer_shaped(text)
