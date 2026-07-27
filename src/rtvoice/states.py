"""Domain turn states, derived from SoulX-Duplug's four wire states.

SoulX-Duplug's shipped server emits: idle | nonidle | speak | blank.
The paper's five semantic states are recovered as follows:

  blank            -> (no event; insufficient audio buffered)
  idle             -> user_idle
  nonidle          -> user_nonidle
  speak            -> user_complete, or user_backchannel if the text is a backchannel
  nonidle -> idle  -> user_incomplete (inferred: the model declined the turn)

The last is the important one: there is no wire state for "paused but not
finished". The absence of a `speak` between speech and silence IS the signal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

# Mirrors utils/backchannel_utils.py in Soul-AILab/SoulX-Duplug, English subset.
_BACKCHANNELS = {
    "", "mm", "mhm", "mm hm", "mmhm", "uh huh", "uhhuh", "ah", "oh", "ok",
    "okay", "yeah", "yep", "yes", "right", "sure", "hmm", "huh", "i see",
}


def is_backchannel(text: str) -> bool:
    cleaned = text.strip().lower().rstrip(".,!?").strip()
    return cleaned in _BACKCHANNELS


# --- Confirmation vocabulary -------------------------------------------------
#
# This lives beside the backchannel vocabulary on purpose. The two used to sit
# in different modules and disagree: "sure" and "right" were backchannels
# (which the turn policy promotes to real answers when a question is pending)
# but were NOT accepted as affirmatives, so answering a destructive-write
# confirmation with "sure" came back "cancelled by user". One file, one source
# of truth, and a test pins the subset relation.

# Backchannels that carry assent. Deliberately a strict subset: "mm", "hmm",
# "huh", "ah", "oh" and "i see" are acknowledgements of hearing, not of
# agreeing, and must keep default-denying a destructive write.
AFFIRMATIVE_BACKCHANNELS = {"ok", "okay", "yeah", "yep", "yes", "right", "sure"}

AFFIRMATIVE_WORDS = AFFIRMATIVE_BACKCHANNELS | {
    "yup", "ya", "correct", "confirm", "confirmed", "confirming",
    "alright", "absolutely", "definitely", "certainly", "affirmative",
}
AFFIRMATIVE_PHRASES = {
    "do it", "go ahead", "go for it", "sounds good", "please do", "yes please",
    "that's right", "thats right", "that's fine", "thats fine",
}

# Checked BEFORE affirmatives, always. "don't do it", "that is not okay" and
# "no, don't confirm it" all contain affirmative words.
NEGATION_WORDS = {
    "no", "nope", "nah", "not", "dont", "don't", "never", "cancel", "stop",
    "wait", "forget", "negative", "abort", "undo", "skip",
}
NEGATION_PHRASES = {"do not", "never mind", "nevermind", "hold on", "forget it",
                    "hold off"}

# Words that may appear around a yes/no without turning the utterance into a
# request of its own: hesitations, politeness, and the connective words that
# make up the affirmative/negation phrases above.
_ANSWER_FILLERS = {
    "um", "uh", "er", "erm", "hmm", "hm", "mm", "mhm", "mmhm", "uhhuh", "huh",
    "well", "please", "thanks", "thank", "you", "i", "it", "its", "it's",
    "that", "that's", "thats", "this", "is", "sorry",
    "do", "go", "for", "ahead", "good", "sounds", "fine", "mind", "hold",
    "off", "on", "forget",
}

ANSWER_VOCABULARY = (
    AFFIRMATIVE_WORDS
    | NEGATION_WORDS
    | _ANSWER_FILLERS
    | {w for phrase in AFFIRMATIVE_PHRASES | NEGATION_PHRASES for w in phrase.split()}
)

# A yes/no answer is short. Anything longer is a sentence, and a sentence
# carries content of its own.
MAX_ANSWER_TOKENS = 6

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def is_answer_shaped(text: str) -> bool:
    """Is this utterance shaped like an answer to a yes/no question?

    THE core property of this system is that a misheard, truncated or merely
    adjacent utterance must never mutate device data. Asking only "does an
    affirmative word appear somewhere in this string" is not enough: while a
    destructive write is awaiting confirmation, "okay so what's on my calendar
    tomorrow" and "yes I was talking to my colleague, ignore that" both
    contain one, and both committed the write.

    So the gate is default-deny by construction: an utterance is answer-shaped
    only when it is short AND made up entirely of yes/no words, negations and
    conversational filler. Any word carrying new content -- a verb, an object,
    a name, a digit -- disqualifies it. Only then does the caller apply the
    negation-first affirmative test.

    An empty or whitespace-only transcript is never an answer: an empty
    `speak` frame is real (it is in the backchannel vocabulary), and treating
    it as an answer default-denied a pending task the user never spoke about.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens or len(tokens) > MAX_ANSWER_TOKENS:
        return False
    return all(token in ANSWER_VOCABULARY for token in tokens)


class UserState(str, Enum):
    IDLE = "user_idle"
    NONIDLE = "user_nonidle"
    BACKCHANNEL = "user_backchannel"
    COMPLETE = "user_complete"
    INCOMPLETE = "user_incomplete"


@dataclass(frozen=True)
class TurnEvent:
    state: UserState
    transcript: str
    t_ms: int


class StateAdapter:
    """Stateful mapper from wire dicts to domain events. Pure; no I/O."""

    def __init__(self) -> None:
        self._prev_wire: str | None = None
        self._partial: str = ""

    def feed(self, wire: dict, t_ms: int) -> list[TurnEvent]:
        ws = wire.get("state")
        if ws == "blank":
            return []

        events: list[TurnEvent] = []

        if ws == "idle":
            if self._prev_wire == "nonidle":
                events.append(TurnEvent(UserState.INCOMPLETE, self._partial, t_ms))
            events.append(TurnEvent(UserState.IDLE, "", t_ms))
            self._partial = ""

        elif ws == "nonidle":
            self._partial = wire.get("asr_buffer", "")
            events.append(TurnEvent(UserState.NONIDLE, self._partial, t_ms))

        elif ws == "speak":
            text = wire.get("text", "")
            state = UserState.BACKCHANNEL if is_backchannel(text) else UserState.COMPLETE
            events.append(TurnEvent(state, text, t_ms))
            self._partial = ""

        self._prev_wire = ws
        return events
