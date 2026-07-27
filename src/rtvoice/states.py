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
