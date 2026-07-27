"""Turn policy: a pure function from (domain event, system state) to actions.

No models, no I/O. This is the spec's turn-policy table made executable.
"""
from __future__ import annotations

from dataclasses import dataclass

from .states import TurnEvent, UserState


@dataclass(frozen=True)
class Stop:
    """Halt TTS immediately. Reflexive — no inference, harmless, reversible."""


@dataclass(frozen=True)
class SendUtterance:
    text: str


@dataclass(frozen=True)
class AskConcierge:
    trigger: str


Action = Stop | SendUtterance | AskConcierge


@dataclass
class PolicyState:
    speaking: bool = False
    pending_question: str | None = None


def decide(event: TurnEvent, state: PolicyState) -> list[Action]:
    if event.state in (UserState.IDLE, UserState.INCOMPLETE):
        # INCOMPLETE: the user paused mid-sentence. Wait. This is the point.
        return []

    if event.state is UserState.NONIDLE:
        # Barge-in is reflexive, unless we are holding a question, in which
        # case speech is the answer to it rather than an interruption.
        if state.speaking and state.pending_question is None:
            return [Stop()]
        return []

    # A bare "yes"/"ok" reads as a backchannel by wording, but it is a real
    # answer when we are holding a question. Disambiguate here, in the layer
    # that has the dialogue context - the adapter deliberately does not.
    #
    # An EMPTY transcript is the exception that proves the rule: "" is in the
    # backchannel vocabulary, so a `speak` frame with no text arrives here as
    # a BACKCHANNEL. Promoting silence to an answer sent an empty
    # clarification_answer to the reasoner, which default-denied it and
    # cancelled a pending destructive task the user had said nothing about.
    # Silence is never an answer.
    answers_question = (
        event.state is UserState.BACKCHANNEL
        and state.pending_question is not None
        and bool(event.transcript.strip())
    )
    if event.state is UserState.BACKCHANNEL and not answers_question:
        # "mm hm" mid-utterance is not a turn grab. Keep speaking.
        return []

    if event.state is UserState.COMPLETE or answers_question:
        actions: list[Action] = []
        if state.speaking:
            actions.append(Stop())
        # Every finalised utterance goes to the reasoner - it is the gatekeeper.
        # The concierge responds in parallel, never waiting for it.
        actions.append(SendUtterance(text=event.transcript))
        actions.append(AskConcierge(trigger="user_turn"))
        return actions

    return []
