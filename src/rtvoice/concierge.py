"""The concierge: the conversational voice of the system.

It talks to the person. It does not touch the phone's data and it never
reports results -- the reasoner authors those and they are spoken directly
(see Orchestrator._speak_facts). The concierge exists to make the exchange
feel like a conversation while slower work happens behind it: greeting,
acknowledging, asking for the clarification the reasoner needs, and keeping
the person informed that something is underway.

WHY THIS RETURNS PLAIN TEXT
---------------------------
It used to return a typed speech act as JSON, with a `relay` variant that had
to cite a task and repeat its wording verbatim. That machinery existed so a
small model could not invent facts while reporting results. It no longer
reports results at all, so the entire apparatus -- the JSON envelope, the
citation validation, and two regex nets that tried to rescue replies when the
envelope came back malformed -- was protecting a capability that had already
moved elsewhere.

The cost of keeping it was severe and measured: forcing structured output on a
small conversational model produced bare sentences instead of JSON (1/6 valid
on a 1.5B), pseudo-JSON that got read aloud verbatim internals and all, and a
prompt that grew past a hundred lines of argument about output format. Plain
text removes every one of those failure modes and puts a fast, small model
back within reach, which is what makes the conversation feel live.

The safety property that mattered is unchanged, and is now structural rather
than validated: the concierge is never given the phone's data, so it has
nothing to leak. Facts reach the person only from the reasoner.
"""
from __future__ import annotations

import re
from collections import deque
from typing import Optional

import httpx

from .registry import TaskRegistry

# Deliberately short. This is sent on every turn, and every line costs latency
# in a component whose whole job is to answer quickly. It says what the system
# is, what it can do, and what the concierge must not claim -- nothing else.
SYSTEM_PROMPT = """You are the voice of a phone assistant, speaking aloud.

The system you speak for can read and change what is on this phone: contacts,
messages, calendar entries and saved places. A separate reasoning component
does that work and holds the data. You never see the data yourself.

So: never say the system lacks access or cannot look something up -- it can.
The real answer is spoken separately the moment the reasoner has it, so do not
invent it, and do not repeat the person's question back at them.

Talk like a person, not a status light. Be warm, be brief, and vary how you
say things -- you are having a conversation, and hearing the same stock phrase
every single turn is worse than hearing nothing. React to the particular thing
that was said: pick up on a name, a detail, a mood. Mention what is being
looked up if it makes the sentence more natural than a bare acknowledgement.
Never open two consecutive replies the same way.

Reply with one or two short spoken sentences, and nothing else. No JSON, no
labels, no quotes around it, no placeholder text. If there is genuinely
nothing worth saying, reply with nothing at all.

Match what was actually said: greet a greeting, answer small talk, and when
the person asks for something on the phone, acknowledge it in your own words
rather than answering as if you already knew. Never simply echo back what
they said -- "Hello." answered with "Hello." is a mirror, not a conversation.

Questions about what this system can do are YOURS to answer, and you can
answer them properly: it looks things up and makes changes across the
contacts, messages, calendar and saved places on this phone -- finding
records, renaming them, adding them, deleting them. Say so naturally and
concretely, in a sentence or two, rather than deflecting.

You may be shown a LOOKING UP line naming what the reasoner is fetching right
now. That is background state for you, never something to read out word for
word, and never something to answer yourself."""

# Long enough for a spoken sentence, short enough that a rambling model gets
# cut off rather than monologuing at the person.
MAX_REPLY_CHARS = 240

# How many previous replies are held to check for repetition and shown back to
# the model. Enough to catch "I'm onto it." every turn; short enough that a
# genuinely apt short reply ("Sure.") is not banned for the whole session.
RECENT_REPLIES = 4


# The labels this module puts in front of its own state, matched only in
# LABEL-SHAPED position: at the start of the reply and followed by a colon.
# The bare words are ordinary English -- "Looking up your calendar now." is a
# perfectly good thing to say, and must not be silenced for beginning with two
# words that also happen to name one of our headings.
_CONTROL_ECHO = re.compile(
    r"^\s*(LOOKING UP|TASKS|ALREADY SAID|CURRENT STATE|AVOID OPENING)\b[^:\n]{0,20}:",
    re.IGNORECASE,
)

# How many opening words identify a reply. Three, not more: "I'm onto it." and
# "I'm onto it right now!" are the same tic, and a longer key lets it slip
# through by adding a word.
_PHRASE_KEY_WORDS = 3


def _phrase_key(text: str) -> str:
    """A reply's identity for repetition purposes.

    Compares the OPENING of the sentence, case- and punctuation-insensitively.
    Two replies that merely start alike are treated as the same, which is the
    intent: the prompt asks for no two consecutive replies opening the same
    way, and the only cost of a false positive is one re-roll.
    """
    words = "".join(c for c in text.lower() if c.isalnum() or c.isspace()).split()
    return " ".join(words[:_PHRASE_KEY_WORDS])


def clean_reply(text: str) -> str:
    """Normalise a model reply into something safe to speak.

    Small models wrap replies in quotes, prefix them with a speaker label, or
    fence them as code even when asked for a bare sentence. None of that
    should reach a text-to-speech engine, and a placeholder like "..." is read
    aloud literally, so it is dropped entirely rather than spoken.
    """
    reply = (text or "").strip()

    if reply.startswith("```"):
        reply = reply.split("\n", 1)[-1] if "\n" in reply else ""
        reply = reply.rsplit("```", 1)[0].strip()

    # "Assistant: hello" / "concierge - hello"
    for label in ("assistant:", "concierge:", "reply:", "response:"):
        if reply.lower().startswith(label):
            reply = reply[len(label) :].strip()

    if len(reply) >= 2 and reply[0] == reply[-1] and reply[0] in "\"'":
        reply = reply[1:-1].strip()

    # Last line of defence against the model reading this system's internal
    # state out loud. The real fix is positional (see respond(): the state
    # block goes at the top of the single system message, with the whole
    # conversation between it and the point of generation) -- but the failure
    # mode here is speaking internal bookkeeping AT a person, twice observed
    # in live use, so a reply that opens with one of our own labels is
    # discarded rather than spoken. Silence is always a safe reply; this one
    # never was.
    if _CONTROL_ECHO.match(reply):
        return ""

    # A reply that is only punctuation or an ellipsis is a placeholder, not
    # speech. Saying nothing is better than saying "dot dot dot".
    if not any(ch.isalnum() for ch in reply):
        return ""

    if len(reply) > MAX_REPLY_CHARS:
        cut = reply[:MAX_REPLY_CHARS]
        stop = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
        reply = cut[: stop + 1] if stop > 40 else cut.rstrip() + "."

    return reply


class Concierge:
    """Speaks to the person. Holds no device data, reports no results."""

    def __init__(
        self,
        base_url: str = "http://localhost:8001/v1",
        model: str = "Qwen/Qwen2.5-3B-Instruct",
        timeout: float = 20.0,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self._client = httpx.AsyncClient(timeout=timeout)
        # The last few things actually said. A small model asked to
        # acknowledge a request converges hard on one phrase and then says it
        # every single turn, which is the single most robot-like thing this
        # component does. The prompt asks for variety; this makes the model
        # face what it already said, and re-rolls once if it repeats anyway.
        self._recent: deque[str] = deque(maxlen=RECENT_REPLIES)
        # Kept for the instruments: how often a reply had to be discarded as
        # unspeakable. It is no longer a JSON-schema violation count, but it
        # measures the same thing -- generations this component could not use.
        self.violations = 0
        # How often a reply had to be re-rolled for repeating a recent one.
        self.repeats = 0

    async def respond(
        self,
        registry: TaskRegistry,
        history: list[dict],
        trigger: str,
        in_flight: Optional[str] = None,
    ) -> str:
        """Return one short spoken sentence, or "" to stay silent.

        `in_flight` is what the reasoner was just handed, if anything. Without
        it the concierge is asked to speak at the exact moment nothing is
        known yet -- which is how it ended up denying capabilities it has,
        telling a person it had no information about their calendar while a
        calendar lookup was already running. Being told what is underway is
        what lets "let me check that" be a true statement rather than a guess.
        """
        # Phrased as terse STATE, never as an instruction. A system message
        # worded as a direction ("say you are onto it, do not answer it") sat
        # immediately before the model's turn and got reproduced verbatim as
        # the spoken reply -- the person heard "Say you are onto it. Do not
        # answer." The behavioural rule for this line lives in SYSTEM_PROMPT
        # instead, well away from the generation point.
        context = []
        if in_flight:
            context.append(f"LOOKING UP: {in_flight}")
        tasks = registry.fact_block()
        if tasks:
            context.append(f"TASKS:\n{tasks}")
        if self._recent:
            # Also terse state, for the same reason the LOOKING UP line is:
            # anything phrased as an instruction this close to the generation
            # point gets read out loud verbatim by a small model.
            context.append("ALREADY SAID (do not reuse):\n"
                           + "\n".join(f"- {r}" for r in self._recent))

        # ONE system message, always, with the state folded into it -- never a
        # second one appended after it. A separate context message sits
        # immediately before the model's turn, and a small model reproduces
        # the most recent instruction-shaped text it can see: the person heard
        # "Say you are onto it. Do not answer." spoken aloud, and later, after
        # that was reworded to terse state, heard "LOOKING UP: Please change
        # the contact." read out as the reply. Rewording it was treating the
        # symptom. Position was the cause, so the state now goes at the very
        # start of the prompt with the whole conversation between it and the
        # point of generation.
        system = SYSTEM_PROMPT
        if context:
            system = f"{SYSTEM_PROMPT}\n\nCURRENT STATE (never read aloud):\n" \
                     + "\n\n".join(context)
        messages = [{"role": "system", "content": system}]
        messages.extend(history[-8:])

        reply = await self._generate(messages, temperature=0.8)

        # One re-roll, and only when it actually repeated itself. Doing this
        # unconditionally would double every turn's latency in the one
        # component whose whole job is to answer fast.
        if reply and _phrase_key(reply) in {_phrase_key(r) for r in self._recent}:
            self.repeats += 1
            retry = messages + [{
                "role": "system",
                "content": f"AVOID OPENING: {reply}",
            }]
            fresh = await self._generate(retry, temperature=1.0)
            if fresh:
                reply = fresh

        if not reply:
            self.violations += 1
        else:
            self._recent.append(reply)
        return reply

    async def _generate(self, messages: list[dict], temperature: float) -> str:
        try:
            resp = await self._client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 60,
                    "temperature": temperature,
                },
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
        except Exception:
            # The conversational layer is not allowed to take down a turn.
            # The reasoner's work continues regardless and its result is
            # spoken by a path that does not involve this component at all.
            return ""
        return clean_reply(raw)

    async def aclose(self) -> None:
        await self._client.aclose()
