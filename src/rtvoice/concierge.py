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
If you do not have an answer yet, say you are checking. The real answer is
spoken separately the moment the reasoner has it, so do not invent it, and do
not repeat the person's question back at them.

Reply with one short spoken sentence, and nothing else. No JSON, no labels,
no quotes around it, no placeholder text. If there is genuinely nothing worth
saying, reply with nothing at all.

Match what was actually said: greet a greeting, answer small talk, and when
the person asks for something on the phone, say you are onto it rather than
answering as if you already knew."""

# Long enough for a spoken sentence, short enough that a rambling model gets
# cut off rather than monologuing at the person.
MAX_REPLY_CHARS = 240


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
        # Kept for the instruments: how often a reply had to be discarded as
        # unspeakable. It is no longer a JSON-schema violation count, but it
        # measures the same thing -- generations this component could not use.
        self.violations = 0

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
        context = []
        if in_flight:
            context.append(
                f"You have just passed this to the reasoner and it is working on "
                f"it now: {in_flight!r}. Say you are onto it. Do not answer it."
            )
        tasks = registry.fact_block()
        if tasks:
            context.append(f"Tasks already known:\n{tasks}")

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if context:
            messages.append({"role": "system", "content": "\n\n".join(context)})
        messages.extend(history[-8:])

        try:
            resp = await self._client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 60,
                    "temperature": 0.7,
                },
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
        except Exception:
            # The conversational layer is not allowed to take down a turn.
            # The reasoner's work continues regardless and its result is
            # spoken by a path that does not involve this component at all.
            self.violations += 1
            return ""

        reply = clean_reply(raw)
        if not reply:
            self.violations += 1
        return reply

    async def aclose(self) -> None:
        await self._client.aclose()
