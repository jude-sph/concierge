"""The concierge: owns the conversation, never asserts task facts.

Three structural measures replace any inspection of generated prose:
  1. Narrow context  - it sees only the current fact block, no history of state
  2. Split authorship - the reasoner writes facts; the concierge frames them
  3. Typed speech acts - validation is a schema check on typed fields
"""
from __future__ import annotations

import json
import re
from typing import Literal, Optional

import httpx
from pydantic import BaseModel

from .registry import TaskRegistry

SYSTEM_PROMPT = """You are the spoken voice of a phone assistant.

WHO YOU ARE, AND WHAT THE SYSTEM CAN ACTUALLY DO:
The system you are the voice of CAN read and modify what's on this phone -
contacts, messages, calendar entries, saved places. It can look any of that
up, change it, delete it, add to it. A separate reasoning component is the
one that actually does this: it queries the phone, and it owns every fact
about what's on it. You never hold that data yourself and you never see it
directly - only the reasoner's results, and only once they're ready, land in
CURRENT TASKS below and get handed to you a moment later.

This split is about WHO holds the data, never about what the system CAN do.
So when the user asks about their phone's data - "what's on my calendar",
"do you have my contacts", "can you check my messages" - that is real work
for the reasoner, not a request you personally can't help with. NEVER say or
imply "I don't have access", "I don't have information about X", or anything
else that denies the capability - it's false, and it makes a system that
just hasn't answered YET sound like one that is broken. Instead, talk the
way a capable person would while someone else looks something up: "Let me
check your calendar.", "One sec, pulling that up.", "Sure, let me look." The
real answer follows on its own, spoken separately, once the reasoner has it.

Not knowing a fact YET is completely different from the system being unable
to find out - keep that distinction sharp, because the rule below never
softens: you may describe yourself as *looking something up*, but you may
never *state* what it turns out to be until it's actually in front of you.

RULES:
- State a task fact ONLY if it appears in CURRENT TASKS below. This is
  absolute and applies no matter how confident you feel or how obviously
  the system CAN do something - capability is never a fact you're allowed to
  invent details for.
- To report a result: act="relay", cite the task id, and include that task's
  exact wording verbatim. Frame it however you like, never reword the fact.
- Don't know a fact yet? Say you're checking, or that you'll find out -
  never guess at task status, and never claim the system can't do the
  lookup just because you don't have the answer in hand yet.
- Every reply is spoken aloud: one short, natural sentence. NEVER output a
  placeholder like "..." - that gets read aloud literally. Write a real
  sentence, or empty text if there is truly nothing worth saying.
- Trigger "user_turn": reply right away, but make the reply FIT what was
  actually said - never a default "On it" regardless of the utterance:
    - A greeting, thanks, or small talk gets a matching conversational
      reply (act="chat"). "Hello" -> "Hi there!", never "On it."
    - A question you can just answer conversationally gets answered
      (act="chat" or "ask") - don't acknowledge a question as if it were
      a task.
    - A request to look up, change, add, or remove something on the phone
      (contacts, messages, calendar, places) IS real work for the reasoner,
      even though you don't have the answer yourself - acknowledge it as
      being looked into (act="acknowledge", e.g. "Let me check your
      calendar.", "Sure, looking that up.") rather than answering as if you
      already know, and never as a denial of capability.
    - Only an instruction or request that hands the reasoner real work
      to do gets a short acknowledgement (act="acknowledge", e.g. "On it.",
      "Sure, one sec.") - filler like this belongs ONLY here, never as a
      reflex to every turn.
- Trigger "reasoner_update": speak only if there's something worth relaying,
  asking, or aborting - otherwise reply with empty text.

OUTPUT FORMAT - read carefully:
Your ENTIRE reply must be ONE JSON object and nothing else: no prose before
or after it, no markdown code fences, no bare sentence on its own. The words
you'd speak go ONLY inside that object's "text" field - they are a field
VALUE, never your reply by itself.

WRONG (a bare sentence - this is not JSON, it will be rejected):
  On it.

RIGHT (the same words, as the "text" field of a JSON object):
  {"act": "acknowledge", "text": "On it."}

More valid envelopes - again, the quoted words are field values inside the
object, never output on their own:
  {"act": "relay", "cites": "t1", "text": "Found it - renamed 47 contacts."}
  {"act": "ask", "text": "Which one did you mean?"}
  {"act": "abort", "cites": "t1"}
  {"act": "chat", "text": "Sure, what else?"}
"""

# A model that ignores the envelope instruction and emits the bare spoken
# sentence instead of the JSON object wrapping it (empirically: raw='On it.'
# against a real 7B model, on every call of a six-call run). A prompt rule is
# not a guarantee, so this is a second, structural line of defence: if the
# body fails to parse as JSON at all, but looks like ordinary short spoken
# prose rather than a mangled/truncated JSON attempt, salvage it as a safe
# act instead of losing the whole turn to a JSONDecodeError. Anything that
# still contains brace/bracket punctuation is presumed to be broken JSON, not
# prose, and is left to raise - only true bare-text replies are salvaged.
_BARE_REPLY_MAX_CHARS = 200

# A live session produced a body with NO brace/bracket punctuation that still
# was not prose: a pseudo-JSON serialization of the speech act itself --
# act="relay", cites="t3", text="...verbatim internals...". The brace/bracket
# check above let it straight through, and it was spoken aloud verbatim,
# internals and all. This is the second, narrower net: any of the schema's
# own field names used as a key (quoted or not, "=" or ":" as the separator),
# or the general shape of a serialized key="value" pair regardless of field
# name, marks `text` as structured data rather than a sentence a person
# would say. When in doubt here, the reply is NOT salvaged -- it is left to
# raise, exactly like broken JSON -- because silence is far better than
# reading internals aloud.
_SPEECH_ACT_FIELD_RE = re.compile(
    r"""["']?\b(act|cites|text|kind|task_id|understood_as)\b["']?\s*[:=]""",
    re.IGNORECASE,
)
_KEY_VALUE_RE = re.compile(r"""[A-Za-z_]\w*\s*=\s*["']""")


def _looks_like_structured_data(text: str) -> bool:
    """True if `text` reads as serialized fields rather than a sentence a
    person would actually say."""
    return bool(_SPEECH_ACT_FIELD_RE.search(text) or _KEY_VALUE_RE.search(text))


def _looks_like_bare_reply(text: str) -> bool:
    text = text.strip()
    if not text or len(text) > _BARE_REPLY_MAX_CHARS:
        return False
    if "{" in text or "[" in text:
        return False
    return not _looks_like_structured_data(text)


# A reply that is nothing but an unfilled template slot - the model copying
# "..." (or a bracketed placeholder) straight out of the format spec above
# instead of writing a real sentence. Caught here, not just warned against in
# the prompt, because "the model was told not to" is not a guarantee: this
# is exactly the failure a live session produced. Matches the WHOLE reply
# (not e.g. a trailing ellipsis inside real prose like "hold on...").
_PLACEHOLDER_RE = re.compile(r"^(\.{2,}|…|\[[.\s…]*\]|<[.\s…]*>)$")

SPEECH_ACT_SCHEMA = {
    "type": "object",
    "properties": {
        "act": {"type": "string", "enum": ["relay", "ask", "acknowledge", "abort", "chat"]},
        "text": {"type": "string"},
        "cites": {"type": ["string", "null"]},
    },
    "required": ["act"],
}


class SpeechAct(BaseModel):
    act: Literal["relay", "ask", "acknowledge", "abort", "chat"]
    text: str = ""
    cites: Optional[str] = None


def validate_act(act: SpeechAct, registry: TaskRegistry) -> tuple[bool, str]:
    """Schema-level check on typed fields, plus one narrow structural check on
    the text: is it a real sentence, or an unfilled placeholder copied from
    the format spec? That is a shape check ("is this a template slot"), not an
    inspection of what the sentence claims - facts are still policed only via
    the relay/cites/verbatim-span checks below.
    """
    if act.text and _PLACEHOLDER_RE.match(act.text.strip()):
        return False, f"reply text is a placeholder, not a real sentence: {act.text!r}"
    if act.act == "relay":
        if not act.cites:
            return False, "relay requires cites"
        if registry.get(act.cites) is None:
            return False, f"unknown task id: {act.cites}"
        span = registry.verbatim_span(act.cites)
        if not span:
            return False, f"no fact recorded yet for task {act.cites}"
        if span not in act.text:
            return False, f"relay must contain the verbatim span: {span!r}"
    if act.act == "abort":
        if not act.cites or act.cites not in registry.live_ids():
            return False, "abort requires a live task id"
    return True, ""


class Concierge:
    """Async LLM client for generating validated speech acts.

    Attributes:
        violations: Count of invalid speech act generations (not turns). Incremented
            each time a model generation fails schema validation, or arrives as a bare,
            unparseable-as-JSON reply salvaged into a safe act (see `_complete`). A
            single turn can contribute up to 2 to this counter (initial attempt plus
            one re-prompt before fallback). Use this metric to track generation
            quality, not turn success rate.
    """
    def __init__(
        self,
        base_url: str = "http://localhost:8001/v1",
        model: str = "Qwen/Qwen3-4B",
        timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self._client = httpx.AsyncClient(timeout=timeout)
        self.violations = 0  # invalid generations, not turns

    async def _complete(self, messages: list[dict]) -> tuple[SpeechAct, bool]:
        """Returns (act, salvaged). `salvaged` is True when the body wasn't
        valid JSON at all but looked like ordinary short spoken prose, so it
        was wrapped into a safe chat act rather than raising - the caller
        still must count that as an invalid generation."""
        resp = await self._client.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                # One short spoken sentence plus a little JSON scaffolding
                # (act/cites/text) - even the longest verbatim span this ever
                # has to carry (an unfiltered-delete confirmation) fits with
                # room to spare. Kept well under the old 200 so a slow model
                # can't burn the turn's latency budget on an oversized reply.
                "max_tokens": 100,
                "temperature": 0.6,
                "guided_json": SPEECH_ACT_SCHEMA,
            },
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        try:
            return SpeechAct(**json.loads(content)), False
        except json.JSONDecodeError:
            if not _looks_like_bare_reply(content):
                raise
            # Salvage, never relay: a bare reply carries no citation and no
            # verified verbatim span, so the only thing it can ever become is
            # a safe, non-factual act. act="chat" is hardcoded here - never
            # derived from the model's text - so this path can never produce
            # a "relay".
            return SpeechAct(act="chat", text=content.strip()), True

    async def respond(
        self, registry: TaskRegistry, history: list[dict], trigger: str
    ) -> SpeechAct:
        """Generate one speech act. Re-prompts once on schema violation, then
        falls back to acknowledge. Never rewrites the model's output."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"CURRENT TASKS:\n{registry.fact_block()}"},
            *history,
            {"role": "user", "content": f"[trigger: {trigger}]"},
        ]

        act, salvaged = await self._complete(messages)
        ok, err = validate_act(act, registry)
        if ok:
            if salvaged:
                self.violations += 1
            return act

        self.violations += 1
        messages.append({"role": "system", "content": f"Rejected: {err}. Try again."})
        act, salvaged = await self._complete(messages)
        ok, _ = validate_act(act, registry)
        if ok:
            if salvaged:
                self.violations += 1
            return act

        self.violations += 1
        return SpeechAct(act="acknowledge", text="Let me check on that.")

    async def aclose(self) -> None:
        await self._client.aclose()
