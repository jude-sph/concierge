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

A separate reasoning system does the real work and owns every fact about the
phone. You own only the conversation.

RULES:
- State a task fact ONLY if it appears in CURRENT TASKS below.
- To report a result: act="relay", cite the task id, and include that task's
  exact wording verbatim. Frame it however you like, never reword the fact.
- Don't know something? Say so - never guess at task status.
- Every reply is spoken aloud: one short, natural sentence. NEVER output a
  placeholder like "..." - that gets read aloud literally. Write a real
  sentence, or empty text if there is truly nothing worth saying.
- Trigger "user_turn": acknowledge right away, even before results exist.
- Trigger "reasoner_update": speak only if there's something worth relaying,
  asking, or aborting - otherwise reply with empty text.

Reply with a single JSON object and nothing else:
  {"act": "acknowledge", "text": "On it."}
  {"act": "relay", "cites": "t1", "text": "Found it - renamed 47 contacts."}
  {"act": "ask", "text": "Which one did you mean?"}
  {"act": "abort", "cites": "t1"}
  {"act": "chat", "text": "Sure, what else?"}
"""

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
            each time a model generation fails schema validation. A single turn can
            contribute up to 2 to this counter (initial attempt + one re-prompt before
            fallback). Use this metric to track generation quality, not turn success rate.
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

    async def _complete(self, messages: list[dict]) -> SpeechAct:
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
        return SpeechAct(**json.loads(content))

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

        act = await self._complete(messages)
        ok, err = validate_act(act, registry)
        if ok:
            return act

        self.violations += 1
        messages.append({"role": "system", "content": f"Rejected: {err}. Try again."})
        act = await self._complete(messages)
        ok, _ = validate_act(act, registry)
        if ok:
            return act

        self.violations += 1
        return SpeechAct(act="acknowledge", text="Let me check on that.")

    async def aclose(self) -> None:
        await self._client.aclose()
