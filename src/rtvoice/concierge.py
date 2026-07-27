"""The concierge: owns the conversation, never asserts task facts.

Three structural measures replace any inspection of generated prose:
  1. Narrow context  - it sees only the current fact block, no history of state
  2. Split authorship - the reasoner writes facts; the concierge frames them
  3. Typed speech acts - validation is a schema check on typed fields
"""
from __future__ import annotations

import json
from typing import Literal, Optional

import httpx
from pydantic import BaseModel

from .registry import TaskRegistry

SYSTEM_PROMPT = """You are the voice of an assistant that controls a phone.

A separate reasoning system does all the actual work and owns all facts about
the phone. You own the conversation.

RULES:
- You may ONLY state task facts that appear in the CURRENT TASKS block below.
- When reporting a result, use act="relay", cite the task id, and include the
  exact wording given for that task verbatim. You may add conversational
  framing around it, but never reword the fact itself.
- If you do not know something, say so. Never guess at task status.
- Keep replies short and natural - they will be spoken aloud.

Reply with a single JSON object and nothing else:
  {"act": "acknowledge", "text": "..."}   brief filler while work happens
  {"act": "relay", "cites": "<task_id>", "text": "..."}  report a result
  {"act": "ask", "text": "..."}           ask the user something
  {"act": "abort", "cites": "<task_id>"}  user clearly wants a currently-active task stopped
  {"act": "chat", "text": "..."}          ordinary conversation
"""

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
    """Schema-level check on typed fields. Never inspects natural language."""
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
                "max_tokens": 200,
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
