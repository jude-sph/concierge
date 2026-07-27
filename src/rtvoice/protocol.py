"""Messages between orchestrator and reasoner. Async, unordered, both directions."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

ReasonerKind = Literal[
    "ack", "need_clarification", "progress", "done", "failed",
    "confirm_required", "noop",
]

OrchestratorKind = Literal[
    "utterance", "clarification_answer", "cancel", "nudge",
]


class ReasonerMessage(BaseModel):
    kind: ReasonerKind
    seq: int = 0
    task_id: Optional[str] = None
    understood_as: str = ""
    question: str = ""
    missing: str = ""
    options: list[str] = []
    status: str = ""
    result: str = ""
    reason: str = ""
    verbatim_text: str = ""


class OrchestratorMessage(BaseModel):
    kind: OrchestratorKind
    seq: int = 0
    task_id: Optional[str] = None
    text: str = ""
    raw_transcript: str = ""
    t_ms: int = 0
