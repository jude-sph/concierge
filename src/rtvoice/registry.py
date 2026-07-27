"""Single source of truth for task state, mirrored from reasoner messages.

The concierge never infers status; it reads fact_block(). Verbatim spans are
stored separately so the concierge's relays can be checked against them.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from .protocol import ReasonerMessage


class TaskStatus(str, Enum):
    PENDING = "pending"
    AWAITING_CONFIRM = "awaiting_confirm"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL = {TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED}


@dataclass
class Task:
    task_id: str
    understood_as: str
    status: TaskStatus = TaskStatus.PENDING
    detail: str = ""
    created_ms: int = field(default_factory=lambda: int(time.monotonic() * 1000))
    updated_ms: int = field(default_factory=lambda: int(time.monotonic() * 1000))


class TaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._verbatim: dict[str, str] = {}

    def apply(self, msg: ReasonerMessage) -> None:
        if msg.kind == "noop" or msg.task_id is None:
            return

        tid = msg.task_id
        if tid not in self._tasks:
            self._tasks[tid] = Task(task_id=tid, understood_as=msg.understood_as)

        task = self._tasks[tid]

        # Terminal-state guard: once a task reaches a terminal state, don't mutate it.
        # Stale messages arriving out of order must not regress status or clobber verbatim spans.
        if task.status in TERMINAL:
            return

        task.updated_ms = int(time.monotonic() * 1000)

        if msg.kind == "ack":
            task.understood_as = msg.understood_as or task.understood_as
            task.status = TaskStatus.PENDING
        elif msg.kind == "confirm_required":
            task.status = TaskStatus.AWAITING_CONFIRM
            task.detail = msg.verbatim_text
            self._verbatim[tid] = msg.verbatim_text
        elif msg.kind == "need_clarification":
            task.status = TaskStatus.AWAITING_CONFIRM
            task.detail = msg.missing
        elif msg.kind == "progress":
            task.status = TaskStatus.RUNNING
            task.detail = msg.status
        elif msg.kind == "done":
            task.status = TaskStatus.DONE
            task.detail = msg.result
            self._verbatim[tid] = msg.result
        elif msg.kind == "failed":
            task.status = TaskStatus.FAILED
            task.detail = msg.reason
            self._verbatim[tid] = msg.reason

    def mark_cancelled(self, task_id: str) -> None:
        if task_id in self._tasks:
            task = self._tasks[task_id]
            # Don't change status if already in a terminal state
            if task.status not in TERMINAL:
                task.status = TaskStatus.CANCELLED

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def all(self) -> list[Task]:
        return list(self._tasks.values())

    def live_ids(self) -> list[str]:
        return [t.task_id for t in self._tasks.values() if t.status not in TERMINAL]

    def verbatim_span(self, task_id: str) -> str | None:
        return self._verbatim.get(task_id)

    def fact_block(self) -> str:
        """The complete, current, authoritative fact set given to the concierge.

        Deliberately contains no history — gaps in stale state are what
        invented task status is made of.
        """
        if not self._tasks:
            return "No tasks are in progress."
        lines = ["Current tasks (these are the ONLY task facts you may state):"]
        for t in self._tasks.values():
            line = f"- [{t.task_id}] {t.understood_as} — status: {t.status.value}"
            if t.detail:
                line += f' — exact wording to use: "{t.detail}"'
            lines.append(line)
        return "\n".join(lines)
