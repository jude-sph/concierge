"""Append-only JSONL event log.

Everything downstream renders from this: the live UI subscribes, replay reads
a file, and tests assert on it. One artifact, three consumers.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Event:
    kind: str
    t_ms: int
    data: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"kind": self.kind, "t_ms": self.t_ms, "data": self.data})

    @staticmethod
    def from_dict(d: dict) -> "Event":
        return Event(kind=d["kind"], t_ms=d["t_ms"], data=d.get("data", {}))


class _AsyncSubscription:
    """Async subscription to event log with explicit cleanup."""

    def __init__(self, log: EventLog) -> None:
        self._log = log
        self._queue: asyncio.Queue = asyncio.Queue()
        self._log._queues.append(self._queue)

    def __aiter__(self):
        return self

    async def __anext__(self) -> Event:
        return await self._queue.get()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.unsubscribe()
        return False

    def unsubscribe(self) -> None:
        """Deterministically remove this subscription from the log."""
        if self._queue in self._log._queues:
            self._log._queues.remove(self._queue)


class EventLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._t0 = time.monotonic()
        self._queues: list[asyncio.Queue] = []

    def append(self, kind: str, **data) -> Event:
        ev = Event(kind=kind, t_ms=int((time.monotonic() - self._t0) * 1000), data=data)
        self._fh.write(ev.to_json() + "\n")
        self._fh.flush()
        for q in self._queues:
            q.put_nowait(ev)
        return ev

    def subscribe(self) -> _AsyncSubscription:
        """Subscribe to the log. Yields events as they are appended.

        The returned subscription can be used in async for loops:
            async for event in log.subscribe():
                process(event)

        Or as an async context manager for deterministic cleanup:
            async with log.subscribe() as sub:
                async for event in sub:
                    process(event)
        """
        return _AsyncSubscription(self)

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._queues.clear()
        self.close()
        return False

    @classmethod
    def read(cls, path: str | Path) -> list[Event]:
        """Read all events from a log file.

        Gracefully handles a truncated final line (from a crash mid-write),
        but raises on corrupted lines anywhere else.
        """
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        events = []
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                events.append(Event.from_dict(json.loads(line)))
            except json.JSONDecodeError:
                # Tolerate malformed final line (crash during append)
                if i == len(lines) - 1:
                    continue
                # Raise on corruption anywhere earlier
                raise
        return events
