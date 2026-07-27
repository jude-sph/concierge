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

    async def subscribe(self):
        q: asyncio.Queue = asyncio.Queue()
        self._queues.append(q)
        try:
            while True:
                yield await q.get()
        finally:
            self._queues.remove(q)

    def close(self) -> None:
        self._fh.close()

    @classmethod
    def read(cls, path: str | Path) -> list[Event]:
        return [
            Event.from_dict(json.loads(line))
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
