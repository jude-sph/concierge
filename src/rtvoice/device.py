"""Journaled fake device memory with two-phase commit.

Destructive writes stage into a working copy and are only persisted on
commit(), so an abort before commit leaves nothing behind. Every operation is
journaled, which is what makes "want me to undo the 12 I already changed?"
answerable.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

from .cancellation import CancellationToken


class DeviceState:
    def __init__(self, state_path: str | Path, journal_path: str | Path) -> None:
        self.state_path = Path(state_path)
        self.journal_path = Path(journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self._committed = json.loads(self.state_path.read_text(encoding="utf-8"))
        self._working = copy.deepcopy(self._committed)

    def _journal(self, op: str, **kw) -> None:
        rec = {"op": op, "ts": time.time(), **kw}
        with self.journal_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    @staticmethod
    def _matches(row: dict, where: dict | None) -> bool:
        return where is None or all(row.get(k) == v for k, v in where.items())

    def query(self, table: str, where: dict | None = None) -> list[dict]:
        return [r for r in self._working.get(table, []) if self._matches(r, where)]

    def update(
        self,
        table: str,
        set_fields: dict,
        where: dict | None = None,
        token: CancellationToken | None = None,
    ) -> int:
        """Stage an update. Raises Cancelled if the token fires mid-loop."""
        n = 0
        for row in self._working.get(table, []):
            if token is not None:
                token.check()
            if self._matches(row, where):
                row.update(set_fields)
                n += 1
        self._journal("update", table=table, set=set_fields, where=where, rows=n)
        return n

    def commit(self) -> None:
        self._committed = copy.deepcopy(self._working)
        self.state_path.write_text(
            json.dumps(self._committed, indent=2), encoding="utf-8"
        )
        self._journal("commit")

    def rollback(self) -> None:
        self._working = copy.deepcopy(self._committed)
        self._journal("rollback")

    def snapshot(self) -> dict:
        return copy.deepcopy(self._working)
