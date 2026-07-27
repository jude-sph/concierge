"""Journaled fake device memory with two-phase commit.

Destructive writes stage into a working copy and are only persisted on
commit(), so an abort before commit leaves nothing behind. Every operation is
journaled, which is what makes "want me to undo the 12 I already changed?"
answerable.
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
import time
from pathlib import Path

from .cancellation import Cancelled, CancellationToken


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
        return [copy.deepcopy(r) for r in self._working.get(table, []) if self._matches(r, where)]

    def update(
        self,
        table: str,
        set_fields: dict,
        where: dict | None = None,
        token: CancellationToken | None = None,
    ) -> int:
        """Stage an update. Raises Cancelled if the token fires mid-loop."""
        n = 0
        try:
            for row in self._working.get(table, []):
                if token is not None:
                    token.check()
                if self._matches(row, where):
                    row.update(set_fields)
                    n += 1
            self._journal("update", table=table, set=set_fields, where=where, rows=n)
            return n
        except Cancelled:
            self._journal("update", table=table, set=set_fields, where=where, rows=n, cancelled=True)
            raise

    def delete(
        self,
        table: str,
        where: dict | None = None,
        token: CancellationToken | None = None,
    ) -> int:
        """Stage a delete. Raises Cancelled if the token fires mid-loop.

        `where=None` empties the table. That is the most destructive thing this
        device can be asked to do, and it is deliberately expressible rather
        than special-cased away here -- the guard belongs at the point of
        consent (the reasoner names the true count before anything happens),
        not at the point of execution, which is also where an undo would have
        to reach.

        Cancellation leaves the partially-applied state in the working copy,
        exactly as `update` does, so the journal and `rollback()` tell the same
        story for both: n rows already gone, none of it committed.
        """
        rows = self._working.get(table, [])
        kept: list[dict] = []
        n = 0
        i = 0
        try:
            for i, row in enumerate(rows):
                if token is not None:
                    token.check()
                if self._matches(row, where):
                    n += 1
                else:
                    kept.append(row)
            self._working[table] = kept
            self._journal("delete", table=table, where=where, rows=n)
            return n
        except Cancelled:
            # Everything from the row that was interrupted onward is untouched.
            self._working[table] = kept + rows[i:]
            self._journal("delete", table=table, where=where, rows=n, cancelled=True)
            raise

    def insert(
        self,
        table: str,
        record: dict,
        token: CancellationToken | None = None,
    ) -> dict:
        """Stage a new record and return it, id included.

        The id is assigned here, never taken from the caller: a duplicate id
        silently makes every `where={"id": n}` afterwards ambiguous, and the
        caller in this system is a language model.
        """
        if token is not None:
            token.check()
        rows = self._working.setdefault(table, [])
        next_id = max(
            (r["id"] for r in rows if isinstance(r.get("id"), int)), default=0
        ) + 1
        created = {"id": next_id, **copy.deepcopy(record)}
        created["id"] = next_id  # last word, whatever the caller passed
        rows.append(created)
        self._journal("insert", table=table, record=created)
        return copy.deepcopy(created)

    def commit(self) -> None:
        self._committed = copy.deepcopy(self._working)
        # Atomic write: write to temp file in same directory, then replace.
        # os.replace() is atomic on POSIX systems, preventing partial-file corruption on crash.
        fd, temp_path = tempfile.mkstemp(dir=self.state_path.parent, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(self._committed, indent=2))
            os.replace(temp_path, self.state_path)
        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise
        self._journal("commit")

    def rollback(self) -> None:
        self._working = copy.deepcopy(self._committed)
        self._journal("rollback")

    def snapshot(self) -> dict:
        return copy.deepcopy(self._working)
