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
    def _value_matches(actual, expected) -> bool:
        """Equality across the type sloppiness of a model-authored filter.

        The `where` clause is written by a language model from spoken input,
        and the stored value is whatever the fixture happens to hold. Two
        mismatches show up constantly and mean nothing semantically:

          * a number quoted as a string. The planner emitted
            `where={"id": "11"}` against a row holding `{"id": 11}`, and the
            reasoner reported "no contacts matched" for a contact plainly
            visible on screen.
          * a difference in case or surrounding whitespace, which for a name
            arriving via speech recognition is not a difference at all.

        This stays EXACT-VALUE matching -- it is deliberately not substring or
        prefix matching, because this predicate also selects the rows that
        `delete` removes, and a loosened one would silently widen the blast
        radius of every destructive write.

        Booleans are compared strictly: in Python `True == 1`, so without this
        guard a filter of `starred=1` would match `starred=False`'s sibling
        rows by accident, and `starred=True` would match a literal 1.
        """
        if isinstance(actual, bool) or isinstance(expected, bool):
            return actual is expected

        if actual == expected:
            return True

        # "11" vs 11, and "3.0" vs 3.
        if isinstance(actual, (int, float)) and isinstance(expected, str):
            actual, expected = expected, actual
        if isinstance(actual, str) and isinstance(expected, (int, float)):
            try:
                return float(actual.strip()) == float(expected)
            except ValueError:
                return False

        if isinstance(actual, str) and isinstance(expected, str):
            return actual.strip().casefold() == expected.strip().casefold()

        return False

    @classmethod
    def _matches(cls, row: dict, where: dict | None) -> bool:
        if where is None:
            return True
        return all(cls._value_matches(row.get(k), v) for k, v in where.items())

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
        removed: list[dict] = []
        i = 0
        try:
            for i, row in enumerate(rows):
                if token is not None:
                    token.check()
                if self._matches(row, where):
                    removed.append(row)
                else:
                    kept.append(row)
            self._working[table] = kept
            # The removed rows go in the journal, not just their count: an
            # update can be described by its `set` fields, but a delete that
            # recorded only "3 rows" would leave nothing to undo from.
            self._journal("delete", table=table, where=where,
                          rows=len(removed), removed=removed)
            return len(removed)
        except Cancelled:
            # Everything from the row that was interrupted onward is untouched.
            self._working[table] = kept + rows[i:]
            self._journal("delete", table=table, where=where, rows=len(removed),
                          removed=removed, cancelled=True)
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

    def reload(self) -> None:
        """Discard every in-memory change and re-read `state_path` from disk.

        Used by the orchestrator's /reset to restore a pristine fixture
        between demo runs. Re-reading the SAME path this instance was
        constructed with (rather than replacing it with a new DeviceState
        object) matters for two reasons: the demo machine keeps a pristine
        copy outside the repo and copies it into that path at launch, so the
        path is the right source of truth; and every other holder of this
        object (notably the reasoner, which keeps its own `self.device`
        reference) sees the reloaded data without needing to be re-wired.
        """
        self._committed = json.loads(self.state_path.read_text(encoding="utf-8"))
        self._working = copy.deepcopy(self._committed)
        self._journal("reload")

    def snapshot(self) -> dict:
        return copy.deepcopy(self._working)
