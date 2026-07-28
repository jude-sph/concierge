import json
import pytest
from rtvoice.cancellation import Cancelled, CancellationToken
from rtvoice.device import DeviceState


@pytest.fixture
def dev(tmp_path):
    state = tmp_path / "device_state.json"
    state.write_text(json.dumps({
        "contacts": [
            {"id": 1, "first_name": "Sarah", "group": "work"},
            {"id": 2, "first_name": "Marcus", "group": "work"},
            {"id": 3, "first_name": "Priya", "group": "family"},
        ]
    }))
    return DeviceState(state, tmp_path / "journal.jsonl")


def test_query_all(dev):
    assert len(dev.query("contacts")) == 3


def test_query_with_where(dev):
    assert len(dev.query("contacts", {"group": "work"})) == 2


def test_update_is_not_visible_until_commit(dev):
    n = dev.update("contacts", {"first_name": "Hans"})
    assert n == 3
    on_disk = json.loads(dev.state_path.read_text())
    assert on_disk["contacts"][0]["first_name"] == "Sarah"
    dev.commit()
    on_disk = json.loads(dev.state_path.read_text())
    assert all(c["first_name"] == "Hans" for c in on_disk["contacts"])


def test_rollback_discards_uncommitted_writes(dev):
    dev.update("contacts", {"first_name": "Hans"})
    dev.rollback()
    dev.commit()
    assert [c["first_name"] for c in dev.query("contacts")] == ["Sarah", "Marcus", "Priya"]


def test_update_respects_where(dev):
    dev.update("contacts", {"first_name": "Hans"}, {"group": "work"})
    dev.commit()
    names = {c["id"]: c["first_name"] for c in dev.query("contacts")}
    assert names == {1: "Hans", 2: "Hans", 3: "Priya"}


def test_cancellation_stops_mid_update_and_rolls_back(dev):
    class CancelAfterOne(CancellationToken):
        def __init__(self):
            super().__init__()
            self.n = 0

        def check(self):
            self.n += 1
            if self.n > 1:
                raise Cancelled()

    with pytest.raises(Cancelled):
        dev.update("contacts", {"first_name": "Hans"}, token=CancelAfterOne())

    # Verify ordering: exactly one row should be mutated before cancellation
    working_names = [c["first_name"] for c in dev.query("contacts")]
    assert working_names == ["Hans", "Marcus", "Priya"], \
        "Cancellation test must verify order: token.check() happens BEFORE mutation"

    dev.rollback()
    dev.commit()
    assert [c["first_name"] for c in dev.query("contacts")] == ["Sarah", "Marcus", "Priya"]


def test_journal_records_every_write(dev):
    dev.update("contacts", {"first_name": "Hans"}, {"group": "work"})
    dev.commit()
    entries = [json.loads(l) for l in dev.journal_path.read_text().splitlines() if l.strip()]
    assert any(e["op"] == "update" and e["table"] == "contacts" for e in entries)
    assert any(e["op"] == "commit" for e in entries)


def test_query_returns_deep_copies(dev):
    """Mutations of query results must not affect subsequent queries."""
    result = dev.query("contacts")
    result[0]["first_name"] = "MUTATED"
    # Re-query should show original data, not the mutation
    requeried = dev.query("contacts")
    assert requeried[0]["first_name"] == "Sarah", \
        "query() must return deep copies, not live references"


# --- delete ------------------------------------------------------------------


def test_delete_respects_where(dev):
    assert dev.delete("contacts", {"group": "work"}) == 2
    assert [c["first_name"] for c in dev.query("contacts")] == ["Priya"]


def test_delete_without_a_filter_empties_the_table(dev):
    assert dev.delete("contacts") == 3
    assert dev.query("contacts") == []


def test_delete_is_not_visible_until_commit(dev):
    dev.delete("contacts", {"group": "work"})
    on_disk = json.loads(dev.state_path.read_text())
    assert len(on_disk["contacts"]) == 3
    dev.commit()
    assert len(json.loads(dev.state_path.read_text())["contacts"]) == 1


def test_rollback_restores_deleted_rows(dev):
    dev.delete("contacts")
    dev.rollback()
    assert [c["first_name"] for c in dev.query("contacts")] == ["Sarah", "Marcus", "Priya"]
    dev.commit()
    assert len(json.loads(dev.state_path.read_text())["contacts"]) == 3


def test_cancellation_stops_mid_delete_and_rolls_back(dev):
    class CancelAfterOne(CancellationToken):
        def __init__(self):
            super().__init__()
            self.n = 0

        def check(self):
            self.n += 1
            if self.n > 1:
                raise Cancelled()

    with pytest.raises(Cancelled):
        dev.delete("contacts", token=CancelAfterOne())

    # exactly one row gone: the token is checked BEFORE each record is decided
    assert [c["first_name"] for c in dev.query("contacts")] == ["Marcus", "Priya"]

    dev.rollback()
    dev.commit()
    assert [c["first_name"] for c in dev.query("contacts")] == ["Sarah", "Marcus", "Priya"]


def test_journal_records_delete_and_cancelled_delete(dev):
    dev.delete("contacts", {"group": "work"})

    class Immediate(CancellationToken):
        def check(self):
            raise Cancelled()

    with pytest.raises(Cancelled):
        dev.delete("contacts", token=Immediate())

    entries = [json.loads(l) for l in dev.journal_path.read_text().splitlines() if l.strip()]
    done = next(e for e in entries if e["op"] == "delete" and not e.get("cancelled"))
    assert done["table"] == "contacts" and done["rows"] == 2
    # the rows themselves, not just how many: nothing else could undo a delete
    assert [r["first_name"] for r in done["removed"]] == ["Sarah", "Marcus"]
    cancelled = next(e for e in entries if e["op"] == "delete" and e.get("cancelled"))
    assert cancelled["rows"] == 0 and cancelled["removed"] == []


# --- insert ------------------------------------------------------------------


def test_insert_assigns_an_id_and_returns_the_record(dev):
    created = dev.insert("contacts", {"first_name": "Hans", "group": "work"})
    assert created["id"] == 4
    assert created["first_name"] == "Hans"
    assert len(dev.query("contacts")) == 4


def test_insert_ignores_a_caller_supplied_id(dev):
    """A duplicate id makes every later where={"id": n} ambiguous, and the
    caller in this system is a language model."""
    created = dev.insert("contacts", {"id": 1, "first_name": "Hans"})
    assert created["id"] == 4
    assert len(dev.query("contacts", {"id": 1})) == 1


def test_insert_into_a_table_that_does_not_exist_yet(dev):
    created = dev.insert("reminders", {"title": "call the dentist"})
    assert created["id"] == 1
    assert dev.query("reminders") == [created]


def test_insert_is_not_visible_until_commit(dev):
    dev.insert("contacts", {"first_name": "Hans"})
    assert len(json.loads(dev.state_path.read_text())["contacts"]) == 3
    dev.commit()
    assert len(json.loads(dev.state_path.read_text())["contacts"]) == 4


def test_rollback_discards_a_staged_insert(dev):
    dev.insert("contacts", {"first_name": "Hans"})
    dev.rollback()
    assert [c["first_name"] for c in dev.query("contacts")] == ["Sarah", "Marcus", "Priya"]


def test_insert_honours_a_fired_token(dev):
    class Immediate(CancellationToken):
        def check(self):
            raise Cancelled()

    with pytest.raises(Cancelled):
        dev.insert("contacts", {"first_name": "Hans"}, token=Immediate())
    assert len(dev.query("contacts")) == 3


def test_insert_returns_a_copy_not_a_live_reference(dev):
    created = dev.insert("contacts", {"first_name": "Hans"})
    created["first_name"] = "MUTATED"
    assert dev.query("contacts", {"id": 4})[0]["first_name"] == "Hans"


def test_journal_records_insert(dev):
    dev.insert("contacts", {"first_name": "Hans"})
    entries = [json.loads(l) for l in dev.journal_path.read_text().splitlines() if l.strip()]
    rec = next(e for e in entries if e["op"] == "insert")
    assert rec["table"] == "contacts"
    assert rec["record"]["first_name"] == "Hans"
    assert rec["record"]["id"] == 4


def test_journal_records_cancelled_update(dev):
    """Cancelled updates must be journaled with partial row count and cancelled flag."""
    class CancelAfterOne(CancellationToken):
        def __init__(self):
            super().__init__()
            self.n = 0

        def check(self):
            self.n += 1
            if self.n > 1:
                raise Cancelled()

    with pytest.raises(Cancelled):
        dev.update("contacts", {"first_name": "Hans"}, token=CancelAfterOne())

    entries = [json.loads(l) for l in dev.journal_path.read_text().splitlines() if l.strip()]
    # Should have a journal entry for the update, marked as cancelled, with rows=1
    cancelled_update = next(
        (e for e in entries if e["op"] == "update" and e.get("cancelled")),
        None
    )
    assert cancelled_update is not None, "Cancelled update must be journaled"
    assert cancelled_update["rows"] == 1, "Journal must record how many rows were changed before cancellation"
    assert cancelled_update["cancelled"] is True


# --- matching a filter a language model wrote --------------------------------
#
# The `where` clause is authored by a model from spoken input, and the stored
# value is whatever the fixture holds. Live, the planner emitted
# `where={"id": "11"}` against a row holding `{"id": 11}` and the reasoner
# reported "no contacts matched" for a contact plainly visible on screen.

def _dev(tmp_path, rows):
    import json
    state = tmp_path / "device_state.json"
    state.write_text(json.dumps({"contacts": rows}))
    return DeviceState(state, tmp_path / "journal.jsonl")


def test_a_number_quoted_as_a_string_still_matches(tmp_path):
    device = _dev(tmp_path, [{"id": 11, "first_name": "Omar"}])
    assert [r["first_name"] for r in device.query("contacts", {"id": "11"})] == ["Omar"]


def test_a_string_field_queried_as_a_number_still_matches(tmp_path):
    device = _dev(tmp_path, [{"id": 1, "ext": "204"}])
    assert len(device.query("contacts", {"ext": 204})) == 1


def test_a_name_matches_regardless_of_case_or_spacing(tmp_path):
    """A difference in case, for a name arriving via speech recognition, is
    not a difference."""
    device = _dev(tmp_path, [{"id": 1, "first_name": "Sarah"}])
    assert len(device.query("contacts", {"first_name": "sarah"})) == 1
    assert len(device.query("contacts", {"first_name": " Sarah "})) == 1


def test_matching_is_still_exact_not_partial(tmp_path):
    """This predicate also selects the rows `delete` removes. Loosening it to
    substrings would silently widen the blast radius of every write."""
    device = _dev(tmp_path, [{"id": 1, "first_name": "Sarah"}])
    assert device.query("contacts", {"first_name": "Sar"}) == []
    assert device.query("contacts", {"first_name": "Sarah Chen"}) == []


def test_a_boolean_never_matches_a_number(tmp_path):
    """In Python `True == 1`. Without a guard, `favourite=1` would match a
    row holding False's sibling values, and `favourite=True` a literal 1."""
    device = _dev(tmp_path, [{"id": 1, "favourite": True},
                             {"id": 2, "favourite": 1}])
    assert [r["id"] for r in device.query("contacts", {"favourite": True})] == [1]
    assert [r["id"] for r in device.query("contacts", {"favourite": 1})] == [2]


def test_a_non_numeric_string_does_not_match_a_number(tmp_path):
    device = _dev(tmp_path, [{"id": 1, "first_name": "Omar"}])
    assert device.query("contacts", {"id": "latest"}) == []


def test_a_missing_field_matches_nothing(tmp_path):
    device = _dev(tmp_path, [{"id": 1}])
    assert device.query("contacts", {"nickname": "Bo"}) == []


def test_the_looser_match_reaches_deletes_too(tmp_path):
    """Deliberate: the same predicate, so what the user was told would be
    affected is what is actually affected."""
    device = _dev(tmp_path, [{"id": 11, "first_name": "Omar"}])
    assert device.delete("contacts", {"id": "11"}) == 1
