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
    token = CancellationToken()

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
    dev.rollback()
    dev.commit()
    assert [c["first_name"] for c in dev.query("contacts")] == ["Sarah", "Marcus", "Priya"]


def test_journal_records_every_write(dev):
    dev.update("contacts", {"first_name": "Hans"}, {"group": "work"})
    dev.commit()
    entries = [json.loads(l) for l in dev.journal_path.read_text().splitlines() if l.strip()]
    assert any(e["op"] == "update" and e["table"] == "contacts" for e in entries)
    assert any(e["op"] == "commit" for e in entries)
