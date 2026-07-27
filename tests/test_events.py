import json
import pytest
from rtvoice.events import Event, EventLog


def test_append_writes_jsonl(tmp_path):
    log = EventLog(tmp_path / "events.jsonl")
    log.append("turn_state", state="user_complete", transcript="hi")
    lines = (tmp_path / "events.jsonl").read_text().strip().split("\n")
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["kind"] == "turn_state"
    assert rec["data"]["state"] == "user_complete"


def test_timestamps_are_monotonic_offsets_from_session_start(tmp_path):
    log = EventLog(tmp_path / "e.jsonl")
    a = log.append("a")
    b = log.append("b")
    assert a.t_ms >= 0
    assert b.t_ms >= a.t_ms


def test_read_roundtrips(tmp_path):
    p = tmp_path / "e.jsonl"
    log = EventLog(p)
    log.append("x", v=1)
    log.append("y", v=2)
    events = EventLog.read(p)
    assert [e.kind for e in events] == ["x", "y"]
    assert events[1].data["v"] == 2


@pytest.mark.asyncio
async def test_subscribers_receive_appended_events(tmp_path):
    log = EventLog(tmp_path / "e.jsonl")
    received = []

    async def consume():
        async for ev in log.subscribe():
            received.append(ev.kind)
            if len(received) == 2:
                return

    import asyncio
    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    log.append("one")
    log.append("two")
    await asyncio.wait_for(task, timeout=1.0)
    assert received == ["one", "two"]
