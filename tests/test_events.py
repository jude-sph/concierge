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


# Tests for Finding 1: Handle malformed final line gracefully
def test_read_handles_malformed_final_line(tmp_path):
    """Malformed final line (crash during write) is silently skipped."""
    p = tmp_path / "e.jsonl"
    log = EventLog(p)
    log.append("valid", x=1)
    # Simulate a crash mid-write by appending incomplete JSON
    with open(p, "a") as f:
        f.write('{"kind": "truncated"')
    log.close()

    # read() should return the valid event and skip the truncated line
    events = EventLog.read(p)
    assert len(events) == 1
    assert events[0].kind == "valid"


def test_read_raises_on_malformed_earlier_line(tmp_path):
    """Malformed line anywhere earlier than the final line raises JSONDecodeError."""
    p = tmp_path / "e.jsonl"
    # Write a malformed line first, then a valid event
    with open(p, "w") as f:
        f.write('{"incomplete"\n')
        f.write('{"kind": "valid", "t_ms": 0, "data": {}}\n')

    # read() should raise on the corrupted line
    with pytest.raises(json.JSONDecodeError):
        EventLog.read(p)


# Tests for Finding 2: Deterministic subscriber cleanup
@pytest.mark.asyncio
async def test_unsubscribe_removes_queue_deterministically(tmp_path):
    """Calling unsubscribe() deterministically removes the queue without relying on GC."""
    log = EventLog(tmp_path / "e.jsonl")
    sub = log.subscribe()

    # Queue should be in the log's subscriber list
    assert sub._queue in log._queues
    assert len(log._queues) == 1

    # Explicitly unsubscribe
    sub.unsubscribe()

    # Queue should be removed immediately, not waiting for GC
    assert sub._queue not in log._queues
    assert len(log._queues) == 0


@pytest.mark.asyncio
async def test_eventlog_context_manager_cleanup(tmp_path):
    """EventLog used as context manager deterministically clears all subscribers on exit."""
    import asyncio

    async def create_and_subscribe(log):
        sub = log.subscribe()
        # Simulate a consumer that breaks early
        async for _ in sub:
            break

    with EventLog(tmp_path / "e.jsonl") as log:
        # Create multiple subscriptions
        sub1 = log.subscribe()
        sub2 = log.subscribe()
        assert len(log._queues) == 2

        # Exit context manager

    # All queues should be cleared
    assert len(log._queues) == 0


@pytest.mark.asyncio
async def test_subscription_async_context_manager(tmp_path):
    """Individual subscriptions can use async context manager for cleanup."""
    log = EventLog(tmp_path / "e.jsonl")

    # Create and use subscription as async context manager
    async with log.subscribe() as sub:
        assert sub._queue in log._queues

    # After exiting, queue should be removed deterministically
    assert len(log._queues) == 0
