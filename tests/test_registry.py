from rtvoice.protocol import ReasonerMessage
from rtvoice.registry import TaskRegistry, TaskStatus


def msg(kind, **kw):
    return ReasonerMessage(kind=kind, **kw)


def test_ack_creates_a_pending_task():
    r = TaskRegistry()
    r.apply(msg("ack", task_id="t1", understood_as="rename contacts to Hans"))
    t = r.get("t1")
    assert t.status == TaskStatus.PENDING
    assert t.understood_as == "rename contacts to Hans"


def test_confirm_required_moves_to_awaiting_confirm_and_stores_verbatim():
    r = TaskRegistry()
    r.apply(msg("ack", task_id="t1", understood_as="rename"))
    r.apply(msg("confirm_required", task_id="t1",
                verbatim_text="This will rename 47 contacts. Confirm?"))
    assert r.get("t1").status == TaskStatus.AWAITING_CONFIRM
    assert r.verbatim_span("t1") == "This will rename 47 contacts. Confirm?"


def test_done_stores_result_verbatim():
    r = TaskRegistry()
    r.apply(msg("ack", task_id="t1", understood_as="rename"))
    r.apply(msg("done", task_id="t1", result="renamed 47 contacts"))
    assert r.get("t1").status == TaskStatus.DONE
    assert r.verbatim_span("t1") == "renamed 47 contacts"


def test_live_ids_excludes_terminal_tasks():
    r = TaskRegistry()
    r.apply(msg("ack", task_id="t1", understood_as="a"))
    r.apply(msg("ack", task_id="t2", understood_as="b"))
    r.apply(msg("done", task_id="t2", result="ok"))
    assert r.live_ids() == ["t1"]


def test_fact_block_contains_only_current_state():
    r = TaskRegistry()
    r.apply(msg("ack", task_id="t1", understood_as="rename contacts"))
    r.apply(msg("progress", task_id="t1", status="scanning"))
    block = r.fact_block()
    assert "t1" in block
    assert "rename contacts" in block
    assert "scanning" in block


def test_noop_is_ignored():
    r = TaskRegistry()
    r.apply(msg("noop"))
    assert r.live_ids() == []
