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


def test_terminal_state_no_regress_on_progress():
    """Stale progress message after done should not regress status or clobber result."""
    r = TaskRegistry()
    r.apply(msg("ack", task_id="t1", understood_as="rename"))
    r.apply(msg("done", task_id="t1", result="renamed 47 contacts"))
    assert r.get("t1").status == TaskStatus.DONE
    assert r.verbatim_span("t1") == "renamed 47 contacts"

    # Stale progress message arrives out of order
    r.apply(msg("progress", task_id="t1", status="scanning"))
    assert r.get("t1").status == TaskStatus.DONE
    assert r.verbatim_span("t1") == "renamed 47 contacts"


def test_terminal_state_no_regress_on_ack():
    """Stale ack message after done should not regress status."""
    r = TaskRegistry()
    r.apply(msg("ack", task_id="t1", understood_as="rename"))
    r.apply(msg("done", task_id="t1", result="done"))
    assert r.get("t1").status == TaskStatus.DONE
    assert r.live_ids() == []

    # Stale ack arrives out of order
    r.apply(msg("ack", task_id="t1", understood_as="rename"))
    assert r.get("t1").status == TaskStatus.DONE
    assert r.live_ids() == []


def test_stale_confirm_required_does_not_clobber_done():
    """Stale confirm_required after done should not overwrite result verbatim."""
    r = TaskRegistry()
    r.apply(msg("ack", task_id="t1", understood_as="rename"))
    r.apply(msg("confirm_required", task_id="t1",
                verbatim_text="This will rename 47 contacts. Confirm?"))
    assert r.verbatim_span("t1") == "This will rename 47 contacts. Confirm?"

    # User confirmed and task completed
    r.apply(msg("done", task_id="t1", result="renamed 47 contacts"))
    assert r.verbatim_span("t1") == "renamed 47 contacts"

    # Stale confirm_required from earlier arrives out of order
    r.apply(msg("confirm_required", task_id="t1",
                verbatim_text="This will rename 47 contacts. Confirm?"))
    # Verbatim should remain the done result, not be clobbered by stale confirmation
    assert r.verbatim_span("t1") == "renamed 47 contacts"


def test_mark_cancelled_does_not_regress_done():
    """mark_cancelled() on an already-done task should not change its status."""
    r = TaskRegistry()
    r.apply(msg("ack", task_id="t1", understood_as="rename"))
    r.apply(msg("done", task_id="t1", result="completed"))
    assert r.get("t1").status == TaskStatus.DONE

    # Stale or erroneous cancel arrives
    r.mark_cancelled("t1")
    assert r.get("t1").status == TaskStatus.DONE


def test_need_clarification_awaits_confirm():
    """need_clarification message should move task to AWAITING_CONFIRM."""
    r = TaskRegistry()
    r.apply(msg("ack", task_id="t1", understood_as="rename"))
    r.apply(msg("need_clarification", task_id="t1", missing="which contacts?"))
    assert r.get("t1").status == TaskStatus.AWAITING_CONFIRM
    assert r.get("t1").detail == "which contacts?"


def test_failed_stores_reason_verbatim():
    """failed message should set status and store reason as verbatim."""
    r = TaskRegistry()
    r.apply(msg("ack", task_id="t1", understood_as="delete"))
    r.apply(msg("failed", task_id="t1", reason="Cannot delete protected contacts"))
    assert r.get("t1").status == TaskStatus.FAILED
    assert r.verbatim_span("t1") == "Cannot delete protected contacts"


def test_fact_block_excludes_stale_detail():
    """fact_block should not include detail from regressed status."""
    r = TaskRegistry()
    r.apply(msg("ack", task_id="t1", understood_as="rename contacts"))
    r.apply(msg("progress", task_id="t1", status="scanning"))
    block = r.fact_block()
    assert "scanning" in block
    assert "running" in block

    # Task completes
    r.apply(msg("done", task_id="t1", result="renamed 47 contacts"))
    block = r.fact_block()
    # Block should contain the done result, not the old scanning status
    assert "renamed 47 contacts" in block
    assert "scanning" not in block
    assert "running" not in block
    assert "done" in block
