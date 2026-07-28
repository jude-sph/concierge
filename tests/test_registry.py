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


def test_fact_block_says_what_is_underway():
    r = TaskRegistry()
    r.apply(msg("ack", task_id="t1", understood_as="rename contacts"))
    r.apply(msg("progress", task_id="t1", status="scanning"))
    block = r.fact_block()
    assert "rename contacts" in block
    assert "running" in block


def test_fact_block_hides_the_internal_task_id():
    """It is spoken context for a small model, and "t1" is not a word anyone
    should hear out loud. Nothing cites tasks any more -- the `relay` speech
    act that needed the id was removed with the rest of that machinery."""
    r = TaskRegistry()
    r.apply(msg("ack", task_id="t1", understood_as="rename contacts"))
    assert "t1" not in r.fact_block()


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


def test_fact_block_never_hands_over_wording_to_repeat():
    """The concierge is told what is happening, never what to say about it.

    This block used to append `exact wording to use: "<detail>"` to every
    line, written for the `relay` speech act where the concierge reproduced
    the reasoner's sentence verbatim. Results now reach the person from
    Orchestrator._speak_facts and never pass through the concierge, so all
    that instruction could still do was invite a second model to restate the
    first one's -- and it did, on a destructive write.
    """
    r = TaskRegistry()
    r.apply(msg("ack", task_id="t1", understood_as="rename contacts"))
    r.apply(msg("done", task_id="t1", result="renamed 47 contacts"))

    block = r.fact_block()

    assert "renamed 47 contacts" not in block
    assert "rename contacts" in block and "done" in block


def test_a_pending_confirmation_is_flagged_as_not_to_be_restated():
    """Measured: the reasoner asked "Delete 1 message from Marcus Webb sent
    2026-07-27?" and the concierge spoke "Delete messages from Marcus Webb
    sent yesterday?" -- the COUNT gone, and the person agreeing to something
    reworded. The verbatim question has already been spoken by then."""
    r = TaskRegistry()
    r.apply(msg("ack", task_id="t1", understood_as="delete messages"))
    r.apply(msg("confirm_required", task_id="t1",
                question="Delete 1 message from Marcus Webb sent 2026-07-27?"))

    block = r.fact_block()

    assert "Delete 1 message" not in block
    assert "rephrase" in block.lower() or "again" in block.lower()
    assert "awaiting_confirm" in block


def test_a_task_can_be_dropped_without_trace():
    """For the placeholder shown while the reasoner is still planning: it
    exists so the panel can say work has started rather than staying blank
    until a result arrives, and must vanish once the real tasks land.
    Distinct from mark_cancelled, which is an outcome a person may be told."""
    r = TaskRegistry()
    r.apply(msg("ack", task_id="pending-3", understood_as="working out: hello"))
    assert r.get("pending-3") is not None

    r.drop("pending-3")

    assert r.get("pending-3") is None
    assert r.all() == []
    r.drop("pending-3")  # idempotent
