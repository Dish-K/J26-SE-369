import asyncio
from dataclasses import replace
from datetime import timedelta, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from backend.app.classifier.flag_identity import flag_id_for_segment
from backend.app.classifier.lifecycle_recovery import CheckpointConflict, RecoverableSegmentLifecycle
from backend.app.classifier.segment_lifecycle import LifecycleInputConflict, SegmentLifecycle
from backend.app.classifier.submission_coordinator import (
    SubmissionCoordinator, SubmissionInputConflict, SubmissionPlan,
)
from tests.classifier.flag_persistence_support import BASE, SESSION, TASK, make_window
from tests.classifier.test_lifecycle_recovery import Connection, Pool


def run(coroutine):
    return asyncio.run(coroutine)


def plan_for(**changes):
    values = dict(session_id=SESSION, task_id=TASK,
                  model_version="task12-storage-fixture-v1", started_at=BASE,
                  submitted_at=BASE + timedelta(seconds=47))
    return SubmissionPlan(**(values | changes))


def fixture():
    plan = plan_for()
    windows = [make_window(n, base=BASE-timedelta(seconds=20),
                           category="no_ai" if n < 4 else "external_ai")
               for n in range(7)]
    pool = Pool(Connection(windows))
    coordinator = run(SubmissionCoordinator.open(pool, plan, create_if_missing=True))
    return plan, windows, pool, coordinator


def feed(coordinator, plan, windows):
    for requirement, window in zip(plan.required_windows, windows):
        run(coordinator.accept_window(window, now=requirement.ready_at))


@pytest.fixture
def saved_flags(monkeypatch):
    saved, calls = {}, []

    async def save(pool, segment, **kwargs):
        identity = flag_id_for_segment(segment)
        calls.append(identity)
        if identity in saved:
            assert saved[identity] == segment
        saved[identity] = segment
        return identity

    monkeypatch.setattr("backend.app.classifier.segment_lifecycle.save_segment_flag", save)
    return saved, calls


def test_official_grid_includes_leading_and_tail_windows_with_normal_grace():
    plan = plan_for()
    assert [int((w.window_start-BASE).total_seconds()) for w in plan.required_windows] == [-20,-10,0,10,20,30,40]
    assert [int((w.ready_at-BASE).total_seconds()) for w in plan.required_windows] == [12,22,32,42,52,62,72]
    assert plan.feature_schema_version == "1.0.0"


def test_exact_submission_boundary_excludes_zero_coverage_window():
    plan = plan_for(submitted_at=BASE+timedelta(seconds=50))
    assert plan.required_windows[-1].window_start == BASE+timedelta(seconds=40)


def test_non_aligned_task_start_uses_shared_epoch_grid():
    plan = plan_for(started_at=BASE+timedelta(seconds=7))
    assert plan.required_windows[0].window_start == BASE-timedelta(seconds=20)


def test_timezone_normalization_preserves_instants():
    zone = timezone(timedelta(hours=5, minutes=30))
    assert plan_for(started_at=BASE.astimezone(zone)).required_windows == plan_for().required_windows


@pytest.mark.parametrize("changes", [
    {"started_at": BASE.replace(tzinfo=None)},
    {"submitted_at": BASE.replace(tzinfo=None)},
    {"submitted_at": BASE-timedelta(seconds=1)},
    {"session_id": "untrusted-id"}, {"model_version": ""},
])
def test_invalid_plan_metadata_is_rejected(changes):
    with pytest.raises(ValueError):
        plan_for(**changes)


def test_too_early_window_rejected_without_checkpoint_or_state_change():
    plan, windows, pool, coordinator = fixture()
    with pytest.raises(ValueError, match="normal end plus grace"):
        run(coordinator.accept_window(windows[0], now=plan.required_windows[0].ready_at-timedelta(microseconds=1)))
    assert pool.connection.row is None
    assert coordinator.progress(now=BASE).lifecycle.processed_window_count == 0


def test_window_can_be_accepted_at_exact_ready_time():
    plan, windows, pool, coordinator = fixture()
    state = run(coordinator.accept_window(windows[0], now=plan.required_windows[0].ready_at))
    assert state.lifecycle.processed_window_count == 1
    assert pool.connection.row["revision"] == 1


@pytest.mark.parametrize("kind", ["skip", "outside", "wrong_end", "wrong_stream"])
def test_nonplan_or_missing_predecessor_windows_fail_closed(kind):
    plan, windows, pool, coordinator = fixture()
    bad = {"skip": windows[1], "outside": make_window(7),
           "wrong_end": replace(windows[0], window_end=windows[0].window_end+timedelta(seconds=1)),
           "wrong_stream": replace(windows[0], task_id=uuid4())}[kind]
    with pytest.raises(SubmissionInputConflict):
        run(coordinator.accept_window(bad, now=BASE+timedelta(seconds=100)))
    assert pool.connection.row is None


def test_clock_alone_never_finishes_missing_prediction_windows():
    plan, windows, pool, coordinator = fixture()
    progress = run(coordinator.advance(now=BASE+timedelta(hours=1), telemetry_complete=True))
    assert progress.phase == "waiting_windows" and len(progress.pending_windows) == 7
    assert not progress.lifecycle.finished and pool.connection.row is None


def test_submission_at_47_keeps_current_flag_active_until_tail_predictions():
    plan, windows, pool, coordinator = fixture()
    feed(coordinator, plan, windows[:5])
    progress = run(coordinator.advance(now=BASE+timedelta(seconds=53), telemetry_complete=True))
    assert progress.phase == "waiting_windows" and len(progress.pending_windows) == 2
    assert len(progress.lifecycle.active_segments) == 1 and not progress.lifecycle.finished
    assert progress.next_ready_at == BASE+timedelta(seconds=62)


def test_complete_predictions_still_require_explicit_telemetry_completion():
    plan, windows, pool, coordinator = fixture()
    feed(coordinator, plan, windows)
    progress = run(coordinator.advance(now=BASE+timedelta(seconds=72), telemetry_complete=False))
    assert progress.phase == "waiting_telemetry" and not progress.lifecycle.finished
    assert pool.connection.row["revision"] == 7


@pytest.mark.parametrize("value", [1, "true", None])
def test_truthy_nonboolean_delivery_signals_are_rejected(value):
    plan, windows, pool, coordinator = fixture()
    with pytest.raises(ValueError, match="boolean"):
        run(coordinator.advance(now=BASE, telemetry_complete=value))


def test_all_gates_finish_and_save_one_segment_with_three_contributors(saved_flags):
    plan, windows, pool, coordinator = fixture()
    feed(coordinator, plan, windows)
    original = coordinator.progress(now=BASE+timedelta(seconds=72)).lifecycle.active_segments[0]
    progress = run(coordinator.advance(now=BASE+timedelta(seconds=72), telemetry_complete=True))
    assert progress.phase == "complete" and progress.lifecycle.finished
    assert not progress.lifecycle.active_segments and not progress.lifecycle.pending_segments
    saved, calls = saved_flags
    identity = flag_id_for_segment(original)
    assert calls == [identity] and saved[identity].prediction_ids == tuple(w.prediction_id for w in windows[4:])
    assert saved[identity].confidence == original.confidence
    again = run(coordinator.advance(now=BASE+timedelta(seconds=73), telemetry_complete=False))
    assert again.phase == "complete" and calls == [identity]


def test_reopening_pending_submission_restores_prefix_without_relocalization(saved_flags):
    plan, windows, pool, coordinator = fixture()
    feed(coordinator, plan, windows[:5])
    original = coordinator.progress(now=BASE+timedelta(seconds=52)).lifecycle.active_segments[0]
    recovered = run(SubmissionCoordinator.open(pool, plan))
    for requirement, window in zip(plan.required_windows[5:], windows[5:]):
        run(recovered.accept_window(window, now=requirement.ready_at))
    progress = run(recovered.advance(now=BASE+timedelta(seconds=72), telemetry_complete=True))
    assert progress.phase == "complete"
    assert flag_id_for_segment(progress.lifecycle.closed_segments[0]) == flag_id_for_segment(original)


def test_failed_flag_save_stays_pending_and_reopen_retries(monkeypatch, saved_flags):
    plan, windows, pool, coordinator = fixture()
    feed(coordinator, plan, windows)
    target = "backend.app.classifier.segment_lifecycle.save_segment_flag"
    original_save = __import__("backend.app.classifier.segment_lifecycle", fromlist=["save_segment_flag"]).save_segment_flag
    monkeypatch.setattr(target, AsyncMock(side_effect=OSError("database unavailable")))
    with pytest.raises(OSError):
        run(coordinator.advance(now=BASE+timedelta(seconds=72), telemetry_complete=True))
    assert coordinator.progress(now=BASE+timedelta(seconds=72)).phase == "saving_flags"
    recovered = run(SubmissionCoordinator.open(pool, plan))
    monkeypatch.setattr(target, original_save)
    assert run(recovered.advance(now=BASE+timedelta(seconds=73), telemetry_complete=False)).phase == "complete"


def test_completed_recovery_revalidates_flags_through_task12(saved_flags):
    plan, windows, pool, coordinator = fixture()
    feed(coordinator, plan, windows)
    run(coordinator.advance(now=BASE+timedelta(seconds=72), telemetry_complete=True))
    recovered = run(SubmissionCoordinator.open(pool, plan))
    assert recovered.progress(now=BASE+timedelta(seconds=73)).phase == "saving_flags"
    assert run(recovered.advance(now=BASE+timedelta(seconds=73), telemetry_complete=False)).phase == "complete"
    saved, calls = saved_flags
    assert len(saved) == 1 and len(calls) == 2 and calls[0] == calls[1]


def test_failed_finish_checkpoint_does_not_close_visible_state(monkeypatch):
    plan, windows, pool, coordinator = fixture()
    feed(coordinator, plan, windows)
    monkeypatch.setattr("backend.app.classifier.lifecycle_recovery.write_checkpoint",
                        AsyncMock(side_effect=OSError("offline")))
    with pytest.raises(OSError):
        run(coordinator.advance(now=BASE+timedelta(seconds=72), telemetry_complete=True))
    assert not coordinator.progress(now=BASE+timedelta(seconds=72)).lifecycle.finished
    assert not run(SubmissionCoordinator.open(pool, plan)).progress(now=BASE).lifecycle.finished


def test_duplicate_delivery_is_safe_but_changed_prediction_is_rejected():
    plan, windows, pool, coordinator = fixture()
    now = plan.required_windows[0].ready_at
    run(coordinator.accept_window(windows[0], now=now))
    assert run(coordinator.accept_window(windows[0], now=now)).lifecycle.processed_window_count == 1
    with pytest.raises(LifecycleInputConflict):
        run(coordinator.accept_window(replace(windows[0], confidence=.1), now=now))


@pytest.mark.parametrize("kind", ["gap", "wrong_model", "premature_finish"])
def test_incompatible_recovered_stream_cannot_be_adopted(kind):
    plan, windows, pool, coordinator = fixture()
    flow = SegmentLifecycle(session_id=plan.session_id, task_id=plan.task_id,
                            model_version="other" if kind == "wrong_model" else plan.model_version)
    if kind != "wrong_model":
        flow.accept_window(windows[1] if kind == "gap" else windows[0])
    if kind == "premature_finish":
        flow.finish(pending_live_processing_complete=True)
    with pytest.raises(SubmissionInputConflict):
        SubmissionCoordinator(plan, RecoverableSegmentLifecycle(pool, flow, 0))


def test_stale_submission_worker_cannot_replace_newer_progress():
    plan, windows, pool, coordinator = fixture()
    feed(coordinator, plan, windows[:1])
    stale = run(SubmissionCoordinator.open(pool, plan))
    feed(coordinator, plan, windows[:2])
    with pytest.raises(CheckpointConflict):
        run(stale.accept_window(windows[1], now=plan.required_windows[1].ready_at))
    assert stale.progress(now=BASE).lifecycle.processed_window_count == 1


def test_all_clean_windows_complete_without_fabricating_any_flag(saved_flags):
    plan = plan_for()
    windows = [make_window(n, base=BASE-timedelta(seconds=20), category="no_ai") for n in range(7)]
    pool = Pool(Connection(windows))
    coordinator = run(SubmissionCoordinator.open(pool, plan, create_if_missing=True))
    feed(coordinator, plan, windows)
    assert run(coordinator.advance(now=BASE+timedelta(seconds=72), telemetry_complete=True)).phase == "complete"
    assert saved_flags == ({}, [])


def test_instantaneous_task_requires_delivery_confirmation_but_no_windows(saved_flags):
    plan = plan_for(submitted_at=BASE)
    assert plan.required_windows == ()
    coordinator = run(SubmissionCoordinator.open(Pool(Connection([])), plan, create_if_missing=True))
    assert run(coordinator.advance(now=BASE, telemetry_complete=False)).phase == "waiting_telemetry"
    assert run(coordinator.advance(now=BASE, telemetry_complete=True)).phase == "complete"
    assert saved_flags == ({}, [])


def test_instantaneous_task_cannot_finish_before_its_submission_time(saved_flags):
    plan = plan_for(submitted_at=BASE)
    coordinator = run(SubmissionCoordinator.open(Pool(Connection([])), plan, create_if_missing=True))
    progress = run(coordinator.advance(now=BASE-timedelta(seconds=1), telemetry_complete=True))
    assert progress.phase == "waiting_windows" and not progress.lifecycle.finished
    assert progress.next_ready_at == BASE


def test_partial_multi_flag_save_retries_only_unacknowledged_flags(monkeypatch, saved_flags):
    plan = plan_for()
    windows = [make_window(n, base=BASE-timedelta(seconds=20),
                           category="no_ai" if n < 4 else "external_ai",
                           ranges=((0,2,10,15),(0,2,40,45))) for n in range(7)]
    pool = Pool(Connection(windows))
    coordinator = run(SubmissionCoordinator.open(pool, plan, create_if_missing=True))
    feed(coordinator, plan, windows)
    import backend.app.classifier.segment_lifecycle as lifecycle_module
    original_save = lifecycle_module.save_segment_flag
    failed = False

    async def fail_second(pool, segment, **kwargs):
        nonlocal failed
        if segment.start_line == 40 and not failed:
            failed = True
            raise OSError("second flag failed")
        return await original_save(pool, segment, **kwargs)

    monkeypatch.setattr(lifecycle_module, "save_segment_flag", fail_second)
    with pytest.raises(OSError):
        run(coordinator.advance(now=BASE+timedelta(seconds=72), telemetry_complete=True))
    progress = coordinator.progress(now=BASE+timedelta(seconds=72))
    assert progress.phase == "saving_flags" and len(progress.lifecycle.pending_segments) == 1
    assert run(coordinator.advance(now=BASE+timedelta(seconds=73), telemetry_complete=False)).phase == "complete"
    saved, calls = saved_flags
    assert len(saved) == 2 and len(calls) == 2
