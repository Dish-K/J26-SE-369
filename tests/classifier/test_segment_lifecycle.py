import asyncio
from copy import deepcopy
from dataclasses import replace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from backend.app.classifier.flag_identity import flag_id_for_segment
from backend.app.classifier.segment_builder import build_segment_drafts
from backend.app.classifier.segment_lifecycle import (
    LifecycleInputConflict, SegmentLifecycle, persist_closed_segments,
)
from tests.classifier.flag_persistence_support import make_window


def coordinator(window):
    return SegmentLifecycle(session_id=window.session_id, task_id=window.task_id,
                            model_version=window.model_version)


def edit_for(window, version, *, start=2, inserted=5):
    return {
        "session_id": window.session_id, "task_id": window.task_id,
        "load_seq": 0, "version_id": version, "batch_seq": version,
        "seq_in_batch": 0, "change_index": 0,
        "start_line": start, "end_line": start, "inserted_lines": inserted,
        "net_line_delta": inserted, "origin": 0,
    }


def test_first_detection_is_active_and_not_yet_pending_storage():
    w = make_window()
    flow = coordinator(w)
    state = flow.accept_window(w)
    assert len(state.active_segments) == 1
    assert state.closed_segments == state.pending_segments == ()
    assert state.persisted_flag_ids == ()
    assert state.processed_window_count == 1 and not state.finished
    assert state.active_segments == build_segment_drafts([w])


def test_active_growth_keeps_stable_id_and_highest_real_representative():
    a, b = make_window(confidence=.7), make_window(1, confidence=.9, ranges=((0,2,14,20),))
    flow = coordinator(a)
    first = flow.accept_window(a).active_segments[0]
    second = flow.accept_window(b).active_segments[0]
    assert flag_id_for_segment(first) == flag_id_for_segment(second)
    assert (second.start_line, second.end_line) == (10, 20)
    assert second.confidence == .9 and second.class_probabilities == b.class_probabilities
    assert second.representative_prediction_id == b.prediction_id
    assert second.prediction_ids == (a.prediction_id, b.prediction_id)
    assert flow.snapshot().closed_segments == ()


@pytest.mark.parametrize("boundary", ["clean", "below", "unlocalized", "category", "gap", "reload", "version", "disconnected"])
def test_existing_builder_boundaries_close_previous_segment(boundary):
    a = make_window()
    b = make_window(1)
    if boundary == "clean":
        b = make_window(1, category="no_ai")
    elif boundary == "below":
        b = replace(b, ai_positive=False, regions=())
    elif boundary == "unlocalized":
        b = replace(b, regions=())
    elif boundary == "category":
        b = make_window(1, category="ide_ai")
    elif boundary == "gap":
        b = make_window(2)
    elif boundary == "reload":
        b = make_window(1, ranges=((1,2,10,15),))
    elif boundary == "version":
        b = make_window(1, ranges=((0,3,10,15),))
    elif boundary == "disconnected":
        b = make_window(1, ranges=((0,2,40,45),))
    flow = coordinator(a)
    original = flow.accept_window(a).active_segments[0]
    state = flow.accept_window(b)
    assert state.closed_segments == (original,)
    assert state.pending_segments == (original,)
    assert len(state.active_segments) == (0 if boundary in {"clean","below","unlocalized"} else 1)
    assert flow.finish(pending_live_processing_complete=True).closed_segments == build_segment_drafts([a, b])


def test_one_of_several_segments_can_close_while_another_continues():
    a = make_window(ranges=((0,2,10,15),(0,2,30,35)))
    b = make_window(1, ranges=((0,2,31,36),))
    flow = coordinator(a)
    first = flow.accept_window(a)
    state = flow.accept_window(b)
    assert state.closed_segments == (first.active_segments[0],)
    assert flag_id_for_segment(state.active_segments[0]) == flag_id_for_segment(first.active_segments[1])
    assert state.active_segments[0].window_count == 2


def test_ambiguous_bridge_closes_originals_and_remains_independent():
    a = make_window(ranges=((0,2,10,10),(0,2,12,12)))
    b = make_window(1, ranges=((0,2,11,11),))
    c = make_window(2, ranges=((0,2,11,11),), confidence=.9)
    flow = coordinator(a)
    originals = flow.accept_window(a).active_segments
    state = flow.accept_window(b)
    assert state.closed_segments == originals
    assert len(state.active_segments) == 1
    state = flow.accept_window(c)
    assert state.closed_segments == originals
    assert state.active_segments[0].window_count == 2
    assert flow.finish(pending_live_processing_complete=True).closed_segments == build_segment_drafts([a,b,c])


def test_cross_version_safe_alignment_reuses_builder_and_identity():
    a = make_window()
    b = make_window(1, ranges=((0,3,15,20),), confidence=.9)
    flow = coordinator(a)
    old = flow.accept_window(a).active_segments[0]
    history = [edit_for(a, 3)]
    state = flow.accept_window(b, edit_history=history)
    assert len(state.active_segments) == 1 and not state.closed_segments
    current = state.active_segments[0]
    assert flag_id_for_segment(current) == flag_id_for_segment(old)
    assert (current.version_id, current.start_line, current.end_line) == (3,15,20)
    assert state.active_segments == build_segment_drafts([a,b], edit_history=history)


def test_exact_window_delivery_retry_does_not_add_members_or_windows():
    w = make_window()
    flow = coordinator(w)
    before = flow.accept_window(w)
    assert flow.accept_window(deepcopy(w)) == before
    assert flow.snapshot().processed_window_count == 1


@pytest.mark.parametrize("change", ["payload", "identity", "feature", "order"])
def test_changed_or_out_of_order_windows_are_rejected_without_state_change(change):
    a, b = make_window(), make_window(1)
    flow = coordinator(a)
    flow.accept_window(a)
    before = flow.accept_window(b)
    bad = {
        "payload": replace(a, regions=()),
        "identity": replace(a, prediction_id=uuid4()),
        "feature": replace(make_window(2), feature_vector_id=a.feature_vector_id),
        "order": make_window(0),
    }[change]
    with pytest.raises((LifecycleInputConflict, ValueError)):
        flow.accept_window(bad)
    assert flow.snapshot() == before


@pytest.mark.parametrize("field", ["session_id", "task_id", "model_version"])
def test_unrelated_streams_require_separate_coordinators(field):
    w = make_window()
    flow = coordinator(w)
    wrong = replace(w, **{field: "other-model" if field == "model_version" else uuid4()})
    with pytest.raises(ValueError, match="another lifecycle stream"):
        flow.accept_window(wrong)
    assert flow.snapshot().processed_window_count == 0


def test_invalid_source_evidence_is_rejected_without_consuming_window():
    w = make_window()
    bad = replace(w, regions=(replace(w.regions[0], edit_events=()),))
    flow = coordinator(w)
    with pytest.raises(ValueError):
        flow.accept_window(bad)
    assert flow.snapshot().processed_window_count == 0
    assert len(flow.accept_window(w).active_segments) == 1


def test_snapshots_and_caller_inputs_cannot_mutate_internal_evidence():
    w = make_window()
    flow = coordinator(w)
    state = flow.accept_window(w)
    original = deepcopy(flow.snapshot())
    w.regions[0].edit_events[0]["batch_seq"] = 999
    w.class_probabilities["external_ai"] = .1
    state.active_segments[0].class_probabilities["external_ai"] = .2
    state.active_segments[0].members[0].edit_events[0]["batch_seq"] = 1000
    assert flow.snapshot() == original


def test_interruption_and_reading_state_do_not_close_an_active_segment():
    a, b = make_window(), make_window(1)
    flow = coordinator(a)
    before = flow.accept_window(a)
    assert flow.snapshot() == flow.snapshot() == before
    # The coordinator has no wall-clock timeout or disconnect-triggered finish.
    resumed = flow.accept_window(b)
    assert not resumed.closed_segments and resumed.active_segments[0].window_count == 2


def test_submission_requires_processing_barrier_and_finish_is_idempotent():
    w = make_window()
    flow = coordinator(w)
    before = flow.accept_window(w)
    with pytest.raises(ValueError, match="must be complete"):
        flow.finish(pending_live_processing_complete=False)
    assert flow.snapshot() == before
    finished = flow.finish(pending_live_processing_complete=True)
    assert finished.finished and not finished.active_segments
    assert finished.closed_segments == finished.pending_segments == before.active_segments
    assert flow.finish(pending_live_processing_complete=True) == finished
    assert flow.accept_window(w) == finished
    with pytest.raises(RuntimeError, match="finished"):
        flow.accept_window(make_window(1))


def test_empty_or_clean_only_submission_does_not_invent_a_flag_or_score():
    w = make_window(category="no_ai")
    flow = coordinator(w)
    assert flow.finish(pending_live_processing_complete=True).closed_segments == ()
    flow = coordinator(w)
    flow.accept_window(w)
    state = flow.finish(pending_live_processing_complete=True)
    assert state.active_segments == state.closed_segments == ()
    assert not hasattr(state, "ai_confidence")


def test_late_history_cannot_rewrite_previously_accepted_segments():
    a = make_window()
    b = make_window(1, ranges=((0,3,15,20),))
    c = make_window(2, ranges=((0,3,15,20),))
    flow = coordinator(a)
    flow.accept_window(a)
    before = flow.accept_window(b)  # no history: existing builder keeps two segments
    with pytest.raises(LifecycleInputConflict, match="revise an accepted"):
        flow.accept_window(c, edit_history=[edit_for(a,3)])
    assert flow.snapshot() == before


def test_existing_history_is_immutable_and_exact_redelivery_across_calls_is_safe():
    a = make_window()
    b = make_window(1, ranges=((0,3,15,20),))
    change = edit_for(a,3)
    flow = coordinator(a)
    flow.accept_window(a)
    before = flow.accept_window(b, edit_history=[change])
    assert flow.accept_window(b, edit_history=[deepcopy(change)]) == before
    with pytest.raises(LifecycleInputConflict, match="evidence was changed"):
        flow.accept_window(make_window(2), edit_history=[{**change, "inserted_lines": 6}])
    assert flow.snapshot() == before


def test_duplicate_history_in_one_delivery_cannot_authorize_mapping():
    a = make_window()
    flow = coordinator(a)
    flow.accept_window(a)
    change = edit_for(a,3)
    with pytest.raises(LifecycleInputConflict, match="Duplicate edit identity"):
        flow.accept_window(make_window(1, ranges=((0,3,15,20),)), edit_history=[change,change])
    assert flow.snapshot().processed_window_count == 1


def test_ordered_replay_of_original_inputs_has_identical_ids_and_states():
    a = make_window(ranges=((0,2,10,15),(0,2,40,45)))
    b = make_window(1, ranges=((0,2,14,20),), confidence=.9)
    c = make_window(2, category="no_ai")
    first, replay = coordinator(a), coordinator(a)
    for w in [a,b,c]:
        assert first.accept_window(w) == replay.accept_window(deepcopy(w))
    assert first.finish(pending_live_processing_complete=True) == replay.finish(pending_live_processing_complete=True)


def test_every_prefix_and_completed_result_match_existing_builder():
    # Broad regression over ordered single-version, mixed-category and no-region
    # streams. Tests the coordinator against the unchanged builder as the oracle.
    import random
    rng = random.Random(43)
    windows = []
    first = make_window()
    flow = coordinator(first)
    for n in range(60):
        category = rng.choice(["external_ai", "external_ai", "ide_ai", "no_ai"])
        ranges = rng.choice([((0,2,10,15),), ((0,2,14,20),), ((0,2,10,10),(0,2,12,12)),
                             ((0,2,11,11),), ((0,2,40,45),), ()])
        w = make_window(n, category=category, confidence=rng.choice([.7,.8,.9]), ranges=ranges)
        windows.append(w)
        state = flow.accept_window(w)
        actual = {flag_id_for_segment(s): s for s in state.active_segments + state.closed_segments}
        expected = {flag_id_for_segment(s): s for s in build_segment_drafts(windows)}
        assert actual == expected
    assert flow.finish(pending_live_processing_complete=True).closed_segments == build_segment_drafts(windows)


def test_only_closed_flags_are_saved_and_successful_saves_are_not_repeated(monkeypatch):
    a, clean = make_window(), make_window(1, category="no_ai")
    flow = coordinator(a)
    flow.accept_window(a)
    save = AsyncMock(side_effect=lambda pool, seg: flag_id_for_segment(seg))
    monkeypatch.setattr("backend.app.classifier.segment_lifecycle.save_segment_flag", save)
    pool = object()
    assert asyncio.run(persist_closed_segments(flow,pool)) == ()
    save.assert_not_awaited()
    state = flow.accept_window(clean)
    expected = flag_id_for_segment(state.closed_segments[0])
    assert asyncio.run(persist_closed_segments(flow,pool)) == (expected,)
    assert flow.snapshot().persisted_flag_ids == (expected,)
    assert flow.snapshot().pending_segments == ()
    assert asyncio.run(persist_closed_segments(flow,pool)) == ()
    assert save.await_count == 1


@pytest.mark.parametrize("error", [RuntimeError("database unavailable"), asyncio.CancelledError()])
def test_failure_after_some_success_keeps_remaining_flags_pending_for_retry(monkeypatch,error):
    w = make_window(ranges=((0,2,10,15),(0,2,40,45)))
    flow = coordinator(w)
    flow.accept_window(w)
    state = flow.finish(pending_live_processing_complete=True)
    ids = tuple(flag_id_for_segment(s) for s in state.closed_segments)
    calls = []
    async def flaky(pool,seg):
        calls.append(flag_id_for_segment(seg))
        if len(calls) == 2:
            raise error
        return flag_id_for_segment(seg)
    monkeypatch.setattr("backend.app.classifier.segment_lifecycle.save_segment_flag",flaky)
    with pytest.raises(type(error)):
        asyncio.run(persist_closed_segments(flow,object()))
    assert flow.snapshot().persisted_flag_ids == (ids[0],)
    assert tuple(flag_id_for_segment(s) for s in flow.snapshot().pending_segments) == (ids[1],)
    async def success(pool,seg):
        calls.append(flag_id_for_segment(seg))
        return flag_id_for_segment(seg)
    monkeypatch.setattr("backend.app.classifier.segment_lifecycle.save_segment_flag",success)
    assert asyncio.run(persist_closed_segments(flow,object())) == (ids[1],)
    assert calls == [ids[0],ids[1],ids[1]]


def test_wrong_persistence_id_is_not_acknowledged(monkeypatch):
    w = make_window()
    flow = coordinator(w)
    flow.accept_window(w)
    flow.finish(pending_live_processing_complete=True)
    monkeypatch.setattr("backend.app.classifier.segment_lifecycle.save_segment_flag",AsyncMock(return_value=uuid4()))
    with pytest.raises(RuntimeError,match="unexpected segment identity"):
        asyncio.run(persist_closed_segments(flow,object()))
    assert len(flow.snapshot().pending_segments) == 1


def test_real_task12_mapper_and_persistence_receive_only_final_closed_payload():
    # Uses the verified Task 12 implementation and its transactional connection
    # double. The separate PostgreSQL smoke command tests the real SQL boundary.
    from tests.classifier.test_flag_persistence import Connection, Pool
    a = make_window(confidence=.7)
    b = make_window(1, confidence=.9, ranges=((0,2,14,20),))
    clean = make_window(2, category="no_ai")
    flow = coordinator(a)
    pool = Pool(Connection([a,b,clean]))
    flow.accept_window(a)
    assert asyncio.run(persist_closed_segments(flow,pool)) == ()
    assert not pool.connection.flags
    flow.accept_window(b)
    assert asyncio.run(persist_closed_segments(flow,pool)) == ()
    state = flow.accept_window(clean)
    saved = asyncio.run(persist_closed_segments(flow,pool))
    assert len(saved) == 1 and saved[0] == flag_id_for_segment(state.closed_segments[0])
    row = pool.connection.flags[saved[0]]
    assert row["detected_end_line"] == 20 and row["window_count"] == 2
    assert row["representative_prediction_id"] == b.prediction_id
    assert len(pool.connection.links) == 2
