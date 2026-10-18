from dataclasses import replace
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest

from backend.app.classifier.localization import LocalizedCodeRegion
from backend.app.classifier.flag_identity import flag_id_for_segment
from backend.app.classifier.segment_builder import ClassifiedWindow, build_segment_drafts

_BASE = datetime(2026, 10, 18, tzinfo=timezone.utc)
_SESSION = uuid4()
_TASK = uuid4()


def _window(n, *, ranges=((0, 2, 10, 15),), confidence=0.85):
    prediction_id, feature_vector_id = uuid4(), uuid4()
    start = _BASE + timedelta(seconds=10 * n)
    probs = {"no_ai": 0.05, "ide_ai": 0.95-confidence, "external_ai": confidence}
    regions = tuple(
        LocalizedCodeRegion(
            prediction_id=prediction_id,
            feature_vector_id=feature_vector_id,
            session_id=_SESSION,
            task_id=_TASK,
            model_version="fixture-v1",
            predicted_category="external_ai",
            confidence=confidence,
            window_start=start,
            window_end=start + timedelta(seconds=30),
            load_seq=load,
            version_id=version,
            start_line=lo,
            end_line=hi,
            edit_events=({"load_seq": load, "version_id": version,
                          "batch_seq": n, "seq_in_batch": index,
                          "change_index": 0},),
        )
        for index, (load, version, lo, hi) in enumerate(ranges)
    )
    return ClassifiedWindow(
        prediction_id=prediction_id,
        feature_vector_id=feature_vector_id,
        session_id=_SESSION,
        task_id=_TASK,
        model_version="fixture-v1",
        predicted_category="external_ai",
        class_probabilities=probs,
        confidence=confidence,
        ai_positive=True,
        window_start=start,
        window_end=start + timedelta(seconds=30),
        regions=regions,
    )


def test_same_segment_rebuilt_and_saved_again_receives_same_uuid():
    first = _window(0)
    a, = build_segment_drafts([first])
    b, = build_segment_drafts([first])
    assert isinstance(flag_id_for_segment(a), UUID)
    assert flag_id_for_segment(a) == flag_id_for_segment(b)


def test_two_regions_from_one_prediction_have_different_uuids():
    first = _window(0, ranges=((0, 2, 10, 15), (0, 2, 30, 35)))
    a, b = build_segment_drafts([first])
    assert a.prediction_ids == b.prediction_ids
    assert flag_id_for_segment(a) != flag_id_for_segment(b)


def test_merged_prediction_and_confidence_do_not_change_existing_identity():
    first, second = _window(0, confidence=0.7), _window(1, ranges=((0, 2, 15, 20),), confidence=0.9)
    before, = build_segment_drafts([first])
    after, = build_segment_drafts([first, second])
    assert after.window_count == 2
    assert after.confidence == 0.9
    assert (before.start_line, before.end_line) != (after.start_line, after.end_line)
    assert flag_id_for_segment(before) == flag_id_for_segment(after)


def test_separate_identical_episode_source_but_new_prediction_has_new_id():
    first = _window(0)
    a, = build_segment_drafts([first])
    new_source = replace(first.regions[0], prediction_id=uuid4())
    b = replace(a, members=(new_source,))
    assert (a.start_line, a.end_line, a.category) == (b.start_line, b.end_line, b.category)
    assert flag_id_for_segment(a) != flag_id_for_segment(b)


def test_page_load_and_version_distinguish_independent_flags():
    pair = _window(0, ranges=((0, 2, 10, 15), (1, 2, 10, 15), (0, 3, 10, 15)))
    segments = build_segment_drafts([pair])
    assert len(set(map(flag_id_for_segment, segments))) == 3


def test_single_line_bridge_stays_independent_from_neighbors():
    first = _window(0, ranges=((0, 2, 10, 10), (0, 2, 12, 12)))
    bridge = _window(1, ranges=((0, 2, 11, 11),))
    segments = build_segment_drafts([first, bridge])
    assert len(segments) == 3
    assert len(set(map(flag_id_for_segment, segments))) == 3


def test_missing_or_inconsistent_original_event_rejected():
    original, = build_segment_drafts([_window(0)])
    region = original.members[0]
    with pytest.raises(ValueError, match="no original edit evidence"):
        flag_id_for_segment(replace(original, members=(replace(region, edit_events=()),)))
    bad = dict(region.edit_events[0])
    bad["batch_seq"] = None
    with pytest.raises(ValueError, match="batch_seq"):
        flag_id_for_segment(replace(original, members=(replace(region, edit_events=(bad,)),)))
    bad = dict(region.edit_events[0])
    bad["load_seq"] = 9
    with pytest.raises(ValueError, match="document identity"):
        flag_id_for_segment(replace(original, members=(replace(region, edit_events=(bad,)),)))


def test_mismatched_first_region_rejected():
    original, = build_segment_drafts([_window(0)])
    with pytest.raises(ValueError, match="does not match"):
        flag_id_for_segment(replace(original, task_id=uuid4()))
