from dataclasses import replace
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest

from backend.app.classifier.flag_identity import flag_id_for_segment
from backend.app.classifier.flag_payload import map_segment_to_flag
from backend.app.classifier.localization import LocalizedCodeRegion
from backend.app.classifier.segment_builder import ClassifiedWindow, build_segment_drafts

_BASE = datetime(2026, 10, 18, 9, 0, tzinfo=timezone.utc)
_SESSION, _TASK = uuid4(), uuid4()


def _window(n, *, category="external_ai", confidence=.8, ranges=((0, 2, 10, 15),)):
    prediction_id, feature_id = uuid4(), uuid4()
    first = _BASE + timedelta(seconds=10 * n)
    last = first + timedelta(seconds=30)
    other = "ide_ai" if category == "external_ai" else "external_ai"
    probs = {"no_ai": .05, other: .95 - confidence, category: confidence}
    regions = tuple(
        LocalizedCodeRegion(
            prediction_id=prediction_id, feature_vector_id=feature_id,
            session_id=_SESSION, task_id=_TASK, model_version="fixture-v1",
            predicted_category=category, confidence=confidence,
            window_start=first, window_end=last,
            load_seq=load, version_id=version, start_line=lo, end_line=hi,
            edit_events=({"load_seq": load, "version_id": version,
                          "batch_seq": n, "seq_in_batch": index,
                          "change_index": 0},),
        )
        for index, (load, version, lo, hi) in enumerate(ranges)
    )
    return ClassifiedWindow(
        prediction_id=prediction_id, feature_vector_id=feature_id,
        session_id=_SESSION, task_id=_TASK, model_version="fixture-v1",
        predicted_category=category, class_probabilities=probs,
        confidence=confidence, ai_positive=True,
        window_start=first, window_end=last, regions=regions,
    )


def test_single_window_exact_mapping_and_no_invented_evidence():
    original = _window(0)
    segment, = build_segment_drafts([original])
    data = map_segment_to_flag(segment)
    assert isinstance(data.flag_id, UUID)
    assert data.flag_id == flag_id_for_segment(segment)
    assert data.task_id == _TASK
    assert data.model_version == "fixture-v1"
    assert data.category == "external_ai"
    assert (data.detected_start_line, data.detected_end_line) == (10, 15)
    assert (data.load_seq, data.version_id) == (0, 2)
    assert data.evidence_kinds == ()
    assert data.segment_basis == "window_run"
    assert data.window_count == 1
    assert (data.detected_at, data.detected_until) == (original.window_start, original.window_end)
    assert data.prediction_ids == (original.prediction_id,)
    assert data.representative_prediction_id == original.prediction_id
    assert data.class_probabilities == dict(original.class_probabilities)
    assert data.confidence == original.confidence


def test_merged_windows_use_first_start_last_end_and_highest_actual_member():
    a, b = _window(0, confidence=.7), _window(1, confidence=.9, ranges=((0, 2, 14, 20),))
    segment, = build_segment_drafts([a, b])
    data = map_segment_to_flag(segment)
    assert data.detected_at == _BASE
    assert data.detected_until == _BASE + timedelta(seconds=40)
    assert data.window_count == 2
    assert data.prediction_ids == (a.prediction_id, b.prediction_id)
    assert data.representative_prediction_id == b.prediction_id
    assert data.confidence == .9
    assert data.class_probabilities == dict(b.class_probabilities)
    assert data.segment_basis == "window_run"
    assert data.flag_id == flag_id_for_segment(segment)


def test_independent_same_prediction_regions_remain_separate():
    first = _window(0, ranges=((0, 2, 8, 8), (0, 2, 30, 32)))
    a, b = build_segment_drafts([first])
    left, right = map_segment_to_flag(a), map_segment_to_flag(b)
    assert left.flag_id != right.flag_id
    assert left.prediction_ids == right.prediction_ids == (first.prediction_id,)


def test_bridge_remains_three_distinct_identifiers_and_no_category_heuristics():
    first = _window(0, ranges=((0, 2, 10, 10), (0, 2, 12, 12)))
    bridge = _window(1, ranges=((0, 2, 11, 11),))
    records = [map_segment_to_flag(seg) for seg in build_segment_drafts([first, bridge])]
    assert len(records) == len({rec.flag_id for rec in records}) == 3
    assert all(rec.segment_basis == "window_run" for rec in records)
    assert all(rec.evidence_kinds == () for rec in records)


def test_canonical_evidence_names_explicit_only_and_deduplicated():
    segment, = build_segment_drafts([_window(0)])
    data = map_segment_to_flag(
        segment, verified_evidence_kinds=["paste", "paste", "typing_burst"]
    )
    assert data.evidence_kinds == ("paste", "typing_burst")
    for kinds in (["unverified"], ["paste", None], "paste"):
        with pytest.raises(ValueError):
            map_segment_to_flag(segment, verified_evidence_kinds=kinds)


def test_both_positive_categories_supported_without_modifying_classification():
    w = _window(0, category="ide_ai", confidence=.8)
    segment, = build_segment_drafts([w])
    rec = map_segment_to_flag(segment)
    assert rec.category == "ide_ai"
    assert rec.confidence == .8
    assert rec.class_probabilities == dict(w.class_probabilities)


def test_rejects_draft_with_forged_or_inconsistent_representative():
    w = _window(0)
    seg, = build_segment_drafts([w])
    with pytest.raises(ValueError, match="Representative prediction"):
        map_segment_to_flag(replace(seg, representative_prediction_id=uuid4()))
    with pytest.raises(ValueError, match="Segment confidence"):
        map_segment_to_flag(replace(seg, confidence=.9))


def test_rejects_incorrect_observation_period():
    seg, = build_segment_drafts([_window(0)])
    with pytest.raises(ValueError, match="observation timestamps"):
        map_segment_to_flag(replace(seg, last_window_end=seg.last_window_end + timedelta(seconds=1)))
    with pytest.raises(ValueError, match="timezone-aware"):
        map_segment_to_flag(replace(seg, first_window_start=seg.first_window_start.replace(tzinfo=None)))


def test_rejects_missing_member_evidence_and_mismatched_provenance():
    seg, = build_segment_drafts([_window(0)])
    region = seg.members[0]
    with pytest.raises(ValueError, match="edit-event evidence"):
        map_segment_to_flag(replace(seg, members=(region, replace(region, edit_events=(), prediction_id=uuid4()))))
    with pytest.raises(ValueError, match="mismatched provenance"):
        map_segment_to_flag(replace(seg, members=(region, replace(region, task_id=uuid4(), prediction_id=uuid4()))))


def test_rejects_out_of_db_range_and_invalid_probabilities():
    seg, = build_segment_drafts([_window(0)])
    with pytest.raises(ValueError, match="database range"):
        map_segment_to_flag(replace(seg, load_seq=32768))
    with pytest.raises(ValueError, match="probabilities"):
        map_segment_to_flag(replace(seg, class_probabilities={"external_ai": 1.0}))


def test_repeated_completed_draft_maps_to_same_identity_and_metadata():
    w = _window(0)
    first, = build_segment_drafts([w])
    again, = build_segment_drafts([w])
    assert map_segment_to_flag(first) == map_segment_to_flag(again)
