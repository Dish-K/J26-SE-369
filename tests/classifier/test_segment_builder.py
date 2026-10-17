from dataclasses import replace
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from backend.app.classifier.localization import LocalizedCodeRegion
from backend.app.classifier.segment_builder import ClassifiedWindow, build_segment_drafts

_BASE = datetime(2026, 10, 17, tzinfo=timezone.utc)
_SESSION = uuid4()
_TASK = uuid4()
_MODEL = "fixture-model-v1"


def window(n: int, *, category="external_ai", positive=True, task=None,
           model=_MODEL, confidence=.85, session=_SESSION, ranges=((0, 2, 10, 15),)):
    task = task or _TASK
    prediction_id, feature_id = uuid4(), uuid4()
    start = _BASE + timedelta(seconds=10 * n)
    end = start + timedelta(seconds=30)
    probabilities = {
        "external_ai": confidence if category == "external_ai" else .05,
        "ide_ai": confidence if category == "ide_ai" else .05,
        "no_ai": confidence if category == "no_ai" else .05,
    }
    other = [name for name in probabilities if name != category]
    probabilities[other[0]] = round((1 - confidence) / 2, 10)
    probabilities[other[1]] = 1 - confidence - probabilities[other[0]]
    regions = tuple(
        LocalizedCodeRegion(
            prediction_id=prediction_id, feature_vector_id=feature_id,
            session_id=session, task_id=task, model_version=model,
            predicted_category=category, confidence=confidence,
            window_start=start, window_end=end, load_seq=load, version_id=version,
            start_line=lo, end_line=hi,
            edit_events=({"load_seq": load, "version_id": version,
                          "batch_seq": n, "seq_in_batch": i, "change_index": 0},),
        )
        for i, (load, version, lo, hi) in enumerate(ranges)
    ) if positive else ()
    return ClassifiedWindow(
        prediction_id=prediction_id, feature_vector_id=feature_id,
        session_id=session, task_id=task, model_version=model,
        predicted_category=category, class_probabilities=probabilities,
        confidence=confidence, ai_positive=positive,
        window_start=start, window_end=end, regions=regions,
    )


def test_overlapping_consecutive_same_version_merges_with_full_provenance():
    first = window(0, ranges=((0, 2, 12, 20),), confidence=.7)
    second = window(1, ranges=((0, 2, 18, 25),), confidence=.9)
    segments = build_segment_drafts([second, first])
    assert len(segments) == 1
    segment, = segments
    assert (segment.start_line, segment.end_line) == (12, 25)
    assert segment.prediction_ids == (first.prediction_id, second.prediction_id)
    assert segment.window_count == 2
    assert segment.representative_prediction_id == second.prediction_id
    assert segment.confidence == .9
    assert segment.class_probabilities == dict(second.class_probabilities)
    assert [r.edit_events[0]["batch_seq"] for r in segment.members] == [0, 1]


def test_directly_adjacent_ranges_merge_but_one_line_gap_does_not():
    assert len(build_segment_drafts([window(0), window(1, ranges=((0, 2, 16, 18),))])) == 1
    assert len(build_segment_drafts([window(0), window(1, ranges=((0, 2, 17, 18),))])) == 2


def test_clean_or_below_threshold_breaks_episodes():
    a, clean, b = window(0), window(1, category="no_ai", positive=False), window(2)
    assert len(build_segment_drafts([a, clean, b])) == 2
    below = window(1, positive=False)
    assert len(build_segment_drafts([a, below, b])) == 2


def test_skipped_window_does_not_create_false_continuity():
    assert len(build_segment_drafts([window(0), window(2)])) == 2


def test_separate_categories_tasks_and_models_never_merge():
    assert len(build_segment_drafts([window(0), window(1, category="ide_ai")])) == 2
    assert len(build_segment_drafts([window(0), window(1, task=uuid4())])) == 2
    assert len(build_segment_drafts([window(0), window(1, model="other-model")])) == 2


def test_separate_regions_in_same_window_remain_distinct():
    pair = window(0, ranges=((0, 2, 8, 8), (0, 2, 30, 32)))
    drafts = build_segment_drafts([pair])
    assert [(d.start_line, d.end_line) for d in drafts] == [(8, 8), (30, 32)]
    assert all(d.prediction_ids == (pair.prediction_id,) for d in drafts)


def test_identical_lines_different_document_versions_do_not_merge():
    a = window(0, ranges=((0, 7, 10, 15),))
    b = window(1, ranges=((0, 8, 10, 15),))
    c = window(2, ranges=((1, 7, 10, 15),))
    assert len(build_segment_drafts([a, b, c])) == 3


def test_representative_probability_ties_keep_first_prediction():
    a, b = window(0), window(1)
    segment, = build_segment_drafts([a, b])
    assert segment.representative_prediction_id == a.prediction_id
    assert segment.class_probabilities == dict(a.class_probabilities)


def test_valid_region_after_unlocalized_positive_window_starts_new_segment():
    a = window(0)
    empty = window(1, ranges=())
    c = window(2)
    assert len(build_segment_drafts([a, empty, c])) == 2


def test_does_not_mutate_localized_regions_or_assign_answer_confidence():
    first, second = window(0), window(1)
    before = first.regions[0]
    segment, = build_segment_drafts([first, second])
    assert first.regions[0] is before
    assert not hasattr(segment, "ai_confidence")
    assert segment.members == (first.regions[0], second.regions[0])


def test_single_line_bridge_stays_independent():
    one = window(0, ranges=((0, 2, 10, 10), (0, 2, 12, 12)))
    bridge = window(1, ranges=((0, 2, 11, 11),))
    drafts = build_segment_drafts([one, bridge])
    assert [(s.start_line, s.end_line) for s in drafts] == [
        (10, 10), (12, 12), (11, 11),
    ]
    assert [s.prediction_ids for s in drafts] == [
        (one.prediction_id,), (one.prediction_id,), (bridge.prediction_id,),
    ]


def test_validation_rejects_mismatched_prediction_identity_and_probabilities():
    valid = window(0)
    with pytest.raises(ValueError, match="Localized region"):
        build_segment_drafts([replace(valid, regions=(replace(valid.regions[0], task_id=uuid4()),))])
    with pytest.raises(ValueError, match="Confidence"):
        build_segment_drafts([replace(valid, confidence=.1)])
    with pytest.raises(ValueError, match="Duplicate"):
        build_segment_drafts([valid, valid])
    with pytest.raises(ValueError, match="non-positive"):
        build_segment_drafts([replace(valid, ai_positive=False)])


def test_separate_streams_can_be_interleaved():
    task2 = uuid4()
    a, b, c, d = window(0), window(0, task=task2), window(1), window(1, task=task2)
    segments = build_segment_drafts([a, b, c, d])
    assert len(segments) == 2
    assert all(s.window_count == 2 for s in segments)


def edit(version, *, start, end=None, inserted_lines=0, load=0, task=_TASK,
         session=_SESSION, batch=None, seq=0, change=0):
    end = start if end is None else end
    return {
        "session_id": session, "task_id": task, "load_seq": load,
        "version_id": version, "batch_seq": version if batch is None else batch,
        "seq_in_batch": seq, "change_index": change,
        "start_line": start, "end_line": end,
        "inserted_lines": inserted_lines,
        "net_line_delta": inserted_lines - (end - start),
        "origin": 0,
    }


def test_cross_version_insertion_above_range_merges_after_safe_rebase():
    old = window(0, ranges=((0, 2, 10, 15),), confidence=.75)
    new = window(1, ranges=((0, 3, 15, 20),), confidence=.9)
    change = edit(3, start=2, inserted_lines=5)
    segment, = build_segment_drafts([old, new], edit_history=[change])
    assert (segment.load_seq, segment.version_id, segment.start_line, segment.end_line) == (0, 3, 15, 20)
    assert segment.prediction_ids == (old.prediction_id, new.prediction_id)
    assert segment.members[0].version_id == 2
    assert segment.members[0].start_line == 10
    assert segment.representative_prediction_id == new.prediction_id


def test_cross_version_deletion_above_range_rebases_up():
    old = window(0, ranges=((0, 2, 10, 15),))
    new = window(1, ranges=((0, 3, 7, 12),))
    result, = build_segment_drafts([old, new], edit_history=[edit(3, start=2, end=5)])
    assert (result.version_id, result.start_line, result.end_line) == (3, 7, 12)


def test_multichange_version_uses_common_pre_edit_coordinate_space():
    first = window(0, ranges=((0, 2, 10, 15),))
    second = window(1, ranges=((0, 3, 14, 19),))
    changes = [
        edit(3, start=1, inserted_lines=2, seq=0, change=0),
        edit(3, start=4, inserted_lines=2, seq=0, change=1),
    ]
    seg, = build_segment_drafts([first, second], edit_history=list(reversed(changes)))
    assert (seg.start_line, seg.end_line) == (14, 19)


def test_cross_version_region_intersected_by_edit_does_not_merge():
    first = window(0, ranges=((0, 2, 10, 15),))
    second = window(1, ranges=((0, 3, 10, 15),))
    assert len(build_segment_drafts([first, second], edit_history=[edit(3, start=12, end=13, inserted_lines=1)])) == 2


def test_cross_version_region_deleted_later_is_not_mapped_to_replacement():
    first = window(0, ranges=((0, 2, 10, 15),))
    second = window(1, ranges=((0, 3, 10, 15),))
    assert len(build_segment_drafts([first, second], edit_history=[edit(3, start=10, end=15)])) == 2


def test_missing_intermediate_version_keeps_segments_independent():
    first = window(0, ranges=((0, 2, 10, 15),))
    second = window(1, ranges=((0, 4, 10, 15),))
    assert len(build_segment_drafts([first, second], edit_history=[edit(4, start=40, inserted_lines=1)])) == 2


def test_same_line_numbers_different_versions_do_not_merge_without_history():
    first = window(0, ranges=((0, 2, 10, 15),))
    second = window(1, ranges=((0, 3, 10, 15),))
    assert len(build_segment_drafts([first, second])) == 2


def test_reload_boundary_never_merges_even_with_history():
    first = window(0, ranges=((0, 2, 10, 15),))
    second = window(1, ranges=((1, 3, 10, 15),))
    assert len(build_segment_drafts([first, second], edit_history=[edit(3, start=40, inserted_lines=1, load=1)])) == 2


def test_invalid_or_duplicate_edit_history_fails_closed():
    first = window(0, ranges=((0, 2, 10, 15),))
    second = window(1, ranges=((0, 3, 15, 20),))
    base = edit(3, start=2, inserted_lines=5)
    assert len(build_segment_drafts([first, second], edit_history=[{**base, "net_line_delta": 99}])) == 2
    assert len(build_segment_drafts([first, second], edit_history=[base, base])) == 2


def test_older_region_arriving_in_later_window_maps_to_latest_reference():
    first = window(0, ranges=((0, 4, 15, 20),), confidence=.8)
    second = window(1, ranges=((0, 3, 10, 15),), confidence=.9)
    seg, = build_segment_drafts([first, second], edit_history=[edit(4, start=2, inserted_lines=5)])
    assert (seg.version_id, seg.start_line, seg.end_line) == (4, 15, 20)
    assert seg.representative_prediction_id == second.prediction_id
    assert seg.members[1].version_id == 3


def test_history_from_another_task_does_not_authorize_mapping():
    first = window(0, ranges=((0, 2, 10, 15),))
    second = window(1, ranges=((0, 3, 15, 20),))
    wrong_task_history = edit(3, start=2, inserted_lines=5, task=uuid4())
    assert len(build_segment_drafts([first, second], edit_history=[wrong_task_history])) == 2


def test_multiple_versions_map_all_intervening_edits_in_order():
    old = window(0, ranges=((0, 2, 10, 15),))
    new = window(1, ranges=((0, 4, 13, 18),))
    events = [edit(3, start=2, inserted_lines=5), edit(4, start=4, end=6)]
    seg, = build_segment_drafts([old, new], edit_history=list(reversed(events)))
    assert (seg.version_id, seg.start_line, seg.end_line) == (4, 13, 18)

def test_multiline_bridge_preserves_three_independent_regions():
    separate = window(0, ranges=((0, 2, 8, 17), (0, 2, 26, 34)), confidence=.86)
    bridging = window(1, ranges=((0, 2, 18, 25),), confidence=.92)
    drafts = build_segment_drafts([separate, bridging])
    assert [(s.start_line, s.end_line) for s in drafts] == [
        (8, 17), (26, 34), (18, 25),
    ]
    assert all(s.window_count == 1 for s in drafts)
    assert [s.representative_prediction_id for s in drafts] == [
        separate.prediction_id, separate.prediction_id, bridging.prediction_id,
    ]


def test_repeated_bridge_merges_only_with_its_own_segment():
    first = window(0, ranges=((0, 2, 8, 17), (0, 2, 26, 34)), confidence=.75)
    bridging = window(1, ranges=((0, 2, 18, 25),), confidence=.80)
    repeated = window(2, ranges=((0, 2, 18, 25),), confidence=.95)
    drafts = build_segment_drafts([first, bridging, repeated])
    assert len(drafts) == 3
    assert [s.window_count for s in drafts] == [1, 1, 2]
    assert drafts[2].prediction_ids == (bridging.prediction_id, repeated.prediction_id)
    assert drafts[2].representative_prediction_id == repeated.prediction_id
    assert drafts[2].class_probabilities == dict(repeated.class_probabilities)
    assert drafts[2].members == (bridging.regions[0], repeated.regions[0])


def test_bridge_and_originals_can_all_receive_new_evidence_independently():
    original = window(0, ranges=((0, 2, 8, 17), (0, 2, 26, 34)))
    middle = window(1, ranges=((0, 2, 8, 17), (0, 2, 26, 34), (0, 2, 18, 25)))
    subsequent = window(2, ranges=((0, 2, 8, 17), (0, 2, 26, 34), (0, 2, 18, 25)))
    drafts = build_segment_drafts([original, middle, subsequent])
    assert len(drafts) == 3
    assert [(s.start_line, s.end_line) for s in drafts] == [
        (8, 17), (26, 34), (18, 25),
    ]
    assert all(s.prediction_ids[-1] == subsequent.prediction_id for s in drafts)
    assert [s.window_count for s in drafts] == [3, 3, 2]


def test_nonmatching_wide_bridge_does_not_rewrite_original_segments():
    original = window(0, ranges=((0, 2, 10, 12), (0, 2, 18, 20)))
    bridge = window(1, ranges=((0, 2, 11, 19),))
    drafts = build_segment_drafts([original, bridge])
    assert [(s.start_line, s.end_line) for s in drafts] == [
        (10, 12), (18, 20), (11, 19),
    ]
    assert all(s.window_count == 1 for s in drafts)


def test_safe_cross_version_bridge_keeps_original_versions_and_ranges():
    original = window(0, ranges=((0, 2, 8, 17), (0, 2, 26, 34)), confidence=.83)
    bridge = window(1, ranges=((0, 3, 23, 30),), confidence=.91)
    drafts = build_segment_drafts([original, bridge], edit_history=[edit(3, start=2, inserted_lines=5)])
    assert len(drafts) == 3
    assert [(s.load_seq, s.version_id, s.start_line, s.end_line) for s in drafts] == [
        (0, 2, 8, 17), (0, 2, 26, 34), (0, 3, 23, 30),
    ]
    assert drafts[0].members[0].version_id == 2
    assert drafts[1].members[0].version_id == 2
    assert drafts[2].members[0].version_id == 3


def test_two_regions_in_one_window_do_not_share_one_prior_segment():
    broad = window(0, ranges=((0, 2, 10, 20),))
    separated = window(1, ranges=((0, 2, 10, 12), (0, 2, 18, 20)))
    drafts = build_segment_drafts([broad, separated])
    assert len(drafts) == 2
    assert drafts[0].prediction_ids == (broad.prediction_id, separated.prediction_id)
    assert drafts[1].prediction_ids == (separated.prediction_id,)
    assert drafts[1].members == (separated.regions[1],)


def test_clean_window_after_bridge_starts_new_episodes():
    first = window(0, ranges=((0, 2, 10, 12), (0, 2, 14, 16)))
    bridge = window(1, ranges=((0, 2, 13, 13),))
    clean = window(2, category="no_ai", positive=False)
    return_to_bridge = window(3, ranges=((0, 2, 13, 13),))
    drafts = build_segment_drafts([first, bridge, clean, return_to_bridge])
    assert len(drafts) == 4
    assert drafts[-1].prediction_ids == (return_to_bridge.prediction_id,)


def test_bridging_single_line_keeps_a_valid_independent_flag_draft():
    first = window(0, ranges=((0, 2, 10, 10), (0, 2, 12, 12)))
    bridge = window(1, ranges=((0, 2, 11, 11),))
    drafts = build_segment_drafts([first, bridge])
    for draft in drafts:
        assert draft.category == "external_ai"
        assert draft.task_id == _TASK
        assert draft.model_version == _MODEL
        assert draft.confidence == .85
        assert draft.class_probabilities == dict(first.class_probabilities)
        assert draft.prediction_ids
        assert draft.window_count == len(draft.prediction_ids)
        assert draft.first_window_start <= draft.last_window_end
        assert draft.members


def test_later_wide_region_does_not_expand_bridge_across_old_independent_flags():
    separate = window(0, ranges=((0, 2, 8, 17), (0, 2, 26, 34)))
    bridging = window(1, ranges=((0, 2, 18, 25),))
    wide = window(2, ranges=((0, 2, 8, 34),))
    segments = build_segment_drafts([separate, bridging, wide])
    assert [(s.start_line, s.end_line) for s in segments] == [
        (8, 17), (26, 34), (18, 25), (8, 34),
    ]
