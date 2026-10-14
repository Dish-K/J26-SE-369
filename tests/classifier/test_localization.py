import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from backend.app.classifier.localization import fetch_prediction_edit_evidence
from shared_features import EDIT_EVIDENCE_SQL_ASYNCPG


class FakeConnection:
    def __init__(self, prediction, edit_events=()):
        self.prediction = prediction
        self.edit_events = edit_events
        self.calls = []

    async def fetchrow(self, sql, *args):
        self.calls.append(("fetchrow", sql, args))
        return self.prediction

    async def fetch(self, sql, *args):
        self.calls.append(("fetch", sql, args))
        return self.edit_events


def example_prediction(**overrides):
    now = datetime.now(timezone.utc)
    prediction = {
        "prediction_id": uuid4(),
        "feature_vector_id": uuid4(),
        "session_id": uuid4(),
        "task_id": uuid4(),
        "model_version": "controlled-fixture-v1",
        "predicted_category": "ide_ai",
        "confidence": 0.90,
        "ai_positive": True,
        "window_start": now - timedelta(seconds=30),
        "window_end": now,
        "run_id": "live",
    }
    prediction.update(overrides)
    return prediction


def test_positive_prediction_fetches_c1_evidence_using_exact_window_and_task():
    prediction = example_prediction()
    event = {
        "session_id": prediction["session_id"],
        "task_id": prediction["task_id"],
        "load_seq": 2,
        "version_id": 7,
        "change_index": 0,
        "touched_start_line": 10,
        "touched_end_line": 11,
    }
    connection = FakeConnection(prediction, [event])
    result = asyncio.run(fetch_prediction_edit_evidence(connection, prediction["prediction_id"]))

    assert result is not None
    assert result.predicted_category == "ide_ai"
    assert result.prediction_id == prediction["prediction_id"]
    assert result.edit_events == (event,)
    assert connection.calls[1] == (
        "fetch",
        EDIT_EVIDENCE_SQL_ASYNCPG,
        (
            prediction["session_id"],
            prediction["task_id"],
            prediction["window_start"],
            prediction["window_end"],
        ),
    )


def test_clean_prediction_does_not_fetch_or_localize():
    prediction = example_prediction(ai_positive=False, predicted_category="no_ai")
    connection = FakeConnection(prediction)
    result = asyncio.run(fetch_prediction_edit_evidence(connection, prediction["prediction_id"]))
    assert result is None
    assert [name for name, _, _ in connection.calls] == ["fetchrow"]


def test_positive_prediction_with_no_edits_keeps_empty_evidence():
    prediction = example_prediction(predicted_category="external_ai")
    result = asyncio.run(
        fetch_prediction_edit_evidence(FakeConnection(prediction), prediction["prediction_id"])
    )
    assert result is not None
    assert result.edit_events == ()


def test_missing_prediction_fails_visibly():
    with pytest.raises(LookupError):
        asyncio.run(fetch_prediction_edit_evidence(FakeConnection(None), uuid4()))


def test_final_stream_is_not_silently_mixed_with_live():
    prediction = example_prediction(run_id="final")
    with pytest.raises(ValueError, match="run_id='live'"):
        asyncio.run(fetch_prediction_edit_evidence(FakeConnection(prediction), prediction["prediction_id"]))


def test_inconsistent_positive_category_is_rejected():
    prediction = example_prediction(ai_positive=True, predicted_category="no_ai")
    with pytest.raises(ValueError, match="AI-associated"):
        asyncio.run(fetch_prediction_edit_evidence(FakeConnection(prediction), prediction["prediction_id"]))


from backend.app.classifier.localization import (
    PredictionEditEvidence,
    localize_prediction_edit_evidence,
)


def fixture_edit(prediction, *, load=0, version=2, start=5, end=None,
                 seq=1, batch=0, change=0, inserted=3, removed=0, **other):
    if end is None:
        end = start
    event = {
        "session_id": prediction["session_id"],
        "task_id": prediction["task_id"],
        "load_seq": load,
        "version_id": version,
        "touched_start_line": start,
        "touched_end_line": end,
        "replaced_start_line": start,
        "replaced_end_line": start,
        "batch_seq": batch,
        "seq_in_batch": seq,
        "change_index": change,
        "inserted_len": inserted,
        "removed_len": removed,
    }
    event.update(other)
    return event


def as_evidence(prediction, *rows):
    return PredictionEditEvidence(
        prediction_id=prediction["prediction_id"],
        feature_vector_id=prediction["feature_vector_id"],
        session_id=prediction["session_id"],
        task_id=prediction["task_id"],
        model_version=prediction["model_version"],
        predicted_category=prediction["predicted_category"],
        confidence=prediction["confidence"],
        window_start=prediction["window_start"],
        window_end=prediction["window_end"],
        edit_events=rows,
    )


def test_two_disconnected_edits_stay_independent():
    pred = example_prediction()
    rows = (fixture_edit(pred, start=8, end=8, seq=1),
            fixture_edit(pred, start=30, end=32, seq=2))
    result = localize_prediction_edit_evidence(as_evidence(pred, *rows))
    assert [(r.start_line, r.end_line) for r in result.regions] == [(8, 8), (30, 32)]
    assert all(r.predicted_category == "ide_ai" and r.confidence == .9 for r in result.regions)
    assert len(result.unlocalized_events) == 0


def test_same_version_adjacent_and_overlapping_regions_join_but_not_gaps():
    pred = example_prediction()
    rows = (
        fixture_edit(pred, start=10, end=12, seq=1),
        fixture_edit(pred, start=14, end=16, seq=2),
        fixture_edit(pred, start=13, end=13, seq=3),
        fixture_edit(pred, start=40, end=41, seq=4),
        fixture_edit(pred, start=15, end=17, seq=5),
    )
    result = localize_prediction_edit_evidence(as_evidence(pred, *rows))
    assert [(r.start_line, r.end_line) for r in result.regions] == [(10, 17), (40, 41)]
    assert [r["seq_in_batch"] for r in result.regions[0].edit_events] == [1, 2, 3, 5]


def test_different_document_versions_never_union_without_rebasing():
    pred = example_prediction()
    rows = (fixture_edit(pred, version=2, start=10, end=12, seq=1),
            fixture_edit(pred, version=3, start=12, end=13, seq=2),
            fixture_edit(pred, version=2, load=1, start=10, end=12, seq=3))
    result = localize_prediction_edit_evidence(as_evidence(pred, *rows))
    assert len(result.regions) == 3
    assert [(r.load_seq, r.version_id) for r in result.regions] == [(0, 2), (0, 3), (1, 2)]


def test_missing_invalid_locations_are_preserved_as_unlocalized_evidence():
    pred = example_prediction()
    bad = fixture_edit(pred, start=None, end=20)
    bad2 = fixture_edit(pred, start=12, end=11, seq=2)
    result = localize_prediction_edit_evidence(as_evidence(pred, bad, bad2))
    assert result.regions == ()
    assert result.unlocalized_events == (bad, bad2)


def test_option_a_deletion_only_does_not_flag_shifted_post_edit_code():
    pred = example_prediction()
    deleted = fixture_edit(
        pred, start=10, end=10, version=7, inserted=0, removed=90,
        replaced_start_line=10, replaced_end_line=15, net_line_delta=-5,
    )
    result = localize_prediction_edit_evidence(as_evidence(pred, deleted))
    assert result.regions == ()
    assert result.unlocalized_events == (deleted,)
    assert result.unlocalized_reasons == ("deletion_only",)
    assert result.prediction_id == pred["prediction_id"]
    assert result.unlocalized_events[0]["replaced_end_line"] == 15


def test_deletion_does_not_block_valid_regions_or_merge_through_deleted_lines():
    pred = example_prediction()
    before = fixture_edit(pred, seq=1, start=8, end=8, inserted=4)
    deleted = fixture_edit(
        pred, seq=2, start=9, end=18, inserted=0, removed=30,
        replaced_start_line=9, replaced_end_line=18,
    )
    after = fixture_edit(pred, seq=3, start=20, end=21, inserted=4)
    result = localize_prediction_edit_evidence(as_evidence(pred, before, deleted, after))
    assert [(r.start_line, r.end_line) for r in result.regions] == [(8, 8), (20, 21)]
    assert result.unlocalized_events == (deleted,)
    assert result.unlocalized_reasons == ("deletion_only",)
    assert [r.edit_events for r in result.regions] == [(before,), (after,)]


def test_missing_or_bad_lengths_are_not_treated_as_safe_insertion():
    pred = example_prediction()
    missing = fixture_edit(pred, inserted=None, removed=4)
    negative = fixture_edit(pred, seq=2, inserted=-1)
    good = fixture_edit(pred, seq=3, start=30)
    result = localize_prediction_edit_evidence(as_evidence(pred, missing, negative, good))
    assert [(r.start_line, r.end_line) for r in result.regions] == [(30, 30)]
    assert result.unlocalized_events == (missing, negative)
    assert result.unlocalized_reasons == (
        "missing_or_invalid_edit_lengths", "missing_or_invalid_edit_lengths",
    )


def test_no_change_event_is_not_a_new_flagged_code_region():
    pred = example_prediction()
    no_op = fixture_edit(pred, inserted=0, removed=0)
    result = localize_prediction_edit_evidence(as_evidence(pred, no_op))
    assert result.regions == ()
    assert result.unlocalized_events == (no_op,)
    assert result.unlocalized_reasons == ("no_code_change",)


def test_mixed_unmappable_event_reasons_follow_original_event_order():
    pred = example_prediction()
    invalid = fixture_edit(pred, seq=1, start=0, end=2)
    deleted = fixture_edit(pred, seq=2, inserted=0, removed=1)
    missing = fixture_edit(pred, seq=3, inserted=None)
    valid = fixture_edit(pred, seq=4, start=11, end=12)
    result = localize_prediction_edit_evidence(
        as_evidence(pred, invalid, deleted, missing, valid)
    )
    assert [(r.start_line, r.end_line) for r in result.regions] == [(11, 12)]
    assert result.unlocalized_events == (invalid, deleted, missing)
    assert result.unlocalized_reasons == (
        "invalid_location_or_version", "deletion_only", "missing_or_invalid_edit_lengths",
    )


def test_task_11_handoff_keeps_version_and_full_prediction_provenance():
    pred = example_prediction(model_version="dummy-fixture-v1", predicted_category="external_ai")
    edit = fixture_edit(
        pred, load=3, version=11, start=40, end=42, batch=7, seq=9, change=2,
    )
    result = localize_prediction_edit_evidence(as_evidence(pred, edit))
    region, = result.regions
    assert (
        region.prediction_id, region.feature_vector_id, region.session_id,
        region.task_id, region.model_version, region.predicted_category,
        region.confidence, region.window_start, region.window_end,
    ) == (
        pred["prediction_id"], pred["feature_vector_id"], pred["session_id"],
        pred["task_id"], pred["model_version"], pred["predicted_category"],
        pred["confidence"], pred["window_start"], pred["window_end"],
    )
    assert (region.load_seq, region.version_id, region.start_line, region.end_line) == (3, 11, 40, 42)
    assert region.edit_events == (edit,)
    assert region.edit_events[0]["change_index"] == 2
    assert region.edit_events[0]["batch_seq"] == 7


def test_cross_task_evidence_is_rejected_not_silently_attributed():
    pred = example_prediction()
    bad = fixture_edit(pred, task_id=uuid4())
    with pytest.raises(ValueError, match="session/task"):
        localize_prediction_edit_evidence(as_evidence(pred, bad))


def test_empty_evidence_does_not_invent_a_region():
    pred = example_prediction()
    result = localize_prediction_edit_evidence(as_evidence(pred))
    assert result.regions == ()
    assert result.unlocalized_events == ()


def test_multiple_categories_remain_independent_per_prediction():
    pred = example_prediction()
    different = example_prediction(predicted_category="external_ai")
    one = localize_prediction_edit_evidence(as_evidence(pred, fixture_edit(pred, start=9)))
    other = localize_prediction_edit_evidence(as_evidence(different, fixture_edit(different, start=9)))
    assert one.regions[0].predicted_category == "ide_ai"
    assert other.regions[0].predicted_category == "external_ai"
    assert one.regions[0].prediction_id != other.regions[0].prediction_id
