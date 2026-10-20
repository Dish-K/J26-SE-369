import asyncio
import copy
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import timedelta
from uuid import uuid4

import pytest

from backend.app.classifier.flag_identity import flag_id_for_segment
from backend.app.classifier.flag_persistence import (
    FlagPersistenceConflict, fetch_saved_flag, save_segment_flag,
    save_segment_flag_on_connection,
)
from backend.app.classifier.segment_builder import build_segment_drafts
from tests.classifier.flag_persistence_support import make_window, stored_prediction


class Connection:
    """Small transactional double; the separate smoke test exercises real SQL."""
    def __init__(self, windows):
        self.predictions = {w.prediction_id: stored_prediction(w) for w in windows}
        self.flags, self.links = {}, set()
        self.fail_after_first_link = None
        self.fail_after_commit = False
        self.transactions = []

    @asynccontextmanager
    async def transaction(self):
        original = copy.deepcopy((self.flags, self.links))
        self.transactions.append("begin")
        try:
            yield
        except BaseException:
            self.flags, self.links = original
            self.transactions.append("rollback")
            raise
        else:
            self.transactions.append("commit")
            if self.fail_after_commit:
                self.fail_after_commit = False
                raise OSError("simulated lost commit acknowledgement")

    async def fetch(self, sql, value):
        if "FROM classifier_window_predictions" in sql:
            return [self.predictions[v] for v in value if v in self.predictions]
        return [{"prediction_id": p} for f, p in self.links if f == value]

    async def fetchval(self, sql, *args):
        # Check named SQL columns instead of borrowing production serialization.
        columns = [s.strip() for s in sql.split("(", 1)[1].split(")", 1)[0].split(",")]
        assert len(columns) == len(args) == 16
        row = dict(zip(columns, args))
        if row["flag_id"] in self.flags:
            return None
        self.flags[row["flag_id"]] = row
        return row["flag_id"]

    async def fetchrow(self, sql, flag_id):
        row = self.flags.get(flag_id)
        if row is None:
            return None
        result = dict(row)
        if "AS prediction_ids" in sql:
            result["prediction_ids"] = sorted(p for f, p in self.links if f == flag_id)
        return result

    async def executemany(self, sql, pairs):
        for pair in pairs:
            self.links.add(pair)
            if self.fail_after_first_link:
                raise self.fail_after_first_link


class Pool:
    def __init__(self, connection):
        self.connection, self.acquired, self.released = connection, 0, 0

    @asynccontextmanager
    async def acquire(self):
        self.acquired += 1
        try:
            yield self.connection
        finally:
            self.released += 1


def run(coroutine):
    return asyncio.run(coroutine)


def test_merged_flag_stores_approved_fields_and_all_links():
    a, b = make_window(), make_window(1, confidence=.9, ranges=((0, 2, 14, 20),))
    segment, = build_segment_drafts([a, b])
    conn = Connection([a, b])
    flag_id = run(save_segment_flag_on_connection(conn, segment))
    row = run(fetch_saved_flag(conn, flag_id))
    assert row["representative_prediction_id"] == b.prediction_id
    assert row["class_probabilities"] == b.class_probabilities
    assert row["confidence"] == b.confidence
    assert (row["detected_start_line"], row["detected_end_line"]) == (10, 20)
    assert (row["load_seq"], row["version_id"]) == (0, 2)
    assert row["segment_basis"] == "window_run"
    assert row["evidence_kinds"] == ()
    assert row["window_count"] == 2
    assert row["detected_at"] == a.window_start
    assert row["detected_until"] == b.window_end
    assert set(row["prediction_ids"]) == {a.prediction_id, b.prediction_id}


def test_identical_retry_returns_same_id_without_duplicating_rows():
    w = make_window()
    seg, = build_segment_drafts([w])
    conn = Connection([w])
    first = run(save_segment_flag_on_connection(conn, seg))
    before = copy.deepcopy((conn.flags, conn.links))
    assert run(save_segment_flag_on_connection(conn, seg)) == first
    assert (conn.flags, conn.links) == before


def test_independent_regions_share_prediction_but_keep_separate_ids():
    w = make_window(ranges=((0, 2, 10, 15), (0, 2, 30, 35)))
    conn = Connection([w])
    ids = [run(save_segment_flag_on_connection(conn, s)) for s in build_segment_drafts([w])]
    assert len(set(ids)) == len(conn.flags) == len(conn.links) == 2


def test_changed_historical_payload_is_not_silently_overwritten():
    w = make_window()
    seg, = build_segment_drafts([w])
    conn = Connection([w])
    run(save_segment_flag_on_connection(conn, seg))
    before = copy.deepcopy((conn.flags, conn.links))
    with pytest.raises(FlagPersistenceConflict, match="detected_end_line"):
        run(save_segment_flag_on_connection(conn, replace(seg, end_line=40)))
    assert (conn.flags, conn.links) == before


def test_corrupt_existing_membership_is_reported_without_repair_or_overwrite():
    w = make_window()
    seg, = build_segment_drafts([w])
    conn = Connection([w])
    run(save_segment_flag_on_connection(conn, seg))
    conn.links.clear()
    with pytest.raises(FlagPersistenceConflict, match="membership"):
        run(save_segment_flag_on_connection(conn, seg))
    assert conn.links == set()


def test_verified_evidence_is_preserved_and_retry_order_is_irrelevant():
    w = make_window()
    seg, = build_segment_drafts([w])
    conn = Connection([w])
    result = run(save_segment_flag_on_connection(conn, seg, verified_evidence_kinds=["paste", "typing_burst"]))
    assert run(save_segment_flag_on_connection(conn, seg, verified_evidence_kinds=["typing_burst", "paste"])) == result
    with pytest.raises(FlagPersistenceConflict, match="evidence_kinds"):
        run(save_segment_flag_on_connection(conn, seg))


def test_missing_prediction_fails_before_any_flag_is_saved():
    seg, = build_segment_drafts([make_window()])
    conn = Connection([])
    with pytest.raises(LookupError, match="already be stored"):
        run(save_segment_flag_on_connection(conn, seg))
    assert not conn.flags and not conn.links


@pytest.mark.parametrize("field,value", [
    ("session_id", uuid4()), ("task_id", uuid4()),
    ("feature_vector_id", uuid4()), ("ai_positive", False),
    ("predicted_category", "ide_ai"), ("model_version", "other-model"),
    ("run_id", "replay-1"), ("confidence", .3),
])
def test_stored_prediction_mismatch_is_rejected(field, value):
    w = make_window()
    seg, = build_segment_drafts([w])
    conn = Connection([w])
    conn.predictions[w.prediction_id][field] = value
    with pytest.raises(ValueError, match="provenance"):
        run(save_segment_flag_on_connection(conn, seg))
    assert not conn.flags and not conn.links


def test_stored_window_time_mismatch_is_rejected():
    w = make_window()
    seg, = build_segment_drafts([w])
    conn = Connection([w])
    conn.predictions[w.prediction_id]["window_end"] += timedelta(seconds=1)
    with pytest.raises(ValueError, match="provenance"):
        run(save_segment_flag_on_connection(conn, seg))


def test_stored_representative_probabilities_are_checked_exactly():
    w = make_window()
    seg, = build_segment_drafts([w])
    conn = Connection([w])
    conn.predictions[w.prediction_id]["class_probabilities"] = {"no_ai": .1, "ide_ai": .1, "external_ai": .8}
    with pytest.raises(ValueError, match="representative"):
        run(save_segment_flag_on_connection(conn, seg))


@pytest.mark.parametrize("error", [RuntimeError("link insert failed"), asyncio.CancelledError()])
def test_link_failure_or_cancellation_rolls_back_entire_save(error):
    a, b = make_window(), make_window(1)
    seg, = build_segment_drafts([a, b])
    conn = Connection([a, b])
    conn.fail_after_first_link = error
    with pytest.raises(type(error)):
        run(save_segment_flag_on_connection(conn, seg))
    assert not conn.flags and not conn.links
    assert conn.transactions == ["begin", "rollback"]
    conn.fail_after_first_link = None
    run(save_segment_flag_on_connection(conn, seg))
    assert len(conn.flags) == 1 and len(conn.links) == 2


def test_retry_after_lost_commit_acknowledgement_does_not_duplicate():
    w = make_window()
    seg, = build_segment_drafts([w])
    conn = Connection([w])
    conn.fail_after_commit = True
    with pytest.raises(OSError):
        run(save_segment_flag_on_connection(conn, seg))
    assert run(save_segment_flag_on_connection(conn, seg)) == flag_id_for_segment(seg)
    assert len(conn.flags) == len(conn.links) == 1


def test_pool_entry_point_releases_connection_on_success_and_conflict():
    w = make_window()
    seg, = build_segment_drafts([w])
    pool = Pool(Connection([w]))
    run(save_segment_flag(pool, seg))
    with pytest.raises(FlagPersistenceConflict):
        run(save_segment_flag(pool, replace(seg, end_line=30)))
    assert pool.acquired == pool.released == 2


def test_invalid_evidence_fails_before_acquiring_pool_connection():
    w = make_window()
    seg, = build_segment_drafts([w])
    pool = Pool(Connection([w]))
    with pytest.raises(ValueError):
        run(save_segment_flag(pool, seg, verified_evidence_kinds=["guessed"]))
    assert pool.acquired == pool.released == 0


def test_read_back_handles_default_json_text_and_missing_flag():
    w = make_window()
    seg, = build_segment_drafts([w])
    conn = Connection([w])
    flag_id = run(save_segment_flag_on_connection(conn, seg))
    assert isinstance(conn.flags[flag_id]["class_probabilities"], str)
    assert run(fetch_saved_flag(conn, flag_id))["class_probabilities"] == w.class_probabilities
    assert run(fetch_saved_flag(conn, uuid4())) is None


def test_read_back_and_retry_support_project_jsonb_dict_codec():
    w = make_window()
    seg, = build_segment_drafts([w])
    conn = Connection([w])
    flag_id = run(save_segment_flag_on_connection(conn, seg))
    conn.flags[flag_id]["class_probabilities"] = json.loads(conn.flags[flag_id]["class_probabilities"])
    assert run(save_segment_flag_on_connection(conn, seg)) == flag_id
    assert run(fetch_saved_flag(conn, flag_id))["class_probabilities"] == w.class_probabilities
