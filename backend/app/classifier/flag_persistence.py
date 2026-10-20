from __future__ import annotations

import json
from typing import Any, Sequence
from uuid import UUID

import asyncpg

from .flag_payload import FlagInsertValues, map_segment_to_flag
from .segment_builder import SegmentDraft


class FlagPersistenceConflict(ValueError):
    """A stable flag ID already exists with different fields or membership."""


_FLAG_COLUMNS = (
    "flag_id", "task_id", "model_version", "category",
    "representative_prediction_id", "class_probabilities", "confidence",
    "detected_start_line", "detected_end_line", "load_seq", "version_id",
    "evidence_kinds", "segment_basis", "window_count", "detected_at",
    "detected_until",
)
_INSERT_FLAG_SQL = """
INSERT INTO classifier_flags (
    flag_id, task_id, model_version, category, representative_prediction_id,
    class_probabilities, confidence, detected_start_line, detected_end_line,
    load_seq, version_id, evidence_kinds, segment_basis, window_count,
    detected_at, detected_until
) VALUES ($1,$2,$3,$4,$5,$6::jsonb,$7,$8,$9,$10,$11,$12::text[],$13,$14,$15,$16)
ON CONFLICT (flag_id) DO NOTHING
RETURNING flag_id
"""
_MEMBERS_SQL = """
SELECT p.prediction_id, p.feature_vector_id, p.task_id, p.model_version,
       p.predicted_category, p.class_probabilities, p.confidence, p.ai_positive,
       fv.session_id, fv.window_start, fv.window_end, fv.run_id
FROM classifier_window_predictions AS p
JOIN feature_vectors AS fv
  ON fv.feature_vector_id = p.feature_vector_id AND fv.task_id = p.task_id
WHERE p.prediction_id = ANY($1::uuid[])
ORDER BY p.prediction_id
FOR SHARE OF p, fv
"""
_INSERT_LINK_SQL = """
INSERT INTO classifier_flag_predictions (flag_id, prediction_id)
VALUES ($1, $2)
"""


def _probabilities(value: Any) -> dict[str, float]:
    # Default asyncpg JSONB decoding is text; a project codec may return a dict.
    return json.loads(value) if isinstance(value, str) else dict(value)


def _arguments(values: FlagInsertValues) -> tuple[Any, ...]:
    return tuple(
        json.dumps(values.class_probabilities) if name == "class_probabilities"
        else list(values.evidence_kinds) if name == "evidence_kinds"
        else getattr(values, name)
        for name in _FLAG_COLUMNS
    )


async def fetch_saved_flag(
    connection: asyncpg.Connection, flag_id: UUID,
) -> dict[str, Any] | None:
    """Read one historical flag and its prediction IDs in one SQL snapshot.

    Prediction IDs are sorted for deterministic read-back, not window order.
    Final submitted-code coordinates remain in the existing separate table.
    """
    row = await connection.fetchrow(
        """
        SELECT f.*, ARRAY(
            SELECT fp.prediction_id FROM classifier_flag_predictions AS fp
            WHERE fp.flag_id = f.flag_id ORDER BY fp.prediction_id
        ) AS prediction_ids
        FROM classifier_flags AS f WHERE f.flag_id = $1
        """, flag_id,
    )
    if row is None:
        return None
    result = dict(row)
    result["class_probabilities"] = _probabilities(result["class_probabilities"])
    result["evidence_kinds"] = tuple(result["evidence_kinds"])
    result["prediction_ids"] = tuple(result["prediction_ids"])
    return result


async def _validate_stored_members(
    connection: asyncpg.Connection, segment: SegmentDraft, values: FlagInsertValues,
) -> None:
    rows = await connection.fetch(_MEMBERS_SQL, list(values.prediction_ids))
    indexed = {row["prediction_id"]: row for row in rows}
    if set(indexed) != set(values.prediction_ids):
        raise LookupError("All contributing predictions must already be stored")

    for member in segment.members:
        row = indexed[member.prediction_id]
        if (
            row["feature_vector_id"] != member.feature_vector_id
            or row["task_id"] != values.task_id
            or row["session_id"] != segment.session_id
            or row["model_version"] != values.model_version
            or row["predicted_category"] != values.category
            or not row["ai_positive"]
            or row["run_id"] != "live"
            or row["window_start"] != member.window_start
            or row["window_end"] != member.window_end
            or abs(row["confidence"] - member.confidence) > 1e-6
        ):
            raise ValueError("Stored prediction does not match segment provenance")

    representative = indexed[values.representative_prediction_id]
    if (
        _probabilities(representative["class_probabilities"])
        != values.class_probabilities
        or representative["confidence"] != values.confidence
    ):
        raise ValueError("Flag must preserve the stored representative prediction")


async def _save(
    connection: asyncpg.Connection, segment: SegmentDraft, values: FlagInsertValues,
) -> UUID:
    # A nested asyncpg transaction becomes a savepoint for test/caller transactions.
    async with connection.transaction():
        await _validate_stored_members(connection, segment, values)
        inserted = await connection.fetchval(_INSERT_FLAG_SQL, *_arguments(values))
        if inserted is None:
            # Serializes same-ID retries without overwriting the historical row.
            existing = await connection.fetchrow(
                "SELECT * FROM classifier_flags WHERE flag_id = $1 FOR UPDATE",
                values.flag_id,
            )
            if existing is None:
                raise FlagPersistenceConflict("Flag disappeared during retry; retry the transaction")
            for name in _FLAG_COLUMNS:
                actual = existing[name]
                expected = getattr(values, name)
                if name == "class_probabilities":
                    actual = _probabilities(actual)
                elif name == "evidence_kinds":
                    actual, expected = set(actual), set(expected)
                if actual != expected:
                    raise FlagPersistenceConflict(
                        f"Existing flag {values.flag_id} differs in {name}"
                    )
            links = await connection.fetch(
                "SELECT prediction_id FROM classifier_flag_predictions WHERE flag_id = $1",
                values.flag_id,
            )
            if {row["prediction_id"] for row in links} != set(values.prediction_ids):
                raise FlagPersistenceConflict("Existing flag has different prediction membership")
        else:
            await connection.executemany(
                _INSERT_LINK_SQL,
                [(values.flag_id, prediction_id) for prediction_id in values.prediction_ids],
            )
    return values.flag_id


async def save_segment_flag_on_connection(
    connection: asyncpg.Connection,
    segment: SegmentDraft,
    *,
    verified_evidence_kinds: Sequence[str] = (),
) -> UUID:
    """Store one supplied segment and every prediction link atomically.

    Predictions must have been persisted first. Identical retries return the
    existing UUID; a changed draft or different evidence under that UUID raises
    FlagPersistenceConflict. Database/connection errors propagate to the caller.
    After an uncertain commit, retry the same segment and evidence values.
    """
    values = map_segment_to_flag(segment, verified_evidence_kinds=verified_evidence_kinds)
    return await _save(connection, segment, values)


async def save_segment_flag(
    pool: asyncpg.Pool,
    segment: SegmentDraft,
    *,
    verified_evidence_kinds: Sequence[str] = (),
) -> UUID:
    """Pool-based entry point using the project's existing asyncpg pool."""
    values = map_segment_to_flag(segment, verified_evidence_kinds=verified_evidence_kinds)
    async with pool.acquire() as connection:
        return await _save(connection, segment, values)
