"""Read-only access to C2 output for C4.

Table and column names follow migrations 002 (``feature_vectors``) and 005
(``classifier_window_predictions``, ``classifier_flags``,
``classifier_flag_predictions``). C4 never writes to these tables.

The reader returns typed schemas. A row that violates the C2 contract raises
``C2RecordInvalidError``; a missing row raises the matching ``*NotFoundError``.
"""

import json
from collections.abc import Mapping
from typing import Any, Protocol, TypeVar
from uuid import UUID

import asyncpg
from pydantic import BaseModel, ValidationError

from backend.app.explainability.errors import (
    C2RecordInvalidError,
    FeatureVectorNotFoundError,
    FlagNotFoundError,
    PredictionNotFoundError,
)
from backend.app.explainability.schemas import ClassifierFlag, FeatureVector, WindowPrediction

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class C2OutputReader(Protocol):
    """The C2 data C4 depends on. Implemented by Postgres and by test fixtures."""

    async def get_flag(self, flag_id: UUID) -> ClassifierFlag: ...

    async def get_prediction(self, prediction_id: UUID) -> WindowPrediction: ...

    async def get_feature_vector(self, feature_vector_id: UUID) -> FeatureVector: ...

    async def get_contributing_predictions(self, flag_id: UUID) -> list[WindowPrediction]: ...

    async def get_task_window_predictions(
        self, task_id: UUID, model_version: str
    ) -> list[WindowPrediction]: ...


_FLAG_SQL = """
SELECT flag_id, task_id, model_version, category, representative_prediction_id,
       class_probabilities, confidence, detected_start_line, detected_end_line,
       evidence_kinds, segment_basis, window_count, detected_at, detected_until,
       created_at
  FROM classifier_flags
 WHERE flag_id = $1
"""

_PREDICTION_COLUMNS = """
       p.prediction_id, p.feature_vector_id, p.task_id, p.model_version,
       p.predicted_category, p.class_probabilities, p.confidence, p.ai_positive,
       p.predicted_at
"""

_PREDICTION_SQL = f"""
SELECT {_PREDICTION_COLUMNS}
  FROM classifier_window_predictions AS p
 WHERE p.prediction_id = $1
"""

_CONTRIBUTING_SQL = f"""
SELECT {_PREDICTION_COLUMNS}
  FROM classifier_flag_predictions AS m
  JOIN classifier_window_predictions AS p ON p.prediction_id = m.prediction_id
  JOIN feature_vectors AS v ON v.feature_vector_id = p.feature_vector_id
 WHERE m.flag_id = $1
 ORDER BY v.window_start, v.window_end, p.prediction_id
"""

_TASK_PREDICTIONS_SQL = f"""
SELECT {_PREDICTION_COLUMNS}
  FROM classifier_window_predictions AS p
  JOIN feature_vectors AS v ON v.feature_vector_id = p.feature_vector_id
 WHERE p.task_id = $1
   AND p.model_version = $2
 ORDER BY v.window_start, v.window_end, p.prediction_id
"""

_FEATURE_VECTOR_SQL = """
SELECT feature_vector_id, session_id, task_id, window_start, window_end,
       run_id, schema_version, features
  FROM feature_vectors
 WHERE feature_vector_id = $1
"""

# asyncpg returns json/jsonb as text unless a codec is registered on the pool.
# The shared pool registers none, so decode here.
_JSON_COLUMNS = ("class_probabilities", "features")


def _decode(row: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(row)
    for column in _JSON_COLUMNS:
        if isinstance(data.get(column), str):
            data[column] = json.loads(data[column])
    return data


def parse_c2_row(schema: type[SchemaT], row: Mapping[str, Any], what: str) -> SchemaT:
    """Parse one C2 row into its schema, or raise ``C2RecordInvalidError``."""
    try:
        return schema.model_validate(_decode(row))
    except (ValidationError, json.JSONDecodeError) as exc:
        raise C2RecordInvalidError(f"{what} violates the C2 output contract: {exc}") from exc


class PostgresC2Repository:
    """``C2OutputReader`` over the shared asyncpg pool (``backend.app.db.pool``)."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get_flag(self, flag_id: UUID) -> ClassifierFlag:
        row = await self._pool.fetchrow(_FLAG_SQL, flag_id)
        if row is None:
            raise FlagNotFoundError(f"classifier flag {flag_id} does not exist")
        return parse_c2_row(ClassifierFlag, row, f"classifier flag {flag_id}")

    async def get_prediction(self, prediction_id: UUID) -> WindowPrediction:
        row = await self._pool.fetchrow(_PREDICTION_SQL, prediction_id)
        if row is None:
            raise PredictionNotFoundError(f"window prediction {prediction_id} does not exist")
        return parse_c2_row(WindowPrediction, row, f"window prediction {prediction_id}")

    async def get_feature_vector(self, feature_vector_id: UUID) -> FeatureVector:
        row = await self._pool.fetchrow(_FEATURE_VECTOR_SQL, feature_vector_id)
        if row is None:
            raise FeatureVectorNotFoundError(f"feature vector {feature_vector_id} does not exist")
        return parse_c2_row(FeatureVector, row, f"feature vector {feature_vector_id}")

    async def get_contributing_predictions(self, flag_id: UUID) -> list[WindowPrediction]:
        rows = await self._pool.fetch(_CONTRIBUTING_SQL, flag_id)
        return [
            parse_c2_row(WindowPrediction, row, f"contributing prediction of flag {flag_id}")
            for row in rows
        ]

    async def get_task_window_predictions(
        self, task_id: UUID, model_version: str
    ) -> list[WindowPrediction]:
        rows = await self._pool.fetch(_TASK_PREDICTIONS_SQL, task_id, model_version)
        return [
            parse_c2_row(WindowPrediction, row, f"window prediction of task {task_id}")
            for row in rows
        ]
