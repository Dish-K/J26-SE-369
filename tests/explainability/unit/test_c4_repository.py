"""PostgresC2Repository against a fake asyncpg pool.

No database is available to these unit tests. ``FakeAsyncpgPool`` returns rows
shaped like asyncpg records from migrations 002/005 — JSONB columns as text,
which is what the shared pool returns without a codec.
"""

import asyncio
from typing import Any
from uuid import UUID

import pytest

from backend.app.explainability.errors import (
    C2RecordInvalidError,
    FeatureVectorNotFoundError,
    FlagNotFoundError,
    PredictionNotFoundError,
)
from backend.app.explainability.persistence.repository import PostgresC2Repository
from tests.explainability.fixtures.prototype_c2 import C2Records, FakeAsyncpgPool


def _external(records: C2Records) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    flag = records.flags[records.flag_ids_by_name["external_ai_segment"]]
    prediction = records.predictions[flag["representative_prediction_id"]]
    vector = records.feature_vectors[prediction["feature_vector_id"]]
    return flag, prediction, vector


def _repository(records: C2Records) -> tuple[PostgresC2Repository, FakeAsyncpgPool]:
    pool = FakeAsyncpgPool(records)
    return PostgresC2Repository(pool), pool  # type: ignore[arg-type]


def test_get_flag_returns_typed_flag(records: C2Records) -> None:
    flag, _, _ = _external(records)
    repository, pool = _repository(records)
    result = asyncio.run(repository.get_flag(flag["flag_id"]))
    assert result.flag_id == flag["flag_id"]
    assert result.class_probabilities == flag["class_probabilities"]
    assert result.evidence_kinds == ("paste", "no_typing_errors")
    query, args = pool.calls[0]
    assert "FROM classifier_flags" in query and "representative_prediction_id" in query
    assert args == (flag["flag_id"],)


def test_get_prediction_returns_representative(records: C2Records) -> None:
    flag, prediction, _ = _external(records)
    repository, _ = _repository(records)
    result = asyncio.run(repository.get_prediction(flag["representative_prediction_id"]))
    assert result.prediction_id == prediction["prediction_id"]
    assert result.feature_vector_id == prediction["feature_vector_id"]


def test_get_feature_vector_returns_linked_vector(records: C2Records) -> None:
    _, prediction, vector = _external(records)
    repository, _ = _repository(records)
    result = asyncio.run(repository.get_feature_vector(prediction["feature_vector_id"]))
    assert result.feature_vector_id == vector["feature_vector_id"]
    assert result.features == vector["features"]


def test_contributing_predictions_preserve_provenance(records: C2Records) -> None:
    flag, _, _ = _external(records)
    repository, pool = _repository(records)
    members = asyncio.run(repository.get_contributing_predictions(flag["flag_id"]))
    assert [m.prediction_id for m in members] == records.members[flag["flag_id"]]
    assert flag["representative_prediction_id"] in {m.prediction_id for m in members}
    assert "classifier_flag_predictions" in pool.calls[0][0]


def test_task_window_predictions_include_clean_windows_in_time_order(
    records: C2Records,
) -> None:
    flag, _, _ = _external(records)
    repository, pool = _repository(records)
    rows = asyncio.run(
        repository.get_task_window_predictions(flag["task_id"], flag["model_version"])
    )
    assert pool.calls[0][1] == (flag["task_id"], flag["model_version"])
    assert [r.predicted_category for r in rows] == [
        "external_ai",
        "external_ai",
        "external_ai",
        "no_ai",
    ]


@pytest.mark.parametrize(
    ("method", "error"),
    [
        ("get_flag", FlagNotFoundError),
        ("get_prediction", PredictionNotFoundError),
        ("get_feature_vector", FeatureVectorNotFoundError),
    ],
)
def test_missing_rows_raise_explicit_errors(
    records: C2Records, method: str, error: type[Exception]
) -> None:
    repository, _ = _repository(records)
    with pytest.raises(error):
        asyncio.run(getattr(repository, method)(UUID(int=0)))


def test_contract_violating_row_raises(records: C2Records) -> None:
    flag, _, _ = _external(records)
    flag["class_probabilities"] = {"no_ai": 0.5, "ide_ai": 0.5, "external_ai": 0.5}
    repository, _ = _repository(records)
    with pytest.raises(C2RecordInvalidError):
        asyncio.run(repository.get_flag(flag["flag_id"]))
