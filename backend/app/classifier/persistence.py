from __future__ import annotations

import json
from uuid import UUID

import asyncpg

from .inference import WindowPrediction

async def save_window_prediction(
        pool: asyncpg.Pool,
        prediction: WindowPrediction
) -> UUID:
    """
    Persist one Classifier window prediction.

    One feature vector may have one prediction per model version.
    The database uniqueness constraint protects against duplicates.

    Returns:
        UUID of the created classifier_window_predictions row.
    """

    query = """
        INSERT INTO classifier_window_predictions (
            feature_vector_id,
            task_id,
            model_version,
            predicted_category,
            class_probabilities,
            confidence,
            ai_positive
        )
        VALUES (
            $1,
            $2,
            $3,
            $4,
            $5::jsonb,
            $6,
            $7
        )
        RETURNING prediction_id
    """

    probabilities_json = json.dumps(
        prediction.class_probabilities
    )

    async with pool.acquire() as connection:
        prediction_id = await connection.fetchval(
            query,
            prediction.feature_vector_id,
            prediction.task_id,
            prediction.model_version,
            prediction.predicted_category,
            probabilities_json,
            prediction.confidence,
            prediction.ai_positive,
        )

    if prediction_id is None:
        raise RuntimeError(
            "Database did not return prediction_id"
            "after saving window prediction"
        )

    return prediction_id