
import asyncio
import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import asyncpg

from backend.app.classifier.artifact import load_artifact
from backend.app.classifier.inference import predict_window
from backend.app.classifier.persistence import save_window_prediction
from backend.app.db.pool import create_db_pool, close_db_pool


ARTIFACT_PATH = "ml/models/dummy-contract-v1/classifier.joblib"


async def main():
    # 1. Load the integration-only model.
    artifact = load_artifact(ARTIFACT_PATH)

    assert artifact.model_version == "dummy-contract-v1"
    assert artifact.schema_version == "fixture-schema-v0"

    # 2. Open the existing project's database pool.
    pool = await create_db_pool()

    try:
        session_id = uuid4()
        task_id = uuid4()
        candidate_id = uuid4()

        now = datetime.now(timezone.utc)
        task_start = now - timedelta(seconds=40)
        window_start = now - timedelta(seconds=30)

        # These are test-only features, NOT research data.
        features = {
            "fixture_feature_1": 0.5,
            "fixture_feature_2": 0.5,
        }

        # 3. Create real database rows with valid foreign keys.
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO sessions
                        (session_id, candidate_id, language, started_at)
                    VALUES ($1, $2, 'python', $3)
                    """,
                    session_id,
                    candidate_id,
                    task_start,
                )

                await conn.execute(
                    """
                    INSERT INTO tasks
                        (task_id, session_id, seq, depth,
                         origin, created_by, status,
                         created_at, started_at)
                    VALUES
                        ($1, $2, 0, 0,
                         'original', 'interviewer', 'active',
                         $3, $4)
                    """,
                    task_id,
                    session_id,
                    task_start,
                    task_start,
                )

                feature_vector_id = await conn.fetchval(
                    """
                    INSERT INTO feature_vectors
                        (session_id, task_id,
                         window_start, window_end,
                         run_id, schema_version, features)
                    VALUES
                        ($1, $2, $3, $4, 'live', $5, $6::jsonb)
                    RETURNING feature_vector_id
                    """,
                    session_id,
                    task_id,
                    window_start,
                    now,
                    artifact.schema_version,
                    json.dumps(features),
                )

        
        # 4. Retrieve the feature vector directly from PostgreSQL.
        async with pool.acquire() as conn:
            stored_vector = await conn.fetchrow(
                """
                SELECT feature_vector_id, task_id,
                    schema_version, features
                FROM feature_vectors
                WHERE feature_vector_id = $1
                """,
                feature_vector_id,
            )

        assert stored_vector is not None
        assert stored_vector["task_id"] == task_id
        assert stored_vector["schema_version"] == artifact.schema_version

        stored_features = json.loads(stored_vector["features"])

        assert stored_features == features

        print("Feature vector successfully loaded from PostgreSQL")

        # 5. Run inference using the database-retrieved features.
        prediction = predict_window(
            artifact=artifact,
            feature_vector_id=stored_vector["feature_vector_id"],
            task_id=stored_vector["task_id"],
            features=stored_features,
        )


        # 6. Save through your real persistence implementation.
        prediction_id = await save_window_prediction(
            pool, prediction
        )

        # 7. Read back the stored result and verify its contents.
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    prediction_id,
                    feature_vector_id,
                    task_id,
                    model_version,
                    predicted_category,
                    class_probabilities,
                    confidence,
                    ai_positive
                FROM classifier_window_predictions
                WHERE prediction_id = $1
                """,
                prediction_id,
            )

            count = await conn.fetchval(
                """
                SELECT count(*)
                FROM classifier_window_predictions
                WHERE feature_vector_id = $1
                  AND model_version = $2
                """,
                feature_vector_id,
                artifact.model_version,
            )

        assert row is not None
        assert count == 1
        assert row["feature_vector_id"] == feature_vector_id
        assert row["task_id"] == task_id
        assert row["model_version"] == prediction.model_version
        assert row["predicted_category"] == prediction.predicted_category
        assert row["ai_positive"] == prediction.ai_positive
        assert abs(row["confidence"] - prediction.confidence) < 1e-9
        assert json.loads(row["class_probabilities"]) == prediction.class_probabilities

        print("Session ID:", session_id)
        print("Feature vector ID:", feature_vector_id)
        print("Prediction ID:", prediction_id)
        print("Predicted category:", row["predicted_category"])
        print("Confidence:", row["confidence"])
        print("AI positive:", row["ai_positive"])
        print("Matching database rows:", count)

        # 8. Verify that the database rejects a duplicate prediction.
        try:
            await save_window_prediction(pool, prediction)
        except asyncpg.UniqueViolationError:
            print("Duplicate correctly rejected")
        else:
            raise AssertionError("Duplicate prediction was accepted")

        print("DATABASE PERSISTENCE TEST PASSED")

    finally:
        await close_db_pool(pool)


if __name__ == "__main__":
    asyncio.run(main())