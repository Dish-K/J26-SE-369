from datetime import datetime, timedelta, timezone
from uuid import uuid4

from backend.app.classifier.localization import LocalizedCodeRegion
from backend.app.classifier.segment_builder import ClassifiedWindow


BASE = datetime(2026, 10, 18, 9, 0, tzinfo=timezone.utc)
SESSION, TASK = uuid4(), uuid4()


def make_window(
    n=0, *, category="external_ai", confidence=.8,
    ranges=((0, 2, 10, 15),), session_id=SESSION, task_id=TASK,
    model_version="task12-storage-fixture-v1", base=BASE,
):
    prediction_id, feature_id = uuid4(), uuid4()
    start = base + timedelta(seconds=10 * n)
    other = "ide_ai" if category == "external_ai" else "external_ai"
    probs = {"no_ai": .05, other: .95 - confidence, category: confidence}
    positive = category != "no_ai"
    if not positive:
        probs = {"no_ai": confidence, "ide_ai": (1-confidence)/2,
                 "external_ai": (1-confidence)/2}
        ranges = ()
    regions = tuple(
        LocalizedCodeRegion(
            prediction_id=prediction_id, feature_vector_id=feature_id,
            session_id=session_id, task_id=task_id, model_version=model_version,
            predicted_category=category, confidence=confidence,
            window_start=start, window_end=start + timedelta(seconds=30),
            load_seq=load, version_id=version, start_line=lo, end_line=hi,
            edit_events=({
                "session_id": session_id, "task_id": task_id,
                "load_seq": load, "version_id": version,
                "batch_seq": n, "seq_in_batch": index, "change_index": 0,
                "start_line": lo, "end_line": lo,
                "inserted_lines": hi-lo, "origin": 0,
            },),
        ) for index, (load, version, lo, hi) in enumerate(ranges)
    )
    return ClassifiedWindow(
        prediction_id=prediction_id, feature_vector_id=feature_id,
        session_id=session_id, task_id=task_id, model_version=model_version,
        predicted_category=category, class_probabilities=probs,
        confidence=confidence, ai_positive=positive,
        window_start=start, window_end=start + timedelta(seconds=30), regions=regions,
    )


def stored_prediction(window):
    return {
        "prediction_id": window.prediction_id,
        "feature_vector_id": window.feature_vector_id,
        "session_id": window.session_id, "task_id": window.task_id,
        "model_version": window.model_version,
        "predicted_category": window.predicted_category,
        "class_probabilities": dict(window.class_probabilities),
        "confidence": window.confidence, "ai_positive": window.ai_positive,
        "window_start": window.window_start, "window_end": window.window_end,
        "run_id": "live",
    }
