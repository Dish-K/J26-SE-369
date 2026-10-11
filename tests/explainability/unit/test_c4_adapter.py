from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from backend.app.explainability.errors import (
    FeatureValueError,
    MissingFeatureError,
    ModelInferenceError,
)
from backend.app.explainability.model.adapter import ClassifierAdapter
from backend.app.explainability.model.artifact_loader import ArtifactLoader
from tests.explainability.fixtures.prototype_c2 import (
    FEATURE_NAMES,
    MODEL_VERSION,
    PROTOTYPE_THRESHOLD,
    C2Records,
    artifact_dict,
    write_artifact,
)


@pytest.fixture
def adapter(artifact_dir: Path) -> ClassifierAdapter:
    return ClassifierAdapter(ArtifactLoader(artifact_dir).load(MODEL_VERSION))


@pytest.fixture
def external_features(records: C2Records) -> dict[str, Any]:
    flag = records.flags[records.flag_ids_by_name["external_ai_segment"]]
    prediction = records.predictions[flag["representative_prediction_id"]]
    return records.feature_vectors[prediction["feature_vector_id"]]["features"]


def test_metadata_accessors(adapter: ClassifierAdapter) -> None:
    assert adapter.get_model_version() == MODEL_VERSION
    assert adapter.get_model_family() == "LogisticRegression"
    assert adapter.get_feature_names() == FEATURE_NAMES
    assert set(adapter.get_class_names()) == {"no_ai", "ide_ai", "external_ai"}
    assert adapter.get_threshold() == PROTOTYPE_THRESHOLD


# --- feature ordering (mandatory) ------------------------------------------


def test_shuffled_feature_dict_is_prepared_in_artifact_order(
    adapter: ClassifierAdapter, external_features: dict[str, Any]
) -> None:
    shuffled = dict(reversed(list(external_features.items())))
    assert list(shuffled) != list(FEATURE_NAMES)

    prepared = adapter.prepare_features(shuffled)

    assert tuple(prepared.columns) == FEATURE_NAMES
    assert prepared.iloc[0].tolist() == [float(external_features[n]) for n in FEATURE_NAMES]


def test_insertion_order_does_not_change_probabilities(
    adapter: ClassifierAdapter, external_features: dict[str, Any]
) -> None:
    ordered = {name: external_features[name] for name in FEATURE_NAMES}
    shuffled = dict(reversed(list(ordered.items())))
    assert adapter.predict_proba(ordered) == adapter.predict_proba(shuffled)


def test_ndarray_fitted_model_receives_artifact_order(tmp_path: Path) -> None:
    """A model fitted without column names still gets columns by name, not dict order."""
    names = ["a", "b", "c"]
    rng = np.random.default_rng(0)
    x = rng.normal(size=(90, 3))
    y = np.array(["no_ai", "ide_ai", "external_ai"])[np.argmax(x, axis=1)]
    model = LogisticRegression().fit(x, y)  # ndarray: no feature_names_in_
    write_artifact(tmp_path, artifact_dict(model, feature_names=names))
    adapter = ClassifierAdapter(ArtifactLoader(tmp_path).load(MODEL_VERSION))

    features = {"c": 3.0, "a": 1.0, "b": 2.0}
    prepared = adapter.prepare_features(features)
    assert prepared.iloc[0].tolist() == [1.0, 2.0, 3.0]

    expected = model.predict_proba(np.array([[1.0, 2.0, 3.0]]))[0]
    by_name = adapter.predict_proba(features)
    for column, label in enumerate(model.classes_):
        assert by_name[str(label)] == pytest.approx(expected[column], abs=1e-15)


# --- inference ---------------------------------------------------------------


def test_predict_proba_maps_columns_to_the_right_classes(
    adapter: ClassifierAdapter, external_features: dict[str, Any], prototype_model: Any
) -> None:
    raw = prototype_model.predict_proba(adapter.prepare_features(external_features))[0]
    probabilities = adapter.predict_proba(external_features)
    for column, label in enumerate(prototype_model.classes_):
        assert probabilities[str(label)] == raw[column]
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-12)


def test_predict_is_the_highest_probability_class(
    adapter: ClassifierAdapter, external_features: dict[str, Any]
) -> None:
    probabilities = adapter.predict_proba(external_features)
    assert adapter.predict(external_features) == max(probabilities, key=probabilities.get)
    assert adapter.predict(external_features) == "external_ai"


def test_preprocessing_pipeline_is_used(
    adapter: ClassifierAdapter, external_features: dict[str, Any], prototype_model: Any
) -> None:
    """Skipping the pipeline's scaler would give different probabilities."""
    prepared = adapter.prepare_features(external_features)
    unscaled = prototype_model.named_steps["classify"].predict_proba(prepared.to_numpy())[0]
    with_pipeline = adapter.predict_proba_batch(prepared)[0]
    assert not np.allclose(unscaled, with_pipeline)


def test_batch_rejects_wrong_column_order(
    adapter: ClassifierAdapter, external_features: dict[str, Any]
) -> None:
    prepared = adapter.prepare_features(external_features)
    with pytest.raises(FeatureValueError):
        adapter.predict_proba_batch(prepared[list(reversed(FEATURE_NAMES))])


def test_missing_required_feature_fails(
    adapter: ClassifierAdapter, external_features: dict[str, Any]
) -> None:
    features = {k: v for k, v in external_features.items() if k != "paste_ratio"}
    with pytest.raises(MissingFeatureError, match="paste_ratio"):
        adapter.prepare_features(features)


def test_extra_stored_features_are_ignored(
    adapter: ClassifierAdapter, external_features: dict[str, Any]
) -> None:
    assert "min_line" in external_features  # stored, but not a model feature
    assert "min_line" not in adapter.prepare_features(external_features).columns


def test_non_numeric_feature_fails(
    adapter: ClassifierAdapter, external_features: dict[str, Any]
) -> None:
    with pytest.raises(FeatureValueError, match="pause_ms"):
        adapter.prepare_features({**external_features, "pause_ms": "long"})


def test_null_feature_is_not_imputed(
    adapter: ClassifierAdapter, external_features: dict[str, Any]
) -> None:
    prepared = adapter.prepare_features({**external_features, "pause_ms": None})
    assert pd.isna(prepared.loc[0, "pause_ms"])
    # The prototype pipeline has no imputer, so inference fails explicitly.
    with pytest.raises(ModelInferenceError):
        adapter.predict_proba({**external_features, "pause_ms": None})
