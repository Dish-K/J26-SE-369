"""SHAP strategy selection. Selection itself needs no SHAP; building does."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from backend.app.explainability.errors import ExplanationBackgroundMismatchError
from backend.app.explainability.explanations.background import XaiBackground, prepare_background
from backend.app.explainability.model.adapter import ClassifierAdapter
from backend.app.explainability.model.artifact_loader import ArtifactLoader, LoadedArtifact
from backend.app.explainability.model.explainer_factory import ExplainerFactory
from backend.app.explainability.schemas import ModelArtifactMetadata
from tests.explainability.fixtures.prototype_c2 import (
    CLASS_ORDER,
    FEATURE_NAMES,
    MODEL_VERSION,
    _training_frame,
    artifact_dict,
    background_frame,
    background_reference,
    write_artifact,
)


def _adapter_for(model: Any, tmp_path: Path) -> ClassifierAdapter:
    write_artifact(tmp_path, artifact_dict(model, model_family=None))
    return ClassifierAdapter(ArtifactLoader(tmp_path).load(MODEL_VERSION))


def _fit(estimator: Any) -> Any:
    frame, labels = _training_frame()
    return estimator.fit(frame, labels)


def test_logistic_regression_pipeline_uses_linear(adapter: ClassifierAdapter) -> None:
    kind, reason = ExplainerFactory().select_kind(adapter)
    assert kind == "linear"
    assert "LogisticRegression" in reason


def test_random_forest_uses_tree(tmp_path: Path) -> None:
    model = _fit(RandomForestClassifier(n_estimators=10, max_depth=3, random_state=0))
    kind, _ = ExplainerFactory().select_kind(_adapter_for(model, tmp_path))
    assert kind == "tree"


def test_scaled_random_forest_uses_tree(tmp_path: Path) -> None:
    model = _fit(
        Pipeline(
            [("scale", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=5, random_state=0))]
        )
    )
    kind, _ = ExplainerFactory().select_kind(_adapter_for(model, tmp_path))
    assert kind == "tree"


def test_unknown_classifier_falls_back_to_model_agnostic(tmp_path: Path) -> None:
    kind, reason = ExplainerFactory().select_kind(_adapter_for(_fit(GaussianNB()), tmp_path))
    assert kind == "model_agnostic"
    assert "GaussianNB" in reason


def test_non_elementwise_preprocessing_forces_model_agnostic(tmp_path: Path) -> None:
    """A linear model behind a non-affine transform must not get LinearExplainer."""
    model = _fit(
        Pipeline(
            [
                ("quantile", QuantileTransformer(n_quantiles=50)),
                ("classify", LogisticRegression(max_iter=2000)),
            ]
        )
    )
    kind, reason = ExplainerFactory().select_kind(_adapter_for(model, tmp_path))
    assert kind == "model_agnostic"
    assert "QuantileTransformer" in reason


class _FakeXGBClassifier:
    """Stands in for xgboost.XGBClassifier (xgboost is not a dependency)."""

    __module__ = "xgboost.sklearn"

    def __init__(self) -> None:
        self.classes_ = np.array([0, 1, 2])

    def predict_proba(self, rows: Any) -> np.ndarray:
        return np.full((len(rows), 3), 1 / 3)


def test_xgboost_is_detected_from_the_object_and_routed_explicitly() -> None:
    metadata = ModelArtifactMetadata(
        model_version=MODEL_VERSION,
        model_family="LogisticRegression",  # free-text label must not drive selection
        feature_schema_version="v",
        feature_names=FEATURE_NAMES,
        class_order=CLASS_ORDER,
        classification_threshold=0.7,
    )
    artifact = LoadedArtifact(
        metadata=metadata,
        model=_FakeXGBClassifier(),
        model_family="XGBClassifier",
        path=Path("unused"),
        output_classes=CLASS_ORDER,
    )
    kind, reason = ExplainerFactory().select_kind(ClassifierAdapter(artifact))
    assert kind == "model_agnostic"
    assert "XGBoost" in reason


def test_background_with_other_columns_is_rejected(
    adapter: ClassifierAdapter, artifact_dir: Path
) -> None:
    metadata = ArtifactLoader(artifact_dir).load(MODEL_VERSION).metadata
    background = prepare_background(background_reference(), background_frame(), metadata)
    narrowed = XaiBackground(reference=background.reference, frame=background.frame.iloc[:, :-1])
    with pytest.raises(ExplanationBackgroundMismatchError):
        ExplainerFactory().create(adapter, narrowed)
