"""Model-family-neutral inference over a loaded C2 artifact.

Everything in C4 that runs the classifier goes through this adapter, so input
ordering, preprocessing and class mapping are decided in exactly one place:

* input columns follow ``artifact.feature_names``, never dict insertion order;
* extra keys in the stored Contract-A payload are ignored, missing ones fail;
* the artifact's fitted object (typically an sklearn Pipeline) is called as a
  whole, so C2's preprocessing is reused rather than recreated;
* probability columns are named by the model's own column order
  (``LoadedArtifact.output_classes``), resolved and checked by the loader.
"""

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from backend.app.explainability.errors import (
    FeatureValueError,
    MissingFeatureError,
    ModelInferenceError,
)
from backend.app.explainability.model.artifact_loader import LoadedArtifact


class ClassifierAdapter:
    def __init__(self, artifact: LoadedArtifact) -> None:
        self._artifact = artifact
        self._model = artifact.model
        self._feature_names = artifact.metadata.feature_names
        self._class_names = artifact.output_classes
        # Pass a named DataFrame only when the model was fitted on one;
        # otherwise sklearn warns and the names would carry no guarantee.
        self._fitted_with_names = getattr(self._model, "feature_names_in_", None) is not None

    def get_model_version(self) -> str:
        return self._artifact.metadata.model_version

    def get_model_family(self) -> str:
        return self._artifact.model_family

    def get_feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    def get_class_names(self) -> tuple[str, ...]:
        """Class of each ``predict_proba_batch`` column, in column order."""
        return self._class_names

    def get_threshold(self) -> float:
        return self._artifact.metadata.classification_threshold

    def get_fitted_model(self) -> Any:
        """The artifact's fitted object, for explainers that must inspect it.

        Read-only: callers must not refit or mutate it.
        """
        return self._model

    def model_input(self, rows: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Rows in artifact feature order, shaped the way the fitted model expects."""
        return self._as_model_input(rows)

    def prepare_features(self, features: Mapping[str, Any]) -> pd.DataFrame:
        """One-row frame of the model features, in artifact order.

        ``None`` (JSON null) becomes NaN. Whether NaN is acceptable is the
        model pipeline's decision under the Contract-A missing-value policy;
        C4 does not impute.
        """
        missing = [name for name in self._feature_names if name not in features]
        if missing:
            raise MissingFeatureError(f"feature vector lacks required model features: {missing}")

        row = [self._coerce(name, features[name]) for name in self._feature_names]
        return pd.DataFrame([row], columns=list(self._feature_names), dtype=float)

    def predict_proba(self, features: Mapping[str, Any]) -> dict[str, float]:
        """Class probabilities for one feature vector, keyed by class name."""
        probabilities = self.predict_proba_batch(self.prepare_features(features))[0]
        return {name: float(p) for name, p in zip(self._class_names, probabilities, strict=True)}

    def predict(self, features: Mapping[str, Any]) -> str:
        """Predicted category: the highest-probability class (C2 README §9.1)."""
        probabilities = self.predict_proba(features)
        return max(probabilities, key=probabilities.__getitem__)

    def predict_proba_batch(self, rows: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Probabilities for many rows already in artifact feature order.

        Columns of the result follow ``get_class_names()``. Intended for later
        phases (SHAP/LIME/DiCE) that perturb rows in bulk.
        """
        matrix = self._as_model_input(rows)
        try:
            output = np.asarray(self._model.predict_proba(matrix), dtype=float)
        except (ValueError, TypeError) as exc:
            raise ModelInferenceError(f"model {self.get_model_version()!r} failed: {exc}") from exc

        expected = (matrix.shape[0], len(self._class_names))
        if output.shape != expected:
            raise ModelInferenceError(
                f"predict_proba returned shape {output.shape}, expected {expected}"
            )
        return output

    def _as_model_input(self, rows: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        if isinstance(rows, pd.DataFrame):
            if tuple(rows.columns) != self._feature_names:
                raise FeatureValueError(
                    "input columns are not in artifact feature order: "
                    f"{list(rows.columns)} vs {list(self._feature_names)}"
                )
            return rows if self._fitted_with_names else rows.to_numpy(dtype=float)

        matrix = np.asarray(rows, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self._feature_names):
            raise FeatureValueError(
                f"expected a 2-D array with {len(self._feature_names)} columns, got {matrix.shape}"
            )
        if self._fitted_with_names:
            return pd.DataFrame(matrix, columns=list(self._feature_names))
        return matrix

    @staticmethod
    def _coerce(name: str, value: Any) -> float:
        if value is None:
            return math.nan
        # bool is a Real subclass; Contract-A flags such as copy_before_blur
        # are treated as 0/1 counts rather than rejected.
        if isinstance(value, Real):
            return float(value)
        raise FeatureValueError(
            f"feature {name!r} must be numeric or null, got {type(value).__name__}: {value!r}"
        )
