"""Choose and build the SHAP strategy for a loaded C2 model.

Selection inspects the fitted object itself (final estimator type and the
preprocessing steps in front of it), never a free-text artifact field.

Strategies — every one attributes to the ORIGINAL Contract-A features:

* ``linear`` — sklearn linear classifier whose preprocessing is purely
  per-feature affine scaling. ``shap.LinearExplainer`` on the scaled inputs.
  A per-feature affine map leaves each feature's Shapley value unchanged, so
  attributions stay one-to-one with the raw features. Output space ``raw``:
  the per-class ``decision_function`` (softmax logit for multinomial models),
  NOT probability.
* ``tree`` — sklearn decision tree / random forest / extra trees behind the
  same kind of scaling. Interventional ``shap.TreeExplainer``; these models'
  raw output is the averaged leaf class distribution, i.e. ``predict_proba``,
  so the output space is ``probability``.
* ``model_agnostic`` — everything else, including any non-elementwise
  preprocessing and model families whose tree path is not yet verified here
  (XGBoost, gradient boosting). Explains the WHOLE artifact pipeline through
  ``ClassifierAdapter.predict_proba_batch`` with an independent masker over
  the C4 background: exact enumeration up to ``max_exact_features``
  features, a seeded permutation explainer above. Output space
  ``probability``.

``shap`` is imported lazily so the rest of C4 does not require it.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.app.explainability.config import ShapConfig
from backend.app.explainability.errors import (
    ExplanationBackgroundMismatchError,
    UnsupportedModelForShapError,
)
from backend.app.explainability.explanations.background import XaiBackground
from backend.app.explainability.model.adapter import ClassifierAdapter

# Preprocessing steps known to act on each feature independently with an
# affine map, which preserves per-feature Shapley attributions.
_ELEMENTWISE_SCALERS = frozenset(
    {
        ("sklearn.preprocessing._data", "StandardScaler"),
        ("sklearn.preprocessing._data", "MinMaxScaler"),
        ("sklearn.preprocessing._data", "MaxAbsScaler"),
        ("sklearn.preprocessing._data", "RobustScaler"),
    }
)
_SKLEARN_PROBABILITY_TREES = frozenset(
    {
        ("sklearn.tree._classes", "DecisionTreeClassifier"),
        ("sklearn.tree._classes", "ExtraTreeClassifier"),
        ("sklearn.ensemble._forest", "RandomForestClassifier"),
        ("sklearn.ensemble._forest", "ExtraTreesClassifier"),
    }
)


def _type_key(obj: Any) -> tuple[str, str]:
    return type(obj).__module__, type(obj).__name__


@dataclass(frozen=True)
class ShapStrategy:
    """A ready SHAP explainer plus everything needed to interpret its output.

    ``attributions`` and ``model_outputs`` both take raw rows in artifact
    feature order. ``model_outputs`` returns, per class column, the exact
    quantity the attributions add up to — computed independently of SHAP.
    """

    kind: str
    reason: str
    explainer_type: str
    output_space: str
    output_description: str
    model_version: str
    model_family: str
    background_id: str
    background_version: str
    class_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    _attributions: Callable[[np.ndarray], tuple[Any, Any]]
    _model_outputs: Callable[[np.ndarray], np.ndarray]

    def attributions(self, rows: np.ndarray) -> tuple[Any, Any]:
        """(SHAP values, base values) exactly as the explainer returned them."""
        return self._attributions(rows)

    def model_outputs(self, rows: np.ndarray) -> np.ndarray:
        outputs = np.asarray(self._model_outputs(rows), dtype=float)
        expected = (rows.shape[0], len(self.class_names))
        if outputs.shape != expected:
            raise UnsupportedModelForShapError(
                f"model output has shape {outputs.shape}, expected {expected}"
            )
        return outputs


def _full_masker(shap: Any, data: np.ndarray) -> Any:
    """Independent masker over EVERY background row.

    SHAP's default silently subsamples to 100 rows; the explanation must use
    exactly the versioned background it reports.
    """
    return shap.maskers.Independent(data, max_samples=data.shape[0])


def _split_pipeline(model: Any) -> tuple[list[Any], Any]:
    steps = getattr(model, "steps", None)
    if steps:
        return [step for _, step in steps[:-1]], steps[-1][1]
    return [], model


def _preprocessing_is_elementwise(steps: list[Any]) -> bool:
    return all(step in (None, "passthrough") or _type_key(step) in _ELEMENTWISE_SCALERS for step in steps)


class ExplainerFactory:
    """Builds and caches SHAP strategies per (model, background, config)."""

    def __init__(self, config: ShapConfig | None = None) -> None:
        self._config = config or ShapConfig()
        self._cache: dict[tuple[Any, ...], ShapStrategy] = {}

    @property
    def config(self) -> ShapConfig:
        return self._config

    def create(self, adapter: ClassifierAdapter, background: XaiBackground) -> ShapStrategy:
        if background.feature_names != adapter.get_feature_names():
            raise ExplanationBackgroundMismatchError(
                "background columns do not match the model features: "
                f"{list(background.feature_names)} vs {list(adapter.get_feature_names())}"
            )
        model = adapter.get_fitted_model()
        # id(model) guards against two different artifacts that claim the
        # same model_version; the key never matches an incompatible model.
        key = (
            adapter.get_model_version(),
            id(model),
            background.background_id,
            background.background_version,
            self._config,
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        strategy = self._build(adapter, background)
        self._cache[key] = strategy
        return strategy

    def select_kind(self, adapter: ClassifierAdapter) -> tuple[str, str]:
        """(strategy kind, human-readable reason) for this fitted model."""
        steps, final = _split_pipeline(adapter.get_fitted_model())
        n_classes = len(adapter.get_class_names())
        n_features = len(adapter.get_feature_names())

        if not _preprocessing_is_elementwise(steps):
            names = [type(step).__name__ for step in steps]
            return "model_agnostic", (
                f"preprocessing {names} is not a known per-feature affine scaling; "
                "explaining the full pipeline on raw features"
            )

        module, name = _type_key(final)
        coef = getattr(final, "coef_", None)
        if (
            module.startswith("sklearn.linear_model")
            and callable(getattr(final, "decision_function", None))
            and coef is not None
            and np.shape(coef) == (n_classes, n_features)
        ):
            return "linear", f"{name} with per-class coefficients and elementwise scaling"

        if (module, name) in _SKLEARN_PROBABILITY_TREES:
            return "tree", f"{name}: interventional TreeExplainer, raw output = predict_proba"

        if module.startswith("xgboost"):
            return "model_agnostic", (
                f"{name}: XGBoost TreeExplainer path not enabled until a C2 XGBoost artifact "
                "exists to verify its output space; explaining predict_proba model-agnostically"
            )
        return "model_agnostic", f"{name}: no specialised SHAP strategy; model-agnostic"

    def _build(self, adapter: ClassifierAdapter, background: XaiBackground) -> ShapStrategy:
        kind, reason = self.select_kind(adapter)
        builders = {
            "linear": self._build_linear,
            "tree": self._build_tree,
            "model_agnostic": self._build_model_agnostic,
        }
        return builders[kind](adapter, background, reason)

    def _common(self, adapter: ClassifierAdapter, background: XaiBackground) -> dict[str, Any]:
        return {
            "model_version": adapter.get_model_version(),
            "model_family": adapter.get_model_family(),
            "background_id": background.background_id,
            "background_version": background.background_version,
            "class_names": adapter.get_class_names(),
            "feature_names": adapter.get_feature_names(),
        }

    @staticmethod
    def _preprocess(adapter: ClassifierAdapter) -> Callable[[np.ndarray], np.ndarray]:
        steps, _ = _split_pipeline(adapter.get_fitted_model())
        model = adapter.get_fitted_model()
        if not steps:
            return lambda rows: np.asarray(adapter.model_input(rows), dtype=float)
        preprocessing = model[:-1]
        return lambda rows: np.asarray(
            preprocessing.transform(adapter.model_input(rows)), dtype=float
        )

    def _build_linear(
        self, adapter: ClassifierAdapter, background: XaiBackground, reason: str
    ) -> ShapStrategy:
        import shap

        _, final = _split_pipeline(adapter.get_fitted_model())
        preprocess = self._preprocess(adapter)
        explainer = shap.LinearExplainer(final, _full_masker(shap, preprocess(background.matrix())))

        return ShapStrategy(
            kind="linear",
            reason=reason,
            explainer_type=type(explainer).__name__,
            output_space="raw",
            output_description=(
                "per-class decision_function of the final linear estimator "
                "(softmax logit for a multinomial model), not probability"
            ),
            _attributions=lambda rows: (
                explainer.shap_values(preprocess(rows)),
                explainer.expected_value,
            ),
            _model_outputs=lambda rows: final.decision_function(preprocess(rows)),
            **self._common(adapter, background),
        )

    def _build_tree(
        self, adapter: ClassifierAdapter, background: XaiBackground, reason: str
    ) -> ShapStrategy:
        import shap

        _, final = _split_pipeline(adapter.get_fitted_model())
        preprocess = self._preprocess(adapter)
        explainer = shap.TreeExplainer(
            final,
            data=_full_masker(shap, preprocess(background.matrix())),
            feature_perturbation="interventional",
        )

        return ShapStrategy(
            kind="tree",
            reason=reason,
            explainer_type=type(explainer).__name__,
            output_space="probability",
            output_description="class probability (predict_proba of the full pipeline)",
            # C4 runs its own reconstruction check, with its own diagnostics.
            _attributions=lambda rows: (
                explainer.shap_values(preprocess(rows), check_additivity=False),
                explainer.expected_value,
            ),
            _model_outputs=adapter.predict_proba_batch,
            **self._common(adapter, background),
        )

    def _build_model_agnostic(
        self, adapter: ClassifierAdapter, background: XaiBackground, reason: str
    ) -> ShapStrategy:
        import shap

        masker = _full_masker(shap, background.matrix())

        def predict(rows: np.ndarray) -> np.ndarray:
            return adapter.predict_proba_batch(np.asarray(rows, dtype=float))

        if len(adapter.get_feature_names()) <= self._config.max_exact_features:
            explainer = shap.explainers.Exact(predict, masker)
        else:
            explainer = shap.explainers.Permutation(
                predict, masker, seed=self._config.permutation_seed
            )

        def attributions(rows: np.ndarray) -> tuple[Any, Any]:
            explanation = explainer(rows)
            return explanation.values, explanation.base_values

        return ShapStrategy(
            kind="model_agnostic",
            reason=reason,
            explainer_type=type(explainer).__name__,
            output_space="probability",
            output_description="class probability (predict_proba of the full pipeline)",
            _attributions=attributions,
            _model_outputs=adapter.predict_proba_batch,
            **self._common(adapter, background),
        )
