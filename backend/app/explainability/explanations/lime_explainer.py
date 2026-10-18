"""Seeded LIME explanation of the same prediction and class SHAP explains.

LIME's sampling is random, so each explanation builds a fresh
``LimeTabularExplainer`` with the configured seed: the same instance, model,
background and config give the same result. Repeated-run stability across
many seeds is Phase 3.

Canonical names come from LIME's structured feature indices (``local_exp``),
cross-checked against the display condition string.
"""

import logging
import math

import numpy as np
from lime.lime_tabular import LimeTabularExplainer

from backend.app.explainability.config import C2_CONTRACT_TOLERANCE, LimeConfig
from backend.app.explainability.errors import LimeExplanationError
from backend.app.explainability.explanations.alignment import (
    check_alignment,
    checked_target_probability,
    instance_row,
)
from backend.app.explainability.explanations.attribution import rank_attributions
from backend.app.explainability.explanations.background import XaiBackground
from backend.app.explainability.explanations.feature_normalizer import canonical_feature_name
from backend.app.explainability.model.adapter import ClassifierAdapter
from backend.app.explainability.schemas import (
    AnalysisContext,
    LimeExplanationResult,
    LimeFeatureContribution,
)

logger = logging.getLogger(__name__)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float).ravel()
    if array.size != 1 or not math.isfinite(array[0]):
        return None
    return float(array[0])


class LimeExplainer:
    def __init__(self, config: LimeConfig | None = None) -> None:
        self._config = config or LimeConfig()

    @property
    def config(self) -> LimeConfig:
        return self._config

    def explain(
        self,
        context: AnalysisContext,
        adapter: ClassifierAdapter,
        background: XaiBackground,
        *,
        seed: int | None = None,
    ) -> LimeExplanationResult:
        """Explain with ``seed`` if given, else the configured seed.

        The override lets repeated-run stability vary only the seed without
        mutating shared configuration.
        """
        seed = self._config.seed if seed is None else seed
        check_alignment(context, adapter, background)
        prediction = context.representative_prediction
        class_index = context.target_class_index
        feature_names = adapter.get_feature_names()
        instance = instance_row(context, adapter)[0]

        explainer = LimeTabularExplainer(
            background.matrix(),
            mode="classification",
            feature_names=list(feature_names),
            class_names=list(adapter.get_class_names()),
            discretize_continuous=self._config.discretize_continuous,
            random_state=seed,
        )
        # Every LIME model call goes through the validated Phase 1 adapter.
        explanation = explainer.explain_instance(
            instance,
            adapter.predict_proba_batch,
            labels=(class_index,),
            num_features=len(feature_names),
            num_samples=self._config.num_samples,
        )

        if class_index not in explanation.local_exp:
            raise LimeExplanationError(
                f"LIME did not explain requested class index {class_index}; "
                f"got {sorted(explanation.local_exp)}"
            )
        structured = explanation.local_exp[class_index]
        conditions = explanation.as_list(label=class_index)
        if len(structured) != len(feature_names) or len(conditions) != len(structured):
            raise LimeExplanationError(
                f"LIME returned {len(structured)} weights for {len(feature_names)} features"
            )

        # LIME scores the unperturbed instance first; it must be the
        # adapter's own prediction for this vector.
        lime_probability = float(np.asarray(explanation.predict_proba)[class_index])
        target_probability = checked_target_probability(
            context, adapter, adapter.predict_proba(context.model_input)
        )
        if not math.isclose(
            lime_probability, target_probability, rel_tol=0.0, abs_tol=C2_CONTRACT_TOLERANCE
        ):
            raise LimeExplanationError(
                f"LIME saw probability {lime_probability} for the instance, "
                f"adapter gives {target_probability}"
            )

        weights = [0.0] * len(feature_names)
        condition_by_index: dict[int, str] = {}
        for (index, weight), (condition, listed_weight) in zip(structured, conditions, strict=True):
            index = int(index)
            if not math.isclose(float(weight), float(listed_weight), rel_tol=0.0, abs_tol=1e-12):
                raise LimeExplanationError("LIME structured and listed weights disagree")
            if index in condition_by_index:
                raise LimeExplanationError(f"LIME reported feature index {index} twice")
            canonical_feature_name(index, condition, feature_names)
            weights[index] = float(weight)
            condition_by_index[index] = condition

        features = tuple(
            LimeFeatureContribution(
                name=ranked.name,
                feature_index=ranked.index,
                value=context.model_input[ranked.name],
                condition=condition_by_index[ranked.index],
                weight=ranked.attribution,
                absolute_weight=ranked.magnitude,
                rank=ranked.rank,
                direction=ranked.direction,
            )
            for ranked in rank_attributions(feature_names, weights)
        )
        logger.info(
            "LIME explained prediction %s (feature vector %s, model %s, target %s[%d]) "
            "seed=%d samples=%d background %s@%s",
            prediction.prediction_id,
            prediction.feature_vector_id,
            adapter.get_model_version(),
            context.target_class,
            class_index,
            seed,
            self._config.num_samples,
            background.background_id,
            background.background_version,
        )
        return LimeExplanationResult(
            prediction_id=prediction.prediction_id,
            feature_vector_id=prediction.feature_vector_id,
            model_version=adapter.get_model_version(),
            target_class=context.target_class,
            target_class_index=class_index,
            target_probability=target_probability,
            background_id=background.background_id,
            background_version=background.background_version,
            seed=seed,
            num_samples=self._config.num_samples,
            discretize_continuous=self._config.discretize_continuous,
            intercept=_optional_float(explanation.intercept.get(class_index)),
            local_prediction=_optional_float(explanation.local_pred),
            score=_optional_float(explanation.score),
            features=features,
        )
