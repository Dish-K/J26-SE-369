"""Local and global SHAP explanations of C2 predictions.

Local: explains exactly the representative prediction in an AnalysisContext,
for the representative predicted class, and verifies additive reconstruction

    base_value + sum(shap_values) == model_output   (same output space)

where ``model_output`` is computed independently of SHAP (see
``ShapStrategy.model_outputs``). The target-class probability is always
reported separately, because the SHAP output space may not be probability.

Global: class-specific mean |SHAP| over a caller-supplied set of analysis
samples, which is kept distinct from the background the explainer masks with.
"""

import logging

import numpy as np
import pandas as pd

from backend.app.explainability.errors import ShapReconstructionError
from backend.app.explainability.explanations.alignment import (
    check_alignment,
    checked_target_probability,
    instance_row,
    target_index,
)
from backend.app.explainability.explanations.attribution import (
    rank_attributions,
    select_class_attributions,
    select_class_base_values,
)
from backend.app.explainability.explanations.background import (
    XaiBackground,
    order_feature_frame,
)
from backend.app.explainability.model.adapter import ClassifierAdapter
from backend.app.explainability.model.explainer_factory import ExplainerFactory, ShapStrategy
from backend.app.explainability.schemas import (
    AnalysisContext,
    GlobalShapBeeswarm,
    GlobalShapFeatureImportance,
    GlobalShapResult,
    ShapExplanationResult,
    ShapFeatureContribution,
)

logger = logging.getLogger(__name__)


def verify_reconstruction(
    *,
    base_value: float,
    attributions: np.ndarray,
    model_output: float,
    tolerance: float,
    strategy: ShapStrategy,
) -> tuple[float, float]:
    """Return (reconstructed_output, error) or raise ShapReconstructionError."""
    reconstructed = float(base_value + np.sum(attributions))
    error = abs(float(model_output) - reconstructed)
    if not error <= tolerance:
        raise ShapReconstructionError(
            "SHAP reconstruction failed: "
            f"model_version={strategy.model_version!r} model_family={strategy.model_family!r} "
            f"explainer={strategy.explainer_type} output_space={strategy.output_space} "
            f"model_output={model_output!r} reconstructed_output={reconstructed!r} "
            f"error={error!r} tolerance={tolerance!r}"
        )
    return reconstructed, error


class ShapExplainer:
    def __init__(self, factory: ExplainerFactory | None = None) -> None:
        self._factory = factory or ExplainerFactory()

    @property
    def tolerance(self) -> float:
        return self._factory.config.reconstruction_tolerance

    def _class_attributions(
        self, strategy: ShapStrategy, rows: np.ndarray, class_index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        values, base_values = strategy.attributions(rows)
        n_classes = len(strategy.class_names)
        phi = select_class_attributions(
            values,
            n_instances=rows.shape[0],
            n_features=len(strategy.feature_names),
            n_classes=n_classes,
            class_index=class_index,
        )
        base = select_class_base_values(
            base_values, n_instances=rows.shape[0], n_classes=n_classes, class_index=class_index
        )
        return phi, base

    def explain(
        self,
        context: AnalysisContext,
        adapter: ClassifierAdapter,
        background: XaiBackground,
    ) -> ShapExplanationResult:
        check_alignment(context, adapter, background)
        prediction = context.representative_prediction
        class_index = context.target_class_index

        strategy = self._factory.create(adapter, background)
        row = instance_row(context, adapter)
        phi, base = self._class_attributions(strategy, row, class_index)
        model_output = float(strategy.model_outputs(row)[0, class_index])
        reconstructed, error = verify_reconstruction(
            base_value=float(base[0]),
            attributions=phi[0],
            model_output=model_output,
            tolerance=self.tolerance,
            strategy=strategy,
        )
        target_probability = checked_target_probability(
            context, adapter, adapter.predict_proba(context.model_input)
        )

        features = tuple(
            ShapFeatureContribution(
                name=ranked.name,
                value=context.model_input[ranked.name],
                shap_value=ranked.attribution,
                absolute_shap_value=ranked.magnitude,
                rank=ranked.rank,
                direction=ranked.direction,
            )
            for ranked in rank_attributions(strategy.feature_names, phi[0].tolist())
        )
        logger.info(
            "SHAP explained prediction %s (feature vector %s, model %s/%s, target %s[%d]) "
            "with %s/%s on background %s@%s; reconstruction error %.3g",
            prediction.prediction_id,
            prediction.feature_vector_id,
            strategy.model_version,
            strategy.model_family,
            context.target_class,
            class_index,
            strategy.kind,
            strategy.explainer_type,
            strategy.background_id,
            strategy.background_version,
            error,
        )
        return ShapExplanationResult(
            prediction_id=prediction.prediction_id,
            feature_vector_id=prediction.feature_vector_id,
            model_version=strategy.model_version,
            model_family=strategy.model_family,
            target_class=context.target_class,
            target_class_index=class_index,
            target_probability=target_probability,
            strategy=strategy.kind,
            strategy_reason=strategy.reason,
            explainer_type=strategy.explainer_type,
            output_space=strategy.output_space,
            output_description=strategy.output_description,
            background_id=strategy.background_id,
            background_version=strategy.background_version,
            base_value=float(base[0]),
            model_output=model_output,
            reconstructed_output=reconstructed,
            reconstruction_error=error,
            reconstruction_tolerance=self.tolerance,
            features=features,
        )

    def explain_global(
        self,
        adapter: ClassifierAdapter,
        background: XaiBackground,
        analysis_samples: pd.DataFrame,
        *,
        target_class: str,
        dataset_reference: str,
        include_beeswarm: bool = True,
    ) -> GlobalShapResult:
        """Mean |SHAP| for ``target_class`` over ``analysis_samples``.

        ``analysis_samples`` are the instances summarised; ``background`` is
        only the masking/reference distribution. They may coincide for a small
        prototype, but the caller decides — and names the samples in
        ``dataset_reference``.
        """
        class_index = target_index(adapter, target_class)
        strategy = self._factory.create(adapter, background)
        frame = order_feature_frame(
            analysis_samples, strategy.feature_names, what=f"analysis samples {dataset_reference!r}"
        )
        rows = frame.to_numpy(dtype=float)

        phi, base = self._class_attributions(strategy, rows, class_index)
        outputs = strategy.model_outputs(rows)[:, class_index]
        errors = [
            verify_reconstruction(
                base_value=float(base[i]),
                attributions=phi[i],
                model_output=float(outputs[i]),
                tolerance=self.tolerance,
                strategy=strategy,
            )[1]
            for i in range(rows.shape[0])
        ]

        mean_abs = np.abs(phi).mean(axis=0)
        mean_signed = phi.mean(axis=0)
        ranked = rank_attributions(strategy.feature_names, mean_abs.tolist())
        importance = tuple(
            GlobalShapFeatureImportance(
                name=item.name,
                mean_abs_shap=float(mean_abs[item.index]),
                mean_shap=float(mean_signed[item.index]),
                rank=item.rank,
            )
            for item in ranked
        )
        beeswarm = (
            GlobalShapBeeswarm(
                feature_names=strategy.feature_names,
                feature_values=tuple(tuple(map(float, r)) for r in rows),
                shap_values=tuple(tuple(map(float, r)) for r in phi),
            )
            if include_beeswarm
            else None
        )
        logger.info(
            "Global SHAP for model %s target %s over %d samples (%s); max reconstruction error %.3g",
            strategy.model_version,
            target_class,
            rows.shape[0],
            dataset_reference,
            max(errors),
        )
        return GlobalShapResult(
            model_version=strategy.model_version,
            model_family=strategy.model_family,
            feature_schema_version=background.reference.schema_version,
            target_class=target_class,
            target_class_index=class_index,
            strategy=strategy.kind,
            explainer_type=strategy.explainer_type,
            output_space=strategy.output_space,
            background_id=strategy.background_id,
            background_version=strategy.background_version,
            dataset_reference=dataset_reference,
            sample_count=rows.shape[0],
            max_reconstruction_error=max(errors),
            feature_importance=importance,
            beeswarm=beeswarm,
        )
