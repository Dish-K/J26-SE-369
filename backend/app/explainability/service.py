"""Phase 1 orchestration: flag_id -> validated, reproduced AnalysisContext.

This is the single entry point later explanation phases build on. They take
the AnalysisContext (and ``adapter_for``) instead of re-querying C2 data.

Order of work:
    flag -> representative prediction -> contributing predictions
         -> feature vector -> artifact (by the flag's exact model_version)
         -> provenance + schema checks -> re-run the model
         -> compare with stored C2 probabilities -> AnalysisContext
"""

import json
import logging
import math
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import ValidationError

from backend.app.explainability.config import (
    DEFAULT_REPRODUCTION_TOLERANCE,
    ExplainabilitySettings,
)
from backend.app.explainability.errors import (
    ExplainabilityContractError,
    FeatureRegistryError,
    ModelArtifactNotFoundError,
)
from backend.app.explainability.feature_registry import FeatureRegistry, load_feature_registry
from backend.app.explainability.model.adapter import ClassifierAdapter
from backend.app.explainability.model.artifact_loader import ArtifactLoader
from backend.app.explainability.persistence.repository import C2OutputReader
from backend.app.explainability.schemas import (
    AnalysisContext,
    BackgroundReference,
    ModelArtifactMetadata,
)
from backend.app.explainability.validation import (
    compare_reproduction,
    validate_artifact_compatibility,
    validate_background_reference,
    validate_contributors,
    validate_feature_vector_link,
    validate_representative,
)

logger = logging.getLogger(__name__)


def load_dev_background_reference(path: Path) -> BackgroundReference:
    """TEMPORARY: read a background descriptor until C2 artifacts carry one."""
    try:
        return BackgroundReference.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ExplainabilityContractError(
            f"invalid development XAI background descriptor {path}: {exc}"
        ) from exc


class AnalysisService:
    def __init__(
        self,
        reader: C2OutputReader,
        artifact_loader: ArtifactLoader,
        registry: FeatureRegistry,
        *,
        reproduction_tolerance: float = DEFAULT_REPRODUCTION_TOLERANCE,
        dev_background_ref: BackgroundReference | None = None,
    ) -> None:
        self._reader = reader
        self._artifacts = artifact_loader
        self._registry = registry
        self._tolerance = reproduction_tolerance
        self._dev_background_ref = dev_background_ref

    @classmethod
    def from_settings(
        cls, reader: C2OutputReader, settings: ExplainabilitySettings
    ) -> "AnalysisService":
        if settings.model_artifact_dir is None:
            raise ModelArtifactNotFoundError("C4_MODEL_ARTIFACT_DIR is not configured")
        if settings.feature_registry_path is None:
            raise FeatureRegistryError("C4_FEATURE_REGISTRY_PATH is not configured")
        dev_background = (
            load_dev_background_reference(settings.xai_background_ref_path)
            if settings.xai_background_ref_path is not None
            else None
        )
        return cls(
            reader,
            ArtifactLoader(settings.model_artifact_dir),
            load_feature_registry(settings.feature_registry_path),
            reproduction_tolerance=settings.reproduction_tolerance,
            dev_background_ref=dev_background,
        )

    def adapter_for(self, model_version: str) -> ClassifierAdapter:
        return ClassifierAdapter(self._artifacts.load(model_version))

    async def build_context(self, flag_id: UUID) -> AnalysisContext:
        logger.info("Resolving C2 flag %s", flag_id)
        flag = await self._reader.get_flag(flag_id)

        prediction = await self._reader.get_prediction(flag.representative_prediction_id)
        validate_representative(flag, prediction)

        contributors = await self._reader.get_contributing_predictions(flag.flag_id)
        validate_contributors(flag, prediction, contributors)

        feature_vector = await self._reader.get_feature_vector(prediction.feature_vector_id)
        validate_feature_vector_link(prediction, feature_vector)
        logger.info(
            "Flag %s -> representative prediction %s -> feature vector %s (model_version=%s)",
            flag.flag_id,
            prediction.prediction_id,
            feature_vector.feature_vector_id,
            flag.model_version,
        )

        # Load strictly by the flag's own model_version; never "latest".
        artifact = self._artifacts.load(flag.model_version)
        metadata = artifact.metadata
        validate_artifact_compatibility(flag, metadata, feature_vector, self._registry)
        feature_metadata = self._registry.resolve(metadata.feature_names)

        background_ref, background_source = self._resolve_background(metadata)

        adapter = ClassifierAdapter(artifact)
        model_input = adapter.prepare_features(feature_vector.features)
        reproduced = adapter.predict_proba(feature_vector.features)
        try:
            reproduction = compare_reproduction(prediction, reproduced, self._tolerance)
        except ExplainabilityContractError:
            logger.error(
                "C2 reproduction FAILED for prediction %s (model_version=%s)",
                prediction.prediction_id,
                flag.model_version,
            )
            raise
        logger.info(
            "C2 prediction %s reproduced (max |diff| %.3g <= %g); C2 -> C4 contract valid",
            prediction.prediction_id,
            reproduction.max_abs_difference,
            reproduction.tolerance,
        )

        row = model_input.iloc[0]
        return AnalysisContext(
            flag=flag,
            representative_prediction=prediction,
            contributing_prediction_ids=tuple(member.prediction_id for member in contributors),
            feature_vector=feature_vector,
            artifact=metadata,
            model_version=metadata.model_version,
            model_family=artifact.model_family,
            feature_schema_version=metadata.feature_schema_version,
            target_class=flag.category,
            target_class_index=artifact.output_classes.index(flag.category),
            feature_names=metadata.feature_names,
            class_names=artifact.output_classes,
            model_input={
                name: None if math.isnan(row[name]) else float(row[name])
                for name in metadata.feature_names
            },
            feature_metadata={
                name: entry.model_dump(exclude_none=True)
                for name, entry in feature_metadata.items()
            },
            reproduction=reproduction,
            background_ref=background_ref,
            background_ref_source=background_source,
        )

    def _resolve_background(
        self, metadata: ModelArtifactMetadata
    ) -> tuple[BackgroundReference | None, Literal["artifact", "configured_dev"] | None]:
        if metadata.xai_background_ref is not None:
            validate_background_reference(metadata.xai_background_ref, metadata)
            return metadata.xai_background_ref, "artifact"
        if self._dev_background_ref is not None:
            validate_background_reference(self._dev_background_ref, metadata)
            logger.warning(
                "Using TEMPORARY configured XAI background %s; artifact %s carries none",
                self._dev_background_ref.reference_id,
                metadata.model_version,
            )
            return self._dev_background_ref, "configured_dev"
        return None, None
