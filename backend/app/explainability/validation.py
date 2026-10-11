"""Cross-record C2 -> C4 consistency checks.

Single-row invariants (probability sums, confidence-matches-class, ...) are
enforced when rows are parsed into schemas. The checks here relate several
records to each other and to the model artifact. Each one raises a specific
``ExplainabilityContractError``; none of them repairs data.
"""

from collections.abc import Sequence

from backend.app.explainability.config import C2_CONTRACT_TOLERANCE, LIVE_FEATURE_RUN_ID
from backend.app.explainability.errors import (
    FeatureSchemaMismatchError,
    FeatureVectorProvenanceError,
    ModelVersionMismatchError,
    ProbabilityMismatchError,
    RepresentativePredictionMismatchError,
)
from backend.app.explainability.feature_registry import FeatureRegistry
from backend.app.explainability.schemas import (
    BackgroundReference,
    ClassifierFlag,
    FeatureVector,
    ModelArtifactMetadata,
    ReproductionResult,
    WindowPrediction,
)


def _max_abs_difference(a: dict[str, float], b: dict[str, float]) -> float:
    return max(abs(a[name] - b[name]) for name in a)


def _winning_classes(probabilities: dict[str, float], tolerance: float) -> set[str]:
    best = max(probabilities.values())
    return {name for name, p in probabilities.items() if best - p <= tolerance}


def validate_representative(flag: ClassifierFlag, prediction: WindowPrediction) -> None:
    """The prediction is the flag's representative and the flag copied it unchanged."""
    if prediction.prediction_id != flag.representative_prediction_id:
        raise RepresentativePredictionMismatchError(
            f"flag {flag.flag_id} names representative {flag.representative_prediction_id}, "
            f"got prediction {prediction.prediction_id}"
        )
    if prediction.model_version != flag.model_version:
        raise ModelVersionMismatchError(
            f"flag {flag.flag_id} model_version {flag.model_version!r} != representative "
            f"prediction model_version {prediction.model_version!r}"
        )
    if prediction.task_id != flag.task_id:
        raise RepresentativePredictionMismatchError(
            f"representative prediction belongs to task {prediction.task_id}, "
            f"flag belongs to task {flag.task_id}"
        )
    if prediction.predicted_category != flag.category:
        raise RepresentativePredictionMismatchError(
            f"flag category {flag.category!r} != representative predicted_category "
            f"{prediction.predicted_category!r}"
        )
    if not prediction.ai_positive:
        raise RepresentativePredictionMismatchError(
            f"representative prediction {prediction.prediction_id} is not ai_positive"
        )
    if abs(flag.confidence - prediction.confidence) > C2_CONTRACT_TOLERANCE:
        raise ProbabilityMismatchError(
            f"flag confidence {flag.confidence} != representative confidence "
            f"{prediction.confidence}"
        )
    difference = _max_abs_difference(flag.class_probabilities, prediction.class_probabilities)
    if difference > C2_CONTRACT_TOLERANCE:
        raise ProbabilityMismatchError(
            f"flag class_probabilities differ from the representative's by {difference}: "
            f"{flag.class_probabilities} vs {prediction.class_probabilities}"
        )
    if prediction.predicted_category not in _winning_classes(
        prediction.class_probabilities, C2_CONTRACT_TOLERANCE
    ):
        raise RepresentativePredictionMismatchError(
            f"predicted_category {prediction.predicted_category!r} is not the highest-"
            f"probability class in {prediction.class_probabilities}"
        )


def validate_contributors(
    flag: ClassifierFlag,
    representative: WindowPrediction,
    contributors: Sequence[WindowPrediction],
) -> None:
    """C2 segment rules: members recorded, all compatible, representative is the max."""
    if len(contributors) != flag.window_count:
        raise RepresentativePredictionMismatchError(
            f"flag {flag.flag_id} has window_count {flag.window_count} but "
            f"{len(contributors)} contributing predictions are recorded"
        )
    ids = {member.prediction_id for member in contributors}
    if representative.prediction_id not in ids:
        raise RepresentativePredictionMismatchError(
            f"representative {representative.prediction_id} is not among the flag's "
            "contributing predictions"
        )
    for member in contributors:
        if (
            member.task_id != flag.task_id
            or member.model_version != flag.model_version
            or member.predicted_category != flag.category
            or not member.ai_positive
        ):
            raise RepresentativePredictionMismatchError(
                f"contributing prediction {member.prediction_id} does not match the flag's "
                "task/model/category or is not ai_positive"
            )
    highest = max(member.confidence for member in contributors)
    if highest - representative.confidence > C2_CONTRACT_TOLERANCE:
        raise RepresentativePredictionMismatchError(
            f"representative confidence {representative.confidence} is below the highest "
            f"member confidence {highest}; C2 must select the highest-confidence member"
        )


def validate_feature_vector_link(
    prediction: WindowPrediction, feature_vector: FeatureVector
) -> None:
    if feature_vector.feature_vector_id != prediction.feature_vector_id:
        raise FeatureVectorProvenanceError(
            f"prediction {prediction.prediction_id} references feature vector "
            f"{prediction.feature_vector_id}, got {feature_vector.feature_vector_id}"
        )
    if feature_vector.task_id != prediction.task_id:
        raise FeatureVectorProvenanceError(
            f"feature vector task {feature_vector.task_id} != prediction task {prediction.task_id}"
        )
    if feature_vector.run_id != LIVE_FEATURE_RUN_ID:
        raise FeatureVectorProvenanceError(
            f"feature vector {feature_vector.feature_vector_id} has run_id "
            f"{feature_vector.run_id!r}; only {LIVE_FEATURE_RUN_ID!r} vectors are explainable"
        )


def validate_artifact_compatibility(
    flag: ClassifierFlag,
    artifact: ModelArtifactMetadata,
    feature_vector: FeatureVector,
    registry: FeatureRegistry,
) -> None:
    """Same model version and one Contract-A version across all three sources.

    Compatibility is exact string equality: no cross-version compatibility
    rules have been agreed between C1, C2 and C4.
    """
    if artifact.model_version != flag.model_version:
        raise ModelVersionMismatchError(
            f"artifact model_version {artifact.model_version!r} != flag model_version "
            f"{flag.model_version!r}"
        )
    versions = {
        "artifact.feature_schema_version": artifact.feature_schema_version,
        "feature_vector.schema_version": feature_vector.schema_version,
        "feature_registry.schema_version": registry.schema_version,
    }
    if len(set(versions.values())) != 1:
        raise FeatureSchemaMismatchError(f"feature schema versions disagree: {versions}")


def validate_background_reference(
    reference: BackgroundReference, artifact: ModelArtifactMetadata
) -> None:
    if reference.schema_version != artifact.feature_schema_version:
        raise FeatureSchemaMismatchError(
            f"XAI background {reference.reference_id!r} uses schema "
            f"{reference.schema_version!r}, artifact uses {artifact.feature_schema_version!r}"
        )


def compare_reproduction(
    stored: WindowPrediction,
    reproduced: dict[str, float],
    tolerance: float,
) -> ReproductionResult:
    """The model re-run by C4 must give the probabilities C2 persisted."""
    difference = _max_abs_difference(stored.class_probabilities, reproduced)
    winners = _winning_classes(reproduced, tolerance)
    # A tie within tolerance is resolved toward the stored category rather
    # than by dict order; a genuine change of winner still fails below.
    reproduced_category = (
        stored.predicted_category
        if stored.predicted_category in winners
        else max(reproduced, key=reproduced.__getitem__)
    )
    result = ReproductionResult(
        stored_probabilities=dict(stored.class_probabilities),
        reproduced_probabilities=reproduced,
        max_abs_difference=difference,
        tolerance=tolerance,
        stored_category=stored.predicted_category,
        reproduced_category=reproduced_category,
    )
    if difference > tolerance:
        raise ProbabilityMismatchError(
            f"re-running {stored.model_version!r} on feature vector {stored.feature_vector_id} "
            f"does not reproduce stored C2 probabilities (max |diff| {difference:.3g} > "
            f"{tolerance:g}): stored {stored.class_probabilities}, reproduced {reproduced}"
        )
    if stored.predicted_category not in winners:
        raise RepresentativePredictionMismatchError(
            f"reproduced category {reproduced_category!r} != stored "
            f"{stored.predicted_category!r}"
        )
    return result
