"""Explicit C2 -> C4 integration failures.

Every check C4 performs on C2 output raises one of these rather than returning
``None``, ``0`` or an empty container. A broken contract must stop the analysis
with a diagnostic, never be smoothed over.
"""


class ExplainabilityContractError(Exception):
    """Base class for every C2 -> C4 contract failure."""


# --- missing records / artifacts -------------------------------------------


class FlagNotFoundError(ExplainabilityContractError):
    pass


class PredictionNotFoundError(ExplainabilityContractError):
    pass


class FeatureVectorNotFoundError(ExplainabilityContractError):
    pass


class ModelArtifactNotFoundError(ExplainabilityContractError):
    pass


# --- malformed records / artifacts -----------------------------------------


class C2RecordInvalidError(ExplainabilityContractError):
    """A row read from the C2 tables does not satisfy the C2 output contract."""


class ModelArtifactInvalidError(ExplainabilityContractError):
    """An artifact exists but does not satisfy the C2 artifact contract."""


class FeatureRegistryError(ExplainabilityContractError):
    """The shared feature registry is missing or malformed."""


# --- cross-record consistency ----------------------------------------------


class ModelVersionMismatchError(ExplainabilityContractError):
    pass


class FeatureSchemaMismatchError(ExplainabilityContractError):
    pass


class MissingFeatureError(ExplainabilityContractError):
    """A feature the model requires is absent from the vector or the registry."""


class FeatureValueError(ExplainabilityContractError):
    """A required model feature holds a value the model cannot consume."""


class ClassOrderMismatchError(ExplainabilityContractError):
    pass


class ProbabilityMismatchError(ExplainabilityContractError):
    pass


class RepresentativePredictionMismatchError(ExplainabilityContractError):
    """The flag's representative prediction does not satisfy C2 provenance rules."""


class FeatureVectorProvenanceError(ExplainabilityContractError):
    """The feature vector is not the live input behind the representative prediction."""


class ModelInferenceError(ExplainabilityContractError):
    """The loaded model raised while scoring a validated feature vector."""


# --- Phase 2: SHAP / LIME explanation failures -------------------------------


class ExplanationError(ExplainabilityContractError):
    """An explanation could not be produced or failed its own validity checks."""


class UnsupportedModelForShapError(ExplanationError):
    pass


class ExplanationBackgroundMismatchError(ExplanationError):
    """The XAI background is not compatible with the model or the analysis context."""


class ExplanationShapeError(ExplanationError):
    """An explainer returned attributions whose layout cannot be interpreted safely."""


class TargetClassNotFoundError(ExplanationError):
    pass


class ShapReconstructionError(ExplanationError):
    """base_value + sum(SHAP values) does not reproduce the explained model output."""


class LimeExplanationError(ExplanationError):
    pass


class CanonicalFeatureMappingError(ExplanationError):
    """An explainer's feature label cannot be mapped to exactly one model feature."""


# --- Phase 3: explanation reliability metrics -------------------------------


class ExplanationAlignmentError(ExplanationError):
    """Explanations or metric results being combined do not describe the same
    prediction, feature vector, model version, target class and features."""


class InvalidMetricConfigurationError(ExplanationError):
    """A reliability metric was configured with values it cannot use (e.g. K out of range)."""
