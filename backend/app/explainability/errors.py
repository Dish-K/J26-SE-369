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
