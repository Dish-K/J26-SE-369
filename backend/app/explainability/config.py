"""C4 configuration, read from environment variables.

No value here is a research parameter. These are integration settings only.
"""

import os
from dataclasses import dataclass
from pathlib import Path

# Numerical tolerance written into C2's own database contract
# (005_classifier_output.sql: probability sum, confidence-matches-class and
# flag-matches-representative checks all use 0.000001). C4 compares persisted
# C2 values with the same tolerance so it is never looser than C2 itself.
C2_CONTRACT_TOLERANCE = 1e-6

# Tolerance for re-running the model and comparing against stored C2
# probabilities. Defaults to the C2 contract tolerance: a reproduction error
# larger than C2 itself accepts indicates a real integration fault.
DEFAULT_REPRODUCTION_TOLERANCE = C2_CONTRACT_TOLERANCE

# Only unmodified telemetry produces explainable C2 flags. Degraded copies
# written by C1's reliability study use other run_id values.
LIVE_FEATURE_RUN_ID = "live"


def _optional_path(name: str) -> Path | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return Path(value)


def _float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number") from exc
    if not 0.0 < parsed < 1.0:
        raise RuntimeError(f"{name} must be between 0 and 1 (exclusive)")
    return parsed


@dataclass(frozen=True)
class ShapConfig:
    """Development / PP1 SHAP settings. Not research constants.

    reconstruction_tolerance:
        Max |model_output - (base_value + sum(shap_values))| in the explainer's
        own output space. Every strategy used here (linear, interventional tree,
        exact/permutation) is additive by construction, so the only error is
        floating-point; 1e-6 matches the C2 contract tolerance.
    max_exact_features:
        Model-agnostic strategy: exact Shapley enumeration (2^n coalitions) up
        to this many features, a seeded permutation explainer above it.
    permutation_seed:
        Seed for the permutation explainer only.
    """

    reconstruction_tolerance: float = C2_CONTRACT_TOLERANCE
    max_exact_features: int = 12
    permutation_seed: int = 42


@dataclass(frozen=True)
class LimeConfig:
    """Development / PP1 LIME settings. Not research constants.

    One seeded run per explanation; repeated-run stability is Phase 3.
    """

    seed: int = 42
    num_samples: int = 5000
    discretize_continuous: bool = True


# Attributions whose magnitude is at most this are reported as "neutral"
# rather than toward/away from the target class. Purely numerical: it only
# absorbs floating-point noise around exact zero.
ATTRIBUTION_NEUTRAL_EPSILON = 1e-12


@dataclass(frozen=True)
class ExplainabilitySettings:
    """Locations of C2 artifacts and the shared registry, plus tolerances.

    Environment variables:
        C4_MODEL_ARTIFACT_DIR       directory holding ``<model_version>/artifact.joblib``
        C4_FEATURE_REGISTRY_PATH    shared feature registry YAML
        C4_XAI_BACKGROUND_REF_PATH  TEMPORARY development descriptor for the
                                    XAI background set, used only when the
                                    artifact carries no ``xai_background_ref``
        C4_REPRODUCTION_TOLERANCE   absolute per-class probability tolerance
    """

    model_artifact_dir: Path | None
    feature_registry_path: Path | None
    xai_background_ref_path: Path | None = None
    reproduction_tolerance: float = DEFAULT_REPRODUCTION_TOLERANCE

    @classmethod
    def from_env(cls) -> "ExplainabilitySettings":
        return cls(
            model_artifact_dir=_optional_path("C4_MODEL_ARTIFACT_DIR"),
            feature_registry_path=_optional_path("C4_FEATURE_REGISTRY_PATH"),
            xai_background_ref_path=_optional_path("C4_XAI_BACKGROUND_REF_PATH"),
            reproduction_tolerance=_float(
                "C4_REPRODUCTION_TOLERANCE", DEFAULT_REPRODUCTION_TOLERANCE
            ),
        )
