"""C4 configuration, read from environment variables.

No value here is a research parameter. These are integration settings only.
"""

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from backend.app.explainability.errors import InvalidMetricConfigurationError

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


# ---------------------------------------------------------------------------
# Phase 3: explanation reliability (Agreement, Stability, Faithfulness, ECS)
#
# Every aggregation below is a PROPOSED C4 design (ECS_CRS_Mathematical_Design
# Part A), versioned so a later change becomes a new version rather than a
# silent redefinition. The primary K values were fixed as PROVISIONAL
# DEVELOPMENT values before any Phase 3 result was computed; all configured
# K values are still reported for sensitivity analysis.
# ---------------------------------------------------------------------------

AGREEMENT_VERSION = "agreement_v1_hierarchical"
STABILITY_VERSION = "stability_v1"
FAITHFULNESS_VERSION = "faithfulness_v1"
ECS_VERSION = "ecs_v1_equal_weight"
PERTURBATION_METHOD = "background_feature_replacement_v1"

_WEIGHT_SUM_TOLERANCE = 1e-9


def _check_top_k(values: tuple[int, ...], primary: int, what: str) -> None:
    if not values or len(set(values)) != len(values) or any(k < 1 for k in values):
        raise InvalidMetricConfigurationError(
            f"{what}: top_k_values must be distinct positive integers, got {values}"
        )
    if primary not in values:
        raise InvalidMetricConfigurationError(
            f"{what}: primary_top_k {primary} is not one of {values}"
        )


@dataclass(frozen=True)
class AgreementConfig:
    """SHAP-LIME agreement. Ties in |attribution| are broken by canonical
    model feature order (Phase 2 ranking); correlations are tie-aware."""

    top_k_values: tuple[int, ...] = (3, 5)
    primary_top_k: int = 5  # provisional development value
    version: str = AGREEMENT_VERSION

    def __post_init__(self) -> None:
        _check_top_k(self.top_k_values, self.primary_top_k, "agreement")


@dataclass(frozen=True)
class StabilityConfig:
    """Repeated seeded LIME. Seeds are base_seed, base_seed + 1, ...; with the
    default base_seed the first run equals the Phase 2 single LIME run."""

    repetitions: int = 20  # development configuration, not a constant
    base_seed: int = 42
    top_k_values: tuple[int, ...] = (3, 5)
    primary_top_k: int = 5  # provisional development value
    version: str = STABILITY_VERSION

    def __post_init__(self) -> None:
        if self.repetitions < 2:
            raise InvalidMetricConfigurationError("stability needs at least 2 repetitions")
        _check_top_k(self.top_k_values, self.primary_top_k, "stability")

    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.base_seed, self.base_seed + self.repetitions))


@dataclass(frozen=True)
class FaithfulnessConfig:
    """Perturbation faithfulness in target-class PROBABILITY space.

    For each K the K top-ranked and K bottom-ranked features are each replaced
    by values from the SAME ``perturbation_samples`` background rows (chosen
    with ``perturbation_seed``). ``min_total_effect`` guards the degenerate
    case where neither perturbation moves the probability: the score is then
    not computable rather than 0. All values are development settings.
    """

    ranking_source: str = "shap"
    top_k_values: tuple[int, ...] = (1, 3, 5)
    primary_top_k: int = 3  # provisional development value
    perturbation_method: str = PERTURBATION_METHOD
    perturbation_samples: int = 50
    perturbation_seed: int = 42
    epsilon: float = 1e-12
    min_total_effect: float = 1e-3
    version: str = FAITHFULNESS_VERSION

    def __post_init__(self) -> None:
        _check_top_k(self.top_k_values, self.primary_top_k, "faithfulness")
        if self.ranking_source not in ("shap", "lime", "consensus"):
            raise InvalidMetricConfigurationError(
                f"unknown faithfulness ranking_source {self.ranking_source!r}"
            )
        if self.perturbation_samples < 1:
            raise InvalidMetricConfigurationError("perturbation_samples must be >= 1")
        if not (self.epsilon > 0 and self.min_total_effect >= 0):
            raise InvalidMetricConfigurationError("epsilon must be > 0 and min_total_effect >= 0")


@dataclass(frozen=True)
class EcsWeights:
    agreement: float
    stability: float
    faithfulness: float

    def __post_init__(self) -> None:
        values = (self.agreement, self.stability, self.faithfulness)
        if not all(math.isfinite(w) and w >= 0 for w in values):
            raise InvalidMetricConfigurationError(f"ECS weights must be finite and >= 0: {values}")
        if abs(sum(values) - 1.0) > _WEIGHT_SUM_TOLERANCE:
            raise InvalidMetricConfigurationError(f"ECS weights must sum to 1, got {sum(values)}")

    def as_dict(self) -> dict[str, float]:
        return {
            "agreement": self.agreement,
            "stability": self.stability,
            "faithfulness": self.faithfulness,
        }


EQUAL_ECS_WEIGHTS = EcsWeights(1 / 3, 1 / 3, 1 / 3)

# Sensitivity scenarios from the ECS design (§11), with exact weights.
ECS_SENSITIVITY_SCENARIOS: tuple[tuple[str, EcsWeights], ...] = (
    ("equal_weight", EQUAL_ECS_WEIGHTS),
    ("agreement_emphasis", EcsWeights(0.50, 0.25, 0.25)),
    ("stability_emphasis", EcsWeights(0.25, 0.50, 0.25)),
    ("faithfulness_emphasis", EcsWeights(0.25, 0.25, 0.50)),
)


@dataclass(frozen=True)
class EcsConfig:
    weights: EcsWeights = EQUAL_ECS_WEIGHTS
    version: str = ECS_VERSION
    sensitivity_scenarios: tuple[tuple[str, EcsWeights], ...] = field(
        default=ECS_SENSITIVITY_SCENARIOS
    )


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
