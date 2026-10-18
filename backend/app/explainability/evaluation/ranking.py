"""Set and rank statistics shared by Agreement and Stability.

Top-K lists come from Phase 2 rank order: descending |attribution|, ties
broken by canonical model feature order, so the order never depends on dict
iteration. Correlations are computed on |attribution| with SciPy's tie-aware
estimators (Spearman with average ranks, Kendall tau-b). An undefined
correlation (e.g. constant magnitudes) is reported as not computable — it is
never turned into 0, 0.5 or 1.
"""

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from backend.app.explainability.errors import InvalidMetricConfigurationError


def check_top_k(k: int, feature_count: int, what: str) -> None:
    if not 1 <= k <= feature_count:
        raise InvalidMetricConfigurationError(
            f"{what}: K={k} must be between 1 and the {feature_count} model features"
        )


def top_k(ranked_names: Sequence[str], k: int) -> tuple[str, ...]:
    return tuple(ranked_names[:k])


def overlap_at_k(first: Sequence[str], second: Sequence[str], k: int) -> float:
    return len(set(first) & set(second)) / k


def jaccard(first: Sequence[str], second: Sequence[str]) -> float:
    union = set(first) | set(second)
    if not union:
        raise InvalidMetricConfigurationError("Jaccard of two empty sets is undefined")
    return len(set(first) & set(second)) / len(union)


def normalize_correlation(value: float) -> float:
    """Map a correlation from [-1, 1] to [0, 1]; clips float noise at the ends."""
    return min(1.0, max(0.0, (value + 1.0) / 2.0))


@dataclass(frozen=True)
class RankCorrelation:
    spearman: float | None
    kendall: float | None
    reason: str | None = None

    @property
    def computable(self) -> bool:
        return self.reason is None

    @property
    def spearman_normalized(self) -> float | None:
        return None if self.spearman is None else normalize_correlation(self.spearman)

    @property
    def kendall_normalized(self) -> float | None:
        return None if self.kendall is None else normalize_correlation(self.kendall)

    @property
    def rank_agreement(self) -> float | None:
        if not self.computable:
            return None
        return (self.spearman_normalized + self.kendall_normalized) / 2  # type: ignore[operator]


def rank_correlation(first: Sequence[float], second: Sequence[float]) -> RankCorrelation:
    """Tie-aware Spearman rho and Kendall tau-b of two aligned magnitude vectors."""
    x = np.asarray(first, dtype=float)
    y = np.asarray(second, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise InvalidMetricConfigurationError("rank correlation needs two aligned 1-D vectors")
    if x.size < 2:
        return RankCorrelation(None, None, "fewer than two features to rank")

    with warnings.catch_warnings():
        # Constant input makes both statistics undefined; SciPy warns and
        # returns NaN, which is reported below as not computable.
        warnings.simplefilter("ignore")
        spearman = float(stats.spearmanr(x, y).statistic)
        kendall = float(stats.kendalltau(x, y).statistic)

    if not (math.isfinite(spearman) and math.isfinite(kendall)):
        return RankCorrelation(
            None,
            None,
            "rank correlation undefined: at least one ranking has constant attribution magnitudes",
        )
    return RankCorrelation(spearman, kendall)
