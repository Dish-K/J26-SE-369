"""Repeated-seed LIME Stability (stability_v1) — a proposed C4 aggregation.

LIME runs ``repetitions`` times on the SAME prediction, feature vector,
model, target class, background and LIME configuration; only the seed
changes (deterministic schedule). Over all R(R-1)/2 unique run pairs:

    S_set  = mean pairwise Top-K Jaccard
    S_rank = mean pairwise (rho' + tau') / 2 over all features, |weight|
    S      = (S_set + S_rank) / 2

Every run is kept, including its local-fit score; low-fit runs are never
dropped. If any planned run fails, Stability is not computable — the
denominator is never silently reduced.
"""

import itertools
import logging
import statistics

from backend.app.explainability.config import StabilityConfig
from backend.app.explainability.errors import ExplanationAlignmentError, ExplanationError
from backend.app.explainability.evaluation.ranking import check_top_k, jaccard, rank_correlation
from backend.app.explainability.explanations.alignment import check_alignment
from backend.app.explainability.explanations.background import XaiBackground
from backend.app.explainability.explanations.lime_explainer import LimeExplainer
from backend.app.explainability.model.adapter import ClassifierAdapter
from backend.app.explainability.schemas import (
    AnalysisContext,
    FailedLimeRun,
    LimeExplanationResult,
    LimeFitSummary,
    StabilityAtKResult,
    StabilityResult,
)

logger = logging.getLogger(__name__)


def _summary(values: list[float]) -> tuple[float | None, float | None, float | None, float | None]:
    if not values:
        return None, None, None, None
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return statistics.fmean(values), min(values), max(values), std


def _fit_summary(runs: list[LimeExplanationResult]) -> LimeFitSummary:
    scores = [run.score for run in runs if run.score is not None]
    mean, minimum, maximum, std = _summary(scores)
    return LimeFitSummary(
        count=len(scores),
        mean=mean,
        median=statistics.median(scores) if scores else None,
        minimum=minimum,
        maximum=maximum,
        std=std,
    )


def _check_run_provenance(context: AnalysisContext, run: LimeExplanationResult) -> None:
    prediction = context.representative_prediction
    expected = {
        "prediction_id": prediction.prediction_id,
        "feature_vector_id": prediction.feature_vector_id,
        "model_version": context.model_version,
        "target_class": context.target_class,
        "target_class_index": context.target_class_index,
    }
    for name, value in expected.items():
        if getattr(run, name) != value:
            raise ExplanationAlignmentError(
                f"LIME run with seed {run.seed} explains a different {name}"
            )


class StabilityEvaluator:
    def __init__(
        self, lime_explainer: LimeExplainer | None = None, config: StabilityConfig | None = None
    ) -> None:
        self._lime = lime_explainer or LimeExplainer()
        self._config = config or StabilityConfig()

    @property
    def config(self) -> StabilityConfig:
        return self._config

    def evaluate(
        self, context: AnalysisContext, adapter: ClassifierAdapter, background: XaiBackground
    ) -> StabilityResult:
        # Contract problems are errors, raised before any run.
        check_alignment(context, adapter, background)
        feature_count = len(context.feature_names)
        for k in self._config.top_k_values:
            check_top_k(k, feature_count, "stability")

        seeds = self._config.seeds()
        runs: list[LimeExplanationResult] = []
        failures: list[FailedLimeRun] = []
        for seed in seeds:
            try:
                run = self._lime.explain(context, adapter, background, seed=seed)
            except ExplanationError as exc:
                failures.append(
                    FailedLimeRun(seed=seed, error_type=type(exc).__name__, message=str(exc))
                )
                continue
            _check_run_provenance(context, run)
            runs.append(run)

        result = self._result(context, background, seeds, runs, failures)
        logger.info(
            "Stability %s for prediction %s: %d/%d runs, K=%d status=%s score=%s",
            result.version,
            result.prediction_id,
            result.successful_runs,
            result.repetitions_requested,
            result.primary_top_k,
            result.status,
            result.score,
        )
        return result

    def _result(
        self,
        context: AnalysisContext,
        background: XaiBackground,
        seeds: tuple[int, ...],
        runs: list[LimeExplanationResult],
        failures: list[FailedLimeRun],
    ) -> StabilityResult:
        common = dict(
            version=self._config.version,
            prediction_id=context.representative_prediction.prediction_id,
            feature_vector_id=context.representative_prediction.feature_vector_id,
            model_version=context.model_version,
            target_class=context.target_class,
            target_class_index=context.target_class_index,
            repetitions_requested=len(seeds),
            successful_runs=len(runs),
            failed_runs=tuple(failures),
            seeds=seeds,
            lime_num_samples=self._lime.config.num_samples,
            lime_discretize_continuous=self._lime.config.discretize_continuous,
            background_id=background.background_id,
            background_version=background.background_version,
            primary_top_k=self._config.primary_top_k,
            lime_fit=_fit_summary(runs),
            runs=tuple(runs),
        )

        if failures:
            reason = (
                f"{len(failures)} of {len(seeds)} planned LIME runs failed "
                f"(seeds {[f.seed for f in failures]}); all runs are required"
            )
            return StabilityResult(
                **common,
                pair_count=0,
                rank_status="not_computable",
                rank_reason=reason,
                at_k=tuple(
                    StabilityAtKResult(k=k, pair_count=0, status="not_computable", reason=reason)
                    for k in self._config.top_k_values
                ),
                status="not_computable",
                reason=reason,
            )

        # Unique unordered index pairs: no self-pairs, (i, j) never also (j, i).
        pairs = list(itertools.combinations(range(len(runs)), 2))
        universe = sorted(f.name for f in runs[0].features)
        magnitudes = [
            [{f.name: f.absolute_weight for f in run.features}[n] for n in universe]
            for run in runs
        ]
        correlations = [rank_correlation(magnitudes[i], magnitudes[j]) for i, j in pairs]
        undefined = [c.reason for c in correlations if not c.computable]
        if undefined:
            rank_stability = mean_rho = mean_tau = None
            rank_status, rank_reason = "not_computable", f"{len(undefined)} run pairs: {undefined[0]}"
        else:
            mean_rho = statistics.fmean(c.spearman_normalized for c in correlations)  # type: ignore[misc]
            mean_tau = statistics.fmean(c.kendall_normalized for c in correlations)  # type: ignore[misc]
            rank_stability = statistics.fmean(c.rank_agreement for c in correlations)  # type: ignore[misc]
            rank_status, rank_reason = "computed", None

        at_k = []
        for k in self._config.top_k_values:
            tops = [[f.name for f in run.features][:k] for run in runs]
            jaccards = [jaccard(tops[i], tops[j]) for i, j in pairs]
            mean, minimum, maximum, std = _summary(jaccards)
            if rank_stability is None:
                score, status, reason = None, "not_computable", rank_reason
            else:
                score, status, reason = (mean + rank_stability) / 2, "computed", None  # type: ignore[operator]
            at_k.append(
                StabilityAtKResult(
                    k=k,
                    pair_count=len(pairs),
                    set_stability=mean,
                    min_pairwise_jaccard=minimum,
                    max_pairwise_jaccard=maximum,
                    std_pairwise_jaccard=std,
                    status=status,
                    stability_score=score,
                    reason=reason,
                )
            )
        primary = next(r for r in at_k if r.k == self._config.primary_top_k)
        return StabilityResult(
            **common,
            pair_count=len(pairs),
            mean_spearman_normalized=mean_rho,
            mean_kendall_normalized=mean_tau,
            rank_stability=rank_stability,
            rank_status=rank_status,
            rank_reason=rank_reason,
            at_k=tuple(at_k),
            status=primary.status,
            score=primary.stability_score,
            reason=primary.reason,
        )
