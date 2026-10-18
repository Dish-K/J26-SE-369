"""Agreement (agreement_v1_hierarchical). Expected values are computed by hand."""

import pytest
from scipy import stats

from backend.app.explainability.config import AgreementConfig
from backend.app.explainability.errors import (
    ExplanationAlignmentError,
    InvalidMetricConfigurationError,
)
from backend.app.explainability.evaluation.agreement import AgreementEvaluator
from backend.app.explainability.evaluation.ranking import (
    jaccard,
    normalize_correlation,
    overlap_at_k,
    rank_correlation,
)
from tests.explainability.fixtures.explanation_builders import lime_result, shap_result

DESCENDING = {"A": 6.0, "B": 5.0, "C": 4.0, "D": 3.0, "E": 2.0, "F": 1.0}
K3 = AgreementConfig(top_k_values=(3,), primary_top_k=3)


def _evaluate(shap: dict, lime: dict, config: AgreementConfig = K3):  # type: ignore[no-untyped-def]
    return AgreementEvaluator(config).evaluate(shap_result(shap), lime_result(lime))


# --- set statistics ----------------------------------------------------------


def test_overlap_and_jaccard_basic_values() -> None:
    assert overlap_at_k(["a", "b", "c"], ["a", "b", "c"], 3) == 1.0
    assert jaccard(["a", "b", "c"], ["c", "b", "a"]) == 1.0
    assert overlap_at_k(["a", "b", "c", "d", "e"], ["a", "b", "c", "x", "y"], 5) == 0.6
    assert overlap_at_k(["a", "b"], ["c", "d"], 2) == 0.0
    assert jaccard(["a", "b"], ["c", "d"]) == 0.0


def test_identical_explanations_agree_fully() -> None:
    result = _evaluate(DESCENDING, DESCENDING)
    at3 = result.at_k[0]
    assert at3.overlap_at_k == at3.jaccard == at3.feature_set_agreement == 1.0
    assert result.spearman_raw == pytest.approx(1.0)
    assert result.kendall_raw == pytest.approx(1.0)
    assert result.spearman_normalized == result.kendall_normalized == 1.0
    assert result.rank_agreement == 1.0
    assert at3.direction_agreement == 1.0
    assert result.score == pytest.approx(1.0)


def test_reversed_ranking_disagrees_fully() -> None:
    reversed_lime = {name: 7.0 - value for name, value in DESCENDING.items()}
    result = _evaluate(DESCENDING, reversed_lime)
    at3 = result.at_k[0]
    assert result.spearman_raw == pytest.approx(-1.0)
    assert result.kendall_raw == pytest.approx(-1.0)
    assert result.rank_agreement == pytest.approx(0.0)
    assert at3.intersection == ()
    assert at3.overlap_at_k == at3.jaccard == 0.0
    # documented v1 convention when the Top-K sets share nothing
    assert at3.direction_status == "no_common_top_k_features"
    assert at3.direction_agreement == 0.0
    assert result.score == pytest.approx(0.0)


def test_golden_hand_calculated_agreement() -> None:
    """SHAP order A,B,C,D,E,F; LIME order A,C,D,B,E,F with C pointing away.

    Top-3: SHAP {A,B,C}, LIME {A,C,D} -> intersection {A,C}
      Overlap@3 = 2/3, Jaccard = 2/4, A_set = (2/3 + 1/2) / 2
    Ranks SHAP (1,2,3,4,5,6), LIME (1,4,2,3,5,6): d^2 = 0,4,1,1,0,0 -> 6
      rho = 1 - 6*6 / (6*(36-1)) = 1 - 36/210
      discordant pairs (B,C),(B,D) of 15 -> tau = (13 - 2) / 15
    Direction over {A,C}: A same, C opposite -> 1/2
    """
    lime = {"A": 6.0, "B": 3.0, "C": -5.0, "D": 4.0, "E": 2.0, "F": 1.0}
    result = _evaluate(DESCENDING, lime)
    at3 = result.at_k[0]

    rho, tau = 1 - 36 / 210, 11 / 15
    a_set = (2 / 3 + 0.5) / 2
    a_rank = ((rho + 1) / 2 + (tau + 1) / 2) / 2
    a_direction = 0.5

    assert at3.shap_top_features == ("A", "B", "C")
    assert at3.lime_top_features == ("A", "C", "D")
    assert at3.intersection == ("A", "C")
    assert at3.overlap_at_k == pytest.approx(2 / 3)
    assert at3.jaccard == pytest.approx(0.5)
    assert at3.feature_set_agreement == pytest.approx(a_set)
    assert result.spearman_raw == pytest.approx(rho)
    assert result.kendall_raw == pytest.approx(tau)
    assert result.rank_agreement == pytest.approx(a_rank)
    assert at3.same_direction_features == ("A",)
    assert at3.different_direction_features == ("C",)
    assert at3.direction_agreement == pytest.approx(a_direction)
    assert result.score == pytest.approx((a_set + a_rank + a_direction) / 3)
    assert 0.0 <= result.score <= 1.0


def test_mixed_directions_fraction() -> None:
    lime = {name: (-v if name in {"B", "C"} else v) for name, v in DESCENDING.items()}
    at3 = _evaluate(DESCENDING, lime).at_k[0]
    assert at3.intersection == ("A", "B", "C")
    assert at3.direction_agreement == pytest.approx(1 / 3)


def test_multiple_k_are_reported_separately() -> None:
    result = _evaluate(DESCENDING, DESCENDING, AgreementConfig())
    assert [r.k for r in result.at_k] == [3, 5]
    assert result.primary_top_k == 5
    assert result.score == result.at_k[1].agreement_score


# --- ties and undefined correlations ------------------------------------------


def test_ties_are_broken_by_canonical_order_and_are_deterministic() -> None:
    tied = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "E": 0.2, "F": 0.1}
    first = _evaluate(tied, DESCENDING)
    second = _evaluate(tied, DESCENDING)
    assert first == second
    assert first.at_k[0].shap_top_features == ("A", "B", "C")
    # tie-aware statistics: agree with SciPy's average-rank / tau-b handling
    names = sorted(tied)
    expected = stats.spearmanr([tied[n] for n in names], [DESCENDING[n] for n in names]).statistic
    assert first.spearman_raw == pytest.approx(expected)


def test_constant_magnitudes_make_rank_agreement_not_computable() -> None:
    constant = {name: 1.0 for name in DESCENDING}
    result = _evaluate(DESCENDING, constant)
    assert result.rank_status == "not_computable"
    assert result.rank_reason and "constant" in result.rank_reason
    assert result.spearman_raw is None and result.rank_agreement is None
    assert result.status == "not_computable" and result.score is None
    # set/direction components are still reported, not hidden
    assert result.at_k[0].feature_set_agreement == 1.0


def test_rank_correlation_needs_two_features() -> None:
    assert not rank_correlation([1.0], [1.0]).computable


def test_normalization_maps_correlation_range() -> None:
    assert normalize_correlation(1.0) == 1.0
    assert normalize_correlation(-1.0) == 0.0
    assert normalize_correlation(0.0) == 0.5


# --- configuration and alignment ---------------------------------------------


def test_k_larger_than_feature_count_fails() -> None:
    with pytest.raises(InvalidMetricConfigurationError):
        _evaluate(DESCENDING, DESCENDING, AgreementConfig(top_k_values=(7,), primary_top_k=7))


@pytest.mark.parametrize(
    ("values", "primary"), [((0,), 0), ((-1, 3), 3), ((3, 3), 3), ((3, 5), 4)]
)
def test_invalid_k_configuration_fails(values: tuple[int, ...], primary: int) -> None:
    with pytest.raises(InvalidMetricConfigurationError):
        AgreementConfig(top_k_values=values, primary_top_k=primary)


@pytest.mark.parametrize(
    "override",
    [
        {"prediction_id": "00000000-0000-0000-0000-0000000000b2"},
        {"feature_vector_id": "00000000-0000-0000-0000-0000000000b3"},
        {"model_version": "other-model"},
        {"target_class": "ide_ai", "target_class_index": 1},
        {"target_class_index": 2},
    ],
)
def test_mismatched_explanations_are_rejected(override: dict) -> None:
    with pytest.raises(ExplanationAlignmentError):
        AgreementEvaluator(K3).evaluate(
            shap_result(DESCENDING), lime_result(DESCENDING, **override)
        )


def test_different_feature_sets_are_rejected() -> None:
    other = {**{k: v for k, v in DESCENDING.items() if k != "F"}, "G": 1.0}
    with pytest.raises(ExplanationAlignmentError, match="different features"):
        AgreementEvaluator(K3).evaluate(shap_result(DESCENDING), lime_result(other))
