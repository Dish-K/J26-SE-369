import pytest

from backend.app.explainability.errors import CanonicalFeatureMappingError
from backend.app.explainability.explanations.feature_normalizer import (
    canonical_feature_name,
    feature_name_from_condition,
    feature_name_from_index,
)
from tests.explainability.fixtures.prototype_c2 import FEATURE_NAMES


# --- primary mechanism: structured feature index -----------------------------


def test_index_maps_to_artifact_feature_name() -> None:
    for index, name in enumerate(FEATURE_NAMES):
        assert feature_name_from_index(index, FEATURE_NAMES) == name


@pytest.mark.parametrize("index", [-1, len(FEATURE_NAMES), 2.0, True, "3"])
def test_invalid_index_fails(index: object) -> None:
    with pytest.raises(CanonicalFeatureMappingError):
        feature_name_from_index(index, FEATURE_NAMES)  # type: ignore[arg-type]


# --- fallback / cross-check: display conditions ------------------------------


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        ("paste_ratio > 0.72", "paste_ratio"),
        ("paste_ratio <= 0.25", "paste_ratio"),
        ("0.25 < paste_ratio <= 0.72", "paste_ratio"),
        ("-10.0 < pause_ms <= 500.0", "pause_ms"),
        ("backspace_rate <= 0.03", "backspace_rate"),
        ("pause_ms >= 1e4", "pause_ms"),
        ("  focus_switch_count>1.46 ", "focus_switch_count"),
        ("keystroke_count", "keystroke_count"),
    ],
)
def test_condition_maps_to_feature(condition: str, expected: str) -> None:
    assert feature_name_from_condition(condition, FEATURE_NAMES) == expected


def test_overlapping_names_never_match_by_substring() -> None:
    names = ("paste_ratio", "paste_ratio_prev1", "ratio")
    assert feature_name_from_condition("paste_ratio > 0.5", names) == "paste_ratio"
    assert feature_name_from_condition("paste_ratio_prev1 > 0.5", names) == "paste_ratio_prev1"
    assert feature_name_from_condition("0.1 < ratio <= 0.2", names) == "ratio"


@pytest.mark.parametrize("condition", ["typing_speed > 3", "paste > 0.5", "0.2 < = paste_ratio", ""])
def test_unknown_or_malformed_condition_fails(condition: str) -> None:
    with pytest.raises(CanonicalFeatureMappingError):
        feature_name_from_condition(condition, FEATURE_NAMES)


def test_ambiguous_condition_fails() -> None:
    names = ("a", "a <= 1")  # the condition text is itself a feature name
    with pytest.raises(CanonicalFeatureMappingError, match="ambiguous"):
        feature_name_from_condition("a <= 1", names)


def test_index_and_condition_must_agree() -> None:
    index = FEATURE_NAMES.index("paste_ratio")
    assert canonical_feature_name(index, "paste_ratio > 0.38", FEATURE_NAMES) == "paste_ratio"
    with pytest.raises(CanonicalFeatureMappingError, match="names 'pause_ms'"):
        canonical_feature_name(index, "pause_ms > 1.0", FEATURE_NAMES)
