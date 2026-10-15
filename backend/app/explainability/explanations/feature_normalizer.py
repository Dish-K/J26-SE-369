"""Map explainer feature labels back to canonical Contract-A feature names.

The primary mechanism is the structured feature index an explainer reports
(LIME's ``as_map``): index i is ``artifact.feature_names[i]``. Display strings
such as ``"0.45 < paste_ratio <= 0.72"`` are parsed only as a fallback and as
a cross-check. The parser matches whole feature names inside the complete
condition grammar, so ``paste_ratio`` can never match ``paste_ratio_prev1``.
"""

import re
from collections.abc import Sequence

from backend.app.explainability.errors import CanonicalFeatureMappingError

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_OP = r"(?:<=|>=|<|>)"


def feature_name_from_index(index: int, feature_names: Sequence[str]) -> str:
    if isinstance(index, bool) or not isinstance(index, int):
        raise CanonicalFeatureMappingError(f"feature index must be an int, got {index!r}")
    if not 0 <= index < len(feature_names):
        raise CanonicalFeatureMappingError(
            f"feature index {index} outside 0..{len(feature_names) - 1}"
        )
    return feature_names[index]


def _condition_patterns(name: str) -> tuple[re.Pattern[str], ...]:
    escaped = re.escape(name)
    return (
        re.compile(rf"^\s*{escaped}\s*$"),
        re.compile(rf"^\s*{escaped}\s*{_OP}\s*{_NUMBER}\s*$"),
        re.compile(rf"^\s*{_NUMBER}\s*{_OP}\s*{escaped}\s*$"),
        re.compile(rf"^\s*{_NUMBER}\s*{_OP}\s*{escaped}\s*{_OP}\s*{_NUMBER}\s*$"),
    )


def feature_name_from_condition(condition: str, feature_names: Sequence[str]) -> str:
    """Return the single model feature a LIME-style condition refers to."""
    matches = [
        name
        for name in feature_names
        if any(pattern.match(condition) for pattern in _condition_patterns(name))
    ]
    if not matches:
        raise CanonicalFeatureMappingError(
            f"condition {condition!r} does not refer to any model feature"
        )
    if len(matches) > 1:
        raise CanonicalFeatureMappingError(
            f"condition {condition!r} is ambiguous between features {matches}"
        )
    return matches[0]


def canonical_feature_name(
    index: int, condition: str | None, feature_names: Sequence[str]
) -> str:
    """Name from the structured index, cross-checked against the display condition."""
    name = feature_name_from_index(index, feature_names)
    if condition is not None:
        parsed = feature_name_from_condition(condition, feature_names)
        if parsed != name:
            raise CanonicalFeatureMappingError(
                f"feature index {index} is {name!r} but its condition {condition!r} "
                f"names {parsed!r}"
            )
    return name
