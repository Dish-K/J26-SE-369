from pathlib import Path
from typing import Any

import numpy as np
import pytest

from backend.app.explainability.errors import ExplanationBackgroundMismatchError
from backend.app.explainability.explanations.background import (
    XaiBackground,
    load_background,
    prepare_background,
)
from backend.app.explainability.model.artifact_loader import ArtifactLoader
from backend.app.explainability.schemas import ModelArtifactMetadata
from tests.explainability.fixtures.prototype_c2 import (
    BACKGROUND_VERSION,
    FEATURE_NAMES,
    MODEL_VERSION,
    background_frame,
    background_reference,
)


@pytest.fixture
def metadata(artifact_dir: Path) -> ModelArtifactMetadata:
    return ArtifactLoader(artifact_dir).load(MODEL_VERSION).metadata


def test_valid_background_loads_with_metadata(background: XaiBackground) -> None:
    assert background.feature_names == FEATURE_NAMES
    assert background.background_version == BACKGROUND_VERSION
    assert background.reference.split == "training_reference"
    assert background.matrix().shape == (len(background_frame()), len(FEATURE_NAMES))


def test_columns_are_reordered_by_artifact_and_extras_dropped(
    metadata: ModelArtifactMetadata,
) -> None:
    frame = background_frame()
    shuffled = frame[list(reversed(FEATURE_NAMES))].assign(min_line=7)
    prepared = prepare_background(background_reference(), shuffled, metadata)
    assert prepared.feature_names == FEATURE_NAMES
    np.testing.assert_array_equal(prepared.matrix(), frame.to_numpy(dtype=float))


@pytest.mark.parametrize(
    "override",
    [
        {"schema_version": "contract-a-v2"},
        {"compatible_model_version": "synthetic-prototype-v1"},
        {"training_dataset_version": "other-training-set"},
        {"sample_count": 3},
    ],
)
def test_incompatible_metadata_fails(
    metadata: ModelArtifactMetadata, override: dict[str, Any]
) -> None:
    with pytest.raises(ExplanationBackgroundMismatchError):
        prepare_background(background_reference(**override), background_frame(), metadata)


def test_missing_model_feature_fails(metadata: ModelArtifactMetadata) -> None:
    frame = background_frame().drop(columns=["paste_ratio"])
    with pytest.raises(ExplanationBackgroundMismatchError, match="paste_ratio"):
        prepare_background(background_reference(), frame, metadata)


def test_missing_values_are_rejected_not_filled(metadata: ModelArtifactMetadata) -> None:
    frame = background_frame()
    frame.loc[0, "pause_ms"] = np.nan
    with pytest.raises(ExplanationBackgroundMismatchError, match="pause_ms"):
        prepare_background(background_reference(), frame, metadata)


def test_non_numeric_values_fail(metadata: ModelArtifactMetadata) -> None:
    frame = background_frame().astype({"pause_ms": object})
    frame.loc[0, "pause_ms"] = "long"
    with pytest.raises(ExplanationBackgroundMismatchError):
        prepare_background(background_reference(), frame, metadata)


def test_empty_background_fails(metadata: ModelArtifactMetadata) -> None:
    with pytest.raises(ExplanationBackgroundMismatchError):
        prepare_background(
            background_reference(sample_count=None), background_frame().iloc[:0], metadata
        )


@pytest.mark.parametrize(
    ("override", "match"),
    [({"uri": None}, "no uri"), ({"uri": "absent.csv"}, "not found"), ({"uri": "bg.parquet"}, "csv")],
)
def test_load_failures(
    metadata: ModelArtifactMetadata, background_dir: Path, override: dict[str, Any], match: str
) -> None:
    with pytest.raises(ExplanationBackgroundMismatchError, match=match):
        load_background(background_reference(**override), metadata, base_dir=background_dir)
