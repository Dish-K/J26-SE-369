"""The C4-owned XAI background sample used by SHAP and LIME.

A background is a versioned matrix of model-feature rows (identified by a
``BackgroundReference``). Before any explainer sees it, it is checked against
the exact C2 artifact: same Contract-A schema version, every model feature
present, columns put into ``artifact.feature_names`` order, and no missing
values. No feature is ever fabricated or imputed — the C1/C2 missing-value
policy is not yet defined, so an incomplete background is rejected outright.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.explainability.errors import ExplanationBackgroundMismatchError
from backend.app.explainability.schemas import BackgroundReference, ModelArtifactMetadata


@dataclass(frozen=True)
class XaiBackground:
    reference: BackgroundReference
    frame: pd.DataFrame  # columns exactly the model features, in artifact order

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.frame.columns)

    @property
    def background_id(self) -> str:
        return self.reference.reference_id

    @property
    def background_version(self) -> str:
        return self.reference.dataset_version

    def matrix(self) -> np.ndarray:
        return self.frame.to_numpy(dtype=float, copy=True)


def order_feature_frame(
    frame: pd.DataFrame, feature_names: tuple[str, ...], *, what: str
) -> pd.DataFrame:
    """Select model features by name, in model order; reject gaps and non-numbers.

    Extra columns are dropped (they are not model inputs). Missing columns and
    missing values raise: nothing is filled in.
    """
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ExplanationBackgroundMismatchError(f"{what} lacks model features: {missing}")
    if frame.empty:
        raise ExplanationBackgroundMismatchError(f"{what} has no rows")

    ordered = frame.loc[:, list(feature_names)]
    try:
        ordered = ordered.astype(float)
    except (TypeError, ValueError) as exc:
        raise ExplanationBackgroundMismatchError(f"{what} has non-numeric values: {exc}") from exc

    if not np.isfinite(ordered.to_numpy()).all():
        bad = [name for name in feature_names if not np.isfinite(ordered[name]).all()]
        raise ExplanationBackgroundMismatchError(
            f"{what} has missing or non-finite values in {bad}; no missing-value policy is "
            "defined for the shared feature contract, so they are not imputed"
        )
    return ordered.reset_index(drop=True)


def prepare_background(
    reference: BackgroundReference,
    frame: pd.DataFrame,
    artifact: ModelArtifactMetadata,
) -> XaiBackground:
    if reference.schema_version != artifact.feature_schema_version:
        raise ExplanationBackgroundMismatchError(
            f"background {reference.reference_id!r} uses schema {reference.schema_version!r}, "
            f"model {artifact.model_version!r} uses {artifact.feature_schema_version!r}"
        )
    if (
        reference.compatible_model_version is not None
        and reference.compatible_model_version != artifact.model_version
    ):
        raise ExplanationBackgroundMismatchError(
            f"background {reference.reference_id!r} is declared for model "
            f"{reference.compatible_model_version!r}, not {artifact.model_version!r}"
        )
    if (
        reference.training_dataset_version is not None
        and artifact.training_dataset_version is not None
        and reference.training_dataset_version != artifact.training_dataset_version
    ):
        raise ExplanationBackgroundMismatchError(
            f"background {reference.reference_id!r} was drawn from training data "
            f"{reference.training_dataset_version!r}, model was trained on "
            f"{artifact.training_dataset_version!r}"
        )

    ordered = order_feature_frame(
        frame, artifact.feature_names, what=f"background {reference.reference_id!r}"
    )
    if reference.sample_count is not None and reference.sample_count != len(ordered):
        raise ExplanationBackgroundMismatchError(
            f"background {reference.reference_id!r} declares {reference.sample_count} rows, "
            f"found {len(ordered)}"
        )
    return XaiBackground(reference=reference, frame=ordered)


def load_background(
    reference: BackgroundReference, artifact: ModelArtifactMetadata, *, base_dir: Path
) -> XaiBackground:
    """Read the CSV named by ``reference.uri`` (relative to ``base_dir``)."""
    if not reference.uri:
        raise ExplanationBackgroundMismatchError(
            f"background {reference.reference_id!r} has no uri to load"
        )
    path = Path(reference.uri)
    if not path.is_absolute():
        path = base_dir / path
    if path.suffix.lower() != ".csv":
        raise ExplanationBackgroundMismatchError(f"background must be a .csv file: {path}")
    if not path.is_file():
        raise ExplanationBackgroundMismatchError(f"background file not found: {path}")
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError) as exc:
        raise ExplanationBackgroundMismatchError(f"{path}: unreadable background: {exc}") from exc
    return prepare_background(reference, frame, artifact)
