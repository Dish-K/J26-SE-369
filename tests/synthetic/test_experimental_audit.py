import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ml.training.audit_experimental_outputs import audit
from shared_features import FEATURE_NAMES, REGISTRY
from synthetic import build_dataset


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SavedDatasetAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.dataset = Path(self.temporary.name)
        self.current_hash = digest(build_dataset.__file__)

    def fixture(self, generator_hash):
        # Small schema-valid rows exercise the real audit without fitting a model.
        values = {
            name: np.nan if spec.nullable else max(0, spec.valid_range[0] or 0)
            for name, spec in REGISTRY.items()
        }
        features = pd.DataFrame(
            [values.copy() for _ in range(3)], columns=FEATURE_NAMES
        )
        labels = pd.DataFrame(
            [
                {"row_id": f"row-{i}", "family_id": f"family-{i}", "label": category}
                for i, category in enumerate(["no_ai", "ide_ai", "external_ai"])
            ]
        )
        windows = labels.assign(
            session_id="test-session",
            task_id="test-task",
            window_start="2026-01-01T00:00:00+00:00",
            window_end="2026-01-01T00:00:30+00:00",
            included=True,
            reason="included_single_known_category",
        )
        features.to_csv(self.dataset / "all_features.csv", index=False)
        features.to_csv(self.dataset / "training_features.csv", index=False)
        labels.to_csv(self.dataset / "training_labels.csv", index=False)
        windows.to_csv(self.dataset / "windows_and_labels.csv", index=False)
        raw = self.dataset / "raw/persona/events.csv"
        raw.parent.mkdir(parents=True)
        raw.write_text("kind,origin,inserted_len,removed_len\n")
        manifest = {
            "holdout_examined": False,
            "experimental_training_ready": True,
            "research_validated": False,
            "generator_sha256": generator_hash,
            "feature_names": list(FEATURE_NAMES),
            "all_windows": 3,
            "included_windows": 3,
            "class_counts": {"no_ai": 1, "ide_ai": 1, "external_ai": 1},
            "output_sha256": {
                str(p.relative_to(self.dataset)): digest(p)
                for p in self.dataset.rglob("*")
                if p.is_file()
            },
        }
        (self.dataset / "dataset_manifest.json").write_text(json.dumps(manifest))

    def test_reused_dataset_accepts_different_current_source_and_preserves_manifest(
        self,
    ):
        self.fixture("0" * 64)
        original_manifest = (self.dataset / "dataset_manifest.json").read_bytes()
        result = audit(self.dataset)
        self.assertEqual(result["audit"], "PASSED")
        self.assertTrue(result["dataset_file_hashes_verified"])
        self.assertFalse(result["generator_source_matches"])
        self.assertEqual(result["dataset_generator_sha256"], "0" * 64)
        self.assertEqual(result["current_generator_sha256"], self.current_hash)
        self.assertEqual(
            (self.dataset / "dataset_manifest.json").read_bytes(), original_manifest
        )

    def test_changed_dataset_file_is_still_rejected_when_source_differs(self):
        self.fixture("0" * 64)
        with (self.dataset / "raw/persona/events.csv").open("a") as stream:
            stream.write("\n")
        with self.assertRaises(AssertionError):
            audit(self.dataset)

    def test_explicit_exact_source_check_rejects_a_different_generator(self):
        self.fixture("0" * 64)
        with self.assertRaisesRegex(ValueError, "Exact source identity"):
            audit(self.dataset, require_generator_match=True)

    def test_explicit_exact_source_check_accepts_the_same_generator(self):
        self.fixture(self.current_hash)
        result = audit(self.dataset, require_generator_match=True)
        self.assertTrue(result["generator_source_matches"])
        self.assertTrue(result["exact_generator_identity_required"])


if __name__ == "__main__":
    unittest.main()
