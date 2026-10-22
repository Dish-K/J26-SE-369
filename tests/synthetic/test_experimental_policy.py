import copy, json, random, unittest
from pathlib import Path
import numpy as np
import pandas as pd
from synthetic.build_dataset import (
    Document,
    label_window,
    load_cache,
    simulate_persona,
    CLASSES,
)
from ml.training.train_experimental import (
    split_groups,
    probability_metrics,
    threshold_search,
)


class PolicyTests(unittest.TestCase):
    def event(self, category="no_ai", time="2026-01-01T00:00:01Z", uncertain=False):
        return {
            "time": time,
            "code_activity": True,
            "categories": [category] if category else [],
            "uncertain": uncertain,
        }

    def bounds(self):
        return pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp(
            "2026-01-01T00:00:30Z"
        )

    def test_same_raw_paste_can_have_different_known_origins(self):
        # No raw event kind is consulted by the labeling function.
        a, b = self.bounds()
        self.assertEqual(label_window([self.event("no_ai")], a, b)[0], "no_ai")
        self.assertEqual(
            label_window([self.event("external_ai")], a, b)[0], "external_ai"
        )

    def test_mixed_excluded(self):
        a, b = self.bounds()
        label, reason, _, cats = label_window(
            [self.event(), self.event("ide_ai")], a, b
        )
        self.assertIsNone(label)
        self.assertEqual(reason, "mixed_contribution_categories")
        self.assertEqual(len(cats), 2)

    def test_uncertain_excluded(self):
        a, b = self.bounds()
        self.assertEqual(
            label_window([self.event("ide_ai", uncertain=True)], a, b)[1],
            "uncertain_provenance",
        )

    def test_no_contributions_are_insufficient(self):
        a, b = self.bounds()
        self.assertEqual(label_window([], a, b)[1], "insufficient_activity")

    def test_half_open_windows(self):
        a, b = self.bounds()
        self.assertEqual(
            label_window([self.event(time=b.isoformat())], a, b)[1],
            "insufficient_activity",
        )

    def test_deletion_lineage_detects_cross_category_edit(self):
        d = Document(4)
        d.apply(5, 0, "ide_ai", "t")
        d.apply(3, 0, "no_ai", "t")
        _, _, origins = d.apply(0, 6, None, "t")
        self.assertEqual(origins, {"no_ai", "ide_ai"})

    def test_deleting_baseline_is_uncertain(self):
        d = Document(4)
        d.apply(2, 0, "no_ai", "t")
        _, _, origins = d.apply(0, 4, None, "t")
        self.assertEqual(origins, {"no_ai", None})

    def test_non_alphabetic_probability_order_is_correct(self):
        p = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
        q = probability_metrics(np.array(CLASSES), p)
        self.assertAlmostEqual(q["log_loss"], -np.mean(np.log([0.8, 0.7, 0.7])))

    def test_threshold_uses_winner_and_does_not_convert_class(self):
        y = np.array(["no_ai", "ide_ai", "external_ai"])
        p = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        best, _ = threshold_search(y, p, [0.5, 0.8, 0.9])
        self.assertEqual(best["threshold"], 0.8)

    def test_group_split_keeps_all_category_variants_together(self):
        rows = [
            {"family_id": f"p{i}", "row_id": f"{i}-{c}", "label": c}
            for i in range(20)
            for c in CLASSES
        ]
        d = pd.DataFrame(rows)
        idx, parts = split_groups(d, 4, [0.7, 0.15, 0.15])
        self.assertFalse(parts["train"] & parts["test"])
        self.assertFalse(parts["validation"] & parts["test"])
        self.assertEqual(sum(len(x) for x in idx.values()), len(d))
        for family, g in d.groupby("family_id"):
            self.assertEqual(sum(family in s for s in parts.values()), 1)

    def test_missing_class_fails_instead_of_relabeling(self):
        d = pd.DataFrame([{"family_id": f"p{i}", "label": "no_ai"} for i in range(20)])
        with self.assertRaisesRegex(ValueError, "missing class"):
            split_groups(d, 4, [0.7, 0.15, 0.15])


if __name__ == "__main__":
    unittest.main(verbosity=2)
