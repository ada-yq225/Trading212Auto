import importlib
import importlib.util
import math
import unittest

from experimental_model import (
    ExperimentSettings,
    build_shared_dataset,
    current_feature_matrix,
)
from test_experimental_model import synthetic_histories


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.settings = ExperimentSettings(minimum_training_samples=80)
        self.histories = synthetic_histories()
        self.dataset = build_shared_dataset(
            self.histories,
            self.settings,
        )
        self.tickers, self.matrix = current_feature_matrix(self.histories)

    def test_all_candidates_are_finite_and_deterministic(self):
        self.assertIsNotNone(
            importlib.util.find_spec("experimental_candidates")
        )
        module = importlib.import_module("experimental_candidates")
        candidate_ids = module.CANDIDATE_IDS
        self.assertEqual(
            candidate_ids,
            (
                "legacy_ensemble",
                "robust_huber",
                "regime_histgb",
                "residual_momentum",
            ),
        )
        for candidate_id in candidate_ids:
            first = module.create_candidate(candidate_id, self.settings)
            second = module.create_candidate(candidate_id, self.settings)
            self.assertGreaterEqual(
                first.fit(self.dataset),
                self.settings.minimum_training_samples,
            )
            second.fit(self.dataset)
            left = first.predict(self.tickers, self.matrix)
            right = second.predict(self.tickers, self.matrix)
            self.assertEqual(set(left), set(self.tickers))
            self.assertEqual(left, right)
            self.assertTrue(
                all(math.isfinite(value) for value in left.values())
            )

    def test_unknown_candidate_is_rejected(self):
        self.assertIsNotNone(
            importlib.util.find_spec("experimental_candidates")
        )
        module = importlib.import_module("experimental_candidates")
        with self.assertRaisesRegex(ValueError, "未知候选"):
            module.create_candidate("unknown", self.settings)


if __name__ == "__main__":
    unittest.main()
