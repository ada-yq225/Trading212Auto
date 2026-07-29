import math
import unittest

from experimental_model import (
    ExperimentSettings,
    ExperimentalEnsemble,
    causal_features,
    create_prediction_batch,
    gate_statistics,
    resolve_shadow_predictions,
    supervised_samples,
)


def synthetic_histories(length: int = 300) -> dict[str, list[float]]:
    return {
        f"STOCK_{stock}": [
            100
            * math.exp(
                (0.0001 + stock * 0.00003) * index
                + 0.004 * math.sin(index / (7 + stock))
            )
            for index in range(length)
        ]
        for stock in range(8)
    }


class ExperimentalModelTests(unittest.TestCase):
    def setUp(self):
        self.settings = ExperimentSettings(
            minimum_training_samples=80,
            minimum_shadow_outcomes=4,
            minimum_shadow_batches=1,
            minimum_hit_rate=0.5,
        )

    def test_features_do_not_read_future_values(self):
        values = synthetic_histories()["STOCK_0"]
        baseline = causal_features(values, 150)
        changed = list(values)
        changed[151:] = [value * 10 for value in changed[151:]]
        self.assertEqual(baseline, causal_features(changed, 150))

    def test_supervised_samples_use_delayed_labels(self):
        features, labels = supervised_samples(synthetic_histories(), self.settings)
        self.assertGreater(len(labels), self.settings.minimum_training_samples)
        self.assertEqual(features.shape[1], 12)
        self.assertTrue(all(-5 <= value <= 5 for value in labels))

    def test_ensemble_fits_and_cross_sectionally_predicts(self):
        histories = synthetic_histories()
        model = ExperimentalEnsemble(self.settings)
        self.assertGreater(model.fit(histories), self.settings.minimum_training_samples)
        predictions = model.predict(histories)
        self.assertEqual(set(predictions), set(histories))
        self.assertAlmostEqual(sum(predictions.values()), 0.0, places=6)

    def test_shadow_gate_requires_forward_outcomes(self):
        state = {
            "outcomes": [
                {
                    "createdAt": 1000.0,
                    "selected": True,
                    "hit": True,
                    "netReturn": 0.002,
                }
                for _ in range(4)
            ]
        }
        result = gate_statistics(state, self.settings)
        self.assertTrue(result["approved"])
        self.assertEqual(result["outcomeCount"], 4)
        self.assertEqual(result["batchCount"], 1)

    def test_prediction_batch_is_resolved_after_horizon(self):
        state = {}
        created = create_prediction_batch(
            state,
            {"A": 1.0, "B": -1.0},
            {"A": 100.0, "B": 100.0},
            now=1000.0,
            settings=self.settings,
        )
        self.assertTrue(created)
        resolve_shadow_predictions(
            state,
            {"A": 102.0, "B": 99.0},
            now=2000.0,
            settings=self.settings,
        )
        self.assertEqual(len(state["pending"]), 0)
        self.assertEqual(len(state["outcomes"]), 2)
        selected = [item for item in state["outcomes"] if item["selected"]]
        self.assertEqual(len(selected), 1)
        self.assertGreater(selected[0]["netReturn"], 0)

    def test_experiment_settings_define_sample_interval(self):
        settings = ExperimentSettings()
        self.assertTrue(hasattr(settings, "sample_interval_seconds"))
        self.assertEqual(settings.sample_interval_seconds, 60.0)

    def test_shadow_resolution_uses_configured_sample_interval(self):
        settings = ExperimentSettings(
            horizon_samples=15,
            sample_interval_seconds=15.0,
            prediction_interval_seconds=0,
        )
        state = {}
        create_prediction_batch(
            state,
            {"A": 1.0, "B": -1.0},
            {"A": 100.0, "B": 100.0},
            now=1000.0,
            settings=settings,
        )

        resolve_shadow_predictions(
            state,
            {"A": 101.0, "B": 99.0},
            now=1224.999,
            settings=settings,
        )
        self.assertEqual(len(state["pending"]), 2)
        self.assertEqual(len(state["outcomes"]), 0)

        resolve_shadow_predictions(
            state,
            {"A": 101.0, "B": 99.0},
            now=1225.0,
            settings=settings,
        )
        self.assertEqual(len(state["pending"]), 0)
        self.assertEqual(len(state["outcomes"]), 2)

    def test_shadow_resolution_defaults_to_one_minute_samples(self):
        settings = ExperimentSettings(
            horizon_samples=15,
            prediction_interval_seconds=0,
        )
        state = {}
        create_prediction_batch(
            state,
            {"A": 1.0},
            {"A": 100.0},
            now=1000.0,
            settings=settings,
        )

        resolve_shadow_predictions(
            state,
            {"A": 101.0},
            now=1899.999,
            settings=settings,
        )
        self.assertEqual(len(state["pending"]), 1)

        resolve_shadow_predictions(
            state,
            {"A": 101.0},
            now=1900.0,
            settings=settings,
        )
        self.assertEqual(len(state["outcomes"]), 1)


if __name__ == "__main__":
    unittest.main()
