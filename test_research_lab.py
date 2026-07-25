import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from research_lab.config import ResearchConfig
from research_lab.features import asset_features, build_panel
from research_lab.models import _split_nested
from research_lab.portfolio import optimize_portfolio
from research_lab.statistics import block_bootstrap, performance_metrics


def synthetic_frame(
    *,
    days: int = 700,
    drift: float = 0.0003,
    phase: float = 0.0,
) -> pd.DataFrame:
    index = pd.bdate_range("2019-01-01", periods=days)
    path = np.asarray(
        [
            100
            * np.exp(
                drift * day
                + 0.02 * np.sin(day / 17 + phase)
                + 0.01 * np.sin(day / 5 + phase)
            )
            for day in range(days)
        ]
    )
    return pd.DataFrame(
        {
            "open": path * 0.998,
            "high": path * 1.012,
            "low": path * 0.988,
            "close": path,
            "adjusted_close": path,
            "volume": 1_000_000
            * (1 + 0.2 * np.sin(np.arange(days) / 11 + phase)),
        },
        index=index,
    )


class ResearchLabTests(unittest.TestCase):
    def setUp(self):
        self.config = ResearchConfig(
            inner_validation_days=80,
            purge_days=21,
            embargo_days=21,
            block_bootstrap_samples=50,
        )

    def test_asset_features_are_causal(self):
        frame = synthetic_frame()
        original = asset_features(frame, self.config)
        cutoff = frame.index[400]
        changed = frame.copy()
        changed.loc[changed.index > cutoff, "adjusted_close"] *= 5
        changed.loc[changed.index > cutoff, "close"] *= 5
        modified = asset_features(changed, self.config)
        feature_columns = [
            column
            for column in original.columns
            if column not in ("future_return", "target", "label_end_date")
        ]
        pd.testing.assert_series_equal(
            original.loc[cutoff, feature_columns],
            modified.loc[cutoff, feature_columns],
        )

    def test_purge_split_separates_training_and_validation(self):
        frames = {
            f"S{index}": synthetic_frame(phase=index)
            for index in range(8)
        }
        sectors = {symbol: "Tech" for symbol in frames}
        panel, _ = build_panel(frames, list(frames), sectors, self.config)
        prediction_date = panel["date"].max()
        train, validation, matured = _split_nested(
            panel,
            prediction_date,
            self.config,
        )
        self.assertGreater(len(matured), 0)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(validation), 0)
        self.assertLess(train["label_end_date"].max(), validation["date"].min())
        all_dates = [pd.Timestamp(date) for date in sorted(panel["date"].unique())]
        training_end_position = all_dates.index(train["label_end_date"].max())
        validation_start_position = all_dates.index(validation["date"].min())
        self.assertGreaterEqual(
            validation_start_position - training_end_position,
            self.config.purge_days + self.config.embargo_days,
        )

    def test_optimizer_enforces_position_and_sector_caps(self):
        symbols = ["A", "B", "C", "D"]
        frames = {
            symbol: synthetic_frame(phase=index)
            for index, symbol in enumerate(symbols)
        }
        sectors = {"A": "Tech", "B": "Tech", "C": "Health", "D": "Energy"}
        config = replace(
            self.config,
            top_n=4,
            maximum_position_weight=0.30,
            maximum_sector_weight=0.40,
        )
        date = frames["A"].index[-1]
        decision = optimize_portfolio(
            {"A": 2.0, "B": 1.5, "C": 1.0, "D": 0.5},
            frames,
            sectors,
            date,
            {},
            0.0,
            config,
        )
        self.assertTrue(
            all(
                weight <= config.maximum_position_weight + 1e-6
                for weight in decision.weights.values()
            )
        )
        tech = decision.weights.get("A", 0) + decision.weights.get("B", 0)
        self.assertLessEqual(tech, config.maximum_sector_weight + 1e-6)
        self.assertLessEqual(sum(decision.weights.values()), decision.gross_target + 1e-6)

    def test_statistics_and_bootstrap_are_deterministic(self):
        index = pd.bdate_range("2023-01-01", periods=300)
        strategy = pd.Series(
            0.0005 + 0.01 * np.sin(np.arange(300)),
            index=index,
        )
        benchmark = pd.Series(
            0.0003 + 0.009 * np.cos(np.arange(300)),
            index=index,
        )
        metrics = performance_metrics(strategy, model_trials=8)
        self.assertIn("deflatedSharpeProbability", metrics)
        first = block_bootstrap(
            strategy,
            benchmark,
            samples=50,
            block_length=21,
            seed=212,
        )
        second = block_bootstrap(
            strategy,
            benchmark,
            samples=50,
            block_length=21,
            seed=212,
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
