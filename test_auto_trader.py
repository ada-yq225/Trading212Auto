import math
import unittest
from datetime import datetime, timezone

from auto_trader import (
    Config,
    SignalMetrics,
    backoff_seconds,
    floor_quantity,
    gross_exposure_for_regime,
    is_regular_market_time,
    load_active_universe,
    load_universe_config,
    market_regime,
    momentum_metrics,
    sample_volatility,
    target_weights,
)


def metrics(score: float, long_return: float, volatility: float) -> SignalMetrics:
    return SignalMetrics(
        score=score,
        short_return=long_return / 3,
        medium_return=long_return / 2,
        long_return=long_return,
        volatility=volatility,
        consistency=0.5 if score > 0 else -0.5,
    )


class StrategyTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(
            short_samples=5,
            medium_samples=10,
            long_samples=20,
            volatility_samples=10,
            top_n=3,
        )

    def test_momentum_metrics_warms_up(self):
        self.assertIsNone(momentum_metrics([100.0] * 20, self.config))

    def test_momentum_metrics_scores_persistent_trend(self):
        rising = [100 * math.exp(0.002 * index) for index in range(40)]
        falling = list(reversed(rising))
        self.assertGreater(momentum_metrics(rising, self.config).score, 0.75)
        self.assertLess(momentum_metrics(falling, self.config).score, -0.75)

    def test_volatility_detects_unstable_path(self):
        smooth = [100 * math.exp(0.001 * index) for index in range(30)]
        noisy = [100 * math.exp(0.001 * index + (0.02 if index % 2 else -0.02)) for index in range(30)]
        self.assertGreater(sample_volatility(noisy, 20), sample_volatility(smooth, 20))

    def test_market_regime_uses_cross_sectional_median(self):
        bullish = {
            "A": metrics(1.0, 0.01, 0.01),
            "B": metrics(1.0, 0.02, 0.01),
            "C": metrics(-1.0, -0.01, 0.01),
        }
        self.assertEqual(market_regime(bullish, self.config)[0], "RISK_ON")
        bearish = {key: metrics(-1.0, -0.02, 0.01) for key in "ABC"}
        self.assertEqual(market_regime(bearish, self.config)[0], "RISK_OFF")

    def test_drawdown_guard_reduces_exposure(self):
        self.assertEqual(
            gross_exposure_for_regime("RISK_ON", 0.0, self.config),
            self.config.risk_on_gross_pct,
        )
        self.assertEqual(
            gross_exposure_for_regime("RISK_ON", 0.09, self.config),
            0.40,
        )
        self.assertEqual(
            gross_exposure_for_regime("RISK_ON", 0.13, self.config),
            0.15,
        )

    def test_target_weights_enforce_stock_and_sector_caps(self):
        signals = {
            "A": metrics(2.0, 0.03, 0.010),
            "B": metrics(1.8, 0.03, 0.012),
            "C": metrics(1.5, 0.02, 0.009),
            "D": metrics(0.2, 0.01, 0.008),
        }
        sectors = {"A": "Tech", "B": "Tech", "C": "Health", "D": "Energy"}
        weights = target_weights(signals, sectors, 0.95, self.config)
        self.assertNotIn("D", weights)
        self.assertTrue(all(weight <= self.config.max_position_pct for weight in weights.values()))
        tech_weight = sum(weights.get(ticker, 0) for ticker in ("A", "B"))
        self.assertLessEqual(tech_weight, self.config.max_sector_pct)
        self.assertLessEqual(sum(weights.values()), 0.95)

    def test_us_market_window(self):
        open_time = datetime(2026, 7, 20, 15, 0, tzinfo=timezone.utc)
        closed_time = datetime(2026, 7, 20, 22, 0, tzinfo=timezone.utc)
        self.assertTrue(is_regular_market_time("AAPL_US_EQ", open_time))
        self.assertFalse(is_regular_market_time("AAPL_US_EQ", closed_time))

    def test_zero_daily_limit_means_unlimited_configuration(self):
        self.assertEqual(Config(max_orders_per_day=0).max_orders_per_day, 0)

    def test_quantity_is_floored_to_three_decimals(self):
        self.assertEqual(floor_quantity(3.96426964), 3.964)
        self.assertEqual(floor_quantity(0.0009), 0.0)

    def test_high_risk_universe_is_loaded(self):
        universe = load_active_universe()
        self.assertIn("TSLA_US_EQ", universe)
        self.assertIn("MSTR_US_EQ", universe)
        self.assertNotIn("KO_US_EQ", universe)

    def test_scout_universe_covers_fifty_stocks(self):
        data = load_universe_config()
        scouts = data.get("scouts", [])
        sectors = data.get("sectors", {})
        self.assertEqual(len(scouts), 42)
        self.assertEqual(len(sectors), 11)
        self.assertEqual(sum(len(items) for items in sectors.values()), 50)
        self.assertEqual(data.get("scout_seed_interval_seconds"), 900)
        self.assertEqual(data.get("scout_min_samples"), 241)
        self.assertTrue(all(0 < item["probe_quantity"] < 0.1 for item in scouts))

    def test_api_backoff_is_exponential_and_capped(self):
        self.assertEqual(backoff_seconds(15, 1), 30)
        self.assertEqual(backoff_seconds(15, 3), 120)
        self.assertEqual(backoff_seconds(15, 20), 300)


if __name__ == "__main__":
    unittest.main()
