import json
import math
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import auto_trader
from auto_trader import (
    Config,
    Position,
    ROOT,
    Runner,
    STRATEGY_VERSION,
    SignalMetrics,
    backoff_seconds,
    floor_quantity,
    gross_exposure_for_regime,
    is_regular_market_time,
    load_active_universe,
    load_universe_config,
    market_regime,
    migrate_strategy_state,
    momentum_metrics,
    sample_volatility,
    target_weights,
)
from experimental_tournament import TournamentResult, migrate_tournament_state


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

    def test_accelerated_strategy_version(self):
        self.assertEqual(
            STRATEGY_VERSION,
            "rational_momentum_ml_v4_fast",
        )

    def test_strategy_migration_resets_interval_dependent_state_once(self):
        state = {
            "strategyVersion": "rational_momentum_ml_v3",
            "priceHistory": {"AMD_US_EQ": [1.0, 2.0]},
            "pricePeaks": {"AMD_US_EQ": 2.0},
            "experimental": {"trainingSamples": 17},
            "lastRebalance": 123.0,
            "portfolioHighWatermark": 5000.0,
            "ordersToday": 17,
            "scoutAttempts": ["AVGO_US_EQ"],
            "promotedScouts": ["TSM_US_EQ"],
        }

        changed = migrate_strategy_state(state, STRATEGY_VERSION)

        self.assertTrue(changed)
        self.assertEqual(state["strategyVersion"], STRATEGY_VERSION)
        self.assertEqual(state["priceHistory"], {})
        self.assertEqual(state["pricePeaks"], {})
        self.assertEqual(state["experimental"], {})
        self.assertEqual(state["lastRebalance"], 0)
        self.assertNotIn("portfolioHighWatermark", state)
        self.assertEqual(state["ordersToday"], 17)
        self.assertEqual(state["scoutAttempts"], ["AVGO_US_EQ"])
        self.assertEqual(state["promotedScouts"], ["TSM_US_EQ"])

        self.assertFalse(migrate_strategy_state(state, STRATEGY_VERSION))

    def test_repository_uses_accelerated_demo_profile(self):
        config = Config.load(ROOT / "strategy.json")
        universe = json.loads(
            (ROOT / "universe.json").read_text(encoding="utf-8")
        )
        self.assertTrue(
            hasattr(config, "experimental_prediction_interval_seconds")
        )
        self.assertEqual(config.poll_seconds, 15)
        self.assertEqual(config.short_samples, 20)
        self.assertEqual(config.medium_samples, 60)
        self.assertEqual(config.long_samples, 180)
        self.assertEqual(config.volatility_samples, 120)
        self.assertEqual(config.rebalance_seconds, 300)
        self.assertEqual(config.cooldown_seconds, 300)
        self.assertEqual(config.experimental_minimum_history, 120)
        self.assertEqual(config.experimental_minimum_training_samples, 120)
        self.assertEqual(config.experimental_minimum_shadow_batches, 20)
        self.assertEqual(config.experimental_minimum_shadow_outcomes, 40)
        self.assertEqual(config.experimental_minimum_hit_rate, 0.53)
        self.assertEqual(config.experimental_prediction_interval_seconds, 225)
        self.assertEqual(config.experimental_compute_budget_seconds, 8)
        self.assertEqual(
            config.experimental_candidate_ids,
            (
                "legacy_ensemble",
                "robust_huber",
                "regime_histgb",
                "residual_momentum",
            ),
        )
        self.assertEqual(config.experimental_early_rejection_batches, 10)
        self.assertEqual(config.experimental_early_rejection_outcomes, 20)
        self.assertEqual(universe["scout_seed_interval_seconds"], 300)
        self.assertEqual(universe["scout_interval_seconds"], 300)

    def test_experimental_prediction_batches_cannot_overlap(self):
        self.assertTrue(
            hasattr(Config, "experimental_prediction_interval_seconds")
        )
        data = json.loads(
            (ROOT / "strategy.json").read_text(encoding="utf-8")
        )
        data["experimental_prediction_interval_seconds"] = 224
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "strategy.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "不能短于"):
                Config.load(path)

    def test_experimental_settings_use_runner_sampling_interval(self):
        config = Config(
            poll_seconds=15,
            rebalance_seconds=300,
            experimental_horizon_samples=15,
        )

        self.assertTrue(
            hasattr(auto_trader, "experiment_settings_from_config")
        )
        settings = auto_trader.experiment_settings_from_config(config)

        self.assertEqual(settings.sample_interval_seconds, 15)
        self.assertEqual(settings.horizon_samples, 15)
        self.assertEqual(
            settings.prediction_interval_seconds,
            config.experimental_prediction_interval_seconds,
        )
        self.assertEqual(
            settings.minimum_shadow_batches,
            config.experimental_minimum_shadow_batches,
        )
        self.assertEqual(
            settings.minimum_shadow_outcomes,
            config.experimental_minimum_shadow_outcomes,
        )

    def test_tournament_migration_does_not_clear_price_history(self):
        state = {
            "strategyVersion": STRATEGY_VERSION,
            "priceHistory": {"AMD_US_EQ": [1.0, 2.0]},
            "experimental": {"trainingSamples": 150},
        }

        migrate_tournament_state(
            state["experimental"],
            Config().experimental_candidate_ids,
        )

        self.assertEqual(
            state["priceHistory"]["AMD_US_EQ"],
            [1.0, 2.0],
        )
        self.assertEqual(
            state["experimental"]["candidates"]["legacy_ensemble"][
                "trainingSamples"
            ],
            150,
        )

    def test_runner_passes_only_in_memory_data_to_tournament(self):
        class FakeTournament:
            def __init__(self):
                self.call = None

            def update(self, **kwargs):
                self.call = kwargs
                return TournamentResult(
                    predictions={"AMD_US_EQ": 0.4},
                    diagnostics={
                        "approved": False,
                        "champion": None,
                        "candidates": {},
                    },
                    events=(
                        {
                            "event": "CANDIDATE_TEST",
                            "candidate": "legacy_ensemble",
                        },
                    ),
                )

        runner = object.__new__(Runner)
        runner.config = Config()
        runner.state = {
            "experimental": {},
            "priceHistory": {
                "AMD_US_EQ": [100.0] * 121,
                "IGNORED_US_EQ": [50.0] * 121,
            },
        }
        runner.price_universe = {"AMD_US_EQ"}
        runner.active_universe = {"AMD_US_EQ"}
        runner.experimental_tournament = FakeTournament()
        position = Position(
            ticker="AMD_US_EQ",
            quantity=1.0,
            available=1.0,
            current_price=100.0,
            unit_account_value=100.0,
            current_value=100.0,
            total_cost=100.0,
            unrealized_pnl=0.0,
        )
        now = datetime(
            2026,
            7,
            29,
            15,
            0,
            tzinfo=timezone.utc,
        ).timestamp()

        with patch.object(auto_trader, "append_journal") as journal:
            predictions, diagnostics = runner._experimental_signals(
                [position],
                now,
            )

        self.assertEqual(predictions, {"AMD_US_EQ": 0.4})
        self.assertFalse(diagnostics["approved"])
        self.assertEqual(
            set(runner.experimental_tournament.call["histories"]),
            {"AMD_US_EQ"},
        )
        self.assertEqual(
            runner.experimental_tournament.call["current_prices"],
            {"AMD_US_EQ": 100.0},
        )
        self.assertEqual(
            runner.experimental_tournament.call["active_universe"],
            {"AMD_US_EQ"},
        )
        self.assertTrue(
            runner.experimental_tournament.call["allow_batch"]
        )
        journal.assert_called_once_with(
            "CANDIDATE_TEST",
            candidate="legacy_ensemble",
        )

    def test_experimental_status_lines_include_candidate_metrics(self):
        diagnostics = {
            "champion": "robust_huber",
            "approved": True,
            "candidates": {
                "robust_huber": {
                    "status": "APPROVED",
                    "batchCount": 20,
                    "outcomeCount": 40,
                    "hitRate": 0.6,
                    "meanNetReturn": 0.002,
                }
            },
        }

        lines = auto_trader.format_experimental_status_lines(diagnostics)

        self.assertIn("冠军 robust_huber", lines[0])
        self.assertIn("robust_huber", lines[1])
        self.assertIn("APPROVED", lines[1])
        self.assertIn("批次 20", lines[1])
        self.assertIn("结果 40", lines[1])
        self.assertIn("命中 60.0%", lines[1])
        self.assertIn("净收益 0.200%", lines[1])

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
        self.assertEqual(data.get("scout_seed_interval_seconds"), 300)
        self.assertEqual(data.get("scout_min_samples"), 241)
        self.assertTrue(all(0 < item["probe_quantity"] < 0.1 for item in scouts))

    def test_api_backoff_is_exponential_and_capped(self):
        self.assertEqual(backoff_seconds(15, 1), 30)
        self.assertEqual(backoff_seconds(15, 3), 120)
        self.assertEqual(backoff_seconds(15, 20), 300)


if __name__ == "__main__":
    unittest.main()
