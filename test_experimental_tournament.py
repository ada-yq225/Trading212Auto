import copy
import importlib
import importlib.util
import math
import time
import unittest

from experimental_candidates import CANDIDATE_IDS
from experimental_model import ExperimentSettings
from experimental_tournament import (
    ExperimentalTournament,
    candidate_diagnostics,
)


def candidate_state(*, batches, hits, returns):
    outcomes = []
    for index, net_return in enumerate(returns):
        outcomes.append(
            {
                "createdAt": float(index % batches),
                "selected": True,
                "hit": index < hits,
                "netReturn": float(net_return),
            }
        )
    return {
        "trainingSamples": 200,
        "pending": [],
        "outcomes": outcomes,
        "frozen": False,
    }


def approved_state(*, mean_return):
    return candidate_state(
        batches=20,
        hits=24,
        returns=[mean_return] * 40,
    )


def failed_state(*, mean_return):
    return candidate_state(
        batches=19,
        hits=30,
        returns=[mean_return] * 40,
    )


def state_with_long_batch_history():
    older = [
        {
            "createdAt": float(batch),
            "selected": True,
            "hit": False,
            "netReturn": -0.1,
        }
        for batch in range(20)
    ]
    recent = [
        {
            "createdAt": float(13 + index % 7),
            "selected": True,
            "hit": index < 24,
            "netReturn": 0.002,
        }
        for index in range(40)
    ]
    return {
        "trainingSamples": 200,
        "pending": [],
        "outcomes": older + recent,
        "frozen": False,
    }


def synthetic_histories(length=300):
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


class FakeCandidate:
    def __init__(self, *, fail=False):
        self.fail = fail

    def fit(self, dataset):
        if self.fail:
            raise RuntimeError("candidate failed")
        return len(dataset.labels)

    def predict(self, tickers, matrix):
        return {
            ticker: float(index - len(tickers) / 2)
            for index, ticker in enumerate(tickers)
        }


class StepClock:
    def __init__(self, values):
        self.values = list(values)
        self.last = self.values[-1]

    def __call__(self):
        if self.values:
            self.last = self.values.pop(0)
        return self.last


def fake_models(failing=frozenset()):
    return {
        candidate_id: FakeCandidate(
            fail=candidate_id in failing
        )
        for candidate_id in CANDIDATE_IDS
    }


def make_tournament(settings, *, models=None, clock=time.monotonic):
    return ExperimentalTournament(
        settings,
        CANDIDATE_IDS,
        compute_budget_seconds=8.0,
        early_rejection_batches=10,
        early_rejection_outcomes=20,
        models=models or fake_models(),
        monotonic=clock,
    )


class TournamentTests(unittest.TestCase):
    def setUp(self):
        self.settings = ExperimentSettings(
            sample_interval_seconds=15.0,
            prediction_interval_seconds=225.0,
        )
        self.state = {}
        self.histories = synthetic_histories()
        self.current_prices = {
            ticker: values[-1]
            for ticker, values in self.histories.items()
        }
        self.active_universe = set(self.histories)

    def test_migration_preserves_legacy_progress_and_is_idempotent(self):
        self.assertIsNotNone(
            importlib.util.find_spec("experimental_tournament")
        )
        module = importlib.import_module("experimental_tournament")
        state = {
            "trainingSamples": 1039,
            "lastPredictions": {"A": 0.5},
            "pending": [{"createdAt": 10.0, "ticker": "A"}],
            "outcomes": [{"createdAt": 5.0, "selected": True}],
            "lastPredictionBatch": 10.0,
            "lastTrainingAttempt": 10.0,
            "lastDiagnostics": {"batchCount": 8},
        }

        changed = module.migrate_tournament_state(state, CANDIDATE_IDS)

        self.assertTrue(changed)
        self.assertEqual(state["schemaVersion"], 1)
        self.assertIsNone(state["champion"])
        legacy = state["candidates"]["legacy_ensemble"]
        self.assertEqual(legacy["trainingSamples"], 1039)
        self.assertEqual(len(legacy["pending"]), 1)
        self.assertEqual(len(legacy["outcomes"]), 1)
        self.assertEqual(state["lastTournamentBatch"], 10.0)
        self.assertEqual(
            state["candidates"]["robust_huber"]["outcomes"],
            [],
        )
        for key in module.LEGACY_KEYS:
            self.assertNotIn(key, state)

        snapshot = copy.deepcopy(state)
        self.assertFalse(
            module.migrate_tournament_state(state, CANDIDATE_IDS)
        )
        self.assertEqual(state, snapshot)

    def test_migration_adds_new_candidate_without_changing_existing_data(self):
        self.assertIsNotNone(
            importlib.util.find_spec("experimental_tournament")
        )
        module = importlib.import_module("experimental_tournament")
        state = {
            "schemaVersion": 1,
            "champion": "legacy_ensemble",
            "lastTournamentBatch": 20.0,
            "candidates": {
                "legacy_ensemble": {
                    **module.empty_candidate_state(),
                    "trainingSamples": 222,
                    "outcomes": [{"createdAt": 10.0}],
                }
            },
        }
        legacy = copy.deepcopy(state["candidates"]["legacy_ensemble"])

        changed = module.migrate_tournament_state(
            state,
            ("legacy_ensemble", "new_challenger"),
        )

        self.assertTrue(changed)
        self.assertEqual(state["candidates"]["legacy_ensemble"], legacy)
        self.assertEqual(
            state["candidates"]["new_challenger"],
            module.empty_candidate_state(),
        )
        self.assertEqual(state["champion"], "legacy_ensemble")
        self.assertEqual(state["lastTournamentBatch"], 20.0)

    def test_early_rejection_requires_both_bad_upper_bounds(self):
        module = importlib.import_module("experimental_tournament")
        weak = candidate_state(
            batches=10,
            hits=4,
            returns=[-0.003] * 20,
        )
        good_return = candidate_state(
            batches=10,
            hits=4,
            returns=[0.004] * 20,
        )

        self.assertTrue(
            module.should_freeze_candidate(
                "robust_huber",
                weak,
                self.settings,
                10,
                20,
            )
        )
        self.assertFalse(
            module.should_freeze_candidate(
                "robust_huber",
                good_return,
                self.settings,
                10,
                20,
            )
        )
        self.assertFalse(
            module.should_freeze_candidate(
                "legacy_ensemble",
                weak,
                self.settings,
                10,
                20,
            )
        )

    def test_early_rejection_never_approves(self):
        module = importlib.import_module("experimental_tournament")
        state = candidate_state(
            batches=10,
            hits=20,
            returns=[0.004] * 20,
        )

        diagnostics = module.candidate_diagnostics(
            state,
            self.settings,
        )

        self.assertFalse(diagnostics["approved"])

    def test_champion_selection_excludes_failed_candidates(self):
        module = importlib.import_module("experimental_tournament")
        candidates = {
            "legacy_ensemble": approved_state(mean_return=0.002),
            "robust_huber": approved_state(mean_return=0.003),
            "regime_histgb": failed_state(mean_return=0.010),
        }

        champion = module.select_champion(candidates, self.settings)

        self.assertEqual(champion, "robust_huber")

    def test_champion_is_revoked_when_gate_fails(self):
        module = importlib.import_module("experimental_tournament")
        candidates = {
            "robust_huber": failed_state(mean_return=-0.001),
        }

        self.assertIsNone(
            module.select_champion(
                candidates,
                self.settings,
                current="robust_huber",
            )
        )

    def test_batch_gate_uses_retained_history_but_metrics_use_recent_window(
        self,
    ):
        diagnostics = candidate_diagnostics(
            state_with_long_batch_history(),
            self.settings,
        )

        self.assertEqual(diagnostics["batchCount"], 20)
        self.assertEqual(diagnostics["outcomeCount"], 40)
        self.assertAlmostEqual(diagnostics["hitRate"], 0.6)
        self.assertAlmostEqual(diagnostics["meanNetReturn"], 0.002)
        self.assertTrue(diagnostics["approved"])

    def test_candidate_failure_is_isolated(self):
        models = fake_models(failing={"regime_histgb"})

        result = make_tournament(
            self.settings,
            models=models,
        ).update(
            self.state,
            self.histories,
            self.current_prices,
            self.active_universe,
            now=1000.0,
            allow_batch=True,
        )

        self.assertIn(
            "legacy_ensemble",
            result.diagnostics["candidates"],
        )
        self.assertEqual(
            result.diagnostics["candidates"]["regime_histgb"]["status"],
            "ERROR",
        )
        self.assertTrue(
            any(
                event["event"] == "CANDIDATE_ERROR"
                for event in result.events
            )
        )
        self.assertTrue(
            self.state["candidates"]["legacy_ensemble"]["pending"]
        )

    def test_three_candidate_failures_freeze_only_that_candidate(self):
        models = fake_models(failing={"regime_histgb"})
        engine = make_tournament(self.settings, models=models)

        for now in (1000.0, 1225.0, 1450.0):
            engine.update(
                self.state,
                self.histories,
                self.current_prices,
                self.active_universe,
                now=now,
                allow_batch=True,
            )

        self.assertTrue(
            self.state["candidates"]["regime_histgb"]["frozen"]
        )
        self.assertFalse(
            self.state["candidates"]["robust_huber"]["frozen"]
        )

    def test_compute_budget_skip_creates_no_fake_batch(self):
        clock = StepClock([0.0, 1.0, 9.0, 9.1])

        result = make_tournament(
            self.settings,
            clock=clock,
        ).update(
            self.state,
            self.histories,
            self.current_prices,
            self.active_universe,
            now=1000.0,
            allow_batch=True,
        )

        skipped = result.diagnostics["candidates"]["robust_huber"]
        self.assertEqual(skipped["status"], "SKIPPED_BUDGET")
        self.assertEqual(
            self.state["candidates"]["robust_huber"]["pending"],
            [],
        )
        self.assertEqual(
            self.state["candidates"]["robust_huber"]["errorCount"],
            0,
        )

    def test_tournament_uses_one_shared_batch_timestamp(self):
        make_tournament(self.settings).update(
            self.state,
            self.histories,
            self.current_prices,
            self.active_universe,
            now=1000.0,
            allow_batch=True,
        )

        created = {
            item["createdAt"]
            for state in self.state["candidates"].values()
            for item in state["pending"]
        }

        self.assertEqual(created, {1000.0})


if __name__ == "__main__":
    unittest.main()
