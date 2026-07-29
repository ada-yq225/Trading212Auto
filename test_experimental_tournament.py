import copy
import importlib
import importlib.util
import unittest

from experimental_candidates import CANDIDATE_IDS
from experimental_model import ExperimentSettings


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


class TournamentTests(unittest.TestCase):
    def setUp(self):
        self.settings = ExperimentSettings()

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


if __name__ == "__main__":
    unittest.main()
