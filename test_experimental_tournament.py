import copy
import importlib
import importlib.util
import unittest

from experimental_candidates import CANDIDATE_IDS


class TournamentTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
