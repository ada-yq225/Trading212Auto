"""State and orchestration for the experimental model tournament."""

from __future__ import annotations

import copy
from typing import Any


TOURNAMENT_SCHEMA_VERSION = 1
LEGACY_KEYS = (
    "trainingSamples",
    "lastPredictions",
    "pending",
    "outcomes",
    "lastPredictionBatch",
    "lastTrainingAttempt",
    "lastDiagnostics",
)


def empty_candidate_state() -> dict[str, Any]:
    return {
        "trainingSamples": 0,
        "lastPredictions": {},
        "pending": [],
        "outcomes": [],
        "lastPredictionBatch": 0.0,
        "lastTrainingAttempt": 0.0,
        "lastDiagnostics": {},
        "frozen": False,
        "errorCount": 0,
    }


def migrate_tournament_state(
    state: dict[str, Any],
    candidate_ids: tuple[str, ...],
) -> bool:
    """Migrate legacy single-model state and add newly configured candidates."""
    if state.get("schemaVersion") not in (None, TOURNAMENT_SCHEMA_VERSION):
        raise ValueError(
            f"unsupported tournament schema: {state.get('schemaVersion')!r}"
        )

    changed = False
    if state.get("schemaVersion") != TOURNAMENT_SCHEMA_VERSION:
        legacy = empty_candidate_state()
        for key in LEGACY_KEYS:
            if key in state:
                legacy[key] = copy.deepcopy(state[key])
        last_batch = float(legacy.get("lastPredictionBatch", 0.0) or 0.0)
        state.clear()
        state.update(
            {
                "schemaVersion": TOURNAMENT_SCHEMA_VERSION,
                "champion": None,
                "lastTournamentBatch": last_batch,
                "candidates": {"legacy_ensemble": legacy},
            }
        )
        changed = True

    if "champion" not in state:
        state["champion"] = None
        changed = True
    if "lastTournamentBatch" not in state:
        state["lastTournamentBatch"] = 0.0
        changed = True
    if not isinstance(state.get("candidates"), dict):
        state["candidates"] = {}
        changed = True

    candidates = state["candidates"]
    for candidate_id in candidate_ids:
        if candidate_id not in candidates:
            candidates[candidate_id] = empty_candidate_state()
            changed = True
    return changed
