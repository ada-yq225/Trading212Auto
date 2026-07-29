"""State and orchestration for the experimental model tournament."""

from __future__ import annotations

import copy
import math
from statistics import mean, stdev
from typing import Any

from experimental_model import ExperimentSettings


TOURNAMENT_SCHEMA_VERSION = 1
ONE_SIDED_95_Z = 1.6448536269514722
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


def wilson_upper(hits: int, count: int) -> float:
    if count <= 0:
        return 1.0
    probability = hits / count
    z2 = ONE_SIDED_95_Z**2
    numerator = (
        probability
        + z2 / (2 * count)
        + ONE_SIDED_95_Z
        * math.sqrt(
            probability * (1 - probability) / count
            + z2 / (4 * count**2)
        )
    )
    return numerator / (1 + z2 / count)


def candidate_diagnostics(
    candidate_state: dict[str, Any],
    settings: ExperimentSettings,
) -> dict[str, Any]:
    selected = [
        item
        for item in candidate_state.get("outcomes", [])
        if item.get("selected")
    ][-settings.minimum_shadow_outcomes :]
    outcome_count = len(selected)
    batch_count = len(
        {float(item["createdAt"]) for item in selected}
    )
    hits = sum(bool(item.get("hit")) for item in selected)
    hit_rate = hits / outcome_count if outcome_count else 0.0
    returns = [
        float(item.get("netReturn") or 0.0) for item in selected
    ]
    mean_net_return = mean(returns) if returns else 0.0
    return_standard_error = (
        stdev(returns) / math.sqrt(outcome_count)
        if outcome_count >= 2
        else math.inf
    )
    mean_upper = (
        mean_net_return + ONE_SIDED_95_Z * return_standard_error
        if outcome_count >= 2
        else math.inf
    )
    training_samples = int(
        candidate_state.get("trainingSamples", 0) or 0
    )
    frozen = bool(candidate_state.get("frozen", False))
    approved = (
        not frozen
        and training_samples >= settings.minimum_training_samples
        and outcome_count >= settings.minimum_shadow_outcomes
        and batch_count >= settings.minimum_shadow_batches
        and hit_rate >= settings.minimum_hit_rate
        and mean_net_return > settings.minimum_mean_net_return
    )
    return {
        "trainingSamples": training_samples,
        "outcomeCount": outcome_count,
        "batchCount": batch_count,
        "hitRate": hit_rate,
        "meanNetReturn": mean_net_return,
        "returnStandardError": return_standard_error,
        "hitRateUpper95": wilson_upper(hits, outcome_count),
        "meanNetReturnUpper95": mean_upper,
        "approved": approved,
        "frozen": frozen,
    }


def should_freeze_candidate(
    candidate_id: str,
    candidate_state: dict[str, Any],
    settings: ExperimentSettings,
    minimum_batches: int,
    minimum_outcomes: int,
) -> bool:
    if candidate_id == "legacy_ensemble":
        return False
    diagnostics = candidate_diagnostics(candidate_state, settings)
    return (
        diagnostics["batchCount"] >= minimum_batches
        and diagnostics["outcomeCount"] >= minimum_outcomes
        and diagnostics["hitRateUpper95"] < settings.minimum_hit_rate
        and diagnostics["meanNetReturnUpper95"]
        <= settings.minimum_mean_net_return
    )


def select_champion(
    candidates: dict[str, dict[str, Any]],
    settings: ExperimentSettings,
    current: str | None = None,
) -> str | None:
    del current
    eligible: list[tuple[str, dict[str, Any]]] = []
    for candidate_id, state in candidates.items():
        diagnostics = candidate_diagnostics(state, settings)
        if diagnostics["approved"] and not diagnostics["frozen"]:
            eligible.append((candidate_id, diagnostics))
    if not eligible:
        return None
    eligible.sort(
        key=lambda item: (
            -(
                float(item[1]["meanNetReturn"])
                - float(item[1]["returnStandardError"])
            ),
            -float(item[1]["hitRate"]),
            item[0],
        )
    )
    return eligible[0][0]
