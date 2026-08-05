"""State and orchestration for the experimental model tournament."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Any, Callable

from experimental_candidates import CandidateModel, create_candidate
from experimental_model import (
    ExperimentSettings,
    build_shared_dataset,
    create_prediction_batch,
    current_feature_matrix,
    resolve_shadow_predictions,
)


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
    all_selected = [
        item
        for item in candidate_state.get("outcomes", [])
        if item.get("selected")
    ]
    selected = all_selected[-settings.minimum_shadow_outcomes :]
    outcome_count = len(selected)
    batch_count = len(
        {float(item["createdAt"]) for item in all_selected}
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


@dataclass(frozen=True)
class TournamentResult:
    predictions: dict[str, float]
    diagnostics: dict[str, Any]
    events: tuple[dict[str, Any], ...]


class ExperimentalTournament:
    def __init__(
        self,
        experiment_settings: ExperimentSettings,
        candidate_ids: tuple[str, ...],
        compute_budget_seconds: float,
        early_rejection_batches: int,
        early_rejection_outcomes: int,
        *,
        models: dict[str, CandidateModel] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.settings = experiment_settings
        self.candidate_ids = tuple(candidate_ids)
        self.compute_budget_seconds = compute_budget_seconds
        self.early_rejection_batches = early_rejection_batches
        self.early_rejection_outcomes = early_rejection_outcomes
        self.models = models or {
            candidate_id: create_candidate(
                candidate_id,
                experiment_settings,
            )
            for candidate_id in self.candidate_ids
        }
        self.monotonic = monotonic

    def update(
        self,
        state: dict[str, Any],
        histories: dict[str, list[float]],
        current_prices: dict[str, float],
        active_universe: set[str],
        now: float,
        allow_batch: bool,
    ) -> TournamentResult:
        migrate_tournament_state(state, self.candidate_ids)
        candidates = state["candidates"]
        events: list[dict[str, Any]] = []
        statuses: dict[str, str] = {}

        for candidate_id in self.candidate_ids:
            candidate_state = candidates[candidate_id]
            resolve_shadow_predictions(
                candidate_state,
                current_prices,
                now,
                self.settings,
            )
            if (
                not candidate_state.get("frozen")
                and should_freeze_candidate(
                    candidate_id,
                    candidate_state,
                    self.settings,
                    self.early_rejection_batches,
                    self.early_rejection_outcomes,
                )
            ):
                candidate_state["frozen"] = True
                statuses[candidate_id] = "FROZEN"
                events.append(
                    {
                        "event": "CANDIDATE_FROZEN",
                        "candidate": candidate_id,
                    }
                )

        last_batch = float(state.get("lastTournamentBatch", 0.0) or 0.0)
        batch_due = (
            allow_batch
            and now - last_batch
            >= self.settings.prediction_interval_seconds
        )
        if batch_due:
            dataset = build_shared_dataset(histories, self.settings)
            prediction_histories = {
                ticker: histories[ticker]
                for ticker in sorted(active_universe)
                if ticker in histories
            }
            tickers, matrix = current_feature_matrix(
                prediction_histories,
            )
            started = self.monotonic()
            for index, candidate_id in enumerate(self.candidate_ids):
                candidate_state = candidates[candidate_id]
                if candidate_state.get("frozen"):
                    statuses[candidate_id] = "FROZEN"
                    continue
                if self.monotonic() - started >= self.compute_budget_seconds:
                    for skipped_id in self.candidate_ids[index:]:
                        if candidates[skipped_id].get("frozen"):
                            statuses[skipped_id] = "FROZEN"
                            continue
                        statuses[skipped_id] = "SKIPPED_BUDGET"
                        events.append(
                            {
                                "event": "CANDIDATE_BUDGET_SKIP",
                                "candidate": skipped_id,
                            }
                        )
                    break
                candidate_state["lastTrainingAttempt"] = now
                try:
                    training_samples = self.models[candidate_id].fit(
                        dataset
                    )
                    predictions = self.models[candidate_id].predict(
                        tickers,
                        matrix,
                    )
                    predictions = {
                        str(ticker): float(value)
                        for ticker, value in predictions.items()
                        if ticker in active_universe
                        and math.isfinite(float(value))
                    }
                    candidate_state["trainingSamples"] = int(
                        training_samples
                    )
                    candidate_state["lastPredictions"] = predictions
                    candidate_state["errorCount"] = 0
                    create_prediction_batch(
                        candidate_state,
                        predictions,
                        current_prices,
                        now,
                        self.settings,
                    )
                except Exception as exc:
                    error_count = int(
                        candidate_state.get("errorCount", 0) or 0
                    ) + 1
                    candidate_state["errorCount"] = error_count
                    statuses[candidate_id] = "ERROR"
                    events.append(
                        {
                            "event": "CANDIDATE_ERROR",
                            "candidate": candidate_id,
                            "error": repr(exc),
                        }
                    )
                    if (
                        candidate_id != "legacy_ensemble"
                        and error_count >= 3
                    ):
                        candidate_state["frozen"] = True
                    continue
            state["lastTournamentBatch"] = now

        previous_champion = state.get("champion")
        champion = select_champion(
            candidates,
            self.settings,
            current=previous_champion,
        )
        state["champion"] = champion
        if champion != previous_champion:
            events.append(
                {
                    "event": "TOURNAMENT_CHAMPION_CHANGED",
                    "previous": previous_champion,
                    "champion": champion,
                }
            )

        candidate_rows: dict[str, dict[str, Any]] = {}
        for candidate_id in self.candidate_ids:
            candidate_state = candidates[candidate_id]
            diagnostics = candidate_diagnostics(
                candidate_state,
                self.settings,
            )
            if candidate_id not in statuses:
                if diagnostics["frozen"]:
                    status = "FROZEN"
                elif (
                    diagnostics["trainingSamples"]
                    < self.settings.minimum_training_samples
                ):
                    status = "WARMUP"
                elif diagnostics["approved"]:
                    status = "APPROVED"
                else:
                    status = "SHADOW"
            else:
                status = statuses[candidate_id]
            row = {
                "status": status,
                "predictionCount": len(
                    candidate_state.get("lastPredictions", {})
                ),
                "pendingCount": len(
                    candidate_state.get("pending", [])
                ),
                "errorCount": int(
                    candidate_state.get("errorCount", 0) or 0
                ),
                **diagnostics,
            }
            candidate_state["lastDiagnostics"] = row
            candidate_rows[candidate_id] = row

        approved = bool(
            champion
            and candidate_rows.get(champion, {}).get("approved")
        )
        predictions = (
            {
                str(ticker): float(value)
                for ticker, value in candidates[champion]
                .get("lastPredictions", {})
                .items()
            }
            if approved and champion
            else {}
        )
        diagnostics = {
            "status": "APPROVED" if approved else "SHADOW",
            "approved": approved,
            "champion": champion,
            "lastTournamentBatch": float(
                state.get("lastTournamentBatch", 0.0) or 0.0
            ),
            "candidates": candidate_rows,
        }
        return TournamentResult(
            predictions=predictions,
            diagnostics=diagnostics,
            events=tuple(events),
        )
