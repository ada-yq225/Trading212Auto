"""Leakage-aware online ensemble used as a gated experimental signal."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = (
    "return_5",
    "return_15",
    "return_60",
    "return_120",
    "volatility_15",
    "volatility_60",
    "positive_fraction_15",
    "positive_fraction_60",
    "drawdown_60",
    "distance_mean_15",
    "distance_mean_60",
    "short_long_interaction",
)


@dataclass(frozen=True)
class ExperimentSettings:
    horizon_samples: int = 15
    minimum_history: int = 120
    training_stride: int = 5
    minimum_training_samples: int = 120
    minimum_shadow_outcomes: int = 40
    minimum_shadow_batches: int = 20
    minimum_hit_rate: float = 0.53
    minimum_mean_net_return: float = 0.0
    assumed_round_trip_cost: float = 0.001
    prediction_interval_seconds: float = 900.0
    top_n: int = 6
    random_state: int = 212


def _log_return(values: list[float], start: int, end: int) -> float:
    if values[start] <= 0 or values[end] <= 0:
        return 0.0
    return math.log(values[end] / values[start])


def _returns(values: list[float], start: int, end: int) -> list[float]:
    return [
        math.log(values[index] / values[index - 1])
        for index in range(start + 1, end + 1)
        if values[index] > 0 and values[index - 1] > 0
    ]


def causal_features(values: list[float], index: int) -> list[float] | None:
    """Build features using values at or before index only."""
    if index < 120 or index >= len(values) or values[index] <= 0:
        return None
    windows = (5, 15, 60, 120)
    returns_by_window = {
        window: _log_return(values, index - window, index) for window in windows
    }
    recent_15 = _returns(values, index - 15, index)
    recent_60 = _returns(values, index - 60, index)

    def volatility(items: list[float]) -> float:
        return float(np.std(items, ddof=1)) if len(items) > 1 else 0.0

    def positive_fraction(items: list[float]) -> float:
        return sum(item > 0 for item in items) / len(items) if items else 0.5

    recent_prices_60 = values[index - 59 : index + 1]
    peak_60 = max(recent_prices_60)
    drawdown_60 = values[index] / peak_60 - 1.0 if peak_60 > 0 else 0.0
    mean_15 = mean(values[index - 14 : index + 1])
    mean_60 = mean(recent_prices_60)
    distance_15 = values[index] / mean_15 - 1.0 if mean_15 > 0 else 0.0
    distance_60 = values[index] / mean_60 - 1.0 if mean_60 > 0 else 0.0
    interaction = returns_by_window[15] * returns_by_window[120]
    return [
        returns_by_window[5],
        returns_by_window[15],
        returns_by_window[60],
        returns_by_window[120],
        volatility(recent_15),
        volatility(recent_60),
        positive_fraction(recent_15),
        positive_fraction(recent_60),
        drawdown_60,
        distance_15,
        distance_60,
        interaction,
    ]


def supervised_samples(
    histories: dict[str, list[float]],
    settings: ExperimentSettings,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[list[float]] = []
    labels: list[float] = []
    for values in histories.values():
        final_feature_index = len(values) - 1 - settings.horizon_samples
        for index in range(
            settings.minimum_history,
            final_feature_index + 1,
            settings.training_stride,
        ):
            row = causal_features(values, index)
            if row is None or values[index] <= 0:
                continue
            future_return = math.log(
                values[index + settings.horizon_samples] / values[index]
            )
            past_returns = _returns(values, index - 60, index)
            past_volatility = max(float(np.std(past_returns, ddof=1)), 1e-6)
            risk_adjusted_label = future_return / (
                past_volatility * math.sqrt(settings.horizon_samples)
            )
            features.append(row)
            labels.append(max(-5.0, min(5.0, risk_adjusted_label)))
    if not features:
        return np.empty((0, len(FEATURE_NAMES))), np.empty((0,))
    return np.asarray(features, dtype=float), np.asarray(labels, dtype=float)


class ExperimentalEnsemble:
    def __init__(self, settings: ExperimentSettings):
        self.settings = settings
        self.linear = make_pipeline(
            StandardScaler(),
            Ridge(alpha=20.0, solver="lsqr"),
        )
        self.tree = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=100,
            max_depth=3,
            min_samples_leaf=30,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=settings.random_state,
        )
        self.training_samples = 0
        self.is_fitted = False

    def fit(self, histories: dict[str, list[float]]) -> int:
        features, labels = supervised_samples(histories, self.settings)
        self.training_samples = len(labels)
        if (
            self.training_samples < self.settings.minimum_training_samples
            or float(np.std(labels)) < 1e-8
        ):
            self.is_fitted = False
            return self.training_samples
        self.linear.fit(features, labels)
        self.tree.fit(features, labels)
        self.is_fitted = True
        return self.training_samples

    def predict(self, histories: dict[str, list[float]]) -> dict[str, float]:
        if not self.is_fitted:
            return {}
        tickers: list[str] = []
        rows: list[list[float]] = []
        for ticker, values in histories.items():
            row = causal_features(values, len(values) - 1)
            if row is not None:
                tickers.append(ticker)
                rows.append(row)
        if not rows:
            return {}
        matrix = np.asarray(rows, dtype=float)
        predictions = 0.5 * self.linear.predict(matrix) + 0.5 * self.tree.predict(matrix)
        center = float(np.mean(predictions))
        scale = max(float(np.std(predictions)), 1e-6)
        return {
            ticker: max(-3.0, min(3.0, float((prediction - center) / scale)))
            for ticker, prediction in zip(tickers, predictions)
        }


def resolve_shadow_predictions(
    experiment_state: dict[str, Any],
    current_prices: dict[str, float],
    now: float,
    settings: ExperimentSettings,
) -> None:
    pending = experiment_state.setdefault("pending", [])
    outcomes = experiment_state.setdefault("outcomes", [])
    unresolved: list[dict[str, Any]] = []
    horizon_seconds = settings.horizon_samples * 60.0
    for item in pending:
        if now - float(item["createdAt"]) < horizon_seconds:
            unresolved.append(item)
            continue
        ticker = str(item["ticker"])
        current_price = current_prices.get(ticker)
        base_price = float(item.get("basePrice") or 0)
        if not current_price or base_price <= 0:
            unresolved.append(item)
            continue
        realized = current_price / base_price - 1.0
        predicted = float(item["prediction"])
        selected = bool(item.get("selected"))
        outcomes.append(
            {
                "createdAt": item["createdAt"],
                "resolvedAt": now,
                "ticker": ticker,
                "prediction": predicted,
                "realizedReturn": realized,
                "hit": (predicted >= 0) == (realized >= 0),
                "selected": selected,
                "netReturn": (
                    realized - settings.assumed_round_trip_cost if selected else 0.0
                ),
            }
        )
    experiment_state["pending"] = unresolved[-500:]
    experiment_state["outcomes"] = outcomes[-500:]


def gate_statistics(
    experiment_state: dict[str, Any],
    settings: ExperimentSettings,
) -> dict[str, Any]:
    selected = [
        item for item in experiment_state.get("outcomes", []) if item.get("selected")
    ][-settings.minimum_shadow_outcomes :]
    count = len(selected)
    batch_count = len({float(item["createdAt"]) for item in selected})
    hit_rate = (
        sum(bool(item.get("hit")) for item in selected) / count if count else 0.0
    )
    mean_net_return = (
        mean(float(item.get("netReturn") or 0.0) for item in selected)
        if count
        else 0.0
    )
    approved = (
        count >= settings.minimum_shadow_outcomes
        and batch_count >= settings.minimum_shadow_batches
        and hit_rate >= settings.minimum_hit_rate
        and mean_net_return > settings.minimum_mean_net_return
    )
    return {
        "outcomeCount": count,
        "batchCount": batch_count,
        "hitRate": hit_rate,
        "meanNetReturn": mean_net_return,
        "approved": approved,
    }


def create_prediction_batch(
    experiment_state: dict[str, Any],
    predictions: dict[str, float],
    current_prices: dict[str, float],
    now: float,
    settings: ExperimentSettings,
) -> bool:
    last_batch = float(experiment_state.get("lastPredictionBatch", 0))
    if now - last_batch < settings.prediction_interval_seconds or not predictions:
        return False
    ranked = sorted(predictions.items(), key=lambda item: item[1], reverse=True)
    selected = {
        ticker for ticker, prediction in ranked[: settings.top_n] if prediction > 0
    }
    pending = experiment_state.setdefault("pending", [])
    for ticker, prediction in predictions.items():
        price = current_prices.get(ticker)
        if not price:
            continue
        pending.append(
            {
                "createdAt": now,
                "ticker": ticker,
                "basePrice": price,
                "prediction": prediction,
                "selected": ticker in selected,
            }
        )
    experiment_state["pending"] = pending[-500:]
    experiment_state["lastPredictionBatch"] = now
    return True
