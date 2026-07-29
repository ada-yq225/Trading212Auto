"""Diverse deterministic candidates for the online model tournament."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from experimental_model import (
    ExperimentSettings,
    ExperimentalEnsemble,
    SharedDataset,
    normalized_predictions,
)


CANDIDATE_IDS = (
    "legacy_ensemble",
    "robust_huber",
    "regime_histgb",
    "residual_momentum",
)


class CandidateModel(Protocol):
    def fit(self, dataset: SharedDataset) -> int:
        raise NotImplementedError

    def predict(
        self,
        tickers: tuple[str, ...],
        matrix: np.ndarray,
    ) -> dict[str, float]:
        raise NotImplementedError


class LegacyCandidate:
    def __init__(self, settings: ExperimentSettings):
        self.model = ExperimentalEnsemble(settings)

    def fit(self, dataset: SharedDataset) -> int:
        return self.model.fit_dataset(dataset)

    def predict(
        self,
        tickers: tuple[str, ...],
        matrix: np.ndarray,
    ) -> dict[str, float]:
        return self.model.predict_matrix(tickers, matrix)


class RobustHuberCandidate:
    def __init__(self, settings: ExperimentSettings):
        self.settings = settings
        self.model = make_pipeline(
            RobustScaler(),
            SGDRegressor(
                loss="huber",
                epsilon=0.1,
                alpha=0.001,
                max_iter=2000,
                tol=1e-5,
                shuffle=False,
                random_state=settings.random_state,
            ),
        )
        self.is_fitted = False

    def fit(self, dataset: SharedDataset) -> int:
        count = len(dataset.labels)
        self.is_fitted = (
            count >= self.settings.minimum_training_samples
            and float(np.std(dataset.labels)) >= 1e-8
        )
        if self.is_fitted:
            self.model.fit(dataset.features[:, :12], dataset.labels)
        return count

    def predict(
        self,
        tickers: tuple[str, ...],
        matrix: np.ndarray,
    ) -> dict[str, float]:
        if not self.is_fitted or not len(matrix):
            return {}
        return normalized_predictions(
            tickers,
            self.model.predict(matrix[:, :12]),
        )


class RegimeHistGBCandidate:
    def __init__(self, settings: ExperimentSettings):
        self.settings = settings
        self.model = HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_iter=120,
            max_depth=3,
            min_samples_leaf=30,
            l2_regularization=2.0,
            early_stopping=False,
            random_state=settings.random_state,
        )
        self.is_fitted = False

    def fit(self, dataset: SharedDataset) -> int:
        count = len(dataset.labels)
        self.is_fitted = (
            count >= self.settings.minimum_training_samples
            and float(np.std(dataset.labels)) >= 1e-8
        )
        if self.is_fitted:
            self.model.fit(dataset.features, dataset.labels)
        return count

    def predict(
        self,
        tickers: tuple[str, ...],
        matrix: np.ndarray,
    ) -> dict[str, float]:
        if not self.is_fitted or not len(matrix):
            return {}
        return normalized_predictions(tickers, self.model.predict(matrix))


class ResidualMomentumCandidate:
    def __init__(self, settings: ExperimentSettings):
        self.settings = settings
        self.is_fitted = False

    def fit(self, dataset: SharedDataset) -> int:
        count = len(dataset.labels)
        self.is_fitted = count >= self.settings.minimum_training_samples
        return count

    def predict(
        self,
        tickers: tuple[str, ...],
        matrix: np.ndarray,
    ) -> dict[str, float]:
        if not self.is_fitted or not len(matrix):
            return {}
        raw = (
            0.25 * (matrix[:, 1] - matrix[:, 12])
            + 0.45 * (matrix[:, 2] - matrix[:, 13])
            + 0.30 * (matrix[:, 3] - matrix[:, 14])
            - 0.15 * matrix[:, 0]
        ) / np.maximum(matrix[:, 5], 1e-6)
        return normalized_predictions(tickers, raw)


def create_candidate(
    candidate_id: str,
    settings: ExperimentSettings,
) -> CandidateModel:
    factories = {
        "legacy_ensemble": LegacyCandidate,
        "robust_huber": RobustHuberCandidate,
        "regime_histgb": RegimeHistGBCandidate,
        "residual_momentum": ResidualMomentumCandidate,
    }
    factory = factories.get(candidate_id)
    if factory is None:
        raise ValueError(f"未知候选模型：{candidate_id}")
    return factory(settings)
