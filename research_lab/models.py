from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from .config import ResearchConfig


@dataclass
class EnsemblePrediction:
    predictions: dict[str, float]
    model_weights: dict[str, float]
    validation_ic: dict[str, float]
    training_rows: int
    validation_rows: int
    feature_importance: dict[str, float]


def model_factory(name: str, seed: int) -> RegressorMixin:
    if name == "ridge":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RobustScaler(),
            Ridge(alpha=20.0, solver="lsqr"),
        )
    if name == "elastic_net":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RobustScaler(),
            ElasticNet(
                alpha=0.002,
                l1_ratio=0.15,
                max_iter=5000,
                random_state=seed,
            ),
        )
    if name == "hist_gradient_boosting":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                learning_rate=0.04,
                max_iter=120,
                max_leaf_nodes=15,
                max_depth=4,
                min_samples_leaf=40,
                l2_regularization=2.0,
                early_stopping=False,
                random_state=seed,
            ),
        )
    if name == "extra_trees":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=160,
                max_depth=8,
                min_samples_leaf=15,
                max_features=0.70,
                n_jobs=-1,
                random_state=seed,
            ),
        )
    if name == "random_fourier_ridge":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            RBFSampler(gamma=0.02, n_components=160, random_state=seed),
            Ridge(alpha=30.0, solver="lsqr"),
        )
    raise ValueError(f"Unknown model: {name}")


def _sample_dates(
    frame: pd.DataFrame,
    stride: int,
) -> pd.DataFrame:
    dates = sorted(frame["date"].dropna().unique())
    selected = set(dates[::stride])
    return frame[frame["date"].isin(selected)]


def _mean_cross_sectional_ic(
    frame: pd.DataFrame,
    predictions: np.ndarray,
) -> float:
    scored = frame[["date", "target"]].copy()
    scored["prediction"] = predictions
    correlations: list[float] = []
    for _, group in scored.groupby("date"):
        if len(group) < 5 or group["prediction"].nunique() < 2:
            continue
        correlation = spearmanr(
            group["prediction"],
            group["target"],
            nan_policy="omit",
        ).statistic
        if np.isfinite(correlation):
            correlations.append(float(correlation))
    return float(np.mean(correlations)) if correlations else -1.0


def _fit_model(
    model: RegressorMixin,
    features: pd.DataFrame,
    target: pd.Series,
) -> RegressorMixin:
    if not np.isfinite(target.to_numpy(dtype=float)).all():
        raise FloatingPointError("Training targets contain non-finite values")
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        model.fit(features, target)
    return model


def _predict_model(
    model: RegressorMixin,
    features: pd.DataFrame,
) -> np.ndarray:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        predictions = np.asarray(model.predict(features), dtype=float)
    if not np.isfinite(predictions).all():
        raise FloatingPointError("Model emitted non-finite predictions")
    return predictions


def _split_nested(
    panel: pd.DataFrame,
    prediction_date: pd.Timestamp,
    config: ResearchConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matured = panel[
        panel["label_end_date"].notna()
        & (panel["label_end_date"] <= prediction_date)
        & panel["target"].notna()
    ].copy()
    dates = sorted(matured["date"].unique())
    separation_days = config.purge_days + config.embargo_days
    if len(dates) < config.inner_validation_days + separation_days + 20:
        sampled = _sample_dates(matured, config.training_stride_days)
        return sampled.iloc[0:0], sampled.iloc[0:0], sampled
    validation_start_index = len(dates) - config.inner_validation_days
    validation_start = pd.Timestamp(dates[validation_start_index])
    purge_index = max(0, validation_start_index - separation_days)
    training_cutoff = pd.Timestamp(dates[purge_index])
    inner_train = _sample_dates(
        matured[matured["label_end_date"] < training_cutoff],
        config.training_stride_days,
    )
    inner_validation = _sample_dates(
        matured[matured["date"] >= validation_start],
        config.training_stride_days,
    )
    sampled = _sample_dates(matured, config.training_stride_days)
    return inner_train, inner_validation, sampled


def _feature_importance(
    fitted_models: dict[str, RegressorMixin],
    feature_columns: list[str],
    weights: dict[str, float],
) -> dict[str, float]:
    combined = np.zeros(len(feature_columns), dtype=float)
    for name, model in fitted_models.items():
        estimator = model.steps[-1][1] if hasattr(model, "steps") else model
        importance: np.ndarray | None = None
        if hasattr(estimator, "feature_importances_"):
            importance = np.asarray(estimator.feature_importances_, dtype=float)
        elif hasattr(estimator, "coef_"):
            coefficients = np.asarray(estimator.coef_, dtype=float).reshape(-1)
            if len(coefficients) == len(feature_columns):
                importance = np.abs(coefficients)
        if importance is None or len(importance) != len(feature_columns):
            continue
        total = float(importance.sum())
        if total > 0:
            combined += weights.get(name, 0.0) * importance / total
    return {
        feature: float(value)
        for feature, value in sorted(
            zip(feature_columns, combined),
            key=lambda item: item[1],
            reverse=True,
        )
        if value > 0
    }


def fit_nested_ensemble(
    panel: pd.DataFrame,
    feature_columns: list[str],
    prediction_date: pd.Timestamp,
    config: ResearchConfig,
) -> EnsemblePrediction:
    inner_train, inner_validation, matured = _split_nested(
        panel,
        prediction_date,
        config,
    )
    prediction_frame = panel[panel["date"] == prediction_date].copy()
    if (
        len(inner_train) < 500
        or len(inner_validation) < 100
        or prediction_frame.empty
    ):
        return EnsemblePrediction({}, {}, {}, len(matured), len(inner_validation), {})

    validation_ic: dict[str, float] = {}
    for name in config.model_names:
        model = _fit_model(
            model_factory(name, config.random_seed),
            inner_train[feature_columns],
            inner_train["target"],
        )
        validation_predictions = _predict_model(
            model,
            inner_validation[feature_columns],
        )
        validation_ic[name] = _mean_cross_sectional_ic(
            inner_validation,
            validation_predictions,
        )

    positive = {
        name: max(0.0, score) ** 2 for name, score in validation_ic.items()
    }
    normalizer = sum(positive.values())
    if normalizer <= 0:
        model_weights = {name: 1.0 / len(config.model_names) for name in config.model_names}
    else:
        model_weights = {
            name: value / normalizer for name, value in positive.items()
        }

    fitted_models: dict[str, RegressorMixin] = {}
    combined = np.zeros(len(prediction_frame), dtype=float)
    for name in config.model_names:
        model = _fit_model(
            model_factory(name, config.random_seed),
            matured[feature_columns],
            matured["target"],
        )
        fitted_models[name] = model
        combined += model_weights[name] * _predict_model(
            model,
            prediction_frame[feature_columns],
        )

    center = float(np.nanmean(combined))
    scale = max(float(np.nanstd(combined)), 1e-8)
    standardized = np.clip((combined - center) / scale, -3.0, 3.0)
    predictions = {
        str(symbol): float(value)
        for symbol, value in zip(prediction_frame["symbol"], standardized)
    }
    return EnsemblePrediction(
        predictions=predictions,
        model_weights=model_weights,
        validation_ic=validation_ic,
        training_rows=len(matured),
        validation_rows=len(inner_validation),
        feature_importance=_feature_importance(
            fitted_models,
            feature_columns,
            model_weights,
        ),
    )
