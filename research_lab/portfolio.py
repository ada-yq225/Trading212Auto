from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from .config import ResearchConfig


@dataclass
class PortfolioDecision:
    weights: dict[str, float]
    regime: str
    gross_target: float
    turnover: float
    estimated_cost: float
    optimizer_success: bool


def market_regime(
    frames: dict[str, pd.DataFrame],
    date: pd.Timestamp,
) -> str:
    if "SPY" not in frames:
        return "NEUTRAL"
    spy = frames["SPY"].loc[:date, "adjusted_close"].dropna()
    if len(spy) < 200:
        return "NEUTRAL"
    return_126 = spy.iloc[-1] / spy.iloc[-127] - 1.0
    above_sma = spy.iloc[-1] > spy.iloc[-200:].mean()
    vix_level = None
    if "^VIX" in frames:
        vix = frames["^VIX"].loc[:date, "adjusted_close"].dropna()
        if len(vix):
            vix_level = float(vix.iloc[-1])
    if (not above_sma and return_126 < 0) or (
        vix_level is not None and vix_level >= 30
    ):
        return "RISK_OFF"
    if above_sma and return_126 > 0.05 and (
        vix_level is None or vix_level < 25
    ):
        return "RISK_ON"
    return "NEUTRAL"


def gross_for_state(
    regime: str,
    drawdown: float,
    config: ResearchConfig,
) -> float:
    gross = {
        "RISK_ON": config.risk_on_gross,
        "NEUTRAL": config.neutral_gross,
        "RISK_OFF": config.risk_off_gross,
    }[regime]
    if drawdown >= config.drawdown_emergency:
        return min(gross, 0.15)
    if drawdown >= config.drawdown_cut:
        return min(gross, 0.40)
    return gross


def _covariance(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    date: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray]:
    series = {
        symbol: np.log(
            frames[symbol].loc[:date, "adjusted_close"].dropna()
        ).diff()
        for symbol in symbols
    }
    returns = pd.DataFrame(series).tail(126).dropna(how="all")
    returns = returns.fillna(0.0)
    if len(returns) < 30:
        covariance = np.eye(len(symbols)) * 0.04
        volatility = np.ones(len(symbols)) * 0.20
        return covariance, volatility
    covariance = (
        LedoitWolf(store_precision=False).fit(returns.values).covariance_ * 252
    )
    volatility = np.sqrt(np.maximum(np.diag(covariance), 1e-8))
    return covariance, volatility


def optimize_portfolio(
    predictions: dict[str, float],
    frames: dict[str, pd.DataFrame],
    sectors: dict[str, str],
    date: pd.Timestamp,
    previous_weights: dict[str, float],
    equity_drawdown: float,
    config: ResearchConfig,
    *,
    cost_bps: float | None = None,
) -> PortfolioDecision:
    ranked = [
        (symbol, prediction)
        for symbol, prediction in sorted(
            predictions.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if prediction > 0 and symbol in frames
    ][: config.top_n]
    regime = market_regime(frames, date)
    gross = gross_for_state(regime, equity_drawdown, config)
    if not ranked:
        turnover = sum(abs(value) for value in previous_weights.values())
        cost_rate = (cost_bps or config.base_transaction_cost_bps) / 10000
        return PortfolioDecision(
            {},
            regime,
            gross,
            turnover,
            turnover * cost_rate,
            True,
        )

    symbols = [symbol for symbol, _ in ranked]
    scores = np.asarray([score for _, score in ranked], dtype=float)
    covariance, volatility = _covariance(frames, symbols, date)
    desirability = np.exp(np.clip(scores, -2.0, 2.0)) / np.maximum(
        volatility,
        0.05,
    )
    target = gross * desirability / desirability.sum()
    previous = np.asarray(
        [previous_weights.get(symbol, 0.0) for symbol in symbols],
        dtype=float,
    )

    def objective(weights: np.ndarray) -> float:
        risk = float(weights @ covariance @ weights)
        forecast_deviation = float(np.sum((weights - target) ** 2))
        turnover_penalty = float(np.sum((weights - previous) ** 2))
        underinvestment = float((weights.sum() - gross) ** 2)
        return risk + 2.0 * forecast_deviation + 0.40 * turnover_penalty + 5.0 * underinvestment

    constraints: list[dict] = [
        {"type": "ineq", "fun": lambda weights: gross - weights.sum()}
    ]
    for sector in sorted({sectors[symbol] for symbol in symbols}):
        indices = [
            index for index, symbol in enumerate(symbols) if sectors[symbol] == sector
        ]
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda weights, selected=indices: (
                    config.maximum_sector_weight - weights[selected].sum()
                ),
            }
        )
    initial = np.minimum(
        target,
        np.full(len(symbols), config.maximum_position_weight),
    )
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, config.maximum_position_weight)] * len(symbols),
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-10},
    )
    optimized = result.x if result.success else initial
    weights = {
        symbol: float(weight)
        for symbol, weight in zip(symbols, optimized)
        if weight > 1e-5
    }
    turnover = sum(
        abs(weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0))
        for symbol in set(weights) | set(previous_weights)
    )
    cost_rate = (cost_bps or config.base_transaction_cost_bps) / 10000
    return PortfolioDecision(
        weights=weights,
        regime=regime,
        gross_target=gross,
        turnover=turnover,
        estimated_cost=turnover * cost_rate,
        optimizer_success=bool(result.success),
    )
