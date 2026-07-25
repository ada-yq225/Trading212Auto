from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew


def drawdown_series(returns: pd.Series) -> pd.Series:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    return equity / equity.cummax() - 1.0


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    benchmark_annual_sharpe: float = 0.0,
) -> float:
    values = returns.dropna().to_numpy(dtype=float)
    if len(values) < 3 or np.std(values, ddof=1) <= 0:
        return 0.0
    daily_sharpe = np.mean(values) / np.std(values, ddof=1)
    benchmark = benchmark_annual_sharpe / math.sqrt(252)
    sample_skew = float(skew(values, bias=False))
    sample_kurtosis = float(kurtosis(values, fisher=False, bias=False))
    denominator = math.sqrt(
        max(
            1e-12,
            1
            - sample_skew * daily_sharpe
            + ((sample_kurtosis - 1) / 4) * daily_sharpe**2,
        )
    )
    statistic = (
        (daily_sharpe - benchmark) * math.sqrt(len(values) - 1) / denominator
    )
    return float(norm.cdf(statistic))


def deflated_sharpe_probability(
    returns: pd.Series,
    model_trials: int,
) -> float:
    sample_size = max(len(returns.dropna()), 2)
    trial_hurdle_daily = norm.ppf(1 - 1 / max(model_trials, 2)) / math.sqrt(
        sample_size
    )
    return probabilistic_sharpe_ratio(
        returns,
        benchmark_annual_sharpe=trial_hurdle_daily * math.sqrt(252),
    )


def performance_metrics(
    returns: pd.Series,
    *,
    model_trials: int = 1,
) -> dict[str, float]:
    values = returns.dropna().astype(float)
    if len(values) < 2:
        return {}
    total_return = float((1.0 + values).prod() - 1.0)
    years = len(values) / 252
    annual_return = (1.0 + total_return) ** (1 / max(years, 1 / 252)) - 1.0
    annual_volatility = float(values.std(ddof=1) * math.sqrt(252))
    sharpe = (
        float(values.mean() / values.std(ddof=1) * math.sqrt(252))
        if values.std(ddof=1) > 0
        else 0.0
    )
    downside = values[values < 0]
    downside_deviation = (
        float(np.sqrt(np.mean(np.square(downside))) * math.sqrt(252))
        if len(downside)
        else 0.0
    )
    sortino = annual_return / downside_deviation if downside_deviation > 0 else 0.0
    drawdowns = drawdown_series(values)
    maximum_drawdown = float(-drawdowns.min())
    calmar = annual_return / maximum_drawdown if maximum_drawdown > 0 else 0.0
    var_95 = float(np.quantile(values, 0.05))
    cvar_95 = float(values[values <= var_95].mean())
    return {
        "observations": float(len(values)),
        "totalReturn": total_return,
        "annualizedReturn": annual_return,
        "annualizedVolatility": annual_volatility,
        "sharpeZeroRate": sharpe,
        "sortinoZeroRate": sortino,
        "calmar": calmar,
        "maxDrawdown": maximum_drawdown,
        "dailyVaR95": var_95,
        "dailyCVaR95": cvar_95,
        "skewness": float(skew(values, bias=False)),
        "pearsonKurtosis": float(kurtosis(values, fisher=False, bias=False)),
        "positiveDayRate": float((values > 0).mean()),
        "probabilisticSharpe": probabilistic_sharpe_ratio(values),
        "deflatedSharpeProbability": deflated_sharpe_probability(
            values,
            model_trials,
        ),
    }


def block_bootstrap(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    samples: int,
    block_length: int,
    seed: int,
) -> dict[str, Any]:
    aligned = pd.concat(
        [strategy_returns.rename("strategy"), benchmark_returns.rename("benchmark")],
        axis=1,
    ).dropna()
    values = aligned.to_numpy(dtype=float)
    if len(values) < block_length * 2:
        return {}
    generator = np.random.default_rng(seed)
    annual_returns: list[float] = []
    sharpes: list[float] = []
    excess_returns: list[float] = []
    blocks_needed = math.ceil(len(values) / block_length)
    maximum_start = len(values) - block_length
    for _ in range(samples):
        starts = generator.integers(0, maximum_start + 1, size=blocks_needed)
        draw = np.concatenate(
            [values[start : start + block_length] for start in starts],
            axis=0,
        )[: len(values)]
        strategy = pd.Series(draw[:, 0])
        benchmark = pd.Series(draw[:, 1])
        strategy_metrics = performance_metrics(strategy)
        benchmark_metrics = performance_metrics(benchmark)
        annual_returns.append(strategy_metrics["annualizedReturn"])
        sharpes.append(strategy_metrics["sharpeZeroRate"])
        excess_returns.append(
            strategy_metrics["annualizedReturn"]
            - benchmark_metrics["annualizedReturn"]
        )
    return {
        "annualizedReturn95CI": [
            float(np.quantile(annual_returns, 0.025)),
            float(np.quantile(annual_returns, 0.975)),
        ],
        "sharpe95CI": [
            float(np.quantile(sharpes, 0.025)),
            float(np.quantile(sharpes, 0.975)),
        ],
        "probabilityAnnualReturnAboveBenchmark": float(
            np.mean(np.asarray(excess_returns) > 0)
        ),
        "medianAnnualExcessReturn": float(np.median(excess_returns)),
    }
