#!/usr/bin/env python3
"""Strict expanding-window backtest for the experimental ML ensemble."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import numpy as np

from auto_trader import (
    Config,
    SignalMetrics,
    gross_exposure_for_regime,
    target_weights,
)
from experimental_model import ExperimentSettings, ExperimentalEnsemble
from research_backtest import load_prices, load_universe, performance


ROOT = Path(__file__).resolve().parent
OUTPUT_FILE = ROOT / "outputs" / "experimental_backtest.json"
TRANSACTION_COST = 0.001
REBALANCE_DAYS = 21


def history_through(series: dict[str, float], date: str) -> list[float]:
    return [value for key, value in series.items() if key <= date]


def live_like_regime(spy_values: list[float]) -> str:
    if len(spy_values) < 200:
        return "WARMUP"
    long_return = spy_values[-1] / spy_values[-121] - 1.0
    above_mean = spy_values[-1] > mean(spy_values[-200:])
    if long_return > 0.05 and above_mean:
        return "RISK_ON"
    if long_return < -0.05 and not above_mean:
        return "RISK_OFF"
    return "NEUTRAL"


def run_expanding_backtest(
    prices: dict[str, dict[str, float]],
    sectors: dict[str, str],
    start: str,
    end: str,
) -> dict:
    settings = ExperimentSettings(
        horizon_samples=15,
        minimum_history=120,
        training_stride=5,
        minimum_training_samples=500,
        minimum_shadow_outcomes=40,
    )
    allocation_config = Config(
        short_samples=21,
        medium_samples=63,
        long_samples=126,
        volatility_samples=63,
        entry_score=0.0,
        top_n=6,
        max_position_pct=0.30,
        max_sector_pct=0.40,
        risk_on_gross_pct=0.95,
        neutral_gross_pct=0.65,
        risk_off_gross_pct=0.25,
    )
    calendar = [date for date in sorted(prices["SPY"]) if start <= date <= end]
    universe = [symbol for symbol in sectors if symbol in prices]
    weights: dict[str, float] = {}
    equity = 1.0
    high_watermark = 1.0
    curve: list[tuple[str, float]] = []
    training_counts: list[int] = []
    prediction_batches = 0
    total_turnover = 0.0

    for index, date in enumerate(calendar):
        if index:
            previous_date = calendar[index - 1]
            portfolio_return = 0.0
            for symbol, weight in weights.items():
                current = prices[symbol].get(date)
                previous = prices[symbol].get(previous_date)
                if current and previous:
                    portfolio_return += weight * (current / previous - 1.0)
            equity *= 1.0 + portfolio_return
            high_watermark = max(high_watermark, equity)

        if index % REBALANCE_DAYS == 0:
            histories = {
                symbol: history_through(prices[symbol], date) for symbol in universe
            }
            model = ExperimentalEnsemble(settings)
            training_count = model.fit(histories)
            training_counts.append(training_count)
            predictions = model.predict(histories)
            spy_values = history_through(prices["SPY"], date)
            regime = live_like_regime(spy_values)
            drawdown = (high_watermark - equity) / high_watermark
            gross = gross_exposure_for_regime(
                regime,
                drawdown,
                allocation_config,
            )
            signal_metrics: dict[str, SignalMetrics] = {}
            for symbol, prediction in predictions.items():
                values = histories[symbol]
                if len(values) < 121 or prediction <= 0:
                    continue
                recent_returns = np.diff(np.log(np.asarray(values[-61:], dtype=float)))
                volatility = max(float(np.std(recent_returns, ddof=1)), 1e-6)
                signal_metrics[symbol] = SignalMetrics(
                    score=prediction,
                    short_return=0.0,
                    medium_return=0.0,
                    long_return=1e-6,
                    volatility=volatility,
                    consistency=0.0,
                )
            new_weights = target_weights(
                signal_metrics,
                sectors,
                gross,
                allocation_config,
            )
            turnover = sum(
                abs(new_weights.get(symbol, 0.0) - weights.get(symbol, 0.0))
                for symbol in set(weights) | set(new_weights)
            )
            equity *= max(0.0, 1.0 - turnover * TRANSACTION_COST)
            total_turnover += turnover
            weights = new_weights
            if predictions:
                prediction_batches += 1
        curve.append((date, equity))

    benchmark = [
        (date, prices["SPY"][date] / prices["SPY"][calendar[0]])
        for date in calendar
    ]
    return {
        "period": {"start": start, "end": end},
        "strategy": performance(curve),
        "benchmarkSPY": performance(benchmark),
        "predictionBatches": prediction_batches,
        "averageTrainingSamples": mean(training_counts) if training_counts else 0,
        "averageTurnover": (
            total_turnover / prediction_batches if prediction_batches else 0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="实验机器学习策略样本外回测")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    symbols, sectors = load_universe()
    prices, errors = load_prices(symbols + ["SPY"], refresh=args.refresh)
    available = {
        symbol: sector for symbol, sector in sectors.items() if symbol in prices
    }
    result = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "50% standardized Ridge + 50% shallow histogram gradient boosting",
        "validationProtocol": (
            "Expanding-window monthly retraining. Each label is delayed 15 trading "
            "days, so no sample is trainable until its future outcome is known."
        ),
        "transactionCostPerUnitTurnover": TRANSACTION_COST,
        "downloadErrors": errors,
        "validation": run_expanding_backtest(
            prices,
            available,
            "2023-01-01",
            "2025-12-31",
        ),
        "limitations": [
            "The current 50-stock universe creates survivorship and selection bias.",
            "Daily historical validation is an analogue of the minute-sampled live model.",
            "The result excludes taxes and models spreads through a fixed turnover cost.",
            "The live model remains in shadow mode until it passes forward Demo gates.",
        ],
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(OUTPUT_FILE)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
