#!/usr/bin/env python3
"""Walk-forward-style research backtest for the strategy's daily analogue."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from auto_trader import (
    Config,
    gross_exposure_for_regime,
    market_regime,
    momentum_metrics,
    target_weights,
)


ROOT = Path(__file__).resolve().parent
CACHE_FILE = ROOT / ".cache" / "backtest_prices.json"
OUTPUT_FILE = ROOT / "outputs" / "research_backtest.json"
TRANSACTION_COST = 0.001
REBALANCE_DAYS = 21
PRESETS = {
    "balanced": {
        "entry_score": 0.75,
        "top_n": 6,
        "max_position_pct": 0.30,
        "max_sector_pct": 0.40,
        "risk_on_gross_pct": 0.95,
        "neutral_gross_pct": 0.65,
        "risk_off_gross_pct": 0.25,
    },
    "aggressive": {
        "entry_score": 0.50,
        "top_n": 4,
        "max_position_pct": 0.35,
        "max_sector_pct": 0.55,
        "risk_on_gross_pct": 0.98,
        "neutral_gross_pct": 0.75,
        "risk_off_gross_pct": 0.35,
    },
    "selective": {
        "entry_score": 1.00,
        "top_n": 3,
        "max_position_pct": 0.40,
        "max_sector_pct": 0.60,
        "risk_on_gross_pct": 0.95,
        "neutral_gross_pct": 0.55,
        "risk_off_gross_pct": 0.20,
    },
}


def load_universe() -> tuple[list[str], dict[str, str]]:
    data = json.loads((ROOT / "universe.json").read_text(encoding="utf-8"))
    symbols: list[str] = []
    sectors: dict[str, str] = {}
    for sector, items in data["sectors"].items():
        for symbol in items:
            symbols.append(symbol)
            sectors[symbol] = sector
    return symbols, sectors


def fetch_daily_prices(symbol: str) -> dict[str, float]:
    start = int(datetime(2018, 1, 1, tzinfo=timezone.utc).timestamp())
    end = int(datetime(2026, 1, 3, tzinfo=timezone.utc).timestamp())
    query = urllib.parse.urlencode(
        {
            "period1": start,
            "period2": end,
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        }
    )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=25) as response:
        result = json.load(response)["chart"]["result"][0]
    timestamps = result.get("timestamp") or []
    adjusted = ((result.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose") or []
    prices: dict[str, float] = {}
    for stamp, value in zip(timestamps, adjusted):
        if value is None or value <= 0:
            continue
        date = datetime.fromtimestamp(stamp, timezone.utc).date().isoformat()
        prices[date] = float(value)
    if len(prices) < 100:
        raise RuntimeError(f"{symbol} 历史数据不足")
    return prices


def load_prices(symbols: list[str], *, refresh: bool) -> tuple[dict[str, dict[str, float]], list[str]]:
    if CACHE_FILE.exists() and not refresh:
        cached = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if all(symbol in cached.get("prices", {}) for symbol in symbols):
            return cached["prices"], cached.get("errors", [])

    prices: dict[str, dict[str, float]] = {}
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(fetch_daily_prices, symbol): symbol for symbol in symbols
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                prices[symbol] = future.result()
            except Exception as exc:
                errors.append(f"{symbol}: {type(exc).__name__}: {exc}")
    payload = {
        "downloadedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "Yahoo Finance chart endpoint",
        "prices": prices,
        "errors": errors,
    }
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp = CACHE_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temp.replace(CACHE_FILE)
    return prices, errors


def price_history_through(
    series: dict[str, float],
    date: str,
    count: int,
) -> list[float]:
    eligible = [value for key, value in series.items() if key <= date]
    return eligible[-count:]


def performance(equity_curve: list[tuple[str, float]]) -> dict[str, float]:
    if len(equity_curve) < 2:
        return {}
    daily_returns = [
        current / previous - 1.0
        for (_, previous), (_, current) in zip(equity_curve, equity_curve[1:])
        if previous > 0
    ]
    years = max(len(daily_returns) / 252.0, 1 / 252)
    total_return = equity_curve[-1][1] / equity_curve[0][1] - 1.0
    annual_return = (1.0 + total_return) ** (1.0 / years) - 1.0
    volatility = statistics.stdev(daily_returns) * math.sqrt(252) if len(daily_returns) > 1 else 0.0
    sharpe = (
        statistics.mean(daily_returns) / statistics.stdev(daily_returns) * math.sqrt(252)
        if len(daily_returns) > 1 and statistics.stdev(daily_returns) > 0
        else 0.0
    )
    high = equity_curve[0][1]
    max_drawdown = 0.0
    for _, equity in equity_curve:
        high = max(high, equity)
        max_drawdown = max(max_drawdown, (high - equity) / high)
    return {
        "totalReturn": total_return,
        "annualizedReturn": annual_return,
        "annualizedVolatility": volatility,
        "sharpeZeroRate": sharpe,
        "maxDrawdown": max_drawdown,
    }


def run_backtest(
    prices: dict[str, dict[str, float]],
    sectors: dict[str, str],
    start: str,
    end: str,
    preset_name: str,
) -> dict[str, Any]:
    preset = PRESETS[preset_name]
    daily_config = Config(
        short_samples=21,
        medium_samples=63,
        long_samples=126,
        volatility_samples=63,
        entry_score=preset["entry_score"],
        exit_score=-0.10,
        top_n=preset["top_n"],
        max_position_pct=preset["max_position_pct"],
        max_sector_pct=preset["max_sector_pct"],
        risk_on_gross_pct=preset["risk_on_gross_pct"],
        neutral_gross_pct=preset["neutral_gross_pct"],
        risk_off_gross_pct=preset["risk_off_gross_pct"],
        market_risk_on_return=0.05,
        market_risk_off_return=-0.05,
    )
    calendar = [
        date for date in sorted(prices["SPY"]) if start <= date <= end
    ]
    equity = 1.0
    high_watermark = 1.0
    weights: dict[str, float] = {}
    curve: list[tuple[str, float]] = []
    turnovers: list[float] = []
    rebalance_count = 0
    universe = [symbol for symbol in sectors if symbol in prices]

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
            signals = {}
            for symbol in universe:
                values = price_history_through(
                    prices[symbol],
                    date,
                    daily_config.long_samples + 1,
                )
                item = momentum_metrics(values, daily_config)
                if item is not None:
                    signals[symbol] = item
            regime, _ = market_regime(signals, daily_config)
            drawdown = (high_watermark - equity) / high_watermark
            gross = gross_exposure_for_regime(regime, drawdown, daily_config)
            new_weights = target_weights(signals, sectors, gross, daily_config)
            turnover = sum(
                abs(new_weights.get(symbol, 0.0) - weights.get(symbol, 0.0))
                for symbol in set(weights) | set(new_weights)
            )
            equity *= max(0.0, 1.0 - turnover * TRANSACTION_COST)
            turnovers.append(turnover)
            weights = new_weights
            rebalance_count += 1
        curve.append((date, equity))

    benchmark = [
        (date, prices["SPY"][date] / prices["SPY"][calendar[0]])
        for date in calendar
        if date in prices["SPY"]
    ]
    return {
        "preset": preset_name,
        "period": {"start": start, "end": end},
        "strategy": performance(curve),
        "benchmarkSPY": performance(benchmark),
        "rebalances": rebalance_count,
        "averageTurnover": statistics.mean(turnovers) if turnovers else 0.0,
        "endingEquity": equity,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="研究版历史回测")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    symbols, sectors = load_universe()
    prices, errors = load_prices(symbols + ["SPY"], refresh=args.refresh)
    if "SPY" not in prices:
        raise RuntimeError("缺少 SPY 基准数据")
    available_sectors = {
        symbol: sector for symbol, sector in sectors.items() if symbol in prices
    }
    training = [
        run_backtest(
            prices,
            available_sectors,
            "2020-01-01",
            "2022-12-31",
            preset_name,
        )
        for preset_name in PRESETS
    ]
    chosen = max(
        training,
        key=lambda item: item["strategy"].get("sharpeZeroRate", float("-inf")),
    )["preset"]
    results = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy": "Cross-sectional multi-horizon momentum with inverse-volatility sizing",
        "dataSource": "Yahoo Finance adjusted daily close, used for research only",
        "transactionCostPerUnitTurnover": TRANSACTION_COST,
        "rebalanceTradingDays": REBALANCE_DAYS,
        "availableStocks": len(available_sectors),
        "downloadErrors": errors,
        "modelSelection": {
            "trainingPeriod": "2020-01-01 through 2022-12-31",
            "criterion": "highest zero-rate Sharpe among three predeclared presets",
            "trainingResults": training,
            "chosenPreset": chosen,
        },
        "validation": run_backtest(
            prices,
            available_sectors,
            "2023-01-01",
            "2025-12-31",
            chosen,
        ),
        "fullPeriodReference": run_backtest(
            prices,
            available_sectors,
            "2020-01-01",
            "2025-12-31",
            chosen,
        ),
        "limitations": [
            "The present-day stock universe creates survivorship and selection bias.",
            "The daily research analogue is not identical to the live intraday sampling implementation.",
            "Yahoo data is an external research source and may contain adjustments or gaps.",
            "Results are hypothetical, exclude taxes and spreads, and do not predict future returns.",
        ],
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(OUTPUT_FILE)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
