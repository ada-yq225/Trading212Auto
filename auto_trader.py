#!/usr/bin/env python3
"""Research-informed long-only momentum strategy for Trading 212 Demo only."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any
from zoneinfo import ZoneInfo

from t212_demo import DEMO_BASE_URL, Trading212Error, make_client


ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "strategy.json"
UNIVERSE_FILE = ROOT / "universe.json"
OUTPUT_DIR = ROOT / "outputs" / "auto_trader"
STATE_FILE = OUTPUT_DIR / "state.json"
JOURNAL_FILE = OUTPUT_DIR / "journal.jsonl"
PID_FILE = OUTPUT_DIR / "runner.pid"
STOP_FILE = OUTPUT_DIR / "STOP"
STRATEGY_VERSION = "rational_momentum_v2"


@dataclass(frozen=True)
class Config:
    poll_seconds: float = 15.0
    short_samples: int = 20
    medium_samples: int = 60
    long_samples: int = 180
    volatility_samples: int = 120
    rebalance_seconds: float = 300.0
    entry_score: float = 0.75
    exit_score: float = -0.10
    top_n: int = 6
    cash_reserve_pct: float = 0.05
    max_position_pct: float = 0.30
    max_sector_pct: float = 0.40
    risk_on_gross_pct: float = 0.95
    neutral_gross_pct: float = 0.65
    risk_off_gross_pct: float = 0.25
    market_risk_on_return: float = 0.002
    market_risk_off_return: float = -0.003
    drawdown_cut_pct: float = 0.08
    drawdown_emergency_pct: float = 0.12
    hard_stop_loss_pct: float = 0.08
    trailing_stop_floor_pct: float = 0.05
    trailing_stop_ceiling_pct: float = 0.15
    minimum_position_value_gbp: float = 3.0
    minimum_order_value_gbp: float = 20.0
    maximum_order_value_gbp: float = 300.0
    rebalance_band_pct: float = 0.015
    cooldown_seconds: float = 300.0
    max_orders_per_day: int = 0
    snapshot_log_seconds: float = 60.0

    @classmethod
    def load(cls, path: Path = CONFIG_FILE) -> "Config":
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        config = cls(**data)
        if not (
            2 <= config.short_samples
            < config.medium_samples
            < config.long_samples
        ):
            raise ValueError("动量采样窗口必须满足 2 <= short < medium < long")
        if config.volatility_samples < config.medium_samples:
            raise ValueError("volatility_samples 不能低于 medium_samples")
        if config.poll_seconds < 1.1:
            raise ValueError("poll_seconds 不能低于 1.1 秒")
        for value in (
            config.cash_reserve_pct,
            config.max_position_pct,
            config.max_sector_pct,
            config.hard_stop_loss_pct,
        ):
            if not 0 < value < 1:
                raise ValueError("百分比参数必须在 0 和 1 之间")
        if not 0 < config.top_n:
            raise ValueError("top_n 必须大于 0")
        return config


@dataclass(frozen=True)
class Position:
    ticker: str
    quantity: float
    available: float
    current_price: float
    unit_account_value: float
    current_value: float
    total_cost: float
    unrealized_pnl: float

    @property
    def pnl_pct(self) -> float:
        return self.unrealized_pnl / self.total_cost if self.total_cost > 0 else 0.0


@dataclass(frozen=True)
class SignalMetrics:
    score: float
    short_return: float
    medium_return: float
    long_return: float
    volatility: float
    consistency: float


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_journal(event: str, **fields: Any) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    record = {"time": utc_now_iso(), "event": event, **fields}
    with JOURNAL_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def atomic_json_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def parse_positions(raw_positions: list[dict[str, Any]]) -> list[Position]:
    result: list[Position] = []
    for raw in raw_positions:
        instrument = raw.get("instrument") or {}
        wallet = raw.get("walletImpact") or {}
        quantity = float(raw.get("quantity") or 0)
        current_value = float(wallet.get("currentValue") or 0)
        if not instrument.get("ticker") or quantity <= 0 or current_value <= 0:
            continue
        result.append(
            Position(
                ticker=str(instrument["ticker"]),
                quantity=quantity,
                available=float(raw.get("quantityAvailableForTrading") or 0),
                current_price=float(raw.get("currentPrice") or 0),
                unit_account_value=current_value / quantity,
                current_value=current_value,
                total_cost=float(wallet.get("totalCost") or 0),
                unrealized_pnl=float(wallet.get("unrealizedProfitLoss") or 0),
            )
        )
    return result


def log_returns(values: list[float]) -> list[float]:
    return [
        math.log(current / previous)
        for previous, current in zip(values, values[1:])
        if previous > 0 and current > 0
    ]


def sample_volatility(values: list[float], count: int) -> float:
    returns = log_returns(values[-(count + 1):])
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((item - mean) ** 2 for item in returns) / (len(returns) - 1)
    return math.sqrt(max(variance, 0.0))


def momentum_metrics(values: list[float], config: Config) -> SignalMetrics | None:
    """Compute a volatility-normalized multi-horizon momentum score."""
    if len(values) <= config.long_samples or any(value <= 0 for value in values):
        return None
    volatility = sample_volatility(values, config.volatility_samples)
    volatility = max(volatility, 1e-6)

    def horizon_return(samples: int) -> float:
        return math.log(values[-1] / values[-1 - samples])

    short_return = horizon_return(config.short_samples)
    medium_return = horizon_return(config.medium_samples)
    long_return = horizon_return(config.long_samples)

    def normalized(value: float, samples: int) -> float:
        return value / (volatility * math.sqrt(samples))

    recent_returns = log_returns(values[-(config.medium_samples + 1):])
    positive_fraction = (
        sum(item > 0 for item in recent_returns) / len(recent_returns)
        if recent_returns
        else 0.5
    )
    consistency = 2.0 * (positive_fraction - 0.5)
    score = (
        0.20 * normalized(short_return, config.short_samples)
        + 0.30 * normalized(medium_return, config.medium_samples)
        + 0.50 * normalized(long_return, config.long_samples)
        + 0.25 * consistency
    )
    return SignalMetrics(
        score=max(-5.0, min(5.0, score)),
        short_return=short_return,
        medium_return=medium_return,
        long_return=long_return,
        volatility=volatility,
        consistency=consistency,
    )


def market_regime(
    metrics: dict[str, SignalMetrics],
    config: Config,
) -> tuple[str, float]:
    if not metrics:
        return "WARMUP", 0.0
    broad_return = median(item.long_return for item in metrics.values())
    if broad_return >= config.market_risk_on_return:
        return "RISK_ON", broad_return
    if broad_return <= config.market_risk_off_return:
        return "RISK_OFF", broad_return
    return "NEUTRAL", broad_return


def gross_exposure_for_regime(
    regime: str,
    drawdown: float,
    config: Config,
) -> float:
    gross = {
        "RISK_ON": config.risk_on_gross_pct,
        "NEUTRAL": config.neutral_gross_pct,
        "RISK_OFF": config.risk_off_gross_pct,
        "WARMUP": config.risk_off_gross_pct,
    }[regime]
    if drawdown >= config.drawdown_emergency_pct:
        return min(gross, 0.15)
    if drawdown >= config.drawdown_cut_pct:
        return min(gross, 0.40)
    return gross


def floor_quantity(value: float, precision: int = 3) -> float:
    """Round down to the quantity precision accepted by current equity orders."""
    factor = 10**precision
    return math.floor(max(value, 0.0) * factor) / factor


def is_regular_market_time(ticker: str, now_utc: datetime | None = None) -> bool:
    now_utc = now_utc or datetime.now(timezone.utc)
    if "_US_" in ticker:
        local = now_utc.astimezone(ZoneInfo("America/New_York"))
        start, end = (9, 35), (15, 50)
    else:
        local = now_utc.astimezone(ZoneInfo("Europe/London"))
        start, end = (8, 5), (16, 20)
    if local.weekday() >= 5:
        return False
    minutes = local.hour * 60 + local.minute
    return start[0] * 60 + start[1] <= minutes <= end[0] * 60 + end[1]


def backoff_seconds(base: float, consecutive_errors: int, cap: float = 300.0) -> float:
    if consecutive_errors <= 0:
        return base
    return min(cap, base * (2 ** min(consecutive_errors, 6)))


def target_weights(
    metrics: dict[str, SignalMetrics],
    sector_by_ticker: dict[str, str],
    gross_exposure: float,
    config: Config,
) -> dict[str, float]:
    """Allocate to the strongest names using bounded inverse-volatility weights."""
    ranked = sorted(
        (
            (item.score, ticker, item)
            for ticker, item in metrics.items()
            if item.score >= config.entry_score and item.long_return > 0
        ),
        reverse=True,
    )[: config.top_n]
    if not ranked:
        return {}
    inverse_vols = [1.0 / item.volatility for _, _, item in ranked]
    middle = median(inverse_vols)
    raw = {
        ticker: max(0.5 * middle, min(2.0 * middle, 1.0 / item.volatility))
        for _, ticker, item in ranked
    }
    total_raw = sum(raw.values())
    weights: dict[str, float] = {}
    sector_weights: dict[str, float] = {}
    for _, ticker, _ in ranked:
        proposed = gross_exposure * raw[ticker] / total_raw
        sector = sector_by_ticker.get(ticker, "Unknown")
        sector_room = config.max_sector_pct - sector_weights.get(sector, 0.0)
        weight = max(0.0, min(proposed, config.max_position_pct, sector_room))
        if weight:
            weights[ticker] = weight
            sector_weights[sector] = sector_weights.get(sector, 0.0) + weight
    return weights


def load_state(config: Config) -> dict[str, Any]:
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    else:
        state = {}
    state.setdefault("startedAt", utc_now_iso())
    state.setdefault("lastCycleAt", None)
    state.setdefault("lastTrade", {})
    state.setdefault("priceHistory", {})
    state.setdefault("watchlist", [])
    today = datetime.now(timezone.utc).date().isoformat()
    if state.get("orderDate") != today:
        state["orderDate"] = today
        state["ordersToday"] = 0
    state.setdefault("ordersToday", 0)
    state["config"] = asdict(config)
    state["environment"] = DEMO_BASE_URL
    return state


def load_universe_config(path: Path = UNIVERSE_FILE) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_active_universe(path: Path = UNIVERSE_FILE) -> set[str]:
    data = load_universe_config(path)
    return {str(ticker) for ticker in data.get("tickers", []) if ticker}


class Runner:
    def __init__(self, config: Config, execute: bool):
        self.config = config
        self.execute = execute
        self.client = make_client()
        self.state = load_state(config)
        if self.state.get("strategyVersion") != STRATEGY_VERSION:
            self.state["strategyVersion"] = STRATEGY_VERSION
            self.state["priceHistory"] = {}
            self.state["pricePeaks"] = {}
            self.state["lastRebalance"] = 0
            self.state.pop("portfolioHighWatermark", None)
        self.universe_config = load_universe_config()
        self.base_universe = {
            str(ticker) for ticker in self.universe_config.get("tickers", []) if ticker
        }
        self.scout_quantities = {
            str(item["ticker"]): float(item["probe_quantity"])
            for item in self.universe_config.get("scouts", [])
            if item.get("ticker") and item.get("probe_quantity")
        }
        self.scout_estimated_values = {
            str(item["ticker"]): float(item.get("estimated_value_gbp", 3.0))
            for item in self.universe_config.get("scouts", [])
            if item.get("ticker")
        }
        promoted = {str(ticker) for ticker in self.state.get("promotedScouts", [])}
        self.active_universe = self.base_universe | promoted
        self.price_universe = self.active_universe | set(self.scout_quantities)
        scout_symbols = {
            str(item["ticker"]): str(item.get("symbol") or "")
            for item in self.universe_config.get("scouts", [])
            if item.get("ticker")
        }
        self.sector_by_ticker: dict[str, str] = {}
        for sector, symbols in self.universe_config.get("sectors", {}).items():
            symbol_set = {str(symbol) for symbol in symbols}
            for ticker in self.price_universe:
                symbol = scout_symbols.get(ticker) or ticker.split("_", 1)[0]
                if symbol in symbol_set:
                    self.sector_by_ticker[ticker] = str(sector)
        self.state["activeUniverse"] = sorted(self.active_universe)
        self.state["scoutUniverse"] = sorted(self.scout_quantities)
        self.running = True
        self.last_snapshot_log = 0.0
        self.last_market_closed_log = 0.0
        self.consecutive_errors = 0
        self.last_error_text: str | None = None

    def stop(self, *_: Any) -> None:
        self.running = False

    def _sleep_interruptibly(self, delay: float) -> None:
        deadline = time.monotonic() + delay
        while self.running and not STOP_FILE.exists() and time.monotonic() < deadline:
            time.sleep(min(0.5, deadline - time.monotonic()))

    def _reset_daily_counter(self) -> None:
        today = datetime.now(timezone.utc).date().isoformat()
        if self.state.get("orderDate") != today:
            self.state["orderDate"] = today
            self.state["ordersToday"] = 0

    def _record_prices(self, positions: list[Position]) -> None:
        history_map = self.state["priceHistory"]
        maxlen = max(
            self.config.long_samples + 1,
            self.config.volatility_samples + 1,
            int(
                self.universe_config.get(
                    "scout_history_samples",
                    self.config.long_samples + 1,
                )
            ),
        )
        for position in positions:
            if self.price_universe and position.ticker not in self.price_universe:
                continue
            values = deque(history_map.get(position.ticker, []), maxlen=maxlen)
            values.append(position.unit_account_value)
            history_map[position.ticker] = list(values)
            peaks = self.state.setdefault("pricePeaks", {})
            peaks[position.ticker] = max(
                float(peaks.get(position.ticker, 0.0)),
                position.unit_account_value,
            )
            if position.ticker not in self.state["watchlist"]:
                self.state["watchlist"].append(position.ticker)

    def _seed_one_scout(
        self,
        account: dict[str, Any],
        positions: list[Position],
        now: float,
    ) -> bool:
        if not self.execute or not self.scout_quantities:
            return False
        seed_interval = float(
            self.universe_config.get("scout_seed_interval_seconds", 900)
        )
        last_seed = float(self.state.get("lastScoutSeed", 0))
        if now - last_seed < seed_interval:
            return False
        existing = {position.ticker for position in positions}
        attempted = set(self.state.get("scoutAttempts", []))
        cash = float((account.get("cash") or {}).get("availableToTrade") or 0)
        total = float(account.get("totalValue") or 0)
        reserve = total * self.config.cash_reserve_pct
        for ticker, quantity in self.scout_quantities.items():
            if ticker in existing or ticker in attempted or not is_regular_market_time(ticker):
                continue
            estimated_value = self.scout_estimated_values.get(ticker, 3.0)
            if cash < estimated_value + reserve:
                return False
            attempted.add(ticker)
            self.state["scoutAttempts"] = sorted(attempted)
            self.state["lastScoutSeed"] = now
            self.state["lastTrade"][ticker] = now
            self.state["lastAttempt"] = {
                "time": utc_now_iso(),
                "ticker": ticker,
                "quantity": quantity,
                "reason": "scout probe",
            }
            atomic_json_write(STATE_FILE, self.state)
            response = self.client.market_order(ticker, quantity, False)
            append_journal(
                "SCOUT_SEED_ORDER",
                request={"ticker": ticker, "quantity": quantity, "type": "MARKET"},
                response=response,
            )
            self.state["ordersToday"] += 1
            self.state["lastOrder"] = response
            return True
        return False

    def _evaluate_scouts(self, positions: list[Position], now: float) -> None:
        interval = float(self.universe_config.get("scout_interval_seconds", 1800))
        last = float(self.state.get("lastScoutEvaluation", 0))
        if now - last < interval:
            return
        self.state["lastScoutEvaluation"] = now
        min_samples = int(
            self.universe_config.get(
                "scout_min_samples",
                self.config.long_samples + 1,
            )
        )
        threshold = float(self.universe_config.get("scout_promotion_score", 0.75))
        existing = {position.ticker for position in positions}
        scores: dict[str, dict[str, float]] = {}
        for ticker in self.scout_quantities:
            if ticker in self.active_universe or ticker not in existing:
                continue
            values = self.state["priceHistory"].get(ticker, [])
            if len(values) < min_samples:
                continue
            metrics = momentum_metrics(values, self.config)
            if metrics is not None:
                scores[ticker] = asdict(metrics)
        append_journal("SCOUT_EVALUATION", scores=scores, threshold=threshold)
        eligible = [
            (item["score"], ticker)
            for ticker, item in scores.items()
            if item["score"] >= threshold and item["long_return"] > 0
        ]
        if not eligible:
            return
        score, ticker = max(eligible)
        self.active_universe.add(ticker)
        self.price_universe.add(ticker)
        promoted = set(self.state.get("promotedScouts", []))
        promoted.add(ticker)
        self.state["promotedScouts"] = sorted(promoted)
        self.state["activeUniverse"] = sorted(self.active_universe)
        append_journal("SCOUT_PROMOTED", ticker=ticker, score=score)

    def _cooldown_ready(self, ticker: str, now: float) -> bool:
        last = float(self.state["lastTrade"].get(ticker, 0))
        return now - last >= self.config.cooldown_seconds

    def _signal_metrics(
        self,
        positions: list[Position],
    ) -> dict[str, SignalMetrics]:
        existing = {position.ticker for position in positions}
        result: dict[str, SignalMetrics] = {}
        for ticker in self.active_universe & existing:
            metrics = momentum_metrics(
                self.state["priceHistory"].get(ticker, []),
                self.config,
            )
            if metrics is not None:
                result[ticker] = metrics
        return result

    def _drawdown(self, account: dict[str, Any]) -> float:
        total = float(account.get("totalValue") or 0)
        high = max(float(self.state.get("portfolioHighWatermark", 0.0)), total)
        self.state["portfolioHighWatermark"] = high
        return (high - total) / high if high > 0 else 0.0

    def _risk_exit(
        self,
        position: Position,
        metrics: SignalMetrics | None,
    ) -> str | None:
        if position.pnl_pct <= -self.config.hard_stop_loss_pct:
            return f"hard stop {position.pnl_pct:.2%}"
        peak = float(self.state.get("pricePeaks", {}).get(position.ticker, 0.0))
        if peak <= 0:
            return None
        volatility = metrics.volatility if metrics else 0.0
        trailing_pct = max(
            self.config.trailing_stop_floor_pct,
            min(
                self.config.trailing_stop_ceiling_pct,
                3.0 * volatility * math.sqrt(self.config.medium_samples),
            ),
        )
        trailing_drawdown = position.unit_account_value / peak - 1.0
        if trailing_drawdown <= -trailing_pct:
            return f"trailing stop {trailing_drawdown:.2%} / {trailing_pct:.2%}"
        return None

    def _candidates(
        self,
        account: dict[str, Any],
        positions: list[Position],
        now: float,
    ) -> tuple[list[tuple[float, str, Position, float, str]], dict[str, Any]]:
        active_positions = [
            position for position in positions if position.ticker in self.active_universe
        ]
        metrics = self._signal_metrics(active_positions)
        regime, broad_return = market_regime(metrics, self.config)
        drawdown = self._drawdown(account)
        gross = gross_exposure_for_regime(regime, drawdown, self.config)
        weights = target_weights(metrics, self.sector_by_ticker, gross, self.config)
        total = float(account.get("totalValue") or 0)
        candidates: list[tuple[float, str, Position, float, str]] = []
        for position in active_positions:
            if not self._cooldown_ready(position.ticker, now):
                continue
            item_metrics = metrics.get(position.ticker)
            risk_reason = self._risk_exit(position, item_metrics)
            if risk_reason:
                target_value = self.config.minimum_position_value_gbp
                priority = 10.0 + abs(position.current_value - target_value) / max(total, 1)
                candidates.append(
                    (priority, "SELL_RISK", position, target_value, risk_reason)
                )
                continue
            target_value = total * weights.get(position.ticker, 0.0)
            if item_metrics and item_metrics.score <= self.config.exit_score:
                target_value = self.config.minimum_position_value_gbp
            elif item_metrics is None:
                target_value = min(
                    position.current_value,
                    total * self.config.max_position_pct,
                )
            target_value = max(
                self.config.minimum_position_value_gbp,
                min(target_value, total * self.config.max_position_pct),
            )
            delta = target_value - position.current_value
            band = max(
                self.config.minimum_order_value_gbp,
                total * self.config.rebalance_band_pct,
            )
            if abs(delta) < band:
                continue
            action = "BUY_TARGET" if delta > 0 else "SELL_TARGET"
            score = item_metrics.score if item_metrics else 0.0
            reason = (
                f"{regime} score={score:.3f} target={target_value:.2f} "
                f"current={position.current_value:.2f}"
            )
            priority = abs(delta) / max(total, 1)
            if delta < 0:
                priority += 1.0
            candidates.append((priority, action, position, target_value, reason))
        diagnostics = {
            "strategyVersion": STRATEGY_VERSION,
            "regime": regime,
            "broadReturn": broad_return,
            "drawdown": drawdown,
            "grossTarget": gross,
            "scores": {ticker: asdict(item) for ticker, item in metrics.items()},
            "targetWeights": weights,
        }
        return sorted(candidates, key=lambda item: item[0], reverse=True), diagnostics

    def _quantity_for_target(
        self,
        action: str,
        position: Position,
        target_value: float,
        account: dict[str, Any],
    ) -> float:
        total = float(account.get("totalValue") or 0)
        cash = float((account.get("cash") or {}).get("availableToTrade") or 0)
        delta = target_value - position.current_value
        if action.startswith("BUY"):
            reserve = total * self.config.cash_reserve_pct
            amount = min(delta, cash - reserve, self.config.maximum_order_value_gbp)
            if amount < self.config.minimum_order_value_gbp:
                return 0.0
            return floor_quantity(amount / position.unit_account_value)
        amount = min(
            -delta,
            self.config.maximum_order_value_gbp,
            position.current_value - self.config.minimum_position_value_gbp,
        )
        if amount < self.config.minimum_order_value_gbp:
            return 0.0
        return floor_quantity(
            min(amount / position.unit_account_value, position.available)
        )

    def cycle(self) -> None:
        self._reset_daily_counter()
        account = self.client.account_summary()
        positions = parse_positions(self.client.positions())
        now = time.time()
        self._record_prices(positions)
        self._evaluate_scouts(positions, now)
        self.state["lastCycleAt"] = utc_now_iso()
        self.state["lastAccount"] = account
        self.state["lastPositions"] = [asdict(item) for item in positions]
        candidates, diagnostics = self._candidates(account, positions, now)
        self.state["lastDiagnostics"] = diagnostics

        if now - self.last_snapshot_log >= self.config.snapshot_log_seconds:
            append_journal(
                "MODEL_SNAPSHOT",
                totalValue=account.get("totalValue"),
                cash=(account.get("cash") or {}).get("availableToTrade"),
                positions={item.ticker: item.current_value for item in positions},
                ordersToday=self.state["ordersToday"],
                diagnostics=diagnostics,
            )
            self.last_snapshot_log = now

        # A value of 0 explicitly means no daily order-count ceiling.
        if self.config.max_orders_per_day > 0 and self.state["ordersToday"] >= self.config.max_orders_per_day:
            atomic_json_write(STATE_FILE, self.state)
            return

        last_rebalance = float(self.state.get("lastRebalance", 0))
        rebalance_due = now - last_rebalance >= self.config.rebalance_seconds
        for _, action, position, target_value, reason in candidates:
            if action != "SELL_RISK" and not rebalance_due:
                continue
            if not is_regular_market_time(position.ticker):
                continue
            quantity = self._quantity_for_target(
                action,
                position,
                target_value,
                account,
            )
            signed_quantity = quantity if action.startswith("BUY") else -quantity
            if quantity <= 0:
                continue
            append_journal(
                "SIGNAL",
                action=action,
                ticker=position.ticker,
                quantity=signed_quantity,
                reason=reason,
                execute=self.execute,
            )
            if self.execute:
                # Record and persist the attempt before calling this non-idempotent
                # endpoint. Even an ambiguous network failure must not be retried
                # on the next cycle.
                self.state["lastTrade"][position.ticker] = now
                self.state["lastAttempt"] = {
                    "time": utc_now_iso(),
                    "ticker": position.ticker,
                    "quantity": signed_quantity,
                    "reason": reason,
                }
                atomic_json_write(STATE_FILE, self.state)
                response = self.client.market_order(position.ticker, signed_quantity, False)
                append_journal("ORDER_SUBMITTED", request={
                    "ticker": position.ticker,
                    "quantity": signed_quantity,
                    "type": "MARKET",
                }, response=response, reason=reason)
                self.state["ordersToday"] += 1
                self.state["lastOrder"] = response
                self.state["lastRebalance"] = now
                if target_value <= self.config.minimum_position_value_gbp + 0.01:
                    self.state.setdefault("pricePeaks", {})[position.ticker] = (
                        position.unit_account_value
                    )
            atomic_json_write(STATE_FILE, self.state)
            return  # At most one non-idempotent order request per cycle.

        if rebalance_due:
            self.state["lastRebalance"] = now
        if self._seed_one_scout(account, positions, now):
            atomic_json_write(STATE_FILE, self.state)
            return
        atomic_json_write(STATE_FILE, self.state)

    def run(self) -> int:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        STOP_FILE.unlink(missing_ok=True)
        PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
        os.chmod(PID_FILE, 0o600)
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        append_journal(
            "RUNNER_STARTED",
            pid=os.getpid(),
            execute=self.execute,
            config=asdict(self.config),
            activeUniverse=sorted(self.active_universe),
            scoutUniverse=sorted(self.scout_quantities),
        )
        print(f"Demo 自动策略已启动，PID={os.getpid()}，execute={self.execute}", flush=True)
        print(f"状态目录：{OUTPUT_DIR}", flush=True)
        try:
            while self.running and not STOP_FILE.exists():
                if self.active_universe and not any(is_regular_market_time(ticker) for ticker in self.active_universe):
                    now = time.time()
                    if now - self.last_market_closed_log >= 3600:
                        append_journal("MARKET_CLOSED_WAIT", retrySeconds=60)
                        self.last_market_closed_log = now
                    self._sleep_interruptibly(60)
                    continue
                started = time.monotonic()
                delay = self.config.poll_seconds
                try:
                    self.cycle()
                    if self.consecutive_errors:
                        append_journal("API_RECOVERED", previousConsecutiveErrors=self.consecutive_errors)
                    self.consecutive_errors = 0
                    self.last_error_text = None
                except Trading212Error as exc:
                    self.consecutive_errors += 1
                    error_text = str(exc)
                    should_log = (
                        error_text != self.last_error_text
                        or self.consecutive_errors & (self.consecutive_errors - 1) == 0
                    )
                    if should_log:
                        append_journal(
                            "API_ERROR",
                            error=error_text,
                            consecutive=self.consecutive_errors,
                        )
                        print(f"API 错误：{exc}", file=sys.stderr, flush=True)
                    self.last_error_text = error_text
                    delay = backoff_seconds(self.config.poll_seconds, self.consecutive_errors)
                except Exception as exc:  # Keep the monitored runner alive and record unexpected errors.
                    append_journal("UNEXPECTED_ERROR", error=repr(exc))
                    print(f"运行错误：{exc!r}", file=sys.stderr, flush=True)
                remaining = max(1.1, delay - (time.monotonic() - started))
                self._sleep_interruptibly(remaining)
        finally:
            append_journal("RUNNER_STOPPED", pid=os.getpid())
            PID_FILE.unlink(missing_ok=True)
        return 0


def process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def show_status() -> int:
    pid = int(PID_FILE.read_text().strip()) if PID_FILE.exists() else 0
    print(f"运行状态：{'运行中' if pid and process_is_running(pid) else '已停止'}")
    if pid:
        print(f"PID：{pid}")
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        account = state.get("lastAccount") or {}
        cash = account.get("cash") or {}
        print(f"最后循环：{state.get('lastCycleAt')}")
        print(f"总资产：{account.get('totalValue')} {account.get('currency', '')}")
        print(f"可用现金：{cash.get('availableToTrade')} {account.get('currency', '')}")
        limit = state.get('config', {}).get('max_orders_per_day', '-')
        limit_label = "无限" if limit == 0 else limit
        print(f"今日订单：{state.get('ordersToday', 0)} / {limit_label}")
        diagnostics = state.get("lastDiagnostics") or {}
        if diagnostics:
            print(
                f"模型：{diagnostics.get('strategyVersion')}，"
                f"状态 {diagnostics.get('regime')}，"
                f"组合回撤 {float(diagnostics.get('drawdown') or 0):.2%}，"
                f"目标风险仓位 {float(diagnostics.get('grossTarget') or 0):.0%}"
            )
        promoted = state.get("promotedScouts", [])
        print(f"侦察晋升：{', '.join(promoted) if promoted else '暂无'}")
        print("持仓：")
        for item in state.get("lastPositions", []):
            print(f"  {item['ticker']}: {item['quantity']}，市值 {item['current_value']:.2f}，浮盈亏 {item['unrealized_pnl']:.2f}")
    if JOURNAL_FILE.exists():
        lines = JOURNAL_FILE.read_text(encoding="utf-8").splitlines()
        orders = [json.loads(line) for line in lines if '"event": "ORDER_SUBMITTED"' in line]
        print(f"策略累计提交订单：{len(orders)}")
        for order in orders[-5:]:
            req = order.get("request") or {}
            response = order.get("response") or {}
            print(f"  {order.get('time')} {req.get('ticker')} {req.get('quantity')}，订单 {response.get('id')} {response.get('status')}")
    return 0


def request_stop() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STOP_FILE.write_text(utc_now_iso(), encoding="utf-8")
    print("已写入停止请求，策略将在当前循环结束后退出。")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trading 212 Demo 自动策略")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="持续运行策略")
    run.add_argument("--execute-demo", action="store_true", help="允许向 Demo 账户提交订单；否则只记录信号")
    sub.add_parser("status", help="显示策略状态与最近订单")
    sub.add_parser("stop", help="请求停止持续策略")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "status":
        return show_status()
    if args.command == "stop":
        return request_stop()
    config = Config.load()
    return Runner(config, execute=args.execute_demo).run()


if __name__ == "__main__":
    raise SystemExit(main())
