from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from .config import ResearchConfig


RETURN_WINDOWS = (1, 5, 21, 63, 126, 252)
VOLATILITY_WINDOWS = (5, 21, 63)
RANK_COLUMNS = (
    "return_5",
    "return_21",
    "return_63",
    "return_126",
    "volatility_21",
    "drawdown_63",
    "volume_z_20",
)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    change = close.diff()
    gain = change.clip(lower=0).rolling(window).mean()
    loss = (-change.clip(upper=0)).rolling(window).mean()
    relative = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + relative)


def _atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window).mean() / frame["close"]


def _downside_volatility(returns: pd.Series, window: int) -> pd.Series:
    downside = returns.where(returns < 0, 0.0)
    return downside.rolling(window).std()


def asset_features(
    frame: pd.DataFrame,
    config: ResearchConfig,
) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    close = frame["adjusted_close"]
    log_returns = np.log(close).diff()
    for window in RETURN_WINDOWS:
        result[f"return_{window}"] = np.log(close / close.shift(window))
    for window in VOLATILITY_WINDOWS:
        result[f"volatility_{window}"] = log_returns.rolling(window).std()
        result[f"downside_volatility_{window}"] = _downside_volatility(
            log_returns,
            window,
        )
    result["skew_21"] = log_returns.rolling(21).skew()
    result["skew_63"] = log_returns.rolling(63).skew()
    result["kurtosis_63"] = log_returns.rolling(63).kurt()
    for window in (21, 63, 252):
        result[f"drawdown_{window}"] = close / close.rolling(window).max() - 1.0
    for window in (10, 20, 50, 100, 200):
        result[f"distance_sma_{window}"] = close / close.rolling(window).mean() - 1.0
    result["rsi_14"] = _rsi(close) / 100.0
    result["atr_14"] = _atr(frame)
    result["intraday_range"] = (frame["high"] - frame["low"]) / close
    result["overnight_gap"] = frame["open"] / close.shift(1) - 1.0
    log_volume = np.log1p(frame["volume"])
    volume_mean = log_volume.rolling(20).mean()
    volume_std = log_volume.rolling(20).std()
    result["volume_z_20"] = (log_volume - volume_mean) / volume_std.replace(0, np.nan)
    result["volume_change_5"] = log_volume - log_volume.shift(5)
    result["momentum_interaction"] = result["return_21"] * result["return_126"]
    result["volatility_ratio"] = result["volatility_21"] / result[
        "volatility_63"
    ].replace(0, np.nan)
    horizon = config.label_horizon_days
    result["future_return"] = close.shift(-horizon) / close - 1.0
    risk_scale = result["volatility_63"].clip(lower=1e-5) * math.sqrt(horizon)
    result["target"] = (result["future_return"] / risk_scale).clip(-5.0, 5.0)
    result["label_end_date"] = pd.Series(
        frame.index,
        index=frame.index,
    ).shift(-horizon)
    return result


def _context_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    context = pd.DataFrame()

    def add_return(symbol: str, name: str, windows: Iterable[int]) -> None:
        nonlocal context
        if symbol not in frames:
            return
        close = frames[symbol]["adjusted_close"]
        for window in windows:
            context[f"{name}_return_{window}"] = np.log(
                close / close.shift(window)
            )

    add_return("SPY", "market", (5, 21, 63, 126, 252))
    add_return("QQQ", "nasdaq", (21, 63, 126))
    add_return("IWM", "small_cap", (21, 63))
    add_return("HYG", "high_yield", (21, 63))
    add_return("TLT", "treasury", (21, 63))
    add_return("GLD", "gold", (21, 63))
    add_return("UUP", "dollar", (21, 63))
    if "SPY" in frames:
        spy = frames["SPY"]["adjusted_close"]
        spy_returns = np.log(spy).diff()
        context["market_volatility_21"] = spy_returns.rolling(21).std()
        context["market_drawdown_252"] = spy / spy.rolling(252).max() - 1.0
        context["market_above_sma_200"] = (
            spy > spy.rolling(200).mean()
        ).astype(float)
    if "^VIX" in frames:
        vix = frames["^VIX"]["adjusted_close"]
        context["vix_level"] = np.log(vix.clip(lower=1e-6))
        context["vix_change_5"] = np.log(vix / vix.shift(5))
        context["vix_change_21"] = np.log(vix / vix.shift(21))
    if {"QQQ", "SPY"} <= frames.keys():
        context["nasdaq_relative_63"] = (
            np.log(
                frames["QQQ"]["adjusted_close"]
                / frames["QQQ"]["adjusted_close"].shift(63)
            )
            - np.log(
                frames["SPY"]["adjusted_close"]
                / frames["SPY"]["adjusted_close"].shift(63)
            )
        )
    if {"HYG", "TLT"} <= frames.keys():
        context["credit_risk_proxy_21"] = np.log(
            frames["HYG"]["adjusted_close"]
            / frames["HYG"]["adjusted_close"].shift(21)
        ) - np.log(
            frames["TLT"]["adjusted_close"]
            / frames["TLT"]["adjusted_close"].shift(21)
        )
    return context.sort_index()


def build_panel(
    frames: dict[str, pd.DataFrame],
    equity_symbols: list[str],
    sectors: dict[str, str],
    config: ResearchConfig,
) -> tuple[pd.DataFrame, list[str]]:
    context = _context_frame(frames)
    pieces: list[pd.DataFrame] = []
    for symbol in equity_symbols:
        if symbol not in frames:
            continue
        features = asset_features(frames[symbol], config)
        features = features.join(context, how="left")
        features["symbol"] = symbol
        features["sector"] = sectors[symbol]
        features["date"] = features.index
        pieces.append(features.reset_index(drop=True))
    panel = pd.concat(pieces, ignore_index=True)
    for column in RANK_COLUMNS:
        panel[f"rank_{column}"] = panel.groupby("date")[column].rank(
            pct=True,
            method="average",
        )
    excluded = {
        "date",
        "symbol",
        "sector",
        "future_return",
        "target",
        "label_end_date",
    }
    feature_columns = [
        column
        for column in panel.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(panel[column])
    ]
    panel = panel.replace([np.inf, -np.inf], np.nan)
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True), feature_columns
