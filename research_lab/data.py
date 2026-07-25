from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ResearchConfig


FIELDS = ("open", "high", "low", "close", "adjusted_close", "volume")


def _timestamp(date: str) -> int:
    return int(datetime.fromisoformat(date).replace(tzinfo=timezone.utc).timestamp())


def fetch_symbol(symbol: str, config: ResearchConfig) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "period1": _timestamp(config.data_start),
            "period2": _timestamp(config.data_end),
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        }
    )
    encoded = urllib.parse.quote(symbol, safe="")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)["chart"]["result"][0]
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    adjusted = (
        ((result.get("indicators") or {}).get("adjclose") or [{}])[0].get(
            "adjclose"
        )
        or []
    )
    rows: list[dict[str, Any]] = []
    for index, stamp in enumerate(timestamps):
        close = (quote.get("close") or [None] * len(timestamps))[index]
        adjusted_close = adjusted[index] if index < len(adjusted) else None
        if close is None or adjusted_close is None or close <= 0 or adjusted_close <= 0:
            continue
        adjustment = adjusted_close / close
        row = {
            "date": datetime.fromtimestamp(stamp, timezone.utc).date().isoformat(),
            "open": _adjusted_value(quote, "open", index, adjustment),
            "high": _adjusted_value(quote, "high", index, adjustment),
            "low": _adjusted_value(quote, "low", index, adjustment),
            "close": float(close * adjustment),
            "adjusted_close": float(adjusted_close),
            "volume": float((quote.get("volume") or [0] * len(timestamps))[index] or 0),
        }
        if all(row[field] is not None for field in ("open", "high", "low")):
            rows.append(row)
    if len(rows) < config.minimum_history_days:
        raise RuntimeError(f"{symbol} has only {len(rows)} usable daily rows")
    return rows


def _adjusted_value(
    quote: dict[str, Any],
    field: str,
    index: int,
    adjustment: float,
) -> float | None:
    values = quote.get(field) or []
    if index >= len(values) or values[index] is None:
        return None
    return float(values[index] * adjustment)


def load_or_download(
    root: Path,
    symbols: list[str],
    config: ResearchConfig,
    *,
    refresh: bool,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    cache_path = config.cache_path(root)
    if cache_path.exists() and not refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        cached = payload.get("data") or {}
        if all(symbol in cached for symbol in symbols):
            return cached, payload["manifest"]

    data: dict[str, list[dict[str, Any]]] = {}
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(fetch_symbol, symbol, config): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                data[symbol] = future.result()
            except Exception as exc:
                errors.append(f"{symbol}: {type(exc).__name__}: {exc}")
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    manifest = {
        "downloadedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "Yahoo Finance chart endpoint",
        "requestedSymbols": len(symbols),
        "availableSymbols": len(data),
        "errors": errors,
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "dateRange": {"start": config.data_start, "end": config.data_end},
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"manifest": manifest, "data": data}
    temp = cache_path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temp.replace(cache_path)
    return data, manifest


def to_frames(
    raw_data: dict[str, list[dict[str, Any]]],
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for symbol, rows in raw_data.items():
        frame = pd.DataFrame(rows)
        if frame.empty:
            continue
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date").sort_index()
        frames[symbol] = frame[list(FIELDS)].astype(float)
    return frames
