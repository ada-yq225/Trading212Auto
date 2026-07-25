from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResearchConfig:
    random_seed: int = 212
    data_start: str = "2017-01-01"
    data_end: str = "2026-01-03"
    validation_start: str = "2023-01-01"
    validation_end: str = "2025-12-31"
    label_horizon_days: int = 21
    rebalance_days: int = 21
    training_stride_days: int = 5
    minimum_history_days: int = 252
    inner_validation_days: int = 252
    purge_days: int = 21
    embargo_days: int = 21
    top_n: int = 8
    maximum_position_weight: float = 0.25
    maximum_sector_weight: float = 0.40
    risk_on_gross: float = 0.95
    neutral_gross: float = 0.70
    risk_off_gross: float = 0.30
    drawdown_cut: float = 0.10
    drawdown_emergency: float = 0.15
    base_transaction_cost_bps: float = 10.0
    cost_scenarios_bps: tuple[float, ...] = (5.0, 10.0, 25.0, 50.0)
    block_bootstrap_samples: int = 1000
    block_length_days: int = 21
    model_names: tuple[str, ...] = (
        "ridge",
        "elastic_net",
        "hist_gradient_boosting",
        "extra_trees",
        "random_fourier_ridge",
    )
    context_symbols: tuple[str, ...] = (
        "SPY",
        "QQQ",
        "IWM",
        "HYG",
        "TLT",
        "GLD",
        "UUP",
        "^VIX",
    )
    output_directory: str = "outputs/research_project"
    cache_file: str = ".cache/research_ohlcv.json"
    notes: tuple[str, ...] = field(
        default_factory=lambda: (
            "Research-only data source; never used as a credential source.",
            "Current constituent selection introduces survivorship bias.",
            "All candidate model trials are counted for multiple-testing diagnostics.",
        )
    )

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ResearchConfig":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        cleaned = {key: value for key, value in values.items() if key in known}
        for name in (
            "cost_scenarios_bps",
            "model_names",
            "context_symbols",
            "notes",
        ):
            if name in cleaned:
                cleaned[name] = tuple(cleaned[name])
        return cls(**cleaned)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def output_path(self, root: Path) -> Path:
        return root / self.output_directory

    def cache_path(self, root: Path) -> Path:
        return root / self.cache_file
