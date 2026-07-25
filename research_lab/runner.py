from __future__ import annotations

import json
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn

from .config import ResearchConfig
from .data import load_or_download, to_frames
from .features import build_panel
from .models import fit_nested_ensemble
from .portfolio import optimize_portfolio
from .statistics import block_bootstrap, performance_metrics


def load_universe(root: Path) -> tuple[list[str], dict[str, str]]:
    data = json.loads((root / "universe.json").read_text(encoding="utf-8"))
    symbols: list[str] = []
    sectors: dict[str, str] = {}
    for sector, items in data["sectors"].items():
        for symbol in items:
            symbols.append(symbol)
            sectors[symbol] = sector
    return symbols, sectors


def _portfolio_return(
    frames: dict[str, pd.DataFrame],
    weights: dict[str, float],
    date: pd.Timestamp,
    previous_date: pd.Timestamp,
) -> tuple[float, dict[str, float]]:
    total = 0.0
    contributions: dict[str, float] = {}
    for symbol, weight in weights.items():
        frame = frames.get(symbol)
        if frame is None or date not in frame.index or previous_date not in frame.index:
            continue
        previous = float(frame.at[previous_date, "adjusted_close"])
        current = float(frame.at[date, "adjusted_close"])
        contribution = weight * (current / previous - 1.0)
        contributions[symbol] = contribution
        total += contribution
    return total, contributions


def _benchmark_returns(
    frames: dict[str, pd.DataFrame],
    calendar: list[pd.Timestamp],
    equity_symbols: list[str],
) -> tuple[pd.Series, pd.Series]:
    spy = frames["SPY"]["adjusted_close"].reindex(calendar).ffill()
    spy_returns = spy.pct_change().fillna(0.0)
    asset_returns = pd.DataFrame(
        {
            symbol: frames[symbol]["adjusted_close"].reindex(calendar).ffill().pct_change()
            for symbol in equity_symbols
            if symbol in frames
        }
    )
    equal_weight = asset_returns.mean(axis=1).fillna(0.0)
    return spy_returns, equal_weight


def _aggregate_dicts(items: list[dict[str, float]]) -> dict[str, float]:
    totals: dict[str, list[float]] = defaultdict(list)
    for item in items:
        for key, value in item.items():
            if np.isfinite(value):
                totals[key].append(float(value))
    return {
        key: float(np.mean(values))
        for key, values in sorted(totals.items())
        if values
    }


def _environment_fingerprint() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikitLearn": sklearn.__version__,
        "executable": sys.executable,
    }


def run_research(
    root: Path,
    config: ResearchConfig,
    *,
    refresh_data: bool,
) -> dict[str, Any]:
    equity_symbols, sectors = load_universe(root)
    requested = list(dict.fromkeys(equity_symbols + list(config.context_symbols)))
    raw_data, manifest = load_or_download(
        root,
        requested,
        config,
        refresh=refresh_data,
    )
    frames = to_frames(raw_data)
    available_equities = [symbol for symbol in equity_symbols if symbol in frames]
    panel, feature_columns = build_panel(
        frames,
        available_equities,
        sectors,
        config,
    )
    calendar = [
        date
        for date in frames["SPY"].index
        if pd.Timestamp(config.validation_start)
        <= date
        <= pd.Timestamp(config.validation_end)
    ]
    rebalance_dates = set(calendar[:: config.rebalance_days])
    current_weights: dict[str, float] = {}
    equity = 1.0
    high_watermark = 1.0
    gross_returns: list[float] = []
    net_returns: list[float] = []
    turnover_series: list[float] = []
    decision_log: list[dict[str, Any]] = []
    model_weights_log: list[dict[str, float]] = []
    validation_ic_log: list[dict[str, float]] = []
    feature_importance_log: list[dict[str, float]] = []
    sector_contribution: dict[str, float] = defaultdict(float)
    regime_counts: Counter[str] = Counter()

    for index, date in enumerate(calendar):
        if index:
            gross_return, contributions = _portfolio_return(
                frames,
                current_weights,
                date,
                calendar[index - 1],
            )
            for symbol, contribution in contributions.items():
                sector_contribution[sectors[symbol]] += contribution
        else:
            gross_return = 0.0
        turnover = 0.0
        estimated_cost = 0.0
        if date in rebalance_dates:
            prediction = fit_nested_ensemble(
                panel,
                feature_columns,
                date,
                config,
            )
            drawdown = (high_watermark - equity) / high_watermark
            decision = optimize_portfolio(
                prediction.predictions,
                frames,
                sectors,
                date,
                current_weights,
                drawdown,
                config,
            )
            current_weights = decision.weights
            turnover = decision.turnover
            estimated_cost = decision.estimated_cost
            regime_counts[decision.regime] += 1
            model_weights_log.append(prediction.model_weights)
            validation_ic_log.append(prediction.validation_ic)
            feature_importance_log.append(prediction.feature_importance)
            decision_log.append(
                {
                    "date": date.date().isoformat(),
                    "regime": decision.regime,
                    "grossTarget": decision.gross_target,
                    "turnover": turnover,
                    "estimatedCost": estimated_cost,
                    "optimizerSuccess": decision.optimizer_success,
                    "trainingRows": prediction.training_rows,
                    "validationRows": prediction.validation_rows,
                    "modelWeights": prediction.model_weights,
                    "validationIC": prediction.validation_ic,
                    "topPredictions": dict(
                        sorted(
                            prediction.predictions.items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )[:10]
                    ),
                    "targetWeights": current_weights,
                }
            )
        net_return = (1.0 + gross_return) * (1.0 - estimated_cost) - 1.0
        equity *= 1.0 + net_return
        high_watermark = max(high_watermark, equity)
        gross_returns.append(gross_return)
        net_returns.append(net_return)
        turnover_series.append(turnover)

    index = pd.DatetimeIndex(calendar)
    gross_series = pd.Series(gross_returns, index=index, name="gross")
    net_series = pd.Series(net_returns, index=index, name="strategy")
    turnover = pd.Series(turnover_series, index=index, name="turnover")
    spy_returns, equal_weight_returns = _benchmark_returns(
        frames,
        calendar,
        available_equities,
    )
    total_trials = len(config.model_names) + 3
    cost_robustness = {}
    for bps in config.cost_scenarios_bps:
        cost_rate = bps / 10000
        scenario = (1.0 + gross_series) * (1.0 - turnover * cost_rate) - 1.0
        cost_robustness[f"{bps:g}bps"] = performance_metrics(
            scenario,
            model_trials=total_trials,
        )

    annual = {
        str(year): performance_metrics(
            net_series[net_series.index.year == year],
            model_trials=total_trials,
        )
        for year in sorted(set(net_series.index.year))
    }
    strategy_metrics = performance_metrics(
        net_series,
        model_trials=total_trials,
    )
    bootstrap_metrics = block_bootstrap(
        net_series,
        spy_returns,
        samples=config.block_bootstrap_samples,
        block_length=config.block_length_days,
        seed=config.random_seed,
    )
    daily_frame = pd.DataFrame(
        {
            "date": [date.date().isoformat() for date in index],
            "grossReturn": gross_series.to_numpy(dtype=float),
            "netReturn": net_series.to_numpy(dtype=float),
            "SPYReturn": spy_returns.to_numpy(dtype=float),
            "equalWeightReturn": equal_weight_returns.to_numpy(dtype=float),
            "turnover": turnover.to_numpy(dtype=float),
            "equity": (1.0 + net_series).cumprod().to_numpy(dtype=float),
        }
    )
    cost_25_metrics = cost_robustness.get("25bps", {})
    promotion_gates = {
        "deflatedSharpeProbabilityAtLeast95Percent": (
            strategy_metrics.get("deflatedSharpeProbability", 0.0) >= 0.95
        ),
        "bootstrapProbabilityBeatingSPYAtLeast70Percent": (
            bootstrap_metrics.get(
                "probabilityAnnualReturnAboveBenchmark",
                0.0,
            )
            >= 0.70
        ),
        "maximumDrawdownAtMost30Percent": (
            strategy_metrics.get("maxDrawdown", 1.0) <= 0.30
        ),
        "sharpeAt25BpsCostAbove0Point5": (
            cost_25_metrics.get("sharpeZeroRate", -1.0) > 0.50
        ),
    }
    report = {
        "runId": (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + manifest["sha256"][:10]
        ),
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": _environment_fingerprint(),
        "researchProtocol": {
            "outerValidation": {
                "start": config.validation_start,
                "end": config.validation_end,
                "type": "expanding-window walk-forward",
            },
            "innerValidationDays": config.inner_validation_days,
            "purgeDays": config.purge_days,
            "embargoDays": config.embargo_days,
            "labelHorizonDays": config.label_horizon_days,
            "rebalanceDays": config.rebalance_days,
            "modelTrialsCounted": total_trials,
        },
        "dataManifest": manifest,
        "universe": {
            "requestedEquities": len(equity_symbols),
            "availableEquities": len(available_equities),
            "features": len(feature_columns),
            "featureNames": feature_columns,
        },
        "models": {
            "candidates": list(config.model_names),
            "averageEnsembleWeights": _aggregate_dicts(model_weights_log),
            "averageValidationIC": _aggregate_dicts(validation_ic_log),
            "averageFeatureImportance": dict(
                sorted(
                    _aggregate_dicts(feature_importance_log).items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:20]
            ),
        },
        "performance": {
            "strategyNet": strategy_metrics,
            "strategyGross": performance_metrics(
                gross_series,
                model_trials=total_trials,
            ),
            "SPY": performance_metrics(spy_returns),
            "equalWeightCurrentUniverse": performance_metrics(equal_weight_returns),
            "blockBootstrap": bootstrap_metrics,
            "costRobustness": cost_robustness,
            "annualSubperiods": annual,
        },
        "candidateDecision": {
            "status": (
                "DEMO_SHADOW_ELIGIBLE"
                if all(promotion_gates.values())
                else "RESEARCH_ONLY"
            ),
            "eligibleForDemoShadow": all(promotion_gates.values()),
            "gates": promotion_gates,
            "policy": (
                "Passing permits Demo shadow observation only. It does not permit "
                "live-capital trading or immediate order influence."
            ),
        },
        "implementation": {
            "rebalanceCount": len(decision_log),
            "averageTurnover": float(turnover[turnover > 0].mean())
            if (turnover > 0).any()
            else 0.0,
            "optimizerSuccessRate": float(
                np.mean([item["optimizerSuccess"] for item in decision_log])
            )
            if decision_log
            else 0.0,
            "regimeCounts": dict(regime_counts),
            "sectorReturnContribution": dict(sector_contribution),
        },
        "dailySeries": daily_frame.to_dict(orient="records"),
        "decisions": decision_log,
        "limitations": [
            "The current stock universe introduces survivorship and selection bias.",
            "Yahoo OHLCV is an external research source and may contain adjustments or gaps.",
            "Daily validation is not identical to minute-level Demo execution.",
            "Fixed-bps costs approximate but do not reproduce every spread or market impact.",
            "Statistical probabilities remain model-dependent and are not profit guarantees.",
        ],
        "config": config.as_dict(),
    }
    return report


def save_report(
    root: Path,
    config: ResearchConfig,
    report: dict[str, Any],
) -> tuple[Path, Path]:
    output = config.output_path(root)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "latest_report.json"
    markdown_path = output / "latest_report.md"
    manifest_path = output / "data_manifest.json"
    config_path = output / "resolved_config.json"
    registry_path = output / "experiment_registry.jsonl"
    daily_path = output / "daily_returns.csv"
    candidate_path = output / "candidate_decision.json"
    _atomic_json(json_path, report)
    _atomic_json(manifest_path, report["dataManifest"])
    _atomic_json(config_path, report["config"])
    _atomic_json(candidate_path, report["candidateDecision"])
    pd.DataFrame(report["dailySeries"]).to_csv(daily_path, index=False)
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    registry_entry = {
        "runId": report["runId"],
        "generatedAt": report["generatedAt"],
        "dataSha256": report["dataManifest"]["sha256"],
        "strategy": report["performance"]["strategyNet"],
        "benchmark": report["performance"]["SPY"],
        "modelWeights": report["models"]["averageEnsembleWeights"],
        "candidateDecision": report["candidateDecision"],
        "environment": report["environment"],
    }
    with registry_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(registry_entry, ensure_ascii=False) + "\n")
    return json_path, markdown_path


def _atomic_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp.replace(path)


def _render_markdown(report: dict[str, Any]) -> str:
    strategy = report["performance"]["strategyNet"]
    benchmark = report["performance"]["SPY"]
    bootstrap = report["performance"]["blockBootstrap"]
    model_weights = report["models"]["averageEnsembleWeights"]
    importance = report["models"]["averageFeatureImportance"]
    candidate = report["candidateDecision"]
    cost_robustness = report["performance"]["costRobustness"]
    annual = report["performance"]["annualSubperiods"]
    lines = [
        "# Trading 212 Quantitative Research Report",
        "",
        f"Run ID: `{report['runId']}`",
        "",
        "## Protocol",
        "",
        "- Expanding-window outer validation",
        f"- {report['researchProtocol']['purgeDays']}-day purge and "
        f"{report['researchProtocol']['embargoDays']}-day embargo",
        f"- {report['researchProtocol']['modelTrialsCounted']} model/strategy trials "
        "counted in deflated-Sharpe diagnostics",
        f"- {report['implementation']['rebalanceCount']} walk-forward rebalances",
        "",
        "## Main Results",
        "",
        "| Metric | Research ensemble | SPY |",
        "|---|---:|---:|",
        f"| Annualized return | {strategy['annualizedReturn']:.2%} | "
        f"{benchmark['annualizedReturn']:.2%} |",
        f"| Annualized volatility | {strategy['annualizedVolatility']:.2%} | "
        f"{benchmark['annualizedVolatility']:.2%} |",
        f"| Sharpe | {strategy['sharpeZeroRate']:.2f} | "
        f"{benchmark['sharpeZeroRate']:.2f} |",
        f"| Maximum drawdown | {strategy['maxDrawdown']:.2%} | "
        f"{benchmark['maxDrawdown']:.2%} |",
        f"| Deflated Sharpe probability | "
        f"{strategy['deflatedSharpeProbability']:.1%} | n/a |",
        "",
        "## Promotion Decision",
        "",
        f"Status: **{candidate['status']}**",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} — {name}"
        for name, passed in candidate["gates"].items()
    )
    lines.extend(
        [
            "",
            candidate["policy"],
            "",
            "Block-bootstrap annualized return 95% interval: "
            f"{bootstrap.get('annualizedReturn95CI', [0, 0])[0]:.2%} to "
            f"{bootstrap.get('annualizedReturn95CI', [0, 0])[1]:.2%}.",
            "",
            "## Cost Stress Test",
            "",
            "| One-way cost | Annualized return | Sharpe | Max drawdown |",
            "|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        f"| {name} | {metrics['annualizedReturn']:.2%} | "
        f"{metrics['sharpeZeroRate']:.2f} | {metrics['maxDrawdown']:.2%} |"
        for name, metrics in cost_robustness.items()
    )
    lines.extend(
        [
            "",
            "## Annual Stability",
            "",
            "| Year | Annual return | Sharpe | Max drawdown |",
            "|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        f"| {year} | {metrics['annualizedReturn']:.2%} | "
        f"{metrics['sharpeZeroRate']:.2f} | {metrics['maxDrawdown']:.2%} |"
        for year, metrics in annual.items()
    )
    lines.extend(
        [
            "",
            "## Average Ensemble Weights",
            "",
        ]
    )
    lines.extend(f"- {name}: {weight:.1%}" for name, weight in model_weights.items())
    lines.extend(["", "## Leading Features", ""])
    lines.extend(f"- {name}: {value:.3f}" for name, value in importance.items())
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is a research result, not a guarantee of future returns. "
            "The current-constituent universe creates survivorship and selection bias. "
            "The candidate may enter the Demo shadow stage only; it must still pass "
            "forward live gates before receiving any allocation influence.",
            "",
        ]
    )
    return "\n".join(lines)
