#!/usr/bin/env python3
"""Run the reproducible quantitative research pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_lab import ResearchConfig
from research_lab.runner import run_research, save_report


ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "research_config.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Trading 212 科研级量化实验")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    args = parser.parse_args()
    values = (
        json.loads(args.config.read_text(encoding="utf-8"))
        if args.config.exists()
        else {}
    )
    config = ResearchConfig.from_dict(values)
    report = run_research(ROOT, config, refresh_data=args.refresh_data)
    json_path, markdown_path = save_report(ROOT, config, report)
    strategy = report["performance"]["strategyNet"]
    benchmark = report["performance"]["SPY"]
    print(f"Run ID: {report['runId']}")
    print(
        f"Strategy: annualized={strategy['annualizedReturn']:.2%}, "
        f"Sharpe={strategy['sharpeZeroRate']:.2f}, "
        f"max drawdown={strategy['maxDrawdown']:.2%}"
    )
    print(
        f"SPY: annualized={benchmark['annualizedReturn']:.2%}, "
        f"Sharpe={benchmark['sharpeZeroRate']:.2f}, "
        f"max drawdown={benchmark['maxDrawdown']:.2%}"
    )
    print(f"JSON: {json_path}")
    print(f"Report: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
