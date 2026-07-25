# Trading 212 Quantitative Research Report

Run ID: `20260725T161933Z-edfd3fa3c4`

## Protocol

- Expanding-window outer validation
- 21-day purge and 21-day embargo
- 8 model/strategy trials counted in deflated-Sharpe diagnostics
- 36 walk-forward rebalances

## Main Results

| Metric | Research ensemble | SPY |
|---|---:|---:|
| Annualized return | 28.01% | 23.18% |
| Annualized volatility | 17.72% | 15.34% |
| Sharpe | 1.48 | 1.44 |
| Maximum drawdown | 21.02% | 18.76% |
| Deflated Sharpe probability | 92.0% | n/a |

## Promotion Decision

Status: **RESEARCH_ONLY**

- FAIL — deflatedSharpeProbabilityAtLeast95Percent
- PASS — bootstrapProbabilityBeatingSPYAtLeast70Percent
- PASS — maximumDrawdownAtMost30Percent
- PASS — sharpeAt25BpsCostAbove0Point5

Passing permits Demo shadow observation only. It does not permit live-capital trading or immediate order influence.

Block-bootstrap annualized return 95% interval: 3.09% to 58.73%.

## Cost Stress Test

| One-way cost | Annualized return | Sharpe | Max drawdown |
|---:|---:|---:|---:|
| 5bps | 28.82% | 1.52 | 20.95% |
| 10bps | 28.01% | 1.48 | 21.02% |
| 25bps | 25.60% | 1.38 | 21.23% |
| 50bps | 21.67% | 1.19 | 21.57% |

## Annual Stability

| Year | Annual return | Sharpe | Max drawdown |
|---:|---:|---:|---:|
| 2023 | 30.19% | 1.85 | 7.94% |
| 2024 | 55.15% | 2.03 | 15.74% |
| 2025 | 3.69% | 0.33 | 21.02% |

## Average Ensemble Weights

- elastic_net: 9.2%
- extra_trees: 18.9%
- hist_gradient_boosting: 21.4%
- random_fourier_ridge: 37.6%
- ridge: 12.9%

## Leading Features

- market_return_126: 0.032
- nasdaq_return_126: 0.030
- treasury_return_63: 0.020
- market_return_63: 0.019
- market_drawdown_252: 0.019
- market_volatility_21: 0.018
- nasdaq_return_63: 0.017
- dollar_return_21: 0.016
- dollar_return_63: 0.016
- market_above_sma_200: 0.016
- small_cap_return_63: 0.015
- market_return_252: 0.015
- credit_risk_proxy_21: 0.015
- nasdaq_relative_63: 0.013
- high_yield_return_63: 0.013
- treasury_return_21: 0.012
- gold_return_63: 0.012
- vix_level: 0.012
- nasdaq_return_21: 0.012
- vix_change_21: 0.011

## Interpretation

This is a research result, not a guarantee of future returns. The current-constituent universe creates survivorship and selection bias. The candidate may enter the Demo shadow stage only; it must still pass forward live gates before receiving any allocation influence.
