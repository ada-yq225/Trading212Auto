# Accelerated Demo Trading and Online Learning Design

Date: 2026-07-29

## Objective

Accelerate both Trading 212 Demo decision-making and online model learning while preserving the existing risk controls and all forward-validation gates.

This change remains locked to:

`https://demo.trading212.com/api/v0`

It does not authorize real-money trading.

## Approved Operating Profile

| Setting | Current | Accelerated |
|---|---:|---:|
| Market sampling interval | 60 seconds | 15 seconds |
| Short momentum window | 15 samples / 15 minutes | 20 samples / 5 minutes |
| Medium momentum window | 60 samples / 60 minutes | 60 samples / 15 minutes |
| Long momentum window | 240 samples / 240 minutes | 180 samples / 45 minutes |
| Volatility window | 120 samples / 120 minutes | 120 samples / 30 minutes |
| Portfolio rebalance interval | 900 seconds | 300 seconds |
| Per-symbol trade cooldown | 900 seconds | 300 seconds |
| Scout seed interval | 900 seconds | 300 seconds |
| Scout evaluation interval | 900 seconds | 300 seconds |
| Experimental minimum history | 120 samples / 120 minutes | 120 samples / 30 minutes |
| Experimental outcome horizon | 15 samples / 15 minutes | 15 samples / 3.75 minutes |
| Experimental prediction interval | 900 seconds | 300 seconds |
| Scout promotion history | 241 samples / 241 minutes | 241 samples / about 60 minutes |

## Unchanged Safety Controls

- Trading 212 Demo endpoint remains hard-coded.
- Only regular US market hours may submit orders.
- At most one non-idempotent order request is submitted per cycle.
- A market order is never automatically retried after an ambiguous response.
- Maximum order value remains £300.
- Minimum ordinary order value remains £20.
- Maximum position weight remains 30%.
- Maximum sector weight remains 40%.
- Cash reserve remains 5%.
- Risk-on, neutral and risk-off gross exposure limits remain unchanged.
- Portfolio drawdown cuts remain at 8% and 12%.
- Hard stop remains 8%.
- Adaptive trailing-stop range remains 5% to 15%.
- Online model influence remains capped at 20%.
- There is no daily order-count ceiling, as previously requested.

## Online-Learning Gates

Acceleration changes the clock, not the evidence requirements. The experimental model must still satisfy all of the following:

- at least 120 supervised training samples;
- at least 20 independent shadow prediction batches;
- at least 40 matured selected outcomes;
- direction hit rate of at least 53%;
- positive mean return after the assumed 0.1% round-trip cost.

Before these gates pass, the model remains `WARMUP` or `SHADOW` and cannot influence order scores.

## State Migration

Existing price history contains observations spaced about 60 seconds apart. Mixing those observations with new 15-second samples would make sample-indexed momentum windows and label horizons inconsistent.

The accelerated profile therefore receives a distinct strategy version. On the first accelerated start, the existing version guard will:

- clear `priceHistory`;
- clear price peaks derived from the old sampling process;
- clear experimental training, pending predictions and outcomes;
- reset the rebalance timestamp and portfolio high-water mark.

It will preserve:

- Trading 212 positions and cash;
- Trading 212 order history;
- the local append-only journal;
- the daily order counter;
- scout attempts and already promoted scout symbols.

The reset is deliberate and auditable. Expected accelerated warm-up is about 30 minutes for continuously held symbols.

## Configuration and Code Changes

1. Update `strategy.json` with the accelerated sampling windows, rebalance interval and cooldown.
2. Update `universe.json` scout seed/evaluation intervals to 300 seconds.
3. Change the strategy version so the state migration happens exactly once.
4. Add tests that assert the accelerated profile and the version-triggered history reset.
5. Update the README operating-profile documentation.

## Rollout

1. Record the pre-change Demo account, positions, process ID and last cycle.
2. Run the new tests in the failing state.
3. Apply the minimal configuration and strategy-version changes.
4. Run the targeted tests and the full test suite.
5. Request graceful shutdown of the old runner and verify it exits.
6. Start exactly one accelerated runner with `/usr/bin/python3`.
7. Verify the endpoint is Demo, the new configuration is active and history was reset.
8. Observe at least two completed 15-second cycles.
9. Verify history growth, process health, and absence of duplicate runner processes.
10. Report any Demo orders produced during rollout.

## Success Criteria

- One and only one accelerated runner is alive.
- The state reports the Demo endpoint.
- Two consecutive cycles complete approximately 15 seconds apart.
- Active price histories grow between those cycles.
- No 60-second price history is reused by the accelerated strategy version.
- Rebalance and cooldown values are both 300 seconds.
- Scout seed/evaluation intervals are both 300 seconds.
- All automated tests pass.
- No API or unexpected-error events occur during rollout verification.

## Rollback

If the accelerated runner produces repeated API errors, duplicate processes, non-finite model output or unsafe order behavior:

1. request a graceful stop;
2. restore the previous configuration and strategy version;
3. restart one Demo runner;
4. verify the restored profile before allowing it to continue.

Rollback does not reverse orders already filled in the Demo account.
