# Tournament Batch-Gate Fix and Demo Resume Design

## Goal

Correct the experimental tournament's independent-batch gate, preserve all
existing forward-test evidence, and resume exactly one Trading 212 Demo runner.
No candidate may influence trading until it passes every existing scientific
gate.

## Confirmed Current State

- The Demo runner is stopped; its last cycle was 2026-07-30 19:50:48 UTC.
- The multi-day sampling gap makes the stored 15-second price series unsafe to
  append to directly: doing so would encode several days of movement as one
  15-second return.
- `legacy_ensemble`, `robust_huber`, and `regime_histgb` each retain 70 selected
  outcomes across 12 distinct forward batches.
- `residual_momentum` remains frozen after weak forward performance.
- The current diagnostic implementation truncates selected outcomes to the
  most recent 40 before counting distinct batches. With roughly six selected
  outcomes per batch, the displayed count stays near seven and prevents the
  20-batch approval gate from being reached.

## Statistical Semantics

`candidate_diagnostics` will maintain two related windows:

1. The performance window is the most recent 40 selected outcomes. Hit rate,
   mean net return, standard error, and confidence bounds continue to use this
   window so recent performance controls approval.
2. The evidence-history window is every selected outcome retained in candidate
   state. `batchCount` is the number of distinct `createdAt` timestamps in that
   retained history.

Approval continues to require all of the following:

- at least 120 training samples;
- at least 40 selected outcomes in the performance window;
- at least 20 distinct forward batches in retained evidence history;
- at least 53% hit rate over the performance window;
- positive mean net return after the existing 0.1% assumed round-trip cost;
- candidate is not frozen.

Early rejection uses the same corrected distinct-batch evidence count and the
existing conservative upper-confidence-bound rule. It still cannot approve a
candidate and still cannot freeze `legacy_ensemble`.

## Sampling-Continuity Reset

At runner startup, a gap longer than 300 seconds since `lastCycleAt` triggers a
continuity reset. The reset clears active `priceHistory`, candidate
`trainingSamples`, `lastPredictions`, `pending`, training-attempt timestamps and
the current champion. It sets the global tournament timestamp to startup time
so the first new attempt respects the 225-second cadence. It does not clear
candidate `outcomes`, frozen flags, error counters, position peaks, account
state, order counters, promoted scouts, or portfolio risk state. A
`SAMPLING_GAP_RESET` journal event records the measured gap and threshold.

This behavior prevents both the main momentum features and experimental model
features from treating a multi-day gap as a 15-second sample. It also prevents
stale pre-gap predictions from influencing allocations after restart. The
threshold is deliberately longer than ordinary cycle jitter but no shorter
than the configured five-minute rebalance interval.

The reset does not change the state schema. The three active candidates retain
12 distinct forward batches, so after rebuilding continuous training history
they need at least eight additional non-overlapping 225-second batches before
they can be approved. Rebuilding sufficient continuous data is expected to take
roughly 35-40 minutes, followed by about 30 minutes for eight batches. During
warmup the runner continues Demo account monitoring and risk management, but
momentum-driven order activity can be limited until its rolling windows refill.

After the gate is satisfied, the existing champion selector chooses only among
approved, non-frozen candidates using conservative net return and hit rate.
Only that single champion may supply the existing maximum 20% allocation-score
weight. There is no forced champion, direct tournament order path, real-account
endpoint, or relaxation of portfolio risk controls.

## Testing and Rollout

Add a regression test whose retained history contains at least 20 batches while
the last 40 selected outcomes span fewer batches. The test must fail before the
fix and prove that:

- `batchCount` uses retained evidence history;
- performance statistics still use only the last 40 selected outcomes;
- a candidate is approved only when all existing gates pass.

Add continuity-reset tests proving that a gap above 300 seconds clears only
interval-dependent samples and stale predictions, preserves outcomes and risk
state, revokes the champion, emits reset metadata, and is idempotent for a
recent `lastCycleAt`. A runner-initialization test must show that the reset is
persisted before live cycling begins.

Run the full test suite and syntax checks. Then snapshot the current state,
start exactly one Demo runner, and verify fresh cycles, preserved 12-batch
evidence, cleared discontinuous price samples, a `SAMPLING_GAP_RESET` event, no
candidate/API errors, and no premature champion. Observe continued 225-second
batches; when at least 20 batches exist, verify that the champion is selected
only if its current cost-adjusted performance remains eligible. Push the
verified fix to `origin/main`.
