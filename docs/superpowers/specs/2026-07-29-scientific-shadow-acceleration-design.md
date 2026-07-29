# Scientific Shadow-Validation Acceleration Design

Date: 2026-07-29

## Objective

Correct the experimental model's shadow-outcome clock so that it uses the
configured 15-second market sampling interval. This reduces unnecessary
waiting without weakening any statistical, trading, or risk-control gate.

The change remains restricted to the Trading 212 Demo endpoint. It does not
authorize real-money trading.

## Root Cause

The accelerated strategy samples market prices every 15 seconds, and its
experimental horizon is configured as 15 samples. Training labels therefore
represent a 225-second horizon.

Shadow outcomes currently mature after:

`horizon_samples * 60 seconds`

The hard-coded 60-second conversion belongs to the previous one-minute
sampling profile. Under the accelerated profile, it makes a 15-sample shadow
prediction wait 900 seconds even though its matching training horizon spans
225 seconds.

## Design

Add the sampling interval to `ExperimentSettings` and calculate the shadow
outcome horizon as:

`horizon_samples * sample_interval_seconds`

`AutoTrader` will pass the configured `poll_seconds` value into the
experimental settings. With the current profile, the result is:

- market sampling: 15 seconds;
- experimental horizon: 15 samples;
- shadow outcome maturity: 225 seconds, or 3 minutes 45 seconds;
- prediction batch interval: unchanged at 300 seconds.

The supervised training feature and label construction remains sample-indexed,
so no model architecture or label definition changes.

## Unchanged Evidence Gates

The model must still satisfy every existing approval condition:

- at least 120 supervised training samples;
- at least 20 distinct shadow prediction batches;
- at least 40 matured selected outcomes;
- direction hit rate of at least 53%;
- positive mean return after the assumed 0.1% round-trip cost.

The experimental signal remains disabled from live Demo order scoring while
its status is `WARMUP` or `SHADOW`.

## State Handling

Existing accelerated price history is already sampled on the correct
15-second clock and must be preserved.

Any pending shadow predictions created by the accelerated runner also use the
same 15-second price stream. They may therefore mature under the corrected
225-second horizon without a state reset. Training samples, completed outcomes,
positions, cash, order history, scout state, and journal history are preserved.

## Tests

Use test-driven development:

1. Add a failing test proving that a 15-sample horizon with a 15-second sample
   interval does not mature before 225 seconds and does mature at 225 seconds.
2. Add a configuration test proving that `AutoTrader` propagates
   `poll_seconds` into the experimental settings.
3. Retain the existing one-minute default behavior test to prevent accidental
   changes for callers that do not supply an accelerated interval.
4. Run the targeted experimental-model and auto-trader tests.
5. Run the complete unit-test suite and `git diff --check`.

## Rollout

1. Capture the current Demo account, runner PID, last cycle, and pending shadow
   state.
2. Implement and verify the time-base correction.
3. Gracefully stop the existing runner and verify that it exits.
4. Start exactly one runner with execution enabled against the Demo endpoint.
5. Verify two consecutive 15-second cycles and confirm there is only one
   detached runner.
6. Confirm that new shadow batches mature after approximately 225 seconds and
   that all approval gates remain unchanged.
7. Push the verified commit to the configured GitHub repository.

No order is submitted merely to test this change. Normal Demo orders may still
be produced by the existing strategy after restart.

## Success Criteria

- The shadow maturity clock equals the configured sampling interval multiplied
  by the configured horizon sample count.
- The current profile matures outcomes after 225 seconds, not 900 seconds.
- All statistical approval gates and trading risk controls are unchanged.
- Existing accelerated state is preserved.
- Exactly one Demo runner is active after rollout.
- No duplicate process, unexpected API error, or real-money endpoint is used.

## Rollback

If the corrected runner produces inconsistent outcome timing, repeated errors,
or duplicate processes, gracefully stop it, restore the previous implementation,
start one Demo runner, and verify its process and endpoint. Rollback does not
reverse Demo orders already filled by the normal strategy.
