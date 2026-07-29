# Parallel Model Tournament Design

Date: 2026-07-29

## Objective

Improve the speed of finding an effective experimental model without weakening
the scientific approval gates or increasing Trading 212 API traffic.

The system will evaluate one incumbent and three diverse challengers on the
same live Demo observations. It may approve a model only after non-overlapping
forward outcomes demonstrate acceptable direction accuracy and positive return
after assumed costs.

This project remains locked to:

`https://demo.trading212.com/api/v0`

It does not authorize real-money trading or imply that any candidate will be
profitable.

## Speed Boundary

The current accelerated profile samples every 15 seconds and predicts over a
15-sample, 225-second horizon.

Prediction batches will move from 300 seconds to 225 seconds. This is the
shortest valid interval that avoids overlapping outcome windows. A shorter
interval would produce correlated windows and would not satisfy the requirement
for distinct scientific validation batches.

The expected minimum wall-clock validation time is:

- incumbent with eight preserved batches: approximately 49 minutes to reach
  20 batches, including final outcome maturity;
- each new challenger: approximately 75 minutes to reach 20 batches, including
  final outcome maturity.

The time estimates are minimum evidence-collection times, not promises of model
approval.

## Candidate Set

Candidate definitions are fixed before new challenger outcomes are collected.
The tournament contains four candidates:

1. `legacy_ensemble`: the existing 50/50 Ridge and histogram gradient boosting
   ensemble, retained as the incumbent and control.
2. `robust_huber`: a robustly scaled Huber regression using the common causal
   feature matrix, intended to reduce sensitivity to price and label outliers.
3. `regime_histgb`: a histogram gradient boosting model using the common
   features plus causal market-median returns and cross-sectional dispersion,
   intended to represent regime-dependent nonlinear behavior.
4. `residual_momentum`: a deterministic blend of idiosyncratic short, medium
   and long momentum after subtracting causal cross-sectional market movement,
   with volatility scaling and a short-term reversal penalty.

These candidates are deliberately structurally different. Adding multiple
parameter variants of the same learner would increase multiple-testing risk
without adding enough model diversity.

## Shared Data and Computation

The runner will construct one causal, timestamp-aligned training matrix per
tournament update. Candidate models select the columns they require from that
matrix instead of rebuilding samples independently.

All candidates reuse the price observations already collected by the trading
cycle. The tournament must not make additional Trading 212 API requests.

The data flow is:

1. the normal runner fetches the account and positions;
2. price histories receive the same 15-second observations used by the base
   strategy;
3. pending outcomes for every candidate are resolved;
4. a shared causal feature and label dataset is built;
5. active candidates fit and predict;
6. one tournament batch records candidate-tagged predictions and base prices;
7. candidate diagnostics and the tournament decision are persisted atomically.

No feature or label may use an observation after its prediction timestamp.

## Compute Budget

Experimental work must not delay risk management or normal Demo execution.

- The tournament receives an eight-second compute budget inside a 15-second
  cycle.
- Outcome resolution and the incumbent run first.
- Challengers run in a stable round-robin order within the remaining budget.
- A candidate that cannot run within the budget records `SKIPPED_BUDGET`; it
  does not receive a synthetic or partial batch.
- Training or prediction exceptions are isolated to the candidate and journaled
  as `CANDIDATE_ERROR`.
- Repeated failure freezes only the failing candidate. It cannot stop the
  runner or change the base strategy.

The budget protects execution latency; it does not relax candidate evidence
requirements.

## Evaluation and Early Rejection

Each candidate owns independent pending predictions, matured outcomes and gate
statistics. All normal approvals retain the existing requirements:

- at least 120 supervised training samples where applicable;
- at least 20 distinct non-overlapping prediction batches;
- at least 40 matured selected outcomes;
- direction hit rate of at least 53%;
- positive mean selected return after the assumed 0.1% round-trip cost.

After at least ten batches and 20 selected outcomes, a challenger may be frozen
early only when both conditions hold:

- the one-sided 95% Wilson upper bound for its hit rate is below 53%; and
- the one-sided 95% upper confidence bound for mean net return is at or below
  zero.

This rule only rejects candidates. It can never approve one early.

## Champion Selection and Revocation

Only candidates that independently pass every approval gate are eligible.

If one candidate is eligible, it becomes champion. If multiple candidates are
eligible, select the candidate with the highest conservative score:

`mean_net_return - standard_error_of_net_return`

Tie-breaking order is higher hit rate, then the fixed candidate identifier.
The selection rule is deterministic.

At most one champion may influence order scoring, and its signal weight remains
capped at 20%. The base strategy retains all stock, sector, gross exposure,
drawdown, stop-loss and order-size controls.

The champion is re-evaluated on the existing rolling outcome window. If it no
longer passes the approval gates, it immediately returns to shadow status and
its order-score influence becomes zero.

## State Schema and Migration

Introduce a versioned experimental tournament state without changing the main
strategy version or clearing accelerated price history.

The existing single-model state migrates into:

`experimental.candidates.legacy_ensemble`

The migration preserves its training count, last predictions, pending
predictions, matured outcomes, batch timestamps and diagnostics. Each new
challenger starts with an empty evaluation state. Account data, positions,
orders, price history, scout state, journal history and the portfolio
high-water mark are unchanged.

The migration is idempotent. A second startup must not duplicate outcomes,
pending predictions or candidate batches.

## Configuration

Add explicit tournament settings to `strategy.json`:

- `experimental_prediction_interval_seconds`: 225;
- `experimental_compute_budget_seconds`: 8;
- `experimental_candidate_ids`: the four fixed identifiers;
- `experimental_early_rejection_batches`: 10;
- `experimental_early_rejection_outcomes`: 20.

Configuration validation must reject a prediction interval shorter than:

`poll_seconds * experimental_horizon_samples`

The existing approval thresholds and 20% signal-weight ceiling remain
unchanged.

## Observability

Persist and journal:

- training duration and sample count per candidate;
- prediction duration and score distribution;
- pending, selected outcome and batch counts;
- hit rate, mean net return and return standard error;
- early-rejection confidence bounds and freeze reason;
- champion identifier, selection score and revocation reason;
- compute-budget skips and isolated candidate errors.

The status command will show a compact row for each candidate and the current
champion. It must clearly distinguish `WARMUP`, `SHADOW`, `FROZEN`, `APPROVED`
and `ERROR`.

## Testing

Use test-driven development for every production change.

Required automated coverage:

1. shared training features and labels remain causal;
2. the 225-second cadence is accepted and any shorter cadence is rejected;
3. the legacy state migration preserves progress and is idempotent;
4. candidate outcomes and gates are isolated from one another;
5. all four candidates produce finite deterministic predictions on synthetic
   data;
6. early rejection requires both pessimistic confidence-bound conditions;
7. no candidate can receive early approval;
8. champion selection excludes candidates that fail any existing gate;
9. deterministic selection and revocation work as specified;
10. a candidate exception or budget skip does not stop other candidates;
11. tournament evaluation makes no additional client request and submits no
    order by itself;
12. the existing Demo endpoint and portfolio risk controls remain unchanged.

Run targeted tests, the complete unit-test suite and `git diff --check`.

## Rollout

1. Capture the current runner, account, candidate progress and error state.
2. Gracefully stop the existing Demo runner.
3. Start exactly one verified runner outside the restricted sandbox so it can
   reach the Demo API.
4. Verify state migration preserved the incumbent and price history.
5. Observe at least two 15-second cycles with no API or unexpected error.
6. Observe one 225-second tournament batch and verify candidate-tagged pending
   predictions.
7. Observe that batch mature no earlier than 225 seconds.
8. Confirm the tournament caused no extra API request or direct order.
9. Push the verified commits to `origin/main`.

Normal Demo orders may still be produced by the unchanged base strategy during
rollout.

## Success Criteria

- Four fixed, diverse candidates run on shared causal data.
- Prediction batches are non-overlapping and occur every 225 seconds.
- The incumbent retains its existing progress.
- Weak candidates can be rejected early but never approved early.
- Approval and revocation enforce every existing statistical gate.
- Experimental processing remains within its compute budget or skips safely.
- No additional Trading 212 API traffic is introduced by the tournament.
- Exactly one Demo runner remains active after rollout.
- The complete test suite passes and the implementation is pushed.

## Rollback

If migration loses state, candidate processing delays execution, repeated
errors occur, or duplicate runners appear:

1. gracefully stop the tournament runner;
2. restore the previous implementation;
3. restore the pre-rollout experimental state snapshot if migration changed it;
4. start exactly one Demo runner;
5. verify the Demo endpoint, 15-second cycles and preserved account state.

Rollback does not reverse Demo orders produced by the base strategy.
