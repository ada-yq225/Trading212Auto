# Scientific Shadow-Validation Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make shadow outcomes mature on the configured 15-second sampling clock so the 15-sample horizon lasts 225 seconds instead of 900 seconds.

**Architecture:** Extend the experimental model settings with an explicit sample interval and use it only for wall-clock maturity of forward predictions. Build those settings through one pure configuration adapter used by `Runner`, preserving the sample-indexed model, persisted state, evidence gates, and all Demo trading controls.

**Tech Stack:** Python 3.9, dataclasses, `unittest`, Trading 212 Demo API, macOS `screen`.

## Global Constraints

- The only trading endpoint is `https://demo.trading212.com/api/v0`.
- Current market sampling remains 15 seconds.
- Current prediction batches remain 300 seconds apart.
- The experimental horizon remains 15 samples.
- Approval still requires 120 training samples, 20 distinct batches, 40 matured selected outcomes, a 53% hit rate, and positive mean return after 0.1% assumed round-trip cost.
- Experimental influence remains capped at 20%.
- Existing price history, experimental state, positions, cash, order history, scout state, and journal history must be preserved.
- Exactly one Demo runner may be active after rollout.

---

### Task 1: Sample-Aware Shadow Maturity

**Files:**
- Modify: `test_experimental_model.py`
- Modify: `experimental_model.py`

**Interfaces:**
- Consumes: `ExperimentSettings.horizon_samples` and the new `ExperimentSettings.sample_interval_seconds`.
- Produces: `resolve_shadow_predictions(...)` that matures pending predictions after `horizon_samples * sample_interval_seconds`.

- [ ] **Step 1: Write the failing accelerated-horizon test**

Add this test to `ExperimentalModelTests`:

```python
def test_shadow_resolution_uses_configured_sample_interval(self):
    settings = ExperimentSettings(
        horizon_samples=15,
        sample_interval_seconds=15.0,
        prediction_interval_seconds=0,
    )
    state = {}
    create_prediction_batch(
        state,
        {"A": 1.0, "B": -1.0},
        {"A": 100.0, "B": 100.0},
        now=1000.0,
        settings=settings,
    )

    resolve_shadow_predictions(
        state,
        {"A": 101.0, "B": 99.0},
        now=1224.999,
        settings=settings,
    )
    self.assertEqual(len(state["pending"]), 2)
    self.assertEqual(len(state["outcomes"]), 0)

    resolve_shadow_predictions(
        state,
        {"A": 101.0, "B": 99.0},
        now=1225.0,
        settings=settings,
    )
    self.assertEqual(len(state["pending"]), 0)
    self.assertEqual(len(state["outcomes"]), 2)
```

- [ ] **Step 2: Run the accelerated-horizon test and verify RED**

Run:

```bash
/usr/bin/python3 -m unittest -v test_experimental_model.ExperimentalModelTests.test_shadow_resolution_uses_configured_sample_interval
```

Expected: `TypeError` because `ExperimentSettings` does not yet accept `sample_interval_seconds`.

- [ ] **Step 3: Add a failing default-clock regression test**

Add:

```python
def test_shadow_resolution_defaults_to_one_minute_samples(self):
    settings = ExperimentSettings(
        horizon_samples=15,
        prediction_interval_seconds=0,
    )
    state = {}
    create_prediction_batch(
        state,
        {"A": 1.0},
        {"A": 100.0},
        now=1000.0,
        settings=settings,
    )

    resolve_shadow_predictions(
        state,
        {"A": 101.0},
        now=1899.999,
        settings=settings,
    )
    self.assertEqual(len(state["pending"]), 1)

    resolve_shadow_predictions(
        state,
        {"A": 101.0},
        now=1900.0,
        settings=settings,
    )
    self.assertEqual(len(state["outcomes"]), 1)
```

- [ ] **Step 4: Add the minimal sample-interval implementation**

Add the backward-compatible field to `ExperimentSettings`:

```python
sample_interval_seconds: float = 60.0
```

Replace the fixed maturity calculation with:

```python
horizon_seconds = (
    settings.horizon_samples * settings.sample_interval_seconds
)
```

- [ ] **Step 5: Run the targeted model tests and verify GREEN**

Run:

```bash
/usr/bin/python3 -m unittest -v test_experimental_model
```

Expected: all experimental-model tests pass, including the 225-second accelerated boundary and 900-second default boundary.

- [ ] **Step 6: Commit the model-clock change**

```bash
git add experimental_model.py test_experimental_model.py
git commit -m "Fix experimental shadow validation clock"
```

---

### Task 2: Propagate the Runner Sampling Configuration

**Files:**
- Modify: `test_auto_trader.py`
- Modify: `auto_trader.py`

**Interfaces:**
- Consumes: `Config`, including `poll_seconds`, `rebalance_seconds`, experimental gates, and `top_n`.
- Produces: `experiment_settings_from_config(config: Config) -> ExperimentSettings`, used by `Runner.__init__`.

- [ ] **Step 1: Write the failing configuration-adapter test**

Import `experiment_settings_from_config` from `auto_trader` and add:

```python
def test_experimental_settings_use_runner_sampling_interval(self):
    config = Config(
        poll_seconds=15,
        rebalance_seconds=300,
        experimental_horizon_samples=15,
    )

    settings = experiment_settings_from_config(config)

    self.assertEqual(settings.sample_interval_seconds, 15)
    self.assertEqual(settings.horizon_samples, 15)
    self.assertEqual(settings.prediction_interval_seconds, 300)
    self.assertEqual(
        settings.minimum_shadow_batches,
        config.experimental_minimum_shadow_batches,
    )
    self.assertEqual(
        settings.minimum_shadow_outcomes,
        config.experimental_minimum_shadow_outcomes,
    )
```

- [ ] **Step 2: Run the adapter test and verify RED**

Run:

```bash
/usr/bin/python3 -m unittest -v test_auto_trader.StrategyTests.test_experimental_settings_use_runner_sampling_interval
```

Expected: import failure because `experiment_settings_from_config` does not yet exist.

- [ ] **Step 3: Implement the pure settings adapter**

Add before `Runner`:

```python
def experiment_settings_from_config(config: Config) -> ExperimentSettings:
    return ExperimentSettings(
        horizon_samples=config.experimental_horizon_samples,
        sample_interval_seconds=config.poll_seconds,
        minimum_history=config.experimental_minimum_history,
        training_stride=config.experimental_training_stride,
        minimum_training_samples=config.experimental_minimum_training_samples,
        minimum_shadow_outcomes=config.experimental_minimum_shadow_outcomes,
        minimum_shadow_batches=config.experimental_minimum_shadow_batches,
        minimum_hit_rate=config.experimental_minimum_hit_rate,
        minimum_mean_net_return=config.experimental_minimum_mean_net_return,
        assumed_round_trip_cost=config.experimental_assumed_round_trip_cost,
        prediction_interval_seconds=config.rebalance_seconds,
        top_n=config.top_n,
    )
```

Replace the inline `ExperimentSettings(...)` construction in `Runner.__init__` with:

```python
self.experiment_settings = experiment_settings_from_config(config)
```

- [ ] **Step 4: Run targeted and complete tests**

Run:

```bash
/usr/bin/python3 -m unittest -v test_auto_trader.StrategyTests.test_experimental_settings_use_runner_sampling_interval
/usr/bin/python3 -m unittest -v
git diff --check
```

Expected: the adapter test and complete suite pass, with no whitespace errors.

- [ ] **Step 5: Commit the Runner propagation change**

```bash
git add auto_trader.py test_auto_trader.py
git commit -m "Propagate sampling interval to experimental model"
```

---

### Task 3: Safe Demo Rollout and Runtime Verification

**Files:**
- Runtime only: `outputs/auto_trader/runner.pid`
- Runtime only: `outputs/auto_trader/state.json`
- Runtime only: `outputs/auto_trader/journal.jsonl`
- Runtime only: `outputs/auto_trader/runner.log`

**Interfaces:**
- Consumes: the verified code, the existing `.env`, and the persisted accelerated state.
- Produces: exactly one detached `t212-demo` runner using a 225-second shadow maturity horizon.

- [ ] **Step 1: Capture pre-rollout evidence**

Run:

```bash
date '+KST %Y-%m-%d %H:%M:%S %Z'
TZ=America/New_York date '+NY %Y-%m-%d %H:%M:%S %Z'
screen -ls
tail -20 outputs/auto_trader/journal.jsonl
/usr/bin/python3 -c 'import json; d=json.load(open("outputs/auto_trader/state.json")); print(json.dumps({"lastCycleAt": d.get("lastCycleAt"), "experimental": d.get("experimental", {}), "environment": d.get("environment"), "ordersToday": d.get("ordersToday")}, ensure_ascii=False, indent=2))'
```

Expected: one detached `t212-demo` screen, Demo endpoint, recent cycles, and persisted experimental state.

- [ ] **Step 2: Request graceful shutdown**

Run:

```bash
/usr/bin/python3 auto_trader.py stop
```

Poll `screen -ls` and the journal for up to 60 seconds. Expected: the old `t212-demo` session exits and a `RUNNER_STOPPED` event appears. Do not start another runner until the old session is gone.

- [ ] **Step 3: Start exactly one verified Demo runner**

Run:

```bash
screen -dmS t212-demo /usr/bin/python3 auto_trader.py run --execute-demo
```

Expected: exactly one detached `t212-demo` session.

- [ ] **Step 4: Verify configuration and two market cycles**

Observe the journal and state for at least two cycles. Verify:

```python
assert state["environment"] == "https://demo.trading212.com/api/v0"
assert state["config"]["poll_seconds"] == 15
assert state["experimental"]["trainingSamples"] >= 120
```

The journal timestamps should advance approximately 15 seconds apart. Existing price history and experimental outcomes must remain present after restart.

- [ ] **Step 5: Verify the 225-second maturity behavior**

Identify a newly created prediction batch in `state["experimental"]["pending"]`, record its `createdAt`, and observe it through subsequent cycles. It must remain pending before 225 seconds and appear in `outcomes` at or after 225 seconds. All approval thresholds in `state["config"]` must retain their specified values.

- [ ] **Step 6: Push the verified implementation**

Run:

```bash
git status --short
git push origin main
```

Expected: the worktree is clean and `origin/main` contains the design, plan, tests, and implementation commits.
