# Accelerated Demo Trading and Online Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch the running Trading 212 Demo strategy from 60-second/15-minute operation to 15-second/5-minute operation without weakening any risk or online-model approval gate.

**Architecture:** Keep the existing runner and model interfaces. Introduce one focused state-migration helper keyed by a new strategy version, update the two JSON operating profiles, and verify the transition through unit tests plus two observed live Demo cycles.

**Tech Stack:** Python 3.9, `unittest`, JSON configuration, Trading 212 Demo API, macOS `screen`.

## Global Constraints

- The only trading endpoint is `https://demo.trading212.com/api/v0`.
- Market sampling is 15 seconds.
- Rebalance and per-symbol cooldown are 300 seconds.
- Scout seed and evaluation intervals are 300 seconds.
- Maximum order value remains £300.
- Maximum stock and sector weights remain 30% and 40%.
- Experimental approval still requires 120 training samples, 20 shadow batches, 40 matured outcomes, 53% hit rate, and positive mean net return after 0.1% round-trip cost.
- The old 60-second price/model state must not be reused after the cadence change.
- Trading 212 positions, cash, remote order history, local journal, order counter, scout attempts, and promoted scouts must be preserved.

---

### Task 1: Versioned State Migration

**Files:**
- Modify: `test_auto_trader.py`
- Modify: `auto_trader.py`

**Interfaces:**
- Consumes: a mutable strategy-state dictionary and a target version string.
- Produces: `migrate_strategy_state(state: dict[str, Any], strategy_version: str) -> bool`, returning `True` only when a migration was performed.

- [ ] **Step 1: Write the failing migration tests**

Add imports for `migrate_strategy_state` and `STRATEGY_VERSION`, then add:

```python
def test_accelerated_strategy_version(self):
    self.assertEqual(STRATEGY_VERSION, "rational_momentum_ml_v4_fast")

def test_strategy_migration_resets_interval_dependent_state_once(self):
    state = {
        "strategyVersion": "rational_momentum_ml_v3",
        "priceHistory": {"AMD_US_EQ": [1.0, 2.0]},
        "pricePeaks": {"AMD_US_EQ": 2.0},
        "experimental": {"trainingSamples": 17},
        "lastRebalance": 123.0,
        "portfolioHighWatermark": 5000.0,
        "ordersToday": 17,
        "scoutAttempts": ["AVGO_US_EQ"],
        "promotedScouts": ["TSM_US_EQ"],
    }

    changed = migrate_strategy_state(state, STRATEGY_VERSION)

    self.assertTrue(changed)
    self.assertEqual(state["strategyVersion"], STRATEGY_VERSION)
    self.assertEqual(state["priceHistory"], {})
    self.assertEqual(state["pricePeaks"], {})
    self.assertEqual(state["experimental"], {})
    self.assertEqual(state["lastRebalance"], 0)
    self.assertNotIn("portfolioHighWatermark", state)
    self.assertEqual(state["ordersToday"], 17)
    self.assertEqual(state["scoutAttempts"], ["AVGO_US_EQ"])
    self.assertEqual(state["promotedScouts"], ["TSM_US_EQ"])

    self.assertFalse(migrate_strategy_state(state, STRATEGY_VERSION))
```

- [ ] **Step 2: Run the targeted tests and verify RED**

Run:

```bash
/usr/bin/python3 -m unittest -v test_auto_trader.StrategyTests.test_accelerated_strategy_version test_auto_trader.StrategyTests.test_strategy_migration_resets_interval_dependent_state_once
```

Expected: import or assertion failure because the helper/new version does not exist.

- [ ] **Step 3: Implement the minimal migration helper**

Set:

```python
STRATEGY_VERSION = "rational_momentum_ml_v4_fast"
```

Add:

```python
def migrate_strategy_state(
    state: dict[str, Any],
    strategy_version: str,
) -> bool:
    if state.get("strategyVersion") == strategy_version:
        return False
    state["strategyVersion"] = strategy_version
    state["priceHistory"] = {}
    state["pricePeaks"] = {}
    state["experimental"] = {}
    state["lastRebalance"] = 0
    state.pop("portfolioHighWatermark", None)
    return True
```

Replace the inline version-reset block in `Runner.__init__` with:

```python
migrate_strategy_state(self.state, STRATEGY_VERSION)
```

- [ ] **Step 4: Run targeted and full tests**

Run:

```bash
/usr/bin/python3 -m unittest -v test_auto_trader.StrategyTests.test_accelerated_strategy_version test_auto_trader.StrategyTests.test_strategy_migration_resets_interval_dependent_state_once
/usr/bin/python3 -m unittest -v
```

Expected: targeted tests pass and all tests pass.

- [ ] **Step 5: Commit the migration**

```bash
git add auto_trader.py test_auto_trader.py
git commit -m "Add versioned accelerated state migration"
```

### Task 2: Accelerated Operating Profile

**Files:**
- Modify: `test_auto_trader.py`
- Modify: `strategy.json`
- Modify: `universe.json`
- Modify: `README.md`

**Interfaces:**
- Consumes: `Config.load(strategy_path)` and the JSON scout configuration.
- Produces: an exact 15-second/5-minute Demo operating profile.

- [ ] **Step 1: Write the failing profile test**

Add:

```python
def test_repository_uses_accelerated_demo_profile(self):
    config = Config.load(ROOT / "strategy.json")
    universe = json.loads((ROOT / "universe.json").read_text(encoding="utf-8"))
    self.assertEqual(config.poll_seconds, 15)
    self.assertEqual(config.short_samples, 20)
    self.assertEqual(config.medium_samples, 60)
    self.assertEqual(config.long_samples, 180)
    self.assertEqual(config.volatility_samples, 120)
    self.assertEqual(config.rebalance_seconds, 300)
    self.assertEqual(config.cooldown_seconds, 300)
    self.assertEqual(config.experimental_minimum_history, 120)
    self.assertEqual(config.experimental_minimum_training_samples, 120)
    self.assertEqual(config.experimental_minimum_shadow_batches, 20)
    self.assertEqual(config.experimental_minimum_shadow_outcomes, 40)
    self.assertEqual(config.experimental_minimum_hit_rate, 0.53)
    self.assertEqual(universe["scout_seed_interval_seconds"], 300)
    self.assertEqual(universe["scout_interval_seconds"], 300)
```

- [ ] **Step 2: Run the profile test and verify RED**

Run:

```bash
/usr/bin/python3 -m unittest -v test_auto_trader.StrategyTests.test_repository_uses_accelerated_demo_profile
```

Expected: failure because the repository still specifies 60/900-second values.

- [ ] **Step 3: Apply the exact accelerated configuration**

In `strategy.json`, set:

```json
"poll_seconds": 15,
"short_samples": 20,
"medium_samples": 60,
"long_samples": 180,
"volatility_samples": 120,
"rebalance_seconds": 300,
"cooldown_seconds": 300
```

Keep all risk and experimental gate values unchanged.

In `universe.json`, set:

```json
"scout_seed_interval_seconds": 300,
"scout_interval_seconds": 300
```

Update the README to describe 15-second sampling, 5/15/45-minute momentum, five-minute rebalancing/cooldown, approximately 30-minute accelerated warm-up, and the unchanged model gates.

- [ ] **Step 4: Run the profile test and full suite**

Run:

```bash
/usr/bin/python3 -m unittest -v test_auto_trader.StrategyTests.test_repository_uses_accelerated_demo_profile
/usr/bin/python3 -m unittest -v
git diff --check
```

Expected: all tests pass and the diff check is clean.

- [ ] **Step 5: Commit the accelerated profile**

```bash
git add strategy.json universe.json README.md test_auto_trader.py
git commit -m "Enable accelerated Demo trading and learning"
```

### Task 3: Safe Live Demo Rollout

**Files:**
- Runtime state only: `outputs/auto_trader/`

**Interfaces:**
- Consumes: the committed accelerated profile and `/usr/bin/python3`.
- Produces: exactly one healthy detached Demo runner using the new profile.

- [ ] **Step 1: Capture pre-rollout evidence**

Run:

```bash
date
/usr/bin/python3 auto_trader.py status
screen -ls
pgrep -fl "auto_trader.py run --execute-demo"
```

Record process ID, last cycle, total value, cash, order count, positions and latest journal time.

- [ ] **Step 2: Stop the old runner gracefully**

Run:

```bash
/usr/bin/python3 auto_trader.py stop
```

Poll for no live old PID and no `t212-demo` screen. If the screen remains after the process exits, terminate only that exact named screen session.

- [ ] **Step 3: Start exactly one accelerated Demo runner**

Run:

```bash
screen -dmS t212-demo /usr/bin/python3 auto_trader.py run --execute-demo
```

- [ ] **Step 4: Verify the first accelerated cycle**

Check `outputs/auto_trader/state.json` and assert:

```python
assert state["environment"] == "https://demo.trading212.com/api/v0"
assert state["strategyVersion"] == "rational_momentum_ml_v4_fast"
assert state["config"]["poll_seconds"] == 15
assert state["config"]["rebalance_seconds"] == 300
assert state["config"]["cooldown_seconds"] == 300
assert max(map(len, state["priceHistory"].values())) <= 2
```

Also verify exactly one `auto_trader.py run --execute-demo` Python process.

- [ ] **Step 5: Observe two consecutive cycles**

Condition-poll `lastCycleAt` for two changes, each with a maximum 45-second observation window. Confirm history length increases and report the actual elapsed time.

- [ ] **Step 6: Inspect errors and rollout orders**

Read journal events after the new `RUNNER_STARTED`. Require no `API_ERROR` or `UNEXPECTED_ERROR`. Report every `ORDER_SUBMITTED` and `SCOUT_SEED_ORDER` event rather than assuming no orders occurred.

- [ ] **Step 7: Final verification and publish**

Run:

```bash
/usr/bin/python3 -m unittest -q
git status --short
git push origin main
git rev-parse HEAD
git ls-remote origin refs/heads/main
```

Expected: all tests pass, source worktree is clean, local and remote commits match, and the accelerated Demo process remains alive.
