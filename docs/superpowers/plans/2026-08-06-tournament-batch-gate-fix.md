# Tournament Batch-Gate Fix and Demo Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the independent-batch approval gate, safely reset discontinuous 15-second samples after a long stop, and resume exactly one Demo runner that activates only a fully approved trained champion.

**Architecture:** Keep statistical gate logic in `experimental_tournament.py` and add one pure continuity-reset function to `auto_trader.py`, invoked during `Runner` initialization before any live cycle. Preserve forward outcomes and risk state while clearing only interval-dependent samples and stale predictions. Roll out through a recoverable state snapshot, verify continuous 225-second batches, and require the existing full gate before champion influence.

**Tech Stack:** Python 3.9, NumPy, scikit-learn, `unittest`, JSON state/journal, Trading 212 Demo API, macOS `screen`, Git.

## Global Constraints

- The only trading endpoint is `https://demo.trading212.com/api/v0`.
- Never force a champion or relax the 120-sample, 40-outcome, 20-batch, 53%-hit-rate, or positive-cost-adjusted-return gates.
- Performance statistics use the most recent 40 selected outcomes; independent-batch evidence uses all retained selected outcomes.
- A sampling gap longer than 300 seconds clears interval-dependent price samples and stale predictions.
- Preserve candidate outcomes, frozen flags, error counters, price peaks, account state, order counters, promoted scouts, and portfolio risk state.
- Prediction batches remain non-overlapping at 225 seconds and tournament compute remains capped at eight seconds per 15-second runner cycle.
- At most one approved champion may influence allocation scoring, with the existing maximum 20% weight.
- Runtime rollout must leave exactly one Demo runner and a valid recoverable pre-rollout state snapshot.
- All production changes follow red-green TDD and the full test suite must pass before rollout or push.

---

### Task 1: Correct Independent-Batch Evidence Counting

**Files:**
- Modify: `test_experimental_tournament.py`
- Modify: `experimental_tournament.py:112-170`

**Interfaces:**
- Consumes: `candidate_diagnostics(candidate_state: dict[str, Any], settings: ExperimentSettings) -> dict[str, Any]`.
- Produces: unchanged diagnostics keys where `outcomeCount`, hit rate, return statistics and confidence bounds use the most recent required outcomes, while `batchCount` uses every retained selected outcome.

- [ ] **Step 1: Add the regression fixture and failing test**

Add this helper and test to `test_experimental_tournament.py`:

```python
def state_with_long_batch_history():
    older = [
        {
            "createdAt": float(batch),
            "selected": True,
            "hit": False,
            "netReturn": -0.1,
        }
        for batch in range(20)
    ]
    recent = [
        {
            "createdAt": float(13 + index % 7),
            "selected": True,
            "hit": index < 24,
            "netReturn": 0.002,
        }
        for index in range(40)
    ]
    return {
        "trainingSamples": 200,
        "pending": [],
        "outcomes": older + recent,
        "frozen": False,
    }


def test_batch_gate_uses_retained_history_but_metrics_use_recent_window(self):
    diagnostics = candidate_diagnostics(
        state_with_long_batch_history(),
        self.settings,
    )
    self.assertEqual(diagnostics["batchCount"], 20)
    self.assertEqual(diagnostics["outcomeCount"], 40)
    self.assertAlmostEqual(diagnostics["hitRate"], 0.6)
    self.assertAlmostEqual(diagnostics["meanNetReturn"], 0.002)
    self.assertTrue(diagnostics["approved"])
```

Import `candidate_diagnostics` directly from `experimental_tournament`.

- [ ] **Step 2: Run the regression test and verify RED**

Run:

```bash
/usr/bin/python3 -m unittest -v \
  test_experimental_tournament.TournamentTests.test_batch_gate_uses_retained_history_but_metrics_use_recent_window
```

Expected: FAIL because the current implementation reports seven batches and does not approve the candidate.

- [ ] **Step 3: Separate evidence history from the performance window**

Replace the first lines of `candidate_diagnostics` with:

```python
all_selected = [
    item
    for item in candidate_state.get("outcomes", [])
    if item.get("selected")
]
selected = all_selected[-settings.minimum_shadow_outcomes :]
outcome_count = len(selected)
batch_count = len(
    {float(item["createdAt"]) for item in all_selected}
)
```

Leave hit count, returns, standard error, confidence bounds, approval conditions and return keys unchanged.

- [ ] **Step 4: Run tournament tests and verify GREEN**

Run:

```bash
LOKY_MAX_CPU_COUNT=1 /usr/bin/python3 -W error::RuntimeWarning \
  -m unittest -v test_experimental_tournament
git diff --check
```

Expected: all tournament tests pass with no runtime warning.

- [ ] **Step 5: Commit the statistical fix**

```bash
git add experimental_tournament.py test_experimental_tournament.py
git commit -m "Fix tournament batch evidence gate"
```

---

### Task 2: Reset Discontinuous Sampling State on Startup

**Files:**
- Modify: `test_auto_trader.py`
- Modify: `auto_trader.py:20-30, 410-480`
- Modify: `README.md`

**Interfaces:**
- Consumes: `state["lastCycleAt"]`, current epoch seconds, and a threshold in seconds.
- Produces: `reset_stale_sampling_state(state: dict[str, Any], now: float, threshold_seconds: float) -> dict[str, float] | None`.
- Runner integration consumes the returned metadata and emits `SAMPLING_GAP_RESET` before live cycling.

- [ ] **Step 1: Read the good-test rules before adding tests**

Read completely:

```bash
sed -n '1,320p' /Users/yq225/.codex/skills/test-driven-development/writing-good-tests.md
```

The production behavior that makes the first test fail is the new pure reset function; the behavior that makes the second fail is Runner startup persistence and journaling.

- [ ] **Step 2: Add failing pure-function continuity tests**

Import `reset_stale_sampling_state` and add:

```python
def stale_tournament_state():
    return {
        "strategyVersion": STRATEGY_VERSION,
        "lastCycleAt": "2026-07-30T19:50:48+00:00",
        "priceHistory": {"AMD_US_EQ": [1.0, 2.0]},
        "pricePeaks": {"AMD_US_EQ": 2.0},
        "ordersToday": 56,
        "promotedScouts": ["LLY_US_EQ"],
        "experimental": {
            "schemaVersion": 1,
            "champion": "legacy_ensemble",
            "lastTournamentBatch": 100.0,
            "candidates": {
                "legacy_ensemble": {
                    "trainingSamples": 1650,
                    "lastPredictions": {"AMD_US_EQ": 1.0},
                    "pending": [{"createdAt": 100.0}],
                    "outcomes": [{"createdAt": 50.0, "selected": True}],
                    "lastPredictionBatch": 100.0,
                    "lastTrainingAttempt": 100.0,
                    "lastDiagnostics": {"approved": True},
                    "frozen": False,
                    "errorCount": 0,
                }
            },
        },
    }


def test_stale_sampling_reset_preserves_forward_evidence_and_risk_state(self):
    state = stale_tournament_state()
    now = datetime(2026, 8, 5, 15, 0, tzinfo=timezone.utc).timestamp()
    metadata = reset_stale_sampling_state(state, now, 300.0)
    self.assertIsNotNone(metadata)
    self.assertGreater(metadata["gapSeconds"], 300.0)
    self.assertEqual(metadata["thresholdSeconds"], 300.0)
    self.assertEqual(state["priceHistory"], {})
    self.assertEqual(state["pricePeaks"], {"AMD_US_EQ": 2.0})
    self.assertEqual(state["ordersToday"], 56)
    self.assertEqual(state["promotedScouts"], ["LLY_US_EQ"])
    experiment = state["experimental"]
    self.assertIsNone(experiment["champion"])
    self.assertEqual(experiment["lastTournamentBatch"], now)
    legacy = experiment["candidates"]["legacy_ensemble"]
    self.assertEqual(legacy["trainingSamples"], 0)
    self.assertEqual(legacy["lastPredictions"], {})
    self.assertEqual(legacy["pending"], [])
    self.assertEqual(len(legacy["outcomes"]), 1)
    self.assertFalse(legacy["frozen"])
    self.assertEqual(legacy["errorCount"], 0)


def test_recent_sampling_state_is_not_reset(self):
    state = stale_tournament_state()
    state["lastCycleAt"] = "2026-08-05T14:59:00+00:00"
    snapshot = copy.deepcopy(state)
    now = datetime(2026, 8, 5, 15, 0, tzinfo=timezone.utc).timestamp()
    self.assertIsNone(reset_stale_sampling_state(state, now, 300.0))
    self.assertEqual(state, snapshot)
```

Import `copy` in `test_auto_trader.py`.

- [ ] **Step 3: Run the pure-function tests and verify RED**

Run:

```bash
/usr/bin/python3 -m unittest -v \
  test_auto_trader.StrategyTests.test_stale_sampling_reset_preserves_forward_evidence_and_risk_state \
  test_auto_trader.StrategyTests.test_recent_sampling_state_is_not_reset
```

Expected: import failure because `reset_stale_sampling_state` does not exist.

- [ ] **Step 4: Implement the minimal pure reset function**

Add near `experiment_settings_from_config`:

```python
def reset_stale_sampling_state(
    state: dict[str, Any],
    now: float,
    threshold_seconds: float,
) -> dict[str, float] | None:
    raw_last_cycle = state.get("lastCycleAt")
    if not raw_last_cycle:
        return None
    try:
        last_cycle = datetime.fromisoformat(
            str(raw_last_cycle).replace("Z", "+00:00")
        ).timestamp()
    except (TypeError, ValueError):
        return None
    gap_seconds = max(0.0, now - last_cycle)
    if gap_seconds <= threshold_seconds:
        return None
    state["priceHistory"] = {}
    experiment = state.setdefault("experimental", {})
    experiment["champion"] = None
    experiment["lastTournamentBatch"] = now
    for candidate in experiment.get("candidates", {}).values():
        candidate["trainingSamples"] = 0
        candidate["lastPredictions"] = {}
        candidate["pending"] = []
        candidate["lastPredictionBatch"] = 0.0
        candidate["lastTrainingAttempt"] = 0.0
        candidate["lastDiagnostics"] = {}
    return {
        "gapSeconds": gap_seconds,
        "thresholdSeconds": float(threshold_seconds),
    }
```

- [ ] **Step 5: Run the pure-function tests and verify GREEN**

Run the two-test command from Step 3 again. Expected: both pass.

- [ ] **Step 6: Add a failing Runner initialization test**

Use the stale fixture, patch only external boundaries, and assert real state behavior:

```python
def test_runner_persists_and_journals_sampling_gap_reset(self):
    state = stale_tournament_state()
    now = datetime(2026, 8, 5, 15, 0, tzinfo=timezone.utc).timestamp()
    with patch.object(
        auto_trader, "make_client", return_value=object()
    ), patch.object(
        auto_trader, "load_state", return_value=state
    ), patch.object(
        auto_trader, "load_universe_config", return_value={}
    ), patch.object(
        auto_trader.time, "time", return_value=now
    ), patch.object(
        auto_trader, "atomic_json_write"
    ) as write_state, patch.object(
        auto_trader, "append_journal"
    ) as journal:
        Runner(Config(), execute=True)
    write_state.assert_called_once_with(auto_trader.STATE_FILE, state)
    journal.assert_called_once()
    event, = journal.call_args.args
    self.assertEqual(event, "SAMPLING_GAP_RESET")
    self.assertGreater(journal.call_args.kwargs["gapSeconds"], 300.0)
    self.assertEqual(state["priceHistory"], {})
    self.assertIsNone(state["experimental"]["champion"])
```

- [ ] **Step 7: Run the Runner test and verify RED**

Run:

```bash
/usr/bin/python3 -m unittest -v \
  test_auto_trader.StrategyTests.test_runner_persists_and_journals_sampling_gap_reset
```

Expected: FAIL because Runner does not invoke or persist the reset.

- [ ] **Step 8: Integrate reset into Runner initialization**

Import `migrate_tournament_state` beside `ExperimentalTournament`. Immediately
after `migrate_strategy_state`, migrate the experimental state and apply the
gap threshold:

```python
migrate_tournament_state(
    self.state.setdefault("experimental", {}),
    config.experimental_candidate_ids,
)
gap_threshold = max(
    config.rebalance_seconds,
    config.poll_seconds * config.experimental_horizon_samples,
)
reset_metadata = reset_stale_sampling_state(
    self.state,
    time.time(),
    gap_threshold,
)
if reset_metadata:
    atomic_json_write(STATE_FILE, self.state)
    append_journal("SAMPLING_GAP_RESET", **reset_metadata)
```

Do not alter `migrate_strategy_state`, `pricePeaks`, outcomes, frozen flags or
order counters.

- [ ] **Step 9: Document continuity behavior**

In README section 9, add one paragraph stating that a startup gap over five
minutes resets only 15-second price samples and stale predictions, preserves
forward outcomes and risk/account state, and requires continuous warmup before
any experimental champion can return.

- [ ] **Step 10: Run auto-trader tests and verify GREEN**

Run:

```bash
LOKY_MAX_CPU_COUNT=1 /usr/bin/python3 -W error::RuntimeWarning \
  -m unittest -v test_auto_trader
git diff --check
```

Expected: all auto-trader tests pass with no runtime warning.

- [ ] **Step 11: Commit the continuity reset**

```bash
git add auto_trader.py test_auto_trader.py README.md
git commit -m "Reset stale sampling windows safely"
```

---

### Task 3: Full Verification and Safe Demo Rollout

**Files:**
- Runtime only: `outputs/auto_trader/state.json`
- Runtime only: `outputs/auto_trader/journal.jsonl`
- Snapshot only: `/private/tmp/trading212-state-before-gap-reset-<UTC timestamp>.json`

**Interfaces:**
- Consumes: verified commits, current Demo credentials, retained candidate outcomes and the stopped runtime state.
- Produces: exactly one live `t212-demo` screen, continuous price samples, preserved 12-batch evidence and auditable reset metadata.

- [ ] **Step 1: Run fresh complete verification**

```bash
LOKY_MAX_CPU_COUNT=1 /usr/bin/python3 -W error::RuntimeWarning -m unittest -v
PYTHONPYCACHEPREFIX=/private/tmp/trading212-pycache \
  /usr/bin/python3 -m py_compile \
  auto_trader.py experimental_model.py experimental_candidates.py \
  experimental_tournament.py t212_demo.py
git diff --check
git status --short
```

Expected: every test passes, compilation exits zero, and the worktree is clean.

- [ ] **Step 2: Capture and validate a recoverable runtime snapshot**

Use an explicit UTC timestamp from `date -u +%Y%m%dT%H%M%SZ`, copy
`outputs/auto_trader/state.json` to the resulting explicit `/private/tmp` path,
then run:

```bash
jq empty /private/tmp/trading212-state-before-gap-reset-<UTC timestamp>.json
shasum -a 256 \
  outputs/auto_trader/state.json \
  /private/tmp/trading212-state-before-gap-reset-<UTC timestamp>.json
```

Expected: valid JSON and matching hashes before startup.

- [ ] **Step 3: Resolve stale runtime markers without deleting state**

Require `screen -ls` to show no `t212-demo` session. If the stale PID file is
present, move only `outputs/auto_trader/runner.pid` to
`/private/tmp/trading212-stale-runner.pid`; do not remove or replace the state or
journal.

- [ ] **Step 4: Start exactly one Demo runner**

```bash
screen -dmS t212-demo /usr/bin/python3 auto_trader.py run --execute-demo
```

- [ ] **Step 5: Verify the startup reset and fresh cycles**

Within two 15-second cycles, require:

- exactly one detached `t212-demo` session;
- environment equals `https://demo.trading212.com/api/v0`;
- a new `RUNNER_STARTED` and one `SAMPLING_GAP_RESET` event;
- `priceHistory` restarted with only fresh short sequences;
- candidate outcome histories still report 12 distinct batches for the three
  active candidates;
- `champion` is null, candidate training samples restart at zero, and the frozen
  residual candidate remains frozen;
- no new `API_ERROR`, `UNEXPECTED_ERROR`, or `CANDIDATE_ERROR`.

- [ ] **Step 6: Monitor conditionally through continuous warmup**

Poll state at intervals no longer than 60 seconds while sending concise user
updates. Require `lastCycleAt` to advance and price-history lengths to grow
continuously. Do not use fixed blocking waits longer than 60 seconds. The first
new prediction batch must not occur before sufficient causal training data is
available; each successful candidate batch must share one timestamp.

- [ ] **Step 7: Verify the 20-batch approval decision**

Continue condition-based monitoring until the active candidates have at least
20 retained distinct batches. Then recompute from state and require:

```text
approval = trainingSamples >= 120
        and recent selected outcomes >= 40
        and retained distinct batches >= 20
        and recent hit rate >= 0.53
        and recent mean net return > 0
        and not frozen
```

If one or more candidates qualify, require exactly one champion and confirm
allocation scoring uses only that champion. If none qualifies, require champion
to remain null while the base Demo strategy continues; do not weaken the gate.

- [ ] **Step 8: Verify errors, rollback artifact and process uniqueness**

Require no repeated errors since the new `RUNNER_STARTED`, validate the snapshot
again with `jq empty`, and require exactly one Demo screen. If state evidence was
lost, price samples include the multi-day gap, or more than one runner exists,
stop the new runner and restore the explicit snapshot before restarting the
previous verified commit. Do not reverse any normal Demo order.

---

### Task 4: Push Verified Main and Confirm Remote State

**Files:**
- No source changes.

**Interfaces:**
- Consumes: verified local `main` commits and a healthy Demo runner.
- Produces: matching local `HEAD` and `origin/main` at the final implementation commit.

- [ ] **Step 1: Run the final verification gate**

```bash
LOKY_MAX_CPU_COUNT=1 /usr/bin/python3 -W error::RuntimeWarning -m unittest -v
git diff --check
git status --short
```

Expected: full suite passes and the worktree is clean.

- [ ] **Step 2: Push the confirmed branch**

```bash
git push origin main
```

- [ ] **Step 3: Verify local and remote commit identity**

```bash
git rev-parse HEAD
git rev-parse refs/remotes/origin/main
screen -ls
```

Expected: both commit hashes match and exactly one `t212-demo` session remains.
