# Parallel Model Tournament Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate four diverse experimental candidates in parallel on shared causal data, using the fastest non-overlapping 225-second cadence and preserving every existing scientific and trading-risk gate.

**Architecture:** Keep feature construction and the incumbent learner in `experimental_model.py`, place diverse candidate implementations in a focused `experimental_candidates.py`, and place state migration, confidence bounds, gates, selection and orchestration in `experimental_tournament.py`. `auto_trader.py` supplies already-fetched price histories to the tournament and consumes only its approved champion, so the feature introduces no Trading 212 API request or direct order path.

**Tech Stack:** Python 3.9, NumPy, scikit-learn, dataclasses, `unittest`, JSON state/configuration, Trading 212 Demo API, macOS `screen`.

## Global Constraints

- The only trading endpoint is `https://demo.trading212.com/api/v0`.
- Market sampling remains 15 seconds and the prediction horizon remains 15 samples.
- Prediction batches occur every 225 seconds and must not overlap.
- Candidate identifiers are `legacy_ensemble`, `robust_huber`, `regime_histgb`, and `residual_momentum`.
- Approval still requires 120 training samples, 20 distinct batches, 40 selected outcomes, 53% hit rate, and positive mean return after 0.1% assumed round-trip cost.
- Early rejection may freeze a weak challenger but may never approve a candidate.
- At most one approved champion may influence order scoring, with weight capped at 20%.
- The tournament compute budget is eight seconds inside the existing 15-second runner cycle.
- Existing accelerated price history and incumbent evaluation progress must be preserved.
- The tournament may not call the Trading 212 client or submit an order.
- Exactly one Demo runner may remain active after rollout.

---

### Task 1: Fastest Non-Overlapping Configuration

**Files:**
- Modify: `test_auto_trader.py`
- Modify: `auto_trader.py`
- Modify: `strategy.json`

**Interfaces:**
- Consumes: `Config.load(path)` and the existing experimental horizon and polling interval.
- Produces: explicit tournament configuration fields and validation that `prediction_interval_seconds >= poll_seconds * horizon_samples`.

- [ ] **Step 1: Write failing configuration tests**

Add these assertions to the repository-profile test:

```python
self.assertEqual(config.experimental_prediction_interval_seconds, 225)
self.assertEqual(config.experimental_compute_budget_seconds, 8)
self.assertEqual(
    config.experimental_candidate_ids,
    (
        "legacy_ensemble",
        "robust_huber",
        "regime_histgb",
        "residual_momentum",
    ),
)
self.assertEqual(config.experimental_early_rejection_batches, 10)
self.assertEqual(config.experimental_early_rejection_outcomes, 20)
```

Add an invalid-cadence test using a temporary JSON file:

```python
def test_experimental_prediction_batches_cannot_overlap(self):
    data = json.loads((ROOT / "strategy.json").read_text(encoding="utf-8"))
    data["experimental_prediction_interval_seconds"] = 224
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "strategy.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "不能短于"):
            Config.load(path)
```

Import `tempfile` and `Path`.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
/usr/bin/python3 -m unittest -v \
  test_auto_trader.StrategyTests.test_repository_uses_accelerated_demo_profile \
  test_auto_trader.StrategyTests.test_experimental_prediction_batches_cannot_overlap
```

Expected: the profile test fails because the new fields do not exist.

- [ ] **Step 3: Add the configuration fields and validation**

Add to `Config`:

```python
experimental_prediction_interval_seconds: float = 225.0
experimental_compute_budget_seconds: float = 8.0
experimental_candidate_ids: tuple[str, ...] = (
    "legacy_ensemble",
    "robust_huber",
    "regime_histgb",
    "residual_momentum",
)
experimental_early_rejection_batches: int = 10
experimental_early_rejection_outcomes: int = 20
```

Normalize the JSON list inside `Config.load` before construction:

```python
if "experimental_candidate_ids" in data:
    data["experimental_candidate_ids"] = tuple(
        str(item) for item in data["experimental_candidate_ids"]
    )
```

Validate:

```python
minimum_interval = (
    config.poll_seconds * config.experimental_horizon_samples
)
if config.experimental_prediction_interval_seconds < minimum_interval:
    raise ValueError(
        "experimental_prediction_interval_seconds "
        "不能短于 poll_seconds * experimental_horizon_samples"
    )
if not 0 < config.experimental_compute_budget_seconds < config.poll_seconds:
    raise ValueError("实验计算预算必须大于0且低于poll_seconds")
```

Add the five exact values to `strategy.json`.

- [ ] **Step 4: Run targeted tests and verify GREEN**

```bash
/usr/bin/python3 -m unittest -v test_auto_trader
git diff --check
```

- [ ] **Step 5: Commit**

```bash
git add auto_trader.py strategy.json test_auto_trader.py
git commit -m "Configure non-overlapping model tournament"
```

---

### Task 2: Shared Causal Dataset

**Files:**
- Modify: `experimental_model.py`
- Modify: `test_experimental_model.py`

**Interfaces:**
- Consumes: synchronized trailing price histories and `ExperimentSettings`.
- Produces: `SharedDataset`, `build_shared_dataset(...)`, `current_feature_matrix(...)`, `ExperimentalEnsemble.fit_dataset(...)`, and `ExperimentalEnsemble.predict_matrix(...)`.

- [ ] **Step 1: Write failing shared-dataset causality tests**

Import the existing `experimental_model` module, then add:

```python
def test_shared_dataset_contains_causal_market_context(self):
    self.assertTrue(
        hasattr(experimental_model, "build_shared_dataset")
    )
    build_shared_dataset = experimental_model.build_shared_dataset
    histories = synthetic_histories()
    baseline = build_shared_dataset(histories, self.settings)
    changed = {ticker: list(values) for ticker, values in histories.items()}
    for values in changed.values():
        values[-10:] = [value * 100 for value in values[-10:]]
    earlier = {
        ticker: values[:-15] for ticker, values in changed.items()
    }
    comparison = build_shared_dataset(earlier, self.settings)
    original = build_shared_dataset(
        {ticker: values[:-15] for ticker, values in histories.items()},
        self.settings,
    )
    np.testing.assert_allclose(comparison.features, original.features)
    self.assertEqual(baseline.features.shape[1], 16)

def test_current_feature_matrix_is_cross_sectionally_aligned(self):
    self.assertTrue(
        hasattr(experimental_model, "current_feature_matrix")
    )
    tickers, matrix = experimental_model.current_feature_matrix(
        synthetic_histories()
    )
    self.assertEqual(len(tickers), 8)
    self.assertEqual(matrix.shape, (8, 16))
    self.assertTrue(np.isfinite(matrix).all())
```

Import NumPy as `np`.

- [ ] **Step 2: Run and verify RED**

```bash
/usr/bin/python3 -m unittest -v \
  test_experimental_model.ExperimentalModelTests.test_shared_dataset_contains_causal_market_context \
  test_experimental_model.ExperimentalModelTests.test_current_feature_matrix_is_cross_sectionally_aligned
```

Expected: assertion failure because the shared dataset API does not exist.

- [ ] **Step 3: Implement the shared dataset**

Add:

```python
CONTEXT_FEATURE_NAMES = (
    "market_return_15",
    "market_return_60",
    "market_return_120",
    "cross_sectional_dispersion_15",
)

@dataclass(frozen=True)
class SharedDataset:
    features: np.ndarray
    labels: np.ndarray
    tickers: tuple[str, ...]
    indices: tuple[int, ...]
```

Implement trailing-age alignment:

```python
def _context_features(
    histories: dict[str, list[float]],
    age_from_latest: int,
) -> list[float] | None:
    rows: list[tuple[float, float, float]] = []
    for values in histories.values():
        index = len(values) - 1 - age_from_latest
        if index < 120:
            continue
        rows.append(tuple(
            _log_return(values, index - window, index)
            for window in (15, 60, 120)
        ))
    if len(rows) < 3:
        return None
    matrix = np.asarray(rows, dtype=float)
    return [
        float(np.median(matrix[:, 0])),
        float(np.median(matrix[:, 1])),
        float(np.median(matrix[:, 2])),
        float(np.std(matrix[:, 0], ddof=1)),
    ]
```

`build_shared_dataset` must append these four values to each existing causal
row, use the same delayed label calculation as `supervised_samples`, and
return empty `(0, 16)` arrays when no rows qualify. `current_feature_matrix`
must use `age_from_latest=0`, sorted ticker order, and only finite rows.

Add matrix-oriented methods:

```python
def fit_dataset(self, dataset: SharedDataset) -> int:
    return self._fit_arrays(dataset.features[:, :12], dataset.labels)

def predict_matrix(
    self,
    tickers: tuple[str, ...],
    matrix: np.ndarray,
) -> dict[str, float]:
    return normalized_predictions(
        tickers,
        0.5 * self.linear.predict(matrix[:, :12])
        + 0.5 * self.tree.predict(matrix[:, :12]),
    )
```

Keep `fit(histories)` and `predict(histories)` as backward-compatible
delegates.

- [ ] **Step 4: Run model tests**

```bash
/usr/bin/python3 -m unittest -v test_experimental_model
git diff --check
```

- [ ] **Step 5: Commit**

```bash
git add experimental_model.py test_experimental_model.py
git commit -m "Build shared causal tournament dataset"
```

---

### Task 3: Diverse Deterministic Candidates

**Files:**
- Create: `experimental_candidates.py`
- Create: `test_experimental_candidates.py`

**Interfaces:**
- Consumes: `SharedDataset`, current ticker tuple and current 16-column matrix.
- Produces: `CANDIDATE_IDS`, `CandidateModel.fit(dataset) -> int`, `CandidateModel.predict(tickers, matrix) -> dict[str, float]`, and `create_candidate(candidate_id, settings)`.

- [ ] **Step 1: Write failing candidate tests**

Create a test file that checks for the new module dynamically, then exercises
every fixed identifier:

```python
class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.settings = ExperimentSettings(minimum_training_samples=80)
        self.histories = synthetic_histories()
        self.dataset = build_shared_dataset(
            self.histories,
            self.settings,
        )
        self.tickers, self.matrix = current_feature_matrix(self.histories)

    def test_all_candidates_are_finite_and_deterministic(self):
        self.assertIsNotNone(
            importlib.util.find_spec("experimental_candidates")
        )
        module = importlib.import_module("experimental_candidates")
        candidate_ids = module.CANDIDATE_IDS
        self.assertEqual(
            candidate_ids,
            (
                "legacy_ensemble",
                "robust_huber",
                "regime_histgb",
                "residual_momentum",
            ),
        )
        for candidate_id in candidate_ids:
            first = module.create_candidate(candidate_id, self.settings)
            second = module.create_candidate(candidate_id, self.settings)
            self.assertGreaterEqual(
                first.fit(self.dataset),
                self.settings.minimum_training_samples,
            )
            second.fit(self.dataset)
            left = first.predict(self.tickers, self.matrix)
            right = second.predict(self.tickers, self.matrix)
            self.assertEqual(set(left), set(self.tickers))
            self.assertEqual(left, right)
            self.assertTrue(all(math.isfinite(value) for value in left.values()))

    def test_unknown_candidate_is_rejected(self):
        self.assertIsNotNone(
            importlib.util.find_spec("experimental_candidates")
        )
        module = importlib.import_module("experimental_candidates")
        with self.assertRaisesRegex(ValueError, "未知候选"):
            module.create_candidate("unknown", self.settings)
```

Import `importlib`, `importlib.util`, `math`, the existing model APIs and the
synthetic-history helper. Do not import the missing candidate module at test
module load time.

- [ ] **Step 2: Run and verify RED**

```bash
/usr/bin/python3 -m unittest -v test_experimental_candidates
```

Expected: assertion failure because `experimental_candidates.py` does not
exist.

- [ ] **Step 3: Implement candidate models**

Use this common protocol:

```python
class CandidateModel(Protocol):
    def fit(self, dataset: SharedDataset) -> int:
        raise NotImplementedError

    def predict(
        self,
        tickers: tuple[str, ...],
        matrix: np.ndarray,
    ) -> dict[str, float]:
        raise NotImplementedError
```

Implement:

- `LegacyCandidate` as an adapter around `ExperimentalEnsemble`.
- `RobustHuberCandidate` with
  `make_pipeline(RobustScaler(), HuberRegressor(epsilon=1.35, alpha=1.0,
  max_iter=200))`, using the first 12 columns.
- `RegimeHistGBCandidate` with deterministic
  `HistGradientBoostingRegressor(learning_rate=0.04, max_iter=120,
  max_depth=3, min_samples_leaf=30, l2_regularization=2.0,
  early_stopping=False, random_state=settings.random_state)`, using all 16
  columns.
- `ResidualMomentumCandidate` with no fitted estimator. After the minimum
  sample count is present, calculate:

```python
raw = (
    0.25 * (matrix[:, 1] - matrix[:, 12])
    + 0.45 * (matrix[:, 2] - matrix[:, 13])
    + 0.30 * (matrix[:, 3] - matrix[:, 14])
    - 0.15 * matrix[:, 0]
) / np.maximum(matrix[:, 5], 1e-6)
```

Normalize every candidate cross-section through
`normalized_predictions(...)` from `experimental_model.py`.

- [ ] **Step 4: Run candidate and model tests**

```bash
/usr/bin/python3 -m unittest -v \
  test_experimental_candidates \
  test_experimental_model
git diff --check
```

- [ ] **Step 5: Commit**

```bash
git add experimental_candidates.py test_experimental_candidates.py
git commit -m "Add diverse experimental challengers"
```

---

### Task 4: Versioned Tournament State Migration

**Files:**
- Create: `experimental_tournament.py`
- Create: `test_experimental_tournament.py`

**Interfaces:**
- Consumes: the current single-model experimental state dictionary.
- Produces: `migrate_tournament_state(state, candidate_ids) -> bool` and schema version `1`.

- [ ] **Step 1: Write failing migration tests**

```python
def test_migration_preserves_legacy_progress_and_is_idempotent(self):
    self.assertIsNotNone(
        importlib.util.find_spec("experimental_tournament")
    )
    module = importlib.import_module("experimental_tournament")
    state = {
        "trainingSamples": 1039,
        "lastPredictions": {"A": 0.5},
        "pending": [{"createdAt": 10.0, "ticker": "A"}],
        "outcomes": [{"createdAt": 5.0, "selected": True}],
        "lastPredictionBatch": 10.0,
        "lastTrainingAttempt": 10.0,
        "lastDiagnostics": {"batchCount": 8},
    }
    changed = module.migrate_tournament_state(state, CANDIDATE_IDS)
    self.assertTrue(changed)
    self.assertEqual(state["schemaVersion"], 1)
    self.assertIsNone(state["champion"])
    legacy = state["candidates"]["legacy_ensemble"]
    self.assertEqual(legacy["trainingSamples"], 1039)
    self.assertEqual(len(legacy["pending"]), 1)
    self.assertEqual(len(legacy["outcomes"]), 1)
    self.assertEqual(
        state["candidates"]["robust_huber"]["outcomes"],
        [],
    )
    snapshot = copy.deepcopy(state)
    self.assertFalse(
        module.migrate_tournament_state(state, CANDIDATE_IDS)
    )
    self.assertEqual(state, snapshot)
```

Add a test proving new candidate identifiers are added empty on an existing
schema without changing legacy data.

Import `copy`, `importlib`, and `importlib.util`. Do not import the missing
tournament module at test module load time.

- [ ] **Step 2: Run and verify RED**

```bash
/usr/bin/python3 -m unittest -v \
  test_experimental_tournament.TournamentTests.test_migration_preserves_legacy_progress_and_is_idempotent
```

Expected: assertion failure because `experimental_tournament.py` does not
exist.

- [ ] **Step 3: Implement migration**

Define:

```python
TOURNAMENT_SCHEMA_VERSION = 1
LEGACY_KEYS = (
    "trainingSamples",
    "lastPredictions",
    "pending",
    "outcomes",
    "lastPredictionBatch",
    "lastTrainingAttempt",
    "lastDiagnostics",
)

def empty_candidate_state() -> dict[str, Any]:
    return {
        "trainingSamples": 0,
        "lastPredictions": {},
        "pending": [],
        "outcomes": [],
        "lastPredictionBatch": 0.0,
        "lastTrainingAttempt": 0.0,
        "lastDiagnostics": {},
        "frozen": False,
        "errorCount": 0,
    }
```

On legacy migration, copy the listed keys into `legacy_ensemble`, set
top-level `lastTournamentBatch` to the preserved legacy
`lastPredictionBatch`, retain no duplicate top-level copies, and initialize
the other fixed candidates through `empty_candidate_state()`. On later calls,
add only missing candidates.

- [ ] **Step 4: Run migration tests**

```bash
/usr/bin/python3 -m unittest -v test_experimental_tournament
git diff --check
```

- [ ] **Step 5: Commit**

```bash
git add experimental_tournament.py test_experimental_tournament.py
git commit -m "Migrate experimental state to tournament schema"
```

---

### Task 5: Conservative Gates, Early Rejection and Champion Selection

**Files:**
- Modify: `experimental_tournament.py`
- Modify: `test_experimental_tournament.py`

**Interfaces:**
- Consumes: candidate outcomes and `ExperimentSettings`.
- Produces: `candidate_diagnostics(...)`, `should_freeze_candidate(...)`, and `select_champion(...)`.

- [ ] **Step 1: Write failing statistical decision tests**

Add helpers that create selected outcomes with fixed hits and returns:

```python
def candidate_state(*, batches, hits, returns):
    outcomes = []
    for index, net_return in enumerate(returns):
        outcomes.append({
            "createdAt": float(index % batches),
            "selected": True,
            "hit": index < hits,
            "netReturn": float(net_return),
        })
    return {
        "trainingSamples": 200,
        "pending": [],
        "outcomes": outcomes,
        "frozen": False,
    }

def approved_state(*, mean_return):
    return candidate_state(
        batches=20,
        hits=24,
        returns=[mean_return] * 40,
    )

def failed_state(*, mean_return):
    return candidate_state(
        batches=19,
        hits=30,
        returns=[mean_return] * 40,
    )
```

Then test:

```python
def test_early_rejection_requires_both_bad_upper_bounds(self):
    weak = candidate_state(batches=10, hits=4, returns=[-0.003] * 20)
    good_return = candidate_state(
        batches=10,
        hits=4,
        returns=[0.004] * 20,
    )
    self.assertTrue(should_freeze_candidate(
        "robust_huber",
        weak,
        self.settings,
        10,
        20,
    ))
    self.assertFalse(
        should_freeze_candidate(
            "robust_huber",
            good_return,
            self.settings,
            10,
            20,
        )
    )
    self.assertFalse(should_freeze_candidate(
        "legacy_ensemble",
        weak,
        self.settings,
        10,
        20,
    ))

def test_early_rejection_never_approves(self):
    state = candidate_state(batches=10, hits=20, returns=[0.004] * 20)
    diagnostics = candidate_diagnostics(state, self.settings)
    self.assertFalse(diagnostics["approved"])

def test_champion_selection_excludes_failed_candidates(self):
    candidates = {
        "legacy_ensemble": approved_state(mean_return=0.002),
        "robust_huber": approved_state(mean_return=0.003),
        "regime_histgb": failed_state(mean_return=0.010),
    }
    champion = select_champion(candidates, self.settings)
    self.assertEqual(champion, "robust_huber")

def test_champion_is_revoked_when_gate_fails(self):
    candidates = {"robust_huber": failed_state(mean_return=-0.001)}
    self.assertIsNone(
        select_champion(candidates, self.settings, current="robust_huber")
    )
```

- [ ] **Step 2: Run and verify RED**

```bash
/usr/bin/python3 -m unittest -v test_experimental_tournament
```

- [ ] **Step 3: Implement conservative statistics**

Use one-sided `z = 1.6448536269514722`.

Wilson upper bound:

```python
def wilson_upper(hits: int, count: int) -> float:
    if count <= 0:
        return 1.0
    p = hits / count
    z2 = ONE_SIDED_95_Z ** 2
    numerator = (
        p
        + z2 / (2 * count)
        + ONE_SIDED_95_Z
        * math.sqrt(p * (1 - p) / count + z2 / (4 * count**2))
    )
    return numerator / (1 + z2 / count)
```

Mean return upper bound is infinity with fewer than two observations;
otherwise `mean + z * stdev / sqrt(count)`.

`candidate_diagnostics` must retain all existing gates, add
`returnStandardError`, `hitRateUpper95`, and `meanNetReturnUpper95`, and set
`approved` only through the existing full gate.

`should_freeze_candidate` must require the minimum early batches and outcomes,
must never freeze `legacy_ensemble`, and must require both pessimistic upper
bounds.

`select_champion` filters for approved, non-frozen candidates and sorts by:

```python
(
    mean_net_return - return_standard_error,
    hit_rate,
    candidate_id,
)
```

Use descending numeric order and ascending identifier as the final stable
tie-break.

- [ ] **Step 4: Run tournament tests**

```bash
/usr/bin/python3 -m unittest -v test_experimental_tournament
git diff --check
```

- [ ] **Step 5: Commit**

```bash
git add experimental_tournament.py test_experimental_tournament.py
git commit -m "Gate and select tournament candidates"
```

---

### Task 6: Budgeted Failure-Isolated Tournament Engine

**Files:**
- Modify: `experimental_tournament.py`
- Modify: `test_experimental_tournament.py`

**Interfaces:**
- Consumes: candidate models, shared histories, current prices, active universe, current time, and an injected monotonic clock.
- Produces: `ExperimentalTournament.update(state: dict[str, Any], histories: dict[str, list[float]], current_prices: dict[str, float], active_universe: set[str], now: float, allow_batch: bool) -> TournamentResult`.

- [ ] **Step 1: Write failing orchestration tests**

Define:

```python
@dataclass(frozen=True)
class TournamentResult:
    predictions: dict[str, float]
    diagnostics: dict[str, Any]
    events: tuple[dict[str, Any], ...]
```

Use deterministic test doubles:

```python
class FakeCandidate:
    def __init__(self, *, fail: bool = False):
        self.fail = fail

    def fit(self, dataset):
        if self.fail:
            raise RuntimeError("candidate failed")
        return len(dataset.labels)

    def predict(self, tickers, matrix):
        return {
            ticker: float(index - len(tickers) / 2)
            for index, ticker in enumerate(tickers)
        }

class StepClock:
    def __init__(self, values):
        self.values = list(values)
        self.last = self.values[-1]

    def __call__(self):
        if self.values:
            self.last = self.values.pop(0)
        return self.last

def fake_models(failing=frozenset()):
    return {
        candidate_id: FakeCandidate(
            fail=candidate_id in failing
        )
        for candidate_id in CANDIDATE_IDS
    }

def make_tournament(settings, *, models=None, clock=time.monotonic):
    return ExperimentalTournament(
        settings,
        CANDIDATE_IDS,
        compute_budget_seconds=8.0,
        early_rejection_batches=10,
        early_rejection_outcomes=20,
        models=models or fake_models(),
        monotonic=clock,
    )
```

Tests must prove:

```python
def test_candidate_failure_is_isolated(self):
    models = fake_models(failing={"regime_histgb"})
    result = make_tournament(self.settings, models=models).update(
        self.state,
        self.histories,
        self.current_prices,
        self.active_universe,
        now=1000.0,
        allow_batch=True,
    )
    self.assertIn("legacy_ensemble", result.diagnostics["candidates"])
    self.assertEqual(
        result.diagnostics["candidates"]["regime_histgb"]["status"],
        "ERROR",
    )
    self.assertTrue(any(
        event["event"] == "CANDIDATE_ERROR"
        for event in result.events
    ))

def test_three_candidate_failures_freeze_only_that_candidate(self):
    models = fake_models(failing={"regime_histgb"})
    engine = make_tournament(self.settings, models=models)
    for now in (1000.0, 1225.0, 1450.0):
        engine.update(
            self.state,
            self.histories,
            self.current_prices,
            self.active_universe,
            now=now,
            allow_batch=True,
        )
    self.assertTrue(
        self.state["candidates"]["regime_histgb"]["frozen"]
    )
    self.assertFalse(
        self.state["candidates"]["robust_huber"]["frozen"]
    )

def test_compute_budget_skip_creates_no_fake_batch(self):
    clock = StepClock([0.0, 1.0, 9.0, 9.1])
    result = make_tournament(self.settings, clock=clock).update(
        self.state,
        self.histories,
        self.current_prices,
        self.active_universe,
        now=1000.0,
        allow_batch=True,
    )
    skipped = result.diagnostics["candidates"]["robust_huber"]
    self.assertEqual(skipped["status"], "SKIPPED_BUDGET")
    self.assertEqual(
        self.state["candidates"]["robust_huber"]["pending"],
        [],
    )

def test_tournament_uses_one_shared_batch_timestamp(self):
    make_tournament(self.settings).update(
        self.state,
        self.histories,
        self.current_prices,
        self.active_universe,
        now=1000.0,
        allow_batch=True,
    )
    created = {
        item["createdAt"]
        for state in self.state["candidates"].values()
        for item in state["pending"]
    }
    self.assertEqual(created, {1000.0})
```

- [ ] **Step 2: Run and verify RED**

```bash
/usr/bin/python3 -m unittest -v test_experimental_tournament
```

- [ ] **Step 3: Implement `ExperimentalTournament`**

Constructor dependencies:

```python
def __init__(
    self,
    experiment_settings: ExperimentSettings,
    candidate_ids: tuple[str, ...],
    compute_budget_seconds: float,
    early_rejection_batches: int,
    early_rejection_outcomes: int,
    *,
    models: dict[str, CandidateModel] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
):
```

`update` must:

1. migrate state;
2. resolve each candidate's pending outcomes;
3. compute diagnostics and freeze eligible weak challengers;
4. determine whether 225 seconds have elapsed since the global
   `lastTournamentBatch`;
5. build the shared dataset and prediction matrix once;
6. fit and predict the incumbent first, then active challengers in fixed order;
7. stop adding candidates when the eight-second budget is exhausted;
8. create candidate prediction batches only for successful finite predictions;
9. select or revoke the champion;
10. return only the approved champion's last predictions.

Candidate exceptions must update only that candidate and append an event:

```python
{
    "event": "CANDIDATE_ERROR",
    "candidate": candidate_id,
    "error": repr(exc),
}
```

Each exception increments `errorCount`; three consecutive exceptions freeze
only that challenger. A successful fit resets `errorCount` to zero. Budget
skips use `CANDIDATE_BUDGET_SKIP` and do not increment the error counter. The
engine has no client dependency.

- [ ] **Step 4: Run all experimental tests**

```bash
/usr/bin/python3 -m unittest -v \
  test_experimental_tournament \
  test_experimental_candidates \
  test_experimental_model
git diff --check
```

- [ ] **Step 5: Commit**

```bash
git add experimental_tournament.py test_experimental_tournament.py
git commit -m "Run budgeted experimental model tournament"
```

---

### Task 7: Runner Integration and Status

**Files:**
- Modify: `auto_trader.py`
- Modify: `test_auto_trader.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `ExperimentalTournament` and the existing price histories and position prices.
- Produces: runner diagnostics, approved champion predictions, candidate journal events and compact status rows.

- [ ] **Step 1: Write failing integration tests**

Test configuration propagation:

```python
self.assertEqual(
    settings.prediction_interval_seconds,
    config.experimental_prediction_interval_seconds,
)
```

Test state migration independently of the main strategy-version migration:

```python
def test_tournament_migration_does_not_clear_price_history(self):
    state = {
        "strategyVersion": STRATEGY_VERSION,
        "priceHistory": {"AMD_US_EQ": [1.0, 2.0]},
        "experimental": {"trainingSamples": 150},
    }
    migrate_tournament_state(
        state["experimental"],
        Config().experimental_candidate_ids,
    )
    self.assertEqual(state["priceHistory"]["AMD_US_EQ"], [1.0, 2.0])
```

Use a fake tournament to prove `_experimental_signals` passes only in-memory
histories/prices and consumes its result without calling `Runner.client`.

Add a status-format test for rows containing candidate identifier, state,
batch count, outcome count, hit rate and mean net return.

- [ ] **Step 2: Run and verify RED**

```bash
/usr/bin/python3 -m unittest -v test_auto_trader
```

- [ ] **Step 3: Integrate the tournament**

`experiment_settings_from_config` must use:

```python
prediction_interval_seconds=(
    config.experimental_prediction_interval_seconds
)
```

Replace `self.experimental_model` with one `ExperimentalTournament`. Replace
the legacy `_experimental_signals` body with a call to `tournament.update`.
Pass `allow_batch=True` only between 10:00 and 15:25 New York time, preserve
atomic state writes, and append every returned event through
`append_journal`.

Allocation scoring must continue to blend only when:

```python
experimental_diagnostics.get("approved")
and experimental_diagnostics.get("champion")
```

Update `show_status` to print a champion line and one compact row per fixed
candidate. Update README with candidate definitions, 225-second cadence,
eight-second budget, preserved approval gates and early-rejection semantics.

- [ ] **Step 4: Run full verification**

```bash
/usr/bin/python3 -m unittest -v
git diff --check
```

- [ ] **Step 5: Commit**

```bash
git add auto_trader.py test_auto_trader.py README.md
git commit -m "Integrate tournament with Demo runner"
```

---

### Task 8: Safe Demo Rollout and Push

**Files:**
- Runtime only: `outputs/auto_trader/`

**Interfaces:**
- Consumes: verified commits, `.env`, current Demo state and the running screen.
- Produces: one live Demo runner with preserved incumbent progress and candidate-tagged 225-second evaluation.

- [ ] **Step 1: Capture pre-rollout evidence and a recoverable state snapshot**

Record the current screen, PID, account, last cycle, price-history lengths,
incumbent batches/outcomes, order count and error events. Copy
`outputs/auto_trader/state.json` to a timestamped file under `/private/tmp`.

- [ ] **Step 2: Run fresh verification**

```bash
/usr/bin/python3 -m unittest -v
git diff --check
git status --short
```

- [ ] **Step 3: Gracefully stop the old runner**

```bash
/usr/bin/python3 auto_trader.py stop
```

Wait no more than 60 seconds and require the old screen to disappear and a
`RUNNER_STOPPED` event to appear.

- [ ] **Step 4: Start exactly one Demo runner outside the restricted sandbox**

```bash
screen -dmS t212-demo /usr/bin/python3 auto_trader.py run --execute-demo
```

Use the previously approved scoped `screen -dmS t212-demo` escalation.

- [ ] **Step 5: Verify runtime migration and cycles**

Require:

- one detached `t212-demo` screen;
- Demo endpoint;
- 15-second cycle progress;
- unchanged price-history lengths except normal growth;
- incumbent batch/outcome counts no lower than the snapshot;
- all four candidate states present;
- no `API_ERROR` or `UNEXPECTED_ERROR` after the production runner start.

- [ ] **Step 6: Observe one tournament batch and maturity**

Verify one common candidate batch timestamp, no outcome before 225 seconds,
and candidate outcomes at or after 225 seconds. Confirm that tournament
evaluation itself did not add a client call or direct order event.

- [ ] **Step 7: Verify the rollback path before push**

Confirm the timestamped `/private/tmp` state snapshot exists and is valid JSON.
Record the previous implementation commit. If migration loses incumbent state,
cycle latency breaches the compute budget repeatedly, errors recur, or more
than one runner appears, stop the tournament runner, restore the previous
commit without deleting the snapshot, restore the captured state file, and
start exactly one prior Demo runner. Do not reverse any normal Demo order.

- [ ] **Step 8: Push**

```bash
git push origin main
```

Verify `HEAD` and `origin/main` match and the worktree is clean.
