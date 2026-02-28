# CHANGELOG

All notable changes to the MTFC carbon prediction pipeline are documented here.

---

## [Unreleased] — 2025

### Fix 1.1 — SALib Sobol sensitivity analysis (N=10 000, 95 % CI)

**File:** `posthoc_sensitivity_visuals.py`

Replaced the hand-rolled Sobol estimator (which produced `S1` sums > 1 due to an
incorrect variance decomposition) with a proper SALib Saltelli sampler and Jansen
estimator.

- `saltelli.sample(problem, n=10_000, calc_second_order=False, seed=seed)`
- `sobol_analyze.analyze(problem, Y, calc_second_order=False, conf_level=0.95)`
- Output columns added to `sobol_indices.csv`: `S1_ci_low`, `S1_ci_high`,
  `ST_ci_low`, `ST_ci_high`
- Renderer `_render_sobol_dual()` now draws 95 % CI error bars on both the S1 and
  ST bar charts.

---

### Fix 1.2 — Bootstrap confidence intervals for copula tail dependence

**File:** `posthoc_sensitivity_visuals.py`

Added `_bootstrap_tail_dependence_ci(u, v, q, n_bootstrap=1_000, seed=42)` using
`numpy` rng to produce non-parametric 95 % CI for both upper and lower tail
dependence coefficients.

- `_copula_analysis()` now calls the bootstrap helper for all 9 pair combinations.
- Output columns added to `copula_tail_dependence.csv`: `upper_tail_ci_low`,
  `upper_tail_ci_high`, `lower_tail_ci_low`, `lower_tail_ci_high`.

---

### Fix 1.3 — `stage_type` field added to `StageMetrics`

**Files:** `schema.py`, `carbon_prediction_pipeline.py`

Added `stage_type: str` field to the `StageMetrics` dataclass (value `"ML"` or
`"physics"`).  `_stage_metric_row()` auto-assigns the type by checking whether
`model_name` is in `{"Physics", "PhysicsFromPredictions"}`.

Physics stages should not be compared as ML model achievements in leaderboards or
papers; this field makes the distinction machine-readable in `metrics_summary.csv`.

---

### Fix 2.1 — Out-of-sample validation documentation

**File:** `carbon_prediction_pipeline.py`

- `_build_model_card()` Limitations section expanded with a detailed explanation of
  why OOS validation is unavailable for Stage 1 CPU and Stage 5 CI (2019-only data
  means the training split exhausts all available labeled rows).
- `save_artifacts()` now writes `oos_validation_results.csv` with one row per
  stage, columns: `stage`, `oos_available`, `reason`, `suggested_remediation`.

---

### Fix 3.1 — Dead stages 4b / 7 documented in metadata and model card

**File:** `carbon_prediction_pipeline.py`

Added three keys to the `run_metadata` dict:

```python
"stage4b_active": bool   # True only if 'stage4_combined_power' model present
"stage7_active":  bool   # True only if 'stage7_energy' model present
"inactive_stage_reason": str  # human-readable explanation when False
```

Also expanded `_build_model_card()` Limitations section to document the feature
flag conditions under which each optional stage activates.

---

### Fix 4.1 — OLS energy demand forecast (replaces compound-interest extrapolation)

**File:** `posthoc_sensitivity_visuals.py`

`_build_energy_forecast_rows()` previously used a hard-coded 2 % compound annual
growth rate. Replaced with an OLS linear regression on 6 years of PJM hourly
demand data (`pjm_hourly_demand_2019_2024_cleaned.csv`):

- Annual median demand computed from hourly data filtered to 50 k–200 k MWh (removes
  known 2021 outliers).
- `scipy.stats.linregress` gives slope, intercept, R², p-value, and SE.
- Three output scenarios: `"Regression forecast (PJM trend)"`,
  `"Regression lower 80% PI"`, `"Regression upper 80% PI"` using t-distribution
  prediction intervals.
- New CSV columns: `pi_80_pct_half_width_frac`, `ols_r2`, `ols_p_value`,
  `ols_n_obs`, `method`.
- Falls back gracefully to flat ±10 % band if data file or scipy unavailable.

---

### Fix 5.1 — Single canonical `PhysicsConfig` (import alias)

**Files:** `posthoc_sensitivity_visuals.py`, `matplotlib_graph_formatter.py`

Removed the duplicate `PhysicsConfig` dataclass definitions in both files.
Both files now import:

```python
from config import PipelineConfig as PhysicsConfig  # alias preserves all call sites
```

`config.py` (`PipelineConfig`) remains the single source of truth for all physics
constants.

---

### Fix 6.1 — `FinancialAssumptions` moved to `config.py`; provenance header in CSV

**Files:** `config.py`, `monetizable_outcomes_analysis.py`

- `FinancialAssumptions` frozen dataclass (13 constants: SCC, energy price, discount
  rate, etc.) moved from `monetizable_outcomes_analysis.py` to `config.py`.
- `monetizable_outcomes_analysis.py` imports `from config import FinancialAssumptions`.
- Added `_write_monetary_csv(path, rows, assumptions)` helper that prepends a
  `#`-prefixed provenance comment line listing all assumption values before the CSV
  header, making assumptions machine-traceable from the output file alone.

---

### Fix 7.1 — Quantile prediction intervals for Stage 1 and Stage 5

**Files:** `random_forest_model.py`, `xgboost_model.py`,
`carbon_prediction_pipeline.py`

Added `predict_quantile_intervals(model, X, quantiles=(0.10, 0.50, 0.90))` to both
model wrappers:

- **Random Forest**: stacks per-tree predictions and takes `numpy.percentile`.
- **XGBoost**: uses `pred_leaf=True` leaf-variance Gaussian approximation for
  quantile bounds (documented approximation; matches ExtraTrees path when an
  ExtraTrees fallback model is available).

`evaluate()` now computes intervals and appends six columns to the stage prediction
DataFrames:

| DataFrame      | New columns |
|----------------|-------------|
| `stage1_df`    | `stage1_cpu_p10`, `stage1_cpu_p50`, `stage1_cpu_p90` |
| `stage5_df`    | `stage5_ci_p10`, `stage5_ci_p50`, `stage5_ci_p90` |

These columns are propagated through `save_artifacts()` to
`stage1_predictions.csv` and `stage5_predictions.csv`.

---

### Fix 8.2 — `actual_emissions` invariant to optional datasets (test fix)

**File:** `carbon_prediction_pipeline.py`

Pre-existing test failure in `test_variant_stage6_target_equivalence_when_gate_mode_matches`:
`actual_emissions` differed between the enhanced (weather_exog=True) and core
pipelines because `actual_pue` was computed with dew-point / wind speed for the
enhanced case.

**Root cause:** optional weather exog was passed to `_physics_pue()` for *both*
prediction and actual paths.

**Fix:** `actual_pue` now always uses `dew=None, wind=None` so the ground-truth
physics calculation is invariant to whether optional weather exog is loaded. Only
`pred_pue` uses the enriched weather inputs.

Additionally: `actual_emissions` is now always derived from `actual_total_power_physics`
(not `actual_total_power_observed`) so sensor calibration data cannot alter the
evaluation ground truth.

All 10 tests pass.
