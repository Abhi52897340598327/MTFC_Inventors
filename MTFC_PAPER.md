# MTFC Final Paper Sync (Run-Backed, No-Leakage)
## Physics-Informed Carbon Risk Modeling, Sensitivity, Energy Forecasting, and Monetization

**Run ID (source of truth):** `carbon_20260224_162633`  
**Date synced:** February 24, 2026  
**Retraining status for post-hoc analyses:** **No retraining** (post-hoc only)  
**Synthetic data used:** **None**  

---

## 1. What Our Model Is

The active model is a **6-stage hybrid pipeline** in [`REAL FINAL FILES/carbon_prediction_pipeline.py`](/Users/abhiraamvenigalla/MTFC_Inventors/REAL%20FINAL%20FILES/carbon_prediction_pipeline.py):

1. **Stage 1 (ML):** CPU utilization forecasting (tree ensemble + leakage-safe persistence blend)
2. **Stage 2 (Physics):** IT power from CPU
3. **Stage 3 (Physics):** PUE from CPU + temperature
4. **Stage 4 (Physics):** Total power = IT power × PUE
5. **Stage 5 (ML):** Carbon intensity forecasting (tree ensemble + leakage-safe persistence blend)
6. **Stage 6 (Physics):** Emissions = Total power × Carbon intensity

### Leakage and integrity checks
- `no_bfill_in_source`: PASS
- `strict_past_stage1`: PASS
- `strict_past_stage5`: PASS
- Chronological holdout used; no future-to-past leakage.

---

## 2. Iteration Results (Best Strict Run)

### 2.1 Holdout metrics
Source: [`metrics_summary.csv`](/Users/abhiraamvenigalla/MTFC_Inventors/REAL%20FINAL%20FILES/outputs/results/carbon_20260224_162633/metrics_summary.csv)

| Stage | Model | RMSE | R2 | Defensible |
|---|---|---:|---:|---:|
| Stage1_CPU | RandomForest[blended_model_persistence] | 0.000275 | 0.6501 | True |
| Stage2_ITPower | Physics | 0.019269 | 0.6501 |  |
| Stage3_PUE | Physics | 0.000014 | 1.0000 |  |
| Stage4_TotalPower | Physics | 0.023697 | 0.9999 |  |
| Stage5_CarbonIntensity | XGBoost[blended_model_persistence] | 10.7863 | 0.8144 | True |
| Stage6_Emissions | PhysicsFromPredictions | 396.0462 | 0.9513 | True |

### 2.2 Baseline uplift
Source: [`baseline_comparison.csv`](/Users/abhiraamvenigalla/MTFC_Inventors/REAL%20FINAL%20FILES/outputs/results/carbon_20260224_162633/baseline_comparison.csv)

| Stage | Model RMSE | Persistence RMSE | Uplift vs Persistence |
|---|---:|---:|---:|
| Stage1_CPU | 0.000275 | 0.000279 | +1.46% |
| Stage5_CarbonIntensity | 10.7863 | 12.0194 | +10.26% |
| Stage6_Emissions | 396.0462 | 436.5651 | +9.28% |

### 2.3 R2>0.9 target status
- Achieved: **Stage3, Stage4, Stage6**
- Not achieved: **Stage1 (0.6501), Stage5 (0.8144)**

Why not (strictly without cheating):
- Stage1 and Stage5 are bounded by noise + limited explanatory signal under strict chronological evaluation.
- CV fold maxima for Stage1 remain in ~0.43 to ~0.67; Stage5 fold maxima peak around ~0.88 (still below 0.9 in strict OOF/holdout).
- No synthetic data, no split manipulation, and no leakage were introduced.

---

## 3. Sensitivity Analyses (Redone)

## 3.1 Sobol Indices (Variance-Based)
Source: [`sobol_indices.csv`](/Users/abhiraamvenigalla/MTFC_Inventors/REAL%20FINAL%20FILES/outputs/analysis/carbon_20260224_162633/sobol_indices.csv), [`sobol_indices_energy.csv`](/Users/abhiraamvenigalla/MTFC_Inventors/REAL%20FINAL%20FILES/outputs/analysis/carbon_20260224_162633/sobol_indices_energy.csv)

### Emissions Sobol
| Parameter | S1 | ST |
|---|---:|---:|
| idle_power_fraction | 0.4942 | 0.4470 |
| temperature_f | 0.3101 | 0.2208 |
| carbon_intensity | 0.3059 | 0.3201 |
| pue_temp_coef | 0.0000 | 0.0134 |
| cpu_utilization | 0.0000 | 0.0002 |
| pue_cpu_coef | 0.0000 | 0.0000 |

### Energy Sobol
| Parameter | S1 | ST |
|---|---:|---:|
| idle_power_fraction | 0.7186 | 0.6611 |
| temperature_f | 0.4653 | 0.3272 |
| pue_temp_coef | 0.0000 | 0.0197 |
| cpu_utilization | 0.0000 | 0.0003 |
| pue_cpu_coef | 0.0001 | 0.0000 |
| carbon_intensity | 0.0000 | 0.0000 |

## 3.2 Tornado OAT (Deterministic)
Source: [`tornado_oat.csv`](/Users/abhiraamvenigalla/MTFC_Inventors/REAL%20FINAL%20FILES/outputs/analysis/carbon_20260224_162633/tornado_oat.csv), [`tornado_oat_energy.csv`](/Users/abhiraamvenigalla/MTFC_Inventors/REAL%20FINAL%20FILES/outputs/analysis/carbon_20260224_162633/tornado_oat_energy.csv)

### Emissions Tornado
| Parameter | Swing (kg/h) | Low vs Baseline | High vs Baseline |
|---|---:|---:|---:|
| idle_power_fraction | 3949.05 | -16.16% | +16.16% |
| carbon_intensity | 3182.65 | -16.54% | +9.51% |
| temperature_f | 2444.42 | -1.48% | +18.52% |
| pue_temp_coef | 120.75 | -0.49% | +0.49% |
| cpu_utilization | 81.57 | -0.41% | +0.25% |
| pue_cpu_coef | 4.07 | -0.02% | +0.02% |

### Energy Tornado
| Parameter | Swing (GWh/yr) | Low vs Baseline | High vs Baseline |
|---|---:|---:|---:|
| idle_power_fraction | 96.94 | -16.16% | +16.16% |
| temperature_f | 60.01 | -1.48% | +18.52% |
| pue_temp_coef | 2.96 | -0.49% | +0.49% |
| cpu_utilization | 2.00 | -0.41% | +0.25% |
| pue_cpu_coef | 0.10 | -0.02% | +0.02% |
| carbon_intensity | 0.00 | 0.00% | 0.00% |

## 3.3 Copulas (Tail Dependence)
Source: [`copula_tail_dependence.csv`](/Users/abhiraamvenigalla/MTFC_Inventors/REAL%20FINAL%20FILES/outputs/analysis/carbon_20260224_162633/copula_tail_dependence.csv)

| Pair | λU(q=0.95) | λL(q=0.05) |
|---|---:|---:|
| temp_vs_energy | 0.9730 | 0.0270 |
| temp_vs_emissions | 0.7838 | 0.0000 |
| carbon_vs_emissions | 0.4865 | 0.9459 |
| temp_vs_carbon | 0.2973 | 0.0000 |
| carbon_vs_energy | 0.2973 | 0.2432 |
| cpu_vs_emissions | 0.1622 | 0.1892 |
| temp_vs_cpu | 0.1351 | 0.0000 |
| cpu_vs_energy | 0.1351 | 0.7838 |
| cpu_vs_carbon | 0.0541 | 0.1351 |

---

## 4. Energy Forecast Integration

Source: [`energy_forecast_scenarios.csv`](/Users/abhiraamvenigalla/MTFC_Inventors/REAL%20FINAL%20FILES/outputs/analysis/carbon_20260224_162633/energy_forecast_scenarios.csv)

| Scenario | Year | Annual Energy (GWh) | Peak (MW) |
|---|---:|---:|---:|
| Conservative 5% | 2026 | 311.53 | 40.55 |
| Conservative 5% | 2030 | 378.66 | 45.20 |
| Conservative 5% | 2035 | 483.28 | 51.77 |
| Moderate 15% | 2026 | 311.53 | 40.55 |
| Moderate 15% | 2030 | 544.86 | 55.69 |
| Moderate 15% | 2035 | 1095.92 | 82.77 |
| Aggressive 30% | 2026 | 311.53 | 40.55 |
| Aggressive 30% | 2030 | 889.75 | 74.70 |
| Aggressive 30% | 2035 | 3303.59 | 160.31 |

---

## 5. Cost-Benefit and Monetizable Outcomes

## 5.1 Scenario monetization
Source: [`scenario_monetization.csv`](/Users/abhiraamvenigalla/MTFC_Inventors/REAL%20FINAL%20FILES/outputs/analysis/carbon_20260224_162633/scenario_monetization.csv)

| Scenario | Total Annual Cost | Delta vs Baseline | Carbon Liability | Electricity Cost | Risk Premium |
|---|---:|---:|---:|---:|---:|
| Current (Baseline) | $51.23M | $0.00M | $20.78M | $22.43M | $0.73M |
| Efficient (Lower PUE + CI) | $42.52M | -$8.71M | $15.68M | $19.93M | $0.60M |
| High Growth | $59.11M | +$7.88M | $24.95M | $24.47M | $1.59M |
| Climate Stress | $59.29M | +$8.06M | $24.31M | $24.26M | $2.51M |

## 5.2 Mitigation economics (10-year)
Source: [`mitigation_cost_benefit.csv`](/Users/abhiraamvenigalla/MTFC_Inventors/REAL%20FINAL%20FILES/outputs/analysis/carbon_20260224_162633/mitigation_cost_benefit.csv)

| Lever | Capex | Annual Net Benefit | Payback | NPV (10y) | NPV Low-High Carbon |
|---|---:|---:|---:|---:|---:|
| Combined Portfolio | $15.20M | $10.58M | 1.44 y | $55.82M | $26.62M to $89.64M |
| PUE Optimization (Cooling) | $14.00M | $6.24M | 2.24 y | $27.84M | $14.74M to $43.02M |
| Cleaner Grid Contracts | $0.20M | $3.37M | 0.06 y | $22.45M | $6.83M to $40.52M |
| Dynamic Workload Shifting | $0.60M | $3.08M | 0.19 y | $20.09M | $12.08M to $29.37M |

## 5.3 Energy-forecast cost exposure
Source: [`energy_forecast_costs.csv`](/Users/abhiraamvenigalla/MTFC_Inventors/REAL%20FINAL%20FILES/outputs/analysis/carbon_20260224_162633/energy_forecast_costs.csv)

- Max projected annual cost: **$505.42M**
- Scenario/year: **Aggressive 30%, 2035**
- Increase vs 2026 baseline-cost year: **~878%**

## 5.4 Monetary master CSV
- File: [`monetary_numbers.csv`](/Users/abhiraamvenigalla/MTFC_Inventors/REAL%20FINAL%20FILES/outputs/analysis/carbon_20260224_162633/monetary_numbers.csv)

---

## 6. Graph Formatting Update (Matplotlib + Smaller Fonts)

Implemented using:
- [`matplotlib_graph_formatter.py`](/Users/abhiraamvenigalla/MTFC_Inventors/REAL%20FINAL%20FILES/matplotlib_graph_formatter.py)
- Style inspired by [`validation_dashboards.py`](/Users/abhiraamvenigalla/MTFC_Inventors/BAD%20FINAL%20MODEL%20NO%20USE/validation_dashboards.py)

Generated readable PNG charts:
- `carbon_intensity_heatmap.png`
- `sobol_global_sensitivity.png`
- `sobol_energy_sensitivity.png`
- `tornado_oat_emissions.png`
- `tornado_oat_energy.png`
- `copula_tail_dashboard.png`
- `copula_energy_tail_bars.png`
- `recommendation_scenario_radar.png`
- `recommendation_energy_peak_projection.png`
- `recommendation_mitigation_impact.png`
- `scenario_cost_stack.png`
- `scenario_carbon_price_band.png`
- `energy_forecast_costs.png`
- `mitigation_npv_payback.png`
- `monetizable_outcomes_dashboard.png`
- `model_performance_dashboard.png`

Manifest:
- [`graph_format_manifest.json`](/Users/abhiraamvenigalla/MTFC_Inventors/REAL%20FINAL%20FILES/outputs/analysis/carbon_20260224_162633/graph_format_manifest.json)

---

## 7. Strict Adversarial Reviewer Audit (PhD-Level)

A strict adversarial mathematical audit was applied to claims, assumptions, and leakage controls.

### Main critiques
1. Stage1/Stage5 do not meet R2 > 0.9 under strict no-leakage chronology.
2. Strong dependence on idle fraction and temperature highlights operational-policy sensitivity.
3. Tail dependence indicates nonlinear compounding risk in extreme weather + carbon events.
4. Cost-benefit depends on explicit market assumptions (SCC, energy price, demand charges), so uncertainty bands must be shown.

### Actions taken from critique
1. Switched Stage6 target mode to pure physics in default config to remove mixed-target distortion.
2. Added leakage-safe OOF blending (model + persistence) for Stage1/Stage5 only.
3. Re-ran full train/eval strictly chronological.
4. Recomputed Sobol, Tornado, Copula, energy forecasting, and monetization on latest run.
5. Re-rendered all charts with smaller fonts and consistent matplotlib/seaborn style.

Result:
- Stage6 reached 0.9513 R2 with defensible uplift.
- Stage1/Stage5 improved materially and remain below 0.9 without leakage or synthetic data.

---

## 8. Final Deliverable Paths

- Results: [`REAL FINAL FILES/outputs/results/carbon_20260224_162633`](/Users/abhiraamvenigalla/MTFC_Inventors/REAL%20FINAL%20FILES/outputs/results/carbon_20260224_162633)
- Analysis: [`REAL FINAL FILES/outputs/analysis/carbon_20260224_162633`](/Users/abhiraamvenigalla/MTFC_Inventors/REAL%20FINAL%20FILES/outputs/analysis/carbon_20260224_162633)
- Updated Markdown paper: [`MTFC_PAPER.md`](/Users/abhiraamvenigalla/MTFC_Inventors/MTFC_PAPER.md)
- Updated LaTeX paper: [`paper.tex`](/Users/abhiraamvenigalla/MTFC_Inventors/paper.tex)

