# MASTER CONTEXT DOCUMENT — MTFC Inventors
### Purpose: Give GitHub Copilot full context on this project so it can write, fix, and extend the codebase and paper correctly.
### Last updated: February 26, 2026
### READ THIS ENTIRE FILE BEFORE TOUCHING ANY CODE OR PAPER.

---

## 1. WHAT THIS PROJECT IS

This is a submission for the **Modeling the Future Challenge (MTFC)**, a high school actuarial modeling competition run by The Actuarial Foundation. The team name is **The Inventors**.

**Topic:** How AI data center energy consumption creates quantifiable financial and environmental risk, and what mitigation strategies reduce that risk.

**Geographic scope:** The state of **Virginia** broadly, with particular focus on the PJM Interconnection grid that serves Virginia. Northern Virginia / Ashburn is the primary case study location due to it being the highest data center density region in the world.

**Case study datacenter:** A **100MW hyperscale AI datacenter in Northern Virginia (Ashburn, VA)**.

**Water consumption is completely out of scope. Do not mention it anywhere in the code or paper.**

---

## 2. THE TWO RISKS — CORE FRAMING

Everything in this project flows from exactly two risks:

### Risk 1 — Grid Stress
AI data center electricity demand in Virginia potentially exceeding grid capacity, causing grid congestion, reliability failures, delayed new connections, and forced curtailment. This risk affects utility companies, PJM grid operators, state regulators, other electricity consumers in Virginia, and the data center operators themselves.

### Risk 2 — Carbon Emissions
Increased energy consumption from AI data centers driving higher CO₂ emissions from Virginia's electricity production. This risk affects environmental regulators, local communities, and society broadly.

These two risks are **not the same** and must be modeled and analyzed separately. Risk 1 is about grid reliability and capacity. Risk 2 is about emissions and environmental cost.

---

## 3. COMPETITION REQUIREMENTS (non-negotiable)

The paper must follow the MTFC Actuarial Process Guide structure:

1. **Problem Statement / Background**
2. **Data Methodology** — sources, reliability, what each dataset contains
3. **Mathematics Methodology** — every model with equations, assumptions, performance metrics
4. **Risk Analysis** — risk characterization, risk projection in dollars, mitigation strategy analysis
5. **Recommendations** — insurance, behavior change, modifying outcomes — each with a cost-benefit analysis
6. **Conclusion**
7. **References**

**What judges look for:**
- Actuarial framing: frequency × severity, expected value of loss, quantified in dollars
- Real data only — no synthetic or fabricated numbers
- Models that are mathematically defensible and honestly reported
- Cost-benefit analyses with specific dollar figures tied to model outputs
- Recommendations linked directly to what the models found
- Clear assumptions with justifications and necessity statements

---

## 4. PROJECT FILE STRUCTURE

```
MTFC_Inventors/
├── REAL FINAL FILES/                ← ALL active code lives here
│   ├── carbon_prediction_pipeline.py  ← Main Risk 2 pipeline (6 stages, 3664 lines)
│   ├── config.py                      ← All physics constants and model parameters
│   ├── random_forest_model.py         ← Stage 1 ML wrapper (CPU forecasting)
│   ├── xgboost_model.py               ← Stage 5 ML wrapper (carbon intensity)
│   ├── schema.py                      ← Data validation contracts
│   ├── utils.py                       ← Metrics, leakage checks, I/O helpers
│   ├── posthoc_sensitivity_visuals.py ← Sensitivity analysis + energy forecast
│   ├── matplotlib_graph_formatter.py  ← All paper figures (PNG generation)
│   ├── monetizable_outcomes_analysis.py ← All financial calculations
│   └── tests/test_pipeline.py         ← End-to-end test suite
├── Data_Sources/cleaned/              ← All input CSVs (read-only, do not modify)
├── MTFC_PAPER.md                      ← Paper draft (numbers may be outdated)
├── paper.tex                          ← LaTeX version (primary paper file)
└── BAD FINAL MODEL NO USE/            ← IGNORE COMPLETELY. Never read or run anything here.
```

---

## 5. RISK 2 — CARBON EMISSIONS PIPELINE (already built)

This pipeline is largely complete. It lives in `carbon_prediction_pipeline.py`.

### The 6-Stage Architecture

| Stage | Type | What it does | Current R² |
|---|---|---|---|
| Stage 1 | ML (Random Forest) | Forecasts CPU utilization from past CPU data | 0.6501 |
| Stage 2 | Physics | Derives IT power from CPU: `P_IT = P_facility × [α_idle + (1-α_idle) × CPU]` | Same as Stage 1 — tautology |
| Stage 3 | Physics | Derives PUE from CPU + temperature: `PUE = b + a_T × max(0, T-65°F) + a_u × CPU` | ~1.000 — tautology |
| Stage 4 | Physics | Total power = IT power × PUE | 0.9999 — tautology |
| Stage 5 | ML (XGBoost) | Forecasts grid carbon intensity (gCO₂/kWh) from past grid data | 0.8144 |
| Stage 6 | Physics | Emissions = Total power × Carbon intensity (kg CO₂/hour) | 0.9513 ✓ |

**Stage 6 R² = 0.9513 is the key result for Risk 2. This is legitimate and must be prominently featured.**

**Physics stages (2, 3, 4) have high R² because they are deterministic functions of their inputs — NOT ML model achievements. Always label them as `stage_type: physics` in all tables and in the paper.**

**Stage 1 (0.6501) and Stage 5 (0.8144) are below 0.9. This is honest and correct under strict no-leakage chronological evaluation. Do NOT inflate these numbers.**

### Baseline Uplift vs Persistence (must always be reported alongside R²)
| Stage | Model RMSE | Persistence RMSE | Uplift |
|---|---|---|---|
| Stage 1 CPU | 0.000275 | 0.000279 | +1.46% |
| Stage 5 Carbon Intensity | 10.786 | 12.019 | +10.26% |
| Stage 6 Emissions | 396.05 | 436.57 | +9.28% |

### Physics Constants (in config.py — single source of truth)
- `P_facility = 100 MW`
- `α_idle = 0.30` (idle power fraction — this is the #1 driver of both energy and emissions)
- `base_pue = 1.10`
- `pue_temp_coef = 0.012` per °F above 65°F
- `pue_cpu_coef = 0.050`
- `pue_max = 2.00`

---

## 6. RISK 1 — GRID STRESS MODEL (needs to be built — highest priority)

**This is the most important missing piece in the entire project.** The paper discusses Risk 1 but the codebase has no dedicated grid stress model. This must be built as a new script: `grid_stress_analysis.py` inside `REAL FINAL FILES/`.

### What Grid Stress Means
When AI data center electricity demand in Virginia grows faster than grid capacity can accommodate, the grid becomes stressed. In stressed conditions: real-time electricity prices spike dramatically, reliability degrades, new large load connections are denied or delayed by utilities, and in extreme cases loads are curtailed. PJM (the grid operator) requires a minimum reserve margin of ~15% above peak demand. As data centers add 100MW+ loads, this margin erodes toward and potentially below the safety threshold.

### Recommended Model: Peak Demand Exceedance Analysis

This is the actuarially correct model because it produces:
- **Frequency:** How many hours per year is the grid stressed
- **Severity:** How much does each stress event cost in dollars
- **Expected value of loss:** Frequency × Severity = Risk 1 dollar quantification

**Step 1 — Establish Virginia baseline grid demand:**
Read `pjm_hourly_demand_2019_2024_cleaned.csv`. Compute:
- Annual peak demand (MW) for each year 2019–2024
- Distribution of hourly demand (mean, 90th percentile, 95th percentile, 99th percentile)
- Trend in annual peak demand growth rate

**Step 2 — Estimate current grid capacity and reserve margin:**
From PJM planning documents or EIA data, establish Virginia's approximate installed capacity (MW). Calculate:
```
reserve_margin = (installed_capacity - peak_demand) / peak_demand × 100
```
If exact capacity data is unavailable, use the PJM-published reserve margin requirement of 15% as the threshold and work backward from the peak demand data.

**Step 3 — Project data center load addition by year:**
Using the three energy growth scenarios (Conservative 5%/yr, Moderate 15%/yr, Aggressive 30%/yr), convert annual energy (GWh) to average MW load, then estimate peak MW load using a load factor of ~0.88 (data centers run near-continuously). Add this MW load to each year's peak demand:
```
new_peak_demand(year) = baseline_peak_demand(year) + datacenter_additional_MW(year)
eroded_reserve_margin(year) = (capacity - new_peak_demand) / new_peak_demand × 100
```

**Step 4 — Count peak coincidence hours (stress events):**
For each hour in the dataset, flag it as a "stress hour" if BOTH conditions are true:
- Grid demand exceeds the 90th percentile of historical hourly demand for that month
- Data center is operating at ≥80% CPU utilization (high load period)

Count `N_stress_hours` per year. These are the frequency of Risk 1 events.

**Step 5 — Quantify financial cost of each stress hour:**
During grid stress, electricity prices spike above normal. Use one of:
- PJM real-time price data if available in the repo
- A price premium model: stress_price = baseline_price × (1 + stress_multiplier), where stress_multiplier scales with demand utilization above 90th percentile
- A conservative fixed assumption: stress price premium = $150–300/MWh above baseline (document the source)

```
grid_stress_cost_per_year = N_stress_hours × datacenter_load_MW × stress_price_premium_per_MWh
```

Also compute curtailment risk cost: if the datacenter is forced to reduce load by X% for Y hours due to PJM curtailment orders, what is the lost revenue/productivity cost? Use a conservative estimate of $500/MWh of curtailed energy as the opportunity cost.

**Step 6 — Build the risk table:**
For each growth scenario and each year 2026–2035, produce:
- Reserve margin remaining (%)
- Year when reserve margin first falls below 15% safety threshold (the "stress threshold crossing year")
- Number of stress hours per year
- Expected cost of stress events per year ($)
- Cumulative expected cost 2026–2035 ($)

### Required Grid Stress Output Files
Create these CSVs in `outputs/analysis/{run_id}/`:
- `grid_stress_reserve_margin.csv` — columns: year, scenario, reserve_margin_pct, eroded_reserve_margin_pct
- `grid_stress_coincidence_hours.csv` — columns: year, scenario, n_stress_hours
- `grid_stress_cost.csv` — columns: year, scenario, stress_cost_dollars, curtailment_risk_dollars, total_grid_stress_cost
- `grid_stress_summary.csv` — one row per scenario for 2030 and 2035 with all key metrics

### Required Grid Stress Figures
Add these to `matplotlib_graph_formatter.py`:

| Figure filename | What it shows | Why it belongs in the paper |
|---|---|---|
| `grid_stress_reserve_margin.png` | Line chart: Virginia reserve margin % from 2026–2035 under all 3 growth scenarios. Draw a red horizontal dashed line at 15% (PJM safety threshold). Label the year each scenario's line crosses below 15%. | This is the single clearest visualization of when Risk 1 becomes critical. The "crossing year" is a concrete, dateable risk event that judges will find compelling. |
| `grid_stress_demand_heatmap.png` | Hour-of-day (y-axis, 0–23) × Month (x-axis, Jan–Dec) heatmap of average Virginia grid demand, colored by MW. Overlay the data center load as a separate color band or annotation. | Shows which hours and months are the highest grid stress risk — directly actionable for workload shifting recommendations. |
| `grid_stress_coincidence_hours.png` | Grouped bar chart: number of annual peak coincidence stress hours (y-axis) from 2026–2035 (x-axis), grouped by growth scenario. | Translates grid stress into a frequency number — essential for actuarial frequency × severity framing. |
| `grid_stress_cost_projection.png` | Line or area chart: annual financial cost of grid stress events ($M, y-axis) from 2026–2035 (x-axis), one line per growth scenario, with shaded uncertainty band (low/high price premium). | Converts grid stress frequency into dollar risk — the core Risk 1 quantification for the paper. |
| `grid_stress_price_spike.png` | Scatter plot: PJM real-time electricity price ($/MWh, y-axis) vs. grid utilization % (x-axis, demand / capacity). Show the non-linear spike behavior at high utilization. | Justifies the price premium assumption and shows reviewers the empirical basis for the stress cost model. |

---

## 7. DATA SOURCES

All data is real-world. No synthetic generation. All CSVs live in `Data_Sources/cleaned/`.

| File | Source | What it contains | Used for |
|---|---|---|---|
| `noaa_global_hourly_dulles_2019_2024_cleaned.csv` | NOAA | Hourly temperature, Dulles/Ashburn VA, 2019–2024 | Stage 3 PUE, Stage 5 features |
| `pjm_hourly_demand_2019_2024_cleaned.csv` | PJM/EIA | Hourly Virginia grid demand, 2019–2024 | Grid stress model (Risk 1), Stage 5 features |
| `pjm_grid_carbon_intensity_2019_full_cleaned.csv` | PJM/EIA | Hourly carbon intensity (gCO₂/kWh), 2019 | Stage 5 training target (Risk 2) |
| `google_cluster_utilization_2019_cellb_hourly_cleaned.csv` | Google | CPU utilization traces 2019 | Stage 1 training target (Risk 2) |
| `virginia_co2_emissions_2015_2023_cleaned.csv` | EIA | Virginia annual CO₂ emissions by fuel type | Energy forecast baseline, Risk 2 context |

**Geographic note:** All grid data is Virginia-scoped within PJM. Reference Virginia as the state, Northern Virginia as the sub-region, and PJM as the grid operator throughout the paper.

---

## 8. ALL FIGURES THE PAPER NEEDS

### Risk 2 — Carbon Emissions Figures (already generating from matplotlib_graph_formatter.py):
| Filename | What it shows | Source data |
|---|---|---|
| `model_performance_dashboard.png` | R² and residuals for all 6 pipeline stages | `metrics_summary.csv` |
| `sobol_global_sensitivity.png` | S1/ST bar chart for emissions parameters | `sobol_indices.csv` |
| `sobol_energy_sensitivity.png` | S1/ST bar chart for energy parameters | `sobol_indices_energy.csv` |
| `tornado_oat_emissions.png` | Tornado bars for emissions sensitivity | `tornado_oat.csv` |
| `tornado_oat_energy.png` | Tornado bars for energy sensitivity | `tornado_oat_energy.csv` |
| `copula_tail_dashboard.png` | Copula scatter plots and tail curves | `copula_tail_dependence.csv` |
| `copula_energy_tail_bars.png` | λU/λL coefficient bar chart | `copula_tail_dependence.csv` |
| `recommendation_scenario_radar.png` | Radar chart of 4 operating scenarios | `scenario_monetization.csv` |
| `recommendation_energy_peak_projection.png` | Fan charts of GWh and MW through 2035 | `energy_forecast_scenarios.csv` |
| `recommendation_mitigation_impact.png` | Emissions reduction per mitigation lever | `mitigation_cost_benefit.csv` |
| `scenario_cost_stack.png` | Stacked cost components per scenario | `scenario_monetization.csv` |
| `scenario_carbon_price_band.png` | Carbon liability vs. carbon price band | `scenario_monetization.csv` |
| `energy_forecast_costs.png` | Annual cost trajectories per growth scenario | `energy_forecast_costs.csv` |
| `mitigation_npv_payback.png` | NPV vs. payback scatter per mitigation lever | `mitigation_cost_benefit.csv` |
| `monetizable_outcomes_dashboard.png` | Executive summary combining all monetization results | `monetary_numbers.csv` |
| `carbon_intensity_heatmap.png` | Hour × Month heatmap of PJM carbon intensity | carbon intensity CSV |

### Risk 1 — Grid Stress Figures (need to be added to matplotlib_graph_formatter.py):
| Filename | What it shows | Source data |
|---|---|---|
| `grid_stress_reserve_margin.png` | Reserve margin % 2026–2035 with 15% red threshold | `grid_stress_reserve_margin.csv` |
| `grid_stress_demand_heatmap.png` | Hour × Month Virginia grid demand with datacenter overlay | `pjm_hourly_demand_2019_2024_cleaned.csv` |
| `grid_stress_coincidence_hours.png` | Annual stress hours by growth scenario | `grid_stress_coincidence_hours.csv` |
| `grid_stress_cost_projection.png` | Annual grid stress cost 2026–2035 by scenario | `grid_stress_cost.csv` |
| `grid_stress_price_spike.png` | Price vs. utilization scatter showing non-linear spike | PJM demand + price data |

---

## 9. MONETIZATION (already built in monetizable_outcomes_analysis.py)

### Financial constants (must live in config.py — not buried in arithmetic):
- Social Cost of Carbon: ~$51/tonne CO₂
- Virginia commercial electricity price: ~$72/MWh
- Discount rate for NPV: ~8%
- Carbon price scenarios: Low $25/t, Base $51/t, High $100/t

### Key numbers the paper must contain:

**Operating scenarios (annual cost):**
| Scenario | Total Annual Cost | Carbon Liability | Electricity Cost | Risk Premium |
|---|---|---|---|---|
| Baseline | $51.23M | $20.78M | $22.43M | $0.73M |
| Efficient | $42.52M | $15.68M | $19.93M | $0.60M |
| High Growth | $59.11M | $24.95M | $24.47M | $1.59M |
| Climate Stress | $59.29M | $24.31M | $24.26M | $2.51M |

**Mitigation levers (10-year NPV):**
| Lever | CapEx | Annual Benefit | Payback | NPV |
|---|---|---|---|---|
| Combined Portfolio | $15.20M | $10.58M | 1.44 yr | $55.82M |
| Cleaner Grid Contracts | $0.20M | $3.37M | 0.06 yr | $22.45M |
| Dynamic Workload Shifting | $0.60M | $3.08M | 0.19 yr | $20.09M |
| PUE Optimization | $14.00M | $6.24M | 2.24 yr | $27.84M |

If you change any financial constant, ALL downstream numbers change. Always update config.py first, then rerun monetizable_outcomes_analysis.py, then update the paper.

---

## 10. PAPER SECTION GUIDE

### Section 1: Background / Introduction
- Establish AI data center energy surge as the problem
- Introduce Northern Virginia / Ashburn as the case study (highest data center density in the world)
- State both risks clearly: Grid Stress (Risk 1) and Carbon Emissions (Risk 2)
- Identify stakeholders for each risk
- Water is not a risk in this paper — do not mention it

### Section 2: Data Methodology
- Table of every dataset: source, what it contains, which model it feeds
- State all data is Virginia-scoped / PJM grid
- State: no synthetic data, chronological train/holdout split (85%/15%), no leakage

### Section 3: Mathematics Methodology
Must cover both risks separately:

**Risk 2 (Carbon):** All 6 pipeline stages. Equations in LaTeX. Physics stages labeled as deterministic. Honest R² with baseline uplift reported.

**Risk 1 (Grid Stress):** Peak Demand Exceedance Analysis. Cover all 6 steps from Section 6 of this doc: reserve margin calculation, load projection, stress hour counting, and cost quantification.

### Section 4: Risk Analysis
1. **Risk Characterization** — What factors drive each risk? Risk 2: idle_power_fraction and carbon intensity dominate (Sobol/Tornado findings). Risk 1: peak coincidence hours and reserve margin erosion.
2. **Risk Projection** — Project both risks to 2030 and 2035 under all three growth scenarios. Express in dollars as expected value of loss.
3. **Risk Mitigation Strategy Analysis** — Evaluate all three strategy types for both risks. Every analysis must reference a specific model output number.

### Section 5: Recommendations
**Insurance:** Environmental liability insurance for datacenter operators. Connect it to the dollar figures from Risk 1 and Risk 2 quantification. Describe mechanism and who is protected.

**Behavior Change:** Lead with **cleaner grid contracts** (payback 0.06 yr = 22 days, NPV $22.45M — the single highest ROI lever). Add **dynamic workload shifting** (shifts load off peak hours, reducing both Risk 1 stress and Risk 2 emissions simultaneously). Use the copula finding (97.3% co-occurrence of extreme heat and extreme power draw) to justify shifting load away from hot summer days.

**Modifying Outcomes:** **PUE optimization** (cooling upgrades). CapEx $14M, NPV $27.84M. Directly reduces idle_power_fraction, which is the #1 Sobol driver. Also include real-time grid monitoring as a modifying outcomes strategy for Risk 1.

Every recommendation must have a cost-benefit table with specific numbers from the model outputs.

---

## 11. COMPLETE LIST OF REQUIRED OUTPUT FILES

After a full run, these files must exist:

### `outputs/results/{run_id}/`
- `metrics_summary.csv` — with `stage_type` column (ML or physics)
- `baseline_comparison.csv`
- `predictions_holdout.csv`
- `model_card.md`

### `outputs/analysis/{run_id}/`
- `sobol_indices.csv`
- `sobol_indices_energy.csv`
- `tornado_oat.csv`
- `tornado_oat_energy.csv`
- `copula_tail_dependence.csv`
- `energy_forecast_scenarios.csv`
- `scenario_monetization.csv`
- `mitigation_cost_benefit.csv`
- `energy_forecast_costs.csv`
- `monetary_numbers.csv`
- `grid_stress_reserve_margin.csv` ← must be built
- `grid_stress_coincidence_hours.csv` ← must be built
- `grid_stress_cost.csv` ← must be built
- `grid_stress_summary.csv` ← must be built
- All 21 PNG figures (16 Risk 2 + 5 Risk 1)

---

## 12. KNOWN ISSUES TO FIX (in order of priority)

**Priority 1 — Grid Stress Model is missing**
Build `grid_stress_analysis.py`. See Section 6 of this doc for the full specification. This is the biggest gap between what the paper claims and what the code produces.

**Priority 2 — PhysicsConfig defined in three files**
`config.py`, `posthoc_sensitivity_visuals.py`, and `matplotlib_graph_formatter.py` each have their own copy of physics constants. Delete the copies in the second and third files. Import from `config.py` everywhere.

**Priority 3 — Financial constants buried in monetizable_outcomes_analysis.py**
Move SCC, electricity price, and discount rate to a `financial_assumptions` block in `config.py`. Print an assumptions table at the top of every financial output CSV.

**Priority 4 — Stage 3/4 R² presented as ML achievement**
Add a `stage_type` column to `metrics_summary.csv`. In the paper, discuss physics stages in a separate paragraph from ML stages.

---

## 13. WHAT NOT TO DO

- **Never read from or run `BAD FINAL MODEL NO USE/`**
- **Never add water consumption** — completely out of scope
- **Never inflate Stage 1 or Stage 5 R²** — no future data, no synthetic records, no relaxed splits
- **Never present Stage 3 or Stage 4 R² as ML results** — they are physics tautologies
- **Never change a physics constant in only one file** — always update config.py first, then remove all duplicate definitions
- **Never add anything to the paper that is not backed by a number in a CSV output file**
- **Never present energy forecast scenarios as statistical model forecasts** — they are scenario projections with stated growth rate assumptions

---

## 14. WHAT WINNING LOOKS LIKE

Based on analysis of the #1 winning MTFC paper (Team 19336, drug overdose topic):

Winners do all of the following — this project should emulate each one:

1. **Every recommendation has a specific cost-benefit analysis** with dollar figures that trace directly back to model outputs. Not general suggestions — specific interventions with numbers.

2. **They used their model to simulate a specific intervention and measure the output change.** They reduced the smoking feature in their GNN and watched risk probability drop. This project's equivalent: reduce idle_power_fraction (via PUE optimization) and measure the resulting drop in expected annual emissions cost.

3. **They acknowledged limitations honestly.** They stated their model excluded midwestern counties. This project should acknowledge that Stage 1 and Stage 5 are below R²=0.9 and explain why (noise bounds under strict chronological evaluation) rather than hiding it.

4. **Geographic specificity.** They used specific Ohio counties and Alabama counties — not just "America." This project uses Northern Virginia / PJM — equally specific. Use Virginia grid data, Virginia emissions data, Virginia cost numbers.

5. **One actionable finding drives the whole recommendations section.** Their finding was smoking = #2 predictor → build smoking deterrence recommendations. This project's equivalent: idle_power_fraction = #1 driver (Sobol S1 = 0.49) → PUE optimization is the core efficiency recommendation. Cleaner grid contracts = fastest payback (22 days) → lead every financial recommendation with this number.

6. **The "worst case scenario" has a specific date or number.** They gave a specific dollar savings amount for 2030. This project should give the specific year when the Aggressive growth scenario crosses below the 15% PJM reserve margin threshold — that is a concrete, dateable risk event that judges will find compelling.
