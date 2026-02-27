# Joint-Risk Operations Playbook (MTFC)

## 1) Danger Rule
- High stress hour: `grid_stress_ratio > 1.00`
- High emissions hour: `pred_emissions >= 27875.32` (top 10% baseline emissions threshold)
- Red hour definition: `high_stress OR high_emissions`

## 2) Joint-Risk Metric
- Percentile stress score: `StressPct_t = percentile(grid_stress_ratio_t)`
- Percentile emissions score: `EmissionsPct_t = percentile(pred_emissions_t)`
- Joint-risk score: `JointRisk_t = (StressPct_t/100) * (EmissionsPct_t/100) * 100`
- Interpretation:
  - `>= 75`: severe co-risk
  - `55-75`: elevated co-risk
  - `< 55`: manageable

## 3) Trigger Thresholds
- RED:
  - `grid_stress_ratio > 1.00` OR `pred_emissions >= P90` OR `JointRisk >= 75`
- AMBER:
  - `0.95 < grid_stress_ratio <= 1.00` OR `P75 <= pred_emissions < P90` OR `55 <= JointRisk < 75`
- GREEN:
  - All lower-risk conditions

## 4) Mitigation Actions
- RED actions (execute within the next dispatch interval):
  - Shift `8%` of flexible compute load out of red hours into low-risk hours.
  - Apply `4%` cooling/efficiency reduction to red-hour facility energy.
  - Pause non-critical batch jobs and defer retries where SLA allows.
- AMBER actions:
  - Pre-stage workload queue for potential RED escalation.
  - Shift up to `3%` voluntary load if queue depth is high.
- GREEN actions:
  - Recover deferred workload while respecting stress headroom.

## 5) Quantified Impact from This Scenario
- Baseline red hours: `5846`
- Mitigated red hours: `2700`
- Red-hour reduction: `3146` (`53.81%`)
- Peak stress reduction: `0.000091` (`0.00%`)
- Emissions reduction: `7586514.93` (`0.68%`)

## 6) Monitoring Cadence
- Recompute danger rule + joint-risk score hourly from updated forecasts.
- Publish red/amber/green state to operators and mentors daily.
- Track KPI deltas weekly:
  - red hours
  - peak stress
  - total emissions
