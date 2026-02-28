# Model Card

Run ID: carbon_20260224_001110
Horizon steps: 1
Seasonal period steps: 24
Selected variant: baseline_core
Stage6 target mode: observed_power_if_available
Calibration valid: True

## Red Flags
- NOT DEFENSIBLE FOR FINAL CLAIMS: negative uplift vs persistence in Stage1_CPU, Stage5_CarbonIntensity, Stage6_Emissions

## Methodology
- Stage 1 and Stage 5 are ML forecasters with rolling time-series CV.
- Stages 2, 3, 4, and 6 are deterministic physics transformations.
- Holdout metrics are computed on a strict final chronological split.

## Leakage Guards
- no_bfill_in_source: PASS
- strict_past_stage1: PASS
- strict_past_stage5: PASS

## Optional Data Usage
- grid_exog: DISABLED
- weather_exog: DISABLED
- power_optional: DISABLED
- cluster_plus_power_optional: DISABLED
- pjm_hourly_demand_optional: DISABLED
- power_calibration: DISABLED

## Headline Holdout Metrics
- Stage1_CPU: RMSE=0.000430, R2=0.147779, Defensible=False
- Stage5_CarbonIntensity: RMSE=14.220820, R2=0.677325, Defensible=False
- Stage6_Emissions: RMSE=14403.126994, R2=-49.139182, Defensible=False

## Baseline Comparison
- Stage1_CPU: model_rmse=0.000430, persistence_rmse=0.000279, uplift=-0.5377
- Stage5_CarbonIntensity: model_rmse=14.220820, persistence_rmse=12.019410, uplift=-0.1832
- Stage6_Emissions: model_rmse=14403.126994, persistence_rmse=14105.802603, uplift=-0.0211

## Limitations
- Accuracy depends on data quality and coverage of CPU + carbon traces.
- If horizon fallback occurred due short sampling horizon feasibility, results are one-step forecasts.