# Model Card

Run ID: carbon_20260224_155159
Horizon steps: 1
Seasonal period steps: 24
Selected variant: baseline_core
Stage6 target mode: observed_power_if_available
Calibration valid: True

## Red Flags
- NOT DEFENSIBLE FOR FINAL CLAIMS: negative uplift vs persistence in Stage6_Emissions

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
- Stage1_CPU: RMSE=0.000276, R2=0.647045, Defensible=True
- Stage5_CarbonIntensity: RMSE=11.013778, R2=0.806452, Defensible=True
- Stage6_Emissions: RMSE=14237.917398, R2=-47.995546, Defensible=False

## Baseline Comparison
- Stage1_CPU: model_rmse=0.000276, persistence_rmse=0.000279, uplift=0.0104
- Stage5_CarbonIntensity: model_rmse=11.013778, persistence_rmse=12.019410, uplift=0.0837
- Stage6_Emissions: model_rmse=14237.917398, persistence_rmse=14105.802603, uplift=-0.0094

## Limitations
- Accuracy depends on data quality and coverage of CPU + carbon traces.
- If horizon fallback occurred due short sampling horizon feasibility, results are one-step forecasts.