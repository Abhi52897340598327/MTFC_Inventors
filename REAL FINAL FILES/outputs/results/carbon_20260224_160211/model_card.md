# Model Card

Run ID: carbon_20260224_160211
Horizon steps: 1
Seasonal period steps: 24
Selected variant: exog_enhanced
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
- grid_exog: ENABLED
- weather_exog: ENABLED
- power_optional: ENABLED
- cluster_plus_power_optional: ENABLED
- pjm_hourly_demand_optional: ENABLED
- power_calibration: ENABLED

## Headline Holdout Metrics
- Stage1_CPU: RMSE=0.000277, R2=0.645527, Defensible=False
- Stage5_CarbonIntensity: RMSE=10.857113, R2=0.811919, Defensible=True
- Stage6_Emissions: RMSE=1123.584492, R2=0.694877, Defensible=False

## Baseline Comparison
- Stage1_CPU: model_rmse=0.000277, persistence_rmse=0.000279, uplift=0.0083
- Stage5_CarbonIntensity: model_rmse=10.857113, persistence_rmse=12.019410, uplift=0.0967
- Stage6_Emissions: model_rmse=1123.584492, persistence_rmse=1113.348422, uplift=-0.0092

## Limitations
- Accuracy depends on data quality and coverage of CPU + carbon traces.
- If horizon fallback occurred due short sampling horizon feasibility, results are one-step forecasts.