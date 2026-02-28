# Model Card

Run ID: carbon_20260223_231616
Horizon steps: 1
Seasonal period steps: 24
Selected variant: baseline_core

## Methodology
- Stage 1 and Stage 5 are ML forecasters with rolling time-series CV.
- Stages 2, 3, 4, and 6 are deterministic physics transformations.
- Holdout metrics are computed on a strict final chronological split.

## Leakage Guards
- no_bfill_in_source: FAIL
- strict_past_stage1: PASS
- strict_past_stage5: PASS

## Optional Data Usage
- grid_exog: DISABLED
- weather_exog: DISABLED
- power_optional: DISABLED
- power_calibration: DISABLED

## Headline Holdout Metrics
- Stage1_CPU: RMSE=0.000398, R2=0.269274, Defensible=False
- Stage5_CarbonIntensity: RMSE=14.043227, R2=0.685334, Defensible=False
- Stage6_Emissions: RMSE=14419.756158, R2=-49.255025, Defensible=False

## Baseline Comparison
- Stage1_CPU: model_rmse=0.000398, persistence_rmse=0.000279, uplift=-0.4239
- Stage5_CarbonIntensity: model_rmse=14.043227, persistence_rmse=12.019410, uplift=-0.1684
- Stage6_Emissions: model_rmse=14419.756158, persistence_rmse=14132.051070, uplift=-0.0204

## Limitations
- Accuracy depends on data quality and coverage of CPU + carbon traces.
- If horizon fallback occurred due short sampling horizon feasibility, results are one-step forecasts.