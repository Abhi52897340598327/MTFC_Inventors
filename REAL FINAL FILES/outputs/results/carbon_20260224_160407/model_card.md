# Model Card

Run ID: carbon_20260224_160407
Horizon steps: 1
Seasonal period steps: 24
Selected variant: baseline_core
Stage6 target mode: observed_power_if_available
Calibration valid: True

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
- Stage1_CPU: RMSE=0.000276, R2=0.647786, Defensible=True
- Stage5_CarbonIntensity: RMSE=10.888355, R2=0.810835, Defensible=True
- Stage6_Emissions: RMSE=858.758723, R2=0.821760, Defensible=True

## Baseline Comparison
- Stage1_CPU: model_rmse=0.000276, persistence_rmse=0.000279, uplift=0.0114
- Stage5_CarbonIntensity: model_rmse=10.888355, persistence_rmse=12.019410, uplift=0.0941
- Stage6_Emissions: model_rmse=858.758723, persistence_rmse=4752.259884, uplift=0.8193

## Limitations
- Accuracy depends on data quality and coverage of CPU + carbon traces.
- If horizon fallback occurred due short sampling horizon feasibility, results are one-step forecasts.