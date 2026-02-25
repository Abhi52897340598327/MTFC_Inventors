# Model Card

Run ID: carbon_20260223_220312
Horizon steps: 1
Seasonal period steps: 24

## Methodology
- Stage 1 and Stage 5 are ML forecasters with rolling time-series CV.
- Stages 2, 3, 4, and 6 are deterministic physics transformations.
- Holdout metrics are computed on a strict final chronological split.

## Leakage Guards
- no_bfill_in_source: PASS
- strict_past_stage1: PASS
- strict_past_stage5: PASS

## Headline Holdout Metrics
- Stage1_CPU: RMSE=0.001511, R2=-0.030444, Defensible=False
- Stage5_CarbonIntensity: RMSE=0.000009, R2=-25792440300062500.000000, Defensible=False
- Stage6_Emissions: RMSE=43.833074, R2=-0.030407, Defensible=False

## Baseline Comparison
- Stage1_CPU: model_rmse=0.001511, persistence_rmse=0.001339, uplift=-0.1283
- Stage5_CarbonIntensity: model_rmse=0.000009, persistence_rmse=0.000000, uplift=nan
- Stage6_Emissions: model_rmse=43.833074, persistence_rmse=38.851105, uplift=-0.1282

## Limitations
- Accuracy depends on data quality and coverage of CPU + carbon traces.
- If horizon fallback occurred due short sampling horizon feasibility, results are one-step forecasts.