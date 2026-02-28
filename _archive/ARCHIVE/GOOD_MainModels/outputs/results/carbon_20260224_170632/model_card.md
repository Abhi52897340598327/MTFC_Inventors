# Model Card

Run ID: carbon_20260224_170632
Horizon steps: 1
Seasonal period steps: 24
Selected variant: exog_enhanced
Stage6 target mode: physics
Calibration valid: False

## Red Flags
- NOT DEFENSIBLE FOR FINAL CLAIMS: negative uplift vs persistence in Stage4_TotalPower_CombinedML, Stage7_EnergyUsageML

## Methodology
- Stage 1 and Stage 5 are ML forecasters with rolling time-series CV.
- Stages 2, 3, 4, and 6 are deterministic physics transformations.
- Additional learned models: combined total power (Stage4_CombinedML) and energy usage (Stage7).
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
- Stage1_CPU: RMSE=0.000275, R2=0.650058, Defensible=True
- Stage4_TotalPower_CombinedML: RMSE=1.137760, R2=0.125649, Defensible=False
- Stage5_CarbonIntensity: RMSE=10.786267, R2=0.814366, Defensible=True
- Stage6_Emissions: RMSE=396.046198, R2=0.951319, Defensible=True
- Stage6_Emissions_CombinedPowerML: RMSE=913.142760, R2=0.798469, Defensible=True
- Stage7_EnergyUsageML: RMSE=1.143627, R2=0.116609, Defensible=False

## Baseline Comparison
- Stage1_CPU: model_rmse=0.000275, persistence_rmse=0.000279, uplift=0.0146
- Stage5_CarbonIntensity: model_rmse=10.786267, persistence_rmse=12.019410, uplift=0.1026
- Stage6_Emissions: model_rmse=396.046198, persistence_rmse=436.565091, uplift=0.0928
- Stage4_TotalPower_CombinedML: model_rmse=1.137760, persistence_rmse=0.942466, uplift=-0.2072
- Stage7_EnergyUsageML: model_rmse=1.143627, persistence_rmse=0.942466, uplift=-0.2134
- Stage6_Emissions_CombinedPowerML: model_rmse=913.142760, persistence_rmse=14132.051070, uplift=0.9354

## Limitations
- Accuracy depends on data quality and coverage of CPU + carbon traces.
- If horizon fallback occurred due short sampling horizon feasibility, results are one-step forecasts.