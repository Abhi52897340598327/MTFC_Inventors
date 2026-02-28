# Post-hoc Sensitivity + Recommendations Visual Report

- Run ID: `carbon_20260224_170934`
- Generated UTC: `2026-02-24 17:11:23`
- Retraining status: `NONE` (no model retraining, no model mutation).
- Method: deterministic post-hoc calculations using active physics equations and existing cleaned datasets.

## Key Sensitivity Findings
### Sobol (Top 3 by first-order S1)
- `idle_power_fraction`: S1=0.4942, ST=0.4470
- `temperature_f`: S1=0.3101, ST=0.2208
- `carbon_intensity`: S1=0.3059, ST=0.3201

### Sobol Energy Forecast Sensitivity (Top 3 by first-order S1)
- `idle_power_fraction`: S1=0.7186, ST=0.6611
- `temperature_f`: S1=0.4653, ST=0.3272
- `pue_cpu_coef`: S1=0.0001, ST=0.0000

### Tornado OAT (Top 3 absolute swings)
- `idle_power_fraction`: swing=3949.05 kg/h, low=-16.2%, high=+16.2%
- `carbon_intensity`: swing=3182.65 kg/h, low=-16.5%, high=+9.5%
- `temperature_f`: swing=2444.42 kg/h, low=-1.5%, high=+18.5%

### Tornado OAT Energy (Top 3 absolute swings)
- `idle_power_fraction`: swing=96.94 GWh/yr, low=-16.2%, high=+16.2%
- `temperature_f`: swing=60.01 GWh/yr, low=-1.5%, high=+18.5%
- `pue_temp_coef`: swing=2.96 GWh/yr, low=-0.5%, high=+0.5%

### Copula Tail Dependence (Top 3 upper-tail at q=0.95)
- `temp_vs_energy`: lambda_U=0.9730, lambda_L=0.0270
- `temp_vs_emissions`: lambda_U=0.7838, lambda_L=0.0000
- `carbon_vs_emissions`: lambda_U=0.4865, lambda_L=0.9459

### Copula Energy Tail Dependence (Top 3 upper-tail at q=0.95)
- `temp_vs_energy`: lambda_U=0.9730, lambda_L=0.0270
- `carbon_vs_energy`: lambda_U=0.2973, lambda_L=0.2432
- `cpu_vs_energy`: lambda_U=0.1351, lambda_L=0.7838

## Generated Diagrams
- `carbon_intensity_heatmap.svg`
- `tornado_oat_emissions.svg`
- `tornado_oat_energy.svg`
- `copula_tail_dashboard.svg`
- `copula_energy_tail_bars.svg`
- `sobol_global_sensitivity.svg`
- `sobol_energy_sensitivity.svg`
- `recommendation_scenario_radar.svg`
- `recommendation_energy_peak_projection.svg`
- `recommendation_mitigation_impact.svg`

## Data Tables
- `sobol_indices.csv`
- `sobol_indices_energy.csv`
- `tornado_oat.csv`
- `tornado_oat_energy.csv`
- `copula_tail_dependence.csv`
- `copula_tail_curves.csv`
- `carbon_intensity_heatmap.csv`
- `recommendation_scenarios.csv`
- `energy_forecast_scenarios.csv`
- `recommendation_mitigation.csv`

## Latest Run Metadata Snapshot
- `selected_variant`: `exog_enhanced`
- `stage6_target_mode`: `physics`
- `holdout_rows_evaluated`: `111`

## Model Card Source
- `/Users/abhiraamvenigalla/MTFC_Inventors/REAL FINAL FILES/outputs/results/carbon_20260224_170934/model_card.md`