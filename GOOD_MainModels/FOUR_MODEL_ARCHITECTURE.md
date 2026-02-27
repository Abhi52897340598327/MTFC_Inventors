# MTFC Four-Model Architecture

This directory now includes a clean, competition-focused four-model path.

## Models

1. `sarimax_energy_usage.py`  
   Model 1: SARIMAX monthly energy forecast (GWh).

2. `model2_sarimax_carbon_emissions.py`  
   Model 2: SARIMAX monthly CO2 emissions forecast (tonnes) with exogenous features:
   - temperature
   - carbon intensity
   - cooling degree days
   - month sin/cos

3. `model3_grid_stress_analysis.py`  
   Model 3: reserve-margin erosion, stress-hour counting, and peak premium grid-stress cost.

4. `model4_monetization_cost_benefit.py`  
   Model 4: monetizes Risk 1 + Risk 2 and computes recommendation economics (CapEx, annual savings, payback, 10-year NPV).

## One-command run

From repository root:

```bash
python GOOD_MainModels/run_mtfc_four_models.py
```

## Core outputs

- `GOOD_MainModels/energy_usage_forecast.csv`
- `GOOD_MainModels/carbon_emissions_forecast.csv`
- `GOOD_MainModels/grid_stress_annual_summary.csv`
- `GOOD_MainModels/annual_risk_monetization.csv`
- `GOOD_MainModels/recommendation_cost_benefit.csv`

## Notes

- The legacy multi-stage pipeline remains unchanged.
- The new path is additive and isolated so paper work can reference only these four models.
