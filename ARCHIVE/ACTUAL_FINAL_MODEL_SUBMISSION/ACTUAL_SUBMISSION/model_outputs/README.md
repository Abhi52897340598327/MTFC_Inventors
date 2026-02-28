# ACTUAL SUBMISSION — Model Outputs

Generated on: 2026-02-24

## What was rerun
- `advanced_sensitivity_analysis.py` (deterministic OAT/tornado + scenario sensitivity visuals)
- `advanced_monte_carlo_validation.py` (copula tail dependence + Sobol variance-based sensitivity)

Pipeline/model training code was not modified during this step.

## Sensitivity Figures
- `sensitivity/tornado_diagram_carbon_tons.png`
- `sensitivity/tornado_diagram_total_power_mw.png`
- `sensitivity/sobol_sensitivity.png`
- `sensitivity/copula_analysis.png`
- `sensitivity/sensitivity_dashboard.png`
- `sensitivity/spider_plot_scenarios.png`
- `sensitivity/scenario_matrix_pue_utilization_carbon_tons.png`
- `sensitivity/sensitivity_analysis_emissions.png`

## Sensitivity Data Outputs
- `sensitivity/validation_results.json`
- `sensitivity/monte_carlo_simulation_results.csv`

## New Virginia Datacenter Map Output
- `maps/virginia_datacenter_location_heatmap.png`

Notes:
- Heatmap combines project context (Ashburn-centered Virginia deployment) with common Virginia datacenter hubs.
- Expanded statewide coverage includes additional central, valley, southside, southwest, and coastal anchors.
- The map is a weighted relative density visualization (not an official census count of facilities).
