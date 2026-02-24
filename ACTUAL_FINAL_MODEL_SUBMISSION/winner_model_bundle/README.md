# MTFC AI Datacenter Digital Twin
## 100MW Virginia AI Datacenter Carbon & Power Modeling Framework

A comprehensive digital twin simulation for modeling power consumption, carbon emissions, and risk analysis of a hyperscale AI datacenter in Northern Virginia (PJM Interconnection).

---

## 📁 Project Structure

```
FINAL MODEL/
├── main.py                          # Main pipeline orchestrator
├── config.py                        # Configuration & hyperparameters
├── data_loader.py                   # Real data loading (NOAA, EIA, Google)
├── data_preparation.py              # Feature selection & preprocessing
├── feature_engineering.py           # Lag features, time features, scaling
├── eda.py                           # Exploratory data analysis
├── models/                          # ML model implementations
│   ├── lstm_model.py               # LSTM neural network
│   ├── xgboost_model.py            # XGBoost gradient boosting
│   ├── sarimax_model.py            # SARIMAX time series
│   └── ensemble.py                 # Weighted ensemble combiner
├── evaluation.py                    # Model evaluation & metrics
├── forecasting.py                   # Time series forecasting
├── utils.py                         # Logging, plotting utilities
└── outputs/
    ├── figures/                    # All generated visualizations
    ├── models/                     # Saved model weights
    └── results/                    # CSV outputs & reports
```

---

## 📦 Data Sources & Model Transparency

### Real Input Data

| Source | Data | File | Records |
|--------|------|------|---------|
| **NOAA ISD** | Hourly temperature (Ashburn, VA) | `ashburn_va_temperature_2019.csv` | 8,760 hourly |
| **EIA-930** | PJM grid carbon intensity & generation | `pjm_carbon_intensity_2019_hourly.csv` | 8,758 hourly |
| **Google** | Public cluster utilization traces | `google_cluster_utilization_2019.csv` | 2,000 samples |
| **EIA-923** | Generation fuel mix data | `eia923_generation_2019.xlsx` | 53 records |
| **Constants** | Datacenter physical parameters | `datacenter_constants.json` | 5 sections |

### Physics-Informed Target Variable

The target variable `pue` (Power Usage Effectiveness) is **calculated** using physics-based thermal models. This is a **digital twin** approach:

| Component | Source | Type |
|-----------|--------|------|
| Temperature | NOAA ISD | ✅ Real measured data |
| Carbon Intensity | EIA-930 | ✅ Real measured data |
| **PUE(T)** | Thermal physics | ✅ Physics equation (Patterson 2008) |

### ❌ Removed Assumptions (Not Backed by Literature)

The following fabricated assumptions were **DELETED** from the model:

| Removed | Original Value | Reason |
|---------|----------------|--------|
| Base utilization | 70% | Arbitrary; no empirical basis |
| Sinusoidal diurnal pattern | ±10% | Fabricated; no measurement data |
| Weekend reduction | -8% | Arbitrary; AI workloads don't follow weekends |
| `total_power_with_pue` target | IT × PUE | Required fabricated utilization |

### Physics Model References

The PUE calculations are based on **peer-reviewed** datacenter thermal engineering:

1. **PUE Definition** (Total Power = IT Power × PUE):
   - The Green Grid (2007). "Green Grid Data Center Power Efficiency Metrics"
   - EPA Report to Congress (2007). "Server and Data Center Energy Efficiency"

2. **Temperature-PUE Relationship**:
   - Patterson, M.K. (2008). "The Effect of Data Center Temperature on Energy Efficiency." *IEEE ITHERM*
   - Capozzoli & Primiceri (2015). "Cooling systems in data centers." *Energy Procedia*, 83, 484-493
   - ASHRAE TC 9.9 (2016). *Thermal Guidelines for Data Processing Environments*

3. **PUE Range (1.15-1.6)**:
   - Shehabi et al. (2016). "US Data Center Energy Usage Report." *LBNL-1005775*

---

## 🐍 Python Scripts

### Core Pipeline

| Script | Purpose |
|--------|---------|
| **main.py** | Master pipeline: loads data → trains models → evaluates → forecasts |
| **config.py** | Central configuration: paths, hyperparameters, feature priorities |
| **data_loader.py** | Loads REAL data + calculates physics-based power from temperature |
| **data_preparation.py** | Prepares training data with optimized feature selection |
| **feature_engineering.py** | Creates lag features, rolling stats, cyclical time encoding |
| **utils.py** | Logging, figure saving, CSV export utilities |

### Machine Learning Models

| Script | Purpose |
|--------|---------|
| **models/lstm_model.py** | LSTM recurrent neural network for sequence modeling |
| **models/xgboost_model.py** | XGBoost gradient boosting with feature importance |
| **models/sarimax_model.py** | Seasonal ARIMA with exogenous variables |
| **models/ensemble.py** | Weighted ensemble combining LSTM, XGBoost, SARIMAX |
| **evaluation.py** | MAE, RMSE, MAPE metrics + residual diagnostics |
| **forecasting.py** | Multi-step ahead forecasting with confidence intervals |

### Advanced Analytics

| Script | Purpose |
|--------|---------|
| **carbon_emissions.py** | Carbon footprint calculations using EIA emission factors |
| **grid_stress.py** | PJM grid stress scoring and peak demand analysis |
| **run_digital_twin.py** | Full digital twin simulation with scenario modeling |
| **sensitivity_analysis.py** | One-at-a-time (OAT) sensitivity tornado diagrams |
| **advanced_sensitivity_analysis.py** | Spider plots, scenario matrices, waterfall charts |
| **future_predictions.py** | Long-term (2025-2035) scenario projections |

### PhD-Level Risk Analysis

| Script | Purpose |
|--------|---------|
| **advanced_monte_carlo_validation.py** | 10,000-path Monte Carlo + EVT + Copula + Sobol |
| **validation_dashboards.py** | Comprehensive validation dashboard |
| **actuarial_risk_analysis.py** | Insurance-grade risk quantification |

---

## 📊 Generated Figures with Data Sources

### Model Performance

| Figure | Description | Data Source(s) |
|--------|-------------|----------------|
| `actual_vs_pred_xgboost.png` | XGBoost predictions vs actual power | NOAA temp → Physics model |
| `actual_vs_pred_lstm.png` | LSTM predictions vs actual | NOAA temp → Physics model |
| `actual_vs_pred_ensemble.png` | Ensemble model predictions | NOAA temp → Physics model |
| `scatter_xgboost.png` | XGBoost scatter plot with R² | NOAA temp → Physics model |
| `scatter_lstm.png` | LSTM scatter plot | NOAA temp → Physics model |
| `scatter_ensemble.png` | Ensemble scatter plot | NOAA temp → Physics model |
| `residuals_xgboost.png` | XGBoost residual distribution | NOAA temp → Physics model |
| `residuals_lstm.png` | LSTM residual distribution | NOAA temp → Physics model |
| `residuals_ensemble.png` | Ensemble residual distribution | NOAA temp → Physics model |
| `all_models_overlay.png` | All models overlaid on test data | NOAA temp → Physics model |
| `metrics_bar_comparison.png` | MAE/RMSE/MAPE comparison | Model evaluation metrics |
| `xgboost_feature_importance.png` | XGBoost native feature importance | All merged datasets |

### Exploratory Data Analysis

| Figure | Description | Data Source(s) |
|--------|-------------|----------------|
| `eda_timeseries.png` | Full time series of power consumption | NOAA temp → Physics power model |
| `eda_distributions.png` | Histograms of all variables | NOAA + EIA-930 + Physics model |
| `eda_correlation_matrix.png` | Pearson correlation heatmap | All merged datasets |
| `eda_hourly_pattern.png` | Average power by hour of day | NOAA temp → Physics model |
| `eda_day_of_week.png` | Power consumption by day of week | NOAA temp → Physics model |
| `eda_monthly_boxplot.png` | Monthly distribution boxplots | NOAA temp → Physics model |
| `eda_seasonal_decomposition.png` | Trend, seasonal, residual decomposition | NOAA temp → Physics model |
| `eda_acf_pacf.png` | Autocorrelation & partial autocorrelation | NOAA temp → Physics model |
| `eda_temp_vs_power.png` | Temperature vs power scatter | NOAA temp + Physics model |

### Feature Importance

| Figure | Description | Data Source(s) |
|--------|-------------|----------------|
| `correlation_heatmap.png` | Full correlation matrix | NOAA + EIA-930 + EIA-923 + Google |
| `feature_importance_comparison.png` | XGBoost vs Permutation importance | All merged datasets |
| `aggregate_importance_ranking.png` | Combined ranking across methods | All merged datasets |

### Carbon & Grid Analysis

| Figure | Description | Data Source(s) |
|--------|-------------|----------------|
| `carbon_monthly_emissions.png` | Monthly CO₂ emissions breakdown | EIA-930 carbon intensity + Physics power |
| `carbon_ei_projection.png` | Emission intensity projections | EIA-930 + scenario modeling |
| `pjm_carbon_intensity_2019.png` | PJM grid carbon intensity over 2019 | **EIA-930** (primary) |
| `pjm_carbon_intensity_monthly.png` | Monthly average carbon intensity | **EIA-930** |
| `pjm_carbon_intensity_heatmap.png` | Hour × Month carbon intensity heatmap | **EIA-930** |
| `grid_demand_heatmap.png` | Grid demand patterns | **EIA-930** generation data |
| `dc_vs_grid_peaks.png` | Datacenter peaks vs grid peaks | EIA-930 + Physics power model |
| `grid_stress_comparison.png` | Grid stress score analysis | EIA-930 + Physics model |

### Sensitivity Analysis

| Figure | Description | Data Source(s) |
|--------|-------------|----------------|
| `tornado_total_energy_gwh.png` | Tornado: impact on total energy | Datacenter constants + Monte Carlo |
| `tornado_peak_demand_mw.png` | Tornado: impact on peak demand | Datacenter constants + Monte Carlo |
| `tornado_co2_metric_tons.png` | Tornado: impact on CO₂ emissions | EIA-930 + Datacenter constants |
| `tornado_diagram_total_power_mw.png` | Tornado: total power sensitivity | Physics model parameters |
| `tornado_diagram_carbon_tons.png` | Tornado: carbon tons sensitivity | EIA-930 + Physics model |
| `tornado_grid_stress_score.png` | Tornado: grid stress sensitivity | EIA-930 + Physics model |
| `spider_plot_scenarios.png` | Spider/radar plot comparing scenarios | Monte Carlo scenario simulations |
| `waterfall_carbon_decomposition.png` | Waterfall: carbon impact decomposition | EIA-930 + Physics model |
| `sensitivity_dashboard.png` | 9-panel comprehensive dashboard | All data sources combined |

### Scenario Matrices

| Figure | Description | Data Source(s) |
|--------|-------------|----------------|
| `scenario_matrix_pue_utilization_carbon_tons.png` | PUE × Utilization → Carbon | Datacenter constants + EIA-930 |
| `scenario_matrix_temperature_f_pue_total_power_mw.png` | Temp × PUE → Power | NOAA + Datacenter constants |
| `scenario_matrix_it_capacity_mw_annual_growth_rate_carbon_tons.png` | Capacity × Growth → Carbon | Datacenter constants + projections |

### Contour Plots

| Figure | Description | Data Source(s) |
|--------|-------------|----------------|
| `contour_temperature_f_pue_total_power_mw.png` | 3D sensitivity: Temp × PUE | NOAA + Datacenter constants |
| `contour_utilization_renewable_pct_carbon_tons.png` | 3D: Utilization × Renewables | EIA-930 + Monte Carlo |

### Future Projections

| Figure | Description | Data Source(s) |
|--------|-------------|----------------|
| `long_term_scenario_projections.png` | 2025-2035 projections (3 scenarios) | Historical trends + growth assumptions |
| `monthly_forecast_heatmap.png` | Monthly power forecast heatmap | XGBoost model + NOAA temp patterns |

### Forecasting

| Figure | Description | Data Source(s) |
|--------|-------------|----------------|
| `forecast_monthly.png` | Monthly forecast with confidence intervals | XGBoost + NOAA temp projections |
| `forecast_annual_summary.png` | Annual energy consumption summary | XGBoost + NOAA temp projections |

### Monte Carlo & Risk Validation

| Figure | Description | Data Source(s) |
|--------|-------------|----------------|
| `comprehensive_validation_dashboard.png` | **12-panel master dashboard** | All sources + Monte Carlo (N=10,000) |
| `evt_diagnostics.png` | EVT: GPD fit, return levels, VaR/CTE | Monte Carlo simulation results |
| `copula_analysis.png` | Copula scatter & tail dependence | Monte Carlo: Temp × Load × Carbon |
| `copula_detailed_analysis.png` | Detailed copula density & functions | Monte Carlo: Temp × Load × Carbon |
| `sobol_sensitivity.png` | Sobol indices: variance decomposition | Monte Carlo + Sobol sequences |
| `walkforward_backtest.png` | Walk-forward out-of-sample validation | Monte Carlo time series |
| `tail_risk_deep_dive.png` | Exceedance probability, Q-Q, compounds | Monte Carlo extreme events |
| `reverse_stress_test.png` | Fragility surface & failure conditions | Optimization on Monte Carlo |

### Actuarial Risk

| Figure | Description | Data Source(s) |
|--------|-------------|----------------|
| `actuarial_risk_heatmap.png` | Risk exposure heatmap | EIA-930 + Physics model |
| `actuarial_risk_monthly_breakdown.png` | Monthly risk breakdown | EIA-930 + Physics model |
| `actuarial_risk_temperature.png` | Temperature-driven risk analysis | NOAA temp + Physics model |

---

## 🔬 Validation Methodologies

### 1. Extreme Value Theory (EVT)
Fits **Generalized Pareto Distribution** to tail exceedances:
$$G_{\xi,\beta}(y) = 1 - \left(1 + \frac{\xi y}{\beta}\right)^{-1/\xi}$$

### 2. Copula-Based Dependency (Sklar's Theorem)
Models tail dependence between Temperature, IT Load, and Carbon Intensity:
$$H(x,y,z) = C(F_X(x), F_Y(y), F_Z(z))$$

### 3. Sobol Global Sensitivity Analysis
Variance-based decomposition with first-order and total-order indices.

### 4. Walk-Forward Backtesting
Strict chronological out-of-sample validation.

### 5. Reverse Stress Testing
Finds minimum conditions for catastrophic failure.

---

## 📈 Key Results

| Metric | Value |
|--------|-------|
| **LSTM R²** | 0.9847 |
| **XGBoost R²** | 0.9988 |
| **Ensemble R²** | 0.9750 |
| **Mean Annual Carbon Liability** | $17M |
| **99th Percentile Liability** | $32.4M |
| **Primary Variance Driver** | Carbon Intensity (54.8%) |

---

## 🚀 Running the Pipeline

```bash
cd "FINAL MODEL"
python3 main.py                           # Full pipeline
python3 eda.py                            # Exploratory analysis
python3 advanced_monte_carlo_validation.py # Monte Carlo + EVT + Copula
python3 validation_dashboards.py          # Generate dashboards
```

---

## 📄 Reports

| Report | Location |
|--------|----------|
| Feature Importance Analysis | `FEATURE_IMPORTANCE_REPORT.md` |
| Model Validation Framework | `outputs/results/MODEL_VALIDATION_REPORT.md` |
| Pipeline Summary | `outputs/results/pipeline_summary.json` |

---

*MTFC Inventors Team — February 2026*
