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
├── outputs/
│   ├── figures/                    # All generated visualizations
│   ├── models/                     # Saved model weights
│   └── results/                    # CSV outputs & reports
└── [Analysis Scripts]              # Specialized analysis modules
```

---

## 🐍 Python Scripts

### Core Pipeline

| Script | Purpose |
|--------|---------|
| **main.py** | Master pipeline: loads data → trains models → evaluates → forecasts |
| **config.py** | Central configuration: paths, hyperparameters, feature priorities |
| **data_loader.py** | Loads REAL data only (NOAA weather, EIA carbon, Google cluster traces) |
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

### Exploratory Data Analysis

| Script | Purpose |
|--------|---------|
| **eda.py** | Time series plots, distributions, correlations, ACF/PACF |
| **feature_importance_analysis.py** | XGBoost, permutation, and correlation-based feature ranking |

### Advanced Analytics

| Script | Purpose |
|--------|---------|
| **carbon_emissions.py** | Carbon footprint calculations using EIA emission factors |
| **grid_stress.py** | PJM grid stress scoring and peak demand analysis |
| **run_digital_twin.py** | Full digital twin simulation with scenario modeling |
| **sensitivity_analysis.py** | One-at-a-time (OAT) sensitivity tornado diagrams |
| **advanced_sensitivity_analysis.py** | Spider plots, scenario matrices, waterfall charts, contour plots |
| **future_predictions.py** | Long-term (2025-2035) scenario projections |

### PhD-Level Risk Analysis

| Script | Purpose |
|--------|---------|
| **advanced_monte_carlo_validation.py** | 10,000-path Monte Carlo + EVT + Copula + Sobol + Stress Testing |
| **validation_dashboards.py** | Comprehensive 12-panel validation dashboard |
| **actuarial_risk_analysis.py** | Insurance-grade risk quantification |
| **pjm_carbon_analysis.py** | PJM-specific carbon intensity analysis |

### Feature Engineering & Model Justification

| Script | Purpose |
|--------|---------|
| **ridge_feature_combiner.py** | Ridge Regression to combine feature importance weights from multiple methods |
| **architecture_ablation_study.py** | Before/after analysis justifying each model component |
| **visualizations_3d.py** | 3D visualization suite complementing 2D figures |

### Data Collection

| Script | Purpose |
|--------|---------|
| **download_noaa_weather.py** | Fetch NOAA temperature data for Ashburn, VA |
| **download_pjm_data.py** | Fetch PJM grid data via EIA API |
| **download_all_data.py** | Master data download orchestrator |

---

## 📊 Generated Figures

### Model Performance (6 figures)

| Figure | Description |
|--------|-------------|
| `actual_vs_pred_xgboost.png` | XGBoost predictions vs actual power consumption |
| `actual_vs_pred_lstm.png` | LSTM predictions vs actual |
| `actual_vs_pred_ensemble.png` | Ensemble model predictions vs actual |
| `scatter_xgboost.png` | XGBoost scatter plot with R² |
| `scatter_lstm.png` | LSTM scatter plot |
| `scatter_ensemble.png` | Ensemble scatter plot |
| `residuals_xgboost.png` | XGBoost residual distribution |
| `residuals_lstm.png` | LSTM residual distribution |
| `residuals_ensemble.png` | Ensemble residual distribution |
| `all_models_overlay.png` | All models overlaid on test data |
| `metrics_bar_comparison.png` | MAE/RMSE/MAPE comparison bar chart |
| `xgboost_feature_importance.png` | XGBoost native feature importance |

### Exploratory Data Analysis (9 figures)

| Figure | Description |
|--------|-------------|
| `eda_timeseries.png` | Full time series of power consumption |
| `eda_distributions.png` | Histograms of all variables |
| `eda_correlation_matrix.png` | Pearson correlation heatmap |
| `eda_hourly_pattern.png` | Average power by hour of day |
| `eda_day_of_week.png` | Power consumption by day of week |
| `eda_monthly_boxplot.png` | Monthly distribution boxplots |
| `eda_seasonal_decomposition.png` | Trend, seasonal, residual decomposition |
| `eda_acf_pacf.png` | Autocorrelation & partial autocorrelation |
| `eda_temp_vs_power.png` | Temperature vs power scatter |

### Feature Importance (3 figures)

| Figure | Description |
|--------|-------------|
| `correlation_heatmap.png` | Full correlation matrix with hierarchical clustering |
| `feature_importance_comparison.png` | XGBoost vs Permutation importance comparison |
| `aggregate_importance_ranking.png` | Combined ranking across all methods |

### Carbon & Grid Analysis (7 figures)

| Figure | Description |
|--------|-------------|
| `carbon_monthly_emissions.png` | Monthly CO₂ emissions breakdown |
| `carbon_ei_projection.png` | Emission intensity projections |
| `pjm_carbon_intensity_2019.png` | PJM grid carbon intensity over 2019 |
| `pjm_carbon_intensity_monthly.png` | Monthly average carbon intensity |
| `pjm_carbon_intensity_heatmap.png` | Hour × Month carbon intensity heatmap |
| `grid_demand_heatmap.png` | Grid demand patterns |
| `dc_vs_grid_peaks.png` | Datacenter peaks vs grid peaks |
| `grid_stress_comparison.png` | Grid stress score analysis |

### Sensitivity Analysis (9 figures)

| Figure | Description |
|--------|-------------|
| `tornado_total_energy_gwh.png` | Tornado: impact on total energy |
| `tornado_peak_demand_mw.png` | Tornado: impact on peak demand |
| `tornado_co2_metric_tons.png` | Tornado: impact on CO₂ emissions |
| `tornado_diagram_total_power_mw.png` | Tornado: total power sensitivity |
| `tornado_diagram_carbon_tons.png` | Tornado: carbon tons sensitivity |
| `tornado_grid_stress_score.png` | Tornado: grid stress sensitivity |
| `spider_plot_scenarios.png` | Spider/radar plot comparing scenarios |
| `waterfall_carbon_decomposition.png` | Waterfall: carbon impact decomposition |
| `sensitivity_dashboard.png` | 9-panel comprehensive sensitivity dashboard |

### Scenario Matrices (3 figures)

| Figure | Description |
|--------|-------------|
| `scenario_matrix_pue_utilization_carbon_tons.png` | PUE × Utilization → Carbon |
| `scenario_matrix_temperature_f_pue_total_power_mw.png` | Temp × PUE → Power |
| `scenario_matrix_it_capacity_mw_annual_growth_rate_carbon_tons.png` | Capacity × Growth → Carbon |

### Contour Plots (2 figures)

| Figure | Description |
|--------|-------------|
| `contour_temperature_f_pue_total_power_mw.png` | 3D sensitivity surface: Temp × PUE |
| `contour_utilization_renewable_pct_carbon_tons.png` | 3D surface: Utilization × Renewables |

### Future Projections (2 figures)

| Figure | Description |
|--------|-------------|
| `long_term_scenario_projections.png` | 2025-2035 projections (3 growth scenarios) |
| `monthly_forecast_heatmap.png` | Monthly power forecast heatmap |

### Forecasting (2 figures)

| Figure | Description |
|--------|-------------|
| `forecast_monthly.png` | Monthly forecast with confidence intervals |
| `forecast_annual_summary.png` | Annual energy consumption summary |

### Monte Carlo & Risk Validation (7 figures)

| Figure | Description |
|--------|-------------|
| `comprehensive_validation_dashboard.png` | **12-panel master dashboard** combining all validation methods |
| `evt_diagnostics.png` | Extreme Value Theory: GPD fit, return levels, VaR/CTE |
| `copula_analysis.png` | Copula scatter plots & tail dependence |
| `copula_detailed_analysis.png` | Detailed copula density & tail functions |
| `sobol_sensitivity.png` | Sobol indices: variance decomposition |
| `walkforward_backtest.png` | Walk-forward out-of-sample validation |
| `tail_risk_deep_dive.png` | Exceedance probability, Q-Q plots, compound events |
| `reverse_stress_test.png` | Fragility surface & minimum failure conditions |

### Actuarial Risk (3 figures)

| Figure | Description |
|--------|-------------|
| `actuarial_risk_heatmap.png` | Risk exposure heatmap |
| `actuarial_risk_monthly_breakdown.png` | Monthly risk breakdown |
| `actuarial_risk_temperature.png` | Temperature-driven risk analysis |

### Ridge Feature Combiner (2 figures)

| Figure | Description |
|--------|-------------|
| `ridge_feature_combiner.png` | Ridge-learned method weights, combined importance ranking, heatmap |
| `ridge_vs_simple_average.png` | Comparison of simple average vs Ridge-weighted importance |

### Architecture Ablation Study (2 figures)

| Figure | Description |
|--------|-------------|
| `architecture_ablation_study.png` | **6-panel dashboard** showing feature group ablation, model comparison, incremental improvements |
| `before_after_component_comparison.png` | Before/After R² improvement for each model component |

### 3D Visualizations (7 figures)

| Figure | Description |
|--------|-------------|
| `3d_temp_hour_power_surface.png` | 3D Surface: Temperature × Hour → Power Consumption |
| `3d_pue_utilization_carbon_surface.png` | 3D Surface: PUE × Utilization → Carbon Emissions |
| `3d_feature_importance_multimethod.png` | 3D Bar Chart: Feature importance across multiple methods |
| `3d_time_series_ribbon.png` | 3D Ribbon: Hour × Day of Year → PUE patterns |
| `3d_monte_carlo_risk_scatter.png` | 3D Scatter: Temperature × Utilization × Carbon Risk |
| `3d_sensitivity_response_surface.png` | 3D Response Surface: Renewable % × Carbon Price sensitivity |
| `3d_carbon_intensity_surface.png` | 3D Surface: Hour × Month → PJM Carbon Intensity |

---

## 🔬 Validation Methodologies

### 1. Extreme Value Theory (EVT)
Fits **Generalized Pareto Distribution** to tail exceedances:
$$G_{\xi,\beta}(y) = 1 - \left(1 + \frac{\xi y}{\beta}\right)^{-1/\xi}$$

- **VaR 99%**: Value at Risk at 99th percentile
- **CTE 99%**: Conditional Tail Expectation (Expected Shortfall)

### 2. Copula-Based Dependency (Sklar's Theorem)
Models tail dependence between Temperature, IT Load, and Carbon Intensity:
$$H(x,y,z) = C(F_X(x), F_Y(y), F_Z(z))$$

- Validates "Double Whammy" hypothesis (heat + dirty grid)

### 3. Sobol Global Sensitivity Analysis
Variance-based decomposition:
- **First-order $S_i$**: Direct variable effect
- **Total-order $S_{Ti}$**: Effect including all interactions

### 4. Walk-Forward Backtesting
Strict chronological out-of-sample validation:
- Expanding window & rolling window methods
- No future information leakage

### 5. Reverse Stress Testing
Finds minimum conditions for catastrophic failure:
- Uses differential evolution optimization
- Maps "fragility surface" of operations

### 6. Ridge Regression Feature Combiner
Combines feature importance weights from multiple methods using regularized regression:
$$\hat{w} = \arg\min_w \|y - Xw\|^2 + \alpha\|w\|^2$$

Where:
- $X$ = normalized importance scores from each method (columns)
- $y$ = actual feature contribution (computed via single-feature R²)
- $\alpha$ = regularization parameter (optimized via cross-validation)

**Methods Combined:**
- Correlation (absolute)
- Gradient Boosting importance
- Random Forest importance
- Mutual Information
- Permutation Importance

### 7. Architecture Ablation Study
Systematic evaluation of each component's contribution:
- **Feature Ablation**: Cumulatively adds feature groups, measures R² improvement
- **Model Ablation**: Compares Linear → Ridge → RF → GB → XGBoost → GRU → Ensemble
- **Before/After Visualization**: Explicit improvement from each addition

---

## 📈 Key Results

| Metric | Value |
|--------|-------|
| **Mean Annual Carbon Liability** | $17M |
| **99th Percentile Liability** | $32.4M |
| **VaR 99.9% (Risk Index)** | 195.5 |
| **Primary Variance Driver** | Carbon Intensity (54.8%) |
| **Walk-Forward MAPE** | 13.4% |
| **Upper Tail Dependence (λ_U)** | 0.174 |

---

## 🚀 Running the Pipeline

### Full Pipeline
```bash
cd "FINAL MODEL"
python3 main.py
```

### Individual Analyses
```bash
# Exploratory Data Analysis
python3 eda.py

# Feature Importance
python3 feature_importance_analysis.py

# Ridge Feature Combiner (combines importance weights)
python3 ridge_feature_combiner.py

# Architecture Ablation Study (justifies each component)
python3 architecture_ablation_study.py

# 3D Visualizations
python3 visualizations_3d.py

# Sensitivity Analysis
python3 sensitivity_analysis.py
python3 advanced_sensitivity_analysis.py

# Future Projections
python3 future_predictions.py

# Monte Carlo & Validation
python3 advanced_monte_carlo_validation.py
python3 validation_dashboards.py

# Digital Twin Simulation
python3 run_digital_twin.py
```

---

## 📦 Dependencies

```
numpy>=1.20
pandas>=1.3
matplotlib>=3.4
seaborn>=0.11
scikit-learn>=1.0
xgboost>=1.5
tensorflow>=2.8
statsmodels>=0.13
scipy>=1.7
```

---

## 📚 Data Sources

| Source | Data | File |
|--------|------|------|
| **NOAA** | Hourly temperature (Ashburn, VA) | `ashburn_va_temperature_2019.csv` |
| **EIA-930** | PJM grid carbon intensity | `pjm_carbon_intensity_2019_hourly.csv` |
| **Google** | Public cluster utilization traces | `google_cluster_utilization_2019.csv` |
| **EIA-923** | Generation fuel mix data | `eia923_generation_2019.xlsx` |
| **Constants** | Datacenter physical parameters | `datacenter_constants.json` |

All data is **REAL** - no synthetic data generation is used. Power consumption is derived from physics-based models driven by actual temperature measurements.

---

## 📄 Reports Generated

| Report | Location |
|--------|----------|
| Feature Importance Analysis | `FEATURE_IMPORTANCE_REPORT.md` |
| Model Validation Framework | `outputs/results/MODEL_VALIDATION_REPORT.md` |
| Pipeline Summary | `outputs/results/pipeline_summary.json` |
| Validation Results | `outputs/risk_analysis/validation_results.json` |

---

## 👥 Authors

MTFC Inventors Team

---

## 📖 Citation

If using this framework for academic work, please cite:
```
MTFC AI Datacenter Digital Twin Framework
Virginia 100MW Hyperscale Datacenter Carbon Modeling
February 2026
```
