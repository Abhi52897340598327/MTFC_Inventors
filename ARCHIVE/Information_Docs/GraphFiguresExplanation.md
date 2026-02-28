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

## 📊 Generated Figures & Business Implications

### Model Performance (12 figures)

| Figure | Description | **Business Implications** |
|--------|-------------|---------------------------|
| `actual_vs_pred_xgboost.png` | XGBoost predictions vs actual power consumption | **Operational Planning**: Shows model accuracy for capacity planning. Tight alignment indicates reliable demand forecasting for utility contracts and cooling system scheduling. Gaps reveal periods where the model struggles (e.g., extreme weather, anomalous workloads). |
| `actual_vs_pred_lstm.png` | LSTM predictions vs actual | **Temporal Pattern Recognition**: LSTM captures sequential dependencies. Performance here indicates how well the model learns daily/weekly cycles—critical for shift scheduling and maintenance windows. |
| `actual_vs_pred_ensemble.png` | Ensemble model predictions vs actual | **Risk Mitigation**: Ensemble combines multiple models' strengths. Superior performance here justifies the computational overhead and provides the most reliable basis for financial planning. |
| `scatter_xgboost.png` | XGBoost scatter plot with R² | **Model Reliability**: R² near 1.0 indicates the model explains nearly all variance. Points far from the diagonal reveal edge cases requiring manual oversight or additional features. |
| `scatter_lstm.png` | LSTM scatter plot | **Deep Learning Viability**: Demonstrates whether neural networks add value over tree-based methods. Wider scatter suggests LSTM may need more data or architectural tuning. |
| `scatter_ensemble.png` | Ensemble scatter plot | **Production Readiness**: This is the "money plot"—tight clustering around the diagonal validates the ensemble for production deployment in automated decision systems. |
| `residuals_xgboost.png` | XGBoost residual distribution | **Systematic Bias Detection**: Normal distribution centered at zero indicates unbiased predictions. Skewness reveals systematic over/under-prediction requiring model recalibration. |
| `residuals_lstm.png` | LSTM residual distribution | **Neural Network Calibration**: Heavy tails indicate the LSTM may produce occasional extreme errors—important for setting safety margins in capacity reserves. |
| `residuals_ensemble.png` | Ensemble residual distribution | **Error Budgeting**: The ensemble's residual spread directly informs the uncertainty buffer needed in power purchase agreements and SLA commitments. |
| `all_models_overlay.png` | All models overlaid on test data | **Model Selection**: Visual comparison reveals which model tracks best during peaks, troughs, and transitions. Informs dynamic model switching strategies for different operating conditions. |
| `metrics_bar_comparison.png` | MAE/RMSE/MAPE comparison bar chart | **Executive Summary**: Single-glance model comparison for stakeholder communication. MAPE < 5% is typically required for utility-grade forecasting; this chart validates that threshold. |
| `xgboost_feature_importance.png` | XGBoost native feature importance | **Investment Prioritization**: Identifies which sensors/data sources drive predictions most. High-importance features justify monitoring investment; low-importance features may be candidates for elimination to reduce data costs. |

### Exploratory Data Analysis (9 figures)

| Figure | Description | **Business Implications** |
|--------|-------------|---------------------------|
| `eda_timeseries.png` | Full time series of power consumption | **Trend Identification**: Reveals long-term growth patterns, seasonality, and anomalies. Upward trends indicate capacity expansion needs; sudden drops may signal outages or efficiency improvements requiring investigation. |
| `eda_distributions.png` | Histograms of all variables | **Operating Envelope**: Shows typical operating ranges. Bimodal distributions suggest distinct operating modes (day/night, weekday/weekend). Helps set alert thresholds and define "normal" operations. |
| `eda_correlation_matrix.png` | Pearson correlation heatmap | **Causal Understanding**: Strong correlations (>0.7) reveal coupled variables. Temperature-power correlation quantifies cooling costs; carbon-hour correlation informs carbon-aware scheduling strategies. |
| `eda_hourly_pattern.png` | Average power by hour of day | **Demand Response Opportunity**: Peak hours identify when to curtail non-critical workloads or shift to battery backup. Trough hours are optimal for maintenance and batch processing. |
| `eda_day_of_week.png` | Power consumption by day of week | **Staffing & Maintenance Windows**: Lower weekend demand may allow scheduled maintenance. Consistent weekday patterns validate workload assumptions for capacity planning. |
| `eda_monthly_boxplot.png` | Monthly distribution boxplots | **Seasonal Budgeting**: Summer peaks drive cooling costs and carbon liability. Wider boxes indicate higher variability requiring larger operational buffers. Informs annual energy procurement strategy. |
| `eda_seasonal_decomposition.png` | Trend, seasonal, residual decomposition | **Forecasting Foundation**: Clean trend separation validates time-series modeling approach. Large residuals indicate external factors (outages, workload spikes) not captured by seasonal patterns. |
| `eda_acf_pacf.png` | Autocorrelation & partial autocorrelation | **Model Selection Guide**: ACF decay rate determines ARIMA order. Strong 24-hour lags validate hourly seasonality. PACF cutoff suggests optimal lag features for ML models. |
| `eda_temp_vs_power.png` | Temperature vs power scatter | **Cooling Efficiency Baseline**: The slope quantifies MW/°F cooling penalty. Steeper slopes indicate poor cooling efficiency; comparison against industry benchmarks (0.5-1.5 MW/°F) guides infrastructure investment. |

### Feature Importance (3 figures)

| Figure | Description | **Business Implications** |
|--------|-------------|---------------------------|
| `correlation_heatmap.png` | Full correlation matrix with hierarchical clustering | **Multicollinearity Detection**: Highly correlated feature clusters may cause model instability. Identifies redundant sensors that can be decommissioned to reduce data infrastructure costs without losing predictive power. |
| `feature_importance_comparison.png` | XGBoost vs Permutation importance comparison | **Robustness Validation**: Agreement between methods confirms genuine importance; disagreement flags features that may be artifacts of model structure. Critical for regulatory reporting where feature justification is required. |
| `aggregate_importance_ranking.png` | Combined ranking across all methods | **Strategic Data Investment**: The definitive guide for data infrastructure priorities. Top-ranked features justify redundant sensors and premium data feeds; bottom-ranked features are candidates for deprecation. |

### Carbon & Grid Analysis (8 figures)

| Figure | Description | **Business Implications** |
|--------|-------------|---------------------------|
| `carbon_monthly_emissions.png` | Monthly CO₂ emissions breakdown | **ESG Reporting**: Direct input for sustainability reports and carbon credit calculations. Summer peaks may trigger carbon offset purchases; winter lows demonstrate seasonal efficiency gains to stakeholders. |
| `carbon_ei_projection.png` | Emission intensity projections | **Decarbonization Roadmap**: Shows trajectory toward net-zero goals. Gap between current and target intensities quantifies required renewable energy procurement or efficiency investments. |
| `pjm_carbon_intensity_2019.png` | PJM grid carbon intensity over 2019 | **Grid Dependency Risk**: High variability reveals exposure to grid fuel mix fluctuations. Justifies on-site renewable generation or battery storage to reduce carbon volatility. |
| `pjm_carbon_intensity_monthly.png` | Monthly average carbon intensity | **Procurement Timing**: Identifies cleanest months for carbon-intensive batch jobs. May inform contract negotiations with renewable energy providers for seasonal PPAs. |
| `pjm_carbon_intensity_heatmap.png` | Hour × Month carbon intensity heatmap | **Carbon-Aware Scheduling**: The "holy grail" for workload optimization. Green cells are optimal for ML training jobs; red cells should be avoided for discretionary computing. Can reduce carbon by 20-40% without hardware changes. |
| `grid_demand_heatmap.png` | Grid demand patterns | **Grid Stability Contribution**: Shows when datacenter load adds stress to already-strained grid. Demand response participation during red zones earns utility incentives and regulatory goodwill. |
| `dc_vs_grid_peaks.png` | Datacenter peaks vs grid peaks | **Peak Shaving Opportunity**: Coincident peaks indicate maximum grid stress contribution. Non-coincident peaks suggest the datacenter may help stabilize grid during off-peak hours—potential ancillary services revenue. |
| `grid_stress_comparison.png` | Grid stress score analysis | **Infrastructure Investment Trigger**: Sustained high stress scores justify on-site generation, battery storage, or utility interconnection upgrades. Quantifies ROI for grid resilience investments. |

### Sensitivity Analysis (9 figures)

| Figure | Description | **Business Implications** |
|--------|-------------|---------------------------|
| `tornado_total_energy_gwh.png` | Tornado: impact on total energy | **OPEX Driver Identification**: Longest bars reveal which variables most impact annual energy costs. Guides where to focus efficiency investments for maximum ROI (e.g., PUE improvement vs. workload optimization). |
| `tornado_peak_demand_mw.png` | Tornado: impact on peak demand | **Capacity Planning**: Peak demand determines utility demand charges (often 30-50% of electricity bill). Variables with largest impact guide strategies to reduce contractual capacity requirements. |
| `tornado_co2_metric_tons.png` | Tornado: impact on CO₂ emissions | **Carbon Strategy Prioritization**: Identifies highest-leverage interventions for decarbonization. If carbon intensity dominates, focus on renewables; if utilization dominates, focus on efficiency. |
| `tornado_diagram_total_power_mw.png` | Tornado: total power sensitivity | **Real-Time Operations**: Informs which control variables to adjust during power emergencies. Variables with largest sensitivity become primary levers for demand response events. |
| `tornado_diagram_carbon_tons.png` | Tornado: carbon tons sensitivity | **Carbon Budget Management**: When approaching carbon caps, this chart identifies fastest paths to reduction. May trigger temporary workload migration to cleaner regions. |
| `tornado_grid_stress_score.png` | Tornado: grid stress sensitivity | **Grid Reliability Contribution**: Shows how datacenter operational choices affect regional grid stability. High sensitivity to datacenter load may trigger utility collaboration discussions. |
| `spider_plot_scenarios.png` | Spider/radar plot comparing scenarios | **Scenario Comparison**: Visualizes trade-offs across multiple KPIs simultaneously. "Aggressive Growth" may excel on capacity utilization but fail on carbon—enables balanced decision-making. |
| `waterfall_carbon_decomposition.png` | Waterfall: carbon impact decomposition | **Attribution Analysis**: Shows how each factor contributes to final carbon footprint step-by-step. Essential for carbon accounting audits and identifying the "long pole" in emissions reduction. |
| `sensitivity_dashboard.png` | 9-panel comprehensive sensitivity dashboard | **Executive Decision Support**: Single-page summary for C-suite briefings. Combines all sensitivity insights into actionable format for board presentations and strategic planning sessions. |

### Scenario Matrices (3 figures)

| Figure | Description | **Business Implications** |
|--------|-------------|---------------------------|
| `scenario_matrix_pue_utilization_carbon_tons.png` | PUE × Utilization → Carbon | **Efficiency vs. Utilization Trade-off**: Reveals the carbon cost of pushing utilization higher with suboptimal PUE. May show diminishing returns—e.g., above 80% utilization, carbon penalties outweigh revenue gains. Informs "sweet spot" operating targets. |
| `scenario_matrix_temperature_f_pue_total_power_mw.png` | Temp × PUE → Power | **Climate Adaptation Planning**: Quantifies power cost of operating in hotter conditions with varying efficiency. Projects infrastructure requirements under climate change scenarios (e.g., +3°C by 2050). |
| `scenario_matrix_it_capacity_mw_annual_growth_rate_carbon_tons.png` | Capacity × Growth → Carbon | **Growth Strategy Carbon Impact**: Maps the carbon implications of expansion plans. Identifies growth rates that exceed carbon budget constraints, potentially triggering phased expansion or renewable procurement requirements. |

### Contour Plots (2 figures)

| Figure | Description | **Business Implications** |
|--------|-------------|---------------------------|
| `contour_temperature_f_pue_total_power_mw.png` | 3D sensitivity surface: Temp × PUE | **Operational Boundary Mapping**: Contour lines define iso-power regions. Reveals "danger zones" where small changes cause large power jumps. Informs automated control system setpoints and alarm thresholds for facility management. |
| `contour_utilization_renewable_pct_carbon_tons.png` | 3D surface: Utilization × Renewables | **Renewable Investment ROI**: Shows how renewable procurement interacts with utilization to reduce carbon. Steeper gradients in renewable direction indicate high ROI for renewable investments; flat regions suggest diminishing returns. |

### Future Projections (2 figures)

| Figure | Description | **Business Implications** |
|--------|-------------|---------------------------|
| `long_term_scenario_projections.png` | 2025-2035 projections (3 growth scenarios) | **Strategic Planning Horizon**: Shows divergence between conservative, moderate, and aggressive growth paths. The gap between scenarios by 2035 quantifies the stakes of growth decisions made today. Essential for 10-year capital planning and land acquisition. |
| `monthly_forecast_heatmap.png` | Monthly power forecast heatmap | **Operational Budgeting**: Year-over-year monthly comparison enables seasonal budgeting. Darker cells (high demand) inform summer staffing, maintenance blackout periods, and utility contract renegotiation timing. |

### Forecasting (2 figures)

| Figure | Description | **Business Implications** |
|--------|-------------|---------------------------|
| `forecast_monthly.png` | Monthly forecast with confidence intervals | **Uncertainty Quantification**: Confidence bands directly inform risk buffers in power contracts. Wider bands in summer months may justify higher capacity reserves or demand response agreements during uncertain periods. |
| `forecast_annual_summary.png` | Annual energy consumption summary | **Financial Planning**: Annual totals drive electricity budget, carbon credit requirements, and depreciation schedules. Year-over-year comparison validates growth assumptions in business plans and investor communications. |

### Monte Carlo & Risk Validation (8 figures)

| Figure | Description | **Business Implications** |
|--------|-------------|---------------------------|
| `comprehensive_validation_dashboard.png` | **12-panel master dashboard** combining all validation methods | **Board-Level Risk Summary**: The definitive risk visualization for executive leadership. Combines statistical rigor with visual accessibility. Should be included in quarterly risk committee reports and insurance renewal documentation. |
| `evt_diagnostics.png` | Extreme Value Theory: GPD fit, return levels, VaR/CTE | **Insurance & Catastrophe Planning**: VaR 99% defines the "100-year event" magnitude. CTE (Expected Shortfall) quantifies average loss given extreme event—critical for self-insurance reserves and captive insurance capitalization. |
| `copula_analysis.png` | Copula scatter plots & tail dependence | **Compound Risk Quantification**: Reveals whether extreme heat AND dirty grid occur together more often than random chance. Upper tail dependence (λ_U = 0.174) indicates 17.4% probability that when one variable is extreme, the other is too—the "double whammy" scenario. |
| `copula_detailed_analysis.png` | Detailed copula density & tail functions | **Regulatory Risk Modeling**: Required for sophisticated enterprise risk management (ERM) frameworks. Demonstrates to regulators and insurers that correlated risks are properly modeled, not naively assumed independent. |
| `sobol_sensitivity.png` | Sobol indices: variance decomposition | **Global Sensitivity Attribution**: Unlike local sensitivity, Sobol captures interactions. If carbon intensity's total-order index (54.8%) far exceeds first-order, it indicates strong interactions with other variables—guiding integrated solutions over isolated fixes. |
| `walkforward_backtest.png` | Walk-forward out-of-sample validation | **Model Trustworthiness**: The gold standard for time-series model validation. Consistent performance across expanding windows proves the model generalizes and isn't overfit to training data. Required for any model used in financial decisions. |
| `tail_risk_deep_dive.png` | Exceedance probability, Q-Q plots, compound events | **Stress Event Anatomy**: Dissects what makes extreme events extreme. Q-Q plots validate distributional assumptions; exceedance curves inform graduated emergency response protocols (e.g., 95th vs. 99th percentile actions). |
| `reverse_stress_test.png` | Fragility surface & minimum failure conditions | **Failure Mode Discovery**: Rather than asking "what happens if X breaks," reverse stress testing asks "what breaks the system?" Identifies the minimum conditions (e.g., 98°F + 95% utilization + 0.9 tCO₂/MWh) that cause catastrophic outcomes—informing operational red lines. |

### Actuarial Risk (3 figures)

| Figure | Description | **Business Implications** |
|--------|-------------|---------------------------|
| `actuarial_risk_heatmap.png` | Risk exposure heatmap | **Insurance Premium Negotiation**: Visualizes when/where risk concentrates. Insurers use similar analyses—presenting this proactively demonstrates risk awareness and may secure better terms. High-risk cells justify targeted risk mitigation investments. |
| `actuarial_risk_monthly_breakdown.png` | Monthly risk breakdown | **Seasonal Risk Budgeting**: Allocates annual risk budget across months. July-August concentration (~40% of annual risk) justifies enhanced monitoring, additional staff, and pre-positioned spare equipment during summer months. |
| `actuarial_risk_temperature.png` | Temperature-driven risk analysis | **Climate Risk Quantification**: The "climate change impact chart." Shows how risk escalates with temperature. Extrapolating to projected future temperatures (IPCC scenarios) quantifies climate adaptation investment requirements. |

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
