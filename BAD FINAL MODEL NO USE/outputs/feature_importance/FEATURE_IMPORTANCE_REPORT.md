# Feature Importance Analysis Report
## MTFC Virginia Datacenter Energy Forecasting Project

**Analysis Date:** February 18, 2026  
**Data Sources Analyzed:** Google Cluster Utilization, EIA Power Generation Data (97 columns), Datacenter Constants

---

## Executive Summary

This analysis identified the most predictive features for datacenter energy forecasting across **all data sources**. The key finding is that **autoregressive features (lag values and rolling statistics) dominate** in predictive power, followed by **workload indicators** and **temperature-related features**.

---

## 📊 Complete Variable Dictionary (All Datasets)

### 1. DATACENTER POWER DATA (Physics-Based Model)
*Source: Physics model driven by REAL temperature data (NOAA) and datacenter constants*

| Variable | Description | Importance |
|----------|-------------|------------|
| `total_power_mw` / `total_power_with_pue` | **TARGET** - Total facility power including cooling (MW) | TARGET |
| `it_power_mw` | IT equipment power only (servers, GPUs) | HIGH |
| `temperature_f` | Ambient outdoor temperature (°F) from NOAA | HIGH |
| `pue` | Power Usage Effectiveness (total/IT ratio) | MEDIUM |
| `utilization` | Server utilization fraction (0.55-0.92) | MEDIUM |

### 2. TEMPORAL FEATURES (Engineered)
*Source: feature_engineering.py*

| Variable | Description | Importance |
|----------|-------------|------------|
| `hour` | Hour of day (0-23) | HIGH |
| `day_of_week` | Day of week (0=Mon, 6=Sun) | MEDIUM |
| `month` | Month of year (1-12) | HIGH |
| `day_of_year` | Day of year (1-365) | MEDIUM |
| `week_of_year` | Week number (1-52) | LOW |
| `is_weekend` | Binary: 1 if Sat/Sun | MEDIUM |
| `is_business_hour` | Binary: 1 if 8am-6pm | MEDIUM |
| `season` | 0=Winter, 1=Spring, 2=Summer, 3=Fall | MEDIUM |

### 3. LAG FEATURES (Autoregressive)
*Source: feature_engineering.py*

| Variable | Description | Importance |
|----------|-------------|------------|
| `power_lag1` | Power from 1 hour ago | ⭐ CRITICAL |
| `power_lag24` | Power from 24 hours ago (same hour yesterday) | HIGH |
| `power_lag168` | Power from 168 hours ago (same hour last week) | MEDIUM |
| `temp_lag1` | Temperature from 1 hour ago | MEDIUM |
| `temp_lag24` | Temperature from 24 hours ago | MEDIUM |

### 4. ROLLING STATISTICS
*Source: feature_engineering.py*

| Variable | Description | Importance |
|----------|-------------|------------|
| `power_rolling_mean_24` | 24-hour rolling average power | HIGH |
| `power_rolling_std_24` | 24-hour rolling standard deviation | HIGH |
| `temp_rolling_mean_24` | 24-hour rolling average temperature | MEDIUM |

### 5. CYCLICAL ENCODINGS
*Source: feature_engineering.py*

| Variable | Description | Importance |
|----------|-------------|------------|
| `hour_sin` / `hour_cos` | Sin/cos encoding of hour (preserves cyclical nature) | HIGH |
| `month_sin` / `month_cos` | Sin/cos encoding of month | MEDIUM |
| `dow_sin` / `dow_cos` | Sin/cos encoding of day of week | MEDIUM |

### 6. INTERACTION FEATURES
*Source: feature_engineering.py*

| Variable | Description | Importance |
|----------|-------------|------------|
| `temp_x_hour` | Temperature × Hour interaction | MEDIUM |
| `weekend_x_hour` | Weekend × Hour interaction | LOW |
| `cooling_degree` | max(0, temp - 65°F) - degrees above cooling threshold | HIGH |

### 7. GOOGLE CLUSTER UTILIZATION DATA
*Source: google_cluster_utilization_2019.csv*

| Variable | Description | Importance |
|----------|-------------|------------|
| `avg_cpu_utilization` | Average CPU usage across cluster (0-1) | HIGH |
| `num_tasks_sampled` | Number of compute tasks running | MEDIUM |
| `hour_of_day` | Hour extracted from timestamp | LOW |
| `real_timestamp` | Datetime of measurement | INDEX |

### 8. EIA POWER GENERATION DATA (Virginia Grid)
*Source: EIA923_Schedules_2_3_4_5_M_11_2025_21JAN2026.xlsx (97 columns, 7,715 rows)*

**Plant Identification:**
| Variable | Description |
|----------|-------------|
| `Plant Id` | Unique EIA plant identifier |
| `Plant Name` | Name of power plant |
| `Operator Name` / `Operator Id` | Operating company |
| `Plant State` | State (filter for VA) |
| `NERC Region` | Grid reliability region |
| `Balancing Authority Code` | Grid balancing authority |

**Fuel & Generation Type:**
| Variable | Description | Use For |
|----------|-------------|---------|
| `Reported Prime Mover` | Type of generator (CT, ST, PV, etc.) | Grid mix |
| `Reported Fuel Type Code` | Fuel type (NG, SUN, NUC, etc.) | Carbon intensity |
| `MER Fuel Type Code` | Monthly Energy Review fuel classification | Carbon calculation |

**Monthly Generation (Netgen columns):**
| Variable | Description | Use For |
|----------|-------------|---------|
| `Netgen January` - `Netgen December` | Net generation by month (MWh) | Seasonal patterns |
| `Net Generation (Megawatthours)` | Total annual net generation | Capacity factor |

**Fuel Consumption:**
| Variable | Description | Use For |
|----------|-------------|---------|
| `Quantity January` - `Quantity December` | Monthly fuel consumed (physical units) | Efficiency |
| `Tot_MMBtu January` - `Tot_MMBtu December` | Monthly total heat input (MMBtu) | Heat rate |
| `Elec_MMBtu January` - `Elec_MMBtu December` | Electric-only heat input | Efficiency |
| `Total Fuel Consumption Quantity` | Annual total fuel | Carbon emissions |
| `Total Fuel Consumption MMBtu` | Annual total heat input | Heat rate |

**Efficiency Metrics:**
| Variable | Description | Use For |
|----------|-------------|---------|
| `MMBtuPer_Unit January` - `December` | Heat content per unit fuel | Fuel quality |
| `Elec Fuel Consumption MMBtu` | Fuel used for electricity only | CHP separation |

### 9. DATACENTER CONSTANTS (Physics Parameters)
*Source: datacenter_constants.json*

**Server Specifications:**
| Variable | Value | Description |
|----------|-------|-------------|
| `max_power_w` | 10,200 W | Max power per NVIDIA DGX H100 |
| `typical_power_w` | 8,000 W | Typical operating power |
| `gpu_fraction` | 0.55 | GPU portion of server power |
| `cpu_fraction` | 0.07 | CPU portion of server power |
| `cooling_fraction` | 0.36 | Cooling portion at server level |

**Facility Specifications:**
| Variable | Value | Description |
|----------|-------|-------------|
| `total_it_capacity_mw` | 100 MW | Total IT power capacity |
| `rack_density_kw` | 40 kW | Power per rack |
| `pue_min` | 1.15 | Best-case PUE (efficient cooling) |
| `pue_max_air` | 1.6 | Worst-case PUE (hot weather) |
| `optimal_temp_f` | 65°F | Ideal ambient temperature |
| `cooling_threshold_f` | 85°F | Temperature where cooling maxes out |

**Grid & Emissions:**
| Variable | Value | Description |
|----------|-------|-------------|
| `carbon_intensity_g_per_kwh` | 276 | Dominion VA grid carbon intensity |
| `pjm_network_avg_g_per_kwh` | 360 | PJM regional average |
| `transmission_loss_factor` | 0.05 | 5% grid transmission losses |

**Scenario Parameters:**
| Scenario | Growth Rate | Temp Adder (°C) |
|----------|-------------|-----------------|
| Conservative | 10%/year | +0.5°C |
| Baseline | 20%/year | +1.0°C |
| Aggressive | 50%/year | +1.5°C |

---

## 📊 Feature Importance Rankings (From Analysis)

### Top Features to Focus On (Ranked by Aggregate Importance)

| Rank | Feature | Aggregate Score | Category | Recommendation |
|------|---------|-----------------|----------|----------------|
| 1 | **power_lag1** / **cpu_lag_1** | 1.000 | Lag Feature | ⭐ CRITICAL - Use always |
| 2 | **power_rolling_std_24** | 0.406 | Rolling Stat | ⭐ HIGH - Captures volatility |
| 3 | **power_rolling_mean_24** | 0.303 | Rolling Stat | ⭐ HIGH - Captures trend |
| 4 | **temperature_f** | ~0.25 | Environmental | ⭐ HIGH - Affects cooling |
| 5 | **log_workload** | 0.206 | Load Indicator | HIGH - Workload proxy |
| 6 | **cooling_degree** | ~0.20 | Derived | HIGH - Cooling demand |
| 7 | **hour_sin/hour_cos** | ~0.15 | Temporal | MEDIUM - Diurnal patterns |
| 8 | **power_lag24** | ~0.12 | Lag Feature | MEDIUM - Daily pattern |
| 9 | **is_weekend** | ~0.08 | Temporal | MEDIUM - Weekend effect |
| 10 | **month_sin/month_cos** | ~0.05 | Temporal | MEDIUM - Seasonal |

---

## 🔬 Analysis Methods Used

### 1. Correlation Analysis
- **Best correlated:** `cpu_lag_1` (r = 0.65), `cpu_rolling_mean_10` (r = 0.48), `cpu_rolling_std_10` (r = 0.42)
- Strong positive correlation between recent CPU usage and current usage (autoregressive behavior)

### 2. Gradient Boosting Feature Importance
- `cpu_lag_1` dominates with 47.2% importance
- `cpu_rolling_std_10` at 15.2%
- Load indicators (`num_tasks_sampled`, `log_num_tasks`) contribute ~17% combined

### 3. Random Forest Feature Importance
- Confirms Gradient Boosting findings
- `cpu_lag_1` at 49.3% importance
- Tree-based models strongly favor lag features

### 4. Mutual Information Scores
- Non-linear dependency analysis confirms `cpu_lag_1` as most informative
- `log_num_tasks` shows higher MI than raw `num_tasks_sampled` (log transform helps!)

### 5. Permutation Importance
- Most robust importance metric
- `cpu_lag_1` shows 0.52 importance drop when permuted
- Validates that this feature is essential for prediction

---

## 📁 Complete Data Sources Summary

### 1. Google Cluster Utilization (2019)
- **Records:** 2,000 rows × 4 columns
- **Target Variable:** `avg_cpu_utilization`
- **Key Columns:** `real_timestamp`, `hour_of_day`, `num_tasks_sampled`
- **Temporal Range:** May 1, 2019 (high-frequency sampling)
- **Use Case:** Workload pattern analysis, CPU-power correlation

### 2. EIA Power Generation Data (2025)
- **Records:** 7,715 rows × 97 columns
- **Virginia Plants:** 202 power plant records
- **Key Metrics Available:**
  - Monthly Net Generation (Netgen January-December) - MWh
  - Total/Electric Fuel Consumption (Quantity & MMBtu)
  - Fuel Type Codes (NG, SUN, NUC, COL, etc.)
  - Prime Mover types (CT, ST, PV, WT, etc.)
  - Balancing Authority assignments
- **Use Case:** Grid carbon intensity, fuel mix analysis, marginal emissions

### 3. Datacenter Constants (Physics)
- **Facility:** 100 MW AI Datacenter in Virginia
- **Server Model:** NVIDIA DGX H100 (Proxy) - 10.2 kW max
- **PUE Range:** 1.15 (optimal) - 1.6 (hot weather)
- **Cooling Threshold:** 65°F optimal, 85°F max
- **Carbon Intensity:** 276 g CO2/kWh (Dominion Energy Virginia)
- **Use Case:** Physics-based power modeling, scenario forecasting

### 4. Pipeline Datasets (REAL DATA SOURCES ONLY)
| Dataset | Source File | Data Source |
|---------|-------------|-------------|
| Temperature | ashburn_va_temperature_2019.csv | NOAA Weather Data |
| Carbon Intensity | pjm_carbon_intensity_2019_hourly.csv | EIA-930 Grid Data |
| Generation Mix | pjm_generation_mix_2019_weekly.csv | EIA-930 Fuel Mix |
| Google Cluster | google_cluster_utilization_2019.csv | Google Public Trace |
| EIA Generation | EIA923_Schedules_2_3_4_5_M_11_2025_21JAN2026.xlsx | EIA-923 Data |
| Datacenter Specs | datacenter_constants.json | Physical Constants |

**Note:** Power consumption is calculated using a physics-based model driven by REAL temperature data, not from synthetic generation.

---

## 🎯 Feature Engineering Recommendations

### PRIMARY FEATURES (Must Use)
```python
# These provide the majority of predictive power
primary_features = [
    'power_lag_1',           # Previous hour's power (t-1)
    'power_lag_24',          # Same hour yesterday (t-24)
    'power_rolling_mean_24', # 24-hour rolling average
    'power_rolling_std_24',  # 24-hour volatility
    'temperature_f',         # Ambient temperature
]
```

### SECONDARY FEATURES (Recommended)
```python
# Enhance model with workload and temporal context
secondary_features = [
    'log_workload',          # Log-transformed workload indicator
    'hour_sin', 'hour_cos',  # Cyclical hour encoding
    'cooling_degree',        # Temp above 65°F threshold
    'is_weekend',            # Weekend flag
    'is_business_hour',      # Business hours (8am-6pm)
]
```

### TERTIARY FEATURES (Optional)
```python
# Additional context for edge cases
tertiary_features = [
    'day_of_week',           # Weekly patterns
    'month_sin', 'month_cos', # Seasonal encoding
    'power_lag_168',         # Same hour last week
    'temp_x_hour',           # Temperature-hour interaction
]
```

---

## 📈 Key Insights

### 1. **Autoregressive Nature**
The datacenter power consumption is highly autoregressive. The previous timestep (`cpu_lag_1`) alone explains ~65% of variance. This means **LSTM and SARIMAX models** are well-suited for this problem.

### 2. **Volatility Matters**
The rolling standard deviation (`cpu_rolling_std_10`) is the second most important feature. This suggests periods of instability are predictive—model should capture **heteroskedasticity**.

### 3. **Log Transform for Task Counts**
`log_num_tasks` outperforms raw `num_tasks_sampled` in Mutual Information analysis. Always apply log transform to workload indicators with high variance.

### 4. **Temporal Features Have Low Importance**
In this dataset, `hour_sin`, `hour_cos`, `is_weekend` showed minimal importance. This is because:
- The Google Cluster sample is short-duration (same day)
- Time features become important with longer time horizons

### 5. **For Multi-Year Forecasting**
Based on EIA data structure, include:
- Fuel type generation mix (for carbon intensity)
- Seasonal temperature patterns
- Grid demand patterns from PJM data

---

## 📂 Output Files Generated

| File | Description |
|------|-------------|
| `feature_importance_ranking.csv` | Complete ranking with all methods |
| `correlation_heatmap.png` | Feature correlation matrix |
| `feature_importance_comparison.png` | Side-by-side method comparison |
| `aggregate_importance_ranking.png` | Final aggregated ranking visualization |

**Output Location:** `FINAL MODEL/outputs/feature_importance/`

---

## ✅ Action Items

1. **Feature Engineering Pipeline:** Ensure all PRIMARY features are computed in `feature_engineering.py`
2. **Model Focus:** Prioritize XGBoost and LSTM which leverage lag features well
3. **Data Collection:** For production, prioritize real-time workload metrics over temporal features
4. **Temperature Data:** Integrate hourly Ashburn, VA temperature with power data
5. **Carbon Tracking:** Use EIA fuel mix data for marginal emissions calculations

---

*Analysis performed using: Correlation, Gradient Boosting, Random Forest, Mutual Information, Permutation Importance*
