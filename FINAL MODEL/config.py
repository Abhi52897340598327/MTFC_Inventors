"""
MTFC Virginia Datacenter Energy Forecasting — Configuration
============================================================
Central configuration for all paths, hyperparameters, physical constants,
scenario definitions, and emissions factors used throughout the pipeline.
"""

import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "Data_Sources")  # Use raw Data_Sources folder
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

for _d in [OUTPUT_DIR, FIGURE_DIR, MODEL_DIR, RESULTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Dataset file names (REAL DATA SOURCES ONLY) ────────────────────────────
# All datasets must be from verified real-world sources:
# - Google Cluster: Public Google cluster trace data
# - EIA: Energy Information Administration official data
# - PJM: PJM Interconnection grid data
# - NOAA: National Oceanic and Atmospheric Administration weather data
DATASETS = {
    "temperature":      "ashburn_va_temperature_2019.csv",          # NOAA weather data
    "carbon_intensity": "pjm_carbon_intensity_2019_hourly.csv",     # EIA-930 grid data
    "carbon_daily":     "pjm_carbon_intensity_2019_daily.csv",      # EIA-930 aggregated
    "generation_mix":   "pjm_generation_mix_2019_weekly.csv",       # EIA-930 fuel mix
    "google_cluster":   "google_cluster_utilization_2019.csv",      # Google public trace
    "eia_generation":   "EIA923_Schedules_2_3_4_5_M_11_2025_21JAN2026.xlsx",  # EIA-923
    "datacenter_specs": "datacenter_constants.json",                # Physical constants
}

def dataset_path(key: str) -> str:
    """Return absolute path for a dataset key."""
    return os.path.join(DATA_DIR, DATASETS[key])


# ── Facility Capacity ───────────────────────────────────────────────────────
# Physical datacenter specifications based on datacenter_constants.json
# All power calculations use physics-based models, NOT synthetic data
FACILITY_CAPACITY_MW = 100.0        # base facility rated IT capacity (MW)

# ── Physical Constants ──────────────────────────────────────────────────────
PUE_LEVELS = {"excellent": 1.1, "good": 1.3, "average": 1.5, "poor": 2.0}
COOLING_THRESHOLD_F = 65.0          # °F where cooling ramps up
COOLING_CF_MIN = 0.15               # minimum cooling fraction of IT load
COOLING_CF_MAX = 0.50               # maximum cooling fraction of IT load
COOLING_ALPHA = 0.01                # cooling increase per °F above threshold
COOLING_TEMP_MAX_F = 100.0          # °F where cooling saturates

# ── Fuel-Specific Emissions Factors (kg CO₂ / MWh) ─────────────────────────
EMISSIONS_FACTORS = {
    "coal":         1_000,
    "natural_gas":  450,
    "petroleum":    850,
    "nuclear":      0,
    "solar":        0,
    "wind":         0,
    "hydro":        0,
    "other":        450,   # conservative fallback
}

# ── Data Split Ratios ───────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ── SARIMAX Hyperparameters ─────────────────────────────────────────────────
SARIMAX_ORDER = (1, 1, 1)
SARIMAX_SEASONAL_ORDER = (1, 1, 1, 24)
SARIMAX_EXOG_COLS = ["temperature_f", "hour_of_day", "is_weekend"]

# ── LSTM Hyperparameters ────────────────────────────────────────────────────
LSTM_LOOKBACK = 24             # 1 day of hourly data (CPU-practical)
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
LSTM_DENSE_UNITS = 16
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 64
LSTM_PATIENCE = 10
LSTM_LR = 0.001

# ── XGBoost Hyperparameters ─────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "objective":        "reg:squarederror",
    "random_state":     42,
    "verbosity":        0,
}
XGB_EARLY_STOPPING = 50

# ── OPTIMIZED FEATURE SELECTION (Based on Feature Importance Analysis) ─────
# Priority ranking from analysis: lag1 > rolling_std > rolling_mean > temp > cooling_degree
PRIORITY_FEATURES = {
    "critical": [  # Must always include - highest predictive power
        "power_lag1",           # Rank 1: 1.000 importance
        "power_rolling_std_24", # Rank 2: 0.406 importance
        "power_rolling_mean_24",# Rank 3: 0.303 importance
    ],
    "high": [  # Strong predictors - include for accuracy
        "temperature_f",        # ~0.25 importance - drives cooling
        "cooling_degree",       # ~0.20 importance - cooling threshold exceeded
        "hour_sin",             # ~0.15 importance - diurnal patterns
        "hour_cos",             # paired with hour_sin
        "power_lag24",          # ~0.12 importance - daily pattern
        "temp_lag1",            # recent temp autocorrelation
    ],
    "medium": [  # Useful - include if not forecasting far ahead
        "is_weekend",           # ~0.08 importance
        "month_sin",            # seasonal patterns
        "month_cos",
        "power_lag168",         # weekly pattern
        "temp_rolling_mean_24", # 24h temp trend
        "is_business_hour",     # workload proxy
        "dow_sin",              # day-of-week pattern
        "dow_cos",
    ],
    "low": [  # Can drop for simpler models / forecasting
        "temp_x_hour",          # interaction term
        "weekend_x_hour",       # interaction term
        "season",               # captured by month encoding
        "day_of_year",          # high cardinality
        "week_of_year",         # redundant with month
    ],
}

# Feature sets for different use cases
FEATURE_SET_FULL = (
    PRIORITY_FEATURES["critical"] + 
    PRIORITY_FEATURES["high"] + 
    PRIORITY_FEATURES["medium"]
)

FEATURE_SET_FORECAST_SAFE = [
    # Features that can be computed without knowing future target values
    "temperature_f", "cooling_degree", "hour_sin", "hour_cos",
    "month_sin", "month_cos", "is_weekend", "is_business_hour",
    "dow_sin", "dow_cos", "temp_x_hour", "season", "hour", "month", "day_of_week",
]

FEATURE_SET_REALTIME = PRIORITY_FEATURES["critical"] + PRIORITY_FEATURES["high"]
# Use REALTIME features when lag values are available (e.g., nowcasting, next-hour)

# ── Sensitivity Analysis Scenarios ──────────────────────────────────────────
BASELINE_SCENARIO = {
    "capacity_mw":      1_000,
    "utilization":      0.70,
    "pue":              1.5,       # matches actual data PUE
    "temp_delta_f":     0.0,
    "grid_year":        2024,
    "load_shift_pct":   0.0,
}

SENSITIVITY_FACTORS = {
    "capacity_mw":      [500, 1_000, 2_000, 4_000],
    "utilization":      [0.50, 0.70, 0.90],
    "pue":              [1.1, 1.3, 1.5],
    "temp_delta_f":     [0, 2, 5, 7, 10],
    "grid_year":        [2024, 2027, 2030],
    "load_shift_pct":   [0.0, 0.20, 0.40],
}

# Grid stress score weights
GSS_WEIGHTS = {
    "cpf":              0.40,
    "peak_contrib":     0.30,
    "variability":      0.20,
    "load_factor":      0.10,
}

# Grid decarbonisation rates (annual EI decline)
GRID_EI_ANNUAL_DECLINE = 0.025   # 2.5% per year

# Forecasting horizon
FORECAST_YEARS = list(range(2024, 2031))

# ── Annual Growth / Climate Parameters (for multi-year forecasting) ────────
ANNUAL_IT_GROWTH_RATE = 0.08         # 8% compound annual growth in IT load
ANNUAL_PUE_IMPROVEMENT = 0.01       # PUE improves ~0.01 per year
ANNUAL_TEMP_WARMING_F = 0.05        # °F warming per year (climate trend)

# Random seed
RANDOM_SEED = 42

# Target column
TARGET_COL = "total_power_with_pue"
