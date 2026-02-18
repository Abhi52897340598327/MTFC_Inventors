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
DATA_DIR = os.path.join(PROJECT_DIR, "Data_Sources", "cleaned")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

for _d in [OUTPUT_DIR, FIGURE_DIR, MODEL_DIR, RESULTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Dataset file names ──────────────────────────────────────────────────────
DATASETS = {
    "power":            "semisynthetic_datacenter_power_2015_2024.csv",
    "temperature":      "ashburn_va_temperature_2019_cleaned.csv",
    "pjm_demand":       "pjm_hourly_demand_2019_2024_cleaned.csv",
    "co2_emissions":    "virginia_co2_emissions_2015_2023_cleaned.csv",
    "elec_consumption": "virginia_electricity_consumption_2015_2024_cleaned.csv",
    "gen_by_fuel":      "virginia_generation_by_fuel_2015_2024_cleaned.csv",
    "renewable_gen":    "virginia_renewable_generation_2015_2024_cleaned.csv",
    "carbon_intensity": "pjm_grid_carbon_intensity_2019_full_cleaned.csv",
    "google_cluster":   "google_cluster_utilization_2019_cleaned.csv",
}

def dataset_path(key: str) -> str:
    """Return absolute path for a dataset key."""
    return os.path.join(DATA_DIR, DATASETS[key])


# ── Facility Capacity ───────────────────────────────────────────────────────
# The raw synthetic data is in per-unit (p.u.) where 1.0 = base facility.
# Multiply by FACILITY_CAPACITY_MW to convert to real megawatts.
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
