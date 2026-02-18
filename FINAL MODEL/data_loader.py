"""
MTFC Virginia Datacenter Energy Forecasting — Data Loader
==========================================================
Load, parse, merge, and validate all cleaned datasets.
Returns structured DataFrames ready for feature engineering.
"""

import pandas as pd
import numpy as np
import config as cfg
from utils import log


# ── Individual loaders ──────────────────────────────────────────────────────

def load_power_data() -> pd.DataFrame:
    """Load datacenter hourly power data.
    
    Supports two formats:
    1. New Physics-Based (2015-2024): Already in MW. Cols: total_power_mw, it_load_mw.
    2. Legacy (2019 only): Per-unit (p.u.). Needs scaling by FACILITY_CAPACITY_MW.
    """
    df = pd.read_csv(cfg.dataset_path("power"))
    
    # Handle timestamp column name differences
    if "datetime" in df.columns:
        df.rename(columns={"datetime": "timestamp"}, inplace=True)
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Check if this is the new MW-based file
    if "total_power_mw" in df.columns:
        log.info("Detected new physics-based dataset (already in MW).")
        # Rename to pipeline standard names
        df.rename(columns={
            "total_power_mw": cfg.TARGET_COL,  # total_power_with_pue
            "it_load_mw": "it_load_component",
            "temp_f": "temperature_f"
        }, inplace=True)
        # No scaling needed
    else:
        # Legacy: Scale per-unit → MW
        log.info("Detected legacy per-unit dataset. Scaling to MW.")
        power_cols = [c for c in df.columns
                      if any(k in c for k in ("load", "power", "base_load"))]
        for c in power_cols:
            if c in df.columns and c not in ("power_utilization", "pue"):
                df[c] = df[c] * cfg.FACILITY_CAPACITY_MW

    log.info(f"Power data loaded: {df.shape}")
    if cfg.TARGET_COL in df.columns:
        log.info(f"  target range: {df[cfg.TARGET_COL].min():.1f} – "
                 f"{df[cfg.TARGET_COL].max():.1f} MW")
    return df



def load_temperature_data() -> pd.DataFrame:
    """Load Ashburn, VA hourly temperature (8,760 rows)."""
    df = pd.read_csv(cfg.dataset_path("temperature"), parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    log.info(f"Temperature data loaded: {df.shape}")
    return df


def load_pjm_demand() -> pd.DataFrame:
    """
    Load PJM hourly grid demand 2019-2024 (52,469 rows).
    Supports EIA format (datetime_utc) or legacy format (period).
    Typical PJM demand: 80-150 GW with diurnal and seasonal patterns.
    """
    df = pd.read_csv(cfg.dataset_path("pjm_demand"))
    
    # Handle API column diffs
    ts_col = "period"
    if "datetime_utc" in df.columns:
        ts_col = "datetime_utc"
    
    if ts_col not in df.columns:
        log.warning(f"PJM demand file missing timestamp column! Cols: {df.columns}")
        # Fallback? or Error
    
    df.rename(columns={ts_col: "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
    
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ──────────────────────────────────────────────────────────────────────────
    # CHECK: Does the file have REAL data now?
    # ──────────────────────────────────────────────────────────────────────────
    if "degree_mwh" in df.columns: # Sometimes EIA uses this, likely 'demand_mwh'
        df.rename(columns={"demand_mwh": "grid_demand_mw"}, inplace=True)
    elif "value" in df.columns:
        df.rename(columns={"value": "grid_demand_mw"}, inplace=True)
    elif "demand_mwh" in df.columns:  # The script I wrote used this
        df.rename(columns={"demand_mwh": "grid_demand_mw"}, inplace=True)
        
    if "grid_demand_mw" in df.columns:
        log.info("Loaded REAL PJM demand data (from EIA).")
        # Ensure numeric
        df["grid_demand_mw"] = pd.to_numeric(df["grid_demand_mw"], errors="coerce")
        # Fill missing with linear interp
        df["grid_demand_mw"] = df["grid_demand_mw"].interpolate(method="linear")
        
        # We don't need synthetic generation if we have real data!
        log.info(f"PJM demand data loaded (REAL): {df.shape}, "
                 f"range: {df['grid_demand_mw'].min():.0f}-{df['grid_demand_mw'].max():.0f} MW")
        return df[['timestamp', 'grid_demand_mw']] # Return clean subset

    # Else: Fallback to SYNTHETIC generation if no real column found
    log.warning("⚠ PJM demand data metadata only — generating SYNTHETIC demand.")
    # Generate synthetic grid demand (PJM typical range ~80-150 GW = 80000-150000 MW)
    np.random.seed(cfg.RANDOM_SEED)
    hours = df["timestamp"].dt.hour.values
    months = df["timestamp"].dt.month.values
    dow = df["timestamp"].dt.dayofweek.values

    # Base load + seasonal + diurnal + weekend effects
    base = 95000  # MW baseline
    seasonal = 15000 * np.sin(2 * np.pi * (months - 1) / 12)      # peaks summer/winter
    seasonal = np.abs(seasonal) + 5000 * (months >= 6).astype(float)  # summer higher
    diurnal = 20000 * np.sin(np.pi * (hours - 5) / 14)              # peaks 12-14h
    diurnal = np.maximum(diurnal, -10000)
    weekend = -8000 * (dow >= 5).astype(float)
    noise = np.random.normal(0, 3000, len(df))

    df["grid_demand_mw"] = np.maximum(50000, base + seasonal + diurnal + weekend + noise)

    log.warning("⚠ PJM demand data is SYNTHETIC — source CSV lacks numerical "
                 "demand values. Grid stress results are illustrative only.")
    log.info(f"PJM demand data loaded (synthetic): {df.shape}, "
             f"range: {df['grid_demand_mw'].min():.0f}-{df['grid_demand_mw'].max():.0f} MW")
    return df


def load_co2_emissions() -> pd.DataFrame:
    """Load Virginia CO₂ emissions 2015-2023 (annual, ~32 rows)."""
    df = pd.read_csv(cfg.dataset_path("co2_emissions"))
    # Try to extract year from period
    df["period"] = pd.to_datetime(df["period"], errors="coerce")
    if df["period"].notna().any():
        df["year"] = df["period"].dt.year
    log.info(f"CO₂ emissions data loaded: {df.shape}")
    return df


def load_electricity_consumption() -> pd.DataFrame:
    """Load Virginia electricity consumption 2015-2024 (monthly, ~720 rows)."""
    df = pd.read_csv(cfg.dataset_path("elec_consumption"))
    df["period"] = pd.to_datetime(df["period"], errors="coerce")
    log.info(f"Electricity consumption data loaded: {df.shape}")
    return df


def load_generation_by_fuel() -> pd.DataFrame:
    """Load Virginia generation by fuel type 2015-2024 (monthly, ~5,000 rows)."""
    df = pd.read_csv(cfg.dataset_path("gen_by_fuel"))
    df["period"] = pd.to_datetime(df["period"], errors="coerce")
    log.info(f"Generation by fuel data loaded: {df.shape}")
    return df


def load_renewable_generation() -> pd.DataFrame:
    """Load Virginia renewable generation 2015-2024 (monthly, ~1,996 rows)."""
    df = pd.read_csv(cfg.dataset_path("renewable_gen"))
    df["period"] = pd.to_datetime(df["period"], errors="coerce")
    log.info(f"Renewable generation data loaded: {df.shape}")
    return df


def load_carbon_intensity() -> pd.DataFrame:
    """Load PJM hourly grid carbon intensity 2019 (8,758 rows)."""
    df = pd.read_csv(cfg.dataset_path("carbon_intensity"))
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    log.info(f"Carbon intensity data loaded: {df.shape}")
    return df


# ── Merge & Validate ────────────────────────────────────────────────────────

def merge_hourly_datasets() -> pd.DataFrame:
    """
    Merge power + temperature + carbon intensity into a single
    hourly DataFrame aligned on timestamp (2019, ~8,760 rows).
    """
    df = load_power_data()
    
    # If using the new physics-based data, we already have high-quality temp data
    if "temperature_f" in df.columns:
        log.info("Using embedded temperature data from power dataset (2015-2024 coverage).")
        # Don't merge legacy temp file to avoid overwriting or key issues
    else:
        # Legacy path: Merge partial temp data
        temp = load_temperature_data()
        temp_cols = [c for c in temp.columns if c != "timestamp"]
        existing_temp_cols = [c for c in temp_cols if c in df.columns]
        if existing_temp_cols:
            df.drop(columns=existing_temp_cols, inplace=True)
        df = pd.merge(df, temp, on="timestamp", how="left")

    # Merge carbon intensity
    if "pjm_grid_carbon_intensity_2019_full_cleaned.csv" in cfg.DATASETS["carbon_intensity"]:
         # Note: The recalculated file is 2019-2024, so it should align well.
         pass # handled by merge below

    carbon = load_carbon_intensity()
    carbon_cols_to_merge = ["timestamp"] + [
        c for c in carbon.columns if c != "timestamp"
    ]
    
    # Ensure timestamps match types
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    carbon["timestamp"] = pd.to_datetime(carbon["timestamp"])
    
    df = pd.merge(df, carbon[carbon_cols_to_merge], on="timestamp", how="left")

    log.info(f"Merged hourly dataset: {df.shape}")
    return df


def validate_dataframe(df: pd.DataFrame, name: str = "DataFrame"):
    """Run sanity checks on a DataFrame."""
    issues = []
    n_null = df.isnull().sum().sum()
    if n_null > 0:
        issues.append(f"{n_null} null values")
    n_inf = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    if n_inf > 0:
        issues.append(f"{n_inf} infinite values")
    if "timestamp" in df.columns:
        ts = df["timestamp"]
        if not ts.is_monotonic_increasing:
            issues.append("timestamps not monotonically increasing")
        if ts.duplicated().any():
            issues.append(f"{ts.duplicated().sum()} duplicate timestamps")
    if issues:
        log.warning(f"[{name}] Validation issues: {', '.join(issues)}")
    else:
        log.info(f"[{name}] Validation passed ✓")
    return issues


# ── PJM demand helpers (for grid stress) ────────────────────────────────────

def get_pjm_demand_2019() -> pd.DataFrame:
    """Return just the 2019 PJM demand for alignment with hourly power data."""
    pjm = load_pjm_demand()
    pjm_2019 = pjm[pjm["timestamp"].dt.year == 2019].copy()
    pjm_2019.reset_index(drop=True, inplace=True)
    log.info(f"PJM demand 2019 subset: {pjm_2019.shape}")
    return pjm_2019


# ── Convenience: Load everything ────────────────────────────────────────────

def load_all():
    """
    Return a dict of all datasets for downstream use.
    Keys: 'hourly', 'pjm_demand', 'co2', 'elec_consumption',
          'gen_by_fuel', 'renewable', 'carbon_intensity'
    """
    hourly = merge_hourly_datasets()
    validate_dataframe(hourly, "hourly")

    return {
        "hourly":           hourly,
        "pjm_demand":       load_pjm_demand(),
        "co2":              load_co2_emissions(),
        "elec_consumption": load_electricity_consumption(),
        "gen_by_fuel":      load_generation_by_fuel(),
        "renewable":        load_renewable_generation(),
        "carbon_intensity": load_carbon_intensity(),
    }


if __name__ == "__main__":
    data = load_all()
    for k, v in data.items():
        print(f"{k:25s} → {v.shape}")
