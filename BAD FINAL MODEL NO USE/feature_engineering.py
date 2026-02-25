"""
MTFC Virginia Datacenter Energy Forecasting — Feature Engineering
==================================================================
Create all derived features per the modeling plan §4.2:
  ‣ Temporal features (hour, day, month, season, weekend, business-hour)
  ‣ Lag features  (t-1, t-24, t-168)
  ‣ Rolling statistics (24 h mean / std)
  ‣ Cyclical encodings (sin / cos for hour and month)
  ‣ Interaction terms (temperature × hour)
"""

import numpy as np
import pandas as pd
import config as cfg
from utils import log


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based features from the timestamp column."""
    ts = df["timestamp"]

    # Basic temporal
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek          # 0=Mon
    df["day_of_year"] = ts.dt.dayofyear
    df["month"] = ts.dt.month
    df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_business_hour"] = ((df["hour"] >= 8) & (df["hour"] <= 18)).astype(int)

    # Season: 0=Winter, 1=Spring, 2=Summer, 3=Fall
    df["season"] = df["month"].map(
        {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
         6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    )

    log.info("Temporal features added")
    return df


def add_lag_features(df: pd.DataFrame, target: str = cfg.TARGET_COL) -> pd.DataFrame:
    """Add lagged versions of the target and temperature."""
    for lag in [1, 24, 168]:
        df[f"power_lag{lag}"] = df[target].shift(lag)

    if "temperature_f" in df.columns:
        for lag in [1, 24]:
            df[f"temp_lag{lag}"] = df["temperature_f"].shift(lag)

    log.info("Lag features added")
    return df


def add_rolling_features(df: pd.DataFrame, target: str = cfg.TARGET_COL) -> pd.DataFrame:
    """Add 24-hour rolling mean and std for power and temperature."""
    df["power_rolling_mean_24"] = df[target].rolling(24, min_periods=1).mean()
    df["power_rolling_std_24"] = df[target].rolling(24, min_periods=1).std().fillna(0)

    if "temperature_f" in df.columns:
        df["temp_rolling_mean_24"] = df["temperature_f"].rolling(24, min_periods=1).mean()

    log.info("Rolling features added")
    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode hour and month as sin/cos for neural-network friendliness."""
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    log.info("Cyclical features added")
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create physically meaningful interaction terms."""
    if "temperature_f" in df.columns:
        df["temp_x_hour"] = df["temperature_f"] * df["hour"]
        df["weekend_x_hour"] = df["is_weekend"] * df["hour"]

        # Cooling degree (temp above threshold) - ASHRAE TC 9.9
        df["cooling_degree"] = np.maximum(
            0, df["temperature_f"] - cfg.COOLING_THRESHOLD_F
        )
    
    # Note: temp_factor is created in data_loader.calculate_pue_from_temperature()
    # and should already exist. If not, create it here.
    if "temp_factor" not in df.columns and "temperature_f" in df.columns:
        optimal_temp = 65.0  # ASHRAE TC 9.9 Class A1
        threshold = 85.0     # ASHRAE TC 9.9 Class A1
        df["temp_factor"] = np.clip(
            (df["temperature_f"] - optimal_temp) / (threshold - optimal_temp), 0, 1
        )

    log.info("Interaction features added")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full feature-engineering pipeline and return the enriched
    DataFrame with NaN rows (from lagging) dropped.
    """
    df = add_temporal_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_cyclical_features(df)
    df = add_interaction_features(df)

    n_before = len(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    log.info(f"Feature engineering complete — dropped {n_before - len(df)} rows "
             f"(lag warm-up). Final shape: {df.shape}")
    return df
