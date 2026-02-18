"""
MTFC Virginia Datacenter Energy Forecasting — Scenario Forecasting
==================================================================
Implements "ML-Driven Simulation" instead of recursive forecasting.

Logic:
1. Generate "Future Scenarios" (DataFrames with future dates/temps).
2. Use trained XGBoost model to predict "Baseline Power" for these conditions.
   (Model captures: "How much power does the facility use at X temp and Y hour?")
3. Apply "Growth Layer": Scale the baseline prediction by annual growth rates.
   (Simulates adding more servers/capacity over time).

This avoids error accumulation from recursive forecasting and allows
physics-based scenario testing (e.g., "What if 2030 is 2°C hotter?").
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config as cfg
from utils import log, save_csv, save_fig

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

# Scenario Definitions
SCENARIOS = {
    "Conservative (5%)": {"growth_rate": 0.05, "temp_increase_c": 0.5},
    "Baseline (15%)":    {"growth_rate": 0.15, "temp_increase_c": 1.0},
    "Aggressive (25%)":  {"growth_rate": 0.25, "temp_increase_c": 1.5},
}

def generate_future_features(start_year, end_year, temp_increase_c=0.0):
    """
    Create a DataFrame of future features (timestamp, hour, month, temp_f, etc.)
    Logic:
    - Timestamps: Hourly from start to end year.
    - Temperature: Cloned from a representative historical year (2024),
      shifted by `temp_increase_c`.
    - Temporal Features: Derived from timestamp.
    """
    log.info(f"Generating future features {start_year}-{end_year} (Temp +{temp_increase_c}°C)...")
    
    # 1. Create Timestamp Index
    future_dates = pd.date_range(
        start=f"{start_year}-01-01", 
        end=f"{end_year}-12-31 23:00", 
        freq="h"
    )
    df = pd.DataFrame({"timestamp": future_dates})
    
    # 2. Add Temporal Features (XGBoost needs these)
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["year"] = df["timestamp"].dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    # 3. Synthesize Future Temperature
    # Strategy: Use a sinusoidal model similar to physics engine, 
    # centered on historical averages + increase
    # Dulles Avg Jan: 35F, July: 78F.
    # Model: T(m, h) = Avg(m) + Diurnal(h) + Increase
    
    # Approx Monthly Avgs (F) for Dulles
    monthly_avgs = {
        1: 34, 2: 37, 3: 45, 4: 55, 5: 64, 6: 73,
        7: 78, 8: 76, 9: 69, 10: 57, 11: 47, 12: 38
    }
    
    # Vectorized mapping
    base_temp = df["month"].map(monthly_avgs)
    
    # Diurnal swing: +/- 10F (Low at 5am, High at 3pm)
    # Cos measure from 15 (3pm). Pi at 3am.
    diurnal = 10 * np.cos((df["hour"] - 15) * 2 * np.pi / 24)
    
    # Climate Change Adder (convert C change to F: deltaF = deltaC * 1.8)
    climate_adder = temp_increase_c * 1.8
    
    # Noise (Weather variability)
    np.random.seed(42)
    noise = np.random.normal(0, 5, size=len(df))
    
    df["temperature_f"] = base_temp + diurnal + climate_adder + noise
    
    # 4. Add Interaction Terms (if model expects them)
    df["temp_x_hour"] = df["temperature_f"] * df["hour"]
    
    return df

def forecast_scenarios(model, feature_cols, start_year=2025, end_year=2030):
    """
    Run the forecasting loop for all defined scenarios.
    
    Args:
        model: Trained XGBoost model (must not rely on lag features!)
        feature_cols: List of columns the model expects.
    """
    all_results = []
    
    for name, params in SCENARIOS.items():
        log.info(f"Forecasting Scenario: {name}")
        rate = params["growth_rate"]
        temp_inc = params["temp_increase_c"]
        
        # 1. Generate Future Feature Set
        future_df = generate_future_features(start_year, end_year, temp_inc)
        
        # 2. Predict Baseline Power (Machine Learning Step)
        # "What would the facility consume given these weather conditions?"
        # (Assuming 2024 capacity)
        X_future = future_df[feature_cols]
        # Ensure column order matches training
        # Handle scaling? If model was trained on scaled data, we need scaler.
        # Ideally passed in. For now assuming unscaled or handle outside.
        
        # NOTE: Model expect input format. 
        # If we trained on scaled data, we MUST scale here.
        # We will assume this function returns the raw DF and forecasting is done in main?
        # No, better to do it here. We need the Scalar passed in.
        pass # Placeholder logic
        
        # Let's return the Future DF for the main loop to process, 
        # or refactor this to accept scaler.
        
    return all_results

def run_forecast(model, scaler, feature_cols):
    """
    Orchestrate the full forecast.
    """
    forecast_results = []
    
    for name, params in SCENARIOS.items():
        rate = params["growth_rate"]
        temp_inc = params["temp_increase_c"]
        
        # 1. Create Future Data
        future = generate_future_features(2025, 2030, temp_inc)
        
        # 2. Scale Features
        # Ensure we only use the columns the scaler was fitted on (forecast safe cols)
        X_future = scaler.transform(future[feature_cols])
        
        # 3. Predict Baseline (2024 Capacity)
        base_pred_mw = model.predict(X_future)
        
        # 4. Apply Growth
        # Year offset: 2025 is year 1
        years_from_base = future["year"] - 2024
        growth_factor = (1 + rate) ** years_from_base
        
        final_pred_mw = base_pred_mw * growth_factor
        
        # Store
        future["total_power_mw"] = final_pred_mw
        future["scenario"] = name
        
        forecast_results.append(future)
        
    final_df = pd.concat(forecast_results, ignore_index=True)
    return final_df

def plot_forecast_comparison(df):
    """Plot the comparison of scenarios."""
    # Aggregating by Year for clean plot
    annual = df.groupby(["year", "scenario"])["total_power_mw"].sum().reset_index()
    # Convert MW -> GWh (sum of hourly MW = MWh, /1000 = GWh)
    annual["total_energy_gwh"] = annual["total_power_mw"] / 1000
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=annual, x="year", y="total_energy_gwh", hue="scenario", marker="o", ax=ax)
    
    ax.set_title("Projected Data Center Energy Consumption (2025-2030)", fontsize=14)
    ax.set_ylabel("Annual Energy (GWh)")
    ax.set_xlabel("Year")
    ax.grid(True, alpha=0.3)
    
    save_fig(fig, "forecast_scenario_comparison")
    
    # Peak Demand Plot
    peak = df.groupby(["year", "scenario"])["total_power_mw"].max().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=peak, x="year", y="total_power_mw", hue="scenario", marker="s", linestyle="--", ax=ax)
    ax.set_title("Projected Peak Power Demand (Grid Stress)", fontsize=14)
    ax.set_ylabel("Peak MW")
    ax.axhline(200, color='r', linestyle=':', label='Substation Limit (200 MW)')
    ax.legend()
    
    save_fig(fig, "forecast_peak_stress")

