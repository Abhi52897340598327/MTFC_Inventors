"""
Generate Semi-Synthetic Datacenter Power Data (2015-2024)
==========================================================
Uses strict Physics-based modeling to create a high-quality target variable
for training the forecasting model.

Data Sources & Logic:
---------------------
1. Weather (Input):
   - Uses REAL NOAA daily max/min temperatures for Dulles, VA (2015-2024).
   - Conversions to hourly use a sinusoidal model (min @ sunrise, max @ 16:00).

2. IT Load (Base Demand):
   - Based on standard "Diurnal" Internet Traffic patterns (Meta/Google studies).
   - Peak: ~20:00 (8 PM), Trough: ~04:00 (4 AM).
   - Weekday/Weekend variance included.
   - Scaled to a 100 MW reference facility.

3. Cooling Power (PUE Physics):
   - PUE (Power Usage Effectiveness) is calculated dynamically based on Outdoor Temp.
   - Curve mimics ASHRAE "Free Cooling" behavior:
     - Below 60°F: Low PUE (1.15) -> Economizer mode.
     - Above 60°F: PUE rises quadratically as mechanical chillers engage.
     - Max PUE ~1.6 at 100°F.

Output:
-------
- Saves to: Data_Sources/semisynthetic_datacenter_power_2015_2024.csv
- Columns: datetime, temp_f, it_load_mw, pue, total_power_mw
"""

import pandas as pd
import numpy as np
import os
from datetime import timedelta

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
FACILITY_CAPACITY_MW = 100.0   # Reference facility size
START_YEAR = 2015
END_YEAR = 2024

# Files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "Data_Sources")
WEATHER_FILE = os.path.join(DATA_DIR, "noaa_dulles_daily_2015_2024.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "semisynthetic_datacenter_power_2015_2024.csv")

# ─── PHYSICS FUNCTIONS ────────────────────────────────────────────────────────

def generate_hourly_temps(daily_df):
    """
    Expand daily min/max temps into hourly readings using a sinusoidal profile.
    Min temp usually occurs around sunrise (06:00), Max around 16:00.
    """
    print("  Generating realistic hourly temperatures from daily NOAA data...")
    
    # Create full hourly range
    hourly_dates = pd.date_range(
        start=f"{START_YEAR}-01-01", 
        end=f"{END_YEAR}-12-31 23:00", 
        freq="h"
    )
    
    # Merge daily data onto hourly
    hourly_df = pd.DataFrame({"datetime": hourly_dates})
    hourly_df["date"] = hourly_df["datetime"].dt.date
    
    # Ensure daily_df "date" is strictly date object
    daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date
    
    merged = pd.merge(hourly_df, daily_df[["date", "min_temp_f", "max_temp_f"]], on="date", how="left")
    
    # Forward fill missing weather days (rare)
    merged = merged.ffill().bfill()
    
    # ─── Vectorized Sinusoidal Interpolation ───
    # Normalized hour: 0 at 06:00 (min), 1 at 16:00 (max)
    # We want a curve that oscillates between min and max.
    # Model: T(h) = (Max+Min)/2 + (Max-Min)/2 * cos(pi * (h - 16) / 12) roughly
    
    hour = merged["datetime"].dt.hour.values
    t_min = merged["min_temp_f"].values
    t_max = merged["max_temp_f"].values
    
    # Simple model: Max at 16:00, Min at 04:00 (approx sunrise)
    # We shift the cosine wave to peak at 16:00
    # cos(0) = 1 (peak). We want peak at h=16. -> cos(2pi * (h-16)/24)
    # Scaled to range [-1, 1], then mapped to [min, max]
    
    # Amplitude = (Max - Min) / 2
    # Midpoint = (Max + Min) / 2
    # Curve = Midpoint + Amplitude * cos(...)
    
    amp = (t_max - t_min) / 2
    mid = (t_max + t_min) / 2
    
    # -cos typically peaks at pi. we want peak at 16.
    # shape: low at 4, high at 16.
    # period = 24h
    
    # Using a shifted negative cosine to put trough at 4 and peak at 16
    hourly_temp = mid + amp * np.cos((hour - 16) * 2 * np.pi / 24)
    
    # Add small random noise to look organic (clouds, wind gusts)
    noise = np.random.normal(0, 1.5, size=len(hourly_temp))
    merged["temp_f"] = hourly_temp + noise
    
    return merged[["datetime", "temp_f"]]

def generate_it_load(timestamps):
    """
    Generate IT Load (MW) based on Meta/Google diurnal patterns.
    - Base load: 60% of capacity (idle servers, constant tasks)
    - Diurnal component: +30% (user activity)
    - Noise: +10%
    """
    print("  Synthesizing IT Load (Meta/Google diurnal patterns)...")
    
    hour = timestamps.dt.hour.values
    day_of_week = timestamps.dt.dayofweek.values # 0=Mon, 6=Sun
    
    # 1. Base Load (always on)
    base_load = FACILITY_CAPACITY_MW * 0.50
    
    # 2. Diurnal "Internet Traffic" Pattern
    # Rises from 06:00, peaks around 20:00 (8 PM), drops to low at 04:00.
    # Model: Sinusoidal peak at 20:00
    diurnal_amp = FACILITY_CAPACITY_MW * 0.25
    diurnal_profile = np.cos((hour - 20) * 2 * np.pi / 24) 
    # Shift cosine from [-1, 1] to [0, 1] roughly for additive load? 
    # Actually, let's just let it swing +/- around a center.
    
    # 3. Weekend Effect (Traffic slightly lower or different, lets assume -5% peak)
    is_weekend = (day_of_week >= 5)
    weekend_factor = np.ones(len(timestamps))
    weekend_factor[is_weekend] = 0.95
    
    # 4. Long-term Growth Trend (Datacenters fill up over time!)
    # Assume 2015 is 60% full, 2024 is 95% full?
    # Or just Assume constant "Capacity" refers to installed hardware?
    # Let's assume a strictly "Stable" facility to isolate weather effects for the model first. 
    # Or better: Add a slight organic growth trend to be realistic.
    # 2015 -> 2024 is 9 years. 2% growth/year.
    years_from_start = (timestamps - timestamps.min()).dt.total_seconds() / (365.25*24*3600)
    growth_trend = 1.0 + (years_from_start * 0.03) # 3% annual growth
    
    # Combine
    it_load = (base_load + (diurnal_profile * diurnal_amp)) * weekend_factor * growth_trend
    
    # Add Random Noise (Burst jobs, batch processing)
    noise = np.random.normal(0, FACILITY_CAPACITY_MW * 0.02, size=len(it_load))
    it_load += noise
    
    # Clip to physical limits (cannot exceed capacity or be negative)
    it_load = np.clip(it_load, 0, FACILITY_CAPACITY_MW)
    
    return it_load

def calculate_pue(temp_f):
    """
    Calculate PUE based on ASHRAE physics.
    - Free Cooling (Economizer): Active when Temp < Threshold (e.g., 60-65F). Low PUE.
    - Mechanical Cooling: Ramps up quadratically as Temp rises.
    """
    # Parameters
    t_threshold = 60.0  # °F where free cooling maxes out
    pue_min = 1.15      # Efficient base state (fans only)
    pue_max_design = 1.6 # At very high temps
    t_max_design = 100.0
    
    # Vectorized calculation
    pue = np.ones_like(temp_f) * pue_min
    
    # For temps above threshold, curve up
    # PUE = Min + k * (T - Threshold)^2
    # Calc k to hit Max at T_max
    k = (pue_max_design - pue_min) / ((t_max_design - t_threshold)**2)
    
    mask_cooling = temp_f > t_threshold
    pue[mask_cooling] = pue_min + k * (temp_f[mask_cooling] - t_threshold)**2
    
    # Add random operational inefficiency noise (filters clogging, etc)
    noise = np.random.normal(0, 0.01, size=len(pue))
    pue += noise
    
    return np.clip(pue, 1.05, 2.0)

# ─── MAIN EXECUTION ───────────────────────────────────────────────────────────

def main():
    print(f"Generating Semi-Synthetic Datacenter Power Data ({START_YEAR}-{END_YEAR})...")
    
    # 1. Load Weather
    if not os.path.exists(WEATHER_FILE):
        print(f"ERROR: Weather file not found at {WEATHER_FILE}")
        print("Please run 'python download_noaa_weather.py --weather' first.")
        return

    daily_weather = pd.read_csv(WEATHER_FILE)
    
    # 2. Interp to Hourly
    df = generate_hourly_temps(daily_weather)
    print(f"  Generated {len(df):,} hourly records.")
    
    # 3. Generate Loads
    df["it_load_mw"] = generate_it_load(df["datetime"])
    df["pue"] = calculate_pue(df["temp_f"].values)
    
    # 4. Total Power = IT * PUE
    df["total_power_mw"] = df["it_load_mw"] * df["pue"]
    
    # 5. Save
    print(f"  Saving to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    
    # 6. Stats
    print("\nData Generation Complete.")
    print("-" * 40)
    print(f"Date Range: {df.datetime.min()} to {df.datetime.max()}")
    print("-" * 40)
    print(f"{'Metric':<15} {'Min':<10} {'Mean':<10} {'Max':<10}")
    print(f"{'Temp (F)':<15} {df.temp_f.min():<10.1f} {df.temp_f.mean():<10.1f} {df.temp_f.max():<10.1f}")
    print(f"{'IT Load (MW)':<15} {df.it_load_mw.min():<10.1f} {df.it_load_mw.mean():<10.1f} {df.it_load_mw.max():<10.1f}")
    print(f"{'PUE':<15} {df.pue.min():<10.2f} {df.pue.mean():<10.2f} {df.pue.max():<10.2f}")
    print(f"{'Total (MW)':<15} {df.total_power_mw.min():<10.1f} {df.total_power_mw.mean():<10.1f} {df.total_power_mw.max():<10.1f}")
    print("-" * 40)

if __name__ == "__main__":
    main()
