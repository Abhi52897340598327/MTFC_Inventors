"""
IMPROVED SYNTHETIC DATACENTER POWER CONSUMPTION GENERATOR
==========================================================

This version generates a FULL YEAR of realistic hourly power data (8,760 hours)
instead of just 1 data point!

Based on:
1. Seasonal temperature patterns (Virginia climate)
2. Realistic datacenter utilization patterns
3. Industry-standard power models
4. Day/night and weekday/weekend variations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data_Sources')
OUTPUT_FILE = os.path.join(DATA_DIR, 'synthetic_datacenter_power_full_year_2019.csv')

# Power model parameters (based on industry research)
BASE_POWER = 0.40  # 40% minimum power even at idle
IT_POWER_FACTOR = 0.35  # 35% of capacity for compute workload
COOLING_BASELINE = 0.20  # 20% baseline cooling
COOLING_TEMP_FACTOR = 0.0015  # Increase per degree F above threshold
COOLING_UTIL_FACTOR = 0.10  # Additional cooling from high utilization
TEMP_THRESHOLD = 65  # °F - below this, minimal additional cooling needed
PUE = 1.5  # Power Usage Effectiveness for 2019 datacenter

def generate_temperature_profile(hours):
    """
    Generate realistic Virginia temperature profile for full year.
    Based on Ashburn/Dulles climate normals.
    """
    temps = []
    
    for hour in range(hours):
        # Day of year (0-364)
        day = hour // 24
        hour_of_day = hour % 24
        
        # Seasonal component (sinusoidal)
        avg_yearly_temp = 58  # Annual average for Ashburn, VA
        seasonal_amplitude = 30  # Summer-winter swing
        seasonal_temp = avg_yearly_temp + seasonal_amplitude * np.sin(2 * np.pi * (day - 80) / 365)
        
        # Daily component (cooler at night, warmer in afternoon)
        daily_amplitude = 12  # Day-night swing
        daily_temp = daily_amplitude * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        
        # Random variation (weather)
        random_variation = np.random.normal(0, 4)
        
        temp = seasonal_temp + daily_temp + random_variation
        temps.append(max(10, min(temp, 105)))  # Realistic bounds
    
    return np.array(temps)

def generate_cpu_utilization(hours):
    """
    Generate realistic datacenter CPU utilization patterns.
    AI datacenters run high utilization (60-85%) with some variation.
    """
    utils = []
    
    for hour in range(hours):
        day_of_week = (hour // 24) % 7
        hour_of_day = hour % 24
        
        # Base utilization: AI datacenters run hot
        base_util = 0.70  # 70% average for AI workloads
        
        # Weekly pattern: slightly lower on weekends (some batch jobs pause)
        if day_of_week >= 5:  # Weekend
            weekly_factor = 0.90  # 10% lower
        else:
            weekly_factor = 1.0
        
        # Daily pattern: slight dip in early morning (fewer inference requests)
        if 2 <= hour_of_day < 6:
            daily_factor = 0.85  # 15% lower during night
        else:
            daily_factor = 1.0
        
        # Random workload variation
        random_variation = np.random.normal(0, 0.05)
        
        util = base_util * weekly_factor * daily_factor + random_variation
        utils.append(np.clip(util, 0.20, 0.95))  # Realistic bounds
    
    return np.array(utils)

def calculate_cooling_load(temp_f, cpu_util):
    """
    Calculate cooling load based on temperature and utilization.
    """
    # Temperature effect
    temp_above_threshold = max(0, temp_f - TEMP_THRESHOLD)
    temp_cooling = COOLING_TEMP_FACTOR * temp_above_threshold
    
    # Utilization effect
    util_cooling = COOLING_UTIL_FACTOR * cpu_util
    
    total_cooling = COOLING_BASELINE + temp_cooling + util_cooling
    
    return min(total_cooling, 0.35)  # Cap at 35%

def generate_full_year_data():
    """
    Generate complete year of hourly power consumption data.
    """
    print("="*70)
    print("GENERATING FULL YEAR SYNTHETIC DATACENTER POWER DATA")
    print("="*70)
    
    # Generate 8,760 hours (full year)
    hours = 8760
    
    print(f"\nGenerating {hours:,} hours of data (full year 2019)...")
    
    # Create timestamp index
    start_date = datetime(2019, 1, 1, 0, 0, 0)
    timestamps = [start_date + timedelta(hours=h) for h in range(hours)]
    
    print("  Generating temperature profile...")
    temperatures_f = generate_temperature_profile(hours)
    
    print("  Generating CPU utilization patterns...")
    cpu_utilization = generate_cpu_utilization(hours)
    
    print("  Calculating power components...")
    
    # Calculate power components for each hour
    base_load = np.full(hours, BASE_POWER)
    it_load = cpu_utilization * IT_POWER_FACTOR
    cooling_load = np.array([
        calculate_cooling_load(temp, util) 
        for temp, util in zip(temperatures_f, cpu_utilization)
    ])
    
    # Small random daily variation
    daily_variation = np.random.normal(0, 0.02, hours)
    
    # Measurement noise
    measurement_noise = np.random.normal(0, 0.01, hours)
    
    # Total power utilization
    power_util = base_load + it_load + cooling_load + daily_variation + measurement_noise
    power_util = np.clip(power_util, 0.30, 1.0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'power_utilization': power_util,
        'cpu_utilization': cpu_utilization,
        'temperature_f': temperatures_f,
        'temperature_c': (temperatures_f - 32) * 5/9,
        'base_load_component': base_load,
        'it_load_component': it_load,
        'cooling_load_component': cooling_load,
        'pue': PUE,
        'total_power_with_pue': power_util * PUE,
        'hour_of_day': [ts.hour for ts in timestamps],
        'day_of_week': [ts.weekday() for ts in timestamps],
        'day_of_year': [ts.timetuple().tm_yday for ts in timestamps],
        'month': [ts.month for ts in timestamps],
        'is_weekend': [1 if ts.weekday() >= 5 else 0 for ts in timestamps]
    })
    
    print(f"✅ Generated {len(df):,} hourly records")
    
    return df

def add_statistics(df):
    """
    Calculate and display summary statistics.
    """
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    stats = {
        'Power Utilization': {
            'Mean': df['power_utilization'].mean(),
            'Min': df['power_utilization'].min(),
            'Max': df['power_utilization'].max(),
            'Std': df['power_utilization'].std()
        },
        'CPU Utilization': {
            'Mean': df['cpu_utilization'].mean(),
            'Min': df['cpu_utilization'].min(),
            'Max': df['cpu_utilization'].max(),
            'Std': df['cpu_utilization'].std()
        },
        'Temperature (°F)': {
            'Mean': df['temperature_f'].mean(),
            'Min': df['temperature_f'].min(),
            'Max': df['temperature_f'].max(),
            'Std': df['temperature_f'].std()
        },
        'Cooling Load': {
            'Mean': df['cooling_load_component'].mean(),
            'Min': df['cooling_load_component'].min(),
            'Max': df['cooling_load_component'].max(),
            'Std': df['cooling_load_component'].std()
        }
    }
    
    for metric, values in stats.items():
        print(f"\n{metric}:")
        for stat_name, value in values.items():
            if 'Utilization' in metric or 'Load' in metric:
                print(f"  {stat_name}: {value:.2%}")
            else:
                print(f"  {stat_name}: {value:.2f}")
    
    # Seasonal analysis
    print("\n" + "-"*70)
    print("SEASONAL PATTERNS")
    print("-"*70)
    
    seasonal_stats = df.groupby('month').agg({
        'power_utilization': 'mean',
        'temperature_f': 'mean',
        'cooling_load_component': 'mean'
    })
    
    print(seasonal_stats.to_string())
    
    # Weekly patterns
    print("\n" + "-"*70)
    print("WEEKLY PATTERNS")
    print("-"*70)
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_stats = df.groupby('day_of_week').agg({
        'power_utilization': 'mean',
        'cpu_utilization': 'mean'
    })
    weekly_stats.index = day_names
    
    print(weekly_stats.to_string())
    
    return df

def main():
    """
    Main execution function.
    """
    print("\n" + "="*80)
    print("SYNTHETIC DATACENTER POWER DATA GENERATOR v2.0")
    print("="*80)
    print("\nVERSION 2 IMPROVEMENTS:")
    print("  ✅ Generates FULL YEAR of data (8,760 hours)")
    print("  ✅ Realistic seasonal temperature patterns")
    print("  ✅ AI datacenter utilization patterns (60-85%)")
    print("  ✅ Daily and weekly variations")
    print("  ✅ Temperature-dependent cooling")
    print("  ✅ Ready for time series forecasting!")
    print("\n" + "="*80)
    
    # Generate data
    df = generate_full_year_data()
    
    # Add statistics
    df = add_statistics(df)
    
    # Save to file
    print("\n" + "="*70)
    print("SAVING DATA")
    print("="*70)
    print(f"Output file: {OUTPUT_FILE}")
    
    df.to_csv(OUTPUT_FILE, index=False)
    
    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"✅ File saved successfully ({file_size_mb:.2f} MB)")
    
    # Show sample
    print("\n" + "="*70)
    print("SAMPLE DATA (first 24 hours)")
    print("="*70)
    print(df[['timestamp', 'power_utilization', 'cpu_utilization', 'temperature_f', 
              'cooling_load_component']].head(24).to_string(index=False))
    
    print("\n" + "="*70)
    print("✅ FULL YEAR SYNTHETIC DATA GENERATION COMPLETE!")
    print("="*70)
    print(f"\nTotal records: {len(df):,}")
    print(f"Time span: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"File size: {file_size_mb:.2f} MB")
    print("\n" + "-"*70)
    print("WHAT YOU CAN DO NOW:")
    print("-"*70)
    print("""
1. TRAIN FORECASTING MODELS:
   - SARIMAX (seasonal patterns)
   - LSTM (temporal dependencies)
   - XGBoost (feature relationships)
   
2. RUN SENSITIVITY ANALYSIS:
   Test scenarios:
   - Temperature +5°F (climate change)
   - Utilization 80% → 90% (more AI workloads)
   - PUE 1.5 → 1.2 (efficiency improvements)
   - Load shifting (move to off-peak hours)

3. CALCULATE METRICS:
   - Annual energy: Sum(power) × 8760 hours
   - Peak demand: Max(power)
   - Carbon emissions: Power × Grid_Carbon_Intensity
   - Grid stress: Power during peak hours

4. MERGE WITH REAL DATA:
   - PJM grid carbon intensity (you have)
   - Actual temperature (you have)
   - This synthetic power fills the missing piece!

5. VISUALIZE:
   - Time series plots
   - Seasonal heatmaps
   - Load duration curves
   - Correlation matrices
    """)
    
    print("="*80)
    print("Ready for MTFC forecasting analysis! 🚀📊")
    print("="*80)

if __name__ == "__main__":
    main()
