"""
SYNTHETIC DATACENTER POWER CONSUMPTION GENERATOR
================================================

Since Google's BigQuery public dataset access is currently restricted,
this script generates realistic synthetic power consumption data based on:
1. CPU utilization patterns (from existing google_cluster_utilization_2019.csv)
2. Temperature effects on cooling load
3. Industry-standard datacenter power models

POWER MODEL:
Power = Base_Load + (CPU_Util × IT_Power_Factor) + Cooling(Temp, CPU_Util) + Variation

Based on research:
- Typical datacenter: 40% base load, 35% compute, 25% cooling
- PUE (Power Usage Effectiveness): 1.4-1.6 for 2019 facilities
- Cooling highly correlated with outdoor temperature
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data_Sources')
OUTPUT_FILE = os.path.join(DATA_DIR, 'synthetic_datacenter_power_2019.csv')

# Power model parameters (based on industry research)
BASE_POWER = 0.40  # 40% minimum power even at idle
IT_POWER_FACTOR = 0.35  # 35% of capacity for compute workload
COOLING_BASELINE = 0.20  # 20% baseline cooling
COOLING_TEMP_FACTOR = 0.0015  # Increase per degree F above threshold
COOLING_UTIL_FACTOR = 0.10  # Additional cooling from high utilization
TEMP_THRESHOLD = 65  # °F - below this, minimal additional cooling needed
PUE = 1.5  # Power Usage Effectiveness for 2019 datacenter

# Variation parameters
DAILY_VARIATION = 0.03  # ±3% daily load variation
NOISE_LEVEL = 0.02  # ±2% measurement noise

def load_utilization_data():
    """Load CPU utilization data."""
    util_file = os.path.join(DATA_DIR, 'google_cluster_utilization_2019.csv')
    print(f"Loading utilization data from: {util_file}")
    
    df = pd.read_csv(util_file)
    print(f"  Loaded {len(df)} utilization records")
    print(f"  Columns: {list(df.columns)}")
    
    return df

def load_temperature_data():
    """Load temperature data - try multiple sources."""
    
    # Try NOAA GSOD data first (just downloaded)
    temp_files = [
        'noaa_gsod_dulles_2019.csv',
        'ashburn_va_temperature_2019.csv',
        'epa_temperature_loudoun_2019.csv'
    ]
    
    for temp_file in temp_files:
        filepath = os.path.join(DATA_DIR, temp_file)
        if os.path.exists(filepath) and os.path.getsize(filepath) > 100:
            print(f"Loading temperature data from: {temp_file}")
            try:
                # Try different encodings
                for encoding in ['utf-16', 'utf-8-sig', 'utf-8', 'latin-1', 'cp1252']:
                    try:
                        df = pd.read_csv(filepath, encoding=encoding)
                        if len(df) > 0:
                            print(f"  Loaded {len(df)} temperature records (encoding: {encoding})")
                            return df, temp_file
                    except (UnicodeDecodeError, pd.errors.EmptyDataError):
                        continue
            except Exception as e:
                print(f"  ⚠️  Failed to load {temp_file}: {e}")
                continue
    
    print("  ⚠️  No temperature data found, will generate synthetic temperatures")
    return None, None

def fahrenheit_to_celsius(temp_f):
    """Convert Fahrenheit to Celsius."""
    return (temp_f - 32) * 5/9

def celsius_to_fahrenheit(temp_c):
    """Convert Celsius to Fahrenheit."""
    return (temp_c * 9/5) + 32

def calculate_cooling_load(temp_f, cpu_util):
    """
    Calculate cooling load based on temperature and utilization.
    
    Cooling increases with:
    - Higher outdoor temperature (less "free cooling")
    - Higher CPU utilization (more heat generation)
    """
    # Temperature effect
    temp_above_threshold = max(0, temp_f - TEMP_THRESHOLD)
    temp_cooling = COOLING_TEMP_FACTOR * temp_above_threshold
    
    # Utilization effect (high compute = more heat)
    util_cooling = COOLING_UTIL_FACTOR * cpu_util
    
    total_cooling = COOLING_BASELINE + temp_cooling + util_cooling
    
    return min(total_cooling, 0.35)  # Cap at 35% of total power

def generate_power_consumption(util_df, temp_df, temp_source):
    """
    Generate synthetic power consumption data.
    
    Args:
        util_df: CPU utilization DataFrame
        temp_df: Temperature DataFrame (or None)
        temp_source: Name of temperature data source
    
    Returns:
        DataFrame with power consumption data
    """
    print("\nGenerating synthetic power consumption data...")
    
    # Parse utilization timestamps
    if 'time' in util_df.columns:
        util_df['timestamp'] = pd.to_datetime(util_df['time'], unit='s')
    elif 'timestamp' in util_df.columns:
        util_df['timestamp'] = pd.to_datetime(util_df['timestamp'])
    elif 'real_timestamp' in util_df.columns:
        util_df['timestamp'] = pd.to_datetime(util_df['real_timestamp'])
    
    # Parse temperature data if available
    if temp_df is not None:
        if 'noaa_gsod' in temp_source:
            # NOAA GSOD format: year, mo, da, temp (in F)
            # Rename columns to standard names for datetime conversion
            temp_df['date'] = pd.to_datetime(
                temp_df[['year', 'mo', 'da']].rename(columns={'mo': 'month', 'da': 'day'})
            )
            temp_df['temp_f'] = temp_df['temp']
        elif 'epa' in temp_source:
            # EPA format: date_local, time_local, sample_measurement (in F)
            temp_df['timestamp'] = pd.to_datetime(temp_df['date_local'] + ' ' + temp_df['time_local'])
            temp_df['temp_f'] = temp_df['sample_measurement']
        else:
            # Existing format: might be hourly with various formats
            if 'TAVG' in temp_df.columns:
                temp_df['temp_c'] = temp_df['TAVG']
                temp_df['temp_f'] = celsius_to_fahrenheit(temp_df['temp_c'])
    
    # Resample utilization to hourly if needed
    util_df = util_df.set_index('timestamp')
    # Only keep numeric columns for resampling
    numeric_cols = util_df.select_dtypes(include=[np.number]).columns
    hourly_util = util_df[numeric_cols].resample('h').mean().reset_index()
    
    print(f"  Resampled to {len(hourly_util)} hourly records")
    
    # Generate temperature if not available
    if temp_df is None or len(temp_df) == 0:
        print("  Generating synthetic temperature data...")
        # Realistic Virginia temperature pattern for 2019
        hourly_util['day_of_year'] = hourly_util['timestamp'].dt.dayofyear
        hourly_util['hour'] = hourly_util['timestamp'].dt.hour
        
        # Seasonal temperature (sinusoidal)
        avg_temp = 58  # Annual average for Ashburn, VA in F
        seasonal_amplitude = 25  # Summer-winter variation
        daily_amplitude = 15  # Day-night variation
        
        hourly_util['temp_f'] = (
            avg_temp +
            seasonal_amplitude * np.sin(2 * np.pi * (hourly_util['day_of_year'] - 80) / 365) +
            daily_amplitude * np.sin(2 * np.pi * (hourly_util['hour'] - 14) / 24) +
            np.random.normal(0, 3, len(hourly_util))
        )
    else:
        # Merge temperature data
        if 'date' in temp_df.columns:
            # Daily data - merge by date
            # Remove timezone info for consistent merging
            hourly_util['date'] = hourly_util['timestamp'].dt.tz_localize(None).dt.normalize()
            temp_daily = temp_df.groupby('date')['temp_f'].mean().reset_index()
            hourly_util = hourly_util.merge(temp_daily, on='date', how='left')
        else:
            # Hourly data - merge by timestamp
            temp_hourly = temp_df.set_index('timestamp').resample('h').mean().reset_index()
            hourly_util = hourly_util.merge(
                temp_hourly[['timestamp', 'temp_f']],
                on='timestamp',
                how='left'
            )
        
        # Fill missing temperatures with interpolation
        hourly_util['temp_f'] = hourly_util['temp_f'].interpolate(method='linear')
    
    # Calculate power components
    print("  Calculating power components...")
    
    # Get CPU utilization (handle different column names)
    if 'cpu_util' in hourly_util.columns:
        cpu_util = hourly_util['cpu_util'].fillna(0.5)
    elif 'average_usage' in hourly_util.columns:
        cpu_util = hourly_util['average_usage'].fillna(0.5)
    elif 'avg_cpu_utilization' in hourly_util.columns:
        cpu_util = hourly_util['avg_cpu_utilization'].fillna(0.5)
    else:
        # Use first numeric column
        numeric_cols = hourly_util.select_dtypes(include=[np.number]).columns
        cpu_util = hourly_util[numeric_cols[0]].fillna(0.5)
    
    # Normalize to 0-1 if needed
    if cpu_util.max() > 1:
        cpu_util = cpu_util / cpu_util.max()
    
    # Base load (constant)
    base_load = np.full(len(hourly_util), BASE_POWER)
    
    # IT load (proportional to CPU utilization)
    it_load = cpu_util * IT_POWER_FACTOR
    
    # Cooling load (function of temperature and utilization)
    cooling_load = hourly_util.apply(
        lambda row: calculate_cooling_load(
            row['temp_f'], 
            cpu_util[row.name]
        ),
        axis=1
    )
    
    # Daily variation (simulates workload patterns)
    hour_of_day = hourly_util['timestamp'].dt.hour
    daily_pattern = DAILY_VARIATION * np.sin(2 * np.pi * (hour_of_day - 10) / 24)
    
    # Measurement noise
    noise = np.random.normal(0, NOISE_LEVEL, len(hourly_util))
    
    # Total power utilization (normalized 0-1)
    power_util = base_load + it_load + cooling_load + daily_pattern + noise
    power_util = np.clip(power_util, 0.3, 1.0)  # Realistic bounds
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        'timestamp': hourly_util['timestamp'],
        'power_utilization': power_util,
        'cpu_utilization': cpu_util,
        'temperature_f': hourly_util['temp_f'],
        'temperature_c': fahrenheit_to_celsius(hourly_util['temp_f']),
        'base_load_component': base_load,
        'it_load_component': it_load,
        'cooling_load_component': cooling_load,
        'pue': PUE,
        'total_power_with_pue': power_util * PUE
    })
    
    print(f"  Generated {len(output_df)} hourly power records")
    
    return output_df

def add_metadata(df):
    """Add metadata and summary statistics."""
    print("\nCalculating summary statistics...")
    
    stats = {
        'avg_power_util': df['power_utilization'].mean(),
        'max_power_util': df['power_utilization'].max(),
        'min_power_util': df['power_utilization'].min(),
        'avg_cpu_util': df['cpu_utilization'].mean(),
        'avg_temp_f': df['temperature_f'].mean(),
        'avg_pue': df['pue'].mean()
    }
    
    print(f"\n  Average Power Utilization: {stats['avg_power_util']:.2%}")
    print(f"  Peak Power Utilization: {stats['max_power_util']:.2%}")
    print(f"  Average CPU Utilization: {stats['avg_cpu_util']:.2%}")
    print(f"  Average Temperature: {stats['avg_temp_f']:.1f}°F")
    print(f"  Power Usage Effectiveness: {stats['avg_pue']:.2f}")
    
    return df

def main():
    """Main execution function."""
    print("="*70)
    print("SYNTHETIC DATACENTER POWER CONSUMPTION GENERATOR")
    print("="*70)
    
    print("\nThis script generates realistic power consumption data since")
    print("Google's BigQuery public dataset is currently inaccessible.")
    print("\nThe model is based on industry research and combines:")
    print("  - CPU utilization patterns (actual data)")
    print("  - Temperature-dependent cooling load")
    print("  - Datacenter power characteristics (PUE, base load, etc.)")
    
    # Load data
    print("\n" + "-"*70)
    util_df = load_utilization_data()
    temp_df, temp_source = load_temperature_data()
    
    # Generate power data
    print("\n" + "-"*70)
    power_df = generate_power_consumption(util_df, temp_df, temp_source)
    
    # Add metadata
    power_df = add_metadata(power_df)
    
    # Save output
    print("\n" + "-"*70)
    print(f"Saving to: {OUTPUT_FILE}")
    power_df.to_csv(OUTPUT_FILE, index=False)
    
    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"✅ File saved successfully ({file_size_mb:.2f} MB)")
    
    # Show sample
    print("\n" + "="*70)
    print("SAMPLE DATA (first 10 rows)")
    print("="*70)
    print(power_df.head(10).to_string(index=False))
    
    print("\n" + "="*70)
    print("✅ SYNTHETIC POWER DATA GENERATION COMPLETE")
    print("="*70)
    print(f"\nOutput file: {OUTPUT_FILE}")
    print(f"Total records: {len(power_df):,}")
    print("\nThis data can now be used for training your forecasting models!")
    print("\nNote: While synthetic, this data reflects realistic datacenter")
    print("power consumption patterns based on industry research and your")
    print("actual CPU utilization data.")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
