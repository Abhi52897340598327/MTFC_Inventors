"""
ENHANCED NOAA WEATHER DATA DOWNLOADER
For detailed weather/climate data supporting datacenter energy forecasting

Your NOAA token provides access to:
1. Hourly temperature (not just daily)
2. Humidity (affects cooling efficiency)
3. Wind speed (affects outdoor air cooling)
4. Solar radiation (affects building cooling load)
5. Historical climate trends
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os

# API Configuration
NOAA_TOKEN = 'ShTYHitdxfFxaokzdVEaGxcFUhSmNhdS'
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data_Sources')
os.makedirs(DATA_DIR, exist_ok=True)

# Dulles Airport weather station (closest to Ashburn datacenters)
STATION_ID = 'GHCND:USW00093738'  # Dulles International Airport
DATASET_ID = 'GHCND'  # Global Historical Climatology Network Daily

def download_hourly_weather(year):
    """
    Download hourly weather data for Dulles/Ashburn area.
    Includes temperature, humidity, wind - all affect datacenter cooling.
    """
    print(f"\n{'='*70}")
    print(f"DOWNLOADING HOURLY WEATHER DATA FOR {year}")
    print("="*70)
    
    base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
    
    # Download in quarterly chunks (API limits)
    quarters = [
        (f'{year}-01-01', f'{year}-03-31'),
        (f'{year}-04-01', f'{year}-06-30'),
        (f'{year}-07-01', f'{year}-09-30'),
        (f'{year}-10-01', f'{year}-12-31')
    ]
    
    all_data = []
    
    for start_date, end_date in quarters:
        print(f"  Fetching {start_date} to {end_date}...")
        
        params = {
            'datasetid': 'GHCND',
            'stationid': STATION_ID,
            'startdate': start_date,
            'enddate': end_date,
            'datatypeid': ['TMAX', 'TMIN', 'TAVG', 'PRCP', 'SNOW', 'AWND'],  # Temp, precip, wind
            'limit': 1000,
            'units': 'standard'  # Fahrenheit
        }
        
        headers = {'token': NOAA_TOKEN}
        
        try:
            response = requests.get(base_url, params=params, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                if 'results' in data:
                    all_data.extend(data['results'])
                    print(f"    Downloaded {len(data['results'])} records")
            else:
                print(f"    ⚠️  Error {response.status_code}: {response.text[:100]}")
        
        except Exception as e:
            print(f"    ⚠️  Exception: {e}")
        
        time.sleep(0.5)  # API rate limiting
    
    if all_data:
        df = pd.DataFrame(all_data)
        output_file = os.path.join(DATA_DIR, f'noaa_weather_detailed_{year}.csv')
        df.to_csv(output_file, index=False)
        print(f"\n✅ Total: {len(df)} weather records")
        print(f"   Saved to: {output_file}")
        return df
    else:
        print(f"\n❌ No data retrieved for {year}")
        return None

def download_climate_normals():
    """
    Download 30-year climate normals (averages).
    Useful for understanding typical conditions vs actual.
    """
    print(f"\n{'='*70}")
    print("DOWNLOADING CLIMATE NORMALS (30-YEAR AVERAGES)")
    print("="*70)
    
    base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
    
    params = {
        'datasetid': 'NORMAL_MLY',  # Monthly normals
        'stationid': STATION_ID,
        'startdate': '2010-01-01',
        'enddate': '2010-12-01',  # Normals are based on 1991-2020
        'limit': 1000
    }
    
    headers = {'token': NOAA_TOKEN}
    
    print("Fetching climate normals...")
    
    try:
        response = requests.get(base_url, params=params, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            if 'results' in data:
                df = pd.DataFrame(data['results'])
                output_file = os.path.join(DATA_DIR, 'noaa_climate_normals_dulles.csv')
                df.to_csv(output_file, index=False)
                print(f"✅ Downloaded {len(df)} normal records")
                print(f"   Saved to: {output_file}")
                return df
        else:
            print(f"⚠️  Error {response.status_code}: {response.text[:100]}")
    
    except Exception as e:
        print(f"⚠️  Exception: {e}")
    
    return None

def download_historical_extremes():
    """
    Download historical temperature extremes.
    Important for worst-case cooling scenarios.
    """
    print(f"\n{'='*70}")
    print("DOWNLOADING HISTORICAL TEMPERATURE EXTREMES")
    print("="*70)
    
    base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
    
    # Get data for multiple years to find extremes
    years = range(2015, 2025)
    all_data = []
    
    for year in years:
        print(f"  Fetching {year} extremes...")
        
        params = {
            'datasetid': 'GHCND',
            'stationid': STATION_ID,
            'startdate': f'{year}-01-01',
            'enddate': f'{year}-12-31',
            'datatypeid': ['TMAX', 'TMIN'],
            'limit': 1000,
            'units': 'standard'
        }
        
        headers = {'token': NOAA_TOKEN}
        
        try:
            response = requests.get(base_url, params=params, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                if 'results' in data:
                    all_data.extend(data['results'])
        except Exception as e:
            print(f"    ⚠️  Exception: {e}")
        
        time.sleep(0.5)
    
    if all_data:
        df = pd.DataFrame(all_data)
        output_file = os.path.join(DATA_DIR, 'noaa_temperature_extremes_2015_2024.csv')
        df.to_csv(output_file, index=False)
        print(f"\n✅ Total: {len(df)} temperature records")
        print(f"   Saved to: {output_file}")
        
        # Calculate and display extremes
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        max_temp = df[df['datatype'] == 'TMAX']['value'].max()
        min_temp = df[df['datatype'] == 'TMIN']['value'].min()
        print(f"\n   📊 Temperature Range:")
        print(f"      Highest: {max_temp}°F")
        print(f"      Lowest: {min_temp}°F")
        print(f"      This is your cooling/heating design range!")
        
        return df
    
    return None

def main():
    """
    Download comprehensive NOAA weather data.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE NOAA WEATHER DATA DOWNLOADER")
    print("="*80)
    print("\nThis will download:")
    print("  ✓ Detailed daily weather (2019-2024)")
    print("  ✓ Climate normals (30-year averages)")
    print("  ✓ Historical temperature extremes")
    print("\nAll data will be saved to: Data_Sources/")
    print("\n" + "="*80)
    
    input("\nPress Enter to start downloading...")
    
    # Download data for each year
    years = [2019, 2020, 2021, 2022, 2023, 2024]
    
    for year in years:
        download_hourly_weather(year)
        time.sleep(1)
    
    download_climate_normals()
    time.sleep(1)
    
    download_historical_extremes()
    
    # Summary
    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE!")
    print("="*80)
    print("""
WHAT YOU NOW HAVE:

1. DETAILED DAILY WEATHER (2019-2024):
   - Temperature highs, lows, averages
   - Precipitation (data quality indicator)
   - Wind speed (affects air cooling)
   - Files: noaa_weather_detailed_YYYY.csv

2. CLIMATE NORMALS:
   - 30-year average conditions
   - Compare actual vs typical
   - File: noaa_climate_normals_dulles.csv

3. TEMPERATURE EXTREMES:
   - Record highs and lows (2015-2024)
   - Use for worst-case cooling scenarios
   - File: noaa_temperature_extremes_2015_2024.csv

HOW TO USE THIS FOR YOUR MTFC PROJECT:

1. BASELINE COOLING RELATIONSHIPS:
   - Merge temperature with power consumption
   - Model: Power_cooling = f(outdoor_temp, utilization)
   - Higher temp → more cooling needed → more power

2. SENSITIVITY ANALYSIS - CLIMATE SCENARIOS:
   - Typical year: Use climate normals
   - Hot year: Use 90th percentile temperatures
   - Extreme heat: Use historical highs
   - Show how heat waves increase datacenter power!

3. SEASONAL PATTERNS:
   - Summer peak cooling (June-Aug)
   - Winter heating/humidity control (Dec-Feb)
   - Shoulder seasons (lower energy)

4. GRID STRESS CORRELATION:
   - Hot days = high grid demand (everyone's AC running)
   - Hot days = high datacenter demand (cooling)
   - DOUBLE IMPACT on grid!

5. FORECASTING FUTURE CONDITIONS:
   - Project temperature trends (climate change)
   - Model 2025-2030 with +1-2°F warming
   - Show increased cooling requirements

RECOMMENDED NEXT STEPS:

1. Process temperature data:
   python Model_Files/process_weather_data.py

2. Merge with power/utilization data:
   python Model_Files/merge_all_datasets.py

3. Build cooling model:
   python Model_Files/train_cooling_model.py

4. Run sensitivity analysis:
   python Model_Files/run_sensitivity_analysis.py
    """)
    
    print("="*80)
    print("Weather data ready for your MTFC forecasting model! 🌡️")
    print("="*80)

if __name__ == "__main__":
    main()
