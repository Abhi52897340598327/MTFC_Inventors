"""
COMPREHENSIVE EIA DATA DOWNLOADER
For MTFC Virginia Datacenter Forecasting Project

This script downloads extensive EIA datasets useful for:
1. Electricity consumption forecasting
2. Generation capacity and trends
3. Carbon emissions analysis
4. Grid stress/reliability metrics
5. Fuel mix evolution (for sensitivity analysis)

Your EIA API key provides access to hundreds of datasets!
"""

import requests
import pandas as pd
import json
from datetime import datetime
import time
import os

# API Configuration
EIA_API_KEY = 'NRjspMDoZtvn3rjwucZ3FbYhgFLWhmAsLPrriyig'
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data_Sources')
os.makedirs(DATA_DIR, exist_ok=True)

def download_state_electricity_consumption():
    """
    Download Virginia electricity consumption by sector (residential, commercial, industrial).
    Historical data for trend analysis and forecasting.
    """
    print("\n" + "="*70)
    print("DOWNLOADING VIRGINIA ELECTRICITY CONSUMPTION")
    print("="*70)
    
    base_url = "https://api.eia.gov/v2/electricity/retail-sales/data/"
    
    params = {
        'api_key': EIA_API_KEY,
        'frequency': 'monthly',
        'data': ['customers', 'price', 'revenue', 'sales'],
        'facets[stateid][]': 'VA',
        'start': '2015-01',
        'end': '2024-12',
        'sort[0][column]': 'period',
        'sort[0][direction]': 'asc',
        'offset': 0,
        'length': 5000
    }
    
    print("Fetching Virginia retail electricity sales...")
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if 'response' in data and 'data' in data['response']:
            df = pd.DataFrame(data['response']['data'])
            output_file = os.path.join(DATA_DIR, 'virginia_electricity_consumption_2015_2024.csv')
            df.to_csv(output_file, index=False)
            print(f"[OK] Downloaded {len(df)} records")
            print(f"   Saved to: {output_file}")
            print(f"   Columns: {list(df.columns)}")
            return df
    else:
        print(f"[FAIL] Error {response.status_code}: {response.text}")
    
    return None

def download_generation_capacity():
    """
    Download Virginia power plant capacity data.
    Shows what generation is available to meet demand.
    """
    print("\n" + "="*70)
    print("DOWNLOADING VIRGINIA GENERATION CAPACITY")
    print("="*70)
    
    base_url = "https://api.eia.gov/v2/electricity/operating-generator-capacity/data/"
    
    params = {
        'api_key': EIA_API_KEY,
        'frequency': 'annual',
        'data': ['nameplate-capacity-mw', 'net-summer-capacity-mw', 'net-winter-capacity-mw'],
        'facets[stateid][]': 'VA',
        'start': '2015',
        'end': '2024',
        'sort[0][column]': 'period',
        'sort[0][direction]': 'desc',
        'offset': 0,
        'length': 5000
    }
    
    print("Fetching Virginia generation capacity...")
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if 'response' in data and 'data' in data['response']:
            df = pd.DataFrame(data['response']['data'])
            output_file = os.path.join(DATA_DIR, 'virginia_generation_capacity_2015_2024.csv')
            df.to_csv(output_file, index=False)
            print(f"[OK] Downloaded {len(df)} records")
            print(f"   Saved to: {output_file}")
            print(f"   Shows capacity by fuel type (coal, gas, solar, wind, etc.)")
            return df
    else:
        print(f"[FAIL] Error {response.status_code}: {response.text}")
    
    return None

def download_generation_trends():
    """
    Download actual electricity generation (not capacity).
    Shows monthly generation by fuel source.
    """
    print("\n" + "="*70)
    print("DOWNLOADING VIRGINIA ELECTRICITY GENERATION BY SOURCE")
    print("="*70)
    
    base_url = "https://api.eia.gov/v2/electricity/electric-power-operational-data/data/"
    
    params = {
        'api_key': EIA_API_KEY,
        'frequency': 'monthly',
        'data': ['generation'],
        'facets[location][]': 'VA',
        'start': '2015-01',
        'end': '2024-12',
        'sort[0][column]': 'period',
        'sort[0][direction]': 'asc',
        'offset': 0,
        'length': 5000
    }
    
    print("Fetching Virginia generation data...")
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if 'response' in data and 'data' in data['response']:
            df = pd.DataFrame(data['response']['data'])
            output_file = os.path.join(DATA_DIR, 'virginia_generation_by_fuel_2015_2024.csv')
            df.to_csv(output_file, index=False)
            print(f"[OK] Downloaded {len(df)} records")
            print(f"   Saved to: {output_file}")
            print(f"   Shows fuel mix trends (coal declining, renewables growing)")
            return df
    else:
        print(f"[FAIL] Error {response.status_code}: {response.text}")
    
    return None

def download_co2_emissions():
    """
    Download Virginia CO2 emissions from electricity generation.
    CRITICAL for your carbon emissions analysis!
    """
    print("\n" + "="*70)
    print("DOWNLOADING VIRGINIA CO2 EMISSIONS DATA")
    print("="*70)
    
    base_url = "https://api.eia.gov/v2/co2-emissions/co2-emissions-aggregates/data/"
    
    params = {
        'api_key': EIA_API_KEY,
        'frequency': 'annual',
        'data': ['value'],
        'facets[stateId][]': 'VA',
        'facets[sectorId][]': 'EC',  # Electric Power sector
        'start': '2015',
        'end': '2023',
        'sort[0][column]': 'period',
        'sort[0][direction]': 'desc',
        'offset': 0,
        'length': 5000
    }
    
    print("Fetching Virginia CO2 emissions...")
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if 'response' in data and 'data' in data['response']:
            df = pd.DataFrame(data['response']['data'])
            output_file = os.path.join(DATA_DIR, 'virginia_co2_emissions_2015_2023.csv')
            df.to_csv(output_file, index=False)
            print(f"[OK] Downloaded {len(df)} records")
            print(f"   Saved to: {output_file}")
            print(f"   Annual CO2 emissions from electricity generation")
            print(f"   Use this to calculate emissions intensity trends!")
            return df
    else:
        print(f"[FAIL] Error {response.status_code}: {response.text}")
    
    return None

def download_hourly_demand():
    """
    Download hourly electricity demand for PJM/Virginia region.
    CRITICAL for grid stress analysis!
    """
    print("\n" + "="*70)
    print("DOWNLOADING HOURLY ELECTRICITY DEMAND (PJM REGION)")
    print("="*70)
    
    base_url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    
    # Download in chunks (API has limits)
    all_data = []
    years = [2019, 2020, 2021, 2022, 2023, 2024]
    
    for year in years:
        print(f"  Fetching {year} data...")
        params = {
            'api_key': EIA_API_KEY,
            'frequency': 'hourly',
            'data': ['value'],
            'facets[respondent][]': 'PJM',
            'facets[type][]': 'D',  # Demand
            'start': f'{year}-01-01T00',
            'end': f'{year}-12-31T23',
            'sort[0][column]': 'period',
            'sort[0][direction]': 'asc',
            'offset': 0,
            'length': 5000
        }
        
        year_data = []
        while True:
            response = requests.get(base_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if 'response' in data and 'data' in data['response']:
                    batch = data['response']['data']
                    if not batch:
                        break
                    year_data.extend(batch)
                    if len(batch) < 5000:
                        break
                    params['offset'] += 5000
                else:
                    break
            else:
                print(f"    [WARN] Error {response.status_code}")
                break
        
        all_data.extend(year_data)
        print(f"    Downloaded {len(year_data)} records for {year}")
        time.sleep(0.5)  # Be nice to API
    
    if all_data:
        df = pd.DataFrame(all_data)
        output_file = os.path.join(DATA_DIR, 'pjm_hourly_demand_2019_2024.csv')
        df.to_csv(output_file, index=False)
        print(f"\n[OK] Total: {len(df)} hourly demand records")
        print(f"   Saved to: {output_file}")
        print(f"   Use this to identify peak demand hours and grid stress!")
        return df
    
    return None

def download_renewable_generation():
    """
    Download renewable energy generation for Virginia.
    Useful for sensitivity analysis (cleaner grid = lower emissions).
    """
    print("\n" + "="*70)
    print("DOWNLOADING VIRGINIA RENEWABLE ENERGY GENERATION")
    print("="*70)
    
    base_url = "https://api.eia.gov/v2/electricity/electric-power-operational-data/data/"
    
    params = {
        'api_key': EIA_API_KEY,
        'frequency': 'monthly',
        'data': ['generation'],
        'facets[location][]': 'VA',
        'facets[fueltypeid][]': ['SUN', 'WND', 'HYC'],  # Solar, Wind, Hydro
        'start': '2015-01',
        'end': '2024-12',
        'sort[0][column]': 'period',
        'sort[0][direction]': 'asc',
        'offset': 0,
        'length': 5000
    }
    
    print("Fetching Virginia renewable generation...")
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if 'response' in data and 'data' in data['response']:
            df = pd.DataFrame(data['response']['data'])
            output_file = os.path.join(DATA_DIR, 'virginia_renewable_generation_2015_2024.csv')
            df.to_csv(output_file, index=False)
            print(f"[OK] Downloaded {len(df)} records")
            print(f"   Saved to: {output_file}")
            print(f"   Shows renewable energy growth trends")
            return df
    else:
        print(f"[FAIL] Error {response.status_code}: {response.text}")
    
    return None

def main():
    """
    Download all comprehensive EIA datasets.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE EIA DATA DOWNLOADER FOR MTFC PROJECT")
    print("="*80)
    print("\nThis will download extensive datasets for:")
    print("  [+] Virginia electricity consumption trends")
    print("  [+] Generation capacity by fuel type")
    print("  [+] Actual generation (fuel mix evolution)")
    print("  [+] CO2 emissions (for carbon analysis)")
    print("  [+] Hourly demand (for grid stress analysis)")
    print("  [+] Renewable energy trends")
    print("\nAll data will be saved to: Data_Sources/")
    print("\n" + "="*80)
    
    input("\nPress Enter to start downloading...")
    
    # Download all datasets
    results = {}
    
    results['consumption'] = download_state_electricity_consumption()
    time.sleep(1)
    
    results['capacity'] = download_generation_capacity()
    time.sleep(1)
    
    results['generation'] = download_generation_trends()
    time.sleep(1)
    
    results['emissions'] = download_co2_emissions()
    time.sleep(1)
    
    results['demand'] = download_hourly_demand()
    time.sleep(1)
    
    results['renewables'] = download_renewable_generation()
    
    # Summary
    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE!")
    print("="*80)
    
    for name, data in results.items():
        status = "[OK]" if data is not None else "[FAIL]"
        print(f"{status} {name.capitalize()}: {'Downloaded' if data is not None else 'Failed'}")
    
    print("\n" + "="*80)
    print("WHAT TO DO NEXT:")
    print("="*80)
    print("""
1. CONSUMPTION DATA: Use to establish baseline trends and forecast future demand
   - See virginia_electricity_consumption_2015_2024.csv
   - Shows growth by sector (residential, commercial, industrial)

2. GENERATION & CAPACITY: Understand the generation fleet
   - virginia_generation_capacity_2015_2024.csv: Total capacity available
   - virginia_generation_by_fuel_2015_2024.csv: What's actually generating
   - Use for sensitivity: "What if more renewables come online?"

3. EMISSIONS DATA: Calculate carbon intensity trends
   - virginia_co2_emissions_2015_2023.csv
   - Emissions per MWh = Total_CO2 / Total_Generation
   - Forecast future emissions intensity (declining as grid cleans up)

4. HOURLY DEMAND: Identify grid stress periods
   - pjm_hourly_demand_2019_2024.csv
   - Find peak demand hours (top 10%)
   - Calculate when datacenters add to peak vs off-peak
   - THIS IS CRITICAL for grid strain analysis!

5. RENEWABLES: Model cleaner grid scenarios
   - virginia_renewable_generation_2015_2024.csv
   - Project renewable growth
   - Lower future emissions per datacenter MWh

6. COMBINE WITH YOUR EXISTING DATA:
   - PJM grid data (you have)
   - Temperature data (you have)
   - Power consumption model (synthetic or BigQuery)
   
7. FOR SENSITIVITY ANALYSIS:
   Test these scenarios:
   a. Datacenter capacity: +500 MW, +1000 MW, +2000 MW
   b. Utilization: 50%, 70%, 90% average
   c. PUE: 1.1 (efficient) vs 1.5 (average)
   d. Grid cleanliness: 2024 vs 2030 (more renewables)
   e. Timing: Load shifting to off-peak hours

8. CALCULATE METRICS:
   - Total energy consumption (MWh/year)
   - Carbon emissions (metric tons CO2/year)
   - Peak demand contribution (MW)
   - Coincident peak factor (% of datacenter during grid peak)
   - Grid stress score (custom metric)
    """)
    
    print("="*80)
    print("Good luck with your MTFC project!")
    print("="*80)

if __name__ == "__main__":
    main()
