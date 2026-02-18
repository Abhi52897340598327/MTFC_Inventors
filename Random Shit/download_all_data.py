"""
DATA DOWNLOAD SCRIPT FOR AI DATACENTER FORECASTING PROJECT
===========================================================

This script downloads all publicly available data needed for the forecasting model.

WHAT THIS SCRIPT DOWNLOADS:
1. PJM 2024 carbon intensity & generation data (EIA API)
2. Weather data for Ashburn, VA 2024 (NOAA API)
3. Instructions for BigQuery power data (requires separate setup)

BEFORE RUNNING:
1. Install required packages: pip install requests pandas
2. Get free API keys:
   - EIA API: https://www.eia.gov/opendata/register.php
   - NOAA API: https://www.ncdc.noaa.gov/cdo-web/token
3. Replace 'YOUR_API_KEY_HERE' below with your actual keys

"""

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import time
import os

# ============================================================================
# API KEYS - REPLACE THESE WITH YOUR OWN
# ============================================================================
EIA_API_KEY = 'NRjspMDoZtvn3rjwucZ3FbYhgFLWhmAsLPrriyig'  # Get from: https://www.eia.gov/opendata/register.php
NOAA_API_KEY = 'ShTYHitdxfFxaokzdVEaGxcFUhSmNhdS'  # Get from: https://www.ncdc.noaa.gov/cdo-web/token

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), 'Data_Sources')
os.makedirs(DATA_DIR, exist_ok=True)

# Date ranges
START_DATE_2024 = '2024-01-01'
END_DATE_2024 = '2024-12-31'

# ============================================================================
# 1. DOWNLOAD PJM CARBON INTENSITY & GENERATION DATA (2024)
# ============================================================================

def download_eia_grid_data():
    """
    Download hourly grid data from EIA Grid Monitor for PJM region.
    
    Data includes:
    - Demand (MW)
    - Generation by fuel type
    - Can be used to calculate carbon intensity
    
    API Docs: https://www.eia.gov/opendata/
    """
    print("\n" + "="*70)
    print("DOWNLOADING EIA GRID DATA FOR PJM REGION (2024)")
    print("="*70)
    
    if EIA_API_KEY == 'YOUR_EIA_API_KEY_HERE':
        print("⚠️  ERROR: Please set your EIA API key first!")
        print("   Get one free at: https://www.eia.gov/opendata/register.php")
        print("   Then replace 'YOUR_EIA_API_KEY_HERE' in this script")
        return None
    
    # EIA API v2 endpoint for electricity data
    # PJM is balancing authority "PJM"
    base_url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    
    params = {
        'api_key': EIA_API_KEY,
        'frequency': 'hourly',
        'data': ['value'],
        'facets[respondent][]': 'PJM',  # PJM Interconnection
        'start': START_DATE_2024 + 'T00',
        'end': END_DATE_2024 + 'T23',
        'sort[0][column]': 'period',
        'sort[0][direction]': 'asc',
        'offset': 0,
        'length': 5000
    }
    
    all_data = []
    
    try:
        print("Fetching data from EIA API...")
        print(f"Date range: {START_DATE_2024} to {END_DATE_2024}")
        
        while True:
            response = requests.get(base_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'response' in data and 'data' in data['response']:
                    batch = data['response']['data']
                    if not batch:
                        break
                    
                    all_data.extend(batch)
                    print(f"  Downloaded {len(all_data)} records so far...")
                    
                    # Check if there's more data
                    if len(batch) < 5000:
                        break
                    
                    params['offset'] += 5000
                    time.sleep(0.5)  # Be nice to the API
                else:
                    print(f"⚠️  Unexpected response format: {data}")
                    break
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                break
        
        if all_data:
            df = pd.DataFrame(all_data)
            output_file = os.path.join(DATA_DIR, 'pjm_grid_data_2024_raw.csv')
            df.to_csv(output_file, index=False)
            print(f"✅ Saved {len(df)} records to: {output_file}")
            print(f"   Columns: {list(df.columns)}")
            return df
        else:
            print("⚠️  No data downloaded")
            return None
            
    except Exception as e:
        print(f"❌ Error downloading EIA data: {e}")
        return None


def download_eia_emissions_data():
    """
    Download emissions data from EIA.
    This provides CO2 emissions by generation source.
    """
    print("\n" + "="*70)
    print("DOWNLOADING EIA EMISSIONS DATA")
    print("="*70)
    
    if EIA_API_KEY == 'YOUR_EIA_API_KEY_HERE':
        print("⚠️  Skipping - API key not set")
        return None
    
    # Note: This is a simplified example. The actual emissions API may differ.
    # You may need to calculate emissions from generation mix using emission factors.
    
    print("Note: Carbon intensity calculation requires:")
    print("  1. Generation by fuel type (from grid data above)")
    print("  2. Emission factors for each fuel:")
    print("     - Coal: ~1000 g CO2/kWh")
    print("     - Natural Gas: ~450 g CO2/kWh")
    print("     - Nuclear/Renewables: ~0 g CO2/kWh")
    print("\nYou'll calculate this in the data processing step.")
    
    return None

# ============================================================================
# 2. DOWNLOAD WEATHER DATA FOR ASHBURN, VA (2024)
# ============================================================================

def download_noaa_weather_data():
    """
    Download hourly temperature data for Ashburn, VA (Dulles Airport station).
    
    Station: GHCND:USW00093738 (Washington Dulles International Airport)
    
    API Docs: https://www.ncdc.noaa.gov/cdo-web/webservices/v2
    """
    print("\n" + "="*70)
    print("DOWNLOADING NOAA WEATHER DATA FOR ASHBURN, VA (2024)")
    print("="*70)
    
    if NOAA_API_KEY == 'YOUR_NOAA_API_KEY_HERE':
        print("⚠️  ERROR: Please set your NOAA API key first!")
        print("   Get one free at: https://www.ncdc.noaa.gov/cdo-web/token")
        print("   (Takes 1-2 days to receive via email)")
        print("   Then replace 'YOUR_NOAA_API_KEY_HERE' in this script")
        return None
    
    base_url = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
    
    headers = {
        'token': NOAA_API_KEY
    }
    
    # Dulles Airport station
    station_id = 'GHCND:USW00093738'
    
    params = {
        'datasetid': 'GHCND',  # Global Historical Climatology Network - Daily
        'stationid': station_id,
        'startdate': START_DATE_2024,
        'enddate': END_DATE_2024,
        'datatypeid': 'TAVG,TMAX,TMIN',  # Temperature average, max, min
        'units': 'metric',
        'limit': 1000
    }
    
    all_data = []
    
    try:
        print(f"Fetching weather data for station: {station_id}")
        print(f"Date range: {START_DATE_2024} to {END_DATE_2024}")
        
        offset = 1
        while True:
            params['offset'] = offset
            response = requests.get(base_url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'results' in data:
                    batch = data['results']
                    all_data.extend(batch)
                    print(f"  Downloaded {len(all_data)} records so far...")
                    
                    if len(batch) < 1000:
                        break
                    
                    offset += 1000
                    time.sleep(0.5)  # NOAA rate limit: 5 requests per second
                else:
                    break
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                break
        
        if all_data:
            df = pd.DataFrame(all_data)
            output_file = os.path.join(DATA_DIR, 'ashburn_va_temperature_2024.csv')
            df.to_csv(output_file, index=False)
            print(f"✅ Saved {len(df)} records to: {output_file}")
            return df
        else:
            print("⚠️  No data downloaded")
            return None
            
    except Exception as e:
        print(f"❌ Error downloading NOAA data: {e}")
        return None


# ============================================================================
# 3. BIGQUERY POWER DATA INSTRUCTIONS
# ============================================================================

def print_bigquery_instructions():
    """
    Print instructions for accessing Google BigQuery power data.
    This cannot be automated without user's Google Cloud credentials.
    """
    print("\n" + "="*70)
    print("GOOGLE BIGQUERY POWER DATA - MANUAL SETUP REQUIRED")
    print("="*70)
    
    instructions = """
    The Google datacenter power consumption data requires BigQuery access.
    This script CANNOT download it automatically (requires your Google Cloud account).
    
    📋 SETUP STEPS:
    
    1. CREATE GOOGLE CLOUD PROJECT (One-time setup)
       → Go to: https://console.cloud.google.com/
       → Click "Create Project"
       → Name it (e.g., "datacenter-forecasting")
       → Enable billing (free tier includes $300 credit)
    
    2. ENABLE BIGQUERY API
       → Go to: https://console.cloud.google.com/apis/library/bigquery.googleapis.com
       → Click "Enable"
    
    3. INSTALL GOOGLE CLOUD SDK
       → Download: https://cloud.google.com/sdk/docs/install
       → Or use pip: pip install google-cloud-bigquery
    
    4. AUTHENTICATE
       → Run in terminal: gcloud auth login
       → Or set credentials: gcloud auth application-default login
    
    5. QUERY THE DATA
       → Use the script: download_bigquery_data.py (separate file)
       → Or query manually via BigQuery console
    
    📊 DATASET INFORMATION:
    
    Project: google.com:google-cluster-data
    Dataset: powerdata_2019
    Tables: 
       - cella_pdu01 through cella_pdu10
       - cellb_pdu01 through cellb_pdu21
       - ... (8 cells total, 57 PDUs)
       - machine_to_pdu_mapping
    
    Fields:
       - time: Timestamp (microseconds since May 1, 2019)
       - measured_power_util: Actual measured power utilization
       - production_power_util: Estimated production power
       - pdu: Power distribution unit ID
    
    💰 COST:
       - Queries cost ~$5 per TB
       - This dataset is small (~1-2 GB)
       - Total cost: ~$0.01 - $0.05
       - Free tier covers it!
    
    📝 SAMPLE QUERY:
    
    SELECT 
        time,
        cell, 
        pdu,
        measured_power_util,
        production_power_util
    FROM `google.com:google-cluster-data.powerdata_2019.cell*`
    WHERE time IS NOT NULL
    ORDER BY time
    LIMIT 1000
    
    Export results as CSV and save to Data_Sources folder.
    
    🔗 DOCUMENTATION:
       - https://github.com/google/cluster-data/blob/master/PowerData2019.md
       - https://cloud.google.com/bigquery/docs/quickstarts
    """
    
    print(instructions)
    
    # Save instructions to file
    instructions_file = os.path.join(os.path.dirname(DATA_DIR), 
                                      'Model_Files', 
                                      'BIGQUERY_SETUP_INSTRUCTIONS.txt')
    with open(instructions_file, 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print(f"\n💾 Instructions saved to: {instructions_file}")


# ============================================================================
# 4. VIRGINIA DATACENTER CAPACITY DATA (RESEARCH REQUIRED)
# ============================================================================

def print_virginia_datacenter_research_guide():
    """
    Guide for researching Virginia datacenter capacity projections.
    """
    print("\n" + "="*70)
    print("VIRGINIA DATACENTER CAPACITY - RESEARCH GUIDE")
    print("="*70)
    
    guide = """
    Northern Virginia (esp. Ashburn) is the world's largest datacenter market.
    You need to research/estimate the total AI datacenter capacity for scenarios.
    
    📊 RESEARCH SOURCES:
    
    1. UTILITY PLANNING DOCUMENTS
       → Dominion Energy Virginia Integrated Resource Plan (IRP)
       → https://www.dominionenergy.com/company/making-energy/Resource-planning
       → Look for: Power demand forecasts, datacenter load growth
    
    2. PUBLIC UTILITY COMMISSION FILINGS
       → Virginia State Corporation Commission
       → https://www.scc.virginia.gov/
       → Search: "datacenter" in recent case filings
    
    3. INDUSTRY REPORTS
       → JLL Datacenter Outlook
       → CBRE Data Center Trends
       → Synergy Research Group
       → Data Center Frontier (news site)
    
    4. NEWS AND ANNOUNCEMENTS
       → Search: "virginia datacenter construction"
       → Look for MW capacity numbers in announcements
       → Companies: AWS, Microsoft, Google, Meta all have VA facilities
    
    5. ACADEMIC SOURCES
       → Virginia Tech reports on datacenter energy
       → UVA research on grid impacts
    
    📈 WHAT YOU NEED:
    
    - Total datacenter capacity in Northern Virginia: ~XXX MW
    - AI-specific capacity (subset): ~XXX MW
    - Growth rate: ~20-30% per year (industry average)
    - PUE (efficiency): 1.1-1.2 for modern facilities
    
    🎯 BASELINE ESTIMATES (if data unavailable):
    
    2019 Baseline: ~1,500 MW total datacenter capacity in Northern VA
    2024 Estimate: ~2,500-3,000 MW (based on 15-20% annual growth)
    AI Subset: ~30-40% of new capacity (750-1,200 MW AI-focused)
    
    2026 Scenario: Add 500-1,000 MW new AI capacity
    2028 Scenario: Add 1,500-2,000 MW new AI capacity
    
    Use these as starting points for your scenarios if you can't find
    exact numbers. Document your assumptions clearly.
    
    💡 TIP: 
    For a science fair project, reasonable estimates with documented 
    assumptions are acceptable. Focus on the modeling methodology - 
    actual capacity numbers matter less than the analytical framework.
    """
    
    print(guide)
    
    guide_file = os.path.join(os.path.dirname(DATA_DIR), 
                               'Information_Docs',
                               'virginia_datacenter_research_guide.txt')
    with open(guide_file, 'w', encoding='utf-8') as f:
        f.write(guide)
    
    print(f"\n💾 Guide saved to: {guide_file}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to download all available data.
    """
    print("\n" + "="*70)
    print("AI DATACENTER FORECASTING - DATA DOWNLOAD SCRIPT")
    print("="*70)
    print(f"\nData will be saved to: {DATA_DIR}")
    print("\nThis script downloads:")
    print("  ✓ PJM grid data (2024) - via EIA API")
    print("  ✓ Weather data (2024) - via NOAA API")
    print("  ℹ BigQuery power data - manual setup required")
    print("  ℹ Virginia capacity - research required")
    
    # Check if API keys are set
    if EIA_API_KEY == 'YOUR_EIA_API_KEY_HERE':
        print("\n⚠️  WARNING: EIA API key not set")
        print("    Get one free at: https://www.eia.gov/opendata/register.php")
    
    if NOAA_API_KEY == 'YOUR_NOAA_API_KEY_HERE':
        print("\n⚠️  WARNING: NOAA API key not set")
        print("    Get one free at: https://www.ncdc.noaa.gov/cdo-web/token")
    
    print("\n" + "-"*70)
    
    # Download what we can
    eia_data = download_eia_grid_data()
    download_eia_emissions_data()
    weather_data = download_noaa_weather_data()
    
    # Print manual instructions
    print_bigquery_instructions()
    print_virginia_datacenter_research_guide()
    
    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    print(f"✅ EIA Grid Data: {'Downloaded' if eia_data is not None else 'Need API key'}")
    print(f"✅ Weather Data: {'Downloaded' if weather_data is not None else 'Need API key'}")
    print(f"⏳ BigQuery Power Data: See instructions above")
    print(f"📚 Virginia Capacity: See research guide above")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Get API keys if you haven't already")
    print("2. Re-run this script to download remaining data")
    print("3. Follow BigQuery setup instructions")
    print("4. Research Virginia datacenter capacity")
    print("5. Run data processing script once all data is collected")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
