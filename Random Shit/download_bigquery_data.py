"""
BIGQUERY POWER DATA DOWNLOAD SCRIPT
====================================

This script downloads Google datacenter power consumption data from BigQuery.

PREREQUISITES:
1. Google Cloud account (free tier OK)
2. Project created in Google Cloud Console
3. BigQuery API enabled
4. Google Cloud SDK installed OR google-cloud-bigquery package
5. Authenticated (gcloud auth login)

INSTALLATION:
pip install google-cloud-bigquery pandas

AUTHENTICATION:
Option A: gcloud auth application-default login
Option B: Set environment variable to service account key
    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"

"""

from google.cloud import bigquery
import pandas as pd
import os
import sys
from datetime import datetime, timedelta

# Configuration
PROJECT_ID = "datacenter-forecasting"  # Replace with your Google Cloud project ID
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data_Sources')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_power_data(cell='a', pdu_num=1, limit=None):
    """
    Download power data for a specific cell and PDU.
    
    Args:
        cell: Cell identifier ('a' through 'h')
        pdu_num: PDU number (varies by cell)
        limit: Max number of rows (None for all data)
    
    Returns:
        pandas DataFrame
    """
    client = bigquery.Client(project=PROJECT_ID)
    
    query = f"""
    SELECT 
        time,
        measured_power_util,
        production_power_util,
        '{cell}' as cell,
        'pdu{pdu_num:02d}' as pdu
    FROM `google.com:google-cluster-data.powerdata_2019.cell{cell}_pdu{pdu_num:02d}`
    WHERE time IS NOT NULL
        AND NOT bad_measurement_data
        AND NOT bad_production_power_data
    ORDER BY time
    {f'LIMIT {limit}' if limit else ''}
    """
    
    print(f"Downloading cell{cell}_pdu{pdu_num:02d}...")
    # Google's public dataset is in EU location
    df = client.query(query, location='EU').to_dataframe()
    print(f"  Downloaded {len(df)} rows")
    
    return df


def download_all_power_data(sample_only=False):
    """
    Download power data from all available cells and PDUs.
    
    Args:
        sample_only: If True, download only first 10000 rows per PDU (faster for testing)
    """
    # Define which PDUs are available for each cell
    # Based on Google's documentation
    cell_pdus = {
        'a': range(1, 11),   # PDUs 1-10
        'b': range(1, 22),   # PDUs 1-21
        'c': range(1, 16),   # PDUs 1-15
        'd': range(1, 13),   # PDUs 1-12
        'e': range(1, 17),   # PDUs 1-16
        'f': range(1, 18),   # PDUs 1-17
        'g': range(1, 15),   # PDUs 1-14
        'h': range(1, 11),   # PDUs 1-10
        # Note: cells i and j have different format (MVPP), handle separately
    }
    
    all_data = []
    
    for cell, pdus in cell_pdus.items():
        print(f"\nProcessing Cell {cell.upper()}...")
        for pdu_num in pdus:
            try:
                df = download_power_data(
                    cell=cell,
                    pdu_num=pdu_num,
                    limit=10000 if sample_only else None
                )
                all_data.append(df)
            except Exception as e:
                print(f"  Error downloading cell{cell}_pdu{pdu_num:02d}: {e}")
                continue
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n✅ Downloaded {len(combined_df)} total rows from {len(all_data)} PDUs")
        return combined_df
    else:
        print("❌ No data downloaded")
        return None


def convert_timestamps(df):
    """
    Convert BigQuery time field to readable datetime.
    
    BigQuery time is microseconds since 600 seconds before May 1, 2019 00:00:00 PT
    """
    print("\nConverting timestamps...")
    
    # Base time: May 1, 2019 00:00:00 PT (UTC-7)
    # Minus 600 seconds offset
    base_time = pd.Timestamp('2019-05-01 00:00:00', tz='US/Pacific') - pd.Timedelta(seconds=600)
    
    # Convert microseconds to timedelta and add to base
    df['datetime'] = base_time + pd.to_timedelta(df['time'], unit='us')
    
    # Add separate date/time columns for convenience
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    
    print("✅ Timestamps converted")
    return df


def aggregate_to_hourly(df):
    """
    Aggregate 5-minute data to hourly averages.
    """
    print("\nAggregating to hourly data...")
    
    # Ensure datetime column exists
    if 'datetime' not in df.columns:
        df = convert_timestamps(df)
    
    # Round datetime to hour
    df['hour_timestamp'] = df['datetime'].dt.floor('H')
    
    # Aggregate by hour, cell, and pdu
    hourly_df = df.groupby(['hour_timestamp', 'cell', 'pdu']).agg({
        'measured_power_util': 'mean',
        'production_power_util': 'mean',
        'time': 'count'  # Count of 5-min measurements per hour
    }).reset_index()
    
    hourly_df.rename(columns={'time': 'measurement_count'}, inplace=True)
    
    print(f"✅ Aggregated to {len(hourly_df)} hourly records")
    return hourly_df


def download_machine_mapping():
    """
    Download the machine-to-PDU mapping table.
    This shows which compute machines are powered by which PDU.
    """
    print("\nDownloading machine-to-PDU mapping...")
    
    client = bigquery.Client(project=PROJECT_ID)
    
    query = """
    SELECT 
        machine_id,
        cell,
        pdu
    FROM `google.com:google-cluster-data.powerdata_2019.machine_to_pdu_mapping`
    ORDER BY cell, pdu, machine_id
    """
    
    df = client.query(query, location='EU').to_dataframe()
    print(f"✅ Downloaded mapping for {len(df)} machines")
    
    return df


def main():
    """
    Main function to download and process all power data.
    """
    print("="*70)
    print("GOOGLE BIGQUERY POWER DATA DOWNLOAD")
    print("="*70)
    
    # Check if project ID is set
    if PROJECT_ID == "YOUR_PROJECT_ID_HERE":
        print("\n❌ ERROR: Please set your Google Cloud PROJECT_ID first!")
        print("   1. Go to: https://console.cloud.google.com/")
        print("   2. Create or select a project")
        print("   3. Copy the Project ID")
        print("   4. Replace 'YOUR_PROJECT_ID_HERE' in this script")
        return
    
    # Check authentication
    try:
        client = bigquery.Client(project=PROJECT_ID)
        client.query("SELECT 1").result()
        print(f"✅ Successfully authenticated with project: {PROJECT_ID}")
    except Exception as e:
        print(f"\n❌ Authentication error: {e}")
        print("\n Please authenticate first:")
        print("   Run: gcloud auth application-default login")
        print("   Or set GOOGLE_APPLICATION_CREDENTIALS environment variable")
        return
    
    print(f"\nData will be saved to: {OUTPUT_DIR}")
    
    # Check for command line arguments
    choice = None
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['--test', '-t', '3']:
            choice = '3'
            print("\nMode: TEST (single PDU)")
        elif arg in ['--sample', '-s', '1']:
            choice = '1'
            print("\nMode: SAMPLE (10K rows per PDU)")
        elif arg in ['--full', '-f', '2']:
            choice = '2'
            print("\nMode: FULL (complete dataset)")
        else:
            print(f"\n⚠️  Unknown argument: {arg}")
            print("Valid options: --test, --sample, --full")
            print("Falling back to interactive mode...\n")
    
    # Interactive mode if no valid command line argument
    if choice is None:
        print("\nDownload options:")
        print("  1. Sample data only (fast, ~1-2 min, 10K rows per PDU)")
        print("  2. Complete dataset (slow, ~20-30 min, all data)")
        print("  3. Single PDU test (fastest, for testing)")
        
        choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '3':
        # Test with single PDU
        print("\nDownloading test data (cell A, PDU 1)...")
        df = download_power_data(cell='a', pdu_num=1, limit=1000)
        
        if df is not None:
            # Convert timestamps
            df = convert_timestamps(df)
            
            # Save
            output_file = os.path.join(OUTPUT_DIR, 'google_power_data_2019_sample.csv')
            df.to_csv(output_file, index=False)
            print(f"\n✅ Sample data saved to: {output_file}")
            print(f"   Shape: {df.shape}")
            print(f"\nFirst few rows:")
            print(df.head())
    
    elif choice in ['1', '2']:
        sample_only = (choice == '1')
        
        # Download all power data
        df = download_all_power_data(sample_only=sample_only)
        
        if df is not None:
            # Convert timestamps
            df = convert_timestamps(df)
            
            # Save raw data
            raw_file = os.path.join(OUTPUT_DIR, 
                                    'google_power_data_2019_raw.csv' if not sample_only 
                                    else 'google_power_data_2019_sample.csv')
            df.to_csv(raw_file, index=False)
            print(f"\n✅ Raw data saved to: {raw_file}")
            
            # Aggregate to hourly
            hourly_df = aggregate_to_hourly(df)
            
            # Save hourly data
            hourly_file = os.path.join(OUTPUT_DIR, 
                                       'google_power_data_2019_hourly.csv' if not sample_only
                                       else 'google_power_data_2019_hourly_sample.csv')
            hourly_df.to_csv(hourly_file, index=False)
            print(f"✅ Hourly data saved to: {hourly_file}")
            
            # Download machine mapping
            try:
                mapping_df = download_machine_mapping()
                mapping_file = os.path.join(OUTPUT_DIR, 'machine_to_pdu_mapping.csv')
                mapping_df.to_csv(mapping_file, index=False)
                print(f"✅ Machine mapping saved to: {mapping_file}")
            except Exception as e:
                print(f"⚠️  Could not download machine mapping: {e}")
            
            # Print summary
            print("\n" + "="*70)
            print("DOWNLOAD COMPLETE")
            print("="*70)
            print(f"Raw data shape: {df.shape}")
            print(f"Hourly data shape: {hourly_df.shape}")
            print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
            print(f"Cells: {sorted(df['cell'].unique())}")
            print(f"PDUs: {len(df.groupby(['cell', 'pdu']))}")
            print("\n✅ Ready for model training!")
    
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
