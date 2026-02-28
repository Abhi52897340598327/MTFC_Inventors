"""
Download real PJM hourly demand data from the EIA Open Data API v2.
Saves to Data_Sources/pjm_hourly_demand_2019_eia.csv

EIA API endpoint: /v2/electricity/rto/region-data/data/
Data: Hourly demand (MWh) for PJM Interconnection, LLC
Source: Form EIA-930 (Hourly Electric Grid Monitor)
"""

import requests
import pandas as pd
import os
import time

API_KEY = "NRjspMDoZtvn3rjwucZ3FbYhgFLWhmAsLPrriyig"
BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "Data_Sources")

def fetch_pjm_demand(start: str, end: str, offset: int = 0, length: int = 5000) -> dict:
    """Fetch PJM hourly demand from EIA API."""
    params = {
        "api_key": API_KEY,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": "PJM",
        "facets[type][]": "D",
        "start": start,
        "end": end,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": offset,
        "length": length,
    }
    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()

def download_year(year: int) -> pd.DataFrame:
    """Download all hourly demand records for a given year."""
    start = f"{year}-01-01T00"
    end = f"{year}-12-31T23"
    
    all_records = []
    offset = 0
    
    # First request to get total count
    data = fetch_pjm_demand(start, end, offset=0)
    total = int(data["response"]["total"])
    records = data["response"]["data"]
    all_records.extend(records)
    print(f"  Year {year}: {total} total records. Fetched {len(records)} (offset 0)")
    
    # Paginate if needed
    while len(all_records) < total:
        offset = len(all_records)
        time.sleep(1)  # Be respectful of rate limits
        data = fetch_pjm_demand(start, end, offset=offset)
        records = data["response"]["data"]
        if not records:
            break
        all_records.extend(records)
        print(f"  Fetched {len(records)} more (offset {offset}), total so far: {len(all_records)}")
    
    df = pd.DataFrame(all_records)
    print(f"  Year {year} complete: {len(df)} records")
    return df

def main():
    print("=" * 60)
    print("Downloading PJM Hourly Demand from EIA API v2")
    print("=" * 60)
    
    # Download 2019-2024 data
    years_to_download = [2020, 2021, 2022, 2023, 2024]
    
    all_dfs = []
    for year in years_to_download:
        print(f"\nDownloading {year}...")
        df = download_year(year)
        all_dfs.append(df)
    
    # Combine newly downloaded years
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Clean up the data
    combined["value"] = pd.to_numeric(combined["value"], errors="coerce")
    combined = combined.rename(columns={
        "period": "datetime_utc",
        "value": "demand_mwh",
    })
    
    # Keep only useful columns
    output = combined[["datetime_utc", "respondent", "demand_mwh"]].copy()
    
    # Load existing 2019 data and merge
    existing_2019 = os.path.join(OUTPUT_DIR, "pjm_hourly_demand_2019_eia.csv")
    if os.path.exists(existing_2019):
        print("\nMerging with existing 2019 data...")
        df_2019 = pd.read_csv(existing_2019)
        output = pd.concat([df_2019, output], ignore_index=True)
    
    output = output.sort_values("datetime_utc").reset_index(drop=True)
    
    # Save combined file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "pjm_hourly_demand_2019_2024_eia.csv")
    output.to_csv(out_path, index=False)
    
    print(f"\n{'=' * 60}")
    print(f"Saved to: {out_path}")
    print(f"Total records: {len(output)}")
    print(f"Date range:    {output['datetime_utc'].min()} → {output['datetime_utc'].max()}")
    print(f"Demand range:  {output['demand_mwh'].min():,.0f} – {output['demand_mwh'].max():,.0f} MWh")
    print(f"Mean demand:   {output['demand_mwh'].mean():,.0f} MWh")
    print(f"File size:     {os.path.getsize(out_path) / 1024:.1f} KB")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
