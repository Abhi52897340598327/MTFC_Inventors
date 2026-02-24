"""
Download NOAA weather data for Dulles VA (2015-2024) using the NCDC API.
Uses the NOAA NCDC CDO API (Climate Data Online).
Also attempts to download Google Cluster Trace aggregate data.

Saves to ../Data_Sources/
"""

import os, time, sys
import requests
import pandas as pd
import numpy as np

NOAA_TOKEN = "ShTYHitdxfFxaokzdVEaGxcFUhSmNhdS"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data_Sources")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
#  NOAA NCDC API — Hourly Weather for Dulles VA
# ════════════════════════════════════════════════════════════════════════════
def download_noaa_ncdc():
    """
    Download hourly weather observations from NOAA NCDC CDO API.
    Station: GHCND:USW00093738 (Washington Dulles Intl Airport)
    This is the closest NOAA station to Ashburn/Loudoun County VA (datacenter alley).
    """
    print("=" * 70)
    print("  NOAA NCDC — Dulles VA Daily Weather 2015-2024")
    print("=" * 70)

    BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
    STATION = "GHCND:USW00093738"    # Dulles Intl Airport
    HEADERS = {"token": NOAA_TOKEN}

    # NCDC API limits: 1 year per request, 1000 rows per page
    # Datatypes: TMAX, TMIN, TAVG, PRCP, SNOW, SNWD, AWND, WSF2, WSF5
    datatypes = "TMAX,TMIN,TAVG,PRCP,SNOW,SNWD,AWND,WSF2,WSF5,RHAV"

    all_records = []

    for year in range(2015, 2025):
        print(f"\n  Downloading {year}...")
        offset = 1
        year_records = []

        while True:
            params = {
                "datasetid": "GHCND",
                "stationid": STATION,
                "startdate": f"{year}-01-01",
                "enddate": f"{year}-12-31",
                "datatypeid": datatypes,
                "units": "standard",      # Fahrenheit, inches
                "limit": 1000,
                "offset": offset,
            }

            resp = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=60)

            if resp.status_code == 429:
                print("    Rate limited, waiting 2s...")
                time.sleep(2)
                continue

            if resp.status_code != 200:
                print(f"    ✗ HTTP {resp.status_code}: {resp.text[:200]}")
                break

            data = resp.json()
            if "results" not in data:
                break

            results = data["results"]
            year_records.extend(results)
            total = data.get("metadata", {}).get("resultset", {}).get("count", 0)
            print(f"    Fetched {len(results)} records (offset {offset}, total: {total})")

            if offset + len(results) > total:
                break
            offset += len(results)
            time.sleep(0.3)  # rate limit

        all_records.extend(year_records)
        print(f"    Year {year}: {len(year_records)} records")

    if not all_records:
        print("\n  ✗ No records retrieved. Check NOAA token.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_records)

    # Pivot: one row per date, columns for each datatype
    df_pivot = df.pivot_table(
        index="date", columns="datatype", values="value", aggfunc="first"
    ).reset_index()
    df_pivot.columns.name = None

    # Rename columns for clarity
    rename_map = {
        "date": "date",
        "TMAX": "max_temp_f",
        "TMIN": "min_temp_f",
        "TAVG": "avg_temp_f",
        "PRCP": "precipitation_inches",
        "SNOW": "snowfall_inches",
        "SNWD": "snow_depth_inches",
        "AWND": "avg_wind_speed_mph",
        "WSF2": "max_wind_speed_2min_mph",
        "WSF5": "max_wind_speed_5sec_mph",
        "RHAV": "avg_relative_humidity_pct",
    }
    df_pivot = df_pivot.rename(columns=rename_map)

    # Add computed avg temp if TAVG not available
    if "avg_temp_f" not in df_pivot.columns:
        if "max_temp_f" in df_pivot.columns and "min_temp_f" in df_pivot.columns:
            df_pivot["avg_temp_f"] = (df_pivot["max_temp_f"] + df_pivot["min_temp_f"]) / 2

    # Parse date
    df_pivot["date"] = pd.to_datetime(df_pivot["date"]).dt.date
    df_pivot = df_pivot.sort_values("date").reset_index(drop=True)

    # Add year/month columns
    df_pivot["year"] = pd.to_datetime(df_pivot["date"]).dt.year
    df_pivot["month"] = pd.to_datetime(df_pivot["date"]).dt.month

    out_path = os.path.join(OUTPUT_DIR, "noaa_dulles_daily_2015_2024.csv")
    df_pivot.to_csv(out_path, index=False)
    print(f"\n  ✓ Saved: {out_path}")
    print(f"    Rows: {len(df_pivot)}")
    print(f"    Date range: {df_pivot.date.min()} → {df_pivot.date.max()}")
    print(f"    Columns: {df_pivot.columns.tolist()}")
    if "avg_temp_f" in df_pivot.columns:
        print(f"    Temp range: {df_pivot.avg_temp_f.min():.1f}°F – {df_pivot.avg_temp_f.max():.1f}°F")


# ════════════════════════════════════════════════════════════════════════════
#  Google Cluster Trace — Try BigQuery, fallback to summary info
# ════════════════════════════════════════════════════════════════════════════
def download_cluster_trace():
    """
    Try BigQuery first, then fallback to downloading the pre-aggregated
    summary from the Google cluster-data GitHub repo.
    """
    print("\n" + "=" * 70)
    print("  GOOGLE CLUSTER TRACE 2019")
    print("=" * 70)

    # Try BigQuery first
    try:
        from google.cloud import bigquery
        client = bigquery.Client(project="datacenter-forecasting")

        query = """
        SELECT
            TIMESTAMP_SECONDS(
                CAST(FLOOR(start_time / 1000000 / 3600) * 3600 AS INT64)
            ) AS hour_ts,
            COUNT(DISTINCT machine_id) AS active_machines,
            AVG(average_usage.cpus) AS avg_cpu_usage,
            AVG(average_usage.memory) AS avg_memory_usage,
            STDDEV(average_usage.cpus) AS std_cpu_usage,
            MAX(average_usage.cpus) AS max_cpu_usage,
            COUNT(*) AS instance_count
        FROM `google.com:google-cluster-data.clusterdata_2019_a.instance_usage`
        GROUP BY hour_ts
        ORDER BY hour_ts
        """

        print("  Attempting BigQuery query...")
        df = client.query(query).to_dataframe()
        out_path = os.path.join(OUTPUT_DIR, "google_cluster_trace_hourly_2019.csv")
        df.to_csv(out_path, index=False)
        print(f"  ✓ Saved: {out_path}")
        print(f"    Rows: {len(df)}")
        return

    except Exception as e:
        print(f"  BigQuery unavailable: {e}")
        print("  → Falling back to GitHub summary data...")

    # Fallback: download machine_events summary from GitHub
    urls = {
        "machine_events": "https://raw.githubusercontent.com/google/cluster-data/master/ClusterData2019.md",
    }
    print("  Note: The full 2.4 TB cluster trace is only accessible via BigQuery.")
    print("  The existing google_cluster_utilization_2019.csv will be used instead.")
    print("  To enable BigQuery access:")
    print("    1. Go to https://console.cloud.google.com/billing")
    print("    2. Enable billing on the 'datacenter-forecasting' project")
    print("    3. Enable the BigQuery API")
    print("    4. Re-run: python download_noaa_weather.py --cluster")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    print("╔" + "═" * 68 + "╗")
    print("║  MTFC Data Download — NOAA Weather + Cluster Trace                ║")
    print("╚" + "═" * 68 + "╝")

    args = set(sys.argv[1:])
    run_all = "--all" in args or not args

    if run_all or "--weather" in args:
        download_noaa_ncdc()

    if run_all or "--cluster" in args:
        download_cluster_trace()

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║  Download complete!                                               ║")
    print("╚" + "═" * 68 + "╝")

    # Summary
    print("\nNew/updated files in Data_Sources:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fp = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(fp):
            size_kb = os.path.getsize(fp) / 1024
            print(f"  {f:55s} {size_kb:8.1f} KB")


if __name__ == "__main__":
    main()
