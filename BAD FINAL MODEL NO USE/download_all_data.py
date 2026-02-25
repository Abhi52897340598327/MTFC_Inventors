"""
Comprehensive data download script for MTFC Datacenter Energy Forecasting.

Downloads from:
  1. BigQuery  — Google Cluster Trace 2019 (hourly CPU/memory workload profiles)
  2. BigQuery  — NOAA GSOD weather data for Dulles VA (2015-2024)
  3. EIA API   — Hourly grid generation by fuel type for PJM (2019-2024)
  4. EIA API   — Hourly grid carbon intensity / emissions for PJM (2019-2024)

All files are saved to ../Data_Sources/
"""

import os, sys, json, time
import requests
import pandas as pd
import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────
EIA_API_KEY  = "NRjspMDoZtvn3rjwucZ3FbYhgFLWhmAsLPrriyig"
GCP_PROJECT  = "datacenter-forecasting"
OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data_Sources")

# NOAA station for Ashburn/Loudoun County VA area
DULLES_STATION = "724030"   # USAF code for Washington Dulles Intl Airport
DULLES_WBAN   = "93738"     # WBAN code

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
#  1. Google Cluster Trace 2019  (BigQuery)
# ════════════════════════════════════════════════════════════════════════════
def download_cluster_trace():
    """
    Query the Google Cluster Trace 2019 dataset in BigQuery.
    Aggregates CPU and memory usage to hourly resolution across all 8 clusters.
    This is a ~2.4 TB dataset; the aggregation query processes ~50-100 GB.
    """
    print("\n" + "=" * 70)
    print("  1. GOOGLE CLUSTER TRACE 2019 (BigQuery)")
    print("=" * 70)

    try:
        from google.cloud import bigquery
    except ImportError:
        print("  ⚠ google-cloud-bigquery not installed. Run:")
        print("    pip install google-cloud-bigquery")
        return

    client = bigquery.Client(project=GCP_PROJECT)

    # Query: aggregate instance-level CPU/memory usage to hourly buckets
    # The trace covers May 2019 (31 days, ~744 hours)
    # start_time and end_time are in microseconds since epoch
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
        APPROX_QUANTILES(average_usage.cpus, 100)[OFFSET(95)] AS p95_cpu_usage,
        APPROX_QUANTILES(average_usage.cpus, 100)[OFFSET(50)] AS p50_cpu_usage,
        COUNT(*) AS instance_count
    FROM `google.com:google-cluster-data.clusterdata_2019_a.instance_usage`
    GROUP BY hour_ts
    ORDER BY hour_ts
    """

    print("  Running BigQuery aggregation (this may take 1-3 minutes)...")
    print("  ⚠ This query processes ~50-100 GB of data from the cluster trace.")

    try:
        df = client.query(query).to_dataframe()
        out_path = os.path.join(OUTPUT_DIR, "google_cluster_trace_hourly_2019.csv")
        df.to_csv(out_path, index=False)
        print(f"  ✓ Saved: {out_path}")
        print(f"    Rows: {len(df)}, Date range: {df.hour_ts.min()} → {df.hour_ts.max()}")
        print(f"    Avg CPU utilization: {df.avg_cpu_usage.mean():.4f}")
        print(f"    Active machines range: {df.active_machines.min()} – {df.active_machines.max()}")
    except Exception as e:
        print(f"  ✗ BigQuery error: {e}")
        print("  → Check your GCP project permissions and billing.")
        print("  → The query reads from google.com:google-cluster-data project.")


# ════════════════════════════════════════════════════════════════════════════
#  2. NOAA GSOD Weather — Dulles VA (BigQuery)
# ════════════════════════════════════════════════════════════════════════════
def download_noaa_weather():
    """
    Query NOAA Global Surface Summary of the Day from BigQuery.
    Gets daily weather for Dulles Airport (near Ashburn datacenter alley)
    for 2015-2024 — much broader range than the existing 2019 CSV.
    """
    print("\n" + "=" * 70)
    print("  2. NOAA GSOD WEATHER — Dulles VA 2015-2024 (BigQuery)")
    print("=" * 70)

    try:
        from google.cloud import bigquery
    except ImportError:
        print("  ⚠ google-cloud-bigquery not installed.")
        return

    client = bigquery.Client(project=GCP_PROJECT)

    # GSOD data is partitioned by year in tables named gsodYYYY
    # We query across multiple years using a wildcard or UNION
    query = """
    SELECT
        CONCAT(year, '-', mo, '-', da) AS date,
        CAST(year AS INT64) AS year,
        CAST(mo AS INT64) AS month,
        CAST(da AS INT64) AS day,
        temp AS avg_temp_f,
        max AS max_temp_f,
        min AS min_temp_f,
        dewp AS dewpoint_f,
        slp AS sea_level_pressure_mb,
        stp AS station_pressure_mb,
        visib AS visibility_miles,
        wdsp AS avg_wind_speed_knots,
        mxspd AS max_wind_speed_knots,
        gust AS max_wind_gust_knots,
        prcp AS precipitation_inches,
        sndp AS snow_depth_inches,
        fog, rain_drizzle, snow_ice_pellets, hail, thunder, tornado_funnel_cloud
    FROM `bigquery-public-data.noaa_gsod.gsod*`
    WHERE
        stn = @station_id
        AND CAST(year AS INT64) BETWEEN 2015 AND 2024
    ORDER BY year, mo, da
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("station_id", "STRING", DULLES_STATION),
        ]
    )

    print(f"  Querying NOAA GSOD for station {DULLES_STATION} (Dulles Intl)...")

    try:
        df = client.query(query, job_config=job_config).to_dataframe()

        # Replace 9999.9 / 999.9 / 99.99 sentinel values with NaN
        sentinel_cols = {
            'avg_temp_f': 9999.9, 'max_temp_f': 9999.9, 'min_temp_f': 9999.9,
            'dewpoint_f': 9999.9, 'sea_level_pressure_mb': 9999.9,
            'station_pressure_mb': 9999.9, 'visibility_miles': 999.9,
            'avg_wind_speed_knots': 999.9, 'max_wind_speed_knots': 999.9,
            'max_wind_gust_knots': 999.9, 'precipitation_inches': 99.99,
            'snow_depth_inches': 999.9,
        }
        for col, sentinel in sentinel_cols.items():
            if col in df.columns:
                df[col] = df[col].replace(sentinel, np.nan)

        out_path = os.path.join(OUTPUT_DIR, "noaa_gsod_dulles_2015_2024.csv")
        df.to_csv(out_path, index=False)
        print(f"  ✓ Saved: {out_path}")
        print(f"    Rows: {len(df)}, Years: {df.year.min()} – {df.year.max()}")
        print(f"    Avg temp range: {df.avg_temp_f.min():.1f}°F – {df.avg_temp_f.max():.1f}°F")
        per_year = df.groupby('year').size()
        print(f"    Per year: {dict(per_year)}")
    except Exception as e:
        print(f"  ✗ BigQuery error: {e}")


# ════════════════════════════════════════════════════════════════════════════
#  3. EIA API — PJM Grid Generation by Fuel Type
# ════════════════════════════════════════════════════════════════════════════
def _eia_fetch(endpoint, params, max_records=None):
    """Helper: paginated fetch from EIA API v2."""
    base_url = f"https://api.eia.gov/v2/{endpoint}"
    params["api_key"] = EIA_API_KEY
    params.setdefault("length", 5000)

    all_records = []
    offset = 0

    while True:
        params["offset"] = offset
        resp = requests.get(base_url, params=params, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        total = int(data["response"]["total"])
        records = data["response"]["data"]
        all_records.extend(records)
        print(f"    Fetched {len(records)} rows (offset {offset}, total available: {total})")
        if len(all_records) >= total or not records:
            break
        if max_records and len(all_records) >= max_records:
            break
        offset = len(all_records)
        time.sleep(0.5)  # rate limit courtesy

    return pd.DataFrame(all_records)


def download_eia_generation():
    """
    Download PJM hourly net generation by energy source (fuel type).
    Useful for computing real-time grid carbon intensity.
    """
    print("\n" + "=" * 70)
    print("  3. EIA — PJM HOURLY GENERATION BY FUEL TYPE (2019-2024)")
    print("=" * 70)

    all_dfs = []
    fuel_types = ["NG", "COL", "NUC", "WND", "SUN", "WAT", "OTH"]
    # NG=Natural Gas, COL=Coal, NUC=Nuclear, WND=Wind, SUN=Solar, WAT=Hydro

    for fuel in fuel_types:
        print(f"\n  Downloading {fuel}...")
        for start_year, end_year in [(2019, 2021), (2022, 2024)]:
            params = {
                "frequency": "hourly",
                "data[0]": "value",
                "facets[respondent][]": "PJM",
                "facets[fueltype][]": fuel,
                "start": f"{start_year}-01-01T00",
                "end": f"{end_year}-12-31T23",
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
            }
            try:
                df = _eia_fetch("electricity/rto/fuel-type-data/data/", params)
                if not df.empty:
                    all_dfs.append(df)
            except Exception as e:
                print(f"    ✗ Error fetching {fuel} {start_year}-{end_year}: {e}")
            time.sleep(1)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined["value"] = pd.to_numeric(combined["value"], errors="coerce")
        out_path = os.path.join(OUTPUT_DIR, "pjm_generation_by_fuel_2019_2024_eia.csv")
        combined.to_csv(out_path, index=False)
        print(f"\n  ✓ Saved: {out_path}")
        print(f"    Total rows: {len(combined)}")
        if "fueltype" in combined.columns:
            print(f"    Fuel types: {combined['fueltype'].unique().tolist()}")
    else:
        print("  ✗ No generation data retrieved.")


def download_eia_interchange():
    """
    Download PJM hourly interchange (power imports/exports with neighbors).
    Useful for understanding grid stress and net power balance.
    """
    print("\n" + "=" * 70)
    print("  4. EIA — PJM HOURLY INTERCHANGE (2019-2024)")
    print("=" * 70)

    all_dfs = []
    for start_year, end_year in [(2019, 2021), (2022, 2024)]:
        print(f"\n  Downloading {start_year}-{end_year}...")
        params = {
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": "PJM",
            "facets[type][]": "TI",   # TI = Total Interchange
            "start": f"{start_year}-01-01T00",
            "end": f"{end_year}-12-31T23",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
        }
        try:
            df = _eia_fetch("electricity/rto/region-data/data/", params)
            if not df.empty:
                all_dfs.append(df)
        except Exception as e:
            print(f"    ✗ Error: {e}")
        time.sleep(1)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined["value"] = pd.to_numeric(combined["value"], errors="coerce")
        out_path = os.path.join(OUTPUT_DIR, "pjm_interchange_2019_2024_eia.csv")
        combined.to_csv(out_path, index=False)
        print(f"\n  ✓ Saved: {out_path}")
        print(f"    Total rows: {len(combined)}")


def download_eia_demand_forecast():
    """
    Download PJM day-ahead demand forecast (2019-2024).
    Comparing forecast vs actual is useful for grid stress analysis.
    """
    print("\n" + "=" * 70)
    print("  5. EIA — PJM HOURLY DEMAND FORECAST (2019-2024)")
    print("=" * 70)

    all_dfs = []
    for start_year, end_year in [(2019, 2021), (2022, 2024)]:
        print(f"\n  Downloading {start_year}-{end_year}...")
        params = {
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": "PJM",
            "facets[type][]": "DF",   # DF = Demand Forecast
            "start": f"{start_year}-01-01T00",
            "end": f"{end_year}-12-31T23",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
        }
        try:
            df = _eia_fetch("electricity/rto/region-data/data/", params)
            if not df.empty:
                all_dfs.append(df)
        except Exception as e:
            print(f"    ✗ Error: {e}")
        time.sleep(1)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined["value"] = pd.to_numeric(combined["value"], errors="coerce")
        out_path = os.path.join(OUTPUT_DIR, "pjm_demand_forecast_2019_2024_eia.csv")
        combined.to_csv(out_path, index=False)
        print(f"\n  ✓ Saved: {out_path}")
        print(f"    Total rows: {len(combined)}")


def download_eia_net_generation():
    """
    Download PJM hourly net generation (total) — 2019-2024.
    """
    print("\n" + "=" * 70)
    print("  6. EIA — PJM HOURLY NET GENERATION (2019-2024)")
    print("=" * 70)

    all_dfs = []
    for start_year, end_year in [(2019, 2021), (2022, 2024)]:
        print(f"\n  Downloading {start_year}-{end_year}...")
        params = {
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": "PJM",
            "facets[type][]": "NG",   # NG = Net Generation
            "start": f"{start_year}-01-01T00",
            "end": f"{end_year}-12-31T23",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
        }
        try:
            df = _eia_fetch("electricity/rto/region-data/data/", params)
            if not df.empty:
                all_dfs.append(df)
        except Exception as e:
            print(f"    ✗ Error: {e}")
        time.sleep(1)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined["value"] = pd.to_numeric(combined["value"], errors="coerce")
        out_path = os.path.join(OUTPUT_DIR, "pjm_net_generation_2019_2024_eia.csv")
        combined.to_csv(out_path, index=False)
        print(f"\n  ✓ Saved: {out_path}")
        print(f"    Total rows: {len(combined)}")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    print("╔" + "═" * 68 + "╗")
    print("║  MTFC Datacenter Energy Forecasting — Data Download Script        ║")
    print("║  Downloading from: BigQuery (Cluster Trace, NOAA) + EIA API       ║")
    print("╚" + "═" * 68 + "╝")
    print(f"\nOutput directory: {OUTPUT_DIR}\n")

    # Parse command-line flags
    run_all = "--all" in sys.argv
    args = set(sys.argv[1:])

    tasks = {
        "--cluster":    ("Google Cluster Trace",     download_cluster_trace),
        "--weather":    ("NOAA GSOD Weather",        download_noaa_weather),
        "--generation": ("EIA Generation by Fuel",   download_eia_generation),
        "--interchange":("EIA Interchange",          download_eia_interchange),
        "--forecast":   ("EIA Demand Forecast",      download_eia_demand_forecast),
        "--netgen":     ("EIA Net Generation",       download_eia_net_generation),
    }

    if not run_all and not args.intersection(tasks.keys()):
        print("Usage: python download_all_data.py [OPTIONS]")
        print()
        print("Options:")
        print("  --all           Download everything")
        print("  --cluster       Google Cluster Trace 2019 (BigQuery; ~50-100 GB query)")
        print("  --weather       NOAA GSOD daily weather 2015-2024 (BigQuery)")
        print("  --generation    PJM hourly generation by fuel type (EIA API)")
        print("  --interchange   PJM hourly interchange/imports (EIA API)")
        print("  --forecast      PJM hourly demand forecast (EIA API)")
        print("  --netgen        PJM hourly net generation (EIA API)")
        print()
        print("Examples:")
        print("  python download_all_data.py --all")
        print("  python download_all_data.py --weather --generation")
        return

    for flag, (name, func) in tasks.items():
        if run_all or flag in args:
            try:
                func()
            except Exception as e:
                print(f"\n  ✗ FAILED: {name} — {e}")

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║  Download complete!                                               ║")
    print("╚" + "═" * 68 + "╝")

    # Print summary of files in Data_Sources
    print("\nFiles in Data_Sources:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fp = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(fp):
            size_mb = os.path.getsize(fp) / (1024 * 1024)
            print(f"  {f:55s} {size_mb:8.2f} MB")


if __name__ == "__main__":
    main()
