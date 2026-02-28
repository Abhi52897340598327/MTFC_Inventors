#!/usr/bin/env python
"""Copy all currently-usable model CSVs into NEW MODEL DATA with path preservation."""

from __future__ import annotations

from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]
DEST_ROOT = ROOT / "NEW MODEL DATA"

USABLE_NOW = [
    "Data_Sources/ashburn_va_temperature_2019.csv",
    "Data_Sources/cleaned/ashburn_va_temperature_2019_cleaned.csv",
    "Data_Sources/cleaned/google_cluster_plus_power_2019_cellb_hourly.csv",
    "Data_Sources/cleaned/google_cluster_utilization_2019_cellb_hourly_cleaned.csv",
    "Data_Sources/cleaned/google_power_utilization_2019_cellb_hourly.csv",
    "Data_Sources/cleaned/noaa_global_hourly_dulles_2019_2024_cleaned.csv",
    "Data_Sources/cleaned/pjm_exogenous_hourly_2019_2024_cleaned.csv",
    "Data_Sources/cleaned/pjm_grid_carbon_intensity_2019_full_cleaned.csv",
    "Data_Sources/cleaned/pjm_hourly_demand_2019_2024_cleaned.csv",
    "Data_Sources/external_downloads/google_cluster_cpu_cellb_hourly_2019.csv",
    "Data_Sources/external_downloads/google_power_cellb_hourly_2019.csv",
    "Data_Sources/google_cluster_utilization_2019.csv",
    "Data_Sources/gpu-price-performance.csv",
    "Data_Sources/hardware-and-energy-cost-to-train-notable-ai-systems.csv",
    "Data_Sources/model_ready/final_training_table_2019_hourly.csv",
    "Data_Sources/model_ready/final_training_table_2019_hourly_with_power.csv",
    "Data_Sources/model_ready/joined_cpu_power_optional.csv",
    "Data_Sources/model_ready/power_optional_feature.csv",
    "Data_Sources/model_ready/stage1_cpu_primary.csv",
    "Data_Sources/model_ready/stage3_temperature_primary.csv",
    "Data_Sources/model_ready/stage3_weather_exog_optional.csv",
    "Data_Sources/model_ready/stage5_carbon_primary.csv",
    "Data_Sources/model_ready/stage5_grid_exog_optional.csv",
    "Data_Sources/monthly-spending-data-center-us.csv",
    "Data_Sources/noaa_dulles_daily_2015_2024.csv",
    "Data_Sources/noaa_gsod_dulles_2019.csv",
    "Data_Sources/pjm_demand_forecast_2019_2024_eia.csv",
    "Data_Sources/pjm_generation_by_fuel_2019_2024_eia.csv",
    "Data_Sources/pjm_hourly_demand_2019_2024_eia.csv",
    "Data_Sources/pjm_interchange_2019_2024_eia.csv",
    "Data_Sources/pjm_net_generation_2019_2024_eia.csv",
    "Data_Sources/share_companies_using_ai_owid_raw.csv",
]


def main() -> int:
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing = 0
    for rel in USABLE_NOW:
        src = ROOT / rel
        if not src.exists():
            print(f"MISSING: {rel}")
            missing += 1
            continue
        dst = DEST_ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    print(f"Copied={copied}, Missing={missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

