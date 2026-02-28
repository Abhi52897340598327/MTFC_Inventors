# Data Audit Summary

- Scanned files: 58
- CSV files: 39
- CSV files with quality flags: 8
- File types: {"csv": 39, "gz": 3, "html": 1, "ipynb": 1, "json": 2, "md": 4, "pdf": 1, "sh": 1, "txt": 3, "xlsx": 3}

## Key Model Data Picks
- stage1_cpu_primary: `cleaned/google_cluster_utilization_2019_cellb_hourly_cleaned.csv` (exists=True, rows=745, flags=)
- stage3_temperature_primary: `cleaned/ashburn_va_temperature_2019_cleaned.csv` (exists=True, rows=8760, flags=)
- stage5_carbon_primary: `cleaned/pjm_grid_carbon_intensity_2019_full_cleaned.csv` (exists=True, rows=52443, flags=)
- power_optional_feature: `cleaned/google_power_utilization_2019_cellb_hourly.csv` (exists=True, rows=743, flags=)
- joined_cpu_power_optional: `cleaned/google_cluster_plus_power_2019_cellb_hourly.csv` (exists=True, rows=743, flags=)
- stage5_grid_exog_optional: `cleaned/pjm_exogenous_hourly_2019_2024_cleaned.csv` (exists=True, rows=52561, flags=)
- stage3_weather_exog_optional: `cleaned/noaa_global_hourly_dulles_2019_2024_cleaned.csv` (exists=True, rows=52608, flags=)
- synthetic_reference_only: `cleaned/semisynthetic_datacenter_power_2015_2024.csv` (exists=True, rows=87672, flags=synthetic_data)

## Copied To model_ready
- `model_ready\stage1_cpu_primary.csv`
- `model_ready\stage3_temperature_primary.csv`
- `model_ready\stage5_carbon_primary.csv`
- `model_ready\power_optional_feature.csv`
- `model_ready\joined_cpu_power_optional.csv`
- `model_ready\stage5_grid_exog_optional.csv`
- `model_ready\stage3_weather_exog_optional.csv`
- `model_ready\synthetic_reference_only.csv`

## Unified Training Tables
- `model_ready/final_training_table_2019_hourly.csv` rows=745
- `model_ready/final_training_table_2019_hourly_with_power.csv` rows=745

## Suggested Cleanup Actions
- Exclude files flagged `synthetic_data` from headline model training/evaluation.
- Exclude files flagged `test_artifact` and `intermediate_export` from production ingestion.
- Prefer UTF-8 files under `cleaned/` or `model_ready/` for pipeline reads.