
import pandas as pd
import numpy as np
import data_loader
import feature_engineering
import data_preparation
import config as cfg

print("DEBUG: Testing pipeline steps...")

try:
    # 1. LOAD
    print("\n--- 1. Data Loader ---")
    df = data_loader.merge_hourly_datasets()
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    if "temperature_f" in df.columns:
        print("PASS: 'temperature_f' present.")
    else:
        print("FAIL: 'temperature_f' MISSING.")

    # 2. FEATURE ENGINEERING
    print("\n--- 2. Feature Engineering ---")
    df_feat = feature_engineering.engineer_features(df)
    print(f"Shape: {df_feat.shape}")
    print(f"Columns: {df_feat.columns.tolist()}")
    
    expected = ["hour", "day_of_week", "temperature_f", "temp_x_hour"]
    for c in expected:
        if c in df_feat.columns:
             print(f"PASS: '{c}' found.")
        else:
             print(f"FAIL: '{c}' NOT found.")
             
    # 3. DATA PREPARATION (Split)
    print("\n--- 3. Data Prep (Split) ---")
    prep = data_preparation.prepare_all(df_feat)
    train = prep["train"]
    print(f"Train Shape: {train.shape}")
    print(f"Train Columns: {train.columns.tolist()}")
    
    # 4. FORECAST COLS
    print("\n--- 4. Forecast Cols ---")
    fc_cols = data_preparation.get_forecast_feature_cols(df_feat)
    print(f"Forecast Cols: {fc_cols}")
    
    # Check if all forecast cols exist in train
    missing = [c for c in fc_cols if c not in train.columns]
    if missing:
        print(f"FAIL: The following forecast cols are missing from Train: {missing}")
    else:
        print("PASS: All forecast cols present in Train.")

except Exception as e:
    print(f"\nCRITICAL FAILURE: {e}")
    import traceback
    traceback.print_exc()
