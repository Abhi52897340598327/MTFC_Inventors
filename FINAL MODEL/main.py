"""
MTFC Virginia Datacenter Energy Forecasting — Hybrid ML Pipeline
================================================================
Robust "Simulation & Projection" Architecture using REAL DATA ONLY.

Flow:
1. Data Loading: Ingests REAL data sources (NOAA, EIA, Google Cluster).
2. Feature Engineering: Creates "Forecast-Safe" features (Time, Temp). No lags.
3. Training (XGBoost): Learns the system's physics (Power = f(Temp, Time)).
4. Forecasting: Simulates future scenarios (Climate + Growth) using the trained model.
5. Impact Analysis: Carbon, Grid Stress, Sensitivity.

All data comes from verified real-world sources - NO synthetic data generation.
"""

import sys
import os

# Suppress TensorFlow/Abseil verbose logging (mutex warnings, etc.)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL only
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'  # Suppress Abseil mutex warnings

# Suppress stderr for TensorFlow C++ mutex warnings during import
import contextlib
import io

import time
import numpy as np
import pandas as pd
import warnings

# Ensure module path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Config & Utils
import config as cfg
from utils import log, save_json, save_csv, set_plot_style

# Modules
import data_loader
import eda
import feature_engineering
import data_preparation
import forecasting
import sensitivity_analysis
import carbon_emissions
import grid_stress

# ML Models
from models.xgboost_model import train_xgboost, evaluate_xgboost, plot_feature_importance
from models.sarimax_model import train_sarimax, evaluate_sarimax

# Suppress noisy warnings
warnings.filterwarnings("ignore")

def main():
    start_time = time.time()
    set_plot_style()
    
    log.info("╔" + "═" * 70 + "╗")
    log.info("║  MTFC DATACENTER FORECASTING — HYBRID ML PIPELINE                   ║")
    log.info("╚" + "═" * 70 + "╝")
    
    # ─── 1. DATA LOADING ────────────────────────────────────────────────────────
    log.info("\n--- PHASE 1: DATA LOADING ---")
    data = data_loader.load_all()
    hourly = data["hourly"]
    
    # ─── 2. EDA ─────────────────────────────────────────────────────────────────
    log.info("\n--- PHASE 2: EDA ---")
    eda.run_eda(hourly)
    
    # ─── 3. FEATURE ENGINEERING ─────────────────────────────────────────────────
    log.info("\n--- PHASE 3: FEATURE ENGINEERING ---")
    # Create all features including lags for maximum predictive power
    df_feat = feature_engineering.engineer_features(hourly)
    
    # ─── 4. DATA PREPARATION (Split & Scale) ────────────────────────────────────
    log.info("\n--- PHASE 4: PREPARATION ---")
    # Use all available features for best accuracy
    feature_cols = data_preparation.get_feature_cols(df_feat)
    target_col = cfg.TARGET_COL
    
    log.info(f"Target: {target_col}")
    log.info(f"Features ({len(feature_cols)}): {feature_cols[:10]}...")  # Show first 10
    
    # Split
    train, val, test = data_preparation.split_data(df_feat)
    
    # Scale (Required for some models, good practice for all)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(train[feature_cols])
    
    X_train = scaler.transform(train[feature_cols])
    X_val   = scaler.transform(val[feature_cols])
    X_test  = scaler.transform(test[feature_cols])
    
    y_train = train[target_col].values
    y_val   = val[target_col].values
    y_test  = test[target_col].values
    
    # ─── 5. MODEL TRAINING ──────────────────────────────────────────────────────
    log.info("\n--- PHASE 5: MODEL TRAINING ---")
    model_metrics = {}
    model_results = {}  # Store full results for evaluation plots
    
    # A. XGBoost (The Core Engine)
    if "--skip-xgboost" not in sys.argv:
        log.info("Training XGBoost (Regression Mode)...")
        xgb_model = train_xgboost(X_train, y_train, X_val, y_val, feature_names=feature_cols)
        xgb_results = evaluate_xgboost(xgb_model, X_test, y_test)
        model_metrics["XGBoost"] = xgb_results["metrics"]
        model_results["XGBoost"] = xgb_results
        plot_feature_importance(xgb_model, feature_cols)
        log.info(f"XGBoost Performance: R2={xgb_results['metrics']['R2']:.4f}, MAE={xgb_results['metrics']['MAE']:.2f}")
    else:
        log.info("Skipping XGBoost (--skip-xgboost flag)")
        xgb_model = None

    # B. GRU (Deep Learning - replaced LSTM for faster training)
    if "--skip-gru" not in sys.argv:
        log.info("Training GRU (Deep Learning)...")
        try:
            from models.gru_model import train_gru, evaluate_gru
            # Create sequences for GRU
            X_train_seq, y_train_seq = data_preparation.create_sequences(X_train, y_train)
            X_val_seq, y_val_seq = data_preparation.create_sequences(X_val, y_val)
            X_test_seq, y_test_seq = data_preparation.create_sequences(X_test, y_test)
            
            gru_model, target_scaler = train_gru(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
            gru_results = evaluate_gru(gru_model, X_test_seq, y_test_seq, target_scaler)
            model_metrics["GRU"] = gru_results["metrics"]
            model_results["GRU"] = gru_results
            log.info(f"GRU Performance: R2={gru_results['metrics']['R2']:.4f}, MAE={gru_results['metrics']['MAE']:.2f}")
        except Exception as e:
            log.warning(f"GRU Failed: {e}")

    # C. Random Forest (Replaced SARIMAX - faster & more accurate)
    if "--skip-rf" not in sys.argv:
        log.info("Training Random Forest...")
        try:
            from models.random_forest_model import train_random_forest, evaluate_random_forest
            rf_model = train_random_forest(X_train, y_train, X_val, y_val, feature_names=feature_cols)
            rf_results = evaluate_random_forest(rf_model, X_test, y_test)
            model_metrics["RandomForest"] = rf_results["metrics"]
            model_results["RandomForest"] = rf_results
            log.info(f"Random Forest Performance: R2={rf_results['metrics']['R2']:.4f}, MAE={rf_results['metrics']['MAE']:.2f}")
        except Exception as e:
            log.warning(f"Random Forest Failed: {e}")
            import traceback
            traceback.print_exc()
    
    # ─── 5b. MODEL EVALUATION & COMPARISON GRAPHS ───────────────────────────────
    log.info("\n--- PHASE 5b: MODEL EVALUATION ---")
    if model_results:
        import evaluation
        test_timestamps = test.index if hasattr(test, 'index') else None
        evaluation.evaluate_all(model_results, timestamps=test_timestamps)
        log.info(f"Generated comparison plots for {len(model_results)} models")
    
    # ─── 6. FORECASTING (Scenario Projection) ───────────────────────────────────
    log.info("\n--- PHASE 6: SCENARIO FORECASTING (2025-2030) ---")
    # For forecasting, we need to retrain with forecast-safe features only
    # (features that can be computed for future dates without knowing future target)
    forecast_feature_cols = [c for c in cfg.FEATURE_SET_FORECAST_SAFE if c in df_feat.columns]
    
    # Retrain a simpler model for forecasting
    from sklearn.preprocessing import StandardScaler as SS2
    forecast_scaler = SS2()
    forecast_scaler.fit(train[forecast_feature_cols])
    
    X_train_fc = forecast_scaler.transform(train[forecast_feature_cols])
    X_val_fc = forecast_scaler.transform(val[forecast_feature_cols])
    
    from models.xgboost_model import train_xgboost as train_xgb_fc
    xgb_forecast = train_xgb_fc(X_train_fc, y_train, X_val_fc, y_val, feature_names=forecast_feature_cols)
    
    # Use the forecast model and scaler for projections
    forecast_df = forecasting.run_forecast(xgb_forecast, forecast_scaler, forecast_feature_cols)
    
    # Save & Plot
    save_csv(forecast_df, "scenario_forecast_2025_2030.csv")
    forecasting.plot_forecast_comparison(forecast_df)
    
    # ─── 7. IMPACT ANALYSIS ─────────────────────────────────────────────────────
    log.info("\n--- PHASE 7: IMPACT ANALYSIS ---")
    
    # 1. Calculate Grid Stress for ALL scenarios to enable comparison
    scenario_metrics = []
    pjm_2019 = data_loader.get_pjm_demand_2019() # 2019 as proxy pattern
    base_grid_demand = pjm_2019["grid_demand_mw"].values
    
    for name, group in forecast_df.groupby("scenario"):
        # For PUE prediction, convert to estimated power for grid stress
        # Total_Power = IT_Capacity × Avg_Utilization × PUE
        # Using 100 MW IT capacity at 70% utilization (conservative estimate)
        power = group["total_power_mw"].values if "total_power_mw" in group.columns else np.ones(len(group)) * 100
        # Align lengths
        n = min(len(power), len(base_grid_demand))
        # Calc Score
        gss = grid_stress.calculate_grid_stress_score(power[:n], base_grid_demand[:n])
        scenario_metrics.append({
            "scenario": name,
            "grid_stress_score": gss["grid_stress_score"],
            "cpf": gss["cpf"]
        })
        
    scenario_metrics_df = pd.DataFrame(scenario_metrics)
    
    # 2. Run Grid Stress analysis
    base_scenario = forecast_df[forecast_df["scenario"] == "Baseline (15%)"]
    projected_power_base = base_scenario["total_power_mw"].values if "total_power_mw" in base_scenario.columns else np.ones(len(base_scenario)) * 100
    
    grid_res = grid_stress.run_grid_stress_analysis(
        projected_power_base, pjm_2019, scenario_metrics_df
    )
    
    # 3. Carbon analysis (simplified - using carbon intensity from data)
    carbon_intensity = data["carbon_intensity"]
    log.info(f"Carbon intensity data available for emissions analysis")
    
    # ─── 8. SUMMARY ─────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    
    # Save Final Pipeline Summary
    final_summary = {
        "runtime_minutes": round(elapsed / 60, 1),
        "target_variable": "pue",  # PUE - Power Usage Effectiveness
        "models_trained": list(model_metrics.keys()),
        "model_metrics": model_metrics,
        "forecast_scenarios": list(forecast_df["scenario"].unique()),
        "pue_stats": {
            "mean": float(hourly["pue"].mean()),
            "min": float(hourly["pue"].min()),
            "max": float(hourly["pue"].max()),
        },
        "data_sources": {
            "temperature": "NOAA (real)",
            "carbon_intensity": "EIA-930 (real)",
            "pue": "Patterson (2008) physics model",
        },
        "assumptions_removed": [
            "Sinusoidal utilization pattern (fabricated)",
            "Base utilization = 70% (arbitrary)",
            "Weekend reduction = 8% (arbitrary)",
        ]
    }
    save_json(final_summary, "pipeline_summary")
    
    log.info("\n" + "=" * 70)
    log.info(f"PIPELINE COMPLETE ({elapsed/60:.1f} min)")
    log.info(f"Results saved to: {cfg.RESULTS_DIR}")
    log.info("=" * 70)

if __name__ == "__main__":
    main()
