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
    # We create features, but KEY POINT: For this Hybrid approach, we rely on 
    # "Forecast Safe" columns (Time, Temp) for the primary model.
    # Lags are used ONLY for the SARIMAX baseline comparison.
    df_feat = feature_engineering.engineer_features(hourly)
    
    # ─── 4. DATA PREPARATION (Split & Scale) ────────────────────────────────────
    log.info("\n--- PHASE 4: PREPARATION ---")
    # Identify Forecast-Safe columns (No lags!)
    feature_cols = data_preparation.get_forecast_feature_cols(df_feat)
    target_col = cfg.TARGET_COL
    
    log.info(f"Target: {target_col}")
    log.info(f"Features ({len(feature_cols)}): {feature_cols}")
    
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
    
    # A. XGBoost (The Core Engine)
    log.info("Training XGBoost (Regression Mode)...")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, feature_names=feature_cols)
    xgb_metrics = evaluate_xgboost(xgb_model, X_test, y_test)
    model_metrics["XGBoost"] = xgb_metrics["metrics"]
    plot_feature_importance(xgb_model, feature_cols)
    
    log.info(f"XGBoost Performance: R2={xgb_metrics['metrics']['R2']:.4f}, MAE={xgb_metrics['metrics']['MAE']:.2f}")

    # B. SARIMAX (Optional Baseline)
    if "--skip-sarimax" not in sys.argv:
        log.info("Training SARIMAX (Statistical Baseline)...")
        try:
            res_sarimax = train_sarimax(train, val)
            eval_sarimax = evaluate_sarimax(res_sarimax, train, test)
            model_metrics["SARIMAX"] = eval_sarimax["metrics"]
            log.info(f"SARIMAX Performance: R2={eval_sarimax['metrics']['R2']:.4f}")
        except Exception as e:
            log.warning(f"SARIMAX Failed: {e}")
    
    # ─── 6. FORECASTING (Scenario Projection) ───────────────────────────────────
    log.info("\n--- PHASE 6: SCENARIO FORECASTING (2025-2030) ---")
    # Use the trained XGBoost + Scaler to simulate future scenarios
    forecast_df = forecasting.run_forecast(xgb_model, scaler, feature_cols)
    
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
        power = group["total_power_mw"].values
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
    
    # 2. Run Detailed Analysis on Baseline
    base_scenario = forecast_df[forecast_df["scenario"] == "Baseline (15%)"]
    projected_power_base = base_scenario["total_power_mw"].values
    
    # Carbon (Baseline)
    carbon_res = carbon_emissions.run_carbon_analysis(
        projected_power_base, data["carbon_intensity"], data["co2"]
    )
    
    # Grid Stress (Baseline + Comparison)
    grid_res = grid_stress.run_grid_stress_analysis(
        projected_power_base, data["pjm_demand"], scenario_metrics_df
    )
    
    # ─── 8. SUMMARY ─────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    
    # Save Final Pipeline Summary
    final_summary = {
        "runtime_minutes": round(elapsed / 60, 1),
        "models_trained": list(model_metrics.keys()),
        "model_metrics": model_metrics,
        "forecast_scenarios": list(forecast_df["scenario"].unique()),
        "baseline_impact": {
            "carbon_total_tons": carbon_res.get("total_2019_tons", 0), # using 2019 key as proxy for 'annual'
            "grid_stress_score": grid_res["grid_stress"]["grid_stress_score"]
        }
    }
    save_json(final_summary, "pipeline_summary")
    
    log.info("\n" + "=" * 70)
    log.info(f"PIPELINE COMPLETE ({elapsed/60:.1f} min)")
    log.info(f"Results saved to: {cfg.RESULTS_DIR}")
    log.info("=" * 70)

if __name__ == "__main__":
    main()
