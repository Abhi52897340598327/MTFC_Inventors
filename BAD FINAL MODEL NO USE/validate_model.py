"""
Digital Twin Validation Module
==============================
Purpose: 
1. Rigorously validate the XGBoost Grid Forecast Model.
2. Verify Physics Engine (PUE) constraints.
3. Output formal metrics for project reporting.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Import Core Logic
from run_digital_twin import load_real_data, train_grid_model, DatacenterTwin, load_constants, OUTPUT_DIR

def validate_xgboost():
    """Train/Test Split Validation of Grid Model."""
    print("\n=== 1. Validating XGBoost Grid Model ===")
    
    # Load Data
    df = load_real_data()
    
    # Train/Test Split (Time Series sensitive: Last 20% is Test)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    print(f"Training Samples: {len(train_df)}")
    print(f"Testing Samples:  {len(test_df)}")
    
    # Training on 80%
    model, features = train_grid_model(train_df)
    
    # Predicting on 20%
    test_df = test_df.copy()
    test_df["predicted_mw"] = model.predict(test_df[features])
    
    # Metrics
    y_true = test_df["grid_demand_mw"]
    y_pred = test_df["predicted_mw"]
    
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"\n--- Validation Metrics (Held-Out Test Set) ---")
    print(f"R-Squared (R2): {r2:.4f}  (Target: > 0.8)")
    print(f"MAE:            {mae:.0f} MW")
    print(f"RMSE:           {rmse:.0f} MW")
    print(f"MAPE:           {mape:.2f}%  (Target: < 5%)")
    
    # Residual Plot
    residuals = y_true - y_pred
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.3)
    plt.axhline(0, color='r', linestyle='--')
    plt.title(f"Residual Plot (R2={r2:.3f})")
    plt.xlabel("Predicted Demand (MW)")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.savefig(os.path.join(OUTPUT_DIR, "validation_residuals.png"))
    print(f"Residual Plot saved to {OUTPUT_DIR}")
    
    return r2

def validate_physics():
    """Sanity Check of Physics Engine."""
    print("\n=== 2. Validating Physics Engine (Thermodynamics) ===")
    
    C = load_constants()
    twin = DatacenterTwin(C)
    
    # Test Range: 0F to 120F
    test_temps = np.arange(0, 120, 1)
    pue_values = twin.get_pue(test_temps)
    
    # Constraint 1: PUE >= 1.0 (Cannot create energy)
    min_pue = pue_values.min()
    print(f"Minimum PUE Observed: {min_pue:.4f}")
    if min_pue < 1.0:
        print("CRITICAL FAIL: PUE < 1.0 violates thermodynamics.")
    else:
        print("PASS: PUE >= 1.0")
        
    # Constraint 2: PUE increases with Temp (above optimal)
    # Check slope at 100F
    pue_100 = twin.get_pue(100)
    pue_101 = twin.get_pue(101)
    if pue_101 > pue_100:
        print("PASS: PUE increases with heat (Cost of Cooling)")
    else:
        print("FAIL: PUE did not increase with heat.")

if __name__ == "__main__":
    validate_xgboost()
    validate_physics()
