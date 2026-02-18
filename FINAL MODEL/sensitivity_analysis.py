"""
Digital Twin Sensitivity Analysis
=================================
Purpose: Quantify how changes in Input Variables affect the Output.
Method: OAT (One-at-a-Time) Analysis.
Metric: % Change in 2030 Carbon Footprint.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import Core Logic
from run_digital_twin import load_real_data, train_grid_model, DatacenterTwin, load_constants, OUTPUT_DIR

def run_sensitivity():
    print("\n=== Sensitivity Analysis (OAT) ===")
    
    # Baseline Setup
    C = load_constants()
    df_hist = load_real_data()
    model, features = train_grid_model(df_hist) # retrain quickly for scope
    
    # Define Baseline Scenario (2030, Aggressive Growth)
    base_year = 2030
    base_growth = 0.30
    base_temp_adder = 1.5
    
    # Helper to run a single simulation
    def get_carbon_impact(temp_adder, growth, pue_factor=1.0):
        # 1. Modify Constants if needed (PUE factor)
        C_mod = C.copy()
        if pue_factor != 1.0:
            # Scale PUE min and max
            # Note: simplistic scaling of the curve
            C_mod["facility_specs"]["pue_min"] *= pue_factor
            
        twin = DatacenterTwin(C_mod)
        
        # 2. Future Weather
        future_dates = pd.date_range(f"{base_year}-01-01", f"{base_year}-12-31 23:00", freq="h")
        df_fut = pd.DataFrame({"timestamp": future_dates})
        df_fut["hour"] = df_fut["timestamp"].dt.hour
        
        # Base on 2022
        base_temp = df_hist[df_hist["timestamp"].dt.year == 2022]["temperature_f"].values[:len(df_fut)]
        if len(base_temp) < len(df_fut): base_temp = np.resize(base_temp, len(df_fut))
        
        df_fut["temperature_f"] = base_temp + (temp_adder * 1.8)
        
        # 3. Simulate
        res = twin.simulate_year(base_year, df_fut, growth)
        return res["carbon_ktons"]
    
    # Calculate Baseline
    base_carbon = get_carbon_impact(base_temp_adder, base_growth)
    print(f"Baseline 2030 Carbon: {base_carbon:.1f} kTons")
    
    # Sensitivity Tests: [Variable, New Value, Label]
    tests = [
        ("Temp", base_temp_adder + 2.0, "Temp + 2C"),
        ("Temp", base_temp_adder - 1.0, "Temp - 1C"),
        ("Growth", 0.40, "Growth 40%"),
        ("Growth", 0.20, "Growth 20%"),
        ("PUE", 1.1, "PUE +10% (Inefficient)"), # Factor 1.1
        ("PUE", 0.9, "PUE -10% (Super Efficient)")
    ]
    
    results = []
    for var, val, label in tests:
        if var == "Temp":
            c = get_carbon_impact(val, base_growth)
        elif var == "Growth":
            c = get_carbon_impact(base_temp_adder, val)
        elif var == "PUE":
            c = get_carbon_impact(base_temp_adder, base_growth, pue_factor=val)
            
        pct_change = ((c - base_carbon) / base_carbon) * 100
        results.append({"Variable": var, "Scenario": label, "Change %": pct_change})
        print(f"  {label:<25} -> {pct_change:+.1f}%")
        
    # Plot
    df_res = pd.DataFrame(results)
    plt.figure(figsize=(10,6))
    sns.barplot(data=df_res, y="Scenario", x="Change %", hue="Variable", dodge=False)
    plt.axvline(0, color="k", linewidth=1)
    plt.title(f"Sensitivity Analysis: Impact on 2030 Carbon Footprint")
    plt.xlabel("% Change from Baseline Output")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "sensitivity_chart.png"))
    print(f"Sensitivity Chart saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_sensitivity()
