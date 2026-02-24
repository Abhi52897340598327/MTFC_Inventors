"""
Advanced Analysis & Recommendations
===================================
Generates high-impact visuals for the Project Board:
1. Tornado Plot: Which variable is the biggest risk?
2. Mitigation Waterfall: How do we reach Net Zero?
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from run_digital_twin import load_real_data, train_grid_model, DatacenterTwin, load_constants, OUTPUT_DIR

def run_tornado_analysis():
    print("\n=== generating Tornado Plot ===")
    
    # 1. Baseline
    C = load_constants()
    df_hist = load_real_data()
    # Retrain XGBoost quickly for this context
    model, features = train_grid_model(df_hist)
    
    base_year = 2030
    base_growth = 0.30
    
    def get_result(temp_adder=1.5, growth=0.30, pue_mult=1.0, grid_dirty_mult=1.0):
        # Config
        C_mod = C.copy()
        C_mod["facility_specs"]["pue_min"] *= pue_mult
        C_mod["grid_specs"]["carbon_intensity_g_per_kwh"] *= grid_dirty_mult
        
        twin = DatacenterTwin(C_mod)
        
        # Weather
        dates = pd.date_range(f"{base_year}-01-01", f"{base_year}-12-31 23:00", freq="h")
        df = pd.DataFrame({"timestamp": dates})
        df["hour"] = df["timestamp"].dt.hour
        
        # Temp Base
        t_base = df_hist[df_hist["timestamp"].dt.year == 2022]["temperature_f"].values[:len(df)]
        if len(t_base) < len(df): t_base = np.resize(t_base, len(df))
        df["temperature_f"] = t_base + (temp_adder * 1.8)
        
        res = twin.simulate_year(base_year, df, growth)
        return res["carbon_ktons"]

    base_val = get_result()
    
    # 2. Parameters to Swing (+/- 10% or realistic steps)
    swings = [
        {"name": "AI Growth Rate", "low_val": 0.20, "high_val": 0.40, "type": "growth"},
        {"name": "Cooling Tech (PUE)", "low_val": 0.9, "high_val": 1.15, "type": "pue"}, # 0.9 = Efficient, 1.15 = Inefficient
        {"name": "Grid Carbon Intensity", "low_val": 0.8, "high_val": 1.2, "type": "grid"},
        {"name": "Climate Temp (+C)", "low_val": 0.5, "high_val": 3.0, "type": "temp"},
    ]
    
    data = []
    
    for s in swings:
        # Calculate Low Case
        args = {"growth": base_growth, "pue_mult": 1.0, "grid_dirty_mult": 1.0, "temp_adder": 1.5}
        
        if s["type"] == "growth": args["growth"] = s["low_val"]
        elif s["type"] == "pue": args["pue_mult"] = s["low_val"]
        elif s["type"] == "grid": args["grid_dirty_mult"] = s["low_val"]
        elif s["type"] == "temp": args["temp_adder"] = s["low_val"]
        
        low_res = get_result(**args)
        
        # Calculate High Case
        args = {"growth": base_growth, "pue_mult": 1.0, "grid_dirty_mult": 1.0, "temp_adder": 1.5}
        
        if s["type"] == "growth": args["growth"] = s["high_val"]
        elif s["type"] == "pue": args["pue_mult"] = s["high_val"]
        elif s["type"] == "grid": args["grid_dirty_mult"] = s["high_val"]
        elif s["type"] == "temp": args["temp_adder"] = s["high_val"]
        
        high_res = get_result(**args)
        
        # Relative to baseline
        data.append({
            "Parameter": s["name"],
            "Low Impact": low_res - base_val,
            "High Impact": high_res - base_val,
            "Range": abs(high_res - low_res)
        })
        
    # Plot Tornado
    df_torn = pd.DataFrame(data).sort_values("Range", ascending=True)
    
    plt.figure(figsize=(10,6))
    y = np.arange(len(df_torn))
    plt.barh(y, df_torn["Low Impact"], color="green", label="Optimistic Case")
    plt.barh(y, df_torn["High Impact"], color="red", label="Pessimistic Case")
    plt.yticks(y, df_torn["Parameter"])
    plt.axvline(0, color='k', linewidth=0.8)
    plt.xlabel("Change in Carbon Emissions (kTons) relative to Baseline")
    plt.title("Tornado Plot: Sensitivity of 2030 Carbon Footprint")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "advanced_tornado.png"))
    print("Tornado Plot Saved.")

def run_mitigation_waterfall():
    print("\n=== Generating Mitigation Waterfall ===")
    
    # Establish Baseline Risk
    # 2030 Aggressive Growth, High Temp, Standard PUE
    
    steps = []
    
    # 1. Baseline
    # (Simulated value aprox 327 ktons from previous runs)
    current_level = 327 
    steps.append({"Label": "2030 Baseline\n(Business as Usual)", "Value": current_level, "Type": "Total"})
    
    # 2. Tech Fix: Liquid Cooling (Reduces PUE ~15%)
    # Impact: ~10% total carbon reduction
    reduction_liquid = -32 
    current_level += reduction_liquid
    steps.append({"Label": "Switch to\nLiquid Cooling", "Value": reduction_liquid, "Type": "Change"})
    
    # 3. Ops Fix: Load Shifting (Move 20% load to night)
    # Night grid is often ~5-10% cleaner (nuclear baseload).
    # Impact: small but free (~2-3%)
    reduction_shifting = -10
    current_level += reduction_shifting
    steps.append({"Label": "AI Load\nShifting", "Value": reduction_shifting, "Type": "Change"})
    
    # 4. Strategic Fix: PPA (Solar/Wind Purchase)
    # Buying 50% renewable energy credits
    reduction_ppa = -142 # 50% of remaining roughly
    current_level += reduction_ppa
    steps.append({"Label": "50% Renewable\nPPA", "Value": reduction_ppa, "Type": "Change"})
    
    # 5. Final State
    steps.append({"Label": "Optimized\nFuture", "Value": current_level, "Type": "Total"})
    
    # Plot Waterfall
    labels = [s["Label"] for s in steps]
    values = [s["Value"] for s in steps]
    types = [s["Type"] for s in steps]
    
    # Cumulative calculation for bars
    # Waterfalls are tricky in pure matplotlib, approximating with bar stack logic
    # Bar Bottoms:
    # 0 -> 327
    # 1 -> 327-32 = 295 (Top is 327, Bottom is 295) -> actually matplotlib bars draw up/down from a point
    
    plt.figure(figsize=(12, 6))
    
    running_total = 0
    for i, step in enumerate(steps):
        if step["Type"] == "Total":
            color = "blue" if i == 0 else "green"
            plt.bar(i, step["Value"], color=color)
            running_total = step["Value"]
            plt.text(i, step["Value"] + 5, f"{step['Value']:.0f}", ha='center', fontweight='bold')
        else:
            # Change
            color = "red" if step["Value"] > 0 else "limegreen"
            # Start from previous total
            prev_total = steps[i-1]["Value"] if i==1 else running_total # simplified for sequential logic
            
            # If negative change (reduction)
            bottom = running_total + step["Value"]
            height = abs(step["Value"])
            plt.bar(i, height, bottom=bottom, color=color)
            
            plt.text(i, bottom - 15, f"{step['Value']:.0f}", ha='center', color='white', fontweight='bold')
            running_total += step["Value"]

    plt.xticks(range(len(labels)), labels)
    plt.ylabel("Annual Carbon Emissions (kTons)")
    plt.title("Mitigation Pathway: Reducing the AI Carbon Footprint")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "advanced_waterfall.png"))
    print("Waterfall Chart Saved.")

if __name__ == "__main__":
    run_tornado_analysis()
    run_mitigation_waterfall()
