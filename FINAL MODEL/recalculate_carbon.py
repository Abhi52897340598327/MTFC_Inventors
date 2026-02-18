"""
Recalculate PJM Carbon Intensity from Generation Mix
====================================================
Regenerates the missing 'pjm_grid_carbon_intensity_2019_full_cleaned.csv'
using the raw generation-by-fuel data.

Input: Data_Sources/pjm_generation_by_fuel_2019_2024_eia.csv
Output: Data_Sources/cleaned/pjm_grid_carbon_intensity_2019_full_cleaned.csv
"""

import pandas as pd
import numpy as np
import os

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "Data_Sources")
CLEANED_DIR = os.path.join(DATA_DIR, "cleaned")
INPUT_FILE = os.path.join(DATA_DIR, "pjm_generation_by_fuel_2019_2024_eia.csv")
OUTPUT_FILE = os.path.join(CLEANED_DIR, "pjm_grid_carbon_intensity_2019_full_cleaned.csv")

# kg CO2 per MWh
EMISSIONS_FACTORS = {
    "Coal": 950,
    "Natural Gas": 400,  # Combined Cycle average
    "Petroleum": 800,
    "Nuclear": 0,
    "Hydro": 0,
    "Solar": 0,
    "Wind": 0,
    "Other": 400,
}

def main():
    print("Recalculating PJM Carbon Intensity...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        return

    # Load raw data
    print(f"  Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # Pivot: period x fueltype -> generation
    # Columns of interest: period, type-name, value
    # Filter for valid rows
    df = df.dropna(subset=["value"])
    
    print("  Pivoting data...")
    pivot = df.pivot_table(
        index="period", 
        columns="type-name", 
        values="value", 
        aggfunc="sum"
    ).fillna(0)
    
    # Calculate Emissions
    total_emissions = 0
    total_gen = 0
    
    print("  Calculating emissions...")
    for fuel, factor in EMISSIONS_FACTORS.items():
        # Match column names (case insensitive partial match)
        cols = [c for c in pivot.columns if fuel.lower() in c.lower()]
        for c in cols:
            gen_mwh = pivot[c]
            emissions = gen_mwh * factor
            total_emissions += emissions
            total_gen += gen_mwh
            # print(f"    + {c}: {gen_mwh.sum():,.0f} MWh -> {emissions.sum():,.0f} kg CO2")
            
    # Also add generation from non-emitting sources that might not be in the dictionary explicitly
    # actually, better to just sum all columns for total_gen?
    # Yes, Carbon Intensity = Total Emissions / Total Generation (All Sources)
    total_gen_all = pivot.sum(axis=1)
    
    CARBON_INTENSITY = total_emissions / total_gen_all
    
    # Format output
    output_df = pd.DataFrame({
        "timestamp": pd.to_datetime(pivot.index),
        "carbon_intensity_kg_per_mwh": CARBON_INTENSITY
    }).sort_values("timestamp")
    
    # Handle NaN (div by zero if grid down? unlikely for PJM)
    output_df = output_df.fillna(0)
    
    # Save
    os.makedirs(CLEANED_DIR, exist_ok=True)
    print(f"  Saving to {OUTPUT_FILE}...")
    output_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"  Done. Generated {len(output_df)} hourly records.")
    print(f"  Avg Intensity: {output_df['carbon_intensity_kg_per_mwh'].mean():.1f} kg/MWh")

if __name__ == "__main__":
    main()
