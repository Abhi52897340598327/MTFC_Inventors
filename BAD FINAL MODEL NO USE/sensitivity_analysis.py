"""
Digital Twin Sensitivity Analysis
=================================
Purpose: Quantify how changes in Input Variables affect the Output.
Method: OAT (One-at-a-Time) Analysis using physics-based calculations.
Metric: % Change in Annual Carbon Emissions.

Physics Equations:
- IT Power = Capacity × (0.3 + 0.7 × Utilization)
- PUE = PUE_base + 0.012 × max(0, Temp - 65°F)
- Total Power = IT Power × PUE
- Carbon Emissions = Total Power × Carbon Intensity × Hours
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, "digital_twin")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Physical constants
FACILITY_MW = 100           # IT capacity (MW)
IDLE_POWER_FRACTION = 0.3   # 30% idle power
COOL_THRESH_F = 65          # Cooling threshold (°F)
BASE_PUE = 1.15             # Base PUE
MAX_PUE = 2.0               # Maximum PUE
HOURS_PER_YEAR = 8760       # Hours in a year


def run_sensitivity():
    """Run physics-based sensitivity analysis."""
    print("\n=== Sensitivity Analysis (OAT - Physics-Based) ===")
    
    # Baseline parameters
    baseline = {
        "temperature_f": 65,
        "utilization": 0.70,
        "pue_base": 1.15,
        "carbon_intensity": 387,  # kg CO2/MWh (PJM average)
        "growth_factor": 1.0,
        "capacity_mw": FACILITY_MW,
    }
    
    def calculate_annual_carbon(params):
        """Calculate annual carbon emissions using physics equations."""
        # Extract parameters
        temp = params.get("temperature_f", 65)
        util = params.get("utilization", 0.70)
        pue_base = params.get("pue_base", BASE_PUE)
        carbon_int = params.get("carbon_intensity", 387)
        growth = params.get("growth_factor", 1.0)
        capacity = params.get("capacity_mw", FACILITY_MW)
        
        # Apply growth to capacity
        effective_capacity = capacity * growth
        
        # IT Power: P_IT = Capacity × (idle + (1-idle) × utilization)
        it_power_mw = effective_capacity * (IDLE_POWER_FRACTION + (1 - IDLE_POWER_FRACTION) * util)
        
        # PUE: Increases with temperature above threshold
        temp_above = max(0, temp - COOL_THRESH_F)
        pue = np.clip(pue_base + 0.012 * temp_above, pue_base, MAX_PUE)
        
        # Total Power
        total_power_mw = it_power_mw * pue
        
        # Annual Energy
        annual_mwh = total_power_mw * HOURS_PER_YEAR
        
        # Carbon Emissions (kg → kilo-tons)
        carbon_kg = annual_mwh * carbon_int
        carbon_ktons = carbon_kg / 1e6
        
        return carbon_ktons
    
    # Calculate Baseline
    base_carbon = calculate_annual_carbon(baseline)
    print(f"Baseline Annual Carbon: {base_carbon:.1f} kTons CO₂")
    
    # Sensitivity Tests: [Variable, New Value, Label]
    tests = [
        ("temperature_f", baseline["temperature_f"] + 20, "Temp + 20°F"),
        ("temperature_f", baseline["temperature_f"] - 10, "Temp - 10°F"),
        ("utilization", 0.90, "Utilization 90%"),
        ("utilization", 0.50, "Utilization 50%"),
        ("growth_factor", 1.5, "Capacity +50%"),
        ("growth_factor", 0.75, "Capacity -25%"),
        ("pue_base", 1.35, "PUE 1.35 (Inefficient)"),
        ("pue_base", 1.10, "PUE 1.10 (Efficient)"),
        ("carbon_intensity", 500, "High Carbon Grid (500)"),
        ("carbon_intensity", 250, "Low Carbon Grid (250)"),
    ]
    
    results = []
    for var, val, label in tests:
        test_params = baseline.copy()
        test_params[var] = val
        carbon = calculate_annual_carbon(test_params)
        pct_change = ((carbon - base_carbon) / base_carbon) * 100
        results.append({
            "Variable": var, 
            "Scenario": label, 
            "Change %": pct_change,
            "Carbon kTons": carbon
        })
        print(f"  {label:<25} -> {pct_change:+.1f}%  ({carbon:.1f} kTons)")
    
    # Plot
    df_res = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color by direction
    colors = ['#2ecc71' if x < 0 else '#e74c3c' for x in df_res['Change %']]
    
    bars = ax.barh(df_res['Scenario'], df_res['Change %'], color=colors, edgecolor='black')
    ax.axvline(0, color='black', linewidth=2)
    ax.set_xlabel('% Change from Baseline', fontsize=12)
    ax.set_title(f'Sensitivity Analysis: Impact on Annual Carbon Emissions\n'
                 f'Baseline: {base_carbon:.1f} kTons CO₂/year', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, val in zip(bars, df_res['Change %']):
        color = 'darkgreen' if val < 0 else 'darkred'
        offset = 1 if val >= 0 else -1
        ha = 'left' if val >= 0 else 'right'
        ax.annotate(f'{val:+.1f}%', 
                   xy=(val + offset, bar.get_y() + bar.get_height()/2),
                   va='center', ha=ha, fontsize=10, fontweight='bold', color=color)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='Decrease (Good)'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Increase (Bad)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sensitivity_chart.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Sensitivity Chart saved to {OUTPUT_DIR}/sensitivity_chart.png")
    
    return df_res


if __name__ == "__main__":
    run_sensitivity()
