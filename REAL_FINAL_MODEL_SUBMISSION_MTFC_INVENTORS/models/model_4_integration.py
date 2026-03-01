"""
MTFC Model 4: Integration Layer
Combines all forecasts and calculates risk metrics.

Mathematical Operations:
1. Merge all time series forecasts
2. Calculate datacenter energy with AI growth
3. Apply interaction terms (renewable intensity, grid stress penalty)
4. Calculate carbon intensity and CO2 emissions by source
5. Compute grid stress metrics

Output: Integrated forecast with all risk indicators
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Directory Configuration
BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / 'REAL FINAL FILES' / 'model_forecasts'
OUTPUT_DIR = BASE_DIR / 'REAL FINAL FILES' / 'model_forecasts'

# Physical Constants
EMISSION_FACTORS = {
    'coal_pct': 2.23,      # lb CO2/kWh
    'gas_pct': 0.91,       # lb CO2/kWh  
    'nuclear_pct': 0.01,   # lb CO2/kWh
    'renewable_pct': 0.03  # lb CO2/kWh
}

CARBON_INTENSITY_FLOOR = 0.03
DATACENTER_BASELINE_SHARE = 0.25
RENEWABLE_INTENSITY_FACTOR = 0.02
GRID_STRESS_THRESHOLD = 35.0
GRID_STRESS_PENALTY_RATE = 0.05
KWH_PER_GWH = 1_000_000
LB_PER_TON = 2_000


def run_integration():
    """
    Integrate all forecasts and calculate comprehensive risk metrics.
    
    Returns:
        DataFrame with integrated forecasts and risk indicators
    """
    
    print("\n" + "=" * 70)
    print(" MODEL 4: INTEGRATION & RISK CALCULATION ".center(70))
    print("=" * 70)
    
    # Load all forecasts
    print("\nLoading forecasts...")
    energy = pd.read_csv(INPUT_DIR / 'forecast_energy.csv')
    coal = pd.read_csv(INPUT_DIR / 'forecast_grid_coal.csv')
    gas = pd.read_csv(INPUT_DIR / 'forecast_grid_gas.csv')
    nuclear = pd.read_csv(INPUT_DIR / 'forecast_grid_nuclear.csv')
    renewable = pd.read_csv(INPUT_DIR / 'forecast_grid_renewable.csv')
    ai = pd.read_csv(INPUT_DIR / 'forecast_ai_multiplier.csv')
    
    # Merge all datasets
    data = energy.merge(coal, on='date') \
                 .merge(gas, on='date') \
                 .merge(nuclear, on='date') \
                 .merge(renewable, on='date') \
                 .merge(ai, on='date')
    
    print(f"✓ Merged {len(data)} months of forecasts")
    
    # ==================================================================
    # DATACENTER ENERGY CALCULATIONS
    # ==================================================================
    
    print("\nCalculating datacenter energy demand...")
    
    # Base datacenter energy (current ~25% of ELECTRICITY consumption)
    data['dc_energy_baseline_gwh'] = data['electricity_gwh'] * DATACENTER_BASELINE_SHARE
    
    # Apply AI growth multiplier
    data['dc_energy_ai_gwh'] = data['dc_energy_baseline_gwh'] * data['ai_multiplier']
    
    # Renewable intermittency adjustment
    data['renewable_share'] = data['renewable_pct'] / 100.0
    data['intensity_factor'] = 1.0 + RENEWABLE_INTENSITY_FACTOR * data['renewable_share']
    data['dc_energy_adjusted_gwh'] = data['dc_energy_ai_gwh'] * data['intensity_factor']
    
    print(f"  Baseline (2024): {data['dc_energy_baseline_gwh'].iloc[0]:.1f} GWh")
    print(f"  With AI (2024): {data['dc_energy_ai_gwh'].iloc[0]:.1f} GWh")
    print(f"  Adjusted (2024): {data['dc_energy_adjusted_gwh'].iloc[0]:.1f} GWh")
    print(f"  Adjusted (2038): {data['dc_energy_adjusted_gwh'].iloc[-1]:.1f} GWh")
    
    # ==================================================================
    # CARBON INTENSITY CALCULATIONS
    # ==================================================================
    
    print("\nCalculating carbon intensity...")
    
    # Weighted average of emission factors
    data['carbon_intensity'] = sum(
        data[source] / 100.0 * factor
        for source, factor in EMISSION_FACTORS.items()
    )
    
    # Apply floor constraint
    data['carbon_intensity'] = np.maximum(data['carbon_intensity'], CARBON_INTENSITY_FLOOR)
    
    print(f"  2024: {data['carbon_intensity'].iloc[0]:.4f} lb CO2/kWh")
    print(f"  2038: {data['carbon_intensity'].iloc[-1]:.4f} lb CO2/kWh")
    print(f"  Change: {((data['carbon_intensity'].iloc[-1]/data['carbon_intensity'].iloc[0] - 1)*100):.1f}%")
    
    # ==================================================================
    # CO2 EMISSIONS BY SOURCE
    # ==================================================================
    
    print("\nCalculating CO2 emissions by source...")
    
    for source, factor in EMISSION_FACTORS.items():
        source_name = source.replace('_pct', '')
        
        # Calculate emissions in pounds
        co2_lb = data['dc_energy_adjusted_gwh'] * KWH_PER_GWH * (data[source] / 100.0) * factor
        
        # Convert to tons
        data[f'co2_{source_name}_tons'] = co2_lb / LB_PER_TON
    
    # Total CO2 emissions
    co2_columns = [f'co2_{s.replace("_pct", "")}_tons' for s in EMISSION_FACTORS.keys()]
    data['co2_total_tons'] = data[co2_columns].sum(axis=1)
    
    # Verification
    calculated_total = data['dc_energy_adjusted_gwh'] * KWH_PER_GWH * data['carbon_intensity'] / LB_PER_TON
    verification_error = np.abs(data['co2_total_tons'] - calculated_total).max()
    assert verification_error < 1.0, f"CO2 calculation error: {verification_error:.2f} tons"
    
    print(f"  Total CO2 (2024): {data['co2_total_tons'].iloc[0]:,.0f} tons")
    print(f"  Total CO2 (2038): {data['co2_total_tons'].iloc[-1]:,.0f} tons")
    print(f"  Cumulative (15yr): {data['co2_total_tons'].sum():,.0f} tons")
    
    # ==================================================================
    # GRID STRESS CALCULATIONS
    # ==================================================================
    
    print("\nCalculating grid stress...")
    
    # Base grid stress (percentage of total ELECTRICITY grid)
    data['grid_stress_pct'] = (data['dc_energy_adjusted_gwh'] / data['electricity_gwh']) * 100.0
    
    # Apply additive penalty above threshold (stress remains a percentage)
    # This represents additional capacity pressure beyond the physical share
    data['stress_penalty_pts'] = np.where(
        data['grid_stress_pct'] > GRID_STRESS_THRESHOLD,
        GRID_STRESS_PENALTY_RATE * (data['grid_stress_pct'] - GRID_STRESS_THRESHOLD),
        0.0
    )
    
    data['grid_stress_adjusted_pct'] = data['grid_stress_pct'] + data['stress_penalty_pts']
    
    max_stress_idx = int(data['grid_stress_adjusted_pct'].idxmax())
    max_stress_date = str(data['date'].iloc[max_stress_idx])
    max_stress_value = float(data['grid_stress_adjusted_pct'].iloc[max_stress_idx])
    
    print(f"  Base stress (2024): {data['grid_stress_pct'].iloc[0]:.2f}%")
    print(f"  Base stress (2038): {data['grid_stress_pct'].iloc[-1]:.2f}%")
    print(f"  Peak adjusted stress: {max_stress_value:.2f}% on {max_stress_date}")
    
    if max_stress_value > 40:
        print(f"  ⚠ WARNING: Peak stress exceeds critical threshold (40%)")
    elif max_stress_value > 35:
        print(f"  ⚠ CAUTION: Peak stress exceeds concern threshold (35%)")
    
    # ==================================================================
    # SAVE OUTPUTS
    # ==================================================================
    
    print("\nSaving output files...")
    
    # Carbon intensity
    carbon_output = data[['date', 'carbon_intensity']]
    carbon_output.to_csv(OUTPUT_DIR / 'forecast_carbon_intensity.csv', index=False)
    print(f"✓ Saved forecast_carbon_intensity.csv")
    
    # CO2 emissions by source
    co2_output_cols = ['date'] + co2_columns + ['co2_total_tons']
    co2_output = data[co2_output_cols]
    co2_output.to_csv(OUTPUT_DIR / 'forecast_co2_emissions.csv', index=False)
    print(f"✓ Saved forecast_co2_emissions.csv")
    
    # Integrated forecast (comprehensive)
    integrated_cols = [
        'date',
        'electricity_gwh',
        'dc_energy_baseline_gwh',
        'dc_energy_ai_gwh',
        'dc_energy_adjusted_gwh',
        'ai_multiplier',
        'coal_pct',
        'gas_pct',
        'nuclear_pct',
        'renewable_pct',
        'renewable_share',
        'intensity_factor',
        'carbon_intensity',
        'co2_coal_tons',
        'co2_gas_tons',
        'co2_nuclear_tons',
        'co2_renewable_tons',
        'co2_total_tons',
        'grid_stress_pct',
        'stress_penalty_pts',
        'grid_stress_adjusted_pct'
    ]
    
    integrated_output = data[integrated_cols]
    integrated_output.to_csv(OUTPUT_DIR / 'forecast_integrated.csv', index=False)
    print(f"✓ Saved forecast_integrated.csv")
    
    print("=" * 70 + "\n")
    
    return data


if __name__ == '__main__':
    run_integration()
