"""
Visualization: Grid Stress Analysis
Dual-panel plot showing grid stress metrics and energy comparison.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / 'REAL FINAL FILES' / 'model_forecasts'
OUTPUT_DIR = BASE_DIR / 'REAL FINAL FILES' / 'visualizations'

def plot_grid_stress():
    data = pd.read_csv(INPUT_DIR / 'forecast_integrated.csv')
    data['date'] = pd.to_datetime(data['date'])
    
    # Load AI multiplier CIs if available
    ai_path = INPUT_DIR / 'forecast_ai_multiplier.csv'
    ai_df = pd.read_csv(ai_path)
    ai_df['date'] = pd.to_datetime(ai_df['date'])
    has_ai_ci = 'ai_multiplier_lower' in ai_df.columns
    if has_ai_ci:
        data = data.merge(ai_df[['date', 'ai_multiplier_lower', 'ai_multiplier_upper']], on='date', how='left')
        data['dc_energy_gwh_lower'] = data['dc_energy_baseline_gwh'] * data['ai_multiplier_lower']
        data['dc_energy_gwh_upper'] = data['dc_energy_baseline_gwh'] * data['ai_multiplier_upper']
        data['dc_share_pct_lower'] = (data['dc_energy_gwh_lower'] / data['electricity_gwh']) * 100
        data['dc_share_pct_upper'] = (data['dc_energy_gwh_upper'] / data['electricity_gwh']) * 100
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Panel 1: DC Share of Grid
    ax1.plot(data['date'], data['dc_share_pct'], 
             linewidth=2.5, color='#1f77b4', label='DC Share of Grid')
    if has_ai_ci:
        ax1.fill_between(data['date'], data['dc_share_pct_lower'], data['dc_share_pct_upper'],
                         alpha=0.2, color='#1f77b4', label='95% CI')
    
    ax1.axhline(y=100, color='darkred', linestyle=':', linewidth=2.5, 
                label='100% of Grid Capacity', alpha=0.7)
    ax1.axhline(y=50, color='orange', linestyle=':', linewidth=2.5, 
                label='50% of Grid Capacity', alpha=0.7)
    
    ax1.set_ylabel('DC Share (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Datacenter Demand as Share of Grid Generation', fontsize=15, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Panel 2: Energy Comparison
    ax2.plot(data['date'], data['dc_energy_gwh'], 
             linewidth=2.5, color='purple', label='Datacenter Energy Demand')
    ax2.plot(data['date'], data['electricity_gwh'], 
             linewidth=2.5, color='gray', linestyle='--', label='Total Grid Generation')
    
    if has_ai_ci:
        ax2.fill_between(data['date'], data['dc_energy_gwh_lower'], data['dc_energy_gwh_upper'],
                         alpha=0.2, color='purple', label='DC Demand 95% CI')
    else:
        ax2.fill_between(data['date'], 0, data['dc_energy_gwh'], 
                         alpha=0.3, color='purple')
    
    ax2.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Energy (GWh)', fontsize=13, fontweight='bold')
    ax2.set_title('Energy Comparison: Datacenter Demand vs Grid Generation',
                  fontsize=15, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'grid_stress_forecast.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved grid_stress_forecast.png")

if __name__ == '__main__':
    plot_grid_stress()
