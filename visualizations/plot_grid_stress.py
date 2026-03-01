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
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Panel 1: Grid Stress
    ax1.plot(data['date'], data['grid_stress_pct'], 
             linewidth=2.5, color='#1f77b4', label='Base Stress')
    ax1.plot(data['date'], data['grid_stress_adjusted_pct'], 
             linewidth=2.5, color='red', linestyle='--', label='Adjusted Stress (with penalty)')
    
    ax1.axhline(y=35, color='orange', linestyle=':', linewidth=2.5, 
                label='Concern Threshold (35%)', alpha=0.7)
    ax1.axhline(y=40, color='darkred', linestyle=':', linewidth=2.5, 
                label='Critical Threshold (40%)', alpha=0.7)
    
    ax1.set_ylabel('Grid Stress (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Datacenter Grid Stress Forecast', fontsize=15, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Panel 2: Energy Comparison
    ax2.plot(data['date'], data['dc_energy_adjusted_gwh'], 
             linewidth=2.5, color='purple', label='Datacenter Energy (adjusted)')
    ax2.plot(data['date'], data['electricity_gwh'], 
             linewidth=2.5, color='gray', linestyle='--', label='Total Grid Electricity')
    
    ax2.fill_between(data['date'], 0, data['dc_energy_adjusted_gwh'], 
                     alpha=0.3, color='purple')
    
    ax2.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Energy (GWh)', fontsize=13, fontweight='bold')
    ax2.set_title('Energy Consumption Comparison', fontsize=15, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'grid_stress_forecast.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved grid_stress_forecast.png")

if __name__ == '__main__':
    plot_grid_stress()
