"""
Visualization: Grid Mix Evolution
Stacked area chart showing grid composition changes over time.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / 'REAL FINAL FILES' / 'model_forecasts'
OUTPUT_DIR = BASE_DIR / 'REAL FINAL FILES' / 'visualizations'

def plot_grid_mix():
    coal = pd.read_csv(INPUT_DIR / 'forecast_grid_coal.csv')
    gas = pd.read_csv(INPUT_DIR / 'forecast_grid_gas.csv')
    nuclear = pd.read_csv(INPUT_DIR / 'forecast_grid_nuclear.csv')
    renewable = pd.read_csv(INPUT_DIR / 'forecast_grid_renewable.csv')
    
    data = coal.merge(gas, on='date') \
              .merge(nuclear, on='date') \
              .merge(renewable, on='date')
    
    data['date'] = pd.to_datetime(data['date'])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11))
    
    # Panel 1: Stacked area (point estimates)
    ax1.stackplot(
        data['date'],
        data['coal_pct'],
        data['gas_pct'],
        data['nuclear_pct'],
        data['renewable_pct'],
        labels=['Coal', 'Natural Gas', 'Nuclear', 'Renewables'],
        colors=['#8B4513', '#FFD700', '#4169E1', '#32CD32'],
        alpha=0.85
    )
    
    ax1.set_ylabel('Grid Composition (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Virginia Grid Mix Evolution Forecast (2024-2038)', 
                 fontsize=15, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Panel 2: Individual sources with 95% CI bands
    source_info = [
        ('renewable_pct', '#32CD32', 'Renewables'),
        ('gas_pct', '#FFD700', 'Natural Gas'),
        ('nuclear_pct', '#4169E1', 'Nuclear'),
        ('coal_pct', '#8B4513', 'Coal'),
    ]
    for col, color, label in source_info:
        ax2.plot(data['date'], data[col], linewidth=2, color=color, label=label)
        lower_col = f'{col}_lower'
        upper_col = f'{col}_upper'
        if lower_col in data.columns:
            ax2.fill_between(data['date'], data[lower_col], data[upper_col],
                             alpha=0.15, color=color)
    
    ax2.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Share (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Individual Source Forecasts with 95% CI', 
                 fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=11, framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'grid_mix_forecast.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved grid_mix_forecast.png")

if __name__ == '__main__':
    plot_grid_mix()
