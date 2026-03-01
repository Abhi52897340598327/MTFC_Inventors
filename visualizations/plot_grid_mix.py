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
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.stackplot(
        data['date'],
        data['coal_pct'],
        data['gas_pct'],
        data['nuclear_pct'],
        data['renewable_pct'],
        labels=['Coal', 'Natural Gas', 'Nuclear', 'Renewables'],
        colors=['#8B4513', '#FFD700', '#4169E1', '#32CD32'],
        alpha=0.85
    )
    
    ax.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax.set_ylabel('Grid Composition (%)', fontsize=13, fontweight='bold')
    ax.set_title('Virginia Grid Mix Evolution Forecast (2024-2038)', 
                 fontsize=15, fontweight='bold', pad=20)
    
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'grid_mix_forecast.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved grid_mix_forecast.png")

if __name__ == '__main__':
    plot_grid_mix()
