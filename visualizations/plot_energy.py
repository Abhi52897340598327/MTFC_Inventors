"""
Visualization: Total Energy Consumption Forecast
Line plot showing 15-year energy consumption trajectory.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / 'REAL FINAL FILES' / 'model_forecasts'
OUTPUT_DIR = BASE_DIR / 'REAL FINAL FILES' / 'visualizations'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_energy_forecast():
    """Generate energy forecast line plot."""
    data = pd.read_csv(INPUT_DIR / 'forecast_energy.csv')
    data['date'] = pd.to_datetime(data['date'])
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(data['date'], data['electricity_gwh'], 
            linewidth=2.5, color='#1f77b4', label='Electricity Consumption')
    
    ax.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax.set_ylabel('Electricity Consumption (GWh)', fontsize=13, fontweight='bold')
    ax.set_title('Virginia Electricity Consumption Forecast (2024-2038)', 
                 fontsize=15, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'energy_forecast.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved energy_forecast.png")

if __name__ == '__main__':
    plot_energy_forecast()
