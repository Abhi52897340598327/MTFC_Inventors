"""
Visualization: Carbon Intensity Forecast
Line plot showing grid carbon intensity evolution with floor reference.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / 'REAL FINAL FILES' / 'model_forecasts'
OUTPUT_DIR = BASE_DIR / 'REAL FINAL FILES' / 'visualizations'

def plot_carbon_intensity():
    """Generate carbon intensity line plot with floor reference."""
    data = pd.read_csv(INPUT_DIR / 'forecast_carbon_intensity.csv')
    data['date'] = pd.to_datetime(data['date'])
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(data['date'], data['carbon_intensity'], 
            linewidth=2.5, color='darkred', label='Carbon Intensity')
    
    ax.axhline(y=0.03, color='green', linestyle=':', linewidth=2.5, 
               label='Lifecycle Floor (100% renewables)', alpha=0.7)
    
    ax.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax.set_ylabel('Carbon Intensity (lb CO₂/kWh)', fontsize=13, fontweight='bold')
    ax.set_title('Grid Carbon Intensity Forecast (2024-2038)', 
                 fontsize=15, fontweight='bold', pad=20)
    
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'carbon_intensity_forecast.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved carbon_intensity_forecast.png")

if __name__ == '__main__':
    plot_carbon_intensity()
