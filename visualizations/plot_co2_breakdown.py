"""
Visualization: CO2 Emissions by Source
Stacked area chart showing CO2 contributions from each energy source.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / 'REAL FINAL FILES' / 'model_forecasts'
OUTPUT_DIR = BASE_DIR / 'REAL FINAL FILES' / 'visualizations'

def plot_co2_breakdown():
    data = pd.read_csv(INPUT_DIR / 'forecast_co2_emissions.csv')
    data['date'] = pd.to_datetime(data['date'])
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.stackplot(
        data['date'],
        data['co2_coal_tons'],
        data['co2_gas_tons'],
        data['co2_nuclear_tons'],
        data['co2_renewable_tons'],
        labels=['Coal', 'Natural Gas', 'Nuclear', 'Renewables'],
        colors=['#8B4513', '#FFD700', '#4169E1', '#32CD32'],
        alpha=0.85
    )
    
    ax.plot(data['date'], data['co2_total_tons'], 
            linewidth=3, color='black', linestyle='--', alpha=0.7,
            label='Total Emissions')
    
    ax.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax.set_ylabel('CO₂ Emissions (tons)', fontsize=13, fontweight='bold')
    ax.set_title('Datacenter CO₂ Emissions by Source (2024-2038)', 
                 fontsize=15, fontweight='bold', pad=20)
    
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Format y-axis for thousands separator
    from matplotlib.ticker import FuncFormatter
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'co2_breakdown_forecast.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved co2_breakdown_forecast.png")

if __name__ == '__main__':
    plot_co2_breakdown()
