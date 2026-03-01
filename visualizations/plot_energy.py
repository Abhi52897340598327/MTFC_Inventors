"""
Visualization: Total Electricity Generation Forecast
Line plot showing historical (2015-2023) and forecast (2024-2038) generation.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
PREP_DIR = BASE_DIR / 'REAL FINAL FILES' / 'prepared_data'
FCAST_DIR = BASE_DIR / 'REAL FINAL FILES' / 'model_forecasts'
OUTPUT_DIR = BASE_DIR / 'REAL FINAL FILES' / 'visualizations'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_energy_forecast():
    """Generate energy forecast line plot with historical context and 95% CI."""
    hist = pd.read_csv(PREP_DIR / 'monthly_energy_consumption.csv')
    hist['date'] = pd.to_datetime(hist['date'])
    fcst = pd.read_csv(FCAST_DIR / 'forecast_energy.csv')
    fcst['date'] = pd.to_datetime(fcst['date'])

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(hist['date'], hist['electricity_gwh'],
            linewidth=2, color='#333333', label='Historical (EIA Actual)')

    ax.plot(fcst['date'], fcst['electricity_gwh'],
            linewidth=2.5, color='#1f77b4', label='SARIMA Forecast')

    # 95% confidence interval band
    if 'electricity_gwh_lower' in fcst.columns:
        ax.fill_between(fcst['date'],
                        fcst['electricity_gwh_lower'],
                        fcst['electricity_gwh_upper'],
                        alpha=0.2, color='#1f77b4', label='95% CI')

    ax.axvline(x=pd.Timestamp('2024-01-01'), color='gray',
               linestyle='--', linewidth=1.5, alpha=0.6, label='Forecast Start')

    ax.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax.set_ylabel('Electricity Generation (GWh)', fontsize=13, fontweight='bold')
    ax.set_title('Virginia Electricity Generation: Historical & Forecast (2015-2038)',
                 fontsize=15, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'energy_forecast.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved energy_forecast.png")

if __name__ == '__main__':
    plot_energy_forecast()