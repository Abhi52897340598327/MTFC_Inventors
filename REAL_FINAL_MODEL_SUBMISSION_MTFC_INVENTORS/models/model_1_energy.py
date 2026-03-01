"""
MTFC Model 1: Total Energy Consumption Forecast
Pure SARIMA model for baseline electricity generation in Virginia.

Uses SARIMA(1,1,1)(1,1,1)₁₂ — no exogenous variables.
Temperature was tested as exogenous but was statistically insignificant
(p=0.51) because the seasonal ARIMA component already captures the same
annual cycle that temperature would predict.

Forecast Horizon: 160 months (~13 years) from 2025-09 to 2038-12
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Directory Configuration
BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR.parent / 'REAL FINAL FILES' / 'prepared_data'
OUTPUT_DIR = BASE_DIR.parent / 'REAL FINAL FILES' / 'model_forecasts'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model Constants
FORECAST_PERIODS = 160  # 2025-09 to 2038-12


def run_energy_forecast():
    """
    Forecast baseline electricity generation using SARIMA.

    Model: SARIMA(1,1,1)(1,1,1)₁₂
    The seasonal component captures Virginia's strong summer/winter
    electricity cycle (~60% swing) directly from the data.
    No exogenous variables — temperature was dropped after testing
    showed p=0.51 (not significant).

    Returns:
        DataFrame with columns: date, electricity_gwh
    """

    # Load prepared data
    energy = pd.read_csv(INPUT_DIR / 'monthly_energy_consumption.csv')
    energy['date'] = pd.to_datetime(energy['date'])
    data = energy.set_index('date')[['electricity_gwh']]
    data.index.freq = 'MS'  # type: ignore[attr-defined]

    # Fit pure SARIMA (no exogenous)
    model = SARIMAX(
        endog=data['electricity_gwh'],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    results = model.fit(disp=False, maxiter=200, method='lbfgs')  # type: ignore[assignment]

    # Generate future dates
    last_date = data.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=FORECAST_PERIODS,
        freq='MS'
    )

    # Generate forecast
    forecast = results.forecast(steps=FORECAST_PERIODS)  # type: ignore[union-attr]

    # Create output dataframe
    output = pd.DataFrame({
        'date': future_dates,
        'electricity_gwh': forecast.values
    })

    # Save to CSV
    output.to_csv(OUTPUT_DIR / 'forecast_energy.csv', index=False)

    # Print summary
    print("\n" + "=" * 70)
    print(" MODEL 1: BASELINE ELECTRICITY FORECAST ".center(70))
    print("=" * 70)
    print(f"\nModel: SARIMA(1,1,1)(1,1,1)₁₂")
    print(f"Exogenous: None (temperature tested, p=0.51, dropped)")
    print(f"Note: AI impact applied separately in integration layer")
    print(f"Training Period: {data.index[0].strftime('%Y-%m')} to {data.index[-1].strftime('%Y-%m')} ({len(data)} months)")
    print(f"Forecast Period: {future_dates[0].strftime('%Y-%m')} to {future_dates[-1].strftime('%Y-%m')} ({FORECAST_PERIODS} months)")
    print(f"\nForecast Results (Baseline Grid):")
    print(f"  Start: {forecast.values[0]:,.1f} GWh")
    print(f"  End:   {forecast.values[-1]:,.1f} GWh")
    print(f"  Growth: {((forecast.values[-1]/forecast.values[0] - 1) * 100):+.1f}%")
    print(f"\n✓ Saved to: {OUTPUT_DIR / 'forecast_energy.csv'}")
    print("=" * 70 + "\n")

    return output


if __name__ == '__main__':
    run_energy_forecast()
