"""
MTFC Model 1: Total Energy Consumption Forecast
SARIMAX model with month, PUE, and temperature as exogenous variables.

Forecast Horizon: 180 months (15 years) from 2024-01 to 2038-12
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Directory Configuration
BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / 'REAL FINAL FILES' / 'prepared_data'
OUTPUT_DIR = BASE_DIR / 'REAL FINAL FILES' / 'model_forecasts'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model Constants
FORECAST_PERIODS = 180
PUE_FUTURE = 1.15
WARMING_RATE_PER_DECADE = 0.5


def run_energy_forecast():
    """
    Forecast baseline electricity consumption (grid load) using SARIMAX.
    
    Model: SARIMAX(1,1,1)(1,1,1)₁₂
    Exogenous: month, pue, temperature (NO AI - that's applied separately)
    
    Returns:
        DataFrame with columns: date, electricity_gwh
    """
    
    # Load prepared data
    energy = pd.read_csv(INPUT_DIR / 'monthly_energy_consumption.csv')
    pue = pd.read_csv(INPUT_DIR / 'monthly_pue.csv')
    temp = pd.read_csv(INPUT_DIR / 'monthly_temperature.csv')
    
    # Merge datasets - Use ELECTRICITY only (what datacenters consume)
    data = energy[['date', 'electricity_gwh']].merge(pue, on='date').merge(temp, on='date')
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    data.index.freq = 'MS'  # type: ignore[attr-defined]
    
    # Extract month feature
    data['month'] = data.index.month  # type: ignore[attr-defined]
    
    # Fit SARIMAX model WITHOUT AI (AI multiplier applied in integration layer)
    model = SARIMAX(
        endog=data['electricity_gwh'],
        exog=data[['month', 'pue', 'temperature']],
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
    
    # Create future exogenous variables
    future_months = np.tile(np.arange(1, 13), 15)  # Repeat 1-12 for 15 years
    future_pue = np.full(FORECAST_PERIODS, PUE_FUTURE)
    
    # Future temperature with climate warming
    historical_avg_temp = data['temperature'].mean()
    
    future_temp = []
    for i in range(FORECAST_PERIODS):
        month_idx = (i % 12) + 1
        year_offset = i / 12.0
        
        # Historical seasonal pattern for this month
        seasonal_temp = data[data.index.month == month_idx]['temperature'].mean()  # type: ignore[attr-defined]
        
        # Add warming trend
        warming = WARMING_RATE_PER_DECADE * (year_offset / 10.0)
        
        future_temp.append(seasonal_temp + warming)
    
    future_temp = np.array(future_temp)
    
    # Assemble exogenous dataframe (NO AI - applied separately in integration)
    future_exog = pd.DataFrame({
        'month': future_months,
        'pue': future_pue,
        'temperature': future_temp
    }, index=future_dates)
    
    # Generate forecast
    forecast = results.forecast(steps=FORECAST_PERIODS, exog=future_exog)  # type: ignore[union-attr]
    
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
    print(f"\nModel: SARIMAX(1,1,1)(1,1,1)₁₂")
    print(f"Exogenous Variables: month, PUE, temperature")
    print(f"Note: AI impact applied separately in integration layer")
    print(f"Training Period: 2015-01 to 2023-12 (108 months)")
    print(f"Forecast Period: 2024-01 to 2038-12 (180 months)")
    print(f"\nFuture Assumptions:")
    print(f"  PUE: {PUE_FUTURE} (constant)")
    print(f"  Temperature: +{WARMING_RATE_PER_DECADE}°F per decade")
    print(f"  Historical avg: {historical_avg_temp:.1f}°F")
    print(f"  Future avg: {future_temp.mean():.1f}°F")
    print(f"\nForecast Results (Baseline Grid):")
    print(f"  2024-01: {forecast.values[0]:,.1f} GWh")
    print(f"  2038-12: {forecast.values[-1]:,.1f} GWh")
    print(f"  Growth: {((forecast.values[-1]/forecast.values[0] - 1) * 100):.1f}%")
    print(f"\n✓ Saved to: {OUTPUT_DIR / 'forecast_energy.csv'}")
    print("=" * 70 + "\n")
    
    return output


if __name__ == '__main__':
    run_energy_forecast()
