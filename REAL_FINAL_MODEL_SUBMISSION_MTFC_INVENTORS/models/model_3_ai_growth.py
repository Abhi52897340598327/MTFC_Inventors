"""
MTFC Model 3: AI Growth Multiplier Forecast
ARIMA model for year-over-year growth rates with floor constraint.

Forecast Horizon: 180 months (15 years)
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
MIN_ANNUAL_AI_GROWTH = 0.05


def run_ai_growth_forecast():
    """
    Forecast AI growth multiplier using ARIMA on YoY growth rates.
    
    Process:
    1. Calculate year-over-year growth rates
    2. Fit ARIMA(2,0,1) model
    3. Forecast growth rates with 5% floor
    4. Convert to cumulative multiplier
    
    Returns:
        DataFrame with columns: date, ai_multiplier
    """
    
    # Load prepared data
    ai = pd.read_csv(INPUT_DIR / 'monthly_ai_proxy.csv')
    ai['date'] = pd.to_datetime(ai['date'])
    ai = ai.set_index('date')
    ai.index.freq = 'MS'  # type: ignore[attr-defined]
    
    # Calculate year-over-year growth rate
    ai['growth_rate'] = ai['ai_proxy'].pct_change(12)
    ai_growth = ai['growth_rate'].dropna()
    
    print("\n" + "=" * 70)
    print(" MODEL 3: AI GROWTH MULTIPLIER FORECAST ".center(70))
    print("=" * 70)
    print(f"\nTraining Period: 2016-01 to 2023-12 (96 months with YoY growth)")
    print(f"Forecast Period: 2024-01 to 2038-12 (180 months)")
    print(f"\nHistorical Growth Statistics:")
    print(f"  Mean YoY Growth: {ai_growth.mean()*100:.2f}%")
    print(f"  Std Dev: {ai_growth.std()*100:.2f}%")
    print(f"  Min: {ai_growth.min()*100:.2f}%")
    print(f"  Max: {ai_growth.max()*100:.2f}%")
    
    # Fit ARIMA model (no seasonality for tech adoption)
    model = SARIMAX(
        endog=ai_growth,
        order=(2, 0, 1),
        seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    results = model.fit(disp=False, maxiter=200, method='lbfgs')  # type: ignore[assignment]
    
    print(f"\nModel: ARIMA(2,0,1)")
    print(f"AIC: {results.aic:.2f}")  # type: ignore[union-attr]
    print(f"BIC: {results.bic:.2f}")  # type: ignore[union-attr]
    
    # Forecast growth rates
    forecast_growth = results.forecast(steps=FORECAST_PERIODS)  # type: ignore[union-attr]
    
    # Apply floor constraint (minimum 5% annual growth)
    forecast_growth_constrained = np.maximum(forecast_growth, MIN_ANNUAL_AI_GROWTH)
    
    floor_applied = np.sum(forecast_growth < MIN_ANNUAL_AI_GROWTH)
    if floor_applied > 0:
        print(f"\n⚠ Floor constraint applied to {floor_applied} months (min {MIN_ANNUAL_AI_GROWTH*100:.1f}% annual)")
    
    # Convert to cumulative multiplier
    # Start from current normalized value (end of training period)
    last_ai_value = ai['ai_proxy'].iloc[-1]

    multiplier_values = [last_ai_value]

    for i, annual_growth in enumerate(forecast_growth_constrained):
        # Convert YoY growth to monthly compound factor
        # If YoY growth is g, then monthly factor is (1+g)^(1/12)
        monthly_factor = (1 + annual_growth) ** (1/12)
        next_value = multiplier_values[-1] * monthly_factor
        multiplier_values.append(next_value)

    # Remove initial value, keep forecasts only
    multiplier_values = multiplier_values[1:]

    # CRITICAL: Normalize so first forecast month (2024-01) = 1.0
    # This means the multiplier represents growth RELATIVE TO NOW.
    # DATACENTER_BASELINE_SHARE in Model 4 represents current DC share (~25%),
    # so the multiplier must start at 1.0 at the forecast start, not at 2015.
    first_forecast_value = multiplier_values[0]
    multiplier_normalized = np.array(multiplier_values) / first_forecast_value
    
    # Generate future dates
    last_date = ai.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=FORECAST_PERIODS,
        freq='MS'
    )
    
    # Create output dataframe
    output = pd.DataFrame({
        'date': future_dates,
        'ai_multiplier': multiplier_normalized
    })
    
    # Save to CSV
    output.to_csv(OUTPUT_DIR / 'forecast_ai_multiplier.csv', index=False)
    
    # Print summary
    print(f"\nForecast Results:")
    print(f"  2024-01: {multiplier_normalized[0]:.3f}x")
    print(f"  2038-12: {multiplier_normalized[-1]:.3f}x")
    print(f"  Total Growth: {((multiplier_normalized[-1]/multiplier_normalized[0] - 1)*100):.1f}%")
    print(f"  CAGR: {((multiplier_normalized[-1]/multiplier_normalized[0])**(1/15) - 1)*100:.2f}%")
    print(f"\n✓ Saved to: {OUTPUT_DIR / 'forecast_ai_multiplier.csv'}")
    print("=" * 70 + "\n")
    
    return output


if __name__ == '__main__':
    run_ai_growth_forecast()
