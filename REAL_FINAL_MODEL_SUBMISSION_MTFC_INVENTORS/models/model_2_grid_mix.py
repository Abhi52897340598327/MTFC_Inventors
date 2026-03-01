"""
MTFC Model 2: Grid Mix Evolution Forecast
Mixed SARIMA/SARIMAX models for each energy source.

Coal, Gas, Nuclear: SARIMA (policy-driven)
Renewable: SARIMAX with AI spending exogenous (PPA-driven)

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
INPUT_DIR = BASE_DIR.parent / 'REAL FINAL FILES' / 'prepared_data'
OUTPUT_DIR = BASE_DIR.parent / 'REAL FINAL FILES' / 'model_forecasts'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model Constants
FORECAST_PERIODS = 160  # 2025-09 to 2038-12
# AI spending projection for renewable exogenous: average forward growth rate
# consistent with Model 3's fitted exponential (~25% initial) decaying with
# a 5-year half-life toward 3.7% floor — averages ~12% over forecast horizon.
AI_SPENDING_GROWTH_PROJECTION = 0.12  # 12% annual (consistent with Model 3)


def run_grid_mix_forecast():
    """
    Forecast grid composition using mixed SARIMA/SARIMAX approach.
    
    Returns:
        DataFrame with normalized grid mix percentages
    """
    
    # Load prepared data
    grid = pd.read_csv(INPUT_DIR / 'monthly_grid_mix.csv')
    ai = pd.read_csv(INPUT_DIR / 'monthly_ai_proxy.csv')
    
    # Merge datasets
    data = grid.merge(ai, on='date')
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    data.index.freq = 'MS'  # type: ignore[attr-defined]
    
    sources = ['coal_pct', 'gas_pct', 'nuclear_pct', 'renewable_pct']
    forecasts = {}
    
    print("\n" + "=" * 70)
    print(" MODEL 2: GRID MIX EVOLUTION FORECAST ".center(70))
    print("=" * 70)
    print(f"\nTraining Period: {data.index[0].strftime('%Y-%m')} to {data.index[-1].strftime('%Y-%m')} ({len(data)} months)")
    print(f"Forecast Period: {FORECAST_PERIODS} months from {data.index[-1].strftime('%Y-%m')}\n")
    
    for source in sources:
        print(f"Fitting {source}...", end=' ')
        
        if source == 'renewable_pct':
            # SARIMAX with AI spending exogenous
            model = SARIMAX(
                endog=data[source],
                exog=data[['ai_proxy']],
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            results = model.fit(disp=False, maxiter=200, method='lbfgs')  # type: ignore[assignment]
            
            # Project future AI spending with exponential growth
            last_ai = data['ai_proxy'].iloc[-1]
            future_ai = np.array([
                last_ai * (1 + AI_SPENDING_GROWTH_PROJECTION) ** (i / 12.0) 
                for i in range(FORECAST_PERIODS)
            ])
            
            future_exog = pd.DataFrame({'ai_proxy': future_ai})
            forecast = results.forecast(steps=FORECAST_PERIODS, exog=future_exog)  # type: ignore[union-attr]
            
            print(f"SARIMAX(1,1,1)(1,1,1)₁₂ with ai_proxy")
            
        else:
            # SARIMA without exogenous
            model = SARIMAX(
                endog=data[source],
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            results = model.fit(disp=False, maxiter=200, method='lbfgs')  # type: ignore[assignment]
            forecast = results.forecast(steps=FORECAST_PERIODS)  # type: ignore[union-attr]
            
            print(f"SARIMA(1,1,1)(1,1,1)₁₂")
        
        forecasts[source] = forecast.values
        print(f"  {data[source].iloc[-1]:.2f}% (2023) → {forecast.values[-1]:.2f}% (2038)")
    
    # Generate future dates
    last_date = data.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=FORECAST_PERIODS,
        freq='MS'
    )
    
    # Create forecast dataframe
    grid_forecast = pd.DataFrame(forecasts, index=future_dates)
    
    # Clip to physical bounds BEFORE normalization
    # (SARIMA can produce negative values for declining trends)
    grid_forecast = grid_forecast.clip(lower=0, upper=100)
    
    # Normalize to enforce compositional constraint (sum = 100%)
    print("\nApplying compositional constraint (normalization)...")
    row_sums = grid_forecast.sum(axis=1)
    grid_forecast = grid_forecast.div(row_sums, axis=0) * 100
    
    # Verify normalization
    verification_sums = grid_forecast.sum(axis=1)
    assert np.allclose(verification_sums, 100.0, atol=1e-6), "Normalization failed"
    print("✓ All rows sum to 100.00%")
    
    # Save individual forecasts
    for source in sources:
        output = pd.DataFrame({
            'date': future_dates,
            source: grid_forecast[source].values
        })
        filename = f"forecast_grid_{source.replace('_pct', '')}.csv"
        output.to_csv(OUTPUT_DIR / filename, index=False)
        print(f"✓ Saved {filename}")
    
    print("\nFinal Grid Composition (2038-12):")
    for source in sources:
        print(f"  {source}: {grid_forecast[source].iloc[-1]:.2f}%")
    
    print("=" * 70 + "\n")
    
    return grid_forecast


if __name__ == '__main__':
    run_grid_mix_forecast()
