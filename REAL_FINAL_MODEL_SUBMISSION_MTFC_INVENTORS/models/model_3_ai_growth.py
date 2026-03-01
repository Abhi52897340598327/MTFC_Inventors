"""
MTFC Model 3: AI Growth Multiplier Forecast
Exponential fit on datacenter spending proxy with gradual
growth-rate decay (modified exponential / Gompertz-like).

The historical AI proxy (DC construction spending normalized to 2015=1.0)
shows strong exponential growth (~20% CAGR). A pure exponential is fit
to historical data, then the growth rate is allowed to decay gradually
toward a long-run floor, producing a realistic S-curve-like trajectory.

Forecast Horizon: 160 months (~13 years) from 2025-09 to 2038-12
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
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

# Growth rate decay derived from technology diffusion literature:
# Infrastructure technologies (internet, mobile, cloud) show growth-rate
# half-lives of 4-8 years. We use 5 years (midpoint), which produces
# a ~5x multiplier over 13 years — consistent with IEA/McKinsey high
# scenarios for datacenter electricity demand growth.
GROWTH_HALFLIFE_YEARS = 5.0  # Technology adoption curve half-life
GROWTH_DECAY_RATE = np.log(2) / (GROWTH_HALFLIFE_YEARS * 12)  # ~0.0116/month

# Long-run floor: US real GDP growth (~2.5%) + efficiency gains (~1%)
# Source: CBO long-term projections, IEA World Energy Outlook baseline
MIN_MONTHLY_GROWTH = 0.003  # ~3.7% annual


def _exp_growth(t, a, b):
    """Pure exponential: y = a * exp(b * t)"""
    return a * np.exp(b * t)


def run_ai_growth_forecast():
    """
    Forecast AI growth multiplier using exponential curve fitting.

    Process:
    1. Fit exponential curve to historical AI proxy (DC spending)
    2. Extrapolate with gradually decaying growth rate
    3. Normalize so forecast start = 1.0

    The growth rate starts at the historical fitted rate and decays
    exponentially toward MIN_MONTHLY_GROWTH, reflecting the natural
    deceleration of technology adoption curves while maintaining
    the exponential character.

    Returns:
        DataFrame with columns: date, ai_multiplier
    """

    # Load prepared data
    ai = pd.read_csv(INPUT_DIR / 'monthly_ai_proxy.csv')
    ai['date'] = pd.to_datetime(ai['date'])
    ai = ai.set_index('date')
    ai.index.freq = 'MS'  # type: ignore[attr-defined]

    # Fit exponential to historical data
    t_hist = np.arange(len(ai))
    y_hist = ai['ai_proxy'].values

    popt, pcov = curve_fit(_exp_growth, t_hist, y_hist,
                           p0=[1.0, 0.02], maxfev=10000)
    a_fit, b_fit = popt

    # Calculate fit quality
    y_fitted = _exp_growth(t_hist, a_fit, b_fit)
    ss_res = np.sum((y_hist - y_fitted) ** 2)
    ss_tot = np.sum((y_hist - y_hist.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot

    initial_monthly_rate = b_fit  # continuous growth rate
    initial_annual_rate = (np.exp(b_fit * 12) - 1)  # discrete annual

    print("\n" + "=" * 70)
    print(" MODEL 3: AI GROWTH MULTIPLIER FORECAST ".center(70))
    print("=" * 70)
    print(f"\nTraining Period: {ai.index[0].strftime('%Y-%m')} to "
          f"{ai.index[-1].strftime('%Y-%m')} ({len(ai)} months)")
    print(f"Forecast Period: {FORECAST_PERIODS} months")
    print(f"\nHistorical AI Proxy:")
    print(f"  Start: {y_hist[0]:.2f}x  End: {y_hist[-1]:.2f}x")
    print(f"  Total growth: {(y_hist[-1]/y_hist[0] - 1)*100:.0f}%")
    print(f"\nExponential Fit:")
    print(f"  y = {a_fit:.4f} * exp({b_fit:.4f} * t)")
    print(f"  R-squared: {r_squared:.4f}")
    print(f"  Initial monthly growth rate: {initial_monthly_rate*100:.2f}%")
    print(f"  Initial annual growth rate: {initial_annual_rate*100:.1f}%")
    print(f"  Growth rate decay: {GROWTH_DECAY_RATE:.4f}/month")
    print(f"  Long-run floor: {(np.exp(MIN_MONTHLY_GROWTH*12)-1)*100:.1f}% annual")

    # Project forward with decaying growth rate
    # Start from the fitted value at end of training
    last_fitted = _exp_growth(t_hist[-1], a_fit, b_fit)
    projected = [last_fitted]

    for i in range(FORECAST_PERIODS):
        # Monthly growth rate decays from initial toward floor
        decayed_rate = MIN_MONTHLY_GROWTH + \
            (b_fit - MIN_MONTHLY_GROWTH) * np.exp(-GROWTH_DECAY_RATE * i)
        next_val = projected[-1] * np.exp(decayed_rate)
        projected.append(next_val)

    projected = np.array(projected[1:])  # drop seed value

    # Normalize so first forecast month = 1.0
    multiplier_normalized = projected / projected[0]

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

    # Summary stats
    cagr = (multiplier_normalized[-1] / multiplier_normalized[0]) ** \
           (12 / FORECAST_PERIODS) - 1
    # Growth rate at end of forecast
    final_monthly_rate = MIN_MONTHLY_GROWTH + \
        (b_fit - MIN_MONTHLY_GROWTH) * np.exp(-GROWTH_DECAY_RATE * (FORECAST_PERIODS - 1))
    final_annual_rate = np.exp(final_monthly_rate * 12) - 1

    print(f"\nForecast Results:")
    print(f"  Start: {multiplier_normalized[0]:.3f}x")
    print(f"  +5yr:  {multiplier_normalized[min(59, FORECAST_PERIODS-1)]:.3f}x")
    print(f"  +10yr: {multiplier_normalized[min(119, FORECAST_PERIODS-1)]:.3f}x")
    print(f"  End:   {multiplier_normalized[-1]:.3f}x")
    print(f"  CAGR:  {cagr*100:.1f}%")
    print(f"  Growth rate at end: {final_annual_rate*100:.1f}% annual")
    print(f"\n✓ Saved to: {OUTPUT_DIR / 'forecast_ai_multiplier.csv'}")
    print("=" * 70 + "\n")

    return output


if __name__ == '__main__':
    run_ai_growth_forecast()
