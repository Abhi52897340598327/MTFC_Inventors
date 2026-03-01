import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

print('forecast_only start', flush=True)

csv_path = '..\\hrl_load_metered_combined_cleaned.csv'
if not os.path.exists(csv_path):
    csv_path = 'hrl_load_metered_combined_cleaned.csv'

print('reading', csv_path, flush=True)
# read only needed columns and disable low-memory to speed parsing
df = pd.read_csv(csv_path, usecols=['datetime_beginning_utc','mw'], 
                 parse_dates=['datetime_beginning_utc'], low_memory=False)
print('rows', len(df), flush=True)

# monthly GWh
va_monthly = df.set_index('datetime_beginning_utc').mw.resample('M').sum().div(1000.0)
va_monthly = va_monthly.dropna()

# linear fit on monthly
x = np.arange(len(va_monthly))
y = va_monthly.values
if len(x) < 2:
    print('not enough data', flush=True)
else:
    m, b = np.polyfit(x, y, 1)
    last = va_monthly.index[-1]
    periods = (2035 - last.year) * 12
    periods = max(periods, 36)
    future_idx = np.arange(len(x), len(x) + periods)
    future_dates = pd.date_range(last + pd.offsets.MonthEnd(1), periods=periods, freq='M')
    forecast = m * future_idx + b

    outdir = os.path.join('..', 'Figures-isaac')
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, '05_linear_forecast.png')

    plt.figure(figsize=(10,5))
    plt.plot(va_monthly.index, va_monthly.values, label='Historical (GWh)')
    plt.plot(future_dates, forecast, label='Linear forecast', color='red')
    plt.title('VA Monthly Consumption & Fast Linear Forecast')
    plt.ylabel('GWh')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print('wrote', out, flush=True)

print('done', flush=True)
