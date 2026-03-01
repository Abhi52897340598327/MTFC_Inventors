import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

print('quick_plots.py start', flush=True)

os.makedirs('Figures-isaac', exist_ok=True)

# Read CSV (assumes file in same folder)
print('reading CSV...', flush=True)
df = pd.read_csv('hrl_load_metered_combined_cleaned.csv', parse_dates=['datetime_beginning_utc'])
print(f'read {len(df):,} rows', flush=True)

# Simple time series (downsample for speed)
df_sorted = df.sort_values('datetime_beginning_utc')
if len(df_sorted) > 5000:
    step = max(1, len(df_sorted) // 5000)
    ts = df_sorted.iloc[::step]
else:
    ts = df_sorted

plt.figure(figsize=(12,4))
plt.plot(ts['datetime_beginning_utc'], ts['mw'], linewidth=0.8)
plt.title('Hourly Load (sampled)')
plt.ylabel('MW')
plt.xlabel('Datetime')
out1 = 'Figures-isaac/quick_hourly_timeseries.png'
plt.tight_layout()
plt.savefig(out1, dpi=120)
plt.close()
print('wrote', out1, flush=True)

# Histogram
plt.figure(figsize=(8,5))
plt.hist(df['mw'].dropna(), bins=60, color='#2E86AB')
plt.title('Load Distribution')
plt.xlabel('MW')
plt.ylabel('Frequency')
out2 = 'Figures-isaac/quick_load_histogram.png'
plt.tight_layout()
plt.savefig(out2, dpi=120)
plt.close()
print('wrote', out2, flush=True)

# Monthly aggregate and simple linear forecast to 2035
print('building monthly series...', flush=True)
va_monthly = df_sorted.set_index('datetime_beginning_utc').mw.resample('M').sum().div(1000.0)
va_monthly = va_monthly.dropna()

# fit linear trend on index ordinal
y = va_monthly.values
x = np.arange(len(y))
if len(x) >= 2:
    m, b = np.polyfit(x, y, 1)
else:
    m, b = 0.0, float(y.mean()) if len(y)>0 else 0.0

last = va_monthly.index[-1]
periods = (2035 - last.year) * 12
future_idx = np.arange(len(x), len(x) + max(12, periods))
future_dates = pd.date_range(last + pd.offsets.MonthEnd(1), periods=len(future_idx), freq='M')
forecast = m * future_idx + b

plt.figure(figsize=(10,5))
plt.plot(va_monthly.index, va_monthly.values, label='Historical (GWh)')
plt.plot(future_dates, forecast, label='Linear forecast', color='tomato')
plt.title('Monthly Consumption & Simple Forecast')
plt.ylabel('GWh')
plt.legend()
out3 = 'Figures-isaac/quick_monthly_forecast.png'
plt.tight_layout()
plt.savefig(out3, dpi=120)
plt.close()
print('wrote', out3, flush=True)

print('DONE', flush=True)
