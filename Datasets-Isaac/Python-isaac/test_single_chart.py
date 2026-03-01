import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
import numpy as np

# Use seaborn style for better aesthetics
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('default')

print("Loading data...")
df = pd.read_csv('hrl_load_metered_combined_cleaned.csv')

print("Creating hourly timeseries chart...")
fig = plt.figure(figsize=(16, 7), facecolor='white')
ax = fig.add_subplot(111, facecolor='#f8f9fa')

x = np.arange(len(df['mw']))
y = df['mw'].values
ax.plot(x, y, linewidth=1.2, color='#2E86AB', alpha=0.8, label='Hourly Load', zorder=2)

coeffs = np.polyfit(x, y, 1)
poly = np.poly1d(coeffs)
regression_line = poly(x)
ax.plot(x, regression_line, linewidth=3, color='#A23B72', label='Linear Trend', zorder=3, linestyle='--')

ax.set_xlabel('Hour Index', fontsize=13, fontweight='bold')
ax.set_ylabel('Load (MW)', fontsize=13, fontweight='bold')
ax.set_title('Hourly Electricity Load Time Series with Linear Regression', fontsize=15, fontweight='bold', pad=20)
ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
ax.legend(fontsize=12, loc='upper left', framealpha=0.95, shadow=True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig('Figures-isaac/01_hourly_timeseries_regression.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Chart 1 complete")
