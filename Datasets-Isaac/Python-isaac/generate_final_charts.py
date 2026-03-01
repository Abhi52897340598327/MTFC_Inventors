"""
Simplified Time Series Analysis - Two key charts:
1. Hourly time series with linear regression
2. Load distribution with 90th percentile line
"""

# quick sanity check to ensure script is executing
print("SCRIPT START")

import pandas as pd
import matplotlib
matplotlib.use('Agg')  
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['lines.linewidth'] = 1.5
matplotlib.rcParams['axes.linewidth'] = 1.2
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use('seaborn-v0_8-darkgrid')

print("Loading data with parsed dates...")
print("about to call read_csv")
# build absolute path for data to allow running from any cwd
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'hrl_load_metered_combined_cleaned.csv')
print(f"reading from {csv_path}")
# specify usecols to reduce memory if file is huge
df = pd.read_csv(csv_path, 
                 usecols=['datetime_beginning_utc','mw'],
                 parse_dates=['datetime_beginning_utc'], low_memory=False)
print("read_csv returned")
# sorting keeps index consistent for time series
if 'datetime_beginning_utc' in df.columns:
    print("sorting dataframe")
    df.sort_values('datetime_beginning_utc', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print("sorting complete")

print(f"Dataset loaded: {len(df):,} records")
print(f"Date range: {df['datetime_beginning_utc'].min()} to {df['datetime_beginning_utc'].max()}")

# create a sampled subset for plotting to keep rendering fast
if len(df) > 500000:
    step = len(df) // 500000
    df_sample = df.iloc[::step].copy()
    print(f"Using sampled dataframe: every {step}th point -> {len(df_sample):,} rows")
else:
    df_sample = df.copy()
    print(f"Using full dataframe for plotting ({len(df_sample):,} rows)")

# downsample for plotting/regression to avoid long computation
max_points = 200000
if len(df) > max_points:
    print(f"Downsampling dataset for regression/plotting to {max_points} points")
    df_sample = df.iloc[::max(1, len(df)//max_points)].copy()
else:
    df_sample = df.copy()


# use figures directory adjacent to script
fig_dir = os.path.join(script_dir, 'Figures-isaac')
os.makedirs(fig_dir, exist_ok=True)
print("\nGenerating charts in", fig_dir, "...")

# run in fast mode to avoid headless rendering hangs; only forecast
# Turning FAST_MODE off now that dependencies are installed and plots are
# simplified.  When troubleshooting we previously skipped the heavy charts,
# but the script is stable now so run everything.
FAST_MODE = False
if FAST_MODE:
    print("FAST MODE enabled: skipping base charts (1–4) and proceeding to forecast.")
else:
    # ============================================================================
    # CHART 1: HOURLY TIME SERIES WITH LINEAR REGRESSION
    # ============================================================================
    print("Generating Chart 1...")
    fig = plt.figure(figsize=(16, 7), facecolor='white')
    ax = fig.add_subplot(111, facecolor='#f8f9fa')

    # Plot the hourly time series (use sample if available)
    x = np.arange(len(df_sample))
    y = df_sample['mw'].values
    ax.plot(x, y, linewidth=1.2, color='#2E86AB', alpha=0.8, label='Hourly Load', zorder=2)

    # Calculate and plot linear regression using numpy polyfit
    print("  computing linear regression with", len(x), "points")
    coeffs = np.polyfit(x, y, 1)  # 1st degree polynomial (linear)
    poly = np.poly1d(coeffs)
    regression_line = poly(x)

    # compute coefficient of determination (R²)
    residuals = y - regression_line
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else float('nan')
    print(f"  regression slope={coeffs[0]:.6f}, intercept={coeffs[1]:.2f}, R²={r_squared:.6f}")

    ax.plot(x, regression_line, linewidth=3, color='#A23B72', label='Linear Trend', zorder=3, linestyle='--')

    ax.set_xlabel('Hour Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Load (MW)', fontsize=13, fontweight='bold')
    ax.set_title('Hourly Electricity Load Time Series with Linear Regression', fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    print("Saving Chart 1...")
    plt.savefig('Figures-isaac/01_hourly_timeseries_regression.png', format='png', dpi=100)
    plt.close()
    print("Saved: 01_hourly_timeseries_regression.png")

    # ============================================================================
    # CHART 2: LOAD DISTRIBUTION WITH 90TH PERCENTILE
    # ============================================================================
    print("Generating Chart 2...")
    percentile_90 = np.percentile(df['mw'], 90)

    fig = plt.figure(figsize=(12, 7), facecolor='white')
    ax = fig.add_subplot(111, facecolor='#f8f9fa')

    # Histogram
    ax.hist(df['mw'], bins=50, color='#06A77D', alpha=0.75, edgecolor='#04523E', linewidth=0.5)

    # Vertical line at 90th percentile
    ax.axvline(x=percentile_90, color='#D62828', linewidth=3, linestyle='--', 
               alpha=0.9)

    ax.set_xlabel('Load (MW)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax.set_title('Virginia Load Distribution', fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    print("Saving Chart 2...")
    plt.savefig('Figures-isaac/02_load_distribution_90th.png', format='png', dpi=100)
    plt.close()
    print("Saved: 02_load_distribution_90th.png")

    # ============================================================================
    # CHART 3: HOURLY PATTERN - HIGHEST HOURS OF STRAIN THROUGHOUT THE DAY
    # ============================================================================
    print("Processing hourly pattern...")
    df['hour'] = df['datetime_beginning_utc'].dt.hour
    hourly_avg = df.groupby('hour')['mw'].mean()
    max_hour_idx = hourly_avg.idxmax()
    max_hour_load = hourly_avg.max()

    print("Generating hourly chart...")
    fig = plt.figure(figsize=(14, 7), facecolor='white')
    ax = fig.add_subplot(111, facecolor='#f8f9fa')

    # Create bar chart with peak hour highlighted
    colors = ['#D62828' if i == max_hour_idx else '#4A90E2' for i in hourly_avg.index]
    bars = ax.bar(hourly_avg.index, hourly_avg.values, color=colors, alpha=0.8, edgecolor='#333333', linewidth=1)

    # Add value label on peak hour
    for i, (idx, val) in enumerate(hourly_avg.items()):
        if idx == max_hour_idx:
            ax.text(idx, val + 200, f'{val:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_xlabel('Hour of Day (UTC)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Load (MW)', fontsize=13, fontweight='bold')
    ax.set_title('Hourly Load Pattern - Highest Strain Throughout the Day', fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(range(0, 24))
    ax.grid(True, alpha=0.4, axis='y', linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    print("Saving hourly chart...")
    plt.savefig('Figures-isaac/03_hourly_strain_pattern.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 03_hourly_strain_pattern.pdf")

    # ============================================================================
    # CHART 4: MONTHLY PATTERN - HIGHEST MONTHS OF STRAIN THROUGHOUT THE YEAR
    # ============================================================================
    print("Processing monthly pattern...")
    df['month'] = df['datetime_beginning_utc'].dt.month
    monthly_avg = df.groupby('month')['mw'].mean()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    max_month_idx = monthly_avg.idxmax()
    max_month_load = monthly_avg.max()

    print("Generating monthly chart...")
    fig = plt.figure(figsize=(14, 7), facecolor='white')
    ax = fig.add_subplot(111, facecolor='#f8f9fa')

    # Create bar chart with peak month highlighted
    colors = ['#D62828' if i == max_month_idx else '#F77F00' for i in monthly_avg.index]
    bars = ax.bar(monthly_avg.index, monthly_avg.values, color=colors, alpha=0.8, edgecolor='#333333', linewidth=1)

    # Add value label on peak month
    for i, (idx, val) in enumerate(monthly_avg.items()):
        if idx == max_month_idx:
            ax.text(idx, val + 200, f'{val:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_xlabel('Month', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Load (MW)', fontsize=13, fontweight='bold')
    ax.set_title('Monthly Load Pattern - Highest Strain Throughout the Year', fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.grid(True, alpha=0.4, axis='y', linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    print("Saving monthly chart...")
    plt.savefig('Figures-isaac/04_monthly_strain_pattern.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 04_monthly_strain_pattern.pdf")



# ============================================================================
# PRINT SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Total Records: {len(df):,} (sample used for charts: {len(df_sample):,})")
print(f"Date Range: {df['datetime_beginning_utc'].min()} to {df['datetime_beginning_utc'].max()}")
print(f"\nLoad (MW):")
print(f"  Mean: {df['mw'].mean():,.2f}")
print(f"  Median: {df['mw'].median():,.2f}")
print(f"  Std Dev: {df['mw'].std():,.2f}")
print(f"  Min: {df['mw'].min():,.2f}")
print(f"  Max: {df['mw'].max():,.2f}")
print(f"  90th Percentile: {percentile_90:,.2f}")
print(f"\nLinear Regression:")
print(f"  Slope: {coeffs[0]:.6f} MW/hour")
print(f"  R²: {r_squared:.6f}")

# Hourly and monthly statistics
print(f"\nHourly Pattern:")
print(f"  Peak Hour: {max_hour_idx}:00 UTC with {max_hour_load:,.2f} MW average")
print(f"  Minimum Hour: {hourly_avg.idxmin()}:00 UTC with {hourly_avg.min():,.2f} MW average")
print(f"  Variation: {max_hour_load - hourly_avg.min():,.2f} MW")

print(f"\nMonthly Pattern:")
print(f"  Peak Month: {month_names[max_month_idx - 1]} with {max_month_load:,.2f} MW average")
print(f"  Minimum Month: {month_names[monthly_avg.idxmin() - 1]} with {monthly_avg.min():,.2f} MW average")
print(f"  Variation: {max_month_load - monthly_avg.min():,.2f} MW")

print("\n" + "="*60)
print("✓ All 4 charts generated and saved to Figures-isaac/")
print("="*60)

# ---------------------------------------------------------------------------
# FORECAST & PLOT (SARIMAX if possible, linear fallback otherwise)
# ---------------------------------------------------------------------------
# build the monthly Virginia series regardless
# for forecast we need full monthly series from full data
va = df.copy()
va_monthly = (
    va.set_index('datetime_beginning_utc')
      .mw.resample('M').sum()
      .div(1000.0)
)

# attempt import ahead of time
have_sarimax = True
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:
    have_sarimax = False

if have_sarimax:
    print("\nBuilding SARIMAX forecast…")
    logy = np.log(va_monthly.clip(lower=1))
    ex = pd.DataFrame(index=logy.index)
    ex['sin_m'] = np.sin(2 * np.pi * ex.index.month / 12)
    ex['cos_m'] = np.cos(2 * np.pi * ex.index.month / 12)
    ex['year'] = ex.index.year
    mod = SARIMAX(logy, exog=ex, order=(1,1,1),
                  seasonal_order=(1,1,1,12),
                  enforce_stationarity=False,
                  enforce_invertibility=False)
    res_m = mod.fit(disp=False)
    last = logy.index[-1]
    horizon = pd.date_range(last + pd.offsets.MonthEnd(1),
                             periods=(2035 - last.year) * 12,
                             freq='M')
    fut_ex = pd.DataFrame(index=horizon)
    fut_ex['sin_m'] = np.sin(2 * np.pi * fut_ex.index.month / 12)
    fut_ex['cos_m'] = np.cos(2 * np.pi * fut_ex.index.month / 12)
    fut_ex['year'] = fut_ex.index.year
    fc = res_m.get_forecast(steps=len(horizon), exog=fut_ex)
    mean = np.exp(fc.predicted_mean)
    ci = fc.conf_int()
    lower = np.exp(ci.iloc[:,0])
    upper = np.exp(ci.iloc[:,1])
else:
    print("\nstatsmodels not installed; using linear-trend fallback.")
    # simple linear extrapolation on log scale
    x = np.arange(len(va_monthly))
    y = np.log(va_monthly.clip(lower=1)).values
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs
    last = va_monthly.index[-1]
    horizon = pd.date_range(last + pd.offsets.MonthEnd(1),
                             periods=(2035 - last.year) * 12,
                             freq='M')
    steps = len(horizon)
    pred_x = np.arange(len(x), len(x) + steps)
    mean = np.exp(intercept + slope * pred_x)
    # rough 95% band ±2 std of residuals
    resid = y - (intercept + slope * x)
    sd = np.std(resid)
    lower = np.exp(intercept + slope * pred_x - 2*sd)
    upper = np.exp(intercept + slope * pred_x + 2*sd)

# plot results
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(va_monthly.index, va_monthly, label='Historical (GWh)')
ax.plot(horizon, mean, label='Forecast', color='#d62728')
ax.fill_between(horizon, lower, upper, color='#d62728', alpha=0.3,
                label='95% CI')
ax.set_title('VA Monthly Consumption & Forecast')
ax.set_ylabel('GWh')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir,'05_sarimax_forecast.pdf'))
plt.close()
print("Saved: 05_sarimax_forecast.pdf")

