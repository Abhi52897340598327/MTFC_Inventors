"""
Time Series Analysis and Visualization of Hourly Electricity Load Data

This script generates 7 simple time series charts without heavy date rendering.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# Set style
plt.style.use('classic')

print("Loading data...")
df = pd.read_csv('hrl_load_metered_combined_cleaned.csv')
df['datetime_beginning_utc'] = pd.to_datetime(df['datetime_beginning_utc'])
df = df.sort_values('datetime_beginning_utc').reset_index(drop=True)

print(f"Dataset loaded: {len(df):,} records")
print(f"Date range: {df['datetime_beginning_utc'].min()} to {df['datetime_beginning_utc'].max()}")

os.makedirs('Figures-isaac', exist_ok=True)
print("\nGenerating charts...")

# ============================================================================
# CHART 1: FULL TIME SERIES
# ============================================================================
fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(range(len(df)), df['mw'], linewidth=0.8, color='#1f77b4')
ax.set_xlabel('Hour Index', fontsize=12)
ax.set_ylabel('Load (MW)', fontsize=12)
ax.set_title('Full Time Series: Hourly Electricity Load', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Figures-isaac/01_full_timeseries.png', dpi=150)
plt.close()
print("✓ Saved: 01_full_timeseries.png")

# ============================================================================
# CHART 2: FIRST 30 DAYS DETAIL VIEW
# ============================================================================
first_30_days = df[df['datetime_beginning_utc'] <= df['datetime_beginning_utc'].min() + pd.Timedelta(days=30)]
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(range(len(first_30_days)), first_30_days['mw'], 
        linewidth=1.2, color='#ff7f0e', marker='o', markersize=2)
ax.set_xlabel('Hour Index', fontsize=12)
ax.set_ylabel('Load (MW)', fontsize=12)
ax.set_title('First 30 Days: Hourly Load Pattern (Detail View)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Figures-isaac/02_first_30days_detail.png', dpi=150)
plt.close()
print("✓ Saved: 02_first_30days_detail.png")

# ============================================================================
# CHART 3: WEEKLY AVERAGES
# ============================================================================
df['week_index'] = df.index // (24 * 7)
weekly_avg = df.groupby('week_index')['mw'].mean().reset_index()

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(weekly_avg['week_index'], weekly_avg['mw'], linewidth=2, color='#2ca02c', marker='o', markersize=5)
ax.set_xlabel('Week Index', fontsize=12)
ax.set_ylabel('Average Load (MW)', fontsize=12)
ax.set_title('Weekly Average Load Trend', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Figures-isaac/03_weekly_averages.png', dpi=150)
plt.close()
print("✓ Saved: 03_weekly_averages.png")

# ============================================================================
# CHART 4: DAILY AVERAGES WITH MIN/MAX RANGE
# ============================================================================
daily_avg = df.groupby(df['datetime_beginning_utc'].dt.date)['mw'].agg(['mean', 'min', 'max']).reset_index()
daily_avg['date_index'] = range(len(daily_avg))

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(daily_avg['date_index'], daily_avg['mean'], 
        linewidth=1.5, color='#d62728', label='Daily Average', marker='o', markersize=3)
ax.fill_between(daily_avg['date_index'], daily_avg['min'], daily_avg['max'],
                alpha=0.2, color='#d62728', label='Min-Max Range')
ax.set_xlabel('Day Index', fontsize=12)
ax.set_ylabel('Load (MW)', fontsize=12)
ax.set_title('Daily Load: Average with Min/Max Range', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('Figures-isaac/04_daily_averages_range.png', dpi=150)
plt.close()
print("✓ Saved: 04_daily_averages_range.png")

# ============================================================================
# CHART 5: HOURLY PATTERN (AVERAGE LOAD BY HOUR OF DAY)
# ============================================================================
df['hour'] = df['datetime_beginning_utc'].dt.hour
hourly_pattern = df.groupby('hour')['mw'].agg(['mean', 'std']).reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(hourly_pattern['hour'], hourly_pattern['mean'], 
        linewidth=2.5, color='#9467bd', marker='o', markersize=8)
ax.fill_between(hourly_pattern['hour'],
                hourly_pattern['mean'] - hourly_pattern['std'],
                hourly_pattern['mean'] + hourly_pattern['std'],
                alpha=0.2, color='#9467bd', label='±1 Std Dev')
ax.set_xlabel('Hour of Day (UTC)', fontsize=12)
ax.set_ylabel('Load (MW)', fontsize=12)
ax.set_title('Average Hourly Load Pattern (Daily Cycle)', fontsize=14, fontweight='bold')
ax.set_xticks(range(0, 24, 2))
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('Figures-isaac/05_hourly_pattern.png', dpi=150)
plt.close()
print("✓ Saved: 05_hourly_pattern.png")

# ============================================================================
# CHART 6: DAY OF WEEK PATTERN
# ============================================================================
df['day_of_week'] = df['datetime_beginning_utc'].dt.day_name()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_pattern = df.groupby('day_of_week')['mw'].agg(['mean', 'std']).reindex(day_order).reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
colors = ['#1f77b4' if day not in ['Saturday', 'Sunday'] else '#ff7f0e' 
          for day in daily_pattern['day_of_week']]
ax.bar(range(len(daily_pattern)), daily_pattern['mean'], 
       color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.errorbar(range(len(daily_pattern)), daily_pattern['mean'], yerr=daily_pattern['std'],
            fmt='none', color='black', capsize=5, capthick=2, linewidth=1)
ax.set_xlabel('Day of Week', fontsize=12)
ax.set_ylabel('Average Load (MW)', fontsize=12)
ax.set_title('Average Load by Day of Week (Weekday vs Weekend)', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(daily_pattern)))
ax.set_xticklabels(daily_pattern['day_of_week'], rotation=45)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('Figures-isaac/06_day_of_week_pattern.png', dpi=150)
plt.close()
print("✓ Saved: 06_day_of_week_pattern.png")

# ============================================================================
# CHART 7: LOAD DISTRIBUTION AND VARIABILITY
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax1.hist(df['mw'], bins=50, color='#17becf', alpha=0.7, edgecolor='black')
ax1.set_xlabel('Load (MW)', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Load Distribution (Histogram)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Box plot by hour
hourly_data = [df[df['hour'] == h]['mw'].values for h in range(24)]
ax2.boxplot(hourly_data, labels=range(24))
ax2.set_xlabel('Hour of Day (UTC)', fontsize=12)
ax2.set_ylabel('Load (MW)', fontsize=12)
ax2.set_title('Load Variability by Hour (Box Plot)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('Figures-isaac/07_load_distribution.png', dpi=150)
plt.close()
print("✓ Saved: 07_load_distribution.png")

# ============================================================================
# PRINT SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Total Records: {len(df):,}")
print(f"Date Range: {df['datetime_beginning_utc'].min()} to {df['datetime_beginning_utc'].max()}")
print(f"\nLoad (MW):")
print(f"  Mean: {df['mw'].mean():,.2f}")
print(f"  Median: {df['mw'].median():,.2f}")
print(f"  Std Dev: {df['mw'].std():,.2f}")
print(f"  Min: {df['mw'].min():,.2f}")
print(f"  Max: {df['mw'].max():,.2f}")
print(f"  Range: {df['mw'].max() - df['mw'].min():,.2f}")

verified_count = df['is_verified'].sum()
unverified_count = len(df) - verified_count
print(f"\nVerified vs Unverified:")
print(f"  Verified: {verified_count:,} ({verified_count/len(df)*100:.1f}%)")
print(f"  Unverified: {unverified_count:,} ({unverified_count/len(df)*100:.1f}%)")
print("\n" + "="*60)
print("✓ All charts generated and saved to Figures-isaac/")
print("="*60)
