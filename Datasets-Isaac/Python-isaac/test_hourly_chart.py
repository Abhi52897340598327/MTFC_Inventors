import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

print("Loading data...")
df = pd.read_csv('hrl_load_metered_combined_cleaned.csv')
df['datetime_beginning_utc'] = pd.to_datetime(df['datetime_beginning_utc'])

print("Extracting hourly patterns...")
df['hour'] = df['datetime_beginning_utc'].dt.hour
hourly_avg = df.groupby('hour')['mw'].mean()
print(f"Hourly averages computed: {hourly_avg.to_dict()}")

print("Creating figure...")
fig, ax = plt.subplots(figsize=(14, 6))
print("Creating bar chart...")
bars = ax.bar(hourly_avg.index, hourly_avg.values, color='#2ca02c', alpha=0.7)

print("Highlighting peak...")
max_hour_idx = hourly_avg.idxmax()
max_hour_load = hourly_avg.max()
bars[max_hour_idx].set_color('#d62728')

ax.set_xlabel('Hour of Day (UTC)', fontsize=12)
ax.set_ylabel('Average Load (MW)', fontsize=12)
ax.set_title('Hourly Load Pattern', fontsize=14, fontweight='bold')
ax.set_xticks(range(0, 24))

print("Saving figure...")
plt.savefig('Figures-isaac/test_hourly.pdf')
print("✓ Test chart saved")
plt.close()

print("Done!")
