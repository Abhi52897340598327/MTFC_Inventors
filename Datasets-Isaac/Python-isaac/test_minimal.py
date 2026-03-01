import sys
print("Starting test...", flush=True)

import pandas as pd
print("pandas imported", flush=True)

import matplotlib
print("matplotlib imported", flush=True)

matplotlib.use('Agg')
print("Agg backend set", flush=True)

import matplotlib.pyplot as plt
print("pyplot imported", flush=True)

import numpy as np
print("numpy imported", flush=True)

print("Loading CSV...", flush=True)
df = pd.read_csv('hrl_load_metered_combined_cleaned.csv')
print(f"CSV loaded: {len(df)} rows", flush=True)

print("Creating simple figure...", flush=True)
fig = plt.figure(figsize=(8, 6))
print("Figure created", flush=True)

ax = fig.add_subplot(111)
print("Subplot added", flush=True)

y = df['mw'].values[:1000]
print("Data extracted", flush=True)

ax.plot(y, linewidth=1)
print("Plot created", flush=True)

plt.savefig('Figures-isaac/test_simple.pdf')
print("Figure saved", flush=True)

plt.close()
print("Figure closed", flush=True)

print("SUCCESS!")
