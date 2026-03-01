import pandas as pd
import os

CSV = 'hrl_load_metered_combined_cleaned.csv'
if not os.path.exists(CSV):
    raise SystemExit(f"File not found: {CSV}")

# Load the CSV
df = pd.read_csv(CSV)
if 'mw' not in df.columns:
    raise SystemExit("Column 'mw' not found")

# Show a few values before
print("Before scaling (first 5 mw values):")
print(df['mw'].head().tolist())

# Scale by 0.165
df['mw'] = df['mw'] * 0.165

# Show a few values after
print("\nAfter scaling (first 5 mw values):")
print(df['mw'].head().tolist())

# Verify the example: 91971.422 * 0.165
example = 91971.422 * 0.165
print(f"\nExample: 91971.422 * 0.165 = {example}")
print(f"First value in CSV after scaling: {df['mw'].iloc[0]}")

# Save back to the same file
df.to_csv(CSV, index=False)
print(f"\nScaled CSV saved to {CSV}")
