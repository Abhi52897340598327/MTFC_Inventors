"""Test Random Forest and generate residual plots."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
from data_loader import load_all
from feature_engineering import engineer_features
from data_preparation import prepare_all
import os

# Load and prepare data
print("Loading data...")
data = load_all()
hourly = data["hourly"]
df = engineer_features(hourly)
prep = prepare_all(df)

X_train, X_val, X_test = prep["X_train"], prep["X_val"], prep["X_test"]
y_train, y_val, y_test = prep["y_train"], prep["y_val"], prep["y_test"]
feature_cols = prep["feature_cols"]

# Train Random Forest
print("Training Random Forest...")
rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)
residuals = y_test - y_pred

# Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nRandom Forest Performance:")
print(f"  R² = {r2:.4f}")
print(f"  MAE = {mae:.4f}")
print(f"  RMSE = {rmse:.4f}")

# Create residual plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Random Forest Residual Analysis", fontsize=14, fontweight='bold')

# 1. Residuals vs Predicted
ax1 = axes[0, 0]
ax1.scatter(y_pred, residuals, alpha=0.5, s=10, c='steelblue')
ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax1.set_xlabel("Predicted Values (MW)")
ax1.set_ylabel("Residuals (MW)")
ax1.set_title("Residuals vs Predicted Values")
ax1.grid(True, alpha=0.3)

# 2. Histogram of Residuals
ax2 = axes[0, 1]
ax2.hist(residuals, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax2.axvline(x=np.mean(residuals), color='orange', linestyle='-', linewidth=2, label=f'Mean: {np.mean(residuals):.3f}')
ax2.set_xlabel("Residuals (MW)")
ax2.set_ylabel("Frequency")
ax2.set_title("Distribution of Residuals")
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Q-Q Plot
ax3 = axes[1, 0]
stats.probplot(residuals, dist="norm", plot=ax3)
ax3.set_title("Q-Q Plot (Normality Check)")
ax3.grid(True, alpha=0.3)

# 4. Actual vs Predicted
ax4 = axes[1, 1]
ax4.scatter(y_test, y_pred, alpha=0.5, s=10, c='steelblue')
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax4.set_xlabel("Actual Values (MW)")
ax4.set_ylabel("Predicted Values (MW)")
ax4.set_title(f"Actual vs Predicted (R² = {r2:.4f})")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs("outputs/figures", exist_ok=True)
plt.savefig("outputs/figures/random_forest_residuals.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"\nResidual Statistics:")
print(f"  Mean: {np.mean(residuals):.4f}")
print(f"  Std:  {np.std(residuals):.4f}")
print(f"  Min:  {np.min(residuals):.4f}")
print(f"  Max:  {np.max(residuals):.4f}")

print("\n✓ Saved: outputs/figures/random_forest_residuals.png")
