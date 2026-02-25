"""
Retrain models with proper configuration for PUE prediction.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import data_loader
import feature_engineering
import config as cfg

# Load and prepare data
print('Loading data...')
data = data_loader.load_all()
hourly = data['hourly']
df = feature_engineering.engineer_features(hourly)

# Use simple features (no scaling needed for tree models)
features = ['temperature_f', 'cooling_degree', 'hour', 'month', 'season']
features = [f for f in features if f in df.columns]

X = df[features].values
y = df['pue'].values

# Split chronologically
n = len(X)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

print(f'Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')
print(f'Features: {features}')
print()

# Train XGBoost WITHOUT scaling
print('Training XGBoost (no scaling)...')
model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    verbosity=0
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# Evaluate
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print('=== XGBoost Results (Unscaled) ===')
print(f'R2:   {r2:.4f}')
print(f'MAE:  {mae:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'MAPE: {mape:.2f}%')
print()
print('Feature Importance:')
for f, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
    print(f'  {f}: {imp:.4f}')
