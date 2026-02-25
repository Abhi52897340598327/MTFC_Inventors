"""
MTFC — Enhanced SARIMAX Model
==============================
Seasonal ARIMA with rich exogenous features for hourly power forecasting.
Uses engineered lag/rolling features to give SARIMAX multivariate learning power.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.preprocessing import StandardScaler
import warnings

import config as cfg
from utils import log, save_pickle, load_pickle, calc_metrics

warnings.filterwarnings("ignore", category=UserWarning)


def prepare_sarimax_features(df: pd.DataFrame) -> tuple:
    """
    Create rich feature set for SARIMAX to compete with XGBoost.
    
    Key insight: SARIMAX struggles because it only sees 4 raw features.
    XGBoost sees 41 engineered features. We'll give SARIMAX similar features.
    """
    exog_df = pd.DataFrame(index=df.index)
    
    # 1. Core features
    exog_df['temperature_f'] = df['temperature_f'].values
    exog_df['is_weekend'] = df['is_weekend'].values if 'is_weekend' in df.columns else 0
    
    # 2. Temporal features (cyclical encoding for better SARIMAX handling)
    if 'hour' in df.columns:
        exog_df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        exog_df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # 3. Temperature-derived features (captures nonlinearity)
    temp = df['temperature_f'].values
    exog_df['temp_squared'] = temp ** 2  # Quadratic effect for cooling
    exog_df['temp_above_65'] = np.maximum(temp - 65, 0)  # Cooling degree threshold
    exog_df['temp_above_80'] = np.maximum(temp - 80, 0)  # High stress threshold
    
    # 4. Lag features (give SARIMAX memory of recent values)
    target = cfg.TARGET_COL
    if target in df.columns:
        power = df[target].values
        for lag in [1, 2, 3, 6, 12, 24]:
            col_name = f'power_lag_{lag}'
            lagged = np.roll(power, lag)
            lagged[:lag] = power[:lag].mean()  # Fill initial lags with mean
            exog_df[col_name] = lagged
        
        # 5. Rolling statistics
        power_series = pd.Series(power)
        exog_df['power_roll_6h_mean'] = power_series.rolling(6, min_periods=1).mean().values
        exog_df['power_roll_24h_mean'] = power_series.rolling(24, min_periods=1).mean().values
        exog_df['power_roll_24h_std'] = power_series.rolling(24, min_periods=1).std().fillna(0).values
    
    # 6. Carbon intensity if available
    if 'carbon_intensity' in df.columns:
        exog_df['carbon_intensity'] = df['carbon_intensity'].values
    
    # 7. Business hour indicator
    if 'is_business_hour' in df.columns:
        exog_df['is_business_hour'] = df['is_business_hour'].values
    elif 'hour' in df.columns:
        exog_df['is_business_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
    
    return exog_df


def train_sarimax(train: pd.DataFrame, val: pd.DataFrame):
    """
    Train enhanced SARIMAX with rich exogenous features.
    
    Returns
    -------
    results, scaler, feature_cols : model, StandardScaler, list of feature names
    """
    import time
    start_time = time.time()
    
    target = cfg.TARGET_COL
    
    # Prepare rich features
    log.info("  [1/6] Preparing exogenous features...")
    exog_train = prepare_sarimax_features(train)
    feature_cols = list(exog_train.columns)
    log.info(f"  [2/6] Feature matrix shape: {exog_train.shape}")
    
    # Standardize features for better SARIMAX convergence
    log.info("  [3/6] Standardizing features...")
    scaler = StandardScaler()
    exog_train_scaled = scaler.fit_transform(exog_train)
    
    y_train = train[target].values

    log.info(f"Training Enhanced SARIMAX — order={cfg.SARIMAX_ORDER}, "
             f"seasonal=(1,0,1,24), features={len(feature_cols)}")
    log.info(f"  Features: {feature_cols}")

    # Use simpler seasonal to avoid overfitting with rich exog
    log.info("  [4/6] Building SARIMAX model...")
    model = SARIMAX(
        y_train,
        exog=exog_train_scaled,
        order=cfg.SARIMAX_ORDER,
        seasonal_order=(1, 0, 1, 24),  # Simpler seasonal, let exog handle complexity
        enforce_stationarity=False,
        enforce_invertibility=False,
        initialization='approximate_diffuse',
    )

    log.info("  [5/6] Fitting model (this may take 1-2 minutes)...")
    fit_start = time.time()
    results = model.fit(disp=False, maxiter=500, method='lbfgs')
    fit_time = time.time() - fit_start
    log.info(f"  [5/6] Model fitted in {fit_time:.1f}s")
    log.info(f"  SARIMAX AIC: {results.aic:.1f}, BIC: {results.bic:.1f}")

    # Ljung-Box residual test
    log.info("  [6/6] Running diagnostics...")
    lb = acorr_ljungbox(results.resid, lags=[24], return_df=True)
    log.info(f"  Ljung-Box (lag 24): stat={lb['lb_stat'].values[0]:.2f}, "
             f"p={lb['lb_pvalue'].values[0]:.4f}")
    
    # Residual statistics
    resid = results.resid
    log.info(f"  Residuals: mean={resid.mean():.3f}, std={resid.std():.3f}, "
             f"min={resid.min():.1f}, max={resid.max():.1f}")

    # Save model and scaler
    save_pickle(results, "sarimax_model")
    save_pickle(scaler, "sarimax_scaler")
    save_pickle(feature_cols, "sarimax_feature_cols")
    
    total_time = time.time() - start_time
    log.info(f"  SARIMAX training complete in {total_time:.1f}s")
    
    return results, scaler, feature_cols


def predict_sarimax(results, exog: np.ndarray, steps: int) -> np.ndarray:
    """Forecast `steps` ahead using the fitted SARIMAX model."""
    forecast = results.forecast(steps=steps, exog=exog)
    return np.asarray(forecast)


def evaluate_sarimax(results, train: pd.DataFrame, test: pd.DataFrame, 
                     scaler=None, feature_cols=None) -> dict:
    """Generate predictions on test set and compute metrics."""
    target = cfg.TARGET_COL
    
    # Prepare features for test set
    exog_test = prepare_sarimax_features(test)
    
    # Use only the columns that were used in training
    if feature_cols:
        exog_test = exog_test[feature_cols]
    
    # Scale using the same scaler
    if scaler is not None:
        exog_test_scaled = scaler.transform(exog_test)
    else:
        exog_test_scaled = exog_test.values
    
    y_pred = predict_sarimax(results, exog_test_scaled, len(test))
    y_true = test[target].values

    metrics = calc_metrics(y_true, y_pred)
    log.info(f"SARIMAX test metrics: {metrics}")
    return {"y_pred": y_pred, "y_true": y_true, "metrics": metrics}
