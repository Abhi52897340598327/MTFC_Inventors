"""
MTFC — SARIMAX Model
=====================
Seasonal ARIMA with eXogenous variables for hourly power forecasting.
Uses statsmodels SARIMAX with order (1,1,1)(1,1,1)₂₄.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings

import config as cfg
from utils import log, save_pickle, calc_metrics

warnings.filterwarnings("ignore", category=UserWarning)


def train_sarimax(train: pd.DataFrame, val: pd.DataFrame):
    """
    Train SARIMAX on the training set with exogenous variables.

    Because full SARIMAX(1,1,1)(1,1,1)₂₄ on ~6,000 rows is very slow,
    we use a practical approach:
      - Use SARIMAX(1,1,1)(1,0,1,24) which is faster but still captures
        daily seasonality with an SAR(1) + SMA(1) term.
      - Exogenous: temperature_f, hour_of_day, is_weekend

    Returns
    -------
    results : SARIMAXResultsWrapper
    """
    target = cfg.TARGET_COL
    exog_cols = [c for c in cfg.SARIMAX_EXOG_COLS if c in train.columns]

    y_train = train[target].values
    exog_train = train[exog_cols].values if exog_cols else None

    log.info(f"Training SARIMAX — order={cfg.SARIMAX_ORDER}, "
             f"seasonal=(1,0,1,24), exog={exog_cols}")

    # Use slightly simpler seasonal to avoid excessive computation
    model = SARIMAX(
        y_train,
        exog=exog_train,
        order=cfg.SARIMAX_ORDER,
        seasonal_order=(1, 0, 1, 24),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    results = model.fit(disp=False, maxiter=200)
    log.info(f"SARIMAX AIC: {results.aic:.1f}, BIC: {results.bic:.1f}")

    # Ljung-Box residual test
    lb = acorr_ljungbox(results.resid, lags=[24], return_df=True)
    log.info(f"Ljung-Box (lag 24): stat={lb['lb_stat'].values[0]:.2f}, "
             f"p={lb['lb_pvalue'].values[0]:.4f}")

    save_pickle(results, "sarimax_model")
    return results


def predict_sarimax(results, exog: np.ndarray, steps: int) -> np.ndarray:
    """Forecast `steps` ahead using the fitted SARIMAX model."""
    forecast = results.forecast(steps=steps, exog=exog)
    return np.asarray(forecast)


def evaluate_sarimax(results, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Generate predictions on test set and compute metrics."""
    target = cfg.TARGET_COL
    exog_cols = [c for c in cfg.SARIMAX_EXOG_COLS if c in test.columns]

    exog_test = test[exog_cols].values if exog_cols else None
    y_pred = predict_sarimax(results, exog_test, len(test))
    y_true = test[target].values

    metrics = calc_metrics(y_true, y_pred)
    log.info(f"SARIMAX test metrics: {metrics}")
    return {"y_pred": y_pred, "y_true": y_true, "metrics": metrics}
