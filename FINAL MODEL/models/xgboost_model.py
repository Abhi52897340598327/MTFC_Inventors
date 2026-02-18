"""
MTFC — XGBoost Model
======================
Gradient-boosted trees for hourly power consumption prediction.
Provides feature importance rankings for interpretability.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

import config as cfg
from utils import log, calc_metrics, save_pickle, save_fig


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  feature_names: list = None):
    """
    Train an XGBoost regressor with early stopping on validation set.

    Returns
    -------
    model : xgb.XGBRegressor
    """
    model = xgb.XGBRegressor(
        **cfg.XGB_PARAMS,
        early_stopping_rounds=cfg.XGB_EARLY_STOPPING,
    )

    log.info(f"Training XGBoost — {cfg.XGB_PARAMS['n_estimators']} trees, "
             f"depth={cfg.XGB_PARAMS['max_depth']}")

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50,
    )

    log.info(f"XGBoost best iteration: {model.best_iteration}")
    save_pickle(model, "xgboost_model")
    return model


def predict_xgboost(model, X: np.ndarray) -> np.ndarray:
    """Run XGBoost inference."""
    return model.predict(X)


def evaluate_xgboost(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate XGBoost on the test set."""
    y_pred = predict_xgboost(model, X_test)
    metrics = calc_metrics(y_test, y_pred)
    log.info(f"XGBoost test metrics: {metrics}")
    return {"y_pred": y_pred, "y_true": y_test, "metrics": metrics}


def plot_feature_importance(model, feature_names: list, top_n: int = 20):
    """Plot top-N feature importances by gain."""
    importance = model.feature_importances_
    idx = np.argsort(importance)[-top_n:]

    names = [feature_names[i] if feature_names else f"f{i}" for i in idx]
    vals = importance[idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(names, vals, color="#2ca02c", edgecolor="white")
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title(f"XGBoost — Top {top_n} Feature Importances")
    return save_fig(fig, "xgboost_feature_importance")
