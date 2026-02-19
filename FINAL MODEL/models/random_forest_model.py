"""
MTFC — Random Forest Model
===========================
Random Forest regressor as SARIMAX replacement.
Fast, accurate, and handles nonlinear relationships well.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import config as cfg
from utils import log, save_pickle, calc_metrics


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        feature_names: list = None) -> RandomForestRegressor:
    """
    Train Random Forest regressor.
    
    Parameters
    ----------
    X_train, y_train : training data
    X_val, y_val : validation data (for logging)
    feature_names : list of feature names for importance analysis
    
    Returns
    -------
    model : trained RandomForestRegressor
    """
    log.info("Training Random Forest — 100 trees, max_depth=10")
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=cfg.RANDOM_SEED,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    
    # Validation metrics
    y_val_pred = model.predict(X_val)
    val_metrics = calc_metrics(y_val, y_val_pred)
    log.info(f"Random Forest validation: R²={val_metrics['R2']:.4f}, MAE={val_metrics['MAE']:.2f}")
    
    # Feature importance (top 5)
    if feature_names is not None:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:5]
        log.info("Top 5 features: " + ", ".join([f"{feature_names[i]}={importances[i]:.3f}" for i in indices]))
    
    save_pickle(model, "random_forest_model")
    return model


def evaluate_random_forest(model: RandomForestRegressor, 
                           X_test: np.ndarray, 
                           y_test: np.ndarray) -> dict:
    """
    Evaluate Random Forest on test set.
    
    Returns
    -------
    dict with 'y_pred', 'y_true', 'metrics'
    """
    y_pred = model.predict(X_test)
    metrics = calc_metrics(y_test, y_pred)
    log.info(f"Random Forest test metrics: {metrics}")
    
    return {
        "y_pred": y_pred,
        "y_true": y_test,
        "metrics": metrics
    }
