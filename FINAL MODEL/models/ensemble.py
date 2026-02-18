"""
MTFC — Ensemble Model
=======================
Inverse-RMSE weighted ensemble of SARIMAX, LSTM, and XGBoost predictions.
Produces confidence intervals from the spread of individual model predictions.
"""

import numpy as np
from utils import log, calc_metrics


def compute_weights(metrics: dict) -> dict:
    """
    Compute inverse-RMSE weights for each model.

    Parameters
    ----------
    metrics : {model_name: {"RMSE": float, ...}}

    Returns
    -------
    weights : {model_name: float}  (sum to 1.0)
    """
    inv_rmse = {k: 1.0 / v["RMSE"] for k, v in metrics.items() if v["RMSE"] > 0}
    total = sum(inv_rmse.values())
    weights = {k: v / total for k, v in inv_rmse.items()}
    log.info(f"Ensemble weights: { {k: f'{v:.3f}' for k, v in weights.items()} }")
    return weights


def ensemble_predict(predictions: dict, weights: dict) -> np.ndarray:
    """
    Weighted average of model predictions.

    Parameters
    ----------
    predictions : {model_name: np.ndarray}
    weights     : {model_name: float}

    Returns
    -------
    y_ens : np.ndarray
    """
    # Align to shortest prediction array
    min_len = min(len(v) for v in predictions.values())
    y_ens = np.zeros(min_len)
    for name, w in weights.items():
        y_ens += w * predictions[name][:min_len]
    return y_ens


def ensemble_confidence_interval(predictions: dict, z: float = 1.96):
    """
    Compute point estimate (mean) and 95 % CI from prediction spread.

    Returns
    -------
    mean, lower, upper : np.ndarray
    """
    min_len = min(len(v) for v in predictions.values())
    stacked = np.column_stack([v[:min_len] for v in predictions.values()])
    mean = stacked.mean(axis=1)
    std  = stacked.std(axis=1)
    return mean, mean - z * std, mean + z * std


def evaluate_ensemble(predictions: dict, y_true: np.ndarray,
                      weights: dict) -> dict:
    """Compute ensemble predictions and metrics."""
    y_ens = ensemble_predict(predictions, weights)
    min_len = min(len(y_ens), len(y_true))
    metrics = calc_metrics(y_true[:min_len], y_ens[:min_len])
    log.info(f"Ensemble test metrics: {metrics}")
    return {"y_pred": y_ens, "y_true": y_true[:min_len], "metrics": metrics}
