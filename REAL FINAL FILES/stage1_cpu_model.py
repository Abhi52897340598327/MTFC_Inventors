"""Pure Random Forest wrapper for Stage 1 CPU forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from utils import calc_metrics, get_logger, save_pickle


DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}


def build_model(params: Optional[Dict] = None) -> RandomForestRegressor:
    """Build a RandomForestRegressor from defaults + overrides."""
    merged = dict(DEFAULT_PARAMS)
    if params:
        merged.update(params)
    return RandomForestRegressor(**merged)


def fit(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    params: Optional[Dict] = None,
    save_artifacts: bool = False,
    artifact_dir: Optional[Path] = None,
    artifact_name: str = "stage1_cpu_random_forest",
) -> Tuple[RandomForestRegressor, Optional[Dict[str, float]]]:
    """Fit the model and optionally evaluate on validation data."""
    log = get_logger("Stage1CPU")
    model = build_model(params=params)
    model.fit(X_train, y_train)

    val_metrics = None
    if X_val is not None and y_val is not None and len(y_val) > 0:
        y_val_pred = model.predict(X_val)
        val_metrics = calc_metrics(y_val, y_val_pred)
        log.info(
            "CPU RF validation metrics: RMSE=%.6f R2=%.6f",
            val_metrics["RMSE"],
            val_metrics["R2"],
        )

    if save_artifacts:
        if artifact_dir is None:
            raise ValueError("artifact_dir is required when save_artifacts=True")
        save_pickle(model, artifact_dir / f"{artifact_name}.pkl")

    return model, val_metrics


def predict(model: RandomForestRegressor, X: np.ndarray) -> np.ndarray:
    """Inference helper."""
    return model.predict(X)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Evaluation helper."""
    return calc_metrics(y_true, y_pred)


__all__ = ["build_model", "fit", "predict", "evaluate", "DEFAULT_PARAMS"]
