"""Pure XGBoost wrapper for Stage 5 carbon intensity forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from utils import calc_metrics, get_logger, save_pickle

try:
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover
    xgb = None
    _XGB_IMPORT_ERROR = exc
else:
    _XGB_IMPORT_ERROR = None


DEFAULT_PARAMS = {
    "model_type": "xgboost",
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.01,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "random_state": 42,
    "verbosity": 0,
}


def build_model(params: Optional[Dict] = None):
    """Build an XGBRegressor from defaults + overrides."""
    merged = dict(DEFAULT_PARAMS)
    if params:
        merged.update(params)
    model_type = str(merged.pop("model_type", "xgboost")).lower()
    if model_type in {"xgb", "xgboost"}:
        if xgb is None:  # pragma: no cover
            raise ImportError(
                "xgboost is required for stage 5 carbon model. Install xgboost to proceed."
            ) from _XGB_IMPORT_ERROR
        return xgb.XGBRegressor(**merged)
    if model_type in {"et", "extra_trees"}:
        et_params = {
            "n_estimators": int(merged.get("n_estimators", 1200)),
            "max_depth": merged.get("max_depth", None),
            "min_samples_split": int(merged.get("min_samples_split", 2)),
            "min_samples_leaf": int(merged.get("min_samples_leaf", 1)),
            "max_features": merged.get("max_features", "sqrt"),
            "random_state": int(merged.get("random_state", 42)),
            "n_jobs": int(merged.get("n_jobs", -1)),
        }
        return ExtraTreesRegressor(**et_params)
    raise ValueError(f"Unsupported model_type='{model_type}' for stage5 model.")


def fit(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    params: Optional[Dict] = None,
    save_artifacts: bool = False,
    artifact_dir: Optional[Path] = None,
    artifact_name: str = "stage5_carbon_xgboost",
):
    """Fit model and optionally evaluate on validation data."""
    log = get_logger("Stage5Carbon")
    model = build_model(params=params)
    model_is_xgb = bool(xgb is not None and isinstance(model, xgb.XGBRegressor))

    fit_kwargs = {}
    if model_is_xgb and X_val is not None and y_val is not None and len(y_val) > 0:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"] = False

    model.fit(X_train, y_train, **fit_kwargs)

    val_metrics = None
    if X_val is not None and y_val is not None and len(y_val) > 0:
        y_val_pred = model.predict(X_val)
        val_metrics = calc_metrics(y_val, y_val_pred)
        log.info(
            "Carbon XGB validation metrics: RMSE=%.6f R2=%.6f",
            val_metrics["RMSE"],
            val_metrics["R2"],
        )

    if save_artifacts:
        if artifact_dir is None:
            raise ValueError("artifact_dir is required when save_artifacts=True")
        save_pickle(model, artifact_dir / f"{artifact_name}.pkl")

    return model, val_metrics


def predict(model, X: np.ndarray) -> np.ndarray:
    """Inference helper."""
    return model.predict(X)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Evaluation helper."""
    return calc_metrics(y_true, y_pred)


__all__ = ["build_model", "fit", "predict", "evaluate", "DEFAULT_PARAMS"]
