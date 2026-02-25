"""Shared utilities for metrics, logging, I/O, and leakage guards."""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Set, Tuple

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_logger(name: str = "CarbonPipeline") -> logging.Logger:
    """Return a configured logger with deterministic formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def create_run_id(prefix: str = "run") -> str:
    """Create a UTC run id suitable for artifact directories."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def ensure_dir(path: Path) -> Path:
    """Create a directory if missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def calc_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    """Compute MAE, RMSE, MAPE, sMAPE, and R2."""
    y_true_arr = np.asarray(y_true, dtype=float).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=float).ravel()

    if y_true_arr.size == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))

    non_zero_mask = np.abs(y_true_arr) > 1e-12
    if non_zero_mask.any():
        mape = float(np.mean(np.abs((y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) / y_true_arr[non_zero_mask])) * 100.0)
    else:
        mape = float("nan")

    denom = np.abs(y_true_arr) + np.abs(y_pred_arr)
    smape_mask = denom > 1e-12
    if smape_mask.any():
        smape = float(np.mean(2.0 * np.abs(y_true_arr[smape_mask] - y_pred_arr[smape_mask]) / denom[smape_mask]) * 100.0)
    else:
        smape = float("nan")

    try:
        r2 = float(r2_score(y_true_arr, y_pred_arr))
    except ValueError:
        r2 = float("nan")

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape, "R2": r2}


def relative_rmse_improvement(baseline_rmse: float, model_rmse: float) -> float:
    """Relative RMSE improvement over baseline."""
    if baseline_rmse <= 0:
        return float("nan")
    return float((baseline_rmse - model_rmse) / baseline_rmse)


def save_dataframe(df: pd.DataFrame, path: Path) -> Path:
    """Persist dataframe as CSV and return written path."""
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    return path


def save_json(data: Dict[str, Any], path: Path) -> Path:
    """Persist dictionary as JSON and return written path."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, default=str)
    return path


def save_pickle(obj: Any, path: Path) -> Path:
    """Persist an object as pickle and return written path."""
    ensure_dir(path.parent)
    with path.open("wb") as fp:
        pickle.dump(obj, fp)
    return path


def save_figure(fig: plt.Figure, path: Path) -> Path:
    """Persist a matplotlib figure and close it."""
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def assert_no_bfill(source_text: str) -> None:
    """Raise if source code includes backward-fill usage."""
    prohibited = [".bfill(", "fillna(method='bfill')", 'fillna(method="bfill")']
    for token in prohibited:
        if token in source_text:
            raise AssertionError(f"Leakage guard failed: prohibited token found: {token}")


def assert_strictly_past_features(
    feature_lag_map: Dict[str, int],
    allow_zero_for: Optional[Set[str]] = None,
) -> None:
    """Ensure engineered features use only historical raw-series values."""
    allow_zero_for = allow_zero_for or set()
    for name, lag in feature_lag_map.items():
        if lag < 0:
            raise AssertionError(f"Feature '{name}' has negative lag ({lag}).")
        if lag == 0 and name not in allow_zero_for:
            raise AssertionError(
                f"Feature '{name}' has zero lag and is not in allow_zero_for. "
                "All data-derived features must be strictly from past values."
            )


def block_bootstrap_metric_ci(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 400,
    block_size: int = 24,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """Block bootstrap confidence interval for a metric."""
    y_true_arr = np.asarray(y_true, dtype=float).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=float).ravel()

    n = len(y_true_arr)
    if n == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    metrics = []

    effective_block = max(1, min(block_size, n))
    max_start = max(1, n - effective_block + 1)

    for _ in range(n_boot):
        idx_parts = []
        while sum(len(part) for part in idx_parts) < n:
            start = int(rng.integers(0, max_start))
            part = np.arange(start, min(start + effective_block, n))
            idx_parts.append(part)
        indices = np.concatenate(idx_parts)[:n]
        m = metric_fn(y_true_arr[indices], y_pred_arr[indices])
        metrics.append(float(m))

    low = float(np.nanpercentile(metrics, 100 * (alpha / 2.0)))
    high = float(np.nanpercentile(metrics, 100 * (1.0 - alpha / 2.0)))
    return low, high


def bootstrap_ci_for_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    n_boot: int,
    block_size: int,
    alpha: float,
    seed: int,
) -> Dict[str, Tuple[float, float]]:
    """Compute block-bootstrap confidence intervals for all reported metrics."""

    def mae_fn(a: np.ndarray, b: np.ndarray) -> float:
        return float(mean_absolute_error(a, b))

    def rmse_fn(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(mean_squared_error(a, b)))

    def mape_fn(a: np.ndarray, b: np.ndarray) -> float:
        mask = np.abs(a) > 1e-12
        if not mask.any():
            return float("nan")
        return float(np.mean(np.abs((a[mask] - b[mask]) / a[mask])) * 100.0)

    def smape_fn(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.abs(a) + np.abs(b)
        mask = denom > 1e-12
        if not mask.any():
            return float("nan")
        return float(np.mean(2.0 * np.abs(a[mask] - b[mask]) / denom[mask]) * 100.0)

    def r2_fn(a: np.ndarray, b: np.ndarray) -> float:
        try:
            return float(r2_score(a, b))
        except ValueError:
            return float("nan")

    metric_fns = {
        "MAE": mae_fn,
        "RMSE": rmse_fn,
        "MAPE": mape_fn,
        "sMAPE": smape_fn,
        "R2": r2_fn,
    }

    ci = {}
    for idx, (name, fn) in enumerate(metric_fns.items()):
        low, high = block_bootstrap_metric_ci(
            y_true,
            y_pred,
            fn,
            n_boot=n_boot,
            block_size=block_size,
            alpha=alpha,
            seed=seed + idx,
        )
        ci[name] = (low, high)

    return ci
