"""
MTFC Virginia Datacenter Energy Forecasting — Utilities
========================================================
Shared helpers: metric computation, publication-quality plotting,
logging, and I/O utilities used across the pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import json
import pickle
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config as cfg

# ── Logging setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("MTFC")


# ── Publication-quality plot defaults ───────────────────────────────────────
def set_plot_style():
    """Apply a clean, publication-quality matplotlib style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize":       (12, 6),
        "figure.dpi":           150,
        "axes.titlesize":       14,
        "axes.labelsize":       12,
        "xtick.labelsize":      10,
        "ytick.labelsize":      10,
        "legend.fontsize":      10,
        "font.family":          "sans-serif",
        "axes.spines.top":      False,
        "axes.spines.right":    False,
    })

set_plot_style()

# Colour palette
COLORS = {
    "sarimax":      "#1f77b4",
    "lstm":         "#ff7f0e",
    "gru":          "#ff7f0e",
    "xgboost":      "#2ca02c",
    "randomforest": "#9467bd",
    "ensemble":     "#d62728",
    "actual":       "#333333",
    "ci":           "#d6d6d6",
}


# ── Metric helpers ──────────────────────────────────────────────────────────
def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return MAE, RMSE, MAPE, R² as a dict."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(mape),
        "R2":   float(r2_score(y_true, y_pred)),
    }


def metrics_table(results: dict) -> pd.DataFrame:
    """Build a comparison DataFrame from {model_name: metrics_dict}."""
    return pd.DataFrame(results).T.round(4)


# ── Plotting helpers ────────────────────────────────────────────────────────
def save_fig(fig: plt.Figure, name: str, tight: bool = True):
    """Save figure to the figures directory as PNG."""
    path = os.path.join(cfg.FIGURE_DIR, f"{name}.png")
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figure saved → {path}")
    return path


def plot_actual_vs_pred(y_true, y_pred, model_name: str, timestamps=None):
    """Overlay actual vs predicted time series."""
    fig, ax = plt.subplots(figsize=(14, 5))
    x = timestamps if timestamps is not None else np.arange(len(y_true))
    ax.plot(x, y_true, label="Actual", color=COLORS["actual"], alpha=0.7, lw=0.8)
    ax.plot(x, y_pred, label=f"{model_name} Predicted",
            color=COLORS.get(model_name.lower(), "#e377c2"), alpha=0.8, lw=0.8)
    ax.set_title(f"{model_name} — Actual vs Predicted Power (MW)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Power (MW)")
    ax.legend()
    return save_fig(fig, f"actual_vs_pred_{model_name.lower()}")


def plot_residuals(y_true, y_pred, model_name: str):
    """Plot residual distribution + residuals over time."""
    residuals = np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Over time
    axes[0].plot(residuals, color=COLORS.get(model_name.lower(), "#e377c2"),
                 alpha=0.5, lw=0.5)
    axes[0].axhline(0, ls="--", color="red", lw=0.8)
    axes[0].set_title(f"{model_name} — Residuals Over Time")
    axes[0].set_ylabel("Residual (MW)")
    # Histogram
    axes[1].hist(residuals, bins=50, edgecolor="white",
                 color=COLORS.get(model_name.lower(), "#e377c2"), alpha=0.7)
    axes[1].set_title(f"{model_name} — Residual Distribution")
    axes[1].set_xlabel("Residual (MW)")
    return save_fig(fig, f"residuals_{model_name.lower()}")


def plot_scatter_pred(y_true, y_pred, model_name: str):
    """45-degree scatter of actual vs predicted."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.3, s=5,
               color=COLORS.get(model_name.lower(), "#e377c2"))
    lims = [min(np.min(y_true), np.min(y_pred)),
            max(np.max(y_true), np.max(y_pred))]
    ax.plot(lims, lims, "r--", lw=1, label="Perfect Prediction")
    ax.set_xlabel("Actual (MW)")
    ax.set_ylabel("Predicted (MW)")
    ax.set_title(f"{model_name} — Prediction Scatter")
    ax.legend()
    return save_fig(fig, f"scatter_{model_name.lower()}")


# ── I/O helpers ─────────────────────────────────────────────────────────────
def save_pickle(obj, name: str):
    """Pickle-save an object to the models directory."""
    path = os.path.join(cfg.MODEL_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    log.info(f"Model saved → {path}")
    return path


def load_pickle(name: str):
    path = os.path.join(cfg.MODEL_DIR, f"{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(data: dict, name: str):
    path = os.path.join(cfg.RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log.info(f"JSON saved → {path}")
    return path


def save_csv(df: pd.DataFrame, name: str):
    path = os.path.join(cfg.RESULTS_DIR, f"{name}.csv")
    df.to_csv(path, index=True)
    log.info(f"CSV saved → {path}")
    return path
