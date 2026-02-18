"""
MTFC Virginia Datacenter Energy Forecasting — Evaluation
==========================================================
Model comparison: metrics table, actual-vs-predicted plots, residual
analysis, feature importance charts, and consolidated comparison figure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as cfg
from utils import (log, calc_metrics, metrics_table, save_fig, save_csv,
                   plot_actual_vs_pred, plot_residuals, plot_scatter_pred,
                   COLORS)


def evaluate_all(results: dict, timestamps=None):
    """
    Consolidated evaluation of all models.

    Parameters
    ----------
    results : dict
        {model_name: {"y_pred": ndarray, "y_true": ndarray, "metrics": dict}}
    timestamps : array-like, optional

    Returns
    -------
    comparison : pd.DataFrame of metrics
    """
    # ─── Metrics Table ───────────────────────────────────────────────────
    metrics_dict = {name: r["metrics"] for name, r in results.items()}
    comparison = metrics_table(metrics_dict)
    log.info(f"\nModel Comparison:\n{comparison}")
    save_csv(comparison, "model_comparison")

    # ─── Per-model plots ─────────────────────────────────────────────────
    for name, r in results.items():
        n = len(r["y_true"])
        # Trim timestamps to match this model's y_true length
        ts = timestamps[-n:] if timestamps is not None and len(timestamps) >= n else None
        plot_actual_vs_pred(r["y_true"], r["y_pred"], name, ts)
        plot_residuals(r["y_true"], r["y_pred"], name)
        plot_scatter_pred(r["y_true"], r["y_pred"], name)

    # ─── Combined overlay (use shortest common length) ───────────────────
    min_len = min(len(r["y_true"]) for r in results.values())
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(min_len)

    # Use the first model's y_true as reference
    ref_ytrue = list(results.values())[0]["y_true"][-min_len:]
    ax.plot(x, ref_ytrue, label="Actual",
            color=COLORS["actual"], alpha=0.6, lw=0.7)

    for name, r in results.items():
        color = COLORS.get(name.lower(), "#9467bd")
        pred = r["y_pred"][-min_len:] if len(r["y_pred"]) >= min_len else r["y_pred"]
        ax.plot(x[:len(pred)], pred,
                label=f"{name}", color=color, alpha=0.7, lw=0.7)
    ax.set_title("All Models vs Actual — Test Set")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Power (MW)")
    ax.legend()
    save_fig(fig, "all_models_overlay")

    # ─── Bar-chart comparison ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for i, metric in enumerate(["MAE", "RMSE", "MAPE", "R2"]):
        vals = [metrics_dict[m][metric] for m in metrics_dict]
        names = list(metrics_dict.keys())
        colors = [COLORS.get(n.lower(), "#9467bd") for n in names]
        axes[i].bar(names, vals, color=colors, edgecolor="white")
        axes[i].set_title(metric)
        axes[i].tick_params(axis="x", rotation=30)
    fig.suptitle("Model Performance Comparison", fontsize=14, y=1.02)
    save_fig(fig, "metrics_bar_comparison")

    return comparison
