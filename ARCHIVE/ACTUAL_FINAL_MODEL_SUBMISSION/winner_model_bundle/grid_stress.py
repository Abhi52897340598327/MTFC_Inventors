"""
MTFC Virginia Datacenter Energy Forecasting — Grid Stress Analysis
====================================================================
Coincident Peak Factor, load factor, grid stress score, and peak-hour
analysis using PJM demand data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config as cfg
from utils import log, save_fig, save_csv


# ── Core Metrics ────────────────────────────────────────────────────────────

def calculate_cpf(dc_power: np.ndarray, grid_demand: np.ndarray,
                  percentile: float = 90) -> float:
    """
    Coincident Peak Factor = DC power at grid peak hours / DC peak power.
    """
    n = min(len(dc_power), len(grid_demand))
    dc = dc_power[:n]
    gd = grid_demand[:n]

    threshold = np.percentile(gd, percentile)
    peak_mask = gd >= threshold

    if peak_mask.sum() == 0:
        return 0.5

    dc_at_peak = dc[peak_mask].mean()
    dc_peak = dc.max()
    cpf = dc_at_peak / dc_peak if dc_peak > 0 else 0
    return float(cpf)


def calculate_load_factor(power: np.ndarray) -> float:
    """Average power / peak power."""
    peak = power.max()
    return float(power.mean() / peak) if peak > 0 else 0


def calculate_grid_stress_score(dc_power: np.ndarray,
                                 grid_demand: np.ndarray) -> dict:
    """
    Compute the composite Grid Stress Score (0-100).
    """
    n = min(len(dc_power), len(grid_demand))
    dc = dc_power[:n]
    gd = grid_demand[:n]

    cpf = calculate_cpf(dc, gd)
    lf = calculate_load_factor(dc)
    cv = float(dc.std() / dc.mean()) if dc.mean() > 0 else 0
    peak_contrib = float(dc.max() / gd.max()) if gd.max() > 0 else 0

    gss = (cfg.GSS_WEIGHTS["cpf"] * cpf * 100 +
           cfg.GSS_WEIGHTS["peak_contrib"] * peak_contrib * 100 +
           cfg.GSS_WEIGHTS["variability"] * min(cv, 1.0) * 100 +
           cfg.GSS_WEIGHTS["load_factor"] * (1 - lf) * 100)

    return {
        "cpf": cpf,
        "load_factor": lf,
        "coefficient_of_variation": cv,
        "peak_contribution": peak_contrib,
        "grid_stress_score": float(gss),
    }


# ── Peak Hour Analysis ──────────────────────────────────────────────────────

def identify_peak_hours(grid_demand: np.ndarray,
                        timestamps: pd.Series = None,
                        top_n: int = 100) -> pd.DataFrame:
    """Identify the top-N peak grid demand hours."""
    idx = np.argsort(grid_demand)[-top_n:][::-1]
    rows = []
    for i in idx:
        row = {"rank": len(rows) + 1, "index": int(i),
               "grid_demand_mw": float(grid_demand[i])}
        if timestamps is not None and i < len(timestamps):
            ts = timestamps.iloc[i]
            row.update({"timestamp": ts, "hour": ts.hour if hasattr(ts, "hour") else 0,
                        "month": ts.month if hasattr(ts, "month") else 0})
        rows.append(row)
    return pd.DataFrame(rows)


# ── Visualisations ──────────────────────────────────────────────────────────

def plot_grid_demand_heatmap(pjm_2019: pd.DataFrame):
    """Heatmap of grid demand by hour × month."""
    df = pjm_2019.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month

    pivot = df.pivot_table(values="grid_demand_mw",
                           index="hour", columns="month", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, cmap="YlOrRd", annot=False, ax=ax)
    ax.set_title("PJM Grid Demand Heatmap (Hour × Month, 2019)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Hour of Day")
    return save_fig(fig, "grid_demand_heatmap")


def plot_dc_vs_grid_peaks(dc_power: np.ndarray, grid_demand: np.ndarray,
                          timestamps: pd.Series = None):
    """Overlay datacenter power and grid demand (normalised)."""
    n = min(len(dc_power), len(grid_demand))
    fig, ax1 = plt.subplots(figsize=(16, 6))

    x = timestamps[:n] if timestamps is not None else np.arange(n)
    ax1.plot(x, grid_demand[:n] / 1000, color="#1f77b4", alpha=0.5, lw=0.5, label="Grid Demand (GW)")
    ax1.set_ylabel("Grid Demand (GW)", color="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(x, dc_power[:n], color="#d62728", alpha=0.6, lw=0.5, label="DC Power (MW)")
    ax2.set_ylabel("Datacenter Power (MW)", color="#d62728")

    ax1.set_title("Datacenter Power vs Grid Demand (2019)")
    ax1.set_xlabel("Time")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    return save_fig(fig, "dc_vs_grid_peaks")


def plot_grid_stress_comparison(scenarios_df: pd.DataFrame):
    """Bar chart comparing grid stress score across scenarios."""
    fig, ax = plt.subplots(figsize=(14, 6))
    scenarios = scenarios_df["scenario"].values
    gss = scenarios_df["grid_stress_score"].values

    colors = ["#2ca02c" if g < 30 else "#ff7f0e" if g < 60 else "#d62728" for g in gss]
    ax.barh(scenarios, gss, color=colors, edgecolor="white")
    ax.axvline(30, ls="--", color="green", alpha=0.5, label="Low Stress Threshold")
    ax.axvline(60, ls="--", color="red", alpha=0.5, label="High Stress Threshold")
    ax.set_xlabel("Grid Stress Score (0-100)")
    ax.set_title("Grid Stress Score by Scenario")
    ax.legend()
    return save_fig(fig, "grid_stress_comparison")


# ── Master Function ─────────────────────────────────────────────────────────

def run_grid_stress_analysis(dc_power: np.ndarray,
                              pjm_df: pd.DataFrame,
                              scenarios_df: pd.DataFrame = None) -> dict:
    """
    Full grid stress analysis.

    Parameters
    ----------
    dc_power    : hourly datacenter power (MW) for 2019
    pjm_df      : PJM hourly demand DataFrame (has timestamp, grid_demand_mw)
    scenarios_df: sensitivity results with 'grid_stress_score' column

    Returns a summary dict.
    """
    # Filter to 2019
    pjm_2019 = pjm_df[pjm_df["timestamp"].dt.year == 2019].copy()
    pjm_2019.reset_index(drop=True, inplace=True)

    grid_demand = pjm_2019["grid_demand_mw"].values

    # Core metrics
    gss = calculate_grid_stress_score(dc_power, grid_demand)
    log.info(f"Grid Stress Score (baseline): {gss['grid_stress_score']:.1f}")
    log.info(f"  CPF: {gss['cpf']:.3f}, LF: {gss['load_factor']:.3f}")

    # Peak hours
    peak_hours = identify_peak_hours(grid_demand, pjm_2019["timestamp"])
    save_csv(peak_hours, "grid_peak_hours")

    # Plots
    plot_grid_demand_heatmap(pjm_2019)
    plot_dc_vs_grid_peaks(dc_power, grid_demand, pjm_2019["timestamp"])

    if scenarios_df is not None and "grid_stress_score" in scenarios_df.columns:
        plot_grid_stress_comparison(scenarios_df)

    return {"grid_stress": gss, "peak_hours": peak_hours}
