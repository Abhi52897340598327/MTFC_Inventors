"""
MTFC Virginia Datacenter Energy Forecasting — Exploratory Data Analysis
========================================================================
Full EDA: time-series plots, seasonal decomposition, correlation matrix,
distributions, stationarity tests (ADF), ACF/PACF, and temperature–power
relationship analysis.  All figures saved to outputs/figures/.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import config as cfg
from utils import log, save_fig, set_plot_style


def run_eda(df: pd.DataFrame) -> dict:
    """
    Execute the complete EDA suite on the hourly merged DataFrame.
    Returns a summary dict of key statistics.
    """
    set_plot_style()
    target = cfg.TARGET_COL
    summary = {}

    # ─── 1. Summary Statistics ───────────────────────────────────────────
    log.info("EDA Step 1 — Summary Statistics")
    desc = df.describe()
    summary["describe"] = desc.to_dict()
    
    cols_to_show = [c for c in [target, 'temperature_f', 'cpu_utilization'] if c in df.columns]
    log.info(f"\n{desc[cols_to_show].round(2)}")

    # ─── 2. Full-Year Time Series ────────────────────────────────────────
    log.info("EDA Step 2 — Time-series plot")
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(df["timestamp"], df[target], lw=0.5, color="#1f77b4")
    ax.set_title("Datacenter Total Power Consumption — Full Year 2019")
    ax.set_ylabel("Power (MW)")
    ax.set_xlabel("Date")
    save_fig(fig, "eda_timeseries")

    # ─── 3. Seasonal Decomposition ───────────────────────────────────────
    log.info("EDA Step 3 — Seasonal decomposition (period=24)")
    decomp = seasonal_decompose(df[target], model="additive", period=24)
    fig = decomp.plot()
    fig.set_size_inches(16, 10)
    fig.suptitle("Seasonal Decomposition (Daily Period = 24 h)", y=1.02)
    save_fig(fig, "eda_seasonal_decomposition")

    # ─── 4. Monthly Box-plots ────────────────────────────────────────────
    log.info("EDA Step 4 — Monthly box-plots")
    fig, ax = plt.subplots(figsize=(12, 5))
    month_col = "month" if "month" in df.columns else df["timestamp"].dt.month
    if isinstance(month_col, str):
        sns.boxplot(x=df[month_col], y=df[target], ax=ax, palette="coolwarm")
    else:
        sns.boxplot(x=month_col, y=df[target], ax=ax, palette="coolwarm")
    ax.set_title("Power Consumption by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Power (MW)")
    save_fig(fig, "eda_monthly_boxplot")

    # ─── 5. Day-of-Week Pattern ──────────────────────────────────────────
    log.info("EDA Step 5 — Day-of-week pattern")
    dow = "day_of_week" if "day_of_week" in df.columns else df["timestamp"].dt.dayofweek
    fig, ax = plt.subplots(figsize=(10, 5))
    if isinstance(dow, str):
        df.groupby(dow)[target].mean().plot(kind="bar", ax=ax, color="#2ca02c")
    else:
        df.assign(dow=dow).groupby("dow")[target].mean().plot(kind="bar", ax=ax, color="#2ca02c")
    ax.set_title("Average Power by Day of Week")
    ax.set_ylabel("Mean Power (MW)")
    ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], rotation=0)
    save_fig(fig, "eda_day_of_week")

    # ─── 6. Hourly Pattern ──────────────────────────────────────────────
    # ─── 6. Hourly Pattern ──────────────────────────────────────────────
    log.info("EDA Step 6 — Hourly pattern")
    hour_col = "hour" if "hour" in df.columns else ("hour_of_day" if "hour_of_day" in df.columns else None)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    if hour_col:
        df.groupby(hour_col)[target].mean().plot(kind="bar", ax=ax, color="#ff7f0e")
    else:
        # derive from timestamp
        df.groupby(df["timestamp"].dt.hour)[target].mean().plot(kind="bar", ax=ax, color="#ff7f0e")
        
    ax.set_title("Average Power by Hour of Day")
    ax.set_ylabel("Mean Power (MW)")
    ax.set_xlabel("Hour")
    save_fig(fig, "eda_hourly_pattern")

    # ─── 7. Correlation Matrix ──────────────────────────────────────────
    log.info("EDA Step 7 — Correlation matrix")
    corr_cols = [target, "cpu_utilization", "temperature_f",
                 "cooling_load", "pue", "it_load", "base_load"]
    corr_cols = [c for c in corr_cols if c in df.columns]
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, square=True)
    ax.set_title("Correlation Matrix — Key Variables")
    save_fig(fig, "eda_correlation_matrix")

    # ─── 8. Distributions ────────────────────────────────────────────────
    log.info("EDA Step 8 — Distributions")
    dist_cols = [target, "temperature_f", "cpu_utilization"]
    dist_cols = [c for c in dist_cols if c in df.columns]
    fig, axes = plt.subplots(1, len(dist_cols), figsize=(5 * len(dist_cols), 4))
    if len(dist_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, dist_cols):
        ax.hist(df[col].dropna(), bins=60, edgecolor="white", alpha=0.7)
        ax.set_title(f"Distribution — {col}")
    save_fig(fig, "eda_distributions")

    # ─── 9. Stationarity Test (ADF) ─────────────────────────────────────
    log.info("EDA Step 9 — Augmented Dickey-Fuller test")
    adf_result = adfuller(df[target].dropna(), autolag="AIC")
    summary["adf"] = {
        "statistic": adf_result[0],
        "p_value":   adf_result[1],
        "lags_used": adf_result[2],
        "nobs":      adf_result[3],
    }
    log.info(f"  ADF statistic: {adf_result[0]:.4f},  p-value: {adf_result[1]:.6f}")
    if adf_result[1] < 0.05:
        log.info("  → Series is stationary (reject H₀)")
    else:
        log.info("  → Series is NON-stationary (fail to reject H₀)")

    # ─── 10. ACF / PACF ─────────────────────────────────────────────────
    log.info("EDA Step 10 — ACF and PACF")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    plot_acf(df[target].dropna(), lags=200, ax=axes[0], alpha=0.05)
    axes[0].set_title("Autocorrelation Function (ACF)")
    plot_pacf(df[target].dropna(), lags=100, ax=axes[1], alpha=0.05, method="ywm")
    axes[1].set_title("Partial Autocorrelation Function (PACF)")
    save_fig(fig, "eda_acf_pacf")

    # ─── 11. Temperature vs Power ────────────────────────────────────────
    log.info("EDA Step 11 — Temperature vs power relationship")
    if "temperature_f" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df["temperature_f"], df[target], alpha=0.15, s=5, color="#d62728")
        # Polynomial fit
        z = np.polyfit(df["temperature_f"].dropna(), df[target].dropna(), 2)
        p = np.poly1d(z)
        t_sorted = np.sort(df["temperature_f"].dropna())
        ax.plot(t_sorted, p(t_sorted), "k-", lw=2, label="Quadratic fit")
        ax.set_xlabel("Temperature (°F)")
        ax.set_ylabel("Power (MW)")
        ax.set_title("Temperature vs Power Consumption (with Quadratic Fit)")
        ax.legend()
        save_fig(fig, "eda_temp_vs_power")

    log.info("EDA complete — all figures saved")
    return summary
