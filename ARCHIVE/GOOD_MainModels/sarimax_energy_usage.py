"""
SARIMAX Forecast — Northern Virginia Data Center Energy Usage (GWh/month)
=========================================================================
Purpose:
    Forecast monthly electricity consumption (GWh) by the modeled
    100MW Northern Virginia AI data center from 2025-01 through 2035-12.

Data source:
    Primary : Data_Sources/cleaned/semisynthetic_datacenter_power_2015_2024.csv
              Columns: datetime, temp_f, it_load_mw, pue, total_power_mw
              Hourly (2015-2024). Aggregated to monthly GWh via:
                  energy_GWh = sum(total_power_mw) / 1000
              (each row represents one hour, so MW × hr = MWh; /1000 → GWh)

    Exogenous variables:
      - month_sin, month_cos  : cyclical month encoding (always available)
      - year_numeric          : e.g. 2015.0, 2015.083, ... captures long-run
                                upward AI demand trend
      - temp_c_monthly        : NOAA Dulles monthly mean temperature (°C)
                                Available 2019-2024; for 2015-2018 the
                                12-month seasonal mean is backfilled.

Run from repo root:
    python GOOD_MainModels/sarimax_energy_usage.py

Outputs:
    GOOD_MainModels/energy_usage_forecast.csv
    GOOD_MainModels/energy_usage_model_summary.txt
    GOOD_Figures/energy_usage_historical_and_forecast.png
    GOOD_Figures/energy_usage_residuals.png
    GOOD_Figures/energy_usage_seasonal_decomposition.png
    GOOD_Figures/energy_usage_annual_totals.png
"""

import os
import sys
import warnings

os.makedirs("GOOD_MainModels", exist_ok=True)
os.makedirs("GOOD_Figures", exist_ok=True)

# ── library checks ──────────────────────────────────────────────────────────
try:
    import pmdarima
    from pmdarima import auto_arima
except ImportError:
    print("Missing required library: pmdarima. Run: pip install pmdarima")
    sys.exit(1)

try:
    import statsmodels
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.graphics.tsaplots import plot_acf
except ImportError:
    print("Missing required library: statsmodels. Run: pip install statsmodels")
    sys.exit(1)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ── file paths ───────────────────────────────────────────────────────────────
DC_PATH     = "Data_Sources/cleaned/semisynthetic_datacenter_power_2015_2024.csv"
NOAA_PATH   = "Data_Sources/cleaned/noaa_global_hourly_dulles_2019_2024_cleaned.csv"
FORECAST_CSV  = "GOOD_MainModels/energy_usage_forecast.csv"
SUMMARY_TXT   = "GOOD_MainModels/energy_usage_model_summary.txt"
FIG_FORECAST  = "GOOD_Figures/energy_usage_historical_and_forecast.png"
FIG_RESIDUALS = "GOOD_Figures/energy_usage_residuals.png"
FIG_DECOMP    = "GOOD_Figures/energy_usage_seasonal_decomposition.png"
FIG_ANNUAL    = "GOOD_Figures/energy_usage_annual_totals.png"

# ═══════════════════════════════════════════════════════════════════════════
# 1.  LOAD AND AGGREGATE DATA
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("SARIMAX — Northern Virginia Data Center Energy Usage")
print("=" * 65)

if not os.path.exists(DC_PATH):
    print(f"ERROR: Data file not found: {DC_PATH}")
    sys.exit(1)

dc = pd.read_csv(DC_PATH, parse_dates=["datetime"])
dc.set_index("datetime", inplace=True)
dc.index = pd.to_datetime(dc.index)

# Each row = 1 hour. Sum MW per month → MWh; divide by 1000 → GWh.
energy_monthly = (
    dc["total_power_mw"]
    .resample("MS")
    .sum()
    .div(1000.0)
    .rename("energy_GWh")
)
energy_monthly.dropna(inplace=True)

print(f"\n[Data loaded] {DC_PATH}")
print(f"  Source selected: semisynthetic_datacenter_power_2015_2024.csv")
print(f"  Reason: Only source with 120 months (2015-2024) of continuous hourly")
print(f"          total facility power — the longest clean monthly energy series")
print(f"          available in the repository.")
print(f"  Hourly rows   : {len(dc):,}")
print(f"  Monthly months: {len(energy_monthly)}")
print(f"  Date range    : {energy_monthly.index.min().date()} → {energy_monthly.index.max().date()}")
print(f"  Missing months: {energy_monthly.isna().sum()}")
print(f"  Min GWh/month : {energy_monthly.min():.2f}")
print(f"  Max GWh/month : {energy_monthly.max():.2f}")
print(f"  Mean GWh/month: {energy_monthly.mean():.2f}")

# ── exog: NOAA monthly temperature ─────────────────────────────────────────
temp_available = False
temp_monthly_full = None
if os.path.exists(NOAA_PATH):
    try:
        noaa = pd.read_csv(NOAA_PATH, parse_dates=["timestamp"])
        noaa["timestamp"] = pd.to_datetime(noaa["timestamp"], utc=True).dt.tz_localize(None)
        noaa.set_index("timestamp", inplace=True)
        temp_m = noaa["temperature_c"].resample("MS").mean().rename("temp_c")
        temp_m.dropna(inplace=True)
        # Build 12-month seasonal mean for backfilling 2015-2018
        seasonal_temp = temp_m.groupby(temp_m.index.month).mean()
        # Construct full-length temperature series aligned to energy index
        temp_full = pd.Series(index=energy_monthly.index, dtype=float, name="temp_c")
        for dt in energy_monthly.index:
            if dt in temp_m.index:
                temp_full[dt] = temp_m[dt]
            else:
                temp_full[dt] = seasonal_temp.get(dt.month, temp_m.mean())
        temp_monthly_full = temp_full
        temp_available = True
        print(f"\n[Exogenous] NOAA temperature loaded: {len(temp_m)} direct months")
        print(f"  2015-2018 period: backfilled with 12-month seasonal mean from NOAA 2019-2024.")
    except Exception as ex:
        print(f"[WARNING] Could not load NOAA temperature: {ex}. Skipping.")
else:
    print(f"[WARNING] NOAA file not found: {NOAA_PATH}. Skipping temperature exog.")

# ── build exogenous feature matrix ─────────────────────────────────────────
def make_exog(date_index: pd.DatetimeIndex,
              temp_series=None,
              seasonal_temp_means=None) -> pd.DataFrame:
    """Cyclical month + year trend + optional temperature."""
    df = pd.DataFrame(index=date_index)
    df["month_sin"]    = np.sin(2 * np.pi * date_index.month / 12)
    df["month_cos"]    = np.cos(2 * np.pi * date_index.month / 12)
    # Year as a fractional numeric feature: 2015.0, 2015.083, ...
    df["year_numeric"] = date_index.year + (date_index.month - 1) / 12.0
    if temp_series is not None:
        df = df.join(temp_series, how="left")
        # Fill any remaining gaps with seasonal means or global mean
        fallback = temp_series.mean() if len(temp_series) > 0 else 12.0
        for idx in df.index[df["temp_c"].isna()]:
            df.loc[idx, "temp_c"] = (
                seasonal_temp_means.get(idx.month, fallback)
                if seasonal_temp_means is not None else fallback
            )
    return df.astype(float)

seasonal_means_for_future = (
    temp_monthly_full.groupby(temp_monthly_full.index.month).mean()
    if temp_available else None
)

exog_all = make_exog(
    energy_monthly.index,
    temp_series=temp_monthly_full if temp_available else None,
    seasonal_temp_means=seasonal_means_for_future,
)

# ═══════════════════════════════════════════════════════════════════════════
# 2.  TRAIN / TEST SPLIT  (80 / 20 chronological)
# ═══════════════════════════════════════════════════════════════════════════
n       = len(energy_monthly)
n_train = int(n * 0.80)
n_test  = n - n_train

y_train    = energy_monthly.iloc[:n_train]
y_test     = energy_monthly.iloc[n_train:]
exog_train = exog_all.iloc[:n_train]
exog_test  = exog_all.iloc[n_train:]

print(f"\n[Split] Total months: {n}  |  Train: {n_train}  |  Test: {n_test}")
print(f"  Train: {y_train.index.min().date()} → {y_train.index.max().date()}")
print(f"  Test : {y_test.index.min().date()}  → {y_test.index.max().date()}")

# ═══════════════════════════════════════════════════════════════════════════
# 3.  AUTO_ARIMA — find best (p,d,q)(P,D,Q,12) order
# ═══════════════════════════════════════════════════════════════════════════
print("\n[auto_arima] Searching for best SARIMAX order (seasonal, m=12)...")
auto_model = auto_arima(
    y_train,
    exogenous=exog_train.values,
    seasonal=True,
    m=12,
    information_criterion="aic",
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    trace=False,
)
order          = auto_model.order
seasonal_order = auto_model.seasonal_order
print(f"  Best order    : ARIMA{order}")
print(f"  Seasonal order: {seasonal_order}")
print(f"  AIC           : {auto_model.aic():.2f}")

# ═══════════════════════════════════════════════════════════════════════════
# 4.  REFIT WITH STATSMODELS ON TRAINING SET
# ═══════════════════════════════════════════════════════════════════════════
print("\n[SARIMAX] Refitting on training set with statsmodels...")
sm_model = SARIMAX(
    y_train,
    exog=exog_train,
    order=order,
    seasonal_order=seasonal_order,
    trend="n",
    enforce_stationarity=False,
    enforce_invertibility=False,
)
result = sm_model.fit(disp=False, maxiter=200)

summary_str = result.summary().as_text()
print("\n" + summary_str)
with open(SUMMARY_TXT, "w") as f:
    f.write(summary_str)
print(f"\n[Saved] Model summary → {SUMMARY_TXT}")

# ═══════════════════════════════════════════════════════════════════════════
# 5.  EVALUATE ON TEST SET
# ═══════════════════════════════════════════════════════════════════════════
test_fc = result.get_forecast(steps=n_test, exog=exog_test)
y_pred  = test_fc.predicted_mean.values.clip(min=0.0)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"\n[Test set performance]")
print(f"  RMSE : {rmse:.4f} GWh/month")
print(f"  MAE  : {mae:.4f} GWh/month")
print(f"  R²   : {r2:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# 6.  REFIT ON FULL DATA
# ═══════════════════════════════════════════════════════════════════════════
print("\n[SARIMAX] Refitting on full historical data before forecasting...")
sm_model_full = SARIMAX(
    energy_monthly,
    exog=exog_all,
    order=order,
    seasonal_order=seasonal_order,
    trend="n",
    enforce_stationarity=False,
    enforce_invertibility=False,
)
result_full = sm_model_full.fit(disp=False, maxiter=200)

# ═══════════════════════════════════════════════════════════════════════════
# 7.  FORECAST 2025-01 → 2035-12  (132 months)
# ═══════════════════════════════════════════════════════════════════════════
N_FORECAST = 132
last_date  = energy_monthly.index.max()
fc_dates   = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=N_FORECAST, freq="MS")

# Future temperature: use 12-month seasonal means (deterministic seasonal pattern)
if temp_available:
    future_temp = pd.Series(
        [seasonal_means_for_future.get(m, temp_monthly_full.mean()) for m in fc_dates.month],
        index=fc_dates,
        name="temp_c",
    )
else:
    future_temp = None

exog_future = make_exog(
    fc_dates,
    temp_series=future_temp,
    seasonal_temp_means=seasonal_means_for_future,
)

fc_result = result_full.get_forecast(steps=N_FORECAST, exog=exog_future)
fc_mean   = fc_result.predicted_mean.clip(lower=0.0)
fc_ci     = fc_result.conf_int(alpha=0.05).clip(lower=0.0)   # 95% CI
fc_ci_80  = fc_result.conf_int(alpha=0.20).clip(lower=0.0)   # 80% CI

# Forecast point estimates at key years (July mid-year representative)
def _fc_at(year: int) -> float:
    target = pd.Timestamp(f"{year}-07-01")
    closest = fc_mean.index[np.argmin(np.abs(fc_mean.index - target))]
    return float(fc_mean[closest])

print(f"\n[Forecast] Range: {fc_dates[0].date()} → {fc_dates[-1].date()}")
print(f"  Monthly point estimate 2030 (Jul): {_fc_at(2030):.2f} GWh")
print(f"  Monthly point estimate 2035 (Jul): {_fc_at(2035):.2f} GWh")

# ═══════════════════════════════════════════════════════════════════════════
# 8.  SAVE FORECAST CSV
# ═══════════════════════════════════════════════════════════════════════════
fc_df = pd.DataFrame({
    "date"         : fc_dates,
    "forecast_GWh" : fc_mean.values,
    "lower_80"     : fc_ci_80.iloc[:, 0].values,
    "upper_80"     : fc_ci_80.iloc[:, 1].values,
    "lower_95"     : fc_ci.iloc[:, 0].values,
    "upper_95"     : fc_ci.iloc[:, 1].values,
})
fc_df.to_csv(FORECAST_CSV, index=False)
print(f"\n[Saved] Forecast CSV → {FORECAST_CSV}")

# ═══════════════════════════════════════════════════════════════════════════
# 9.  FIGURES
# ═══════════════════════════════════════════════════════════════════════════

# ── Figure 1: Historical + test + forecast ────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(y_train.index, y_train.values, color="steelblue", linewidth=1.8,
        label="Historical (train)")
ax.plot(y_test.index, y_test.values, color="darkorange", linewidth=1.8,
        label="Actuals (test)")
ax.plot(y_test.index, y_pred, color="firebrick", linewidth=1.5,
        linestyle="--", label="Model fit (test)")
ax.axvline(y_train.index.max(), color="gray", linestyle="--", alpha=0.7,
           label="Train/Test Split")
ax.plot(fc_dates, fc_mean.values, color="steelblue", linewidth=1.8,
        linestyle="--", label="Forecast 2025–2035")
ax.fill_between(fc_dates, fc_ci.iloc[:,0], fc_ci.iloc[:,1],
                color="steelblue", alpha=0.2, label="95% CI")
ax.fill_between(fc_dates, fc_ci_80.iloc[:,0], fc_ci_80.iloc[:,1],
                color="steelblue", alpha=0.3, label="80% CI")

# Annotate 2035 point estimate
last_fc_val   = float(fc_mean.iloc[-1])
last_fc_upper = float(fc_ci.iloc[-1, 1])
last_fc_lower = float(fc_ci.iloc[-1, 0])
ax.annotate(
    f"2035: {last_fc_val:.1f} GWh\n95% CI [{last_fc_lower:.0f}–{last_fc_upper:.0f}]",
    xy=(fc_dates[-1], last_fc_val),
    xytext=(-100, 15),
    textcoords="offset points",
    fontsize=9,
    arrowprops=dict(arrowstyle="->", color="gray"),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="steelblue"),
)

ax.set_title("Northern Virginia Data Center Energy Usage: SARIMAX Forecast 2025–2035",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Energy Consumption (GWh/month)", fontsize=12)
ax.tick_params(labelsize=10)
ax.legend(fontsize=9, loc="upper left")
ax.set_xlim(energy_monthly.index.min(), fc_dates.max())
fig.tight_layout()
fig.savefig(FIG_FORECAST, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[Saved] Figure → {FIG_FORECAST}")

# ── Figure 2: Residual diagnostics ────────────────────────────────────────
residuals = result_full.resid

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

axes[0].plot(energy_monthly.index, residuals, color="steelblue", linewidth=1.0)
axes[0].axhline(0, color="gray", linestyle="--", linewidth=1.0)
axes[0].set_title("SARIMAX Residual Diagnostics — Data Center Energy Usage",
                   fontsize=14, fontweight="bold")
axes[0].set_xlabel("Year", fontsize=12)
axes[0].set_ylabel("Residual (GWh/month)", fontsize=12)
axes[0].tick_params(labelsize=10)
axes[0].set_xlim(energy_monthly.index.min(), energy_monthly.index.max())

plot_acf(residuals.dropna(), lags=min(20, n_train // 2 - 1), ax=axes[1],
         color="steelblue", alpha=0.8)
axes[1].set_title("ACF of Residuals (should show no significant autocorrelation)",
                   fontsize=12)
axes[1].set_xlabel("Lag (months)", fontsize=12)
axes[1].set_ylabel("Autocorrelation", fontsize=12)
axes[1].tick_params(labelsize=10)

fig.tight_layout()
fig.savefig(FIG_RESIDUALS, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[Saved] Figure → {FIG_RESIDUALS}")

# ── Figure 3: Seasonal decomposition ──────────────────────────────────────
if len(energy_monthly) >= 24:
    decomp = seasonal_decompose(energy_monthly, model="additive", period=12,
                                 extrapolate_trend="freq")
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    pairs = [
        (energy_monthly, "Observed", "steelblue"),
        (decomp.trend,   "Trend",    "darkorange"),
        (decomp.seasonal,"Seasonal", "forestgreen"),
        (decomp.resid,   "Residual", "firebrick"),
    ]
    for ax, (series, label, color) in zip(axes, pairs):
        ax.plot(series.index, series.values, color=color, linewidth=1.5)
        ax.set_ylabel(label, fontsize=11)
        ax.tick_params(labelsize=10)
        ax.set_xlim(energy_monthly.index.min(), energy_monthly.index.max())
    axes[0].set_title("Seasonal Decomposition — Northern Virginia DC Energy Usage",
                      fontsize=14, fontweight="bold")
    axes[-1].set_xlabel("Year", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DECOMP, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Figure → {FIG_DECOMP}")

# ── Figure 4: Annual totals bar chart ─────────────────────────────────────
fc_df_plot = fc_df.copy()
fc_df_plot["year"] = pd.to_datetime(fc_df_plot["date"]).dt.year
annual_fc     = fc_df_plot.groupby("year")["forecast_GWh"].sum()
annual_lower  = fc_df_plot.groupby("year")["lower_95"].sum()
annual_upper  = fc_df_plot.groupby("year")["upper_95"].sum()
error_low  = annual_fc.values - annual_lower.values
error_high = annual_upper.values - annual_fc.values

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(annual_fc.index, annual_fc.values, color="steelblue", alpha=0.8,
              label="Annual Energy Forecast (GWh)", zorder=3)
ax.errorbar(annual_fc.index, annual_fc.values,
            yerr=[error_low, error_high],
            fmt="none", color="black", capsize=5, linewidth=1.5, label="95% CI")

# Annotate each bar
for year, val in zip(annual_fc.index, annual_fc.values):
    ax.text(year, val + max(annual_fc.values) * 0.01, f"{val:.0f}",
            ha="center", va="bottom", fontsize=8, color="black")

ax.set_title("Annual Data Center Energy Usage Forecast 2025–2035 (SARIMAX)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Annual Energy Consumption (GWh/year)", fontsize=12)
ax.tick_params(labelsize=10)
ax.set_xticks(annual_fc.index)
ax.set_xticklabels([str(y) for y in annual_fc.index], rotation=45, ha="right")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_ANNUAL, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[Saved] Figure → {FIG_ANNUAL}")

# ═══════════════════════════════════════════════════════════════════════════
# 10. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("OUTPUTS WRITTEN")
print("  CSVs :")
print(f"    {FORECAST_CSV}")
print(f"    {SUMMARY_TXT}")
print("  Figures :")
print(f"    {FIG_FORECAST}")
print(f"    {FIG_RESIDUALS}")
print(f"    {FIG_DECOMP}")
print(f"    {FIG_ANNUAL}")
print("=" * 65)
