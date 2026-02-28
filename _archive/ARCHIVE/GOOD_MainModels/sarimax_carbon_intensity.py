"""
SARIMAX Forecast — Virginia/PJM Grid Carbon Intensity (gCO2/kWh)
=================================================================
Purpose:
    Forecast monthly mean grid carbon intensity (gCO2/kWh) from
    2025-01 through 2035-12 using SARIMAX, fit to the 2019-2024
    PJM hourly data aggregated to monthly.

Data source:
    Primary   : Data_Sources/cleaned/pjm_grid_carbon_intensity_2019_full_cleaned.csv
                Column: carbon_intensity_kg_per_mwh
                Note  : kg/MWh ≡ g/kWh — no unit conversion needed.
    Exogenous : Month cyclical encoding (sin/cos) — always available.
                PJM monthly mean hourly demand — available 2019-2024,
                projected forward using the 12-month seasonal mean.

Run from repo root:
    python GOOD_MainModels/sarimax_carbon_intensity.py

Outputs:
    GOOD_MainModels/carbon_intensity_forecast.csv
    GOOD_MainModels/carbon_intensity_model_summary.txt
    GOOD_Figures/carbon_intensity_historical_and_forecast.png
    GOOD_Figures/carbon_intensity_residuals.png
    GOOD_Figures/carbon_intensity_seasonal_decomposition.png
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
CI_PATH     = "Data_Sources/cleaned/pjm_grid_carbon_intensity_2019_full_cleaned.csv"
DEMAND_PATH = "Data_Sources/cleaned/pjm_hourly_demand_2019_2024_cleaned.csv"
FORECAST_CSV  = "GOOD_MainModels/carbon_intensity_forecast.csv"
SUMMARY_TXT   = "GOOD_MainModels/carbon_intensity_model_summary.txt"
FIG_FORECAST  = "GOOD_Figures/carbon_intensity_historical_and_forecast.png"
FIG_RESIDUALS = "GOOD_Figures/carbon_intensity_residuals.png"
FIG_DECOMP    = "GOOD_Figures/carbon_intensity_seasonal_decomposition.png"

# ═══════════════════════════════════════════════════════════════════════════
# 1.  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("SARIMAX — Virginia Grid Carbon Intensity Forecast")
print("=" * 65)

# ── primary: PJM carbon intensity ──────────────────────────────────────────
if not os.path.exists(CI_PATH):
    print(f"ERROR: Data file not found: {CI_PATH}")
    sys.exit(1)

ci_hourly = pd.read_csv(CI_PATH, parse_dates=["timestamp"])
ci_hourly.set_index("timestamp", inplace=True)
ci_hourly.index = pd.to_datetime(ci_hourly.index, utc=True).tz_localize(None)

# Aggregate to monthly mean. kg/MWh == g/kWh — values are native gCO2/kWh.
ci_monthly = (
    ci_hourly["carbon_intensity_kg_per_mwh"]
    .resample("MS")
    .mean()
    .rename("carbon_intensity_gco2_per_kwh")
)
ci_monthly.dropna(inplace=True)

print(f"\n[Data loaded] {CI_PATH}")
print(f"  Hourly shape : {ci_hourly.shape}")
print(f"  Monthly shape: {len(ci_monthly)} months")
print(f"  Date range   : {ci_monthly.index.min().date()} → {ci_monthly.index.max().date()}")
print(f"  Missing months: {ci_monthly.isna().sum()}")
print(f"  Units: kg/MWh ≡ g/kWh — no conversion applied.")
print(f"  Min: {ci_monthly.min():.2f}  Max: {ci_monthly.max():.2f}  Mean: {ci_monthly.mean():.2f}")

# ── exogenous: PJM monthly demand ──────────────────────────────────────────
demand_available = False
if os.path.exists(DEMAND_PATH):
    try:
        dem = pd.read_csv(DEMAND_PATH, parse_dates=["datetime_utc"])
        dem.set_index("datetime_utc", inplace=True)
        dem.index = pd.to_datetime(dem.index, utc=True).tz_localize(None)
        dem_monthly = dem["demand_mwh"].resample("MS").mean().rename("demand_mwh")
        dem_monthly.dropna(inplace=True)
        demand_available = True
        print(f"\n[Exogenous] PJM monthly demand loaded: {len(dem_monthly)} months")
    except Exception as ex:
        print(f"[WARNING] Could not load PJM demand data: {ex}. Skipping.")
else:
    print(f"[WARNING] Demand file not found: {DEMAND_PATH}. Skipping.")

# ── build exogenous feature matrix ─────────────────────────────────────────
def make_exog(date_index: pd.DatetimeIndex, demand_series=None) -> pd.DataFrame:
    """Monthly cyclical features + optional demand."""
    df = pd.DataFrame(index=date_index)
    df["month_sin"] = np.sin(2 * np.pi * date_index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * date_index.month / 12)
    if demand_series is not None:
        df = df.join(demand_series, how="left")
        # Fill any missing demand with the historical monthly mean
        monthly_means = demand_series.groupby(demand_series.index.month).mean()
        for idx in df.index[df["demand_mwh"].isna()]:
            df.loc[idx, "demand_mwh"] = monthly_means.get(idx.month, demand_series.mean())
    return df.astype(float)

# ═══════════════════════════════════════════════════════════════════════════
# 2.  TRAIN / TEST SPLIT  (80 / 20 chronological)
# ═══════════════════════════════════════════════════════════════════════════
n      = len(ci_monthly)
n_train = int(n * 0.80)
n_test  = n - n_train

y_train = ci_monthly.iloc[:n_train]
y_test  = ci_monthly.iloc[n_train:]

print(f"\n[Split] Total months: {n}  |  Train: {n_train}  |  Test: {n_test}")
print(f"  Train: {y_train.index.min().date()} → {y_train.index.max().date()}")
print(f"  Test : {y_test.index.min().date()}  → {y_test.index.max().date()}")

demand_for_exog = dem_monthly if demand_available else None
exog_all   = make_exog(ci_monthly.index, demand_for_exog)
exog_train = exog_all.iloc[:n_train]
exog_test  = exog_all.iloc[n_train:]

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
order         = auto_model.order
seasonal_order = auto_model.seasonal_order
print(f"  Best order    : ARIMA{order}")
print(f"  Seasonal order: {seasonal_order}")
print(f"  AIC           : {auto_model.aic():.2f}")

# ═══════════════════════════════════════════════════════════════════════════
# 4.  REFIT WITH STATSMODELS ON TRAINING SET (full diagnostic output)
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
y_pred  = test_fc.predicted_mean.values

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"\n[Test set performance]")
print(f"  RMSE : {rmse:.4f} gCO2/kWh")
print(f"  MAE  : {mae:.4f} gCO2/kWh")
print(f"  R²   : {r2:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# 6.  REFIT ON FULL DATA FOR FORECASTING
# ═══════════════════════════════════════════════════════════════════════════
print("\n[SARIMAX] Refitting on full historical data before forecasting...")
sm_model_full = SARIMAX(
    ci_monthly,
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
N_FORECAST  = 132
last_date   = ci_monthly.index.max()
fc_dates    = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=N_FORECAST, freq="MS")

# Project exog forward: cyclical features are deterministic;
# demand is projected as the 12-month seasonal mean of the training history.
if demand_available:
    monthly_demand_means = dem_monthly.groupby(dem_monthly.index.month).mean()
    future_demand = pd.Series(
        [monthly_demand_means.get(m, dem_monthly.mean()) for m in fc_dates.month],
        index=fc_dates,
        name="demand_mwh",
    )
else:
    future_demand = None

exog_future = make_exog(fc_dates, future_demand)

fc_result = result_full.get_forecast(steps=N_FORECAST, exog=exog_future)
fc_mean   = fc_result.predicted_mean
fc_ci     = fc_result.conf_int(alpha=0.05)    # 95% CI
fc_ci_80  = fc_result.conf_int(alpha=0.20)    # 80% CI

# Guard against unrealistic values (carbon intensity cannot go below 50 g/kWh)
fc_mean   = fc_mean.clip(lower=50.0)
fc_ci     = fc_ci.clip(lower=50.0)
fc_ci_80  = fc_ci_80.clip(lower=50.0)

# ── print key forecast landmarks ──────────────────────────────────────────
def _fc_at(year: int) -> float:
    target = pd.Timestamp(f"{year}-07-01")
    closest = fc_mean.index[np.argmin(np.abs(fc_mean.index - target))]
    return float(fc_mean[closest])

print(f"\n[Forecast] Range: {fc_dates[0].date()} → {fc_dates[-1].date()}")
print(f"  Point estimate 2030 (Jul): {_fc_at(2030):.2f} gCO2/kWh")
print(f"  Point estimate 2035 (Jul): {_fc_at(2035):.2f} gCO2/kWh")

# ═══════════════════════════════════════════════════════════════════════════
# 8.  SAVE FORECAST CSV
# ═══════════════════════════════════════════════════════════════════════════
fc_df = pd.DataFrame({
    "date"        : fc_dates,
    "forecast"    : fc_mean.values,
    "lower_80"    : fc_ci_80.iloc[:, 0].values,
    "upper_80"    : fc_ci_80.iloc[:, 1].values,
    "lower_95"    : fc_ci.iloc[:, 0].values,
    "upper_95"    : fc_ci.iloc[:, 1].values,
})
fc_df.to_csv(FORECAST_CSV, index=False)
print(f"\n[Saved] Forecast CSV → {FORECAST_CSV}")

# ═══════════════════════════════════════════════════════════════════════════
# 9.  FIGURES
# ═══════════════════════════════════════════════════════════════════════════

# ── Figure 1: Historical + test predictions + forecast ────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

# Historical (train)
ax.plot(y_train.index, y_train.values, color="steelblue", linewidth=1.8,
        label="Historical (train)")
# Test actuals
ax.plot(y_test.index, y_test.values, color="darkorange", linewidth=1.8,
        label="Actuals (test)")
# Test predictions
ax.plot(y_test.index, y_pred, color="firebrick", linewidth=1.5,
        linestyle="--", label="Model fit (test)")
# Train/test split line
split_date = y_train.index.max()
ax.axvline(split_date, color="gray", linestyle="--", alpha=0.7, label="Train/Test Split")
# Forecast
ax.plot(fc_dates, fc_mean.values, color="steelblue", linewidth=1.8, linestyle="--",
        label="Forecast 2025–2035")
ax.fill_between(fc_dates, fc_ci.iloc[:,0], fc_ci.iloc[:,1],
                color="steelblue", alpha=0.2, label="95% CI")
ax.fill_between(fc_dates, fc_ci_80.iloc[:,0], fc_ci_80.iloc[:,1],
                color="steelblue", alpha=0.3, label="80% CI")

ax.set_title("Virginia Grid Carbon Intensity: SARIMAX Forecast 2025–2035",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Carbon Intensity (gCO₂/kWh)", fontsize=12)
ax.tick_params(labelsize=10)
ax.legend(fontsize=9, loc="upper right")
ax.set_xlim(ci_monthly.index.min(), fc_dates.max())
fig.tight_layout()
fig.savefig(FIG_FORECAST, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[Saved] Figure → {FIG_FORECAST}")

# ── Figure 2: Residual diagnostics ────────────────────────────────────────
residuals = result_full.resid

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Top: residuals over time
axes[0].plot(ci_monthly.index, residuals, color="steelblue", linewidth=1.0)
axes[0].axhline(0, color="gray", linestyle="--", linewidth=1.0)
axes[0].set_title("SARIMAX Residual Diagnostics — Carbon Intensity", fontsize=14,
                   fontweight="bold")
axes[0].set_xlabel("Year", fontsize=12)
axes[0].set_ylabel("Residual (gCO₂/kWh)", fontsize=12)
axes[0].tick_params(labelsize=10)
axes[0].set_xlim(ci_monthly.index.min(), ci_monthly.index.max())

# Bottom: ACF
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
# Decompose using at least 2 full seasonal cycles (need >= 24 months)
if len(ci_monthly) >= 24:
    decomp = seasonal_decompose(ci_monthly, model="additive", period=12, extrapolate_trend="freq")
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    pairs = [
        (ci_monthly, "Observed", "steelblue"),
        (decomp.trend, "Trend", "darkorange"),
        (decomp.seasonal, "Seasonal", "forestgreen"),
        (decomp.resid, "Residual", "firebrick"),
    ]
    for ax, (series, label, color) in zip(axes, pairs):
        ax.plot(series.index, series.values, color=color, linewidth=1.5)
        ax.set_ylabel(label, fontsize=11)
        ax.tick_params(labelsize=10)
        ax.set_xlim(ci_monthly.index.min(), ci_monthly.index.max())
    axes[0].set_title("Seasonal Decomposition — Virginia Grid Carbon Intensity",
                      fontsize=14, fontweight="bold")
    axes[-1].set_xlabel("Year", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DECOMP, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Figure → {FIG_DECOMP}")
else:
    print("[WARNING] Not enough data for seasonal decomposition (need ≥ 24 months).")

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
print("=" * 65)
