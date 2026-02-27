"""
Model 2 — SARIMAX Carbon Emissions Forecast
============================================
Direct monthly CO2 forecast for the Virginia data-center risk narrative.

Model form:
    y_t = emissions_tonnes_monthly
    X_t = [temp_c, carbon_intensity_g_per_kwh, cooling_degree_days, month_sin, month_cos]

Historical construction (available in this repository):
    emissions_tonnes = monthly_energy_gwh * monthly_carbon_intensity_g_per_kwh

Run from repository root:
    python GOOD_MainModels/model2_sarimax_carbon_emissions.py

Outputs:
    GOOD_MainModels/carbon_emissions_forecast.csv
    GOOD_MainModels/carbon_emissions_model_summary.txt
    GOOD_Figures/carbon_emissions_historical_and_forecast.png
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from pmdarima import auto_arima
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit("Missing required library: pmdarima. Install with `pip install pmdarima`.") from exc

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit("Missing required library: statsmodels. Install with `pip install statsmodels`.") from exc


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data_Sources" / "cleaned"
FIG_DIR = ROOT / "GOOD_Figures"
OUT_DIR = ROOT / "GOOD_MainModels"

DC_POWER_PATH = DATA_DIR / "semisynthetic_datacenter_power_2015_2024.csv"
CARBON_INTENSITY_PATH = DATA_DIR / "pjm_grid_carbon_intensity_2019_full_cleaned.csv"
NOAA_TEMP_PATH = DATA_DIR / "noaa_global_hourly_dulles_2019_2024_cleaned.csv"

OUT_CSV = OUT_DIR / "carbon_emissions_forecast.csv"
OUT_SUMMARY = OUT_DIR / "carbon_emissions_model_summary.txt"
OUT_FIG = FIG_DIR / "carbon_emissions_historical_and_forecast.png"

N_FORECAST_MONTHS = 132  # 2025-01 through 2035-12
CDD_BASE_TEMP_C = 18.0


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")


def _month_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    out["month_sin"] = np.sin(2 * np.pi * index.month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * index.month / 12.0)
    return out


def _monthly_training_frame() -> pd.DataFrame:
    _require_file(DC_POWER_PATH)
    _require_file(CARBON_INTENSITY_PATH)
    _require_file(NOAA_TEMP_PATH)

    dc = pd.read_csv(DC_POWER_PATH, parse_dates=["datetime"])
    dc["datetime"] = pd.to_datetime(dc["datetime"], errors="coerce")
    dc = dc.dropna(subset=["datetime"]).set_index("datetime").sort_index()

    # Hourly MW -> monthly GWh
    energy_monthly = (dc["total_power_mw"].resample("MS").sum() / 1000.0).rename("energy_gwh")

    ci = pd.read_csv(CARBON_INTENSITY_PATH, parse_dates=["timestamp"])
    ci["timestamp"] = pd.to_datetime(ci["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)
    ci = ci.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    ci_monthly = ci["carbon_intensity_kg_per_mwh"].resample("MS").mean().rename("carbon_intensity_g_per_kwh")

    wx = pd.read_csv(NOAA_TEMP_PATH, parse_dates=["timestamp"])
    wx["timestamp"] = pd.to_datetime(wx["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)
    wx = wx.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    temp_monthly = wx["temperature_c"].resample("MS").mean().rename("temp_c")

    frame = pd.concat([energy_monthly, ci_monthly, temp_monthly], axis=1).dropna()
    if frame.empty:
        raise ValueError("No overlapping monthly rows across energy, carbon intensity, and temperature.")

    # Units: GWh * g/kWh == tonnes CO2
    frame["emissions_tonnes"] = frame["energy_gwh"] * frame["carbon_intensity_g_per_kwh"]
    frame["cooling_degree_days"] = np.maximum(frame["temp_c"] - CDD_BASE_TEMP_C, 0.0) * frame.index.days_in_month
    frame = frame.join(_month_features(frame.index))
    return frame


def _future_exog(frame: pd.DataFrame, horizon_months: int) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    last_date = frame.index.max()
    future_idx = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=horizon_months, freq="MS")

    hist_temp = frame["temp_c"]
    temp_monthly_mean = hist_temp.groupby(hist_temp.index.month).mean()
    future_temp = np.array([float(temp_monthly_mean.get(m, hist_temp.mean())) for m in future_idx.month], dtype=float)

    # Carbon-intensity future path: linear trend + historical seasonal residual.
    hist_ci = frame["carbon_intensity_g_per_kwh"]
    t_hist = np.arange(len(hist_ci), dtype=float)
    slope, intercept = np.polyfit(t_hist, hist_ci.values.astype(float), 1)
    trend_fit = intercept + slope * t_hist
    seasonal_resid = hist_ci.values.astype(float) - trend_fit
    seasonal_lookup = (
        pd.DataFrame({"month": hist_ci.index.month, "resid": seasonal_resid})
        .groupby("month")["resid"]
        .mean()
    )
    t_future = np.arange(len(hist_ci), len(hist_ci) + horizon_months, dtype=float)
    future_ci = intercept + slope * t_future + np.array(
        [float(seasonal_lookup.get(int(m), 0.0)) for m in future_idx.month],
        dtype=float,
    )
    future_ci = np.clip(future_ci, 50.0, None)

    future_cdd = np.maximum(future_temp - CDD_BASE_TEMP_C, 0.0) * future_idx.days_in_month
    future = pd.DataFrame(
        {
            "temp_c": future_temp,
            "carbon_intensity_g_per_kwh": future_ci,
            "cooling_degree_days": future_cdd.astype(float),
        },
        index=future_idx,
    )
    future = future.join(_month_features(future_idx))
    return future_idx, future.astype(float)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("MODEL 2 — SARIMAX Carbon Emissions Forecast")
    print("=" * 72)

    frame = _monthly_training_frame()
    y = frame["emissions_tonnes"].astype(float)
    exog_cols = [
        "temp_c",
        "carbon_intensity_g_per_kwh",
        "cooling_degree_days",
        "month_sin",
        "month_cos",
    ]
    exog = frame[exog_cols].astype(float)

    n = len(frame)
    n_train = int(n * 0.80)
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]
    x_train, x_test = exog.iloc[:n_train], exog.iloc[n_train:]

    print(f"Training months: {len(y_train)} | Test months: {len(y_test)}")
    print(f"History range : {y.index.min().date()} to {y.index.max().date()}")

    arima = auto_arima(
        y_train,
        exogenous=x_train.values,
        seasonal=True,
        m=12,
        information_criterion="aic",
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
    )
    order = arima.order
    seasonal_order = arima.seasonal_order
    print(f"Selected order: ARIMA{order} seasonal={seasonal_order}")

    fit_train = SARIMAX(
        y_train,
        exog=x_train,
        order=order,
        seasonal_order=seasonal_order,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=200)

    pred_test = fit_train.get_forecast(steps=len(y_test), exog=x_test).predicted_mean.clip(lower=0.0)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))
    mae = float(mean_absolute_error(y_test, pred_test))
    r2 = float(r2_score(y_test, pred_test)) if len(y_test) > 1 else float("nan")
    print(f"Test RMSE={rmse:.2f} tonnes/month | MAE={mae:.2f} | R2={r2:.3f}")

    fit_full = SARIMAX(
        y,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=200)

    fc_dates, x_future = _future_exog(frame, N_FORECAST_MONTHS)
    fc = fit_full.get_forecast(steps=N_FORECAST_MONTHS, exog=x_future)
    fc_mean = fc.predicted_mean.clip(lower=0.0)
    fc_95 = fc.conf_int(alpha=0.05).clip(lower=0.0)
    fc_80 = fc.conf_int(alpha=0.20).clip(lower=0.0)

    out = pd.DataFrame(
        {
            "date": fc_dates,
            "emissions_tonnes_monthly": fc_mean.values.astype(float),
            "lower_80": fc_80.iloc[:, 0].values.astype(float),
            "upper_80": fc_80.iloc[:, 1].values.astype(float),
            "lower_95": fc_95.iloc[:, 0].values.astype(float),
            "upper_95": fc_95.iloc[:, 1].values.astype(float),
        }
    )
    out["year"] = pd.to_datetime(out["date"]).dt.year
    annual = out.groupby("year", as_index=False)["emissions_tonnes_monthly"].sum().rename(
        columns={"emissions_tonnes_monthly": "emissions_tonnes_annual"}
    )
    out = out.merge(annual, on="year", how="left")
    out = out[
        [
            "date",
            "emissions_tonnes_monthly",
            "emissions_tonnes_annual",
            "lower_95",
            "upper_95",
            "lower_80",
            "upper_80",
        ]
    ]
    out.to_csv(OUT_CSV, index=False)

    summary_lines = [
        "MODEL 2 — SARIMAX Carbon Emissions Forecast",
        f"History: {y.index.min().date()} to {y.index.max().date()} ({len(y)} months)",
        f"ARIMA order: {order}",
        f"Seasonal order: {seasonal_order}",
        f"Test RMSE (tonnes/month): {rmse:.4f}",
        f"Test MAE  (tonnes/month): {mae:.4f}",
        f"Test R2: {r2:.6f}",
        f"Forecast range: {fc_dates.min().date()} to {fc_dates.max().date()}",
        f"2035 annual forecast (tonnes): {annual.loc[annual['year'] == 2035, 'emissions_tonnes_annual'].iloc[0]:,.0f}"
        if (annual["year"] == 2035).any()
        else "2035 annual forecast unavailable.",
        "",
        "Exogenous variables: temp_c, carbon_intensity_g_per_kwh, cooling_degree_days, month_sin, month_cos",
    ]
    OUT_SUMMARY.write_text("\n".join(summary_lines), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y.index, y.values, color="steelblue", linewidth=1.8, label="Historical monthly emissions")
    ax.plot(fc_dates, fc_mean.values, color="firebrick", linewidth=1.8, linestyle="--", label="Forecast 2025-2035")
    ax.fill_between(fc_dates, fc_95.iloc[:, 0], fc_95.iloc[:, 1], color="firebrick", alpha=0.20, label="95% CI")
    ax.fill_between(fc_dates, fc_80.iloc[:, 0], fc_80.iloc[:, 1], color="firebrick", alpha=0.30, label="80% CI")
    ax.axvline(y.index.max(), color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_title("Model 2: Virginia Data-Center CO2 Emissions Forecast", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("CO2 emissions (tonnes/month)", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_SUMMARY}")
    print(f"Saved: {OUT_FIG}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI entrypoint guard
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
