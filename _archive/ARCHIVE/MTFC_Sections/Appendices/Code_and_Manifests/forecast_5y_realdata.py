"""Run a 5-year forecast using only real exogenous data already in Data_Sources.

Outputs:
- Hourly forecast CSV (emissions, energy, stress metrics)
- Yearly summary CSV (forecast years 1..5)
- Graphs for emissions, energy, and grid stress
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Prefer vendored dependencies when present.
DEPS_DIR = Path(__file__).resolve().parent / ".deps"
if DEPS_DIR.exists():
    sys.path.insert(0, str(DEPS_DIR))

from carbon_prediction_pipeline import CarbonForecastPipeline
from config import make_config


def _parse_ts(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    return ts.dt.tz_convert(None)


def _load_real_future_exog(
    data_dir: Path,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    required_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    grid_path = data_dir / "cleaned/pjm_exogenous_hourly_2019_2024_cleaned.csv"
    weather_path = data_dir / "cleaned/noaa_global_hourly_dulles_2019_2024_cleaned.csv"
    demand_raw_path = data_dir / "cleaned/pjm_hourly_demand_2019_2024_cleaned.csv"

    grid = pd.read_csv(grid_path)
    grid["timestamp"] = _parse_ts(grid["timestamp"])

    weather = pd.read_csv(weather_path)
    weather["timestamp"] = _parse_ts(weather["timestamp"])

    demand_raw = pd.read_csv(demand_raw_path)
    demand_raw["timestamp"] = _parse_ts(demand_raw["datetime_utc"])
    demand_raw = (
        demand_raw.groupby("timestamp", as_index=False)
        .agg(demand_mwh_raw=("demand_mwh", "mean"))
        .sort_values("timestamp")
    )

    future = grid.merge(weather, on="timestamp", how="inner")
    future = future.merge(demand_raw, on="timestamp", how="left")

    mask = (future["timestamp"] >= start_ts) & (future["timestamp"] < end_ts)
    future = future.loc[mask].sort_values("timestamp").reset_index(drop=True)

    # Keep only columns we actually need for this pipeline contract.
    cols_needed = ["timestamp"] + [c for c in required_cols if c != "timestamp"]
    cols_needed = [c for c in cols_needed if c in future.columns]
    future = future[cols_needed].copy()

    # Fill demand_mwh_raw from demand_mwh only when raw demand is unavailable.
    if "demand_mwh_raw" in required_cols and "demand_mwh_raw" not in future.columns and "demand_mwh" in future.columns:
        future["demand_mwh_raw"] = future["demand_mwh"]

    missing_required = [c for c in required_cols if c not in future.columns]
    if missing_required:
        raise ValueError(
            "Real future exogenous frame is missing required columns: "
            f"{missing_required}"
        )

    future = future[["timestamp"] + [c for c in required_cols if c != "timestamp"]]

    # Enforce full hourly calendar over 5 years, then fill gaps from adjacent real observations.
    full_ts = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start=start_ts,
                end=end_ts - pd.Timedelta(hours=1),
                freq="h",
            )
        }
    )
    future = full_ts.merge(future, on="timestamp", how="left")

    numeric_cols = [c for c in future.columns if c != "timestamp"]
    missing_before = {c: int(future[c].isna().sum()) for c in numeric_cols}
    rows_with_missing_before = int(future[numeric_cols].isna().any(axis=1).sum())

    for col in numeric_cols:
        future[col] = pd.to_numeric(future[col], errors="coerce").ffill().bfill()

    missing_after = {c: int(future[c].isna().sum()) for c in numeric_cols}
    bad_cols = [c for c in numeric_cols if missing_after[c] > 0]
    if bad_cols:
        raise ValueError(f"Unable to fill missing values in required future columns: {bad_cols}")

    fill_stats = {
        "rows_total": int(len(future)),
        "rows_with_any_missing_before_fill": rows_with_missing_before,
        "total_missing_cells_before_fill": int(sum(missing_before.values())),
        "total_missing_cells_after_fill": int(sum(missing_after.values())),
    }
    return future, fill_stats


def _stress_percentile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref = np.sort(reference[np.isfinite(reference)])
    if ref.size == 0:
        return np.full_like(values, np.nan, dtype=float)
    ranks = np.searchsorted(ref, values, side="right")
    return (ranks / ref.size) * 100.0


def _plot_hourly_series(
    df_hourly: pd.DataFrame,
    value_col: str,
    title: str,
    y_label: str,
    path: Path,
    color_hourly: str,
    color_smooth: str,
    add_reference_one: bool = False,
    y_max: float | None = None,
) -> None:
    work = df_hourly[["timestamp", value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[value_col]).sort_values("timestamp")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(
        work["timestamp"],
        work[value_col],
        color=color_hourly,
        linewidth=0.6,
        alpha=0.5,
        label="Hourly",
    )

    smooth = (
        work.set_index("timestamp")[value_col]
        .rolling(window=24, min_periods=1)
        .mean()
    )
    ax.plot(
        smooth.index,
        smooth.values,
        color=color_smooth,
        linewidth=1.6,
        alpha=0.95,
        label="24h Rolling Mean",
    )

    if add_reference_one:
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0, label="Ratio = 1.0")

    if y_max is not None:
        ax.set_ylim(top=float(y_max))

    ax.set_title(title)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _make_plots(df_yearly: pd.DataFrame, df_hourly: pd.DataFrame, fig_dir: Path) -> dict[str, str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    # Emissions
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df_yearly["forecast_year"], df_yearly["total_emissions"], color="#d95f02")
    ax.set_title("5-Year Forecast: Total Carbon Emissions by Forecast Year")
    ax.set_xlabel("Forecast Year")
    ax.set_ylabel("Total Emissions (model units)")
    p = fig_dir / "forecast_5y_emissions_by_year.png"
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths["emissions"] = str(p)

    # Energy
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_yearly["forecast_year"], df_yearly["total_energy_mwh"], marker="o", color="#1b9e77")
    ax.set_title("5-Year Forecast: Total Energy Consumption by Forecast Year")
    ax.set_xlabel("Forecast Year")
    ax.set_ylabel("Energy (MWh)")
    p = fig_dir / "forecast_5y_energy_by_year.png"
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths["energy"] = str(p)

    # Grid stress
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(
        df_yearly["forecast_year"],
        df_yearly["grid_stress_ratio_p95"],
        marker="o",
        color="#7570b3",
        label="P95 Stress Ratio",
    )
    ax1.set_xlabel("Forecast Year")
    ax1.set_ylabel("Grid Stress Ratio (P95)", color="#7570b3")
    ax1.tick_params(axis="y", labelcolor="#7570b3")
    ax1.set_ylim(top=1.25)

    ax2 = ax1.twinx()
    ax2.plot(
        df_yearly["forecast_year"],
        df_yearly["grid_stress_score_mean"],
        marker="s",
        color="#e7298a",
        label="Mean Stress Score",
    )
    ax2.set_ylabel("Grid Stress Score (0-100)", color="#e7298a")
    ax2.tick_params(axis="y", labelcolor="#e7298a")
    ax1.set_title("5-Year Forecast: Grid Stress by Forecast Year")
    p = fig_dir / "forecast_5y_grid_stress_by_year.png"
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths["grid_stress"] = str(p)

    # Hourly emissions
    p = fig_dir / "forecast_5y_hourly_emissions.png"
    _plot_hourly_series(
        df_hourly=df_hourly,
        value_col="pred_emissions",
        title="5-Year Forecast (Hourly): Carbon Emissions",
        y_label="Predicted Emissions (model units)",
        path=p,
        color_hourly="#fdd0a2",
        color_smooth="#d95f02",
    )
    paths["emissions_hourly"] = str(p)

    # Hourly energy
    p = fig_dir / "forecast_5y_hourly_energy_mwh.png"
    _plot_hourly_series(
        df_hourly=df_hourly,
        value_col="pred_energy_mwh",
        title="5-Year Forecast (Hourly): Energy Consumption",
        y_label="Predicted Energy (MWh)",
        path=p,
        color_hourly="#a6dba0",
        color_smooth="#1b9e77",
    )
    paths["energy_hourly"] = str(p)

    # Hourly grid stress ratio
    p = fig_dir / "forecast_5y_hourly_grid_stress_ratio.png"
    _plot_hourly_series(
        df_hourly=df_hourly,
        value_col="grid_stress_ratio",
        title="5-Year Forecast (Hourly): Grid Stress Ratio",
        y_label="Grid Stress Ratio",
        path=p,
        color_hourly="#cbc9e2",
        color_smooth="#7570b3",
        add_reference_one=True,
        y_max=1.25,
    )
    paths["grid_stress_hourly"] = str(p)

    return paths


def main() -> None:
    cfg = make_config()
    pipe = CarbonForecastPipeline(cfg)

    history = pipe.load_data().sort_values("timestamp").reset_index(drop=True)
    pipe.fit(history)

    start_ts = history["timestamp"].max() + pd.Timedelta(hours=1)
    end_ts = start_ts + pd.DateOffset(years=5)

    required = list(pipe.required_future_exog_columns or ["timestamp", "temperature_c"])
    future_exog, fill_stats = _load_real_future_exog(cfg.data_dir, start_ts, end_ts, required)

    horizon_hours = len(future_exog)
    if horizon_hours < 24 * 365:
        raise ValueError(
            f"Future real-data horizon is too short for 5 years: {horizon_hours} hours"
        )

    pred = pipe.predict(history, future_exog, horizon_hours=horizon_hours)
    merged = pred.merge(future_exog, on="timestamp", how="left")

    # Use model's combined-energy output where available; fallback to physics power * step-hours.
    energy_mwh = merged["pred_energy_mwh_ml"].to_numpy(dtype=float)
    fallback_energy = merged["pred_total_power"].to_numpy(dtype=float) * float(pipe.step_hours)
    merged["pred_energy_mwh"] = np.where(np.isfinite(energy_mwh), energy_mwh, fallback_energy)

    # Grid stress ratio: demand divided by available supply proxy.
    supply = merged["net_generation_mwh"].to_numpy(dtype=float) + merged["interchange_mwh"].to_numpy(dtype=float)
    supply = np.where(np.abs(supply) < 1e-6, np.nan, supply)
    merged["grid_stress_ratio"] = merged["demand_mwh"].to_numpy(dtype=float) / supply

    # Grid stress score: percentile vs historical stress baseline from real observed history.
    hist_supply = history["net_generation_mwh"].to_numpy(dtype=float) + history["interchange_mwh"].to_numpy(dtype=float)
    hist_supply = np.where(np.abs(hist_supply) < 1e-6, np.nan, hist_supply)
    hist_stress = history["demand_mwh"].to_numpy(dtype=float) / hist_supply
    merged["grid_stress_score"] = _stress_percentile(
        merged["grid_stress_ratio"].to_numpy(dtype=float),
        hist_stress,
    )

    # Forecast year index 1..5 using fixed 365-day bins from forecast start.
    elapsed_days = (merged["timestamp"] - start_ts).dt.total_seconds() / 86400.0
    merged["forecast_year"] = (elapsed_days // 365.0).astype(int) + 1
    merged = merged[merged["forecast_year"].between(1, 5)].copy()

    yearly = (
        merged.groupby("forecast_year", as_index=False)
        .agg(
            start_timestamp=("timestamp", "min"),
            end_timestamp=("timestamp", "max"),
            hours=("timestamp", "count"),
            total_emissions=("pred_emissions", "sum"),
            total_energy_mwh=("pred_energy_mwh", "sum"),
            grid_stress_ratio_mean=("grid_stress_ratio", "mean"),
            grid_stress_ratio_p95=("grid_stress_ratio", lambda s: float(np.nanpercentile(s, 95))),
            grid_stress_score_mean=("grid_stress_score", "mean"),
            high_stress_hours_90p=("grid_stress_score", lambda s: int(np.sum(np.asarray(s) >= 90.0))),
        )
        .sort_values("forecast_year")
        .reset_index(drop=True)
    )

    out_tag = f"{pipe.run_id}_5y_realdata"
    result_dir = cfg.results_dir / out_tag
    fig_dir = cfg.figure_dir / out_tag
    result_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    merged.to_csv(result_dir / "forecast_5y_hourly.csv", index=False)
    yearly.to_csv(result_dir / "forecast_5y_yearly_summary.csv", index=False)

    plot_paths = _make_plots(yearly, merged, fig_dir)

    metadata = {
        "run_id": pipe.run_id,
        "output_tag": out_tag,
        "history_start": str(history["timestamp"].min()),
        "history_end": str(history["timestamp"].max()),
        "forecast_start": str(start_ts),
        "forecast_end_exclusive": str(end_ts),
        "forecast_hours": int(len(merged)),
        "required_future_exog_columns": required,
        "future_exog_fill_stats": fill_stats,
        "plots": plot_paths,
    }
    with open(result_dir / "forecast_5y_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Saved:")
    print(result_dir / "forecast_5y_hourly.csv")
    print(result_dir / "forecast_5y_yearly_summary.csv")
    print(result_dir / "forecast_5y_metadata.json")
    for k, v in plot_paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
