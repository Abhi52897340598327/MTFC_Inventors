"""
Model 3 — Grid Stress / Peak Demand Exceedance Analysis
=======================================================
Converts Model 1 energy forecasts into an actuarial-style grid stress estimate.

Inputs:
    GOOD_MainModels/energy_usage_forecast.csv
    Data_Sources/cleaned/pjm_hourly_demand_2019_2024_cleaned.csv

Outputs:
    GOOD_MainModels/grid_stress_annual_summary.csv
    GOOD_MainModels/grid_stress_model_summary.txt
    GOOD_Figures/grid_stress_cost_and_margin.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data_Sources" / "cleaned"
OUT_DIR = ROOT / "GOOD_MainModels"
FIG_DIR = ROOT / "GOOD_Figures"

ENERGY_FORECAST_PATH = OUT_DIR / "energy_usage_forecast.csv"
PJM_DEMAND_PATH = DATA_DIR / "pjm_hourly_demand_2019_2024_cleaned.csv"

OUT_CSV = OUT_DIR / "grid_stress_annual_summary.csv"
OUT_SUMMARY = OUT_DIR / "grid_stress_model_summary.txt"
OUT_FIG = FIG_DIR / "grid_stress_cost_and_margin.png"

RESERVE_MARGIN_REQUIREMENT = 0.15
PEAK_PRICE_PREMIUM_USD_PER_MWH = 300.0
SCENARIO_COLUMN_MAP = {
    "conservative": "lower_95",
    "baseline": "forecast_GWh",
    "aggressive": "upper_95",
}


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")


def _load_energy_forecast() -> pd.DataFrame:
    _require_file(ENERGY_FORECAST_PATH)
    fc = pd.read_csv(ENERGY_FORECAST_PATH, parse_dates=["date"])
    fc["date"] = pd.to_datetime(fc["date"], errors="coerce")
    fc = fc.dropna(subset=["date"]).sort_values("date")
    required = {"forecast_GWh", "lower_95", "upper_95"}
    missing = required - set(fc.columns)
    if missing:
        raise ValueError(f"Energy forecast is missing required columns: {sorted(missing)}")
    return fc


def _load_pjm_hourly() -> pd.DataFrame:
    _require_file(PJM_DEMAND_PATH)
    pjm = pd.read_csv(PJM_DEMAND_PATH, parse_dates=["datetime_utc"])
    pjm["datetime_utc"] = pd.to_datetime(pjm["datetime_utc"], errors="coerce")
    pjm["demand_mw"] = pd.to_numeric(pjm["demand_mwh"], errors="coerce")
    pjm = pjm.dropna(subset=["datetime_utc", "demand_mw"]).sort_values("datetime_utc")
    # Remove obvious sentinel/outlier values from source pulls (e.g., int overflow artifacts).
    pjm = pjm[(pjm["demand_mw"] > 1_000.0) & (pjm["demand_mw"] < 250_000.0)]
    pjm = pjm.groupby("datetime_utc", as_index=False)["demand_mw"].mean()
    return pjm


def _historical_profiles(pjm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    hist = pjm.copy()
    hist["month"] = hist["datetime_utc"].dt.month
    hist["dow"] = hist["datetime_utc"].dt.dayofweek
    hist["hour"] = hist["datetime_utc"].dt.hour

    month_mean = hist.groupby("month")["demand_mw"].transform("mean")
    hist["shape"] = hist["demand_mw"] / month_mean.replace(0.0, np.nan)
    hist["shape"] = hist["shape"].replace([np.inf, -np.inf], np.nan).fillna(1.0)

    demand_profile = hist.groupby(["month", "dow", "hour"], as_index=False)["demand_mw"].mean()
    shape_profile = hist.groupby(["month", "dow", "hour"], as_index=False)["shape"].mean()
    # Planning peak is less brittle than max() and aligns with reserve planning practice.
    planning_peak = float(hist["demand_mw"].quantile(0.995))
    return demand_profile, shape_profile, planning_peak


def _build_future_baseline(
    year: int,
    demand_profile: pd.DataFrame,
) -> pd.DataFrame:
    idx = pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 23:00:00", freq="h")
    out = pd.DataFrame({"timestamp": idx})
    out["month"] = out["timestamp"].dt.month
    out["dow"] = out["timestamp"].dt.dayofweek
    out["hour"] = out["timestamp"].dt.hour

    month_hour = demand_profile.groupby(["month", "hour"], as_index=False)["demand_mw"].mean()
    global_mean = float(demand_profile["demand_mw"].mean())

    out = out.merge(demand_profile, on=["month", "dow", "hour"], how="left")
    out = out.merge(month_hour, on=["month", "hour"], how="left", suffixes=("", "_fallback"))
    out["baseline_demand_mw"] = (
        out["demand_mw"]
        .fillna(out["demand_mw_fallback"])
        .fillna(global_mean)
        .astype(float)
    )
    return out[["timestamp", "month", "dow", "hour", "baseline_demand_mw"]]


def _build_dc_load_profile(
    future_baseline: pd.DataFrame,
    shape_profile: pd.DataFrame,
    monthly_energy_gwh: pd.Series,
) -> pd.Series:
    month_hours = future_baseline.groupby("month")["timestamp"].count().astype(float)
    month_avg_mw = (monthly_energy_gwh * 1000.0) / month_hours

    shape_month_hour = shape_profile.groupby(["month", "hour"], as_index=False)["shape"].mean()
    out = future_baseline.merge(shape_profile, on=["month", "dow", "hour"], how="left")
    out = out.merge(shape_month_hour, on=["month", "hour"], how="left", suffixes=("", "_fallback"))
    out["shape_used"] = out["shape"].fillna(out["shape_fallback"]).fillna(1.0).astype(float)
    out["month_avg_mw"] = out["month"].map(month_avg_mw.to_dict()).fillna(0.0).astype(float)
    return (out["month_avg_mw"] * out["shape_used"]).clip(lower=0.0)


def _annual_results(
    energy_fc: pd.DataFrame,
    demand_profile: pd.DataFrame,
    shape_profile: pd.DataFrame,
    planning_peak_mw: float,
) -> pd.DataFrame:
    capacity_mw = planning_peak_mw * (1.0 + RESERVE_MARGIN_REQUIREMENT)
    safe_threshold_mw = capacity_mw / (1.0 + RESERVE_MARGIN_REQUIREMENT)

    rows: list[dict[str, float | int | str]] = []
    years = sorted(pd.to_datetime(energy_fc["date"]).dt.year.unique().tolist())

    for scenario_name, scenario_col in SCENARIO_COLUMN_MAP.items():
        for year in years:
            year_slice = energy_fc[pd.to_datetime(energy_fc["date"]).dt.year == year].copy()
            if year_slice.empty:
                continue

            monthly_energy = year_slice.groupby(pd.to_datetime(year_slice["date"]).dt.month)[scenario_col].sum()
            future = _build_future_baseline(year, demand_profile=demand_profile)
            dc_load_mw = _build_dc_load_profile(
                future_baseline=future,
                shape_profile=shape_profile,
                monthly_energy_gwh=monthly_energy,
            )

            total_mw = future["baseline_demand_mw"].to_numpy(dtype=float) + dc_load_mw.to_numpy(dtype=float)
            # Stress hours are defined as datacenter operation during grid peak-demand windows.
            # We use the 95th percentile of baseline demand as the peak-window threshold.
            peak_window_threshold_mw = float(np.quantile(future["baseline_demand_mw"].to_numpy(dtype=float), 0.95))
            stress_mask = total_mw >= peak_window_threshold_mw
            stress_hours = int(stress_mask.sum())
            stress_energy_mwh = float(dc_load_mw.to_numpy(dtype=float)[stress_mask].sum())
            stress_cost_usd = float(stress_energy_mwh * PEAK_PRICE_PREMIUM_USD_PER_MWH)
            projected_peak_mw = float(np.max(total_mw))
            reserve_margin_pct = float((capacity_mw - projected_peak_mw) / projected_peak_mw * 100.0)

            annual_energy_gwh = float(monthly_energy.sum())
            annual_avg_dc_mw = float(annual_energy_gwh * 1000.0 / len(future))

            rows.append(
                {
                    "year": int(year),
                    "scenario": scenario_name,
                    "dc_annual_energy_gwh": annual_energy_gwh,
                    "dc_annual_avg_mw": annual_avg_dc_mw,
                    "planning_peak_mw_q995": planning_peak_mw,
                    "capacity_mw_at_15pct_margin": capacity_mw,
                    "projected_peak_mw": projected_peak_mw,
                    "projected_reserve_margin_pct": reserve_margin_pct,
                    "peak_window_threshold_mw_p95": peak_window_threshold_mw,
                    "stress_hours": stress_hours,
                    "stress_energy_mwh": stress_energy_mwh,
                    "peak_price_premium_usd_per_mwh": PEAK_PRICE_PREMIUM_USD_PER_MWH,
                    "grid_stress_cost_usd": stress_cost_usd,
                }
            )

    return pd.DataFrame(rows).sort_values(["scenario", "year"]).reset_index(drop=True)


def _plot_results(results: pd.DataFrame) -> None:
    baseline = results[results["scenario"] == "baseline"].copy()
    if baseline.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(
        baseline["year"],
        baseline["projected_reserve_margin_pct"],
        color="firebrick",
        linewidth=2.0,
        marker="o",
    )
    axes[0].axhline(RESERVE_MARGIN_REQUIREMENT * 100.0, color="gray", linestyle="--", linewidth=1.2)
    axes[0].set_ylabel("Reserve margin (%)")
    axes[0].set_title("Model 3: Baseline Reserve Margin Erosion and Grid Stress Cost", fontsize=14, fontweight="bold")
    axes[0].grid(alpha=0.3)

    axes[1].bar(
        baseline["year"],
        baseline["grid_stress_cost_usd"] / 1e6,
        color="steelblue",
        alpha=0.85,
    )
    axes[1].set_ylabel("Grid stress cost ($M)")
    axes[1].set_xlabel("Year")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_xticks(baseline["year"].tolist())

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("MODEL 3 — Grid Stress / Peak Demand Exceedance Analysis")
    print("=" * 72)

    energy_fc = _load_energy_forecast()
    pjm = _load_pjm_hourly()

    demand_profile, shape_profile, planning_peak_mw = _historical_profiles(pjm)
    results = _annual_results(
        energy_fc=energy_fc,
        demand_profile=demand_profile,
        shape_profile=shape_profile,
        planning_peak_mw=planning_peak_mw,
    )
    if results.empty:
        raise ValueError("Grid stress analysis produced no rows.")

    results.to_csv(OUT_CSV, index=False)
    _plot_results(results)

    baseline = results[results["scenario"] == "baseline"].copy()
    min_margin = float(baseline["projected_reserve_margin_pct"].min()) if not baseline.empty else float("nan")
    max_stress_hours = int(baseline["stress_hours"].max()) if not baseline.empty else 0
    max_cost = float(baseline["grid_stress_cost_usd"].max()) if not baseline.empty else 0.0

    summary = [
        "MODEL 3 — Grid Stress / Peak Demand Exceedance Analysis",
        f"Reserve margin requirement: {RESERVE_MARGIN_REQUIREMENT:.0%}",
        f"Peak price premium used: ${PEAK_PRICE_PREMIUM_USD_PER_MWH:,.0f} per MWh",
        f"Planning peak baseline q99.5 demand (MW): {planning_peak_mw:,.2f}",
        f"Minimum baseline reserve margin through forecast: {min_margin:.2f}%",
        f"Maximum baseline stress hours in one year: {max_stress_hours:,}",
        f"Maximum baseline annual grid stress cost: ${max_cost:,.0f}",
        "",
        "Scenarios:",
        "- conservative uses Model 1 lower_95 energy path",
        "- baseline uses Model 1 point forecast path",
        "- aggressive uses Model 1 upper_95 energy path",
    ]
    OUT_SUMMARY.write_text("\n".join(summary), encoding="utf-8")

    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_SUMMARY}")
    print(f"Saved: {OUT_FIG}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI entrypoint guard
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
