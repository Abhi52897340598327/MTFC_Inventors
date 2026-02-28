"""Define danger rules, compute joint risk, simulate mitigations, and export MTFC-ready artifacts.

This script reads the latest 5-year real-data forecast output and produces:
- clear danger-rule thresholds
- hourly baseline vs mitigated risk metrics
- quantified mitigation impacts
- before/after visuals
- an operations playbook markdown
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib
import numpy as np
import pandas as pd

# Force a non-GUI backend for headless/sandbox execution.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _find_latest_forecast_dir(results_root: Path) -> Path:
    candidates = [p for p in results_root.iterdir() if p.is_dir() and p.name.endswith("_5y_realdata")]
    if not candidates:
        raise FileNotFoundError(f"No '*_5y_realdata' run directories found in {results_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _percentile_from_reference(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref = np.sort(reference[np.isfinite(reference)])
    if ref.size == 0:
        return np.full_like(values, np.nan, dtype=float)
    ranks = np.searchsorted(ref, values, side="right")
    return (ranks / ref.size) * 100.0


def _simulate_mitigation(df: pd.DataFrame, emissions_threshold: float) -> pd.DataFrame:
    work = df.copy()

    base_energy = work["pred_energy_mwh"].to_numpy(dtype=float)
    base_emissions = work["pred_emissions"].to_numpy(dtype=float)
    base_stress = work["grid_stress_ratio"].to_numpy(dtype=float)
    base_demand = work["demand_mwh"].to_numpy(dtype=float)
    supply = (work["net_generation_mwh"] + work["interchange_mwh"]).to_numpy(dtype=float)
    supply = np.where(np.abs(supply) < 1e-9, np.nan, supply)

    high_stress = base_stress > 1.0
    high_emissions = base_emissions >= emissions_threshold
    red_hours = high_stress | high_emissions

    # Mitigation policy assumptions:
    # - load shifting: move 8% of red-hour energy to lower-risk hours
    # - cooling optimization / peak shaving: save 4% during red hours
    shift_frac = 0.08
    efficiency_frac = 0.04

    shifted_out = np.where(red_hours, base_energy * shift_frac, 0.0)
    efficiency_saved = np.where(red_hours, base_energy * efficiency_frac, 0.0)

    mitigated_energy = base_energy - shifted_out - efficiency_saved

    # Redistribute shifted energy to low-risk hours with the most headroom.
    recipient_mask = (work["joint_risk_score_baseline"].to_numpy(dtype=float) <= 40.0) & (base_stress < 0.95)
    if int(np.sum(recipient_mask)) == 0:
        recipient_mask = ~red_hours
    if int(np.sum(recipient_mask)) == 0:
        recipient_mask = np.ones(len(work), dtype=bool)

    headroom = np.clip(1.0 - base_stress[recipient_mask], 1e-6, None)
    weights = headroom / np.sum(headroom)
    redistributed = np.zeros(len(work), dtype=float)
    redistributed[np.where(recipient_mask)[0]] = np.sum(shifted_out) * weights
    mitigated_energy = mitigated_energy + redistributed
    mitigated_energy = np.clip(mitigated_energy, 0.0, None)

    # Grid stress after mitigation via adjusted demand.
    demand_delta = mitigated_energy - base_energy
    mitigated_demand = base_demand + demand_delta
    mitigated_stress = mitigated_demand / supply

    # Emissions after mitigation: same carbon intensity profile, lower/shifted energy.
    energy_ratio = np.divide(mitigated_energy, base_energy, out=np.ones_like(base_energy), where=base_energy > 0)
    mitigated_emissions = base_emissions * energy_ratio

    work["mitigated_energy_mwh"] = mitigated_energy
    work["mitigated_demand_mwh"] = mitigated_demand
    work["mitigated_grid_stress_ratio"] = mitigated_stress
    work["mitigated_emissions"] = mitigated_emissions

    return work


def _make_plots(df: pd.DataFrame, risk_dir: Path, rec_dir: Path) -> Dict[str, str]:
    plot_paths: Dict[str, str] = {}

    # 1) Red hours by year (before vs after)
    yearly = (
        df.groupby("forecast_year", as_index=False)
        .agg(
            red_hours_baseline=("is_red_hour_baseline", "sum"),
            red_hours_mitigated=("is_red_hour_mitigated", "sum"),
            emissions_baseline=("pred_emissions", "sum"),
            emissions_mitigated=("mitigated_emissions", "sum"),
            peak_stress_baseline=("grid_stress_ratio", "max"),
            peak_stress_mitigated=("mitigated_grid_stress_ratio", "max"),
        )
        .sort_values("forecast_year")
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(yearly))
    w = 0.38
    ax.bar(x - w / 2, yearly["red_hours_baseline"], width=w, label="Baseline", color="#d95f02")
    ax.bar(x + w / 2, yearly["red_hours_mitigated"], width=w, label="Mitigated", color="#1b9e77")
    ax.set_xticks(x)
    ax.set_xticklabels(yearly["forecast_year"].astype(int))
    ax.set_xlabel("Forecast Year")
    ax.set_ylabel("Red Hours")
    ax.set_title("Danger Rule Red Hours: Baseline vs Mitigated")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    p = risk_dir / "danger_rule_red_hours_before_after.png"
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    plot_paths["red_hours_before_after"] = str(p)

    # 2) Emissions by year (before vs after)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(yearly["forecast_year"], yearly["emissions_baseline"], marker="o", linewidth=2, label="Baseline", color="#d95f02")
    ax.plot(
        yearly["forecast_year"],
        yearly["emissions_mitigated"],
        marker="o",
        linewidth=2,
        label="Mitigated",
        color="#1b9e77",
    )
    ax.set_xlabel("Forecast Year")
    ax.set_ylabel("Total Emissions (model units)")
    ax.set_title("Annual Emissions: Baseline vs Mitigated")
    ax.grid(alpha=0.25)
    ax.legend()
    p = risk_dir / "emissions_before_after_by_year.png"
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    plot_paths["emissions_before_after"] = str(p)

    # 3) Peak stress by year (before vs after)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        yearly["forecast_year"],
        yearly["peak_stress_baseline"],
        marker="o",
        linewidth=2,
        label="Baseline Peak Stress",
        color="#7570b3",
    )
    ax.plot(
        yearly["forecast_year"],
        yearly["peak_stress_mitigated"],
        marker="o",
        linewidth=2,
        label="Mitigated Peak Stress",
        color="#1b9e77",
    )
    ax.axhline(1.0, linestyle="--", color="#333333", linewidth=1.2, label="Stress Threshold = 1.0")
    ax.set_xlabel("Forecast Year")
    ax.set_ylabel("Peak Grid Stress Ratio")
    ax.set_title("Peak Grid Stress: Baseline vs Mitigated")
    ax.grid(alpha=0.25)
    ax.legend()
    p = risk_dir / "peak_stress_before_after_by_year.png"
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    plot_paths["peak_stress_before_after"] = str(p)

    # 4) Monthly joint-risk trend (before vs after)
    monthly = (
        df.assign(month=df["timestamp"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)
        .agg(
            joint_risk_baseline=("joint_risk_score_baseline", "mean"),
            joint_risk_mitigated=("joint_risk_score_mitigated", "mean"),
        )
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(monthly["month"], monthly["joint_risk_baseline"], linewidth=1.6, label="Baseline", color="#7570b3")
    ax.plot(monthly["month"], monthly["joint_risk_mitigated"], linewidth=1.6, label="Mitigated", color="#1b9e77")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean Joint Risk Score (0-100)")
    ax.set_title("Monthly Joint-Risk Score: Baseline vs Mitigated")
    ax.grid(alpha=0.25)
    ax.legend()
    p = risk_dir / "joint_risk_monthly_before_after.png"
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    plot_paths["joint_risk_monthly_before_after"] = str(p)

    # Copy strongest recommendation graphic into recommendations section.
    rec_plot = rec_dir / "risk_mitigation_red_hours_impact.png"
    data = plt.imread(risk_dir / "danger_rule_red_hours_before_after.png")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(data)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(rec_plot, dpi=180)
    plt.close(fig)
    plot_paths["recommendations_red_hours_impact"] = str(rec_plot)

    return plot_paths


def _write_playbook(
    path: Path,
    thresholds: Dict[str, float],
    summary: Dict[str, float],
) -> None:
    text = f"""# Joint-Risk Operations Playbook (MTFC)

## 1) Danger Rule
- High stress hour: `grid_stress_ratio > {thresholds['stress_threshold']:.2f}`
- High emissions hour: `pred_emissions >= {thresholds['emissions_threshold_p90']:.2f}` (top 10% baseline emissions threshold)
- Red hour definition: `high_stress OR high_emissions`

## 2) Joint-Risk Metric
- Percentile stress score: `StressPct_t = percentile(grid_stress_ratio_t)`
- Percentile emissions score: `EmissionsPct_t = percentile(pred_emissions_t)`
- Joint-risk score: `JointRisk_t = (StressPct_t/100) * (EmissionsPct_t/100) * 100`
- Interpretation:
  - `>= 75`: severe co-risk
  - `55-75`: elevated co-risk
  - `< 55`: manageable

## 3) Trigger Thresholds
- RED:
  - `grid_stress_ratio > {thresholds['stress_threshold']:.2f}` OR `pred_emissions >= P90` OR `JointRisk >= 75`
- AMBER:
  - `0.95 < grid_stress_ratio <= {thresholds['stress_threshold']:.2f}` OR `P75 <= pred_emissions < P90` OR `55 <= JointRisk < 75`
- GREEN:
  - All lower-risk conditions

## 4) Mitigation Actions
- RED actions (execute within the next dispatch interval):
  - Shift `8%` of flexible compute load out of red hours into low-risk hours.
  - Apply `4%` cooling/efficiency reduction to red-hour facility energy.
  - Pause non-critical batch jobs and defer retries where SLA allows.
- AMBER actions:
  - Pre-stage workload queue for potential RED escalation.
  - Shift up to `3%` voluntary load if queue depth is high.
- GREEN actions:
  - Recover deferred workload while respecting stress headroom.

## 5) Quantified Impact from This Scenario
- Baseline red hours: `{int(summary['baseline_red_hours'])}`
- Mitigated red hours: `{int(summary['mitigated_red_hours'])}`
- Red-hour reduction: `{summary['red_hour_reduction_abs']:.0f}` (`{summary['red_hour_reduction_pct']:.2f}%`)
- Peak stress reduction: `{summary['peak_stress_reduction_abs']:.6f}` (`{summary['peak_stress_reduction_pct']:.2f}%`)
- Emissions reduction: `{summary['emissions_reduction_abs']:.2f}` (`{summary['emissions_reduction_pct']:.2f}%`)

## 6) Monitoring Cadence
- Recompute danger rule + joint-risk score hourly from updated forecasts.
- Publish red/amber/green state to operators and mentors daily.
- Track KPI deltas weekly:
  - red hours
  - peak stress
  - total emissions
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    results_root = Path(__file__).resolve().parent / "outputs" / "results"

    risk_dir = repo_root / "MTFC_Sections" / "Risk_Analysis"
    rec_dir = repo_root / "MTFC_Sections" / "Recommendations"
    risk_dir.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)

    run_dir = _find_latest_forecast_dir(results_root)
    hourly_path = run_dir / "forecast_5y_hourly.csv"
    if not hourly_path.exists():
        raise FileNotFoundError(f"Missing forecast hourly file: {hourly_path}")

    df = pd.read_csv(hourly_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    stress_threshold = 1.0
    emissions_threshold_p90 = float(df["pred_emissions"].quantile(0.90))

    stress_pct = _percentile_from_reference(
        df["grid_stress_ratio"].to_numpy(dtype=float),
        df["grid_stress_ratio"].to_numpy(dtype=float),
    )
    emissions_pct = _percentile_from_reference(
        df["pred_emissions"].to_numpy(dtype=float),
        df["pred_emissions"].to_numpy(dtype=float),
    )

    df["stress_pct_baseline"] = stress_pct
    df["emissions_pct_baseline"] = emissions_pct
    df["joint_risk_score_baseline"] = (stress_pct / 100.0) * (emissions_pct / 100.0) * 100.0
    df["is_high_stress_baseline"] = df["grid_stress_ratio"] > stress_threshold
    df["is_high_emissions_baseline"] = df["pred_emissions"] >= emissions_threshold_p90
    df["is_red_hour_baseline"] = df["is_high_stress_baseline"] | df["is_high_emissions_baseline"]

    df = _simulate_mitigation(df, emissions_threshold=emissions_threshold_p90)

    stress_pct_after = _percentile_from_reference(
        df["mitigated_grid_stress_ratio"].to_numpy(dtype=float),
        df["grid_stress_ratio"].to_numpy(dtype=float),
    )
    emissions_pct_after = _percentile_from_reference(
        df["mitigated_emissions"].to_numpy(dtype=float),
        df["pred_emissions"].to_numpy(dtype=float),
    )
    df["stress_pct_mitigated"] = stress_pct_after
    df["emissions_pct_mitigated"] = emissions_pct_after
    df["joint_risk_score_mitigated"] = (stress_pct_after / 100.0) * (emissions_pct_after / 100.0) * 100.0
    df["is_high_stress_mitigated"] = df["mitigated_grid_stress_ratio"] > stress_threshold
    df["is_high_emissions_mitigated"] = df["mitigated_emissions"] >= emissions_threshold_p90
    df["is_red_hour_mitigated"] = df["is_high_stress_mitigated"] | df["is_high_emissions_mitigated"]

    baseline_red_hours = int(df["is_red_hour_baseline"].sum())
    mitigated_red_hours = int(df["is_red_hour_mitigated"].sum())
    red_hour_reduction_abs = baseline_red_hours - mitigated_red_hours
    red_hour_reduction_pct = 100.0 * red_hour_reduction_abs / max(baseline_red_hours, 1)

    baseline_peak_stress = float(np.nanmax(df["grid_stress_ratio"].to_numpy(dtype=float)))
    mitigated_peak_stress = float(np.nanmax(df["mitigated_grid_stress_ratio"].to_numpy(dtype=float)))
    peak_stress_reduction_abs = baseline_peak_stress - mitigated_peak_stress
    peak_stress_reduction_pct = 100.0 * peak_stress_reduction_abs / max(abs(baseline_peak_stress), 1e-9)

    baseline_emissions_total = float(np.nansum(df["pred_emissions"].to_numpy(dtype=float)))
    mitigated_emissions_total = float(np.nansum(df["mitigated_emissions"].to_numpy(dtype=float)))
    emissions_reduction_abs = baseline_emissions_total - mitigated_emissions_total
    emissions_reduction_pct = 100.0 * emissions_reduction_abs / max(abs(baseline_emissions_total), 1e-9)

    summary = {
        "run_dir": str(run_dir),
        "rows": int(len(df)),
        "baseline_red_hours": baseline_red_hours,
        "mitigated_red_hours": mitigated_red_hours,
        "red_hour_reduction_abs": float(red_hour_reduction_abs),
        "red_hour_reduction_pct": float(red_hour_reduction_pct),
        "baseline_peak_stress": baseline_peak_stress,
        "mitigated_peak_stress": mitigated_peak_stress,
        "peak_stress_reduction_abs": float(peak_stress_reduction_abs),
        "peak_stress_reduction_pct": float(peak_stress_reduction_pct),
        "baseline_total_emissions": baseline_emissions_total,
        "mitigated_total_emissions": mitigated_emissions_total,
        "emissions_reduction_abs": float(emissions_reduction_abs),
        "emissions_reduction_pct": float(emissions_reduction_pct),
    }

    thresholds = {
        "stress_threshold": stress_threshold,
        "emissions_threshold_p90": emissions_threshold_p90,
    }

    # Save detailed hourly comparison for traceability.
    hourly_cols = [
        "timestamp",
        "forecast_year",
        "pred_energy_mwh",
        "mitigated_energy_mwh",
        "pred_emissions",
        "mitigated_emissions",
        "grid_stress_ratio",
        "mitigated_grid_stress_ratio",
        "joint_risk_score_baseline",
        "joint_risk_score_mitigated",
        "is_red_hour_baseline",
        "is_red_hour_mitigated",
    ]
    hourly_out = risk_dir / "joint_risk_hourly_before_after.csv"
    df[hourly_cols].to_csv(hourly_out, index=False)

    summary_out = rec_dir / "mitigation_impact_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_out, index=False)

    thresholds_out = risk_dir / "danger_rule_thresholds.json"
    thresholds_out.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    plot_paths = _make_plots(df, risk_dir=risk_dir, rec_dir=rec_dir)

    playbook_out = rec_dir / "joint_risk_operations_playbook.md"
    _write_playbook(playbook_out, thresholds=thresholds, summary=summary)

    manifest = {
        "latest_run_dir": str(run_dir),
        "hourly_comparison_csv": str(hourly_out),
        "summary_csv": str(summary_out),
        "thresholds_json": str(thresholds_out),
        "playbook_md": str(playbook_out),
        "plots": plot_paths,
        "summary": summary,
    }
    manifest_out = rec_dir / "joint_risk_mitigation_manifest.json"
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Wrote:")
    print(hourly_out)
    print(summary_out)
    print(thresholds_out)
    print(playbook_out)
    print(manifest_out)
    for name, path in plot_paths.items():
        print(f"{name}: {path}")
    print("Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
