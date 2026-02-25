"""Matplotlib/seaborn graph formatter for MTFC analysis outputs.

Purpose:
- Regenerate all major analysis charts with consistent, smaller fonts.
- Keep model artifacts untouched (no retraining, no data synthesis).
- Use only existing run outputs and cleaned source data.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
ANALYSIS_ROOT = ROOT / "outputs" / "analysis"
RESULTS_ROOT = ROOT / "outputs" / "results"
DATA_ROOT = PROJECT_ROOT / "Data_Sources" / "cleaned"


@dataclass(frozen=True)
class PhysicsConfig:
    facility_mw: float = 100.0
    idle_power_fraction: float = 0.30
    cooling_threshold_f: float = 65.0
    base_pue: float = 1.10
    max_pue: float = 2.00
    pue_temp_coef: float = 0.012
    pue_cpu_coef: float = 0.050


def _latest_run_id() -> str:
    runs = sorted([p.name for p in RESULTS_ROOT.glob("carbon_*") if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No runs found under {RESULTS_ROOT}")
    return runs[-1]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path)


def _apply_plot_style() -> None:
    # Inspired by BAD FINAL MODEL NO USE/validation_dashboards.py but with smaller fonts.
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.title_fontsize": 7,
            "axes.titleweight": "bold",
            "grid.alpha": 0.25,
            "axes.grid": True,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _pretty_parameter(name: str) -> str:
    mapping = {
        "cpu_utilization": "CPU Utilization",
        "temperature_f": "Temperature (F)",
        "carbon_intensity": "Carbon Intensity",
        "idle_power_fraction": "Idle Power Fraction",
        "pue_temp_coef": "PUE Temp Coef",
        "pue_cpu_coef": "PUE CPU Coef",
    }
    return mapping.get(name, name.replace("_", " ").title())


def plot_carbon_heatmap(run_dir: Path) -> None:
    df = _read_csv(run_dir / "carbon_intensity_heatmap.csv")
    pivot = df.pivot(index="hour", columns="month", values="mean_carbon_intensity").sort_index()

    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    cmap = sns.color_palette("RdYlGn_r", as_cmap=True)
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        annot=True,
        fmt=".0f",
        annot_kws={"size": 6},
        cbar_kws={"label": "Avg Carbon Intensity (kg/MWh)"},
    )
    ax.set_title("Carbon Intensity by Hour and Month\n(When to Schedule Workloads for Lower Emissions)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Hour of Day")
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], rotation=0)
    _save(fig, run_dir / "carbon_intensity_heatmap.png")


def _plot_tornado_generic(df: pd.DataFrame, path: Path, value_col: str, baseline_col: str, title: str, xlabel: str, value_scale: float = 1.0) -> None:
    rows: List[Dict[str, object]] = []
    baseline = float(df[baseline_col].iloc[0]) * value_scale if not df.empty else 0.0
    for _, r in df.iterrows():
        p = _pretty_parameter(str(r["parameter"]))
        low = float(r[f"low_{value_col}"]) * value_scale
        high = float(r[f"high_{value_col}"]) * value_scale
        rows.append({"label": f"{p} (Low)", "value": low, "kind": "low"})
        rows.append({"label": f"{p} (High)", "value": high, "kind": "high"})
    rows.append({"label": "Baseline", "value": baseline, "kind": "base"})

    plot_df = pd.DataFrame(rows)
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11.8, 7.4))
    colors = {"low": "#38c172", "high": "#e74c3c", "base": "#2b8cbe"}
    y = np.arange(len(plot_df))

    left = np.minimum(plot_df["value"].to_numpy(), baseline)
    width = np.abs(plot_df["value"].to_numpy() - baseline)
    ax.barh(y, width, left=left, color=[colors[k] for k in plot_df["kind"]], edgecolor="black", linewidth=0.6)

    ax.axvline(baseline, color="#1f77b4", linestyle="--", linewidth=2.0, label="Baseline")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.set_title(title)
    ax.set_xlabel(xlabel)

    for yi, val in zip(y, plot_df["value"]):
        pct = ((val / baseline) - 1.0) * 100.0 if baseline else 0.0
        ax.text(val, yi, f" {pct:+.1f}%", va="center", ha="left", fontsize=7, color="#333")

    ax.legend(loc="lower right")
    _save(fig, path)


def plot_tornadoes(run_dir: Path) -> None:
    emis = _read_csv(run_dir / "tornado_oat.csv")
    enr = _read_csv(run_dir / "tornado_oat_energy.csv")

    _plot_tornado_generic(
        emis,
        run_dir / "tornado_oat_emissions.png",
        value_col="kg_per_h",
        baseline_col="baseline_kg_per_h",
        title="Sensitivity Analysis: Impact on Carbon Emissions\n(Deterministic OAT, Physics-Based)",
        xlabel="Emissions (tons CO2/hr)",
        value_scale=1.0 / 1000.0,
    )
    _plot_tornado_generic(
        enr,
        run_dir / "tornado_oat_energy.png",
        value_col="kg_per_h",
        baseline_col="baseline_kg_per_h",
        title="Sensitivity Analysis: Impact on Annual Energy\n(Deterministic OAT, Physics-Based)",
        xlabel="Annual Energy (GWh/year)",
        value_scale=1.0,
    )


def _plot_sobol_dual(df: pd.DataFrame, path: Path, subtitle: str) -> None:
    d = df.copy()
    d["label"] = d["parameter"].map(_pretty_parameter)
    d = d.sort_values("S1", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.3), sharey=True)

    axes[0].barh(d["label"], d["S1"], color="#4c84b1", edgecolor="black", linewidth=0.5)
    axes[0].set_title("Standalone Variable Effects\n(First-Order $S_i$)")
    axes[0].set_xlabel("Sobol Index")

    axes[1].barh(d["label"], d["ST"], color="#f67c4b", edgecolor="black", linewidth=0.5)
    axes[1].set_title("Total Variable Effects\n(Total-Order $S_{Ti}$)")
    axes[1].set_xlabel("Sobol Index")

    for ax, col in zip(axes, ["S1", "ST"]):
        for i, v in enumerate(d[col].to_numpy()):
            ax.text(v, i, f" {100*v:.1f}%", va="center", fontsize=7)

    fig.suptitle(f"Sobol Global Sensitivity Analysis\n{subtitle}", fontsize=11, y=1.02, fontweight="bold")
    _save(fig, path)


def plot_sobol(run_dir: Path) -> None:
    df_em = _read_csv(run_dir / "sobol_indices.csv")
    df_en = _read_csv(run_dir / "sobol_indices_energy.csv")
    _plot_sobol_dual(df_em, run_dir / "sobol_global_sensitivity.png", "Variance Decomposition of Carbon Liability")
    _plot_sobol_dual(df_en, run_dir / "sobol_energy_sensitivity.png", "Variance Decomposition of Annual Energy Forecast")


def _pseudo_obs(arr: np.ndarray) -> np.ndarray:
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1)
    return ranks / (len(arr) + 1.0)


def _load_merged_hourly() -> pd.DataFrame:
    cpu = _read_csv(DATA_ROOT / "google_cluster_utilization_2019_cellb_hourly_cleaned.csv")
    temp = _read_csv(DATA_ROOT / "ashburn_va_temperature_2019_cleaned.csv")
    ci = _read_csv(DATA_ROOT / "pjm_grid_carbon_intensity_2019_full_cleaned.csv")

    cpu["key"] = pd.to_datetime(cpu["real_timestamp"], utc=True).dt.floor("h")
    temp["key"] = pd.to_datetime(temp["timestamp"], utc=True).dt.floor("h")
    ci["key"] = pd.to_datetime(ci["timestamp"], utc=True).dt.floor("h")

    cpu_g = cpu.groupby("key", as_index=False).agg(cpu_utilization=("avg_cpu_utilization", "mean"))
    temp_g = temp.groupby("key", as_index=False).agg(temperature_c=("temperature_c", "mean"))
    ci_g = ci.groupby("key", as_index=False).agg(carbon_intensity=("carbon_intensity_kg_per_mwh", "mean"))

    merged = cpu_g.merge(temp_g, on="key", how="inner").merge(ci_g, on="key", how="inner").sort_values("key")
    merged["temperature_f"] = merged["temperature_c"] * 9.0 / 5.0 + 32.0

    cfg = PhysicsConfig()
    cpu_u = merged["cpu_utilization"].to_numpy(dtype=float)
    temp_f = merged["temperature_f"].to_numpy(dtype=float)
    ci_v = merged["carbon_intensity"].to_numpy(dtype=float)

    it_power = cfg.facility_mw * (cfg.idle_power_fraction + (1.0 - cfg.idle_power_fraction) * cpu_u)
    temp_above = np.maximum(0.0, temp_f - cfg.cooling_threshold_f)
    pue = np.clip(cfg.base_pue + cfg.pue_temp_coef * temp_above + cfg.pue_cpu_coef * cpu_u, cfg.base_pue, cfg.max_pue)
    total_power = it_power * pue
    emissions = total_power * ci_v
    annual_energy = total_power * 8760.0 / 1000.0

    merged["emissions"] = emissions
    merged["annual_energy"] = annual_energy
    return merged


def plot_copula_dashboard(run_dir: Path) -> None:
    curve = _read_csv(run_dir / "copula_tail_curves.csv")
    summary = _read_csv(run_dir / "copula_tail_dependence.csv")
    merged = _load_merged_hourly()

    u_temp = _pseudo_obs(merged["temperature_f"].to_numpy())
    u_cpu = _pseudo_obs(merged["cpu_utilization"].to_numpy())
    u_ci = _pseudo_obs(merged["carbon_intensity"].to_numpy())

    pair_plot = [
        ("temp_vs_carbon", "Temperature", "Carbon", u_temp, u_ci),
        ("temp_vs_cpu", "Temperature", "IT Load", u_temp, u_cpu),
        ("cpu_vs_carbon", "IT Load", "Carbon", u_cpu, u_ci),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.5))
    q = 0.95

    for i, (pair, xlabel, ylabel, ux, uy) in enumerate(pair_plot):
        ax = axes[0, i]
        extreme = (ux >= q) & (uy >= q)
        ax.scatter(ux[~extreme], uy[~extreme], s=5, alpha=0.35, color="#1f4aff", edgecolors="none")
        ax.scatter(ux[extreme], uy[extreme], s=14, alpha=0.8, color="#e31a1c", edgecolors="none")
        ax.axvline(q, color="gray", linestyle="--", linewidth=1.0)
        ax.axhline(q, color="gray", linestyle="--", linewidth=1.0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"{xlabel} (uniform)")
        ax.set_ylabel(f"{ylabel} (uniform)")
        ax.set_title(f"Copula Domain: {pair.replace('_', ' ').title()}")

        ax2 = axes[1, i]
        c = curve[curve["pair"] == pair].sort_values("threshold_q")
        ax2.plot(c["threshold_q"], c["lambda_upper"], "o-", color="#e31a1c", label="Upper tail $\\lambda_U$", markersize=3)
        ax2.plot(c["threshold_q"], c["lambda_lower"], "s-", color="#1f4aff", label="Lower tail $\\lambda_L$", markersize=3)

        srow = summary[summary["pair"] == pair]
        if not srow.empty:
            y95 = float(srow.iloc[0]["upper_tail_q95"])
            ax2.axhline(y95, color="#ff7f7f", linestyle="--", linewidth=1.2)
            ax2.text(c["threshold_q"].min(), y95 + 0.005, f"$\\lambda_U$(0.95)={y95:.3f}", fontsize=7)

        ax2.set_xlabel("Threshold Quantile")
        ax2.set_ylabel("Tail Dependence")
        ax2.set_title(f"Tail Dependence: {pair.replace('_', ' ').title()}")
        ax2.set_ylim(0, max(0.4, c[["lambda_upper", "lambda_lower"]].max().max() * 1.15))

    axes[1, 0].legend(loc="upper right")
    fig.suptitle(
        "Copula-Based Dependency Analysis (Sklar's Theorem)\nValidating Compound Tail Events in Datacenter Carbon Risk",
        fontsize=11,
        y=1.02,
        fontweight="bold",
    )
    _save(fig, run_dir / "copula_tail_dashboard.png")


def plot_copula_energy_bars(run_dir: Path) -> None:
    df = _read_csv(run_dir / "copula_tail_dependence.csv")
    d = df[df["pair"].str.endswith("_energy")].copy()
    if d.empty:
        return
    d = d.sort_values("upper_tail_q95", ascending=True)
    d["label"] = d["pair"].str.replace("_", " ")

    fig, ax = plt.subplots(figsize=(9.6, 4.9))
    y = np.arange(len(d))
    ax.barh(y - 0.18, d["upper_tail_q95"], height=0.34, color="#d62728", label="Upper tail $\\lambda_U$")
    ax.barh(y + 0.18, d["lower_tail_q05"], height=0.34, color="#1f77b4", label="Lower tail $\\lambda_L$")
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"])
    ax.set_xlabel("Tail Dependence Coefficient")
    ax.set_title("Energy Tail Dependence (Copula)")
    ax.legend(loc="lower right")
    _save(fig, run_dir / "copula_energy_tail_bars.png")


def plot_radar(run_dir: Path) -> None:
    df = _read_csv(run_dir / "recommendation_scenarios.csv")
    metrics = [
        ("total_power_x_baseline", "Total Power"),
        ("emissions_x_baseline", "Carbon Emissions"),
        ("peak_power_x_baseline", "Peak Demand"),
        ("cooling_x_baseline", "Cooling Load"),
    ]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    colors = {
        "Current (Baseline)": "#2c7fb8",
        "Efficient (Lower PUE + CI)": "#2dbf6f",
        "High Growth": "#de4b39",
        "Climate Stress": "#8c55b5",
    }

    fig = plt.figure(figsize=(7.3, 6.2))
    ax = plt.subplot(111, polar=True)

    for _, row in df.iterrows():
        vals = [float(row[m]) for m, _ in metrics]
        vals += vals[:1]
        name = str(row["scenario"])
        color = colors.get(name, "#555")
        ax.plot(angles, vals, color=color, linewidth=1.8, marker="o", markersize=3.5, label=name)
        ax.fill(angles, vals, color=color, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylim(0, 2.5)
    ax.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_yticklabels(["0.5x", "1.0x", "1.5x", "2.0x", "2.5x"])
    ax.set_title("Scenario Comparison (2030 Projection)\nNormalized to Current Baseline", va="bottom")
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05), frameon=False)
    _save(fig, run_dir / "recommendation_scenario_radar.png")


def plot_energy_peak_projection(run_dir: Path) -> None:
    df = _read_csv(run_dir / "energy_forecast_scenarios.csv")
    df = df.sort_values(["forecast_scenario", "year"])

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4))

    moderate = df[df["forecast_scenario"] == "Moderate 15%"]
    if moderate.empty:
        moderate = df.groupby("year", as_index=False).first()
    axes[0].bar(moderate["year"], moderate["forecast_annual_energy_gwh"], color="#2c7fb8", edgecolor="black", linewidth=0.5)
    axes[0].set_title("Projected Annual Energy Consumption")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Energy (GWh)")

    colors = {"Conservative 5%": "#2c7fb8", "Moderate 15%": "#f47a1f", "Aggressive 30%": "#2e9f44"}
    for scenario, d in df.groupby("forecast_scenario"):
        axes[1].plot(d["year"], d["forecast_peak_mw"], marker="o", linewidth=1.8, markersize=3.5, label=scenario, color=colors.get(scenario, None))
    axes[1].set_title("Projected Peak Demand")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Peak Power (MW)")
    axes[1].legend(frameon=False)

    _save(fig, run_dir / "recommendation_energy_peak_projection.png")


def plot_mitigation_impact(run_dir: Path) -> None:
    df = _read_csv(run_dir / "recommendation_mitigation.csv").sort_values("annual_reduction_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(9.3, 4.8))
    colors = ["#2fbf71" if x < 25 else "#e34a33" for x in df["annual_reduction_pct"]]
    ax.barh(df["lever"], df["annual_reduction_pct"], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Estimated Annual Carbon Reduction (%)")
    ax.set_title("Recommendation Impact Ranking")
    for y, val, diff in zip(df["lever"], df["annual_reduction_pct"], df["difficulty_score"]):
        ax.text(val, y, f" {val:.1f}%  (Difficulty {diff:.1f}/5)", va="center", fontsize=7)
    _save(fig, run_dir / "recommendation_mitigation_impact.png")


def plot_cost_stack(run_dir: Path) -> None:
    df = _read_csv(run_dir / "scenario_monetization.csv")
    comps = [
        ("carbon_liability_usd", "Carbon", "#e15759"),
        ("electricity_cost_usd", "Electricity", "#4e79a7"),
        ("demand_charge_cost_usd", "Demand", "#f28e2b"),
        ("risk_premium_total_usd", "Risk Premium", "#b07aa1"),
    ]

    fig, ax = plt.subplots(figsize=(10.8, 5.3))
    x = np.arange(len(df))
    bottom = np.zeros(len(df))
    for col, lab, color in comps:
        vals = df[col].to_numpy(dtype=float) / 1e6
        ax.bar(x, vals, bottom=bottom, label=lab, color=color, edgecolor="black", linewidth=0.4)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(df["scenario"], rotation=12, ha="right")
    ax.set_ylabel("Annual Cost (Million USD)")
    ax.set_title("Annual Cost Stack by Scenario")
    ax.legend(ncol=2, frameon=False)
    _save(fig, run_dir / "scenario_cost_stack.png")


def plot_carbon_price_band(run_dir: Path) -> None:
    df = _read_csv(run_dir / "scenario_monetization.csv")
    y = np.arange(len(df))
    low = df["total_annual_cost_low_carbon_usd"].to_numpy(dtype=float) / 1e6
    mid = df["total_annual_cost_usd"].to_numpy(dtype=float) / 1e6
    high = df["total_annual_cost_high_carbon_usd"].to_numpy(dtype=float) / 1e6

    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    ax.hlines(y, low, high, color="#4e79a7", linewidth=2.2)
    ax.scatter(mid, y, color="#d62728", s=28, zorder=3, label="Base carbon price")
    ax.scatter(low, y, color="#2ca02c", s=16, zorder=3, label="Low carbon price")
    ax.scatter(high, y, color="#ff7f0e", s=16, zorder=3, label="High carbon price")
    ax.set_yticks(y)
    ax.set_yticklabels(df["scenario"])
    ax.set_xlabel("Total Annual Cost (Million USD)")
    ax.set_title("Annual Cost Exposure Under Carbon Price Bands")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, run_dir / "scenario_carbon_price_band.png")


def plot_energy_forecast_costs(run_dir: Path) -> None:
    df = _read_csv(run_dir / "energy_forecast_costs.csv").sort_values(["forecast_scenario", "year"])
    fig, ax = plt.subplots(figsize=(10.4, 4.9))
    colors = {"Conservative 5%": "#2c7fb8", "Moderate 15%": "#f47a1f", "Aggressive 30%": "#2e9f44"}
    for scenario, d in df.groupby("forecast_scenario"):
        ax.plot(d["year"], d["total_annual_cost_usd"] / 1e6, marker="o", linewidth=1.9, markersize=3.5, label=scenario, color=colors.get(scenario, None))
    ax.set_title("Projected Annual Cost from Energy Forecast")
    ax.set_xlabel("Forecast Year")
    ax.set_ylabel("Total Annual Cost (Million USD)")
    ax.legend(frameon=False)
    _save(fig, run_dir / "energy_forecast_costs.png")


def plot_mitigation_npv(run_dir: Path) -> None:
    df = _read_csv(run_dir / "mitigation_cost_benefit.csv").sort_values("npv_10y_usd", ascending=True)

    fig, ax1 = plt.subplots(figsize=(10.8, 5.3))
    y = np.arange(len(df))
    npv = df["npv_10y_usd"].to_numpy(dtype=float) / 1e6
    payback = df["payback_years"].to_numpy(dtype=float)

    bars = ax1.barh(y, npv, color=["#2ca02c" if v >= 0 else "#d62728" for v in npv], edgecolor="black", linewidth=0.5)
    ax1.axvline(0, color="#1f77b4", linestyle="--", linewidth=1.4)
    ax1.set_yticks(y)
    ax1.set_yticklabels(df["lever"])
    ax1.set_xlabel("10-Year NPV (Million USD)")
    ax1.set_title("Mitigation Economics: NPV with Payback Overlay")

    ax2 = ax1.twiny()
    ax2.plot(payback, y, "o", color="#ff7f0e", markersize=4, label="Payback")
    ax2.set_xlabel("Payback (Years)")

    for rect, val in zip(bars, npv):
        ax1.text(val, rect.get_y() + rect.get_height() / 2.0, f" {val:.2f}M", va="center", fontsize=7)

    _save(fig, run_dir / "mitigation_npv_payback.png")


def plot_monetization_dashboard(run_dir: Path) -> None:
    scen = _read_csv(run_dir / "scenario_monetization.csv")
    miti = _read_csv(run_dir / "mitigation_cost_benefit.csv")

    baseline = scen[scen["scenario"] == "Current (Baseline)"]
    efficient = scen[scen["scenario"] == "Efficient (Lower PUE + CI)"]
    top_mit = miti.sort_values("npv_10y_usd", ascending=False).head(1)

    fig = plt.figure(figsize=(11.8, 6.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.4])

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])

    ax0.axis("off")
    if not baseline.empty:
        b = baseline.iloc[0]
        ax0.text(
            0.0,
            1.0,
            "Baseline Annual Cost\n"
            f"Total: ${b['total_annual_cost_usd']:,.0f}\n"
            f"Carbon: ${b['carbon_liability_usd']:,.0f}\n"
            f"Energy: ${b['electricity_cost_usd']:,.0f}\n"
            f"Risk Premium: ${b['risk_premium_total_usd']:,.0f}",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "#f7f7f7", "edgecolor": "#b0b0b0"},
        )

    ax1.axis("off")
    if not efficient.empty and not top_mit.empty:
        e = efficient.iloc[0]
        t = top_mit.iloc[0]
        ax1.text(
            0.0,
            1.0,
            "Strategic Delta\n"
            f"Efficient vs Baseline: ${e['delta_vs_baseline_total_cost_usd']:,.0f}\n"
            f"Top Mitigation: {t['lever']}\n"
            f"10Y NPV: ${t['npv_10y_usd']:,.0f}\n"
            f"Payback: {t['payback_years']:.2f} years",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "#f7f7f7", "edgecolor": "#b0b0b0"},
        )

    m = miti.sort_values("annual_net_benefit_usd", ascending=True)
    ax2.barh(m["lever"], m["annual_net_benefit_usd"] / 1e6, color="#55a868", edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Annual Net Benefit (Million USD/year)")
    ax2.set_title("Mitigation Annual Net Benefit Ranking")

    fig.suptitle("Monetizable Outcomes Dashboard", fontsize=11, fontweight="bold")
    _save(fig, run_dir / "monetizable_outcomes_dashboard.png")


def plot_model_performance(run_id: str, run_dir: Path) -> None:
    res_dir = RESULTS_ROOT / run_id
    metrics = _read_csv(res_dir / "metrics_summary.csv")
    s1 = _read_csv(res_dir / "stage1_cpu_holdout_predictions.csv")
    s5 = _read_csv(res_dir / "stage5_carbon_holdout_predictions.csv")
    s6 = _read_csv(res_dir / "stage6_emissions_holdout_predictions.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))

    # R2 bars
    core = metrics[metrics["stage"].isin(["Stage1_CPU", "Stage5_CarbonIntensity", "Stage6_Emissions"])].copy()
    axes[0, 0].bar(core["stage"], core["r2"], color=["#4e79a7", "#f28e2b", "#59a14f"], edgecolor="black", linewidth=0.5)
    axes[0, 0].axhline(0.9, color="#d62728", linestyle="--", linewidth=1.2, label="Target R2=0.9")
    axes[0, 0].set_ylim(0, 1.02)
    axes[0, 0].set_title("Holdout R2 by Stage")
    axes[0, 0].tick_params(axis="x", rotation=15)
    axes[0, 0].legend(frameon=False)

    # Actual vs pred (Stage6)
    axes[0, 1].scatter(s6["actual_emissions"], s6["pred_emissions"], s=10, alpha=0.5, color="#2c7fb8", edgecolors="none")
    mn = min(s6["actual_emissions"].min(), s6["pred_emissions"].min())
    mx = max(s6["actual_emissions"].max(), s6["pred_emissions"].max())
    axes[0, 1].plot([mn, mx], [mn, mx], "--", color="#d62728", linewidth=1.2)
    axes[0, 1].set_title("Stage6 Actual vs Predicted")
    axes[0, 1].set_xlabel("Actual")
    axes[0, 1].set_ylabel("Predicted")

    # Residual distributions
    r1 = s1["actual_cpu"] - s1["pred_cpu"]
    r5 = s5["actual_ci"] - s5["pred_ci"]
    r6 = s6["actual_emissions"] - s6["pred_emissions"]
    axes[1, 0].hist(r1, bins=22, alpha=0.6, label="Stage1", color="#4e79a7")
    axes[1, 0].hist(r5, bins=22, alpha=0.6, label="Stage5", color="#f28e2b")
    axes[1, 0].hist(r6, bins=22, alpha=0.6, label="Stage6", color="#59a14f")
    axes[1, 0].set_title("Residual Distributions")
    axes[1, 0].set_xlabel("Actual - Predicted")
    axes[1, 0].legend(frameon=False)

    # Residual time-series (Stage6)
    ts = pd.to_datetime(s6["timestamp"])
    axes[1, 1].plot(ts, r6, color="#1f77b4", linewidth=1.0)
    axes[1, 1].axhline(0, color="#d62728", linestyle="--", linewidth=1.0)
    axes[1, 1].set_title("Stage6 Residuals Over Holdout Time")
    axes[1, 1].set_xlabel("Timestamp")
    axes[1, 1].set_ylabel("Residual")
    axes[1, 1].tick_params(axis="x", rotation=30)

    fig.suptitle("Model Performance Dashboard", fontsize=11, fontweight="bold")
    _save(fig, run_dir / "model_performance_dashboard.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render matplotlib/seaborn charts with small-font formatting.")
    parser.add_argument("--run-id", default=None, help="Run ID (default: latest)")
    args = parser.parse_args()

    run_id = args.run_id or _latest_run_id()
    run_dir = ANALYSIS_ROOT / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Analysis directory not found: {run_dir}")

    _apply_plot_style()

    plot_carbon_heatmap(run_dir)
    plot_tornadoes(run_dir)
    plot_sobol(run_dir)
    plot_copula_dashboard(run_dir)
    plot_copula_energy_bars(run_dir)
    plot_radar(run_dir)
    plot_energy_peak_projection(run_dir)
    plot_mitigation_impact(run_dir)

    plot_cost_stack(run_dir)
    plot_carbon_price_band(run_dir)
    plot_energy_forecast_costs(run_dir)
    plot_mitigation_npv(run_dir)
    plot_monetization_dashboard(run_dir)
    plot_model_performance(run_id, run_dir)

    generated = sorted([p.name for p in run_dir.glob("*.png")])
    manifest = {
        "run_id": run_id,
        "renderer": "matplotlib_seaborn",
        "small_font_style": True,
        "generated_png_files": generated,
    }
    (run_dir / "graph_format_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
