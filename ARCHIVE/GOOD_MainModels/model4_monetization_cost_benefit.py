"""
Model 4 — Monetization and Cost-Benefit Analysis
================================================
Converts forecast outputs from Models 1-3 into dollar impacts and
recommendation economics.

Inputs:
    GOOD_MainModels/energy_usage_forecast.csv
    GOOD_MainModels/carbon_emissions_forecast.csv
    GOOD_MainModels/grid_stress_annual_summary.csv

Outputs:
    GOOD_MainModels/annual_risk_monetization.csv
    GOOD_MainModels/recommendation_cost_benefit.csv
    GOOD_MainModels/model4_monetization_summary.txt
    GOOD_Figures/model4_annual_risk_costs.png
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FinancialAssumptions


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "GOOD_MainModels"
FIG_DIR = ROOT / "GOOD_Figures"

ENERGY_FORECAST_PATH = OUT_DIR / "energy_usage_forecast.csv"
EMISSIONS_FORECAST_PATH = OUT_DIR / "carbon_emissions_forecast.csv"
GRID_STRESS_PATH = OUT_DIR / "grid_stress_annual_summary.csv"

OUT_ANNUAL_RISK = OUT_DIR / "annual_risk_monetization.csv"
OUT_RECS = OUT_DIR / "recommendation_cost_benefit.csv"
OUT_SUMMARY = OUT_DIR / "model4_monetization_summary.txt"
OUT_FIG = FIG_DIR / "model4_annual_risk_costs.png"


@dataclass(frozen=True)
class MitigationLever:
    name: str
    capex_usd: float
    energy_cost_reduction_pct: float
    carbon_liability_reduction_pct: float
    grid_stress_reduction_pct: float


MITIGATION_LEVERS = [
    MitigationLever(
        name="Cleaner grid contracts",
        capex_usd=18_000_000.0,
        energy_cost_reduction_pct=0.02,
        carbon_liability_reduction_pct=0.35,
        grid_stress_reduction_pct=0.05,
    ),
    MitigationLever(
        name="PUE optimization",
        capex_usd=12_000_000.0,
        energy_cost_reduction_pct=0.10,
        carbon_liability_reduction_pct=0.10,
        grid_stress_reduction_pct=0.08,
    ),
    MitigationLever(
        name="Workload shifting",
        capex_usd=7_500_000.0,
        energy_cost_reduction_pct=0.03,
        carbon_liability_reduction_pct=0.07,
        grid_stress_reduction_pct=0.25,
    ),
]


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")


def _annual_energy() -> pd.DataFrame:
    _require_file(ENERGY_FORECAST_PATH)
    e = pd.read_csv(ENERGY_FORECAST_PATH, parse_dates=["date"])
    e["year"] = pd.to_datetime(e["date"], errors="coerce").dt.year
    annual = e.groupby("year", as_index=False)["forecast_GWh"].sum().rename(columns={"forecast_GWh": "energy_gwh"})
    return annual


def _annual_emissions() -> pd.DataFrame:
    _require_file(EMISSIONS_FORECAST_PATH)
    c = pd.read_csv(EMISSIONS_FORECAST_PATH, parse_dates=["date"])
    c["year"] = pd.to_datetime(c["date"], errors="coerce").dt.year
    if "emissions_tonnes_monthly" in c.columns:
        annual = c.groupby("year", as_index=False)["emissions_tonnes_monthly"].sum().rename(
            columns={"emissions_tonnes_monthly": "emissions_tonnes"}
        )
        return annual
    if "emissions_tonnes_annual" in c.columns:
        annual = c.groupby("year", as_index=False)["emissions_tonnes_annual"].mean().rename(
            columns={"emissions_tonnes_annual": "emissions_tonnes"}
        )
        return annual
    raise ValueError("Carbon emissions forecast missing emissions_tonnes_monthly/emissions_tonnes_annual columns.")


def _annual_grid_stress() -> pd.DataFrame:
    _require_file(GRID_STRESS_PATH)
    g = pd.read_csv(GRID_STRESS_PATH)
    if "scenario" not in g.columns or "grid_stress_cost_usd" not in g.columns:
        raise ValueError("Grid stress summary is missing scenario/grid_stress_cost_usd columns.")
    base = g[g["scenario"] == "baseline"][["year", "grid_stress_cost_usd"]].copy()
    base = base.rename(columns={"grid_stress_cost_usd": "grid_stress_cost_usd_baseline"})
    return base


def _npv(cashflows: list[float], discount_rate: float) -> float:
    return float(sum(cf / ((1.0 + discount_rate) ** i) for i, cf in enumerate(cashflows, start=1)))


def _build_annual_risk_table(assumptions: FinancialAssumptions) -> pd.DataFrame:
    energy = _annual_energy()
    emissions = _annual_emissions()
    stress = _annual_grid_stress()

    annual = energy.merge(emissions, on="year", how="inner").merge(stress, on="year", how="left")
    annual["grid_stress_cost_usd_baseline"] = annual["grid_stress_cost_usd_baseline"].fillna(0.0)

    annual["energy_cost_usd"] = annual["energy_gwh"] * 1000.0 * assumptions.energy_price_usd_per_mwh
    annual["carbon_liability_usd"] = annual["emissions_tonnes"] * assumptions.scc_usd_per_ton
    annual["total_risk_cost_usd"] = (
        annual["energy_cost_usd"] + annual["carbon_liability_usd"] + annual["grid_stress_cost_usd_baseline"]
    )

    return annual.sort_values("year").reset_index(drop=True)


def _recommendation_table(
    annual_risk: pd.DataFrame,
    assumptions: FinancialAssumptions,
) -> pd.DataFrame:
    horizon = int(assumptions.analysis_horizon_years)
    risk_slice = annual_risk.head(horizon).copy()
    if risk_slice.empty:
        raise ValueError("No annual risk rows are available for recommendation analysis.")

    rows: list[dict[str, float | str]] = []
    for lever in MITIGATION_LEVERS:
        savings_energy = risk_slice["energy_cost_usd"] * lever.energy_cost_reduction_pct
        savings_carbon = risk_slice["carbon_liability_usd"] * lever.carbon_liability_reduction_pct
        savings_stress = risk_slice["grid_stress_cost_usd_baseline"] * lever.grid_stress_reduction_pct
        annual_savings_series = (savings_energy + savings_carbon + savings_stress).astype(float)

        avg_annual_savings = float(annual_savings_series.mean())
        total_undiscounted = float(annual_savings_series.sum())
        payback_years = math.inf if avg_annual_savings <= 0 else float(lever.capex_usd / avg_annual_savings)
        npv_10y = float(-lever.capex_usd + _npv(annual_savings_series.tolist(), assumptions.discount_rate))

        rows.append(
            {
                "lever": lever.name,
                "capex_usd": lever.capex_usd,
                "avg_annual_cost_reduction_usd": avg_annual_savings,
                "total_cost_reduction_10y_usd": total_undiscounted,
                "payback_years": payback_years,
                "npv_10y_usd": npv_10y,
                "energy_cost_reduction_pct": lever.energy_cost_reduction_pct * 100.0,
                "carbon_liability_reduction_pct": lever.carbon_liability_reduction_pct * 100.0,
                "grid_stress_reduction_pct": lever.grid_stress_reduction_pct * 100.0,
            }
        )

    return pd.DataFrame(rows).sort_values("npv_10y_usd", ascending=False).reset_index(drop=True)


def _plot_annual_risk(annual: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    years = annual["year"].astype(int).tolist()
    x = np.arange(len(years))

    energy_m = annual["energy_cost_usd"].to_numpy(dtype=float) / 1e6
    carbon_m = annual["carbon_liability_usd"].to_numpy(dtype=float) / 1e6
    stress_m = annual["grid_stress_cost_usd_baseline"].to_numpy(dtype=float) / 1e6

    ax.bar(x, energy_m, label="Energy cost", color="#4c78a8")
    ax.bar(x, carbon_m, bottom=energy_m, label="Carbon liability", color="#f58518")
    ax.bar(x, stress_m, bottom=energy_m + carbon_m, label="Grid stress cost", color="#e45756")

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], rotation=45, ha="right")
    ax.set_ylabel("Annual risk cost ($M)")
    ax.set_xlabel("Year")
    ax.set_title("Model 4: Annual Monetized Risk Stack", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("MODEL 4 — Monetization and Cost-Benefit Analysis")
    print("=" * 72)

    assumptions = FinancialAssumptions()
    annual_risk = _build_annual_risk_table(assumptions=assumptions)
    annual_risk.to_csv(OUT_ANNUAL_RISK, index=False)

    recs = _recommendation_table(annual_risk=annual_risk, assumptions=assumptions)
    recs.to_csv(OUT_RECS, index=False)
    _plot_annual_risk(annual_risk)

    total_10y = float(annual_risk.head(assumptions.analysis_horizon_years)["total_risk_cost_usd"].sum())
    summary_lines = [
        "MODEL 4 — Monetization and Cost-Benefit Analysis",
        f"SCC used: ${assumptions.scc_usd_per_ton:,.0f}/tonne",
        f"Electricity price used: ${assumptions.energy_price_usd_per_mwh:,.0f}/MWh",
        f"Discount rate used: {assumptions.discount_rate:.1%}",
        f"Analysis horizon: {assumptions.analysis_horizon_years} years",
        f"10-year total baseline risk cost: ${total_10y:,.0f}",
        "",
        "Top recommendation by NPV:",
        f"- {recs.iloc[0]['lever']}: NPV=${recs.iloc[0]['npv_10y_usd']:,.0f}, payback={recs.iloc[0]['payback_years']:.2f} years",
    ]
    OUT_SUMMARY.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Saved: {OUT_ANNUAL_RISK}")
    print(f"Saved: {OUT_RECS}")
    print(f"Saved: {OUT_SUMMARY}")
    print(f"Saved: {OUT_FIG}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI entrypoint guard
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
