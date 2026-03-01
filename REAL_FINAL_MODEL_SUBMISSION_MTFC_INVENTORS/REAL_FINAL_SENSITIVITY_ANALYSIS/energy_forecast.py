"""
Energy Forecast & Cost Scenarios
=================================
Projects 10-year forward energy demand, emissions, and total costs
under three growth scenarios (Conservative / Moderate / Aggressive).

Outputs
-------
- energy_forecast_scenarios.csv
- energy_forecast_costs.csv
- figures/energy_forecast_scenarios.png
- figures/energy_forecast_cost_comparison.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import (DC_PARAMS, GRID, FINANCE, GROWTH_SCENARIOS,
                    OUTPUT_DIR, FIGURE_DIR, PLOT)


# ── Core projection logic ───────────────────────────────────────────────
def project_scenarios() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (scenarios_df, costs_df) with 10-year projections."""

    base_it_mw = DC_PARAMS["it_capacity_mw"]
    pue_base   = DC_PARAMS["pue_baseline"]
    ci_base    = GRID["carbon_intensity_mean"]
    scc        = FINANCE["scc_usd_per_ton"]
    energy_p   = FINANCE["energy_price_usd_per_mwh"]
    discount   = FINANCE["discount_rate"]
    horizon    = FINANCE["horizon_years"]

    scenario_rows = []
    cost_rows = []

    for name, scen in GROWTH_SCENARIOS.items():
        growth = scen["growth_rate"]
        pue_imp = scen["pue_improvement"]
        ci_imp  = scen["carbon_intensity_improvement"]

        cumul_emissions = 0.0
        cumul_cost = 0.0

        for yr in range(1, horizon + 1):
            it_mw  = base_it_mw * (1 + growth) ** yr
            pue_yr = max(pue_base * (1 - pue_imp) ** yr, 1.05)
            total_mw = it_mw * pue_yr
            annual_mwh = total_mw * 8760

            ci_yr = ci_base * (1 - ci_imp) ** yr
            annual_tons = total_mw * ci_yr * 8760 / 1000

            # Costs
            energy_cost   = annual_mwh * energy_p
            carbon_cost   = annual_tons * scc
            cooling_maint = FINANCE["cooling_maintenance_annual"]
            demand_charge = FINANCE["demand_charge_usd_per_kw_yr"] * total_mw * 1000
            total_annual  = energy_cost + carbon_cost + cooling_maint + demand_charge

            # Discount
            pv_factor = 1 / (1 + discount) ** yr
            pv_cost = total_annual * pv_factor

            cumul_emissions += annual_tons
            cumul_cost += pv_cost

            scenario_rows.append({
                "scenario":               name,
                "year":                   yr,
                "it_capacity_mw":         round(it_mw, 2),
                "pue":                    round(pue_yr, 4),
                "total_power_mw":         round(total_mw, 2),
                "annual_energy_mwh":      round(annual_mwh, 0),
                "carbon_intensity":       round(ci_yr, 2),
                "annual_emissions_tons":  round(annual_tons, 0),
                "cumulative_emissions":   round(cumul_emissions, 0),
            })

            cost_rows.append({
                "scenario":            name,
                "year":                yr,
                "energy_cost":         round(energy_cost, 0),
                "carbon_cost":         round(carbon_cost, 0),
                "demand_charge":       round(demand_charge, 0),
                "cooling_maintenance": round(cooling_maint, 0),
                "total_annual_cost":   round(total_annual, 0),
                "present_value_cost":  round(pv_cost, 0),
                "cumulative_pv_cost":  round(cumul_cost, 0),
            })

    return pd.DataFrame(scenario_rows), pd.DataFrame(cost_rows)


# ── Figures ──────────────────────────────────────────────────────────────
def plot_scenarios(scen_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=PLOT["figsize_wide"])

    metrics = [
        ("total_power_mw", "Total Facility Power (MW)"),
        ("annual_energy_mwh", "Annual Energy (MWh)"),
        ("annual_emissions_tons", "Annual Emissions (t CO₂)"),
        ("cumulative_emissions", "Cumulative Emissions (t CO₂)"),
    ]
    colors = {"Conservative": "#27ae60", "Moderate": "#f39c12", "Aggressive": "#c0392b"}

    for ax, (col, title) in zip(axes.ravel(), metrics):
        for name, grp in scen_df.groupby("scenario"):
            ax.plot(grp["year"], grp[col], marker="o", ms=5, lw=2,
                    color=colors.get(name, "gray"), label=name)
        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "energy_forecast_scenarios.png",
                dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close()


def plot_cost_comparison(cost_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=PLOT["figsize_wide"])
    colors = {"Conservative": "#27ae60", "Moderate": "#f39c12", "Aggressive": "#c0392b"}

    ax = axes[0]
    for name, grp in cost_df.groupby("scenario"):
        ax.plot(grp["year"], grp["total_annual_cost"]/1e6, marker="o", ms=5,
                lw=2, color=colors.get(name, "gray"), label=name)
    ax.set_xlabel("Year"); ax.set_ylabel("Annual Cost ($M)")
    ax.set_title("Total Annual Cost by Scenario", fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name, grp in cost_df.groupby("scenario"):
        ax.plot(grp["year"], grp["cumulative_pv_cost"]/1e6, marker="s", ms=5,
                lw=2, color=colors.get(name, "gray"), label=name)
    ax.set_xlabel("Year"); ax.set_ylabel("Cumulative PV Cost ($M)")
    ax.set_title("Cumulative Present-Value Cost", fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "energy_forecast_cost_comparison.png",
                dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close()


# ── Entry Point ──────────────────────────────────────────────────────────
def run():
    print("  ▶ Running energy forecast & cost scenarios …")
    scen_df, cost_df = project_scenarios()

    scen_df.to_csv(OUTPUT_DIR / "energy_forecast_scenarios.csv", index=False)
    cost_df.to_csv(OUTPUT_DIR / "energy_forecast_costs.csv", index=False)

    plot_scenarios(scen_df)
    plot_cost_comparison(cost_df)

    for name in GROWTH_SCENARIOS:
        sub = cost_df[cost_df["scenario"] == name]
        total_pv = sub["cumulative_pv_cost"].iloc[-1]
        print(f"    ✓ {name:15s}  10-yr PV cost = ${total_pv/1e6:,.1f}M")

    return scen_df, cost_df


if __name__ == "__main__":
    run()
