"""
Recommendation & Mitigation Strategy Tables
=============================================
Builds decision-grade summary tables that an actuarial panel
can use directly: scenario comparison and mitigation-lever
recommendation rankings.

Outputs
-------
- recommendation_scenarios.csv
- recommendation_mitigation.csv
- figures/recommendation_radar.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import (DC_PARAMS, GRID, FINANCE, GROWTH_SCENARIOS,
                    MITIGATION_LEVERS, OUTPUT_DIR, FIGURE_DIR, PLOT)


# ── Scenario recommendation table ───────────────────────────────────────
def build_scenario_recommendations() -> pd.DataFrame:
    rows = []
    base_mw = DC_PARAMS["it_capacity_mw"] * DC_PARAMS["pue_baseline"]
    base_em = base_mw * GRID["carbon_intensity_mean"] * 8760 / 1000

    for name, scen in GROWTH_SCENARIOS.items():
        g = scen["growth_rate"]
        pue_i = scen["pue_improvement"]
        ci_i  = scen["carbon_intensity_improvement"]

        # Year 10 snapshot
        yr10_mw = DC_PARAMS["it_capacity_mw"] * (1+g)**10
        yr10_pue = max(DC_PARAMS["pue_baseline"] * (1-pue_i)**10, 1.05)
        yr10_total = yr10_mw * yr10_pue
        yr10_ci = GRID["carbon_intensity_mean"] * (1-ci_i)**10
        yr10_em = yr10_total * yr10_ci * 8760 / 1000
        yr10_cost = (yr10_total * 8760 * FINANCE["energy_price_usd_per_mwh"]
                     + yr10_em * FINANCE["scc_usd_per_ton"])

        # Cumulative 10-yr
        cumul_em = sum(
            DC_PARAMS["it_capacity_mw"] * (1+g)**y
            * max(DC_PARAMS["pue_baseline"]*(1-pue_i)**y, 1.05)
            * GRID["carbon_intensity_mean"] * (1-ci_i)**y
            * 8760 / 1000
            for y in range(1, 11))

        risk_rating = "Low" if g <= 0.05 else ("Medium" if g <= 0.15 else "High")
        action = ("Maintain current ops" if g <= 0.05
                  else ("Proactive mitigation needed" if g <= 0.15
                        else "Urgent intervention required"))

        rows.append({
            "scenario": name,
            "growth_rate_pct": g * 100,
            "yr10_power_mw": round(yr10_total, 1),
            "yr10_emissions_tons": round(yr10_em, 0),
            "yr10_annual_cost": round(yr10_cost, 0),
            "cumulative_10yr_emissions": round(cumul_em, 0),
            "emission_change_vs_baseline": round((yr10_em / base_em - 1) * 100, 1),
            "risk_rating": risk_rating,
            "recommended_action": action,
        })
    return pd.DataFrame(rows)


# ── Mitigation recommendation table ─────────────────────────────────────
def build_mitigation_recommendations() -> pd.DataFrame:
    base_mw = DC_PARAMS["it_capacity_mw"] * DC_PARAMS["pue_baseline"]
    base_energy_cost = base_mw * 8760 * FINANCE["energy_price_usd_per_mwh"]
    base_carbon_cost = (base_mw * GRID["carbon_intensity_mean"] * 8760 / 1000
                        * FINANCE["scc_usd_per_ton"])

    rows = []
    for name, lev in MITIGATION_LEVERS.items():
        co2_red = lev["emission_reduction_pct"] / 100
        nrg_red = lev.get("energy_saving_pct", 0.0) / 100
        capex = lev["capex_usd"]
        # Emission reduction avoids carbon cost; energy saving avoids energy cost
        annual_sav = base_carbon_cost * co2_red + base_energy_cost * nrg_red
        npv = sum(annual_sav / (1 + FINANCE["discount_rate"])**y
                  for y in range(1, FINANCE["horizon_years"]+1)) - capex
        roi = npv / capex if capex > 0 else 0
        payback = capex / annual_sav if annual_sav > 0 else 0

        # Feasibility score (1-5) — heuristic
        if payback < 1:
            feasibility = 5
        elif payback < 3:
            feasibility = 4
        elif payback < 5:
            feasibility = 3
        else:
            feasibility = 2

        # Impact score (1-5)
        if co2_red >= 0.35:
            impact = 5
        elif co2_red >= 0.20:
            impact = 4
        elif co2_red >= 0.10:
            impact = 3
        else:
            impact = 2

        priority = "★★★" if (feasibility + impact) >= 8 else ("★★" if (feasibility + impact) >= 6 else "★")

        rows.append({
            "lever": name,
            "emission_reduction_pct": lev["emission_reduction_pct"],
            "capex_usd": capex,
            "annual_savings_usd": round(annual_sav, 0),
            "npv_10yr_usd": round(npv, 0),
            "roi_x": round(roi, 1),
            "payback_years": round(payback, 2),
            "feasibility_score": feasibility,
            "impact_score": impact,
            "priority": priority,
        })

    return pd.DataFrame(rows).sort_values("npv_10yr_usd", ascending=False)


# ── Figure: Radar chart ─────────────────────────────────────────────────
def plot_radar(mit_df: pd.DataFrame):
    categories = ["Emission\nReduction", "ROI", "Feasibility", "Impact", "Speed\n(1/Payback)"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ["#27ae60", "#2980b9", "#f39c12", "#8e44ad"]

    for i, (_, row) in enumerate(mit_df.iterrows()):
        # Normalise each dimension to 0-1
        vals = [
            row["emission_reduction_pct"] / 50,      # max ~42%
            min(row["roi_x"] / 120, 1.0),             # max ~112×
            row["feasibility_score"] / 5,
            row["impact_score"] / 5,
            min(1 / max(row["payback_years"], 0.1) / 2, 1.0),  # faster = higher
        ]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", lw=2, color=colors[i % len(colors)],
                label=row["lever"])
        ax.fill(angles, vals, alpha=0.1, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Mitigation Lever Comparison", fontsize=13,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "recommendation_radar.png",
                dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close()


# ── Entry Point ──────────────────────────────────────────────────────────
def run():
    print("  ▶ Building recommendation tables …")
    scen_df = build_scenario_recommendations()
    mit_df  = build_mitigation_recommendations()

    scen_df.to_csv(OUTPUT_DIR / "recommendation_scenarios.csv", index=False)
    mit_df.to_csv(OUTPUT_DIR / "recommendation_mitigation.csv",  index=False)

    plot_radar(mit_df)

    print("    ✓ Scenario recommendations:")
    for _, r in scen_df.iterrows():
        print(f"      {r['scenario']:15s}  risk={r['risk_rating']:6s}  → {r['recommended_action']}")
    print("    ✓ Top mitigation lever:")
    top = mit_df.iloc[0]
    print(f"      {top['lever']}  NPV=${top['npv_10yr_usd']/1e6:,.1f}M  ROI={top['roi_x']:.1f}×  Priority={top['priority']}")
    return scen_df, mit_df


if __name__ == "__main__":
    run()
