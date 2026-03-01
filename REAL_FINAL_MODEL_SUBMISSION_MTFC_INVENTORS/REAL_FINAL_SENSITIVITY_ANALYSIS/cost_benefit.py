"""
Cost-Benefit Analysis & Financial Scenario Monetisation
========================================================
Computes:
  • Scenario monetisation (Baseline / Efficient / High-Growth / Climate-Stress)
  • Risk-premium breakdown
  • Mitigation lever NPV / ROI analysis
  • Full financial assumptions table
  • Aggregate monetary-impact numbers

Outputs
-------
- scenario_monetization.csv
- risk_premium_breakdown.csv
- mitigation_cost_benefit.csv
- financial_assumptions.csv
- monetary_numbers.csv
- figures/cost_benefit_mitigation.png
- figures/scenario_monetization_waterfall.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import (DC_PARAMS, GRID, FINANCE, GROWTH_SCENARIOS,
                    MITIGATION_LEVERS, OUTPUT_DIR, FIGURE_DIR, PLOT)


# ── Scenario monetisation ───────────────────────────────────────────────
_SCENARIO_DEFS = {
    "Baseline": {
        "it_mw": DC_PARAMS["it_capacity_mw"],
        "pue": DC_PARAMS["pue_baseline"],
        "ci": GRID["carbon_intensity_mean"],
        "growth": 0.05,
    },
    "Efficient Ops": {
        "it_mw": DC_PARAMS["it_capacity_mw"],
        "pue": 1.18,
        "ci": GRID["carbon_intensity_mean"] * 0.90,
        "growth": 0.05,
    },
    "High Growth": {
        "it_mw": DC_PARAMS["it_capacity_mw"],
        "pue": DC_PARAMS["pue_baseline"],
        "ci": GRID["carbon_intensity_mean"],
        "growth": 0.30,
    },
    "Climate Stress": {
        "it_mw": DC_PARAMS["it_capacity_mw"],
        "pue": DC_PARAMS["pue_baseline"] + 0.12,
        "ci": GRID["carbon_intensity_mean"] * 1.10,
        "growth": 0.15,
    },
}


def _npv_stream(annual_cost, discount, n_years, growth=0.0):
    """NPV of a cost stream growing at *growth* rate p.a."""
    return sum(annual_cost * (1 + growth) ** yr * (1 + discount) ** (-yr)
               for yr in range(1, n_years + 1))


def compute_scenario_monetisation() -> pd.DataFrame:
    rows = []
    d = FINANCE["discount_rate"]
    n = FINANCE["horizon_years"]
    scc = FINANCE["scc_usd_per_ton"]
    ep  = FINANCE["energy_price_usd_per_mwh"]

    for name, s in _SCENARIO_DEFS.items():
        total_mw = s["it_mw"] * s["pue"]
        annual_mwh = total_mw * 8760
        annual_tons = total_mw * s["ci"] * 8760 / 1000

        energy_cost = annual_mwh * ep
        carbon_cost = annual_tons * scc
        cooling     = FINANCE["cooling_maintenance_annual"]
        demand_chg  = FINANCE["demand_charge_usd_per_kw_yr"] * total_mw * 1000
        total       = energy_cost + carbon_cost + cooling + demand_chg

        npv = _npv_stream(total, d, n, growth=s["growth"])

        # 10-year cumulative with growth
        cumul_emissions = sum(
            (s["it_mw"] * (1 + s["growth"])**yr * s["pue"]
             * s["ci"] * 8760 / 1000)
            for yr in range(1, n + 1))

        rows.append({
            "scenario": name,
            "total_power_mw": round(total_mw, 2),
            "annual_energy_mwh": round(annual_mwh, 0),
            "annual_emissions_tons": round(annual_tons, 0),
            "annual_energy_cost": round(energy_cost, 0),
            "annual_carbon_cost": round(carbon_cost, 0),
            "annual_demand_charge": round(demand_chg, 0),
            "annual_cooling_maint": round(cooling, 0),
            "annual_total_cost": round(total, 0),
            "npv_10yr": round(npv, 0),
            "cumulative_10yr_emissions": round(cumul_emissions, 0),
        })
    return pd.DataFrame(rows)


# ── Risk-premium breakdown ───────────────────────────────────────────────
def compute_risk_premium() -> pd.DataFrame:
    base_mw = DC_PARAMS["it_capacity_mw"] * DC_PARAMS["pue_baseline"]
    base_mwh = base_mw * 8760
    base_tons = base_mw * GRID["carbon_intensity_mean"] * 8760 / 1000

    base_energy = base_mwh * FINANCE["energy_price_usd_per_mwh"]
    base_carbon = base_tons * FINANCE["scc_usd_per_ton"]
    base_total  = base_energy + base_carbon

    rows = [
        {"component": "Energy Procurement", "base_annual": round(base_energy, 0),
         "volatility_pct": 8.5, "risk_premium": round(base_energy * 0.085, 0),
         "risk_adjusted": round(base_energy * 1.085, 0)},
        {"component": "Carbon Liability (SCC)", "base_annual": round(base_carbon, 0),
         "volatility_pct": 15.0, "risk_premium": round(base_carbon * 0.15, 0),
         "risk_adjusted": round(base_carbon * 1.15, 0)},
        {"component": "Demand Charges", "base_annual":
             round(FINANCE["demand_charge_usd_per_kw_yr"] * base_mw * 1000, 0),
         "volatility_pct": 5.0,
         "risk_premium": round(FINANCE["demand_charge_usd_per_kw_yr"] * base_mw * 1000 * 0.05, 0),
         "risk_adjusted": round(FINANCE["demand_charge_usd_per_kw_yr"] * base_mw * 1000 * 1.05, 0)},
        {"component": "Cooling & Maintenance", "base_annual":
             round(FINANCE["cooling_maintenance_annual"], 0),
         "volatility_pct": 3.0,
         "risk_premium": round(FINANCE["cooling_maintenance_annual"] * 0.03, 0),
         "risk_adjusted": round(FINANCE["cooling_maintenance_annual"] * 1.03, 0)},
        {"component": "Peak Breach Penalty",
         "base_annual": round(FINANCE["peak_breach_penalty_usd_per_mwh"]
                              * FINANCE["peak_breach_hours"] * 2, 0),
         "volatility_pct": 25.0,
         "risk_premium": round(FINANCE["peak_breach_penalty_usd_per_mwh"]
                               * FINANCE["peak_breach_hours"] * 2 * 0.25, 0),
         "risk_adjusted": round(FINANCE["peak_breach_penalty_usd_per_mwh"]
                                * FINANCE["peak_breach_hours"] * 2 * 1.25, 0)},
    ]
    df = pd.DataFrame(rows)
    # Totals row
    total = {c: df[c].sum() if c != "component" else "TOTAL"
             for c in df.columns}
    total["volatility_pct"] = round(
        df["risk_premium"].sum() / df["base_annual"].sum() * 100, 1)
    df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)
    return df


# ── Mitigation cost-benefit ─────────────────────────────────────────────
def compute_mitigation_cba() -> pd.DataFrame:
    base_mw = DC_PARAMS["it_capacity_mw"] * DC_PARAMS["pue_baseline"]
    base_energy_cost = base_mw * 8760 * FINANCE["energy_price_usd_per_mwh"]
    base_carbon_cost = (base_mw * GRID["carbon_intensity_mean"] * 8760 / 1000
                        * FINANCE["scc_usd_per_ton"])
    d = FINANCE["discount_rate"]
    n = FINANCE["horizon_years"]

    rows = []
    for name, lev in MITIGATION_LEVERS.items():
        co2_frac = lev["emission_reduction_pct"] / 100.0
        nrg_frac = lev.get("energy_saving_pct", 0.0) / 100.0
        # Emission reduction avoids carbon cost; energy saving avoids energy cost
        annual_savings = base_carbon_cost * co2_frac + base_energy_cost * nrg_frac
        capex = lev["capex_usd"]
        npv = _npv_stream(annual_savings, d, n) - capex
        roi = npv / capex if capex > 0 else float("inf")
        payback = capex / annual_savings if annual_savings > 0 else float("inf")

        rows.append({
            "lever": name,
            "emission_reduction_pct": lev["emission_reduction_pct"],
            "capex_usd": capex,
            "annual_savings_usd": round(annual_savings, 0),
            "npv_10yr_usd": round(npv, 0),
            "roi_x": round(roi, 1),
            "payback_years": round(payback, 2),
        })
    return pd.DataFrame(rows)


# ── Financial assumptions table ──────────────────────────────────────────
def build_financial_assumptions() -> pd.DataFrame:
    return pd.DataFrame([
        {"parameter": "Social Cost of Carbon", "value": FINANCE["scc_usd_per_ton"],
         "unit": "USD/ton CO₂", "source": "EPA 2024 (central, 3% DR)"},
        {"parameter": "Wholesale Energy Price", "value": FINANCE["energy_price_usd_per_mwh"],
         "unit": "USD/MWh", "source": "PJM 2024 avg"},
        {"parameter": "Discount Rate", "value": f"{FINANCE['discount_rate']*100:.0f}%",
         "unit": "—", "source": "WACC for hyperscale DC"},
        {"parameter": "Horizon", "value": FINANCE["horizon_years"],
         "unit": "years", "source": "Model assumption"},
        {"parameter": "Contract Peak", "value": FINANCE["contract_peak_mw"],
         "unit": "MW", "source": "Typical Ashburn utility contract"},
        {"parameter": "Peak Breach Penalty", "value": FINANCE["peak_breach_penalty_usd_per_mwh"],
         "unit": "USD/MWh", "source": "Industry estimate"},
        {"parameter": "Demand Charge", "value": FINANCE["demand_charge_usd_per_kw_yr"],
         "unit": "USD/kW/yr", "source": "Dominion Energy tariff"},
        {"parameter": "Cooling Maintenance", "value": f"${FINANCE['cooling_maintenance_annual']/1e6:.1f}M",
         "unit": "USD/yr", "source": "ASHRAE benchmark"},
    ])


# ── Monetary summary numbers ────────────────────────────────────────────
def build_monetary_summary(scen_df, risk_df, mit_df) -> pd.DataFrame:
    baseline = scen_df[scen_df["scenario"] == "Baseline"].iloc[0]
    high_g   = scen_df[scen_df["scenario"] == "High Growth"].iloc[0]
    risk_tot = risk_df[risk_df["component"] == "TOTAL"].iloc[0]

    rows = [
        {"metric": "Baseline Annual Cost", "value": baseline["annual_total_cost"],
         "unit": "USD", "context": "30 MW DC, PUE 1.30, grid avg"},
        {"metric": "Baseline 10-yr NPV", "value": baseline["npv_10yr"],
         "unit": "USD", "context": "Discounted at 8%"},
        {"metric": "High-Growth 10-yr NPV", "value": high_g["npv_10yr"],
         "unit": "USD", "context": "30% CAGR"},
        {"metric": "Risk Premium (annual)", "value": risk_tot["risk_premium"],
         "unit": "USD", "context": "Composite volatility mark-up"},
        {"metric": "Best Mitigation NPV", "value": mit_df["npv_10yr_usd"].max(),
         "unit": "USD", "context": mit_df.loc[mit_df["npv_10yr_usd"].idxmax(), "lever"]},
        {"metric": "Best Mitigation ROI", "value": mit_df["roi_x"].max(),
         "unit": "x", "context": mit_df.loc[mit_df["roi_x"].idxmax(), "lever"]},
    ]
    return pd.DataFrame(rows)


# ── Figures ──────────────────────────────────────────────────────────────
def plot_mitigation(mit_df):
    fig, axes = plt.subplots(1, 2, figsize=PLOT["figsize_wide"])

    # NPV bar chart
    ax = axes[0]
    colors = ["#27ae60", "#2980b9", "#f39c12", "#8e44ad"]
    ax.barh(mit_df["lever"], mit_df["npv_10yr_usd"]/1e6,
            color=colors[:len(mit_df)], edgecolor="white", height=0.55)
    for i, v in enumerate(mit_df["npv_10yr_usd"]):
        ax.text(v/1e6 + 0.5, i, f"${v/1e6:,.1f}M", va="center", fontsize=10)
    ax.set_xlabel("NPV 10-yr ($M)", fontsize=11)
    ax.set_title("Mitigation Lever NPV", fontsize=12, fontweight="bold")

    # ROI bar chart
    ax = axes[1]
    ax.barh(mit_df["lever"], mit_df["roi_x"],
            color=colors[:len(mit_df)], edgecolor="white", height=0.55)
    for i, v in enumerate(mit_df["roi_x"]):
        ax.text(v + 0.5, i, f"{v:.1f}×", va="center", fontsize=10)
    ax.set_xlabel("Return on Investment (×)", fontsize=11)
    ax.set_title("Mitigation Lever ROI", fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "cost_benefit_mitigation.png",
                dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close()


def plot_scenario_waterfall(scen_df):
    fig, ax = plt.subplots(figsize=(12, 6))

    names = scen_df["scenario"].tolist()
    npvs = scen_df["npv_10yr"].values / 1e6
    colors = ["#3498db", "#27ae60", "#e74c3c", "#e67e22"]

    bars = ax.bar(names, npvs, color=colors, edgecolor="white", width=0.55)
    for bar, v in zip(bars, npvs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"${v:,.0f}M", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("10-Year NPV ($M)", fontsize=12)
    ax.set_title("Scenario Monetisation: 10-Year Net Present Value",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "scenario_monetization_waterfall.png",
                dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close()


# ── Entry Point ──────────────────────────────────────────────────────────
def run():
    print("  ▶ Running cost-benefit analysis …")

    scen_df  = compute_scenario_monetisation()
    risk_df  = compute_risk_premium()
    mit_df   = compute_mitigation_cba()
    fin_df   = build_financial_assumptions()
    money_df = build_monetary_summary(scen_df, risk_df, mit_df)

    scen_df.to_csv(OUTPUT_DIR  / "scenario_monetization.csv",   index=False)
    risk_df.to_csv(OUTPUT_DIR  / "risk_premium_breakdown.csv",  index=False)
    mit_df.to_csv(OUTPUT_DIR   / "mitigation_cost_benefit.csv", index=False)
    fin_df.to_csv(OUTPUT_DIR   / "financial_assumptions.csv",   index=False)
    money_df.to_csv(OUTPUT_DIR / "monetary_numbers.csv",        index=False)

    plot_mitigation(mit_df)
    plot_scenario_waterfall(scen_df)

    print("    ✓ 4 scenarios monetised")
    print(f"    ✓ Best lever: {mit_df.loc[mit_df['npv_10yr_usd'].idxmax(), 'lever']}"
          f" (NPV ${mit_df['npv_10yr_usd'].max()/1e6:,.1f}M)")
    return scen_df, risk_df, mit_df, fin_df, money_df


if __name__ == "__main__":
    run()
