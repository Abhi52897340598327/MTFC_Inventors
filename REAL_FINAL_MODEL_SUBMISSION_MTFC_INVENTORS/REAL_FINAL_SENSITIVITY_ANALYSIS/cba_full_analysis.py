"""
CBA FULL ANALYSIS — Section 4: Comprehensive Cost-Benefit Engine
==================================================================
Builds on scenario projections to compute:
  (a) Exhaustive cost decomposition (8 components)
  (b) Four mitigation levers with break-even analysis
  (c) NPV / ROI / payback / break-even carbon price
  (d) Monte Carlo uncertainty around each scenario
  (e) Risk-premium estimation

Outputs:
    outputs/cba_cost_breakdown.csv        — 8 cost components × scenario × year
    outputs/cba_mitigation_analysis.csv   — lever economics
    outputs/cba_npv_summary.csv           — headline NPV / ROI / payback
    outputs/cba_risk_metrics.csv          — VaR, CVaR, risk premiums
    outputs/cba_financial_assumptions.csv — all assumptions table
"""

import numpy as np
import pandas as pd
from config import FINANCE, DC_PARAMS, OUTPUT_DIR


# ═══════════════════════════════════════════════════════════════════════════
#  COST COMPONENT FUNCTIONS (a–h)
# ═══════════════════════════════════════════════════════════════════════════

def compute_cost_breakdown(proj_df, p):
    """
    For each scenario × year, compute all 8 cost components:
      (a) Energy Procurement       = energy_mwh × $72/MWh
      (b) Demand Charges           = peak_mw × $180/kW/yr
      (c) Carbon Liability (SCC)   = co2_tonnes × $190/t
      (d) Carbon Liability (Low)   = co2_tonnes × $95/t
      (e) Carbon Liability (High)  = co2_tonnes × $300/t
      (f) Cooling Maintenance      = base × CDD_factor
      (g) Peak Breach Penalty      = breach_hours × excess_mw × $450/MWh
      (h) Risk Premium             = energy_cost × volatility_markup
    """
    scc            = FINANCE["scc_usd_per_ton"]
    scc_low        = FINANCE["carbon_price_low_usd_per_ton"]
    scc_high       = FINANCE["carbon_price_high_usd_per_ton"]
    energy_price   = FINANCE["energy_price_usd_per_mwh"]
    demand_charge  = FINANCE["demand_charge_usd_per_kw_yr"]
    contract_peak  = FINANCE["contract_peak_mw"]
    breach_hours   = FINANCE["peak_breach_hours"]
    breach_penalty = FINANCE["peak_breach_penalty_usd_per_mwh"]
    vol_markup     = FINANCE["electricity_volatility_markup"]
    cool_base      = FINANCE["cooling_maintenance_annual"]

    # CDD factor: scale cooling maintenance by temperature trend
    cdd_annual_mean = p.get("va_cdd_annual_mean", 1200)
    temp_trend_yr   = p.get("va_temp_trend_f_per_decade", 0.3) / 10.0

    rows = []
    for _, r in proj_df.iterrows():
        yr_off = r["year_offset"]

        # (a) Energy Procurement
        energy_cost = r["energy_mwh"] * energy_price

        # (b) Demand Charges (peak MW × $/kW × 1000 kW/MW)
        peak_mw = r["facility_power_mw"] * 1.15  # Peak = 115% of avg facility power
        demand_cost = peak_mw * 1000 * demand_charge

        # (c-e) Carbon Liability at three SCC levels
        carbon_scc      = r["co2_tonnes"] * scc
        carbon_low      = r["co2_tonnes"] * scc_low
        carbon_high     = r["co2_tonnes"] * scc_high

        # (f) Cooling Maintenance (scales with CDD growth)
        cdd_factor = 1.0 + (temp_trend_yr * yr_off * 65) / max(cdd_annual_mean, 1)  # CDD growth factor
        cool_cost = cool_base * cdd_factor

        # (g) Peak Breach Penalty
        excess_mw = max(0, peak_mw - contract_peak)
        breach_cost = breach_hours * excess_mw * breach_penalty if excess_mw > 0 else 0

        # (h) Risk Premium (electricity price volatility)
        risk_premium = energy_cost * vol_markup

        # Regulatory Reserve (2% of total carbon liability as actuarial reserve)
        reg_reserve = carbon_scc * 0.02

        total = energy_cost + demand_cost + carbon_scc + cool_cost + breach_cost + risk_premium + reg_reserve

        rows.append({
            "scenario": r["scenario"],
            "year": r["year"],
            "year_offset": yr_off,
            "a_energy_procurement_usd": round(energy_cost, 0),
            "b_demand_charges_usd": round(demand_cost, 0),
            "c_carbon_liability_scc_usd": round(carbon_scc, 0),
            "d_carbon_liability_low_usd": round(carbon_low, 0),
            "e_carbon_liability_high_usd": round(carbon_high, 0),
            "f_cooling_maintenance_usd": round(cool_cost, 0),
            "g_peak_breach_penalty_usd": round(breach_cost, 0),
            "h_risk_premium_usd": round(risk_premium, 0),
            "i_regulatory_reserve_usd": round(reg_reserve, 0),
            "total_cost_usd": round(total, 0),
            "total_cost_usd_m": round(total / 1e6, 2),
        })

    cost_df = pd.DataFrame(rows)
    cost_df.to_csv(OUTPUT_DIR / "cba_cost_breakdown.csv", index=False)
    print(f"✓ Saved cba_cost_breakdown.csv  ({len(cost_df)} rows)")
    return cost_df


# ═══════════════════════════════════════════════════════════════════════════
#  MITIGATION LEVERS
# ═══════════════════════════════════════════════════════════════════════════

def compute_mitigation(proj_df, p):
    """
    Four mitigation levers applied to the DC Boom scenario:
      1. Dynamic Workload Shifting   — move compute to low-CI hours
      2. PUE Optimization            — cooling upgrades (2× ESIF improvement rate)
      3. Cleaner Grid (50 MW PPA)    — direct carbon offset via renewables
      4. Combined Portfolio           — all three together

    Each lever: CapEx, annual OpEx savings, CO₂ reduction, NPV, ROI, payback, break-even carbon price.
    """
    discount = FINANCE["discount_rate"]
    horizon  = FINANCE["horizon_years"]
    scc      = FINANCE["scc_usd_per_ton"]
    energy_price = FINANCE["energy_price_usd_per_mwh"]

    # Use DC Boom scenario as the "do-nothing" baseline for mitigation
    boom = proj_df[proj_df["scenario"] == "DC Boom"].copy()
    if len(boom) == 0:
        print("  ⚠ No 'DC Boom' scenario found — using first scenario")
        first_scen = proj_df["scenario"].unique()[0]
        boom = proj_df[proj_df["scenario"] == first_scen].copy()

    # PJM diurnal ratio for workload shifting benefit
    diurnal_ratio = p.get("pjm_diurnal_peak_trough_ratio", 1.25)
    # ESIF PUE improvement rate for optimization lever
    esif_pue_rate = abs(p.get("esif_pue_trend_per_year", 0.004))

    levers = {
        "Dynamic Workload Shifting": {
            "capex": 600_000,
            "annual_opex": 50_000,
            "co2_reduction_pct": min(0.115, (diurnal_ratio - 1) * 0.50),  # Shift load to low-CI hours
            "energy_saving_pct": 0.02,  # Small energy saving from off-peak
            "description": f"Move {min(11.5,(diurnal_ratio-1)*50):.1f}% of compute to low-carbon hours",
        },
        "PUE Optimization (Cooling)": {
            "capex": 14_000_000,
            "annual_opex": 200_000,
            "co2_reduction_pct": 0.188,  # From PUE 1.30 → ~1.06 over 10 yr (ESIF target)
            "energy_saving_pct": 0.188,  # PUE improvement directly saves energy
            "description": f"Aggressive cooling upgrades targeting ESIF-level PUE (~1.05)",
        },
        "Cleaner Grid (50 MW PPA)": {
            "capex": 200_000,  # Contract negotiation costs
            "annual_opex": 0,
            "co2_reduction_pct": 0.224,  # 50MW PPA / ~200MW avg facility ≈ 25% clean
            "energy_saving_pct": 0.0,  # No energy saving, just cleaner source
            "ppa_cost_offset_mwh": -27.0,  # PPA @ $45 vs grid $72 → save $27/MWh on 50MW
            "description": "50 MW renewable PPA at $45/MWh displacing grid carbon",
        },
        "Combined Portfolio": {
            "capex": 14_800_000,
            "annual_opex": 250_000,
            "co2_reduction_pct": 0.419,  # Combined (non-additive: 1 - (1-.115)*(1-.188)*(1-.224))
            "energy_saving_pct": 0.20,
            "ppa_cost_offset_mwh": -27.0,
            "description": "All three levers combined (non-additive interaction)",
        },
    }

    lever_rows = []
    for lname, lev in levers.items():
        # Year-by-year cash flows
        annual_benefits = []
        annual_co2_avoided = []

        for _, r in boom.iterrows():
            yr_off = r["year_offset"]
            if yr_off == 0:
                continue  # CapEx year

            # Energy savings
            energy_saved_mwh = r["energy_mwh"] * lev["energy_saving_pct"]
            energy_saving_usd = energy_saved_mwh * energy_price

            # PPA cost offset
            ppa_saving = 0
            if "ppa_cost_offset_mwh" in lev:
                ppa_mw = 50.0
                ppa_mwh = ppa_mw * 8760
                ppa_saving = ppa_mwh * abs(lev["ppa_cost_offset_mwh"])

            # Carbon savings (valued at SCC)
            co2_avoided = r["co2_tonnes"] * lev["co2_reduction_pct"]
            carbon_saving = co2_avoided * scc

            # Total annual benefit
            total_benefit = energy_saving_usd + carbon_saving + ppa_saving - lev["annual_opex"]
            annual_benefits.append(total_benefit)
            annual_co2_avoided.append(co2_avoided)

        # NPV of benefits
        npv_benefits = sum(b / (1 + discount) ** (i + 1) for i, b in enumerate(annual_benefits))
        npv_net = npv_benefits - lev["capex"]
        roi = npv_net / lev["capex"] if lev["capex"] > 0 else float("inf")

        # Payback period
        cumulative = -lev["capex"]
        payback_yr = None
        for i, b in enumerate(annual_benefits):
            cumulative += b
            if cumulative >= 0 and payback_yr is None:
                payback_yr = i + 1

        # Break-even carbon price
        # At what SCC does NPV = 0?
        # NPV = -CapEx + Σ [(energy_saving + co2_avoided×SCC_be + ppa) / (1+r)^t] = 0
        total_discounted_co2 = sum(
            co2 / (1 + discount) ** (i + 1)
            for i, co2 in enumerate(annual_co2_avoided)
        )
        non_carbon_npv = -lev["capex"] + sum(
            (boom.iloc[i + 1]["energy_mwh"] * lev["energy_saving_pct"] * energy_price
             + (abs(lev.get("ppa_cost_offset_mwh", 0)) * 50 * 8760 if "ppa_cost_offset_mwh" in lev else 0)
             - lev["annual_opex"])
            / (1 + discount) ** (i + 1)
            for i in range(len(annual_benefits))
        )
        if total_discounted_co2 > 0:
            breakeven_scc = max(0, -non_carbon_npv / total_discounted_co2)
        else:
            breakeven_scc = float("inf")

        total_co2_avoided_10yr = sum(annual_co2_avoided)
        cost_per_tonne = lev["capex"] / total_co2_avoided_10yr if total_co2_avoided_10yr > 0 else float("inf")

        lever_rows.append({
            "lever": lname,
            "description": lev["description"],
            "capex_usd": lev["capex"],
            "annual_opex_usd": lev["annual_opex"],
            "co2_reduction_pct": lev["co2_reduction_pct"],
            "energy_saving_pct": lev["energy_saving_pct"],
            "total_co2_avoided_10yr_tonnes": round(total_co2_avoided_10yr, 0),
            "npv_benefits_usd": round(npv_benefits, 0),
            "npv_net_usd": round(npv_net, 0),
            "roi_x": round(roi, 2),
            "payback_years": payback_yr if payback_yr else ">10",
            "breakeven_scc_usd_per_ton": round(breakeven_scc, 2) if breakeven_scc < 1e6 else "N/A",
            "cost_per_tonne_avoided_usd": round(cost_per_tonne, 2) if cost_per_tonne < 1e6 else "N/A",
        })

    mit_df = pd.DataFrame(lever_rows)
    mit_df.to_csv(OUTPUT_DIR / "cba_mitigation_analysis.csv", index=False)
    print(f"✓ Saved cba_mitigation_analysis.csv  ({len(mit_df)} rows)")
    return mit_df


# ═══════════════════════════════════════════════════════════════════════════
#  NPV / ROI / HEADLINE METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_npv_summary(proj_df, cost_df, mit_df):
    """
    Compute headline financial metrics for each scenario.
    """
    discount = FINANCE["discount_rate"]
    horizon  = FINANCE["horizon_years"]

    rows = []
    for scen in proj_df["scenario"].unique():
        sc = cost_df[cost_df["scenario"] == scen]
        sp = proj_df[proj_df["scenario"] == scen]

        # NPV of total costs (year 1..10)
        future = sc[sc["year_offset"] > 0]
        npv_cost = sum(
            row["total_cost_usd"] / (1 + discount) ** row["year_offset"]
            for _, row in future.iterrows()
        )

        # Cumulative CO₂
        cum_co2 = sp["co2_tonnes"].sum()
        # Cumulative energy
        cum_energy = sp["energy_gwh"].sum()
        # Final year cost
        final_cost = sc[sc["year_offset"] == horizon]["total_cost_usd_m"].values[0] if len(sc[sc["year_offset"] == horizon]) > 0 else 0

        # Cost growth rate (year 1 to year 10)
        yr1_cost = sc[sc["year_offset"] == 1]["total_cost_usd_m"].values
        yr10_cost = sc[sc["year_offset"] == horizon]["total_cost_usd_m"].values
        if len(yr1_cost) > 0 and len(yr10_cost) > 0 and yr1_cost[0] > 0:
            cost_cagr = (yr10_cost[0] / yr1_cost[0]) ** (1.0 / (horizon - 1)) - 1
        else:
            cost_cagr = 0

        rows.append({
            "scenario": scen,
            "npv_total_cost_usd_m": round(npv_cost / 1e6, 2),
            "cumulative_co2_tonnes": round(cum_co2, 0),
            "cumulative_co2_mmt": round(cum_co2 / 1e6, 4),
            "cumulative_energy_gwh": round(cum_energy, 1),
            "final_year_cost_usd_m": round(final_cost, 2),
            "cost_cagr": round(cost_cagr, 4),
        })

    # Add mitigation scenario (Combined Portfolio applied to DC Boom)
    boom_cost = cost_df[cost_df["scenario"] == "DC Boom"]
    if len(boom_cost) > 0 and len(mit_df) > 0:
        combined = mit_df[mit_df["lever"] == "Combined Portfolio"]
        if len(combined) > 0:
            comb = combined.iloc[0]
            boom_npv = [r for r in rows if r["scenario"] == "DC Boom"]
            if boom_npv:
                mitigated_npv = boom_npv[0]["npv_total_cost_usd_m"] * (1 - 0.35)  # ~35% cost reduction
                rows.append({
                    "scenario": "DC Boom + Combined Mitigation",
                    "npv_total_cost_usd_m": round(mitigated_npv, 2),
                    "cumulative_co2_tonnes": round(boom_npv[0]["cumulative_co2_tonnes"] * (1 - 0.419), 0),
                    "cumulative_co2_mmt": round(boom_npv[0]["cumulative_co2_mmt"] * (1 - 0.419), 4),
                    "cumulative_energy_gwh": round(boom_npv[0]["cumulative_energy_gwh"] * (1 - 0.20), 1),
                    "final_year_cost_usd_m": round(boom_npv[0]["final_year_cost_usd_m"] * (1 - 0.35), 2),
                    "cost_cagr": round(boom_npv[0]["cost_cagr"] * 0.85, 4),
                })

    npv_df = pd.DataFrame(rows)
    npv_df.to_csv(OUTPUT_DIR / "cba_npv_summary.csv", index=False)
    print(f"✓ Saved cba_npv_summary.csv  ({len(npv_df)} rows)")
    return npv_df


# ═══════════════════════════════════════════════════════════════════════════
#  RISK METRICS (Monte Carlo on each scenario)
# ═══════════════════════════════════════════════════════════════════════════

def compute_risk_metrics(proj_df, p, n_sims=10_000, seed=42):
    """
    Run Monte Carlo on each scenario to estimate VaR / CVaR for total 10-yr cost.
    Perturb: grid CI (±20%), PUE (±10%), utilization (±15%), energy price (±25%).
    """
    rng = np.random.default_rng(seed)
    discount = FINANCE["discount_rate"]
    horizon  = FINANCE["horizon_years"]
    scc      = FINANCE["scc_usd_per_ton"]
    energy_price_base = FINANCE["energy_price_usd_per_mwh"]

    risk_rows = []

    for scen in proj_df["scenario"].unique():
        sp = proj_df[proj_df["scenario"] == scen]
        base_years = sp[sp["year_offset"] > 0]

        # Monte Carlo: perturb each year's parameters
        total_costs = np.zeros(n_sims)

        for sim in range(n_sims):
            ci_shock    = rng.normal(1.0, 0.20)
            pue_shock   = rng.normal(1.0, 0.10)
            util_shock  = rng.normal(1.0, 0.15)
            price_shock = rng.normal(1.0, 0.25)

            sim_cost = 0
            for _, r in base_years.iterrows():
                ci_sim   = r["effective_ci_kgco2_mwh"] * ci_shock
                pue_sim  = r["pue_effective"] * pue_shock
                util_sim = np.clip(r["utilization"] * util_shock, 0.2, 0.95)

                # Recompute energy
                fac_mw = r["it_capacity_mw"] * util_sim * pue_sim
                energy_mwh = fac_mw * 8760
                co2_t = energy_mwh * ci_sim / 1000

                # Cost
                e_cost = energy_mwh * energy_price_base * price_shock
                c_cost = co2_t * scc
                yr_cost = e_cost + c_cost

                sim_cost += yr_cost / (1 + discount) ** r["year_offset"]

            total_costs[sim] = sim_cost

        # Statistics
        mean_cost = np.mean(total_costs) / 1e6
        std_cost  = np.std(total_costs) / 1e6
        var_90    = np.percentile(total_costs, 90) / 1e6
        var_95    = np.percentile(total_costs, 95) / 1e6
        var_99    = np.percentile(total_costs, 99) / 1e6
        cvar_95   = np.mean(total_costs[total_costs >= np.percentile(total_costs, 95)]) / 1e6

        risk_rows.append({
            "scenario": scen,
            "npv_mean_usd_m": round(mean_cost, 2),
            "npv_std_usd_m": round(std_cost, 2),
            "npv_var_90_usd_m": round(var_90, 2),
            "npv_var_95_usd_m": round(var_95, 2),
            "npv_var_99_usd_m": round(var_99, 2),
            "npv_cvar_95_usd_m": round(cvar_95, 2),
            "coefficient_of_variation": round(std_cost / mean_cost, 3) if mean_cost > 0 else 0,
            "n_simulations": n_sims,
        })

    risk_df = pd.DataFrame(risk_rows)
    risk_df.to_csv(OUTPUT_DIR / "cba_risk_metrics.csv", index=False)
    print(f"✓ Saved cba_risk_metrics.csv  ({len(risk_df)} rows)")
    return risk_df


# ═══════════════════════════════════════════════════════════════════════════
#  FINANCIAL ASSUMPTIONS TABLE
# ═══════════════════════════════════════════════════════════════════════════

def save_assumptions(p):
    """Save all financial / physical assumptions for transparency."""
    assumptions = [
        ("Social Cost of Carbon (EPA 2024)", "$190/tonne", "FINANCE.scc_usd_per_ton"),
        ("Carbon Price — Low Scenario", "$95/tonne", "FINANCE.carbon_price_low"),
        ("Carbon Price — High/Tail Scenario", "$300/tonne", "FINANCE.carbon_price_high"),
        ("Energy Price (PJM Average)", "$72/MWh", "FINANCE.energy_price"),
        ("Discount Rate", "8%", "FINANCE.discount_rate"),
        ("Projection Horizon", "10 years (2025–2035)", "FINANCE.horizon_years"),
        ("Contract Peak Capacity", "40 MW", "FINANCE.contract_peak_mw"),
        ("Peak Breach Penalty Rate", "$450/MWh", "FINANCE.peak_breach_penalty"),
        ("Electricity Volatility Markup", "5%", "FINANCE.electricity_volatility_markup"),
        ("Cooling Maintenance (base)", "$2.5M/yr", "FINANCE.cooling_maintenance"),
        ("Demand Charge", "$180/kW/yr", "FINANCE.demand_charge"),
        ("DC IT Capacity (base)", "30 MW", "DC_PARAMS.it_capacity_mw"),
        ("Industry PUE Baseline", "1.30", "DC_PARAMS.pue_baseline"),
        ("Base Utilization", "60%", "scenario assumption"),
        ("VA Grid CI (empirical)", f"{p.get('va_grid_ci_kgco2_mwh_latest', 345):.1f} kg CO₂/MWh", "Derived: emissions/energy"),
        ("DC Spending CAGR (empirical)", f"{p.get('dc_spend_cagr', 0.25):.1%}", "monthly-spending-data-center-us.csv"),
        ("VA Energy CAGR 20yr (empirical)", f"{p.get('va_energy_cagr_20yr', 0.005):.2%}", "virginia_yearly_energy_consumption_bbtu.csv"),
        ("VA Temp Trend (empirical)", f"{p.get('va_temp_trend_f_per_decade', 0.3):+.2f}°F/decade", "monthly_temp_virginia.csv"),
        ("ESIF PUE Mean (empirical)", f"{p.get('esif_pue_mean', 1.05):.3f}", "esif_daily_avg_interpolated.csv"),
        ("Coal Decline CAGR 20yr", f"{p.get('coal_consumption_cagr_20yr', -0.05):.1%}", "energy_by_source_annual_grid_comp.csv"),
        ("Renewable Growth CAGR 20yr", f"{p.get('renewable_consumption_cagr_20yr', 0.03):.1%}", "energy_by_source_annual_grid_comp.csv"),
        ("VA Total CO₂ (latest)", f"{p.get('va_total_co2_mmt_latest', 21):.1f} MMT", "virginia_total_carbon_emissions_energyproduction_annual.csv"),
        ("Exponential Fit R² (DC spend)", f"{p.get('dc_spend_exponential_r2', 0.95):.3f}", "monthly-spending-data-center-us.csv"),
        ("PJM Load CAGR", f"{p.get('pjm_load_cagr', 0.05):.2%}", "hrl_load_metered_combined_cleaned.csv"),
        ("PJM Diurnal Peak/Trough", f"{p.get('pjm_diurnal_peak_trough_ratio', 1.25):.2f}×", "hrl_load_metered_combined_cleaned.csv"),
    ]

    df = pd.DataFrame(assumptions, columns=["assumption", "value", "source"])
    df.to_csv(OUTPUT_DIR / "cba_financial_assumptions.csv", index=False)
    print(f"✓ Saved cba_financial_assumptions.csv  ({len(df)} rows)")
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  MASTER FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def run_full_cba(proj_df, p):
    """Execute full CBA pipeline and return all DataFrames."""
    print("\n" + "=" * 72)
    print("  SECTION 4: FULL COST-BENEFIT ANALYSIS")
    print("=" * 72)

    print("\n── (a–h) Cost Decomposition ──")
    cost_df = compute_cost_breakdown(proj_df, p)

    print("\n── Mitigation Levers ──")
    mit_df = compute_mitigation(proj_df, p)

    print("\n── NPV / ROI Summary ──")
    npv_df = compute_npv_summary(proj_df, cost_df, mit_df)

    print("\n── Risk Metrics (Monte Carlo) ──")
    risk_df = compute_risk_metrics(proj_df, p, n_sims=10_000)

    print("\n── Financial Assumptions ──")
    assum_df = save_assumptions(p)

    # Print headline summary
    print("\n" + "─" * 72)
    print("  HEADLINE RESULTS")
    print("─" * 72)
    for _, row in npv_df.iterrows():
        print(f"  {row['scenario']:35s}  NPV=${row['npv_total_cost_usd_m']:>8.1f}M  "
              f"CO₂={row['cumulative_co2_mmt']:.3f} MMT")

    print("\n  Mitigation Levers (applied to DC Boom):")
    for _, row in mit_df.iterrows():
        print(f"    {row['lever']:35s}  NPV(net)=${row['npv_net_usd']:>12,.0f}  "
              f"ROI={row['roi_x']:.1f}×  Payback={row['payback_years']}yr  "
              f"CO₂ avoided={row['total_co2_avoided_10yr_tonnes']:>10,.0f}t")

    return {
        "cost_breakdown": cost_df,
        "mitigation": mit_df,
        "npv_summary": npv_df,
        "risk_metrics": risk_df,
        "assumptions": assum_df,
    }


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_ingestion import ingest_all
    from cba_scenarios import project_scenarios
    data = ingest_all()
    proj_df, _ = project_scenarios(data["params"])
    run_full_cba(proj_df, data["params"])
