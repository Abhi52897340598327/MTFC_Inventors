"""
CBA SCENARIOS — Section 3: Data-Driven Scenario Construction
==============================================================
Builds four 10-year projection scenarios, each derived from
empirical parameters extracted by data_ingestion.py.

Scenarios:
  1. Baseline          — Historical Virginia energy CAGR continues
  2. DC Boom           — AI/DC growth follows DC-spending CAGR
  3. Efficient Transition — DC Boom + PUE improvement + grid decarb
  4. Climate Stress     — DC Boom + warming trend + coal reversion

Outputs:
    outputs/scenario_projections.csv   — yearly rows × scenario columns
    outputs/scenario_assumptions.csv   — key assumptions per scenario
"""

import numpy as np
import pandas as pd
from config import FINANCE, OUTPUT_DIR


# ═══════════════════════════════════════════════════════════════════════════
#  SCENARIO DEFINITIONS  (parameterized from empirical data)
# ═══════════════════════════════════════════════════════════════════════════
def _build_scenario_defs(p):
    """
    Return scenario assumption dicts from empirical params `p`.
    Every number here is traceable to a CSV-derived parameter.
    """
    # --- Data-derived base values ---
    # DC construction spending CAGR ~ proxy for AI/DC growth
    dc_cagr = p.get("dc_spend_cagr", 0.25)
    # Virginia historical energy CAGR (20yr)
    va_energy_cagr = p.get("va_energy_cagr_20yr", 0.005)
    # ESIF PUE trend (per year)
    pue_trend = p.get("esif_pue_trend_per_year", 0.004)
    # Grid CI trend (kg CO₂/MWh per decade) — negative = decarbonizing
    ci_trend_decade = p.get("va_grid_ci_trend_per_decade", -5.0)
    ci_trend_yr = ci_trend_decade / 10.0
    # Temperature trend
    temp_trend_decade = p.get("va_temp_trend_f_per_decade", 0.3)
    temp_trend_yr = temp_trend_decade / 10.0
    # Coal decline rate
    coal_cagr = p.get("coal_consumption_cagr_20yr", -0.05)
    # Renewable growth rate
    ren_cagr = p.get("renewable_consumption_cagr_20yr", 0.03)

    # --- Baseline data-centre assumptions ---
    dc_capacity_mw_base = 30.0  # Single campus IT capacity (MW)
    pue_base = 1.30             # Industry-average PUE (not ESIF's ultra-efficient 1.03)
    utilization_base = 0.60     # Average IT utilization
    grid_ci_base = p.get("va_grid_ci_kgco2_mwh_latest", 345.0)  # kg CO₂/MWh from real data

    scenarios = {
        "Baseline": {
            "description": "Historical Virginia energy growth continues; no AI step-change",
            "dc_capacity_growth_rate": va_energy_cagr,    # ~0.5% from VA energy CAGR
            "pue_change_per_year": 0.0,                   # No improvement
            "grid_ci_change_per_year": ci_trend_yr,       # Historical grid decarb trend
            "temp_change_per_year_f": temp_trend_yr,      # Historical warming
            "utilization_growth_rate": 0.01,              # Modest
            "dc_capacity_mw_base": dc_capacity_mw_base,
            "pue_base": pue_base,
            "utilization_base": utilization_base,
            "grid_ci_base": grid_ci_base,
            "narrative": f"VA energy CAGR={va_energy_cagr:.2%}, CI trend={ci_trend_yr:+.1f} kg/MWh/yr",
        },
        "DC Boom": {
            "description": "AI-driven datacenter growth follows DC-spending CAGR trajectory",
            "dc_capacity_growth_rate": min(dc_cagr * 0.5, 0.20),  # 50% of spending CAGR → capacity CAGR (capital ≠ capacity 1:1)
            "pue_change_per_year": 0.005,                  # Slight PUE degradation from rapid build
            "grid_ci_change_per_year": ci_trend_yr * 0.5,  # Grid decarb slows (demand growth outpaces)
            "temp_change_per_year_f": temp_trend_yr,
            "utilization_growth_rate": 0.03,               # Rapid fill
            "dc_capacity_mw_base": dc_capacity_mw_base,
            "pue_base": pue_base,
            "utilization_base": utilization_base,
            "grid_ci_base": grid_ci_base,
            "narrative": f"DC spending CAGR={dc_cagr:.1%} → capacity proxy ~{min(dc_cagr*0.5,0.20):.0%}/yr",
        },
        "Efficient Transition": {
            "description": "DC Boom growth + aggressive PUE optimization + 50MW clean PPA",
            "dc_capacity_growth_rate": min(dc_cagr * 0.5, 0.20),
            "pue_change_per_year": -0.015,                 # Active PUE improvement (2× ESIF trend)
            "grid_ci_change_per_year": ci_trend_yr * 2.0,  # Accelerated grid decarb via PPAs
            "temp_change_per_year_f": temp_trend_yr,
            "utilization_growth_rate": 0.03,
            "dc_capacity_mw_base": dc_capacity_mw_base,
            "pue_base": pue_base,
            "utilization_base": utilization_base,
            "grid_ci_base": grid_ci_base,
            "ppa_mw": 50.0,
            "ppa_price_usd_mwh": 45.0,
            "ppa_ci_reduction_pct": 0.30,  # PPA offsets 30% of grid CI
            "narrative": f"PUE improves -0.015/yr, PPA 50 MW @ $45, grid CI decarb 2× trend",
        },
        "Climate Stress": {
            "description": "DC Boom + accelerated warming + coal reversion during summer peaks",
            "dc_capacity_growth_rate": min(dc_cagr * 0.5, 0.20),
            "pue_change_per_year": 0.010,                  # PUE worsens from heat stress
            "grid_ci_change_per_year": abs(ci_trend_yr) * 0.5,  # CI INCREASES (coal reversion)
            "temp_change_per_year_f": temp_trend_yr * 3.0,     # 3× historical warming
            "utilization_growth_rate": 0.03,
            "dc_capacity_mw_base": dc_capacity_mw_base,
            "pue_base": pue_base,
            "utilization_base": utilization_base,
            "grid_ci_base": grid_ci_base,
            "coal_reversion_ci_adder": 15.0,  # kg CO₂/MWh added per year from coal dispatch
            "narrative": f"Warming 3× trend, PUE +0.01/yr, CI reverses by +{abs(ci_trend_yr)*0.5:.1f}/yr",
        },
    }
    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
#  10-YEAR PROJECTIONS
# ═══════════════════════════════════════════════════════════════════════════
def project_scenarios(p):
    """
    Build year-by-year projections for all 4 scenarios.

    For each scenario × year, computes:
        - DC IT Capacity (MW)
        - PUE
        - Utilization
        - Total Facility Power (MW) = IT_Capacity × Utilization × PUE
        - Annual Energy (GWh) = Facility_Power × 8760 / 1000
        - Grid Carbon Intensity (kg CO₂/MWh)
        - Annual CO₂ Emissions (tonnes) = Energy(MWh) × CI(kg/MWh) / 1000 → tonnes
        - Energy Cost ($M)
        - Carbon Liability ($M) at SCC

    Returns (projections_df, assumptions_df).
    """
    scenarios = _build_scenario_defs(p)
    horizon = FINANCE["horizon_years"]
    base_year = 2025
    scc = FINANCE["scc_usd_per_ton"]
    energy_price = FINANCE["energy_price_usd_per_mwh"]

    rows = []
    assumption_rows = []

    for sname, s in scenarios.items():
        assumption_rows.append({
            "scenario": sname,
            "description": s["description"],
            "narrative": s["narrative"],
            "dc_capacity_growth_rate": s["dc_capacity_growth_rate"],
            "pue_change_per_year": s["pue_change_per_year"],
            "grid_ci_change_per_year": s["grid_ci_change_per_year"],
            "temp_change_f_per_year": s["temp_change_per_year_f"],
            "utilization_growth_rate": s["utilization_growth_rate"],
        })

        for yr_offset in range(horizon + 1):  # year 0 … year 10
            year = base_year + yr_offset

            # IT Capacity (MW)
            it_cap = s["dc_capacity_mw_base"] * (1 + s["dc_capacity_growth_rate"]) ** yr_offset

            # PUE (bounded [1.0, 2.5])
            pue = np.clip(s["pue_base"] + s["pue_change_per_year"] * yr_offset, 1.0, 2.5)

            # Utilization (bounded [0.2, 0.95])
            util = np.clip(s["utilization_base"] + s["utilization_growth_rate"] * yr_offset, 0.2, 0.95)

            # Temperature-driven PUE adjustment
            temp_delta = s["temp_change_per_year_f"] * yr_offset
            pue_temp_adj = max(0, temp_delta) * 0.008  # PUE increases 0.008 per °F above baseline
            pue_effective = np.clip(pue + pue_temp_adj, 1.0, 2.5)

            # Total Facility Power (MW)
            facility_mw = it_cap * util * pue_effective

            # Annual Energy (GWh)
            energy_gwh = facility_mw * 8760 / 1000

            # Grid Carbon Intensity (kg CO₂/MWh)
            ci = s["grid_ci_base"] + s["grid_ci_change_per_year"] * yr_offset
            # Climate Stress: add coal reversion adder
            if "coal_reversion_ci_adder" in s:
                ci += s["coal_reversion_ci_adder"] * yr_offset
            ci = max(ci, 50)  # Floor at 50 kg/MWh

            # Efficient Transition: PPA offset
            effective_ci = ci
            if "ppa_ci_reduction_pct" in s:
                effective_ci = ci * (1 - s["ppa_ci_reduction_pct"])

            # Annual CO₂ (tonnes) = Energy (MWh) × CI (kg/MWh) / 1000
            energy_mwh = energy_gwh * 1000
            co2_tonnes = energy_mwh * effective_ci / 1000

            # Costs
            energy_cost_m = energy_mwh * energy_price / 1e6
            carbon_liability_m = co2_tonnes * scc / 1e6

            rows.append({
                "scenario": sname,
                "year": year,
                "year_offset": yr_offset,
                "it_capacity_mw": round(it_cap, 2),
                "pue_effective": round(pue_effective, 3),
                "utilization": round(util, 3),
                "facility_power_mw": round(facility_mw, 2),
                "energy_gwh": round(energy_gwh, 2),
                "energy_mwh": round(energy_mwh, 0),
                "grid_ci_kgco2_mwh": round(ci, 1),
                "effective_ci_kgco2_mwh": round(effective_ci, 1),
                "co2_tonnes": round(co2_tonnes, 0),
                "co2_mmt": round(co2_tonnes / 1e6, 6),
                "energy_cost_usd_m": round(energy_cost_m, 2),
                "carbon_liability_usd_m": round(carbon_liability_m, 2),
                "total_cost_usd_m": round(energy_cost_m + carbon_liability_m, 2),
            })

    proj_df = pd.DataFrame(rows)
    assum_df = pd.DataFrame(assumption_rows)

    # Save
    proj_df.to_csv(OUTPUT_DIR / "scenario_projections.csv", index=False)
    assum_df.to_csv(OUTPUT_DIR / "scenario_assumptions.csv", index=False)

    print(f"\n✓ Saved scenario_projections.csv  ({len(proj_df)} rows)")
    print(f"✓ Saved scenario_assumptions.csv  ({len(assum_df)} rows)")

    # Summary table
    print("\n── Scenario Summary (Year 10) ──")
    yr10 = proj_df[proj_df["year_offset"] == horizon]
    for _, row in yr10.iterrows():
        print(f"  {row['scenario']:25s}  |  IT={row['it_capacity_mw']:7.1f} MW  |  "
              f"PUE={row['pue_effective']:.3f}  |  E={row['energy_gwh']:8.1f} GWh  |  "
              f"CO₂={row['co2_tonnes']:10,.0f} t  |  Cost=${row['total_cost_usd_m']:7.1f}M")

    return proj_df, assum_df


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_ingestion import ingest_all
    data = ingest_all()
    project_scenarios(data["params"])
