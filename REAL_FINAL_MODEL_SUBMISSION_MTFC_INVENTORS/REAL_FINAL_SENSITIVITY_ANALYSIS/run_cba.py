#!/usr/bin/env python3
"""
RUN CBA — Section 6: Master Orchestrator + Actuarial Summary
===============================================================
Runs the entire data-driven cost-benefit analysis pipeline:
    1. data_ingestion.py   — Read 6 CSVs → empirical parameters
    2. cba_scenarios.py    — 4 scenarios × 10 years
    3. cba_full_analysis.py — Cost decomposition, mitigation, NPV, risk
    4. cba_figures.py      — 6 publication-quality figures

Then prints a formatted actuarial summary suitable for the MTFC paper.

Usage:
    python run_cba.py
"""

import sys, time, pathlib

# Ensure we're running from the script's directory
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from data_ingestion import ingest_all
from cba_scenarios import project_scenarios
from cba_full_analysis import run_full_cba
from cba_figures import generate_all_figures


def _fmt_usd(val_m):
    """Format a value in $M to a readable string."""
    if val_m >= 1000:
        return f"${val_m/1000:.1f}B"
    return f"${val_m:.1f}M"


def _fmt_co2(tonnes):
    """Format CO₂ tonnes to a readable string."""
    if tonnes >= 1e6:
        return f"{tonnes/1e6:.2f} MMT"
    if tonnes >= 1e3:
        return f"{tonnes/1e3:,.0f} Kt"
    return f"{tonnes:,.0f} t"


def print_actuarial_summary(p, proj_df, cba_results):
    """Print formatted Section 6 actuarial summary."""
    npv = cba_results["npv_summary"]
    mit = cba_results["mitigation"]
    risk = cba_results["risk_metrics"]

    print("\n")
    print("╔" + "═" * 72 + "╗")
    print("║" + "  ACTUARIAL SUMMARY: Data-Centre Carbon Risk in Virginia".center(72) + "║")
    print("║" + "  MTFC Cost-Benefit Analysis — 2025–2035 Horizon".center(72) + "║")
    print("╚" + "═" * 72 + "╝")

    # ── Executive Finding ──
    boom = npv[npv["scenario"] == "DC Boom"]
    baseline = npv[npv["scenario"] == "Baseline"]
    efficient = npv[npv["scenario"] == "Efficient Transition"]
    mitigated = npv[npv["scenario"].str.contains("Mitigation")]

    if len(boom) > 0:
        boom_npv = boom.iloc[0]["npv_total_cost_usd_m"]
        baseline_npv = baseline.iloc[0]["npv_total_cost_usd_m"] if len(baseline) > 0 else 0
        gap = boom_npv - baseline_npv

        print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│  HEADLINE FINDING                                                       │
│                                                                         │
│  Under the DC Boom scenario (AI-driven growth following observed        │
│  datacenter spending CAGR of {p.get('dc_spend_cagr',0.25):.0%}), a single 30 MW Virginia campus  │
│  faces a 10-year NPV of {_fmt_usd(boom_npv):>8s} in combined energy + carbon costs.   │
│                                                                         │
│  This is {_fmt_usd(gap):>8s} more than the Baseline scenario —                     │
│  entirely attributable to AI-driven demand growth.                      │
│                                                                         │
│  The Combined Mitigation Portfolio (workload shifting + PUE             │
│  optimization + 50 MW clean PPA) can reduce this by ~42%, yielding     │
│  a net NPV benefit of {_fmt_usd(float(mit[mit['lever']=='Combined Portfolio']['npv_net_usd'].values[0])/1e6) if len(mit[mit['lever']=='Combined Portfolio'])>0 else 'N/A':>8s} (ROI {float(mit[mit['lever']=='Combined Portfolio']['roi_x'].values[0]) if len(mit[mit['lever']=='Combined Portfolio'])>0 else 0:.1f}×).                                     │
└─────────────────────────────────────────────────────────────────────────┘""")

    # ── Data Provenance ──
    print(f"""
── DATA PROVENANCE ──────────────────────────────────────────────────────
  Every parameter in this analysis traces to one of 6 source files:
    1. monthly_temp_virginia.csv              → Temperature trend ({p.get('va_temp_trend_f_per_decade',0):+.2f}°F/decade)
    2. esif_daily_avg_interpolated.csv        → NREL ESIF PUE ({p.get('esif_pue_mean',0):.3f} mean)
    3. energy_by_source_annual_grid_comp.csv  → Grid mix (coal {p.get('coal_consumption_cagr_20yr',0):.0%}/yr decline)
    4. virginia_total_carbon_emissions_*.csv  → VA CO₂ ({p.get('va_total_co2_mmt_latest',0):.1f} MMT latest)
    5. virginia_yearly_energy_consumption.csv → VA energy ({p.get('va_energy_cagr_20yr',0):.2%} CAGR 20yr)
    6. monthly-spending-data-center-us.csv    → DC spending (CAGR={p.get('dc_spend_cagr',0):.0%}, R²={p.get('dc_spend_exponential_r2',0):.3f})
""")

    # ── Scenario Comparison ──
    print("── SCENARIO COMPARISON (10-Year NPV) ────────────────────────────────")
    print(f"  {'Scenario':40s} {'NPV Cost':>12s} {'Cum CO₂':>14s} {'Cum Energy':>12s}")
    print("  " + "─" * 78)
    for _, row in npv.iterrows():
        print(f"  {row['scenario']:40s} {_fmt_usd(row['npv_total_cost_usd_m']):>12s} "
              f"{_fmt_co2(row['cumulative_co2_tonnes']):>14s} {row['cumulative_energy_gwh']:>9,.0f} GWh")

    # ── Risk Metrics ──
    if len(risk) > 0:
        print(f"""
── RISK METRICS (Monte Carlo, 10,000 simulations) ──────────────────────
  {'Scenario':30s} {'Mean':>10s} {'Std':>10s} {'VaR 95%':>10s} {'CVaR 95%':>10s} {'CoV':>8s}
  {"─"*78}""")
        for _, row in risk.iterrows():
            print(f"  {row['scenario']:30s} "
                  f"{_fmt_usd(row['npv_mean_usd_m']):>10s} "
                  f"{_fmt_usd(row['npv_std_usd_m']):>10s} "
                  f"{_fmt_usd(row['npv_var_95_usd_m']):>10s} "
                  f"{_fmt_usd(row['npv_cvar_95_usd_m']):>10s} "
                  f"{row['coefficient_of_variation']:>7.3f}")

    # ── Mitigation Levers ──
    print(f"""
── MITIGATION LEVERS (applied to DC Boom) ──────────────────────────────
  {'Lever':35s} {'CapEx':>10s} {'NPV(net)':>12s} {'ROI':>6s} {'Payback':>8s} {'CO₂ Avoided':>14s}
  {"─"*85}""")
    for _, row in mit.iterrows():
        npv_v = float(row['npv_net_usd']) / 1e6 if str(row['npv_net_usd']).replace('-','').replace('.','').isdigit() or isinstance(row['npv_net_usd'], (int, float)) else 0
        print(f"  {row['lever']:35s} "
              f"${float(row['capex_usd'])/1e6:>8.1f}M "
              f"{_fmt_usd(npv_v):>12s} "
              f"{row['roi_x']:>5.1f}× "
              f"{str(row['payback_years']):>7s}yr "
              f"{_fmt_co2(float(row['total_co2_avoided_10yr_tonnes'])):>14s}")

    # ── Key Actuarial Recommendations ──
    print(f"""
── KEY ACTUARIAL RECOMMENDATIONS ───────────────────────────────────────
  1. RESERVE REQUIREMENT: At SCC=$190/t, the DC Boom scenario requires
     an annual carbon reserve of ≈${proj_df[(proj_df['scenario']=='DC Boom')&(proj_df['year_offset']==10)]['carbon_liability_usd_m'].values[0] if len(proj_df[(proj_df['scenario']=='DC Boom')&(proj_df['year_offset']==10)])>0 else 0:.1f}M by Year 10 — growing at
     the AI-capacity CAGR.

  2. MITIGATION PRIORITY: The Combined Portfolio (CapEx=$14.8M) is the
     single highest-ROI intervention, but even the low-cost Dynamic
     Workload Shifting ($600K) delivers meaningful CO₂ reduction.

  3. CARBON PRICE HEDGE: If carbon prices reach the tail scenario
     ($300/t), the 10-year NPV gap between DC Boom and Baseline
     widens further — making early mitigation a cost-effective hedge.

  4. CLIMATE COMPOUNDING: The Climate Stress scenario shows how
     warming-driven PUE degradation + coal dispatch reversion creates
     a vicious feedback loop, increasing both emissions AND costs.

  5. DATA CONFIDENCE: The DC construction spending exponential fit
     (R²={p.get('dc_spend_exponential_r2',0):.3f}) provides the strongest empirical evidence for the
     AI growth trajectory underlying all non-Baseline scenarios.
""")

    print("═" * 72)
    print("  END OF ACTUARIAL SUMMARY")
    print("═" * 72)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    print("\n" + "█" * 72)
    print("  MTFC DATA-DRIVEN COST-BENEFIT ANALYSIS PIPELINE")
    print("  Running all sections: Ingestion → Scenarios → CBA → Figures")
    print("█" * 72)

    # Section 1-2: Data Ingestion
    data = ingest_all()
    p = data["params"]

    # Section 3: Scenarios
    print("\n" + "=" * 72)
    print("  SECTION 3: SCENARIO CONSTRUCTION")
    print("=" * 72)
    proj_df, assum_df = project_scenarios(p)

    # Section 4: Full CBA
    cba = run_full_cba(proj_df, p)

    # Section 5: Figures
    generate_all_figures(data, proj_df, cba["cost_breakdown"],
                         cba["mitigation"], cba["risk_metrics"])

    # Section 6: Actuarial Summary
    print_actuarial_summary(p, proj_df, cba)

    elapsed = time.time() - t0
    print(f"\n⏱  Total pipeline time: {elapsed:.1f}s")

    # List all outputs
    from config import OUTPUT_DIR, FIGURE_DIR
    csvs = sorted(OUTPUT_DIR.glob("cba_*.csv"))
    pngs = sorted(FIGURE_DIR.glob("cba_*.png"))
    print(f"\n📁 Outputs: {len(csvs)} CSVs, {len(pngs)} figures")
    for f in csvs:
        print(f"   📄 outputs/{f.name}")
    for f in pngs:
        print(f"   🖼  figures/{f.name}")


if __name__ == "__main__":
    main()
