#!/usr/bin/env python3
"""
MTFC Sensitivity Analysis – Master Orchestrator
=================================================
Runs every analysis module in sequence and prints a final summary.

Usage:
    python run_all.py
"""

import sys
import time
from pathlib import Path

# Ensure this directory is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUT_DIR, FIGURE_DIR


def banner(text: str):
    w = 60
    print("\n" + "═" * w)
    print(f"  {text}")
    print("═" * w)


def main():
    t0 = time.time()
    banner("MTFC Data-Centre Emissions – Full Sensitivity Suite")
    print(f"  Outputs → {OUTPUT_DIR}")
    print(f"  Figures → {FIGURE_DIR}\n")

    # ── 1. Monte-Carlo ───────────────────────────────────────────────
    banner("1 / 9  Monte-Carlo Simulation")
    import monte_carlo
    mc_df, mc_summary = monte_carlo.run()

    # ── 2. Copula ────────────────────────────────────────────────────
    banner("2 / 9  Copula Tail-Dependency Analysis")
    import copula_analysis
    cop_dep, cop_curves = copula_analysis.run()

    # ── 3. Base Forecast ─────────────────────────────────────────────
    banner("3 / 9  Base Forecasting (Energy, CO₂, AI CapEx, Grid Stress)")
    import base_forecast
    fc_energy, fc_co2, fc_spend, fc_stress, fc_dash = base_forecast.run()

    # ── 4. Energy Forecast ───────────────────────────────────────────
    banner("4 / 9  Energy Forecast & Cost Scenarios")
    import energy_forecast
    scen_df, cost_df = energy_forecast.run()

    # ── 5. Cost-Benefit Analysis ─────────────────────────────────────
    banner("5 / 9  Cost-Benefit & Scenario Monetisation")
    import cost_benefit
    cba_scen, cba_risk, cba_mit, cba_fin, cba_money = cost_benefit.run()

    # ── 6. Sobol Indices ─────────────────────────────────────────────
    banner("6 / 9  Sobol Sensitivity Indices")
    import sobol_analysis
    sob_em, sob_en = sobol_analysis.run()

    # ── 7. Tornado / OAT ─────────────────────────────────────────────
    banner("7 / 9  Tornado / OAT Sensitivity")
    import tornado_analysis
    torn_em, torn_en = tornado_analysis.run()

    # ── 8. Carbon Heatmap ────────────────────────────────────────────
    banner("8 / 9  Carbon Intensity Heatmap")
    import carbon_heatmap
    heatmap_df = carbon_heatmap.run()

    # ── 9. Loss Characterisation ──────────────────────────────────────
    banner("9 / 9  Actuarial Loss Characterisation")
    import loss_characterization
    lc_dist, lc_exc, lc_base, lc_lev, lc_impact = loss_characterization.run(mc_df=mc_df)

    # ── 9. Recommendations ───────────────────────────────────────────
    banner("BONUS  Recommendation Strategies")
    import recommendations
    rec_scen, rec_mit = recommendations.run()

    # ── Summary ──────────────────────────────────────────────────────
    elapsed = time.time() - t0
    banner("PIPELINE COMPLETE")

    csv_count  = len(list(OUTPUT_DIR.glob("*.csv")))
    fig_count  = len(list(FIGURE_DIR.glob("*.png")))

    print(f"\n  ✅ {csv_count} CSV files written to {OUTPUT_DIR.name}/")
    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"      • {f.name}")

    print(f"\n  ✅ {fig_count} figures written to {FIGURE_DIR.name}/")
    for f in sorted(FIGURE_DIR.glob("*.png")):
        print(f"      • {f.name}")

    print(f"\n  ⏱  Total runtime: {elapsed:.1f} s")
    print()


if __name__ == "__main__":
    main()
