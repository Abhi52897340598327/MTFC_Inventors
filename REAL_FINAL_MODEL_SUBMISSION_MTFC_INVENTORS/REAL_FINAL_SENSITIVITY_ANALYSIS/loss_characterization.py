"""
Actuarial Loss Characterization & Before/After Mitigation Projections
======================================================================
Produces decision-grade numerical tables and figures that an actuarial
panel requires:

  1. **Loss Distribution Table** – VaR, TVaR (CVaR), Expected Shortfall,
     standard deviation, skewness, kurtosis at 90/95/99/99.5 percentiles
     for emissions, energy cost, carbon liability, and total cost.
  2. **Loss Exceedance Table** – P(Loss > threshold) for a ladder of
     monetary thresholds ($30M to $60M).
  3. **Before / After Mitigation Year-by-Year Projections** – 10-year
     tables showing annual emissions, energy cost, carbon cost, and
     total cost under (a) no-action baseline and (b) each mitigation
     lever, for the Moderate growth scenario.
  4. **Mitigation Impact Summary** – tabulates cumulative 10-yr savings
     in emissions (tonnes), energy ($), carbon ($), total ($), and NPV.
  5. **Figures**:
     - Loss exceedance curve (S-curve) for total annual cost
     - Before/after emissions trajectory comparison
     - Before/after total cost trajectory comparison
     - Stacked cost-component comparison (before vs combined portfolio)

Outputs
-------
CSV:
  - loss_distribution_table.csv
  - loss_exceedance_table.csv
  - mitigation_projection_baseline.csv
  - mitigation_projection_by_lever.csv
  - mitigation_impact_summary.csv

Figures:
  - figures/loss_exceedance_curve.png
  - figures/mitigation_emissions_trajectory.png
  - figures/mitigation_cost_trajectory.png
  - figures/mitigation_cost_components.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from config import (DC_PARAMS, GRID, FINANCE, MITIGATION_LEVERS,
                    OUTPUT_DIR, FIGURE_DIR, PLOT)


# ═════════════════════════════════════════════════════════════════════════
#  1.  NORMALISE MC DATAFRAME  (accepts output of monte_carlo.simulate())
# ═════════════════════════════════════════════════════════════════════════

def _normalise_mc(mc_df: pd.DataFrame | None) -> pd.DataFrame:
    """Accept the MC DataFrame from the orchestrator (same population as
    every other module) and rename columns to the canonical names used
    by the loss tables.  Falls back to reading the CSV on disk."""
    if mc_df is None:
        mc_df = pd.read_csv(OUTPUT_DIR / "monte_carlo_results.csv")

    # Map monte_carlo.py column names → canonical names
    rename = {
        "electricity_cost":   "energy_cost",
        "carbon_cost_central": "carbon_cost",
    }
    mc = mc_df.rename(columns=rename)

    # Ensure the columns we need exist
    required = ["annual_emissions_tons", "energy_cost", "carbon_cost",
                "peak_penalty", "total_annual_cost"]
    for col in required:
        if col not in mc.columns:
            raise KeyError(f"MC DataFrame missing required column: {col}")
    return mc[required]


# ═════════════════════════════════════════════════════════════════════════
#  2.  LOSS DISTRIBUTION TABLE
# ═════════════════════════════════════════════════════════════════════════

def build_loss_distribution(mc: pd.DataFrame) -> pd.DataFrame:
    """Full actuarial loss characterization at multiple confidence levels."""
    quantiles = [0.50, 0.75, 0.90, 0.95, 0.99, 0.995]
    metrics = [
        ("Annual Emissions (tonnes CO₂)", "annual_emissions_tons"),
        ("Energy Procurement Cost ($)",   "energy_cost"),
        ("Carbon Liability – SCC ($)",    "carbon_cost"),
        ("Peak Breach Penalty ($)",       "peak_penalty"),
        ("Total Annual Cost ($)",         "total_annual_cost"),
    ]
    rows = []
    for label, col in metrics:
        v = mc[col].values
        row = {
            "loss_category": label,
            "mean": round(v.mean(), 0),
            "std_dev": round(v.std(), 0),
            "coeff_of_variation": round(v.std() / v.mean(), 4) if v.mean() != 0 else 0,
            "skewness": round(float(sp_stats.skew(v)), 4),
            "kurtosis": round(float(sp_stats.kurtosis(v)), 4),
            "minimum": round(v.min(), 0),
        }
        for q in quantiles:
            # Consistent label: "50", "75", … "99.5"
            pctl_str = f"{q*100:g}"   # e.g. "99.5", "95", "50"
            var_val = np.percentile(v, q * 100)
            # TVaR / CVaR = E[X | X >= VaR]
            tail = v[v >= var_val]
            tvar_val = tail.mean() if len(tail) > 0 else var_val
            row[f"VaR_{pctl_str}"] = round(var_val, 0)
            row[f"TVaR_{pctl_str}"] = round(tvar_val, 0)
        row["maximum"] = round(v.max(), 0)
        rows.append(row)
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════
#  3.  LOSS EXCEEDANCE TABLE
# ═════════════════════════════════════════════════════════════════════════

def build_loss_exceedance(mc: pd.DataFrame) -> pd.DataFrame:
    """P(Total Cost > threshold) for a ladder of thresholds."""
    cost = mc["total_annual_cost"].values
    n = len(cost)
    # Build thresholds from median to 99.9th pctl (actuarially useful tail)
    lo = np.percentile(cost, 50)
    hi = np.percentile(cost, 99.9)
    thresholds = np.linspace(lo, hi, 25)
    rows = []
    for t in thresholds:
        exceed = (cost > t).sum()
        rows.append({
            "threshold_usd": round(t, 0),
            "threshold_usd_millions": round(t / 1e6, 2),
            "n_exceedances": exceed,
            "prob_exceedance": round(exceed / n, 6),
            "return_period_years": round(n / exceed, 1) if exceed > 0 else float("inf"),
        })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════
#  4.  BEFORE / AFTER MITIGATION PROJECTIONS (10-year)
# ═════════════════════════════════════════════════════════════════════════

_GROWTH_RATE     = 0.15   # Moderate growth scenario
_PUE_IMPROVE     = 0.015  # Annual PUE improvement
_CI_IMPROVE      = 0.03   # Annual grid CI improvement
_EFFICACY_DECAY  = 0.02   # Mitigation efficacy decays 2 %/yr (BUG #9 fix)
_PEAK_BREACH_P   = 0.35   # Prob of hitting peak in any year (deterministic est.)
_BASE_TOTAL_MW   = DC_PARAMS["it_capacity_mw"] * DC_PARAMS["pue_baseline"]

def _project_year(yr: int, co2_red: float = 0.0, nrg_red: float = 0.0):
    """Project year-*yr* values with optional mitigation reductions.

    Fixes applied:
      - BUG #8:  Cooling maintenance scales with total_mw / baseline.
      - BUG #9:  Mitigation efficacy decays by 2 %/yr.
      - BUG #10: Peak breach penalty included as expected annual value.
    """
    it_mw = DC_PARAMS["it_capacity_mw"] * (1 + _GROWTH_RATE) ** yr
    pue_yr = max(DC_PARAMS["pue_baseline"] * (1 - _PUE_IMPROVE) ** yr, 1.05)
    total_mw = it_mw * pue_yr
    annual_mwh = total_mw * 8760
    ci_yr = GRID["carbon_intensity_mean"] * (1 - _CI_IMPROVE) ** yr

    # ── Before mitigation ────────────────────────────────────────
    annual_tons = total_mw * ci_yr * 8760 / 1000
    energy_cost = annual_mwh * FINANCE["energy_price_usd_per_mwh"]
    carbon_cost = annual_tons * FINANCE["scc_usd_per_ton"]
    demand_chg  = FINANCE["demand_charge_usd_per_kw_yr"] * total_mw * 1000

    # BUG #8 fix: scale cooling with facility size
    cooling = FINANCE["cooling_maintenance_annual"] * (total_mw / _BASE_TOTAL_MW)

    # BUG #10 fix: deterministic expected peak breach penalty
    breach_mw = max(total_mw * 1.075 - FINANCE["contract_peak_mw"], 0)
    peak_penalty = (_PEAK_BREACH_P * breach_mw
                    * FINANCE["peak_breach_penalty_usd_per_mwh"]
                    * FINANCE["peak_breach_hours"])

    # ── After mitigation (with BUG #9 efficacy decay) ────────────
    eff_factor = (1 - _EFFICACY_DECAY) ** (yr - 1)   # yr 1 → full, yr 10 → 83 %
    co2_eff = co2_red * eff_factor
    nrg_eff = nrg_red * eff_factor

    mit_energy = energy_cost * (1 - nrg_eff)
    mit_carbon = carbon_cost * (1 - co2_eff)
    mit_tons   = annual_tons * (1 - co2_eff)

    total_before = energy_cost + carbon_cost + demand_chg + cooling + peak_penalty
    total_after  = mit_energy + mit_carbon + demand_chg + cooling + peak_penalty

    return {
        "year": yr,
        "it_capacity_mw": round(it_mw, 1),
        "pue": round(pue_yr, 4),
        "total_power_mw": round(total_mw, 1),
        "carbon_intensity": round(ci_yr, 1),
        # Before
        "emissions_tons_before": round(annual_tons, 0),
        "energy_cost_before": round(energy_cost, 0),
        "carbon_cost_before": round(carbon_cost, 0),
        "demand_charge": round(demand_chg, 0),
        "cooling_maint": round(cooling, 0),
        "peak_penalty": round(peak_penalty, 0),
        "total_cost_before": round(total_before, 0),
        # After
        "emissions_tons_after": round(mit_tons, 0),
        "energy_cost_after": round(mit_energy, 0),
        "carbon_cost_after": round(mit_carbon, 0),
        "total_cost_after": round(total_after, 0),
        # Savings
        "emissions_saved_tons": round(annual_tons - mit_tons, 0),
        "cost_saved_usd": round(total_before - total_after, 0),
    }


def build_projection_tables():
    """Build baseline + per-lever 10-year projection tables."""
    horizon = FINANCE["horizon_years"]
    d = FINANCE["discount_rate"]

    # Baseline (no mitigation)
    baseline_rows = []
    for yr in range(1, horizon + 1):
        r = _project_year(yr, co2_red=0, nrg_red=0)
        r["lever"] = "No Action (Baseline)"
        baseline_rows.append(r)
    baseline_df = pd.DataFrame(baseline_rows)

    # Per-lever projections
    lever_rows = []
    for name, lev in MITIGATION_LEVERS.items():
        co2_red = lev["emission_reduction_pct"] / 100.0
        nrg_red = lev.get("energy_saving_pct", 0.0) / 100.0
        for yr in range(1, horizon + 1):
            r = _project_year(yr, co2_red=co2_red, nrg_red=nrg_red)
            r["lever"] = name
            lever_rows.append(r)
    lever_df = pd.DataFrame(lever_rows)

    return baseline_df, lever_df


# ═════════════════════════════════════════════════════════════════════════
#  5.  MITIGATION IMPACT SUMMARY TABLE
# ═════════════════════════════════════════════════════════════════════════

def build_impact_summary(baseline_df: pd.DataFrame,
                         lever_df: pd.DataFrame) -> pd.DataFrame:
    """One-row-per-lever summary: cumulative 10yr savings + NPV."""
    d = FINANCE["discount_rate"]
    rows = []

    # Baseline cumulative
    base_cum_emit = baseline_df["emissions_tons_before"].sum()
    base_cum_energy = baseline_df["energy_cost_before"].sum()
    base_cum_carbon = baseline_df["carbon_cost_before"].sum()
    base_cum_total = baseline_df["total_cost_before"].sum()
    base_npv = sum(
        baseline_df.loc[baseline_df["year"] == yr, "total_cost_before"].values[0]
        * (1 + d) ** (-yr)
        for yr in range(1, FINANCE["horizon_years"] + 1))

    rows.append({
        "lever": "No Action (Baseline)",
        "capex_usd": 0,
        "cumul_emissions_tons": round(base_cum_emit, 0),
        "cumul_energy_cost": round(base_cum_energy, 0),
        "cumul_carbon_cost": round(base_cum_carbon, 0),
        "cumul_total_cost": round(base_cum_total, 0),
        "npv_total_cost": round(base_npv, 0),
        "emissions_avoided_tons": 0,
        "total_cost_saved": 0,
        "npv_savings": 0,
        "pct_emission_reduction": 0.0,
        "pct_cost_reduction": 0.0,
    })

    for name, lev in MITIGATION_LEVERS.items():
        sub = lever_df[lever_df["lever"] == name]
        cum_emit = sub["emissions_tons_after"].sum()
        cum_energy = sub["energy_cost_after"].sum()
        cum_carbon = sub["carbon_cost_after"].sum()
        cum_total = sub["total_cost_after"].sum()
        capex = lev["capex_usd"]
        lev_npv = sum(
            sub.loc[sub["year"] == yr, "total_cost_after"].values[0]
            * (1 + d) ** (-yr)
            for yr in range(1, FINANCE["horizon_years"] + 1))

        cum_saved = base_cum_total - cum_total
        emit_avoided = base_cum_emit - cum_emit
        npv_saved = base_npv - lev_npv - capex  # net of CapEx

        rows.append({
            "lever": name,
            "capex_usd": capex,
            "cumul_emissions_tons": round(cum_emit, 0),
            "cumul_energy_cost": round(cum_energy, 0),
            "cumul_carbon_cost": round(cum_carbon, 0),
            "cumul_total_cost": round(cum_total, 0),
            "npv_total_cost": round(lev_npv, 0),
            "emissions_avoided_tons": round(emit_avoided, 0),
            "total_cost_saved": round(cum_saved, 0),
            "npv_savings": round(npv_saved, 0),
            "pct_emission_reduction": round(emit_avoided / base_cum_emit * 100, 2),
            "pct_cost_reduction": round(cum_saved / base_cum_total * 100, 2),
        })

    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════
#  6.  FIGURES
# ═════════════════════════════════════════════════════════════════════════

def plot_loss_exceedance(exc_df: pd.DataFrame):
    """S-curve: P(cost > threshold) vs threshold, with return-period axis."""
    fig, ax = plt.subplots(figsize=(12, 7))
    x = exc_df["threshold_usd_millions"]
    y = exc_df["prob_exceedance"]
    ax.plot(x, y, "o-", color="#8e44ad", lw=2.5, ms=5, zorder=5)
    ax.fill_between(x, y, alpha=0.12, color="#8e44ad")

    # Annotate key quantiles with VaR intersection lines
    for q, c, ls, label in [
        (0.10, "#e74c3c", "--", "VaR₉₀"),
        (0.05, "#e67e22", "-.", "VaR₉₅"),
        (0.01, "#c0392b", ":", "VaR₉₉"),
    ]:
        ax.axhline(q, color=c, ls=ls, lw=1.5, alpha=0.7)
        # Find the threshold at this exceedance probability
        idx_close = np.argmin(np.abs(y.values - q))
        threshold_val = x.values[idx_close]
        ax.plot(threshold_val, q, "D", color=c, ms=8, zorder=10)
        ax.vlines(threshold_val, 0, q, color=c, ls=":", alpha=0.5)
        ax.text(threshold_val + 0.3, q + 0.015,
                f"{label} = ${threshold_val:.1f}M\np = {q:.0%}",
                fontsize=9, color=c, fontweight="bold")

    ax.set_xlabel("Total Annual Cost Threshold ($M)", fontsize=12)
    ax.set_ylabel("Probability of Exceedance", fontsize=12)
    ax.set_title("Loss Exceedance Curve – Total Annual Cost",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(-0.02, max(y) * 1.08)
    ax.grid(True, alpha=0.3)

    # Return period secondary axis
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    yticks = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    ax2.set_yticks(yticks)
    ax2.set_yticklabels([f"1-in-{1/p:.0f} yr" for p in yticks], fontsize=8, color="grey")
    ax2.set_ylabel("Return Period", fontsize=10, color="grey")

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "loss_exceedance_curve.png",
                dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close()


def plot_mitigation_trajectories(baseline_df: pd.DataFrame,
                                  lever_df: pd.DataFrame):
    """Combined 1×2 panel: emissions (left) and cost (right) before/after mitigation."""
    fig, (ax_em, ax_cost) = plt.subplots(1, 2, figsize=(18, 7))
    yrs = baseline_df["year"]
    colors = ["#27ae60", "#2980b9", "#f39c12", "#8e44ad"]
    dashes = [(5, 2), (3, 2, 1, 2), (1, 1), (8, 3)]

    # ── Left panel: Emissions ──
    ax_em.plot(yrs, baseline_df["emissions_tons_before"] / 1000,
               "k-o", lw=2.5, ms=7, label="No Action (Baseline)", zorder=5)
    for i, (name, _) in enumerate(MITIGATION_LEVERS.items()):
        sub = lever_df[lever_df["lever"] == name]
        ax_em.plot(sub["year"], sub["emissions_tons_after"] / 1000,
                   "s-", lw=2, ms=6, color=colors[i % len(colors)],
                   dashes=dashes[i % len(dashes)],
                   label=f"After: {name}", alpha=0.85)
    # Annotate final-year % reduction for Combined Portfolio
    comb_em = lever_df[lever_df["lever"] == "Combined Portfolio"]
    if not comb_em.empty:
        last_base = baseline_df["emissions_tons_before"].iloc[-1]
        last_mit = comb_em["emissions_tons_after"].iloc[-1]
        pct = (last_base - last_mit) / last_base * 100
        ax_em.annotate(f"–{pct:.0f}%", xy=(yrs.iloc[-1], last_mit / 1000),
                       fontsize=11, fontweight="bold", color="#8e44ad",
                       xytext=(5, 10), textcoords="offset points")
    ax_em.set_xlabel("Year", fontsize=12)
    ax_em.set_ylabel("Annual Emissions (thousand tonnes CO₂)", fontsize=12)
    ax_em.set_title("(a) Emissions Trajectory", fontsize=13, fontweight="bold")
    ax_em.legend(fontsize=8, loc="upper left")
    ax_em.grid(True, alpha=0.3)

    # ── Right panel: Cost ──
    ax_cost.plot(yrs, baseline_df["total_cost_before"] / 1e6,
                 "k-o", lw=2.5, ms=7, label="No Action (Baseline)", zorder=5)
    for i, (name, _) in enumerate(MITIGATION_LEVERS.items()):
        sub = lever_df[lever_df["lever"] == name]
        ax_cost.plot(sub["year"], sub["total_cost_after"] / 1e6,
                     "s-", lw=2, ms=6, color=colors[i % len(colors)],
                     dashes=dashes[i % len(dashes)],
                     label=f"After: {name}", alpha=0.85)
    # Annotate final-year % savings for Combined Portfolio
    comb_cost = lever_df[lever_df["lever"] == "Combined Portfolio"]
    if not comb_cost.empty:
        last_base_c = baseline_df["total_cost_before"].iloc[-1]
        last_mit_c = comb_cost["total_cost_after"].iloc[-1]
        pct_c = (last_base_c - last_mit_c) / last_base_c * 100
        ax_cost.annotate(f"–{pct_c:.0f}%", xy=(yrs.iloc[-1], last_mit_c / 1e6),
                         fontsize=11, fontweight="bold", color="#8e44ad",
                         xytext=(5, 10), textcoords="offset points")
    ax_cost.set_xlabel("Year", fontsize=12)
    ax_cost.set_ylabel("Total Annual Cost ($M)", fontsize=12)
    ax_cost.set_title("(b) Cost Trajectory", fontsize=13, fontweight="bold")
    ax_cost.legend(fontsize=8, loc="upper left")
    ax_cost.grid(True, alpha=0.3)

    plt.suptitle("Projected Emissions & Cost: Before vs After Mitigation (Moderate Growth)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "mitigation_emissions_and_cost.png",
                dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close()


def plot_cost_components(baseline_df: pd.DataFrame,
                         lever_df: pd.DataFrame):
    """Side-by-side stacked bar: cost components before vs Combined Portfolio.
    Each bar segment shows its % share of total cost for that year."""
    combined = lever_df[lever_df["lever"] == "Combined Portfolio"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    comp_colors = {"Energy": "#3498db", "Carbon (SCC)": "#e74c3c",
                   "Demand Charges": "#f39c12", "Cooling/Maint": "#95a5a6",
                   "Peak Breach": "#e67e22"}

    for ax, df_src, title, suffix in [
        (axes[0], baseline_df, "Before Mitigation (Baseline)", "before"),
        (axes[1], combined, "After: Combined Portfolio", "after"),
    ]:
        yrs = df_src["year"].values
        e_key = f"energy_cost_{suffix}"
        c_key = f"carbon_cost_{suffix}"

        e = df_src[e_key].values / 1e6
        c = df_src[c_key].values / 1e6
        d_vals = df_src["demand_charge"].values / 1e6
        cl = df_src["cooling_maint"].values / 1e6
        pk = df_src["peak_penalty"].values / 1e6

        totals = e + c + d_vals + cl + pk

        ax.bar(yrs, e, color=comp_colors["Energy"], label="Energy", width=0.6)
        ax.bar(yrs, c, bottom=e, color=comp_colors["Carbon (SCC)"],
               label="Carbon (SCC)", width=0.6)
        ax.bar(yrs, d_vals, bottom=e + c, color=comp_colors["Demand Charges"],
               label="Demand Charges", width=0.6)
        ax.bar(yrs, cl, bottom=e + c + d_vals, color=comp_colors["Cooling/Maint"],
               label="Cooling/Maint", width=0.6)
        ax.bar(yrs, pk, bottom=e + c + d_vals + cl,
               color=comp_colors["Peak Breach"], label="Peak Breach", width=0.6)

        # Add % labels on the two largest components for first & last years
        for yi in [0, len(yrs) - 1]:
            comps = [e[yi], c[yi], d_vals[yi], cl[yi], pk[yi]]
            bottoms = [0, e[yi], e[yi] + c[yi],
                       e[yi] + c[yi] + d_vals[yi],
                       e[yi] + c[yi] + d_vals[yi] + cl[yi]]
            for ci_idx in range(len(comps)):
                pct = comps[ci_idx] / totals[yi] * 100 if totals[yi] > 0 else 0
                if pct >= 15:  # Only label segments ≥ 15%
                    mid_y = bottoms[ci_idx] + comps[ci_idx] / 2
                    ax.text(yrs[yi], mid_y, f"{pct:.0f}%",
                            ha="center", va="center", fontsize=7,
                            fontweight="bold", color="white")

        ax.set_xlabel("Year", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    axes[0].set_ylabel("Annual Cost ($M)", fontsize=12)
    plt.suptitle("Cost Component Breakdown: Before vs After Mitigation",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "mitigation_cost_components.png",
                dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close()


# ═════════════════════════════════════════════════════════════════════════
#  7.  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════

def run(mc_df: pd.DataFrame | None = None):
    """Run loss characterisation.

    Parameters
    ----------
    mc_df : DataFrame, optional
        The Monte-Carlo results DataFrame produced by monte_carlo.run().
        If *None*, falls back to reading ``monte_carlo_results.csv``.
        Using the same population ensures numerical consistency across
        all modules.
    """
    print("  ▶ Running actuarial loss characterization …")

    # Reuse the MC population from the orchestrator (BUG #6 fix)
    mc = _normalise_mc(mc_df)

    # Tables
    loss_dist = build_loss_distribution(mc)
    loss_exc  = build_loss_exceedance(mc)
    baseline_df, lever_df = build_projection_tables()
    impact_df = build_impact_summary(baseline_df, lever_df)

    # Save CSVs
    loss_dist.to_csv(OUTPUT_DIR / "loss_distribution_table.csv", index=False)
    loss_exc.to_csv(OUTPUT_DIR / "loss_exceedance_table.csv", index=False)
    baseline_df.to_csv(OUTPUT_DIR / "mitigation_projection_baseline.csv", index=False)
    lever_df.to_csv(OUTPUT_DIR / "mitigation_projection_by_lever.csv", index=False)
    impact_df.to_csv(OUTPUT_DIR / "mitigation_impact_summary.csv", index=False)

    # Figures
    plot_loss_exceedance(loss_exc)
    plot_mitigation_trajectories(baseline_df, lever_df)
    plot_cost_components(baseline_df, lever_df)

    # Console summary
    print("    ✓ Loss distribution table (5 categories × 6 confidence levels)")
    tc = mc["total_annual_cost"]
    print(f"      Mean total cost:  ${tc.mean()/1e6:,.1f}M")
    print(f"      Std dev:          ${tc.std()/1e6:,.1f}M")
    print(f"      VaR 95%:          ${np.percentile(tc, 95)/1e6:,.1f}M")
    print(f"      TVaR 99%:         ${tc[tc >= np.percentile(tc, 99)].mean()/1e6:,.1f}M")

    print("    ✓ Loss exceedance curve (25-point threshold ladder)")

    print("    ✓ Before/after mitigation projections (10 yr × 5 levers)")
    for _, r in impact_df.iterrows():
        if r["lever"] == "No Action (Baseline)":
            print(f"      Baseline 10yr: {r['cumul_emissions_tons']:,.0f} t CO₂, "
                  f"${r['cumul_total_cost']/1e6:,.0f}M total cost")
        else:
            print(f"      {r['lever']:30s}  "
                  f"saved {r['emissions_avoided_tons']:,.0f} t "
                  f"({r['pct_emission_reduction']:.1f}%), "
                  f"NPV savings ${r['npv_savings']/1e6:,.1f}M")

    return loss_dist, loss_exc, baseline_df, lever_df, impact_df


if __name__ == "__main__":
    run()
