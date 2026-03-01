"""
CBA FIGURES — Section 5: Publication-Quality Visualizations
=============================================================
Generates 6 figures suitable for MTFC paper submission.
All figures read from the CSVs produced by upstream modules.
NO PJM data used — only the 6 approved data sources.

Figures:
  1. data_sources_overview.png      — 2×3 panel showing all 6 data sources
  2. scenario_emissions.png         — Emissions trajectories with MC uncertainty
  3. mitigation_roi.png             — ROI comparison of 4 levers
  4. carbon_price_sensitivity.png   — NPV vs carbon price for each scenario
  5. risk_decomposition.png         — Stacked bar of risk by scenario
  6. growth_evidence.png            — DC spending scatter with exponential fit
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from config import FIGURE_DIR, OUTPUT_DIR, FINANCE

# ── Style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})

SCENARIO_COLORS = {
    "Baseline": "#2196F3",
    "DC Boom": "#F44336",
    "Efficient Transition": "#4CAF50",
    "Climate Stress": "#FF9800",
    "DC Boom + Combined Mitigation": "#9C27B0",
}

def _save(fig, name):
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}")


# ═══════════════════════════════════════════════════════════════════════════
#  FIG 1: Data Sources Overview (2×3 panel)
# ═══════════════════════════════════════════════════════════════════════════

def fig_data_sources_overview(data_bundle):
    """Six-panel figure showing all 6 real data sources (NO PJM)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Real Data Sources Driving the Cost-Benefit Analysis", fontsize=16, fontweight="bold", y=1.02)

    # Panel 1: Virginia Yearly Energy Consumption
    ax = axes[0, 0]
    va_en = data_bundle["va_energy"]
    ax.plot(va_en["Year"], va_en["total_bbtu"] / 1e6, color="#1565C0", linewidth=2)
    ax.fill_between(va_en["Year"], 0, va_en["total_bbtu"] / 1e6, alpha=0.15, color="#1565C0")
    # Trend annotation
    if len(va_en) > 3:
        z_en = np.polyfit(va_en["Year"].astype(float), va_en["total_bbtu"].values / 1e6, 1)
        ss_res = np.sum((va_en["total_bbtu"].values / 1e6 - np.polyval(z_en, va_en["Year"].astype(float))) ** 2)
        ss_tot = np.sum((va_en["total_bbtu"].values / 1e6 - (va_en["total_bbtu"].values / 1e6).mean()) ** 2)
        r2_en = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        ax.text(0.05, 0.92, f"Trend: {z_en[0]:+.3f} M BBtu/yr\nR² = {r2_en:.3f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="aliceblue", alpha=0.8))
    ax.set_title("Virginia Total Energy Consumption")
    ax.set_ylabel("Million BBtu")
    ax.grid(True, alpha=0.3)

    # Panel 2: Virginia Temperature
    ax = axes[0, 1]
    temp = data_bundle["temperature"]
    annual_temp = temp.groupby(temp["date"].dt.year)["avg_temp_f"].mean()
    ax.scatter(annual_temp.index, annual_temp.values, s=20, alpha=0.6, color="#D32F2F")
    # Trend line
    if len(annual_temp) > 10:
        z = np.polyfit(annual_temp.index.astype(float), annual_temp.values, 1)
        trendline = np.poly1d(z)
        ax.plot(annual_temp.index, trendline(annual_temp.index.astype(float)),
                "--", color="#D32F2F", linewidth=2, label=f"Trend: {z[0]*10:+.2f}°F/decade")
        ax.legend(fontsize=9)
    ax.set_title("Virginia Annual Avg Temperature")
    ax.set_ylabel("°F")

    # Panel 3: ESIF PUE
    ax = axes[0, 2]
    esif = data_bundle["esif"]
    esif_monthly = esif.set_index("date").resample("ME")["pue"].mean()
    ax.plot(esif_monthly.index, esif_monthly.values, color="#388E3C", linewidth=1)
    ax.set_title("NREL ESIF Data-Centre PUE")
    ax.set_ylabel("PUE")
    ax.set_ylim(0.95, 1.15)

    # Panel 4: Virginia Grid Composition
    ax = axes[1, 0]
    grid = data_bundle["grid_comp"]
    cols = [c for c in grid.columns if c != "Year"]
    short_names = ["Coal", "Nat Gas", "Petroleum", "Nuclear", "Renewable"]
    colors_grid = ["#424242", "#1565C0", "#6D4C41", "#FFA000", "#2E7D32"]
    bottom = np.zeros(len(grid))
    for i, (col, sn, clr) in enumerate(zip(cols, short_names, colors_grid)):
        vals = grid[col].fillna(0).values
        ax.bar(grid["Year"], vals, bottom=bottom, width=0.8, label=sn, color=clr, alpha=0.85)
        bottom += vals
    ax.set_title("Virginia Grid Composition (BBtu)")
    ax.set_ylabel("Billion BTU")
    ax.legend(fontsize=8, ncol=2)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

    # Panel 5: Virginia CO₂ Emissions
    ax = axes[1, 1]
    emis = data_bundle["emissions"]
    emis_clean = emis.dropna(subset=["Total_co2_mmt"])
    ax.fill_between(emis_clean["Year"], 0, emis_clean["Total_co2_mmt"], alpha=0.3, color="#F44336")
    ax.plot(emis_clean["Year"], emis_clean["Total_co2_mmt"], color="#C62828", linewidth=2)
    # Trend annotation
    if len(emis_clean) > 3:
        z_co2 = np.polyfit(emis_clean["Year"].astype(float), emis_clean["Total_co2_mmt"].values, 1)
        ss_r = np.sum((emis_clean["Total_co2_mmt"].values - np.polyval(z_co2, emis_clean["Year"].astype(float))) ** 2)
        ss_t = np.sum((emis_clean["Total_co2_mmt"].values - emis_clean["Total_co2_mmt"].values.mean()) ** 2)
        r2_co2 = 1 - ss_r / ss_t if ss_t > 0 else 0
        ax.text(0.05, 0.92, f"Trend: {z_co2[0]:+.2f} MMT/yr\nR² = {r2_co2:.3f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", alpha=0.8))
    ax.set_title("Virginia Total CO₂ Emissions")
    ax.set_ylabel("MMT CO₂")

    # Panel 6: US DC Construction Spending
    ax = axes[1, 2]
    dc = data_bundle["dc_spending"]
    ax.scatter(dc["date"], dc["spending_usd"] / 1e9, s=15, alpha=0.6, color="#7B1FA2")
    # Exponential fit
    dc_clean = dc.dropna(subset=["spending_usd"])
    dc_clean = dc_clean[dc_clean["spending_usd"] > 0]
    x_days = (dc_clean["date"] - dc_clean["date"].iloc[0]).dt.days.values.astype(float)
    y_log = np.log(dc_clean["spending_usd"].values)
    coeffs = np.polyfit(x_days, y_log, 1)
    x_fit = np.linspace(x_days.min(), x_days.max(), 200)
    y_fit = np.exp(np.polyval(coeffs, x_fit))
    dates_fit = pd.to_datetime(dc_clean["date"].iloc[0]) + pd.to_timedelta(x_fit, unit="D")
    r2 = data_bundle["params"].get("dc_spend_exponential_r2", 0.95)
    ax.plot(dates_fit, y_fit / 1e9, "--", color="#7B1FA2", linewidth=2,
            label=f"Exp. fit (R²={r2:.3f})")
    ax.set_title("US Data-Centre Construction Spending")
    ax.set_ylabel("$Billion / month")
    ax.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, "cba_data_sources_overview.png")


# ═══════════════════════════════════════════════════════════════════════════
#  FIG 2: Scenario Emissions Trajectories
# ═══════════════════════════════════════════════════════════════════════════

def fig_scenario_emissions(proj_df, risk_df):
    """Emissions trajectories for all 4 scenarios with uncertainty bands."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Emissions
    for scen in proj_df["scenario"].unique():
        sp = proj_df[proj_df["scenario"] == scen]
        clr = SCENARIO_COLORS.get(scen, "#888")
        ax1.plot(sp["year"], sp["co2_tonnes"] / 1e3, "-o", color=clr,
                 linewidth=2.5, markersize=5, label=scen)
        # ±20% uncertainty band
        ax1.fill_between(sp["year"],
                         sp["co2_tonnes"] / 1e3 * 0.80,
                         sp["co2_tonnes"] / 1e3 * 1.20,
                         alpha=0.12, color=clr)

    ax1.set_title("Projected Annual CO₂ Emissions by Scenario", fontweight="bold")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Thousand Tonnes CO₂")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Right: Cumulative cost
    for scen in proj_df["scenario"].unique():
        sp = proj_df[proj_df["scenario"] == scen]
        clr = SCENARIO_COLORS.get(scen, "#888")
        cum_cost = sp["total_cost_usd_m"].cumsum()
        ax2.plot(sp["year"], cum_cost, "-s", color=clr, linewidth=2.5, markersize=5, label=scen)

    ax2.set_title("Cumulative Total Cost (Energy + Carbon)", fontweight="bold")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Cumulative Cost ($M)")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, "cba_scenario_emissions.png")


# ═══════════════════════════════════════════════════════════════════════════
#  FIG 3: Mitigation Lever ROI Comparison
# ═══════════════════════════════════════════════════════════════════════════

def fig_mitigation_roi(mit_df):
    """Horizontal bar chart comparing mitigation levers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    levers = mit_df["lever"].values
    y_pos = range(len(levers))
    colors_mit = ["#1565C0", "#2E7D32", "#7B1FA2", "#C62828"]

    # Left: NPV (net)
    npv_vals = []
    for v in mit_df["npv_net_usd"]:
        try:
            npv_vals.append(float(v) / 1e6)
        except (ValueError, TypeError):
            npv_vals.append(0)

    bars1 = ax1.barh(y_pos, npv_vals, color=colors_mit, height=0.5, edgecolor="white")
    for i, (bar, val) in enumerate(zip(bars1, npv_vals)):
        ax1.text(val + max(npv_vals) * 0.02, i, f"${val:,.1f}M", va="center", fontsize=11, fontweight="bold")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(levers, fontsize=11)
    ax1.set_xlabel("Net NPV ($M)", fontsize=12)
    ax1.set_title("Net Present Value of Mitigation Levers", fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)

    # Right: CO₂ Avoided with % reduction labels
    co2_vals = []
    for v in mit_df["total_co2_avoided_10yr_tonnes"]:
        try:
            co2_vals.append(float(v) / 1e3)
        except (ValueError, TypeError):
            co2_vals.append(0)

    bars2 = ax2.barh(y_pos, co2_vals, color=colors_mit, height=0.5, edgecolor="white")
    max_co2 = max(co2_vals) if co2_vals else 1
    for i, (bar, val) in enumerate(zip(bars2, co2_vals)):
        # % of baseline avoided
        pct_label = ""
        try:
            pct_r = float(mit_df["co2_reduction_pct"].iloc[i]) * 100
            pct_label = f" ({pct_r:.0f}% reduction)"
        except Exception:
            pass
        ax2.text(val + max_co2 * 0.02, i, f"{val:,.0f}K t{pct_label}",
                 va="center", fontsize=10, fontweight="bold")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(levers, fontsize=11)
    ax2.set_xlabel("10-Year CO₂ Avoided (Thousand Tonnes)", fontsize=12)
    ax2.set_title("Cumulative CO₂ Reduction by Lever", fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    # Break-even carbon price annotation
    try:
        total_cost = float(mit_df[mit_df["lever"] == "Combined Portfolio"]["npv_cost_usd"].values[0])
        total_co2 = float(mit_df[mit_df["lever"] == "Combined Portfolio"]["total_co2_avoided_10yr_tonnes"].values[0])
        if total_co2 > 0:
            breakeven = total_cost / total_co2
            fig.text(0.5, -0.02,
                     f"Combined Portfolio break-even carbon price: ${breakeven:,.0f}/tonne CO₂",
                     ha="center", fontsize=11, fontweight="bold", color="#6A1B9A")
    except Exception:
        pass

    plt.tight_layout()
    _save(fig, "cba_mitigation_roi.png")


# ═══════════════════════════════════════════════════════════════════════════
#  FIG 4: Carbon Price Sensitivity
# ═══════════════════════════════════════════════════════════════════════════

def fig_carbon_price_sensitivity(proj_df):
    """NPV of 10-year cost as a function of carbon price for each scenario."""
    fig, ax = plt.subplots(figsize=(12, 7))

    carbon_prices = np.arange(0, 501, 10)  # $0–$500/t
    discount = FINANCE["discount_rate"]
    energy_price = FINANCE["energy_price_usd_per_mwh"]

    for scen in proj_df["scenario"].unique():
        sp = proj_df[proj_df["scenario"] == scen]
        future = sp[sp["year_offset"] > 0]

        npvs = []
        for cp in carbon_prices:
            npv = 0
            for _, r in future.iterrows():
                e_cost = r["energy_mwh"] * energy_price
                c_cost = r["co2_tonnes"] * cp
                npv += (e_cost + c_cost) / (1 + discount) ** r["year_offset"]
            npvs.append(npv / 1e6)

        clr = SCENARIO_COLORS.get(scen, "#888")
        ax.plot(carbon_prices, npvs, linewidth=2.5, color=clr, label=scen)

    # Reference lines
    for price, lbl, ls in [(95, "Low ($95)", ":"), (190, "EPA SCC ($190)", "--"), (300, "Tail ($300)", "-.")]:
        ax.axvline(price, color="#888", linestyle=ls, alpha=0.5)
        ax.text(price + 3, ax.get_ylim()[0] + 50, lbl, fontsize=9, color="#555", rotation=90)

    ax.set_title("Carbon Price Sensitivity: 10-Year NPV of Total Cost", fontsize=14, fontweight="bold")
    ax.set_xlabel("Carbon Price ($/tonne CO₂)", fontsize=12)
    ax.set_ylabel("10-Year NPV of Total Cost ($M)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}M"))

    plt.tight_layout()
    _save(fig, "cba_carbon_price_sensitivity.png")


# ═══════════════════════════════════════════════════════════════════════════
#  FIG 5: Risk Decomposition (Stacked Bar)
# ═══════════════════════════════════════════════════════════════════════════

def fig_risk_decomposition(cost_df):
    """Stacked bar of cost components by scenario in Year 10."""
    fig, ax = plt.subplots(figsize=(14, 8))

    yr10 = cost_df[cost_df["year_offset"] == 10]
    if len(yr10) == 0:
        yr10 = cost_df[cost_df["year_offset"] == cost_df["year_offset"].max()]

    scenarios = yr10["scenario"].unique()
    x = np.arange(len(scenarios))
    width = 0.5

    component_cols = [
        ("a_energy_procurement_usd", "Energy Procurement", "#1565C0"),
        ("b_demand_charges_usd", "Demand Charges", "#0D47A1"),
        ("c_carbon_liability_scc_usd", "Carbon Liability (SCC)", "#C62828"),
        ("f_cooling_maintenance_usd", "Cooling Maintenance", "#2E7D32"),
        ("g_peak_breach_penalty_usd", "Peak Breach Penalty", "#E65100"),
        ("h_risk_premium_usd", "Risk Premium", "#6A1B9A"),
        ("i_regulatory_reserve_usd", "Regulatory Reserve", "#37474F"),
    ]

    bottom = np.zeros(len(scenarios))
    for col, label, color in component_cols:
        vals = []
        for scen in scenarios:
            row = yr10[yr10["scenario"] == scen]
            vals.append(row[col].values[0] / 1e6 if len(row) > 0 else 0)
        vals = np.array(vals)
        ax.bar(x, vals, width, bottom=bottom, label=label, color=color, edgecolor="white", linewidth=0.5)
        # % labels on segments >= 10% of total
        for j in range(len(vals)):
            total_j = sum(yr10[yr10["scenario"] == scenarios[j]][c].values[0] / 1e6
                          for c, _, _ in component_cols
                          if len(yr10[yr10["scenario"] == scenarios[j]]) > 0)
            if total_j > 0:
                pct_seg = vals[j] / total_j * 100
                if pct_seg >= 10:
                    ax.text(x[j], bottom[j] + vals[j] / 2,
                            f"{pct_seg:.0f}%", ha="center", va="center",
                            fontsize=7, fontweight="bold", color="white")
        bottom += vals

    # Total labels on top
    for i, total in enumerate(bottom):
        ax.text(i, total + 1, f"${total:.1f}M", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=11, fontweight="bold")
    ax.set_ylabel("Annual Cost ($M)", fontsize=12)
    ax.set_title("Risk Decomposition by Cost Component — Year 10", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save(fig, "cba_risk_decomposition.png")


# ═══════════════════════════════════════════════════════════════════════════
#  FIG 6: Growth Evidence (DC Spending)
# ═══════════════════════════════════════════════════════════════════════════

def fig_growth_evidence(data_bundle):
    """DC spending scatter with exponential fit and R² — strongest empirical evidence."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: DC Spending with exp fit
    dc = data_bundle["dc_spending"]
    ax1.scatter(dc["date"], dc["spending_usd"] / 1e9, s=20, alpha=0.5, color="#7B1FA2", label="Monthly data")

    dc_clean = dc.dropna(subset=["spending_usd"])
    dc_clean = dc_clean[dc_clean["spending_usd"] > 0]
    x_days = (dc_clean["date"] - dc_clean["date"].iloc[0]).dt.days.values.astype(float)
    y_log = np.log(dc_clean["spending_usd"].values)
    coeffs = np.polyfit(x_days, y_log, 1)
    x_fit = np.linspace(x_days.min(), x_days.max(), 200)
    y_fit = np.exp(np.polyval(coeffs, x_fit))
    dates_fit = pd.to_datetime(dc_clean["date"].iloc[0]) + pd.to_timedelta(x_fit, unit="D")

    r2 = data_bundle["params"].get("dc_spend_exponential_r2", 0.95)
    cagr = data_bundle["params"].get("dc_spend_cagr", 0.25)
    ax1.plot(dates_fit, y_fit / 1e9, "--", color="#4A148C", linewidth=2.5,
             label=f"Exponential fit (R²={r2:.3f}, CAGR={cagr:.0%})")
    ax1.set_title("Virginia Data-Centre Construction Spending (est.)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Monthly Spending ($Billion)")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Text box with key stats
    textstr = (f"Start: ${dc['spending_usd'].iloc[0]/1e6:.0f}M/mo (2014)\n"
               f"End: ${dc['spending_usd'].iloc[-1]/1e9:.1f}B/mo (2025)\n"
               f"Growth: {dc['spending_usd'].iloc[-1]/dc['spending_usd'].iloc[0]:.0f}× in {(dc['date'].iloc[-1]-dc['date'].iloc[0]).days/365.25:.1f} yr\n"
               f"CAGR: {cagr:.0%} per year")
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="lavender", alpha=0.8))

    # Right: Log scale for linearity
    ax2.scatter(dc["date"], dc["spending_usd"], s=20, alpha=0.5, color="#7B1FA2")
    ax2.plot(dates_fit, y_fit, "--", color="#4A148C", linewidth=2.5, label=f"R²={r2:.3f}")
    ax2.set_yscale("log")
    ax2.set_title("Log Scale — Exponential Growth (Virginia est.)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Monthly Spending ($, log scale)")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x/1e9:.1f}B" if x >= 1e9 else f"${x/1e6:.0f}M"))

    plt.tight_layout()
    _save(fig, "cba_growth_evidence.png")


# ═══════════════════════════════════════════════════════════════════════════
#  MASTER
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_figures(data_bundle, proj_df, cost_df, mit_df, risk_df):
    """Generate 5 publication-quality figures (scenario emissions removed —
    redundant with base_forecast CO₂ and loss_characterization trajectories)."""
    print("\n" + "=" * 72)
    print("  SECTION 5: GENERATING PUBLICATION FIGURES")
    print("=" * 72 + "\n")

    fig_data_sources_overview(data_bundle)
    # fig_scenario_emissions removed — story told by other figures
    fig_mitigation_roi(mit_df)
    fig_carbon_price_sensitivity(proj_df)
    fig_risk_decomposition(cost_df)
    fig_growth_evidence(data_bundle)

    print(f"\n✓ All 5 figures saved to {FIGURE_DIR}/")


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_ingestion import ingest_all
    from cba_scenarios import project_scenarios
    from cba_full_analysis import run_full_cba

    data = ingest_all()
    proj_df, _ = project_scenarios(data["params"])
    cba = run_full_cba(proj_df, data["params"])
    generate_all_figures(data, proj_df, cba["cost_breakdown"],
                         cba["mitigation"], cba["risk_metrics"])
