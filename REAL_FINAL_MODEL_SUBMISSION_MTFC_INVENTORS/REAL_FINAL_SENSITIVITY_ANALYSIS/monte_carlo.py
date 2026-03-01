"""
Monte-Carlo Simulation Engine
==============================
Propagates uncertainty through the 6-stage data-centre carbon model:
  Temperature → PUE → IT Load → Total Power → Grid Carbon → Emissions → $ Liability

Outputs
-------
- monte_carlo_results.csv          (per-simulation row)
- monte_carlo_summary.csv          (VaR / CVaR / percentiles)
- figures/monte_carlo_emissions_distribution.png
- figures/monte_carlo_financial_risk.png
- figures/monte_carlo_convergence.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import DC_PARAMS, GRID, FINANCE, MC, OUTPUT_DIR, FIGURE_DIR, PLOT

np.random.seed(MC["seed"])


# ── Physical Model (vectorised) ─────────────────────────────────────────
def simulate(n: int = MC["n_simulations"]) -> pd.DataFrame:
    """Run *n* Monte-Carlo draws through the cascaded pipeline."""

    # --- Stage 1: Temperature (truncated normal, Virginia climate) --------
    temp_f = np.random.normal(60.0, 15.0, n)
    temp_f = np.clip(temp_f, 10, 110)  # physical bounds

    # --- Stage 2: CPU utilisation (truncated normal) ----------------------
    cpu = np.random.normal(
        DC_PARAMS["cpu_utilization_mean"],
        DC_PARAMS["cpu_utilization_std"], n)
    cpu = np.clip(cpu, 5, 100)

    # --- Stage 3: PUE (physics-based) ------------------------------------
    delta_t = np.maximum(temp_f - 65.0, 0)
    pue = (DC_PARAMS["pue_baseline"]
           + DC_PARAMS["pue_temp_coef"] * delta_t
           + DC_PARAMS["pue_cpu_coef"] * cpu)

    # --- Stage 4: IT power (MW) ------------------------------------------
    idle_frac = np.random.uniform(
        DC_PARAMS["idle_power_fraction"] - 0.05,
        DC_PARAMS["idle_power_fraction"] + 0.05, n)
    it_power_mw = DC_PARAMS["it_capacity_mw"] * (
        idle_frac + (1 - idle_frac) * cpu / 100.0)

    # --- Stage 5: Total facility power (MW) ------------------------------
    total_power_mw = it_power_mw * pue

    # --- Stage 6: Grid carbon intensity (kg CO₂ / MWh) -------------------
    ci = np.random.normal(GRID["carbon_intensity_mean"],
                          GRID["carbon_intensity_std"], n)
    ci = np.clip(ci, GRID["carbon_intensity_min"],
                 GRID["carbon_intensity_max"])

    # --- Stage 7: Hourly emissions (kg CO₂) ------------------------------
    emissions_kg_h = total_power_mw * ci  # MW × kg/MWh = kg/h

    # --- Stage 8: Annual roll-up -----------------------------------------
    annual_energy_mwh = total_power_mw * 8760
    annual_emissions_tons = emissions_kg_h * 8760 / 1000.0

    # --- Stage 9: Financial liability ($) --------------------------------
    carbon_cost_central = annual_emissions_tons * FINANCE["scc_usd_per_ton"]
    carbon_cost_low     = annual_emissions_tons * FINANCE["carbon_price_low_usd_per_ton"]
    carbon_cost_high    = annual_emissions_tons * FINANCE["carbon_price_high_usd_per_ton"]
    electricity_cost    = annual_energy_mwh * FINANCE["energy_price_usd_per_mwh"]

    # Peak-demand penalty (probabilistic)
    peak_mw = total_power_mw * np.random.uniform(1.0, 1.15, n)
    breach = np.maximum(peak_mw - FINANCE["contract_peak_mw"], 0)
    peak_penalty = breach * FINANCE["peak_breach_penalty_usd_per_mwh"] * FINANCE["peak_breach_hours"]

    total_cost = electricity_cost + carbon_cost_central + peak_penalty

    return pd.DataFrame({
        "temperature_f":          temp_f,
        "cpu_utilization":        cpu,
        "pue":                    pue,
        "idle_power_fraction":    idle_frac,
        "it_power_mw":            it_power_mw,
        "total_power_mw":         total_power_mw,
        "carbon_intensity":       ci,
        "emissions_kg_per_h":     emissions_kg_h,
        "annual_energy_mwh":      annual_energy_mwh,
        "annual_emissions_tons":  annual_emissions_tons,
        "carbon_cost_central":    carbon_cost_central,
        "carbon_cost_low":        carbon_cost_low,
        "carbon_cost_high":       carbon_cost_high,
        "electricity_cost":       electricity_cost,
        "peak_mw":                peak_mw,
        "peak_penalty":           peak_penalty,
        "total_annual_cost":      total_cost,
    })


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Compute VaR, CVaR, and percentile table."""
    rows = []
    for col in ["emissions_kg_per_h", "annual_emissions_tons",
                "total_power_mw", "total_annual_cost",
                "carbon_cost_central", "carbon_cost_high"]:
        vals = df[col].values
        row = {"metric": col, "mean": vals.mean(), "std": vals.std(),
               "min": vals.min(), "max": vals.max(),
               "median": np.median(vals)}
        for q in MC["confidence_levels"]:
            var = np.percentile(vals, q * 100)
            cvar = vals[vals >= var].mean() if (vals >= var).any() else var
            row[f"VaR_{int(q*100)}"] = var
            row[f"CVaR_{int(q*100)}"] = cvar
        rows.append(row)
    return pd.DataFrame(rows)


# ── Figures ──────────────────────────────────────────────────────────────
def plot_emissions_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=PLOT["figsize_wide"])

    ax = axes[0]
    ax.hist(df["emissions_kg_per_h"], bins=120, color="#2b8c4e", alpha=0.85, edgecolor="white", linewidth=0.3)
    for q, c in zip([0.95, 0.99], ["#e67e22", "#e74c3c"]):
        v = df["emissions_kg_per_h"].quantile(q)
        ax.axvline(v, color=c, ls="--", lw=2, label=f"VaR {int(q*100)}% = {v:,.0f} kg/h")
    ax.set_xlabel("Hourly Emissions (kg CO₂)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Monte-Carlo: Hourly Emissions Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    ax = axes[1]
    ax.hist(df["annual_emissions_tons"], bins=120, color="#2980b9", alpha=0.85, edgecolor="white", linewidth=0.3)
    for q, c in zip([0.95, 0.99], ["#e67e22", "#e74c3c"]):
        v = df["annual_emissions_tons"].quantile(q)
        ax.axvline(v, color=c, ls="--", lw=2, label=f"VaR {int(q*100)}% = {v:,.0f} t/yr")
    ax.set_xlabel("Annual Emissions (tons CO₂)", fontsize=12)
    ax.set_title("Monte-Carlo: Annual Emissions Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "monte_carlo_emissions_distribution.png", dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close()


def plot_financial_risk(df):
    fig, axes = plt.subplots(1, 2, figsize=PLOT["figsize_wide"])

    ax = axes[0]
    vals = df["total_annual_cost"] / 1e6
    ax.hist(vals, bins=120, color="#8e44ad", alpha=0.85, edgecolor="white", linewidth=0.3)
    for q, c in zip([0.95, 0.99], ["#e67e22", "#e74c3c"]):
        v = vals.quantile(q)
        ax.axvline(v, color=c, ls="--", lw=2, label=f"VaR {int(q*100)}% = ${v:,.1f}M")
    ax.set_xlabel("Total Annual Cost ($M)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Monte-Carlo: Total Annual Financial Risk", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    ax = axes[1]
    scenarios = {"Low ($95/t)": df["carbon_cost_low"]/1e6,
                 "Central ($190/t)": df["carbon_cost_central"]/1e6,
                 "High ($300/t)": df["carbon_cost_high"]/1e6}
    colors = ["#27ae60", "#f39c12", "#c0392b"]
    for (lbl, v), col in zip(scenarios.items(), colors):
        ax.hist(v, bins=80, alpha=0.55, color=col, label=lbl, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Annual Carbon Liability ($M)", fontsize=12)
    ax.set_title("Carbon Liability Under Three SCC Scenarios", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "monte_carlo_financial_risk.png", dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close()


def plot_convergence(df):
    """Show that the Monte-Carlo mean stabilises."""
    fig, ax = plt.subplots(figsize=(10, 5))
    cumm = df["emissions_kg_per_h"].expanding().mean()
    ax.plot(cumm.values, color="#2b8c4e", lw=1.2)
    ax.axhline(df["emissions_kg_per_h"].mean(), color="#e74c3c", ls="--", lw=1.5, label="Final mean")
    ax.set_xlabel("Simulation #", fontsize=12)
    ax.set_ylabel("Running Mean Emissions (kg/h)", fontsize=12)
    ax.set_title("Monte-Carlo Convergence", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "monte_carlo_convergence.png", dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close()


# ── Entry Point ──────────────────────────────────────────────────────────
def run():
    print("  ▶ Running Monte-Carlo simulation …")
    df = simulate()
    summary = summarise(df)

    df.to_csv(OUTPUT_DIR / "monte_carlo_results.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "monte_carlo_summary.csv", index=False)

    plot_emissions_distribution(df)
    plot_financial_risk(df)
    plot_convergence(df)

    print(f"    ✓ {len(df):,} simulations complete")
    print(f"    ✓ Mean emissions: {df['emissions_kg_per_h'].mean():,.0f} kg/h")
    print(f"    ✓ 99% VaR emissions: {df['emissions_kg_per_h'].quantile(0.99):,.0f} kg/h")
    print(f"    ✓ 99% VaR total cost: ${df['total_annual_cost'].quantile(0.99)/1e6:,.1f}M")
    return df, summary


if __name__ == "__main__":
    run()
