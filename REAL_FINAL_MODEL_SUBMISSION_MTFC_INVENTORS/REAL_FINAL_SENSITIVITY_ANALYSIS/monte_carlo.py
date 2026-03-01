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
from config import DC_PARAMS, GRID, FINANCE, MC, OUTPUT_DIR, FIGURE_DIR, PLOT, TEMPERATURE

np.random.seed(MC["seed"])


# ── Physical Model (vectorised) ─────────────────────────────────────────
def simulate(n: int = MC["n_simulations"]) -> pd.DataFrame:
    """Run *n* Monte-Carlo draws through the cascaded pipeline."""

    # --- Stage 1: Temperature (from real VA monthly data) ----------------
    temp_f = np.random.normal(TEMPERATURE["mean_f"], TEMPERATURE["std_f"], n)
    temp_f = np.clip(temp_f, TEMPERATURE["min_f"] - 5, TEMPERATURE["max_f"] + 5)

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
    from scipy.stats import gaussian_kde
    fig, axes = plt.subplots(1, 2, figsize=PLOT["figsize_wide"])

    for ax, col, xlabel, title in [
        (axes[0], "emissions_kg_per_h", "Hourly Emissions (kg CO₂)",
         "Monte-Carlo: Hourly Emissions Distribution"),
        (axes[1], "annual_emissions_tons", "Annual Emissions (tons CO₂)",
         "Monte-Carlo: Annual Emissions Distribution"),
    ]:
        vals = df[col].values
        ax.hist(vals, bins=120, density=True, color="#2b8c4e" if "kg" in col else "#2980b9",
                alpha=0.65, edgecolor="white", linewidth=0.3, label="Histogram (density)")

        # KDE overlay
        kde = gaussian_kde(vals)
        x_kde = np.linspace(vals.min(), vals.max(), 300)
        ax.plot(x_kde, kde(x_kde), color="black", lw=2, label="KDE")

        # VaR lines with value labels
        for q, c, ls in [(0.95, "#e67e22", "--"), (0.99, "#e74c3c", "-.")]:
            v = np.percentile(vals, q * 100)
            ax.axvline(v, color=c, ls=ls, lw=2)
            ax.text(v, ax.get_ylim()[1] * 0.92, f"VaR {int(q*100)}%\n{v:,.0f}",
                    color=c, fontsize=9, ha="left", fontweight="bold")

        # TVaR shading beyond VaR 95
        var95 = np.percentile(vals, 95)
        tvar95 = vals[vals >= var95].mean()
        ax.axvspan(var95, vals.max(), alpha=0.08, color="#e74c3c", label=f"TVaR₉₅ region (E={tvar95:,.0f})")

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Probability Density", fontsize=12)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "monte_carlo_emissions_distribution.png", dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close()


def plot_financial_risk(df):
    fig, axes = plt.subplots(1, 2, figsize=PLOT["figsize_wide"])
    from scipy.stats import gaussian_kde

    # Left: Total annual cost with KDE and TVaR shading
    ax = axes[0]
    vals = df["total_annual_cost"].values / 1e6
    ax.hist(vals, bins=120, density=True, color="#8e44ad", alpha=0.65, edgecolor="white", linewidth=0.3)
    kde = gaussian_kde(vals)
    x_kde = np.linspace(vals.min(), vals.max(), 300)
    ax.plot(x_kde, kde(x_kde), color="black", lw=2, label="KDE")

    for q, c, ls in [(0.95, "#e67e22", "--"), (0.99, "#e74c3c", "-.")]:
        v = np.percentile(vals, q * 100)
        ax.axvline(v, color=c, ls=ls, lw=2)
        ax.text(v, ax.get_ylim()[1] * 0.85, f"VaR {int(q*100)}%\n${v:,.1f}M",
                color=c, fontsize=9, ha="left", fontweight="bold")

    var95 = np.percentile(vals, 95)
    tvar95 = vals[vals >= var95].mean()
    ax.axvspan(var95, vals.max(), alpha=0.08, color="#e74c3c",
               label=f"TVaR₉₅ = ${tvar95:,.1f}M")
    ax.set_xlabel("Total Annual Cost ($M)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title("Total Annual Financial Risk", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Right: 3 SCC scenarios as box/violin + strip
    ax = axes[1]
    scenarios = {
        "Low\n($95/t)": df["carbon_cost_low"].values / 1e6,
        "Central\n(EPA $190/t)": df["carbon_cost_central"].values / 1e6,
        "High\n($300/t)": df["carbon_cost_high"].values / 1e6,
    }
    positions = [1, 2, 3]
    colors_box = ["#27ae60", "#f39c12", "#c0392b"]
    bp = ax.boxplot(list(scenarios.values()), positions=positions, widths=0.5,
                    patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=2))
    for patch, c in zip(bp["boxes"], colors_box):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(list(scenarios.keys()), fontsize=10)
    # Annotate medians and 95th percentiles
    for pos, (lbl, v) in zip(positions, scenarios.items()):
        med = np.median(v)
        p95 = np.percentile(v, 95)
        ax.text(pos, p95 + 0.3, f"P95=${p95:.1f}M", ha="center", fontsize=8, fontweight="bold")
    ax.set_ylabel("Annual Carbon Liability ($M)", fontsize=12)
    ax.set_title("Carbon Liability by SCC Scenario", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "monte_carlo_financial_risk.png", dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close()


def plot_convergence(df):
    """Show convergence of both the mean AND tail quantiles (VaR₉₅, VaR₉₉)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    vals = df["emissions_kg_per_h"].values
    n = len(vals)

    # Start from 500 to avoid early-stage noise dominating the plot
    idx = np.arange(500, n)

    running_mean = np.array([vals[:i+1].mean() for i in idx])
    running_p95 = np.array([np.percentile(vals[:i+1], 95) for i in idx])
    running_p99 = np.array([np.percentile(vals[:i+1], 99) for i in idx])

    final_mean = vals.mean()
    final_p95 = np.percentile(vals, 95)
    final_p99 = np.percentile(vals, 99)

    ax.plot(idx, running_mean, color="#2b8c4e", lw=1.5, label=f"Running Mean → {final_mean:,.0f}")
    ax.plot(idx, running_p95, color="#e67e22", lw=1.5, label=f"Running VaR₉₅ → {final_p95:,.0f}")
    ax.plot(idx, running_p99, color="#e74c3c", lw=1.5, label=f"Running VaR₉₉ → {final_p99:,.0f}")

    # ±0.5% convergence band around final mean
    band = 0.005
    ax.fill_between(idx, final_mean * (1 - band), final_mean * (1 + band),
                    alpha=0.15, color="#2b8c4e", label="±0.5% of final mean")
    ax.fill_between(idx, final_p95 * (1 - band), final_p95 * (1 + band),
                    alpha=0.10, color="#e67e22")
    ax.fill_between(idx, final_p99 * (1 - band), final_p99 * (1 + band),
                    alpha=0.10, color="#e74c3c")

    # Final value dashed lines
    ax.axhline(final_mean, color="#2b8c4e", ls=":", lw=1, alpha=0.5)
    ax.axhline(final_p95, color="#e67e22", ls=":", lw=1, alpha=0.5)
    ax.axhline(final_p99, color="#e74c3c", ls=":", lw=1, alpha=0.5)

    ax.set_xlabel("Number of Simulations", fontsize=12)
    ax.set_ylabel("Emissions (kg CO₂/h)", fontsize=12)
    ax.set_title("Monte-Carlo Convergence: Mean, VaR₉₅, and VaR₉₉",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="center right")
    ax.grid(True, alpha=0.2)

    # Convergence statistics
    # Find first index where running mean stays within ±0.5% of final
    converged_at = n
    for i in range(len(idx)):
        if abs(running_mean[i] / final_mean - 1) < band:
            if all(abs(running_mean[j] / final_mean - 1) < band for j in range(i, min(i+100, len(idx)))):
                converged_at = idx[i]
                break
    ax.text(0.02, 0.02,
            f"Mean converges (±0.5%) at n ≈ {converged_at:,}\nN = {n:,} total simulations",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

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
