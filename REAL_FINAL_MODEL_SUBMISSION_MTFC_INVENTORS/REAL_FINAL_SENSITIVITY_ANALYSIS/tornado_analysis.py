"""
Tornado / One-at-a-Time (OAT) Sensitivity Analysis
====================================================
Perturbs each parameter ±20 % from its baseline value while
holding all others constant. Records absolute and percentage
change in both emissions and energy outputs.

Only the emissions tornado is plotted (energy is saved to CSV
only, as it tells a nearly identical ranking story).

Outputs
-------
- tornado_oat.csv            (emissions target)
- tornado_oat_energy.csv     (energy target)
- figures/tornado_oat.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import DC_PARAMS, GRID, OUTPUT_DIR, FIGURE_DIR, PLOT, TEMPERATURE


# ── Parameter baselines and perturbation range ──────────────────────────
PARAM_DEFS = {
    "idle_power_fraction": DC_PARAMS["idle_power_fraction"],
    "cpu_utilization":     DC_PARAMS["cpu_utilization_mean"],
    "temperature_f":       TEMPERATURE["mean_f"],
    "pue_baseline":        DC_PARAMS["pue_baseline"],
    "carbon_intensity":    GRID["carbon_intensity_mean"],
    "it_capacity_mw":      DC_PARAMS["it_capacity_mw"],
    "n_servers":           DC_PARAMS["n_servers"],
}

SWING = 0.20   # ±20 %


# ── Deterministic model evaluation ───────────────────────────────────────
def _eval(params: dict) -> tuple[float, float]:
    """Return (annual_emissions_tons, annual_energy_mwh)."""
    idle   = params["idle_power_fraction"]
    cpu    = params["cpu_utilization"]
    temp_f = params["temperature_f"]
    pue_b  = params["pue_baseline"]
    ci     = params["carbon_intensity"]
    it_cap = params["it_capacity_mw"]

    it_mw = it_cap * (idle + (1 - idle) * cpu / 100)
    delta_t = max(temp_f - 65, 0)
    pue = pue_b + DC_PARAMS["pue_temp_coef"] * delta_t + DC_PARAMS["pue_cpu_coef"] * cpu
    total_mw = it_mw * pue
    energy_mwh = total_mw * 8760
    emissions_tons = total_mw * ci * 8760 / 1000
    return emissions_tons, energy_mwh


# ── Tornado sweep ────────────────────────────────────────────────────────
def compute_tornado() -> tuple[pd.DataFrame, pd.DataFrame]:
    base_params = dict(PARAM_DEFS)
    base_em, base_en = _eval(base_params)

    em_rows, en_rows = [], []

    for name, base_val in PARAM_DEFS.items():
        low_val  = base_val * (1 - SWING)
        high_val = base_val * (1 + SWING)

        # Low perturbation
        p = dict(base_params); p[name] = low_val
        em_low, en_low = _eval(p)

        # High perturbation
        p = dict(base_params); p[name] = high_val
        em_high, en_high = _eval(p)

        em_rows.append({
            "parameter":       name,
            "base_value":      round(base_val, 4),
            "low_value":       round(low_val, 4),
            "high_value":      round(high_val, 4),
            "base_output":     round(base_em, 2),
            "low_output":      round(em_low, 2),
            "high_output":     round(em_high, 2),
            "low_pct_change":  round((em_low - base_em) / base_em * 100, 2),
            "high_pct_change": round((em_high - base_em) / base_em * 100, 2),
            "range_pct":       round(abs(em_high - em_low) / base_em * 100, 2),
        })

        en_rows.append({
            "parameter":       name,
            "base_value":      round(base_val, 4),
            "low_value":       round(low_val, 4),
            "high_value":      round(high_val, 4),
            "base_output":     round(base_en, 2),
            "low_output":      round(en_low, 2),
            "high_output":     round(en_high, 2),
            "low_pct_change":  round((en_low - base_en) / base_en * 100, 2),
            "high_pct_change": round((en_high - base_en) / base_en * 100, 2),
            "range_pct":       round(abs(en_high - en_low) / base_en * 100, 2),
        })

    em_df = pd.DataFrame(em_rows).sort_values("range_pct", ascending=True)
    en_df = pd.DataFrame(en_rows).sort_values("range_pct", ascending=True)
    return em_df, en_df


# ── Figures ──────────────────────────────────────────────────────────────
def _plot_tornado(df: pd.DataFrame, target: str, fname: str):
    fig, ax = plt.subplots(figsize=(12, 7))

    y = np.arange(len(df))
    base_val = df["base_output"].iloc[0]

    # Bars: low side (negative direction) and high side (positive direction)
    low_delta  = df["low_pct_change"].values
    high_delta = df["high_pct_change"].values

    ax.barh(y, low_delta, height=0.6, color="#2980b9", label="−20% perturbation",
            edgecolor="white", linewidth=0.5)
    ax.barh(y, high_delta, height=0.6, color="#e74c3c", label="+20% perturbation",
            edgecolor="white", linewidth=0.5)

    # Annotations with elasticity (% output change / % input change = delta/20)
    for i in range(len(df)):
        if abs(low_delta[i]) > 0.5:
            elast = low_delta[i] / (-SWING * 100)
            ax.text(low_delta[i] - 0.3, i, f"{low_delta[i]:+.1f}% (ε={elast:.2f})",
                    va="center", ha="right", fontsize=8, color="#2980b9")
        if abs(high_delta[i]) > 0.5:
            elast = high_delta[i] / (SWING * 100)
            ax.text(high_delta[i] + 0.3, i, f"{high_delta[i]:+.1f}% (ε={elast:.2f})",
                    va="center", ha="left", fontsize=8, color="#e74c3c")

    ax.set_yticks(y)
    ax.set_yticklabels([p.replace("_", " ").title() for p in df["parameter"]],
                       fontsize=10)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("% Change from Baseline", fontsize=12)
    ax.set_title(f"Tornado Sensitivity (Local OAT, ±20%) – {target}",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    # Base value annotation
    ax.text(0.02, 0.98,
            f"Base case: {base_val:,.0f}\nε = elasticity = Δ%output / Δ%input\n"
            f"Note: OAT — no interaction effects captured",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / fname, dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close()


# ── Entry Point ──────────────────────────────────────────────────────────
def run():
    print("  ▶ Running tornado / OAT sensitivity analysis …")
    em_df, en_df = compute_tornado()

    em_df.to_csv(OUTPUT_DIR / "tornado_oat.csv", index=False)
    en_df.to_csv(OUTPUT_DIR / "tornado_oat_energy.csv", index=False)

    _plot_tornado(em_df, "Annual Emissions (t CO₂)", "tornado_oat.png")
    # Energy tornado saved to CSV only (ranking is near-identical).

    print("    ✓ Tornado (emissions) – top driver:")
    top = em_df.iloc[-1]
    print(f"      {top['parameter']}  range = {top['range_pct']:.1f}%")
    return em_df, en_df


if __name__ == "__main__":
    run()
