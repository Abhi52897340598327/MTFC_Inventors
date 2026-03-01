"""
Carbon Intensity Heatmap (Hour × Month)
=========================================
Generates a 24-row × 12-column heatmap of average grid carbon
intensity (kg CO₂ / MWh):
  • RdYlGn colormap (red = high, green = low)
  • Integer annotations in every cell
  • Y-axis = Hour of Day (0–23)
  • X-axis = Month (Jan–Dec)
  • Colorbar labelled "Avg Carbon Intensity (kg/MWh)"

Data-driven approach:
  1. Monthly CI from virginia_generation_all_years.csv
     (real VA generation by fuel × EPA emission factors)
  2. Diurnal modulation from hrl_load_metered_combined_cleaned.csv
     (higher load → more gas peakers → higher CI)

Outputs
-------
- carbon_intensity_heatmap.csv
- figures/carbon_intensity_heatmap.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from config import GRID, OUTPUT_DIR, FIGURE_DIR, PLOT, DATA_DIR

np.random.seed(42)

# EPA / EIA CO₂ emission factors (kg CO₂ per MWh of generation)
_CO2_FACTORS = {
    "Coal": 1000,
    "Natural Gas": 450,
    "Petroleum": 900,
    "Nuclear": 0,
    "Hydroelectric Conventional": 0,
    "Solar Thermal and Photovoltaic": 0,
    "Wind": 0,
    "Wood and Wood Derived Fuels": 0,
    "Other Biomass": 0,
    "Pumped Storage": 0,
    "Other": 450,
    "Other Gases": 450,
}


def _load_monthly_ci() -> pd.Series:
    """Compute real monthly-average CI (kg/MWh) from VA generation data."""
    fp = DATA_DIR / "virginia_generation_all_years.csv"
    gen = pd.read_csv(fp)
    gen = gen[gen["TYPE OF PRODUCER"] == "Total Electric Power Industry"]
    gen = gen[gen["ENERGY SOURCE"] != "Total"]
    gen["co2_kg"] = gen.apply(
        lambda r: r["GENERATION (Megawatthours)"]
        * _CO2_FACTORS.get(r["ENERGY SOURCE"], 0),
        axis=1,
    )
    monthly = (
        gen.groupby(["YEAR", "MONTH"])
        .agg(total_gen=("GENERATION (Megawatthours)", "sum"),
             total_co2=("co2_kg", "sum"))
        .reset_index()
    )
    monthly["ci"] = monthly["total_co2"] / monthly["total_gen"]
    # Average across recent years (2020+) for each calendar month
    recent = monthly[monthly["YEAR"] >= 2020]
    return recent.groupby("MONTH")["ci"].mean()  # Series indexed 1..12


def _load_diurnal_profile() -> pd.Series:
    """Normalised hourly load profile from PJM data (proxy for dispatch intensity)."""
    fp = DATA_DIR / "hrl_load_metered_combined_cleaned.csv"
    pjm = pd.read_csv(fp)
    pjm["datetime"] = pd.to_datetime(
        pjm["datetime_beginning_ept"], format="mixed", dayfirst=False
    )
    pjm["hour"] = pjm["datetime"].dt.hour
    hourly_avg = pjm.groupby("hour")["mw"].mean()
    # Normalise: 1.0 = daily mean, >1 = above-average dispatch
    return hourly_avg / hourly_avg.mean()  # Series indexed 0..23


def _build_heatmap_data() -> pd.DataFrame:
    """Build 24×12 CI grid from real data.

    For each (hour, month):
        CI(h, m) = monthly_CI(m) × diurnal_factor(h)
    where monthly_CI comes from real VA generation/fuel mix and
    the diurnal factor reflects how much marginal-dispatch CI
    rises above the monthly average during peak-load hours.
    """
    monthly_ci = _load_monthly_ci()       # Series: month 1..12 → CI (kg/MWh)
    diurnal    = _load_diurnal_profile()  # Series: hour 0..23 → multiplier

    # Scale the diurnal effect: at peak load, CI rises ~15% above monthly avg
    # (gas peakers fire); at trough, CI drops ~10% (nuclear/renewables dominate)
    diurnal_effect_strength = 0.18  # ±18% swing driven by load shape
    diurnal_scaled = 1.0 + (diurnal.values - 1.0) * diurnal_effect_strength / (
        diurnal.values.max() - 1.0
    )

    rows = []
    for month in range(1, 13):
        base_ci = monthly_ci.get(month, GRID["carbon_intensity_mean"])
        for hour in range(24):
            ci = base_ci * diurnal_scaled[hour]
            # No artificial clipping — let real data values show through
            rows.append({
                "month": month,
                "hour": hour,
                "carbon_intensity": round(ci, 1),
            })

    return pd.DataFrame(rows)


# ── Figure ───────────────────────────────────────────────────────────────
def plot_heatmap(df: pd.DataFrame):
    pivot = df.pivot(index="hour", columns="month", values="carbon_intensity")

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    hour_labels = [f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}" for h in range(24)]

    fig, ax = plt.subplots(figsize=(14, 10))

    im = ax.imshow(pivot.values, aspect="auto",
                   cmap="RdYlGn_r",
                   vmin=pivot.values.min() - 5,
                   vmax=pivot.values.max() + 5)

    # Annotate EVERY cell with its integer CI value
    ci_min, ci_max = pivot.values.min(), pivot.values.max()
    # Use the colormap to determine text contrast
    ci_range = ci_max - ci_min if ci_max > ci_min else 1
    cmap = plt.cm.RdYlGn_r
    for i in range(24):
        for j in range(12):
            val = pivot.values[i, j]
            # Map value to colormap normalised position
            norm_val = (val - (ci_min - 5)) / ((ci_max + 5) - (ci_min - 5))
            r, g, b, _ = cmap(norm_val)
            # Perceived luminance (ITU-R BT.601)
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            # Dark text on light/medium cells, white only on very dark cells
            color = "white" if lum < 0.35 else "#1a1a1a"
            ax.text(j, i, f"{int(round(val))}", ha="center", va="center",
                    fontsize=7, fontweight="bold", color=color)

    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels, fontsize=11)
    ax.set_yticks(range(24))
    ax.set_yticklabels(hour_labels, fontsize=9)
    ax.set_xlabel("Month", fontsize=13, fontweight="bold")
    ax.set_ylabel("Hour of Day", fontsize=13, fontweight="bold")
    ax.set_title("Average Carbon Intensity by Hour and Month (kg CO₂/MWh)\n"
                 f"Range: {ci_min:.0f} – {ci_max:.0f} kg/MWh  |  Mean: {pivot.values.mean():.0f}",
                 fontsize=14, fontweight="bold", pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label("Avg Carbon Intensity (kg/MWh)", fontsize=12)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "carbon_intensity_heatmap.png",
                dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close()


# ── Entry Point ──────────────────────────────────────────────────────────
def run():
    print("  ▶ Generating carbon intensity heatmap …")
    df = _build_heatmap_data()
    df.to_csv(OUTPUT_DIR / "carbon_intensity_heatmap.csv", index=False)

    plot_heatmap(df)

    pivot = df.pivot(index="hour", columns="month", values="carbon_intensity")
    print(f"    ✓ Heatmap: {pivot.shape[0]} hours × {pivot.shape[1]} months")
    print(f"    ✓ Range: {pivot.values.min():.0f} – {pivot.values.max():.0f} kg/MWh")
    return df


if __name__ == "__main__":
    run()
