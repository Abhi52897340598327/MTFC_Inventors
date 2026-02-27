"""
SARIMAX Forecast — Northern Virginia Datacenter CO₂ Emissions (tonnes/month)
==============================================================================
Purpose:
    Derive monthly CO₂ emission forecasts (2025-2035) by combining the
    outputs of sarimax_carbon_intensity.py and sarimax_energy_usage.py.
    Uncertainty from both forecasts is propagated via bound combination.
    A validation SARIMAX is also fitted directly to the historical
    derived emissions series.

INPUT FILES (must exist before running this script):
    GOOD_MainModels/carbon_intensity_forecast.csv   ← from File 1
    GOOD_MainModels/energy_usage_forecast.csv        ← from File 2

Run FILES IN ORDER:
    1.  python GOOD_MainModels/sarimax_carbon_intensity.py
    2.  python GOOD_MainModels/sarimax_energy_usage.py
    3.  python GOOD_MainModels/sarimax_carbon_emissions.py

Outputs:
    GOOD_MainModels/carbon_emissions_forecast.csv
    GOOD_Figures/carbon_emissions_historical_and_forecast.png
    GOOD_Figures/carbon_emissions_annual_bar.png
    GOOD_Figures/carbon_emissions_decomposed_drivers.png
    GOOD_Figures/carbon_emissions_cost_overlay.png
    GOOD_Figures/carbon_emissions_vs_scenarios.png
"""

import os
import sys
import warnings

os.makedirs("GOOD_MainModels", exist_ok=True)
os.makedirs("GOOD_Figures", exist_ok=True)

# ── library checks ──────────────────────────────────────────────────────────
try:
    import pmdarima
    from pmdarima import auto_arima
except ImportError:
    print("Missing required library: pmdarima. Run: pip install pmdarima")
    sys.exit(1)

try:
    import statsmodels
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:
    print("Missing required library: statsmodels. Run: pip install statsmodels")
    sys.exit(1)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ── file paths ─────────────────────────────────────────────────────────────
CI_FORECAST_CSV  = "GOOD_MainModels/carbon_intensity_forecast.csv"
EU_FORECAST_CSV  = "GOOD_MainModels/energy_usage_forecast.csv"
DC_PATH          = "Data_Sources/cleaned/semisynthetic_datacenter_power_2015_2024.csv"
CI_HIST_PATH     = "Data_Sources/cleaned/pjm_grid_carbon_intensity_2019_full_cleaned.csv"
SCENARIOS_PATH   = "REAL FINAL FILES/outputs/analysis"   # search for energy_forecast_scenarios.csv

EMISSIONS_CSV    = "GOOD_MainModels/carbon_emissions_forecast.csv"
FIG_FORECAST     = "GOOD_Figures/carbon_emissions_historical_and_forecast.png"
FIG_ANNUAL_BAR   = "GOOD_Figures/carbon_emissions_annual_bar.png"
FIG_DECOMP       = "GOOD_Figures/carbon_emissions_decomposed_drivers.png"
FIG_COST         = "GOOD_Figures/carbon_emissions_cost_overlay.png"
FIG_VS_SCENARIOS = "GOOD_Figures/carbon_emissions_vs_scenarios.png"

# Social Cost of Carbon — from REAL FINAL FILES/config.py FinancialAssumptions
# (scc_usd_per_ton = 51.0 as of config.py last updated Feb 2026)
SCC_USD_PER_TONNE = 51.0

# ═══════════════════════════════════════════════════════════════════════════
# 1.  LOAD FORECAST INPUTS
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("SARIMAX — Northern Virginia Datacenter CO₂ Emissions Forecast")
print("=" * 65)

for path, name in [
    (CI_FORECAST_CSV, "sarimax_carbon_intensity.py"),
    (EU_FORECAST_CSV, "sarimax_energy_usage.py"),
]:
    if not os.path.exists(path):
        print(f"\nERROR: Required input file not found:\n  {path}")
        print(f"  → Run {name} first.")
        print("Run sarimax_carbon_intensity.py and sarimax_energy_usage.py first.")
        sys.exit(1)

ci_fc = pd.read_csv(CI_FORECAST_CSV, parse_dates=["date"])
eu_fc = pd.read_csv(EU_FORECAST_CSV, parse_dates=["date"])

ci_fc.set_index("date", inplace=True)
eu_fc.set_index("date", inplace=True)

print(f"\n[Loaded] {CI_FORECAST_CSV}: {len(ci_fc)} rows, "
      f"{ci_fc.index.min().date()} → {ci_fc.index.max().date()}")
print(f"[Loaded] {EU_FORECAST_CSV}: {len(eu_fc)} rows, "
      f"{eu_fc.index.min().date()} → {eu_fc.index.max().date()}")

# ── align on shared date range ──────────────────────────────────────────────
shared_idx = ci_fc.index.intersection(eu_fc.index)
if len(shared_idx) != len(ci_fc) or len(shared_idx) != len(eu_fc):
    print(f"[WARNING] Forecast date ranges differ. Aligning to {len(shared_idx)} shared months.")
ci_fc = ci_fc.loc[shared_idx]
eu_fc = eu_fc.loc[shared_idx]

print(f"[Aligned] Shared forecast range: {shared_idx[0].date()} → {shared_idx[-1].date()}")

# ═══════════════════════════════════════════════════════════════════════════
# 2.  DERIVE MONTHLY EMISSIONS FORECAST
#
#     Formula:
#       emissions(tonnes) = energy_GWh × carbon_intensity_g_per_kwh
#
#     Proof of units:
#       GWh × (g CO₂ / kWh) = GWh × (kg CO₂ / MWh)
#         = 1000 MWh × (kg CO₂ / MWh) = 1000 kg = 1 tonne
#       ∴ emissions(t CO₂) = energy(GWh) × carbon_intensity(kg/MWh)
# ═══════════════════════════════════════════════════════════════════════════
emissions_fc = eu_fc["forecast_GWh"] * ci_fc["forecast"]

# Propagate uncertainty conservatively:
#   upper bound = max energy × max carbon intensity (worst case: more energy AND dirtier grid)
#   lower bound = min energy × min carbon intensity (best case: less energy AND cleaner grid)
emissions_upper_95 = eu_fc["upper_95"] * ci_fc["upper_95"]
emissions_lower_95 = eu_fc["lower_95"] * ci_fc["lower_95"]
emissions_upper_80 = eu_fc["upper_80"] * ci_fc["upper_80"]
emissions_lower_80 = eu_fc["lower_80"] * ci_fc["lower_80"]

# Guard against negative lower bounds
emissions_lower_95 = emissions_lower_95.clip(lower=0.0)
emissions_lower_80 = emissions_lower_80.clip(lower=0.0)

print(f"\n[Emissions derived]")
print(f"  Formula: emissions(t CO₂) = energy(GWh) × carbon_intensity(g/kWh)")
print(f"  Units note: g/kWh ≡ kg/MWh ≡ 1 t/GWh  →  t CO₂ = GWh × kg/MWh")
print(f"  2025-01: {emissions_fc.iloc[0]:.1f} tonnes/month")
print(f"  2035-12: {emissions_fc.iloc[-1]:.1f} tonnes/month")
print(f"  Trend  : {'+' if emissions_fc.iloc[-1] > emissions_fc.iloc[0] else ''}"
      f"{(emissions_fc.iloc[-1] / emissions_fc.iloc[0] - 1) * 100:.1f}% over the forecast period")

# ═══════════════════════════════════════════════════════════════════════════
# 3.  ANNUAL AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════
fc_combined = pd.DataFrame({
    "emissions_tonnes_monthly": emissions_fc,
    "upper_95": emissions_upper_95,
    "lower_95": emissions_lower_95,
    "upper_80": emissions_upper_80,
    "lower_80": emissions_lower_80,
})
fc_combined["year"] = fc_combined.index.year
annual = fc_combined.groupby("year").agg(
    annual_tonnes_central=("emissions_tonnes_monthly", "sum"),
    annual_tonnes_upper_95=("upper_95", "sum"),
    annual_tonnes_lower_95=("lower_95", "sum"),
    annual_tonnes_upper_80=("upper_80", "sum"),
    annual_tonnes_lower_80=("lower_80", "sum"),
)
fc_combined["emissions_tonnes_annual"] = fc_combined["year"].map(annual["annual_tonnes_central"])

# ═══════════════════════════════════════════════════════════════════════════
# 4.  HISTORICAL EMISSIONS SERIES (2019-2024) FOR VALIDATION
#
#     Derived from semisynthetic DC power × PJM hourly carbon intensity.
#     Note: For 2015-2018, no carbon intensity data is available, so the
#     historical series is limited to 2019-2024 (72 months). All monthly
#     values are in tonnes CO₂.
# ═══════════════════════════════════════════════════════════════════════════
hist_emissions_available = False
hist_monthly = None

if os.path.exists(DC_PATH) and os.path.exists(CI_HIST_PATH):
    try:
        dc = pd.read_csv(DC_PATH, parse_dates=["datetime"])
        dc.set_index("datetime", inplace=True)
        energy_m = (dc["total_power_mw"].resample("MS").sum() / 1000.0).rename("energy_GWh")

        ci_h = pd.read_csv(CI_HIST_PATH, parse_dates=["timestamp"])
        ci_h["timestamp"] = pd.to_datetime(ci_h["timestamp"], utc=True).dt.tz_localize(None)
        ci_h.set_index("timestamp", inplace=True)
        ci_m = ci_h["carbon_intensity_kg_per_mwh"].resample("MS").mean().rename("ci_g_per_kwh")

        combined_hist = pd.concat([energy_m, ci_m], axis=1).dropna()
        # emissions(t CO₂) = energy(GWh) × ci(kg/MWh) — units proof in Section 2
        hist_monthly = (combined_hist["energy_GWh"] * combined_hist["ci_g_per_kwh"]).rename("emissions_tonnes")
        hist_emissions_available = True
        print(f"\n[Historical emissions] Derived for {len(hist_monthly)} months "
              f"({hist_monthly.index.min().date()} → {hist_monthly.index.max().date()})")
        print(f"  Mean monthly: {hist_monthly.mean():.1f} t CO₂  |  Max: {hist_monthly.max():.1f}")
    except Exception as e:
        print(f"[WARNING] Could not derive historical emissions: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# 5.  VALIDATION SARIMAX ON HISTORICAL EMISSIONS
#     Checks that the derived forecast is in a plausible range.
# ═══════════════════════════════════════════════════════════════════════════
sarimax_val_rmse = None
if hist_emissions_available and len(hist_monthly) >= 24:
    print("\n[Validation SARIMAX] Fitting on historical 2019-2024 emissions...")
    hist_idx     = hist_monthly.index
    exog_hist    = pd.DataFrame({
        "month_sin": np.sin(2 * np.pi * hist_idx.month / 12),
        "month_cos": np.cos(2 * np.pi * hist_idx.month / 12),
    }, index=hist_idx).astype(float)

    try:
        val_auto = auto_arima(
            hist_monthly,
            exogenous=exog_hist.values,
            seasonal=True,
            m=12,
            information_criterion="aic",
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
        )
        val_sm = SARIMAX(
            hist_monthly,
            exog=exog_hist,
            order=val_auto.order,
            seasonal_order=val_auto.seasonal_order,
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=200)

        val_fitted = val_sm.fittedvalues
        sarimax_val_rmse = float(np.sqrt(mean_squared_error(
            hist_monthly.values, val_fitted.values
        )))
        print(f"  Validation SARIMAX order     : ARIMA{val_auto.order}{val_auto.seasonal_order}")
        print(f"  In-sample RMSE on hist series: {sarimax_val_rmse:.2f} tonnes/month")
        print(f"  (Plausibility check — forecast should produce values in the same order")
        print(f"   of magnitude as the historical series mean: {hist_monthly.mean():.0f} t/month)")
    except Exception as e:
        print(f"[WARNING] Validation SARIMAX failed: {e}")
else:
    print("[WARNING] Not enough historical emissions data for validation SARIMAX (need ≥ 24 months).")

# ═══════════════════════════════════════════════════════════════════════════
# 6.  PRINT FORECAST LANDMARKS
# ═══════════════════════════════════════════════════════════════════════════
def _annual_at(year: int) -> float:
    row = annual[annual.index == year]
    return float(row["annual_tonnes_central"].iloc[0]) if len(row) else float("nan")

print(f"\n[Forecast] Annual emissions:")
for year in [2025, 2028, 2030, 2032, 2035]:
    if year in annual.index:
        print(f"  {year}: {_annual_at(year):,.0f} tonnes CO₂/year")

# ═══════════════════════════════════════════════════════════════════════════
# 7.  SAVE EMISSIONS FORECAST CSV
# ═══════════════════════════════════════════════════════════════════════════
out_df = pd.DataFrame({
    "date"                    : fc_combined.index,
    "emissions_tonnes_monthly": fc_combined["emissions_tonnes_monthly"].values,
    "emissions_tonnes_annual" : fc_combined["emissions_tonnes_annual"].values,
    "lower_95"                : fc_combined["lower_95"].values,
    "upper_95"                : fc_combined["upper_95"].values,
    "lower_80"                : fc_combined["lower_80"].values,
    "upper_80"                : fc_combined["upper_80"].values,
})
out_df.to_csv(EMISSIONS_CSV, index=False)
print(f"\n[Saved] Emissions CSV → {EMISSIONS_CSV}")

# ═══════════════════════════════════════════════════════════════════════════
# 8.  FIGURES
# ═══════════════════════════════════════════════════════════════════════════

fc_dates     = fc_combined.index
fc_monthly   = fc_combined["emissions_tonnes_monthly"]
fc_upper_95  = fc_combined["upper_95"]
fc_lower_95  = fc_combined["lower_95"]
fc_upper_80  = fc_combined["upper_80"]
fc_lower_80  = fc_combined["lower_80"]

# ── Figure 1: Historical + forecast line chart ────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

if hist_emissions_available and hist_monthly is not None:
    ax.plot(hist_monthly.index, hist_monthly.values, color="steelblue", linewidth=1.8,
            label="Historical (derived 2019–2024)")

ax.plot(fc_dates, fc_monthly.values, color="steelblue", linewidth=1.8, linestyle="--",
        label="Forecast 2025–2035")
ax.fill_between(fc_dates, fc_lower_95, fc_upper_95, color="steelblue", alpha=0.2, label="95% CI")
ax.fill_between(fc_dates, fc_lower_80, fc_upper_80, color="steelblue", alpha=0.3, label="80% CI")

if hist_emissions_available and hist_monthly is not None:
    ax.axvline(hist_monthly.index.max(), color="gray", linestyle="--", alpha=0.7,
               label="History / Forecast boundary")

ax.set_title("Northern Virginia Datacenter CO₂ Emissions: Forecast 2025–2035",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("CO₂ Emissions (tonnes/month)", fontsize=12)
ax.tick_params(labelsize=10)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIG_FORECAST, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[Saved] Figure → {FIG_FORECAST}")

# ── Figure 2: Annual total CO₂ bar chart ──────────────────────────────────
error_low  = annual["annual_tonnes_central"] - annual["annual_tonnes_lower_95"]
error_high = annual["annual_tonnes_upper_95"] - annual["annual_tonnes_central"]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(annual.index, annual["annual_tonnes_central"] / 1e3, color="steelblue",
              alpha=0.8, label="Annual CO₂ Forecast (kt CO₂)", zorder=3)
ax.errorbar(annual.index, annual["annual_tonnes_central"] / 1e3,
            yerr=[error_low / 1e3, error_high / 1e3],
            fmt="none", color="black", capsize=5, linewidth=1.5, label="95% CI")

for year, row in annual.iterrows():
    val_kt = row["annual_tonnes_central"] / 1e3
    ax.text(year, val_kt + max(annual["annual_tonnes_central"] / 1e3) * 0.008,
            f"{val_kt:.0f}", ha="center", va="bottom", fontsize=8)

ax.set_title("Northern Virginia Datacenter: Annual CO₂ Emissions Forecast 2025–2035",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Annual CO₂ Emissions (kt CO₂/year)", fontsize=12)
ax.tick_params(labelsize=10)
ax.set_xticks(annual.index)
ax.set_xticklabels([str(y) for y in annual.index], rotation=45, ha="right")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_ANNUAL_BAR, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[Saved] Figure → {FIG_ANNUAL_BAR}")

# ── Figure 3: Decomposed drivers (energy growth vs grid cleanliness) ───────
# Decomposition method:
#   Base year: 2025-01
#   emissions = energy × carbon_intensity
#   Additive decomposition around the base year values:
#     Δemissions = ΔE × CI_base  (pure energy growth effect)
#                + E_base × ΔCI  (pure CI-change effect)
#                + ΔE × ΔCI      (interaction — small, assigned to energy)
#
# Annual version:
#   base_year = 2025 annual totals
# ─────────────────────────────────────────────────────────────────────────
base_year    = annual.index[0]   # first forecast year

fc_ann_eu   = eu_fc["forecast_GWh"].resample("YS").sum()   # annual GWh
fc_ann_ci   = ci_fc["forecast"].resample("YS").mean()      # annual mean gCO2/kWh

base_E  = float(fc_ann_eu.loc[fc_ann_eu.index[0]])   # GWh in base year
base_CI = float(fc_ann_ci.loc[fc_ann_ci.index[0]])   # g/kWh in base year

# Align indices
common_years  = fc_ann_eu.index.intersection(fc_ann_ci.index)
fc_ann_eu     = fc_ann_eu.loc[common_years]
fc_ann_ci     = fc_ann_ci.loc[common_years]

delta_E  = fc_ann_eu  - base_E
delta_CI = fc_ann_ci  - base_CI

# Effect from energy growth (tonnes = GWh × baseline CI)
energy_growth_effect  = delta_E  * base_CI                      # positive = more energy
# Effect from CI change (tonnes = baseline energy × delta CI)
ci_change_effect      = base_E   * delta_CI                     # negative = cleaner grid
# Base-year annual emissions (constant reference line)
base_annual_emissions = base_E * base_CI  # tonnes

years_plot = pd.to_datetime(common_years).year

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(years_plot, energy_growth_effect.values / 1e3,
       color="firebrick", alpha=0.75, label="Δ Emissions from energy growth (kt CO₂)")
ax.bar(years_plot, ci_change_effect.values / 1e3,
       color="forestgreen", alpha=0.75, label="Δ Emissions from grid decarbonisation (kt CO₂)")
ax.axhline(0, color="black", linewidth=0.8)
ax.axhline(base_annual_emissions / 1e3, color="steelblue", linestyle="dashed",
           linewidth=1.5, label=f"Base-year emissions ({base_year}): {base_annual_emissions/1e3:.0f} kt CO₂")

# Annotate the "grid cleanliness offset" on the last bar
last_ci_effect = float(ci_change_effect.iloc[-1]) / 1e3
last_year_val  = years_plot[-1]
ax.annotate(
    f"Grid cleanliness\noffset: {last_ci_effect:.0f} kt CO₂",
    xy=(last_year_val, last_ci_effect / 2),
    xytext=(-60, 0),
    textcoords="offset points",
    fontsize=8,
    arrowprops=dict(arrowstyle="->", color="gray"),
    color="forestgreen",
    fontweight="bold",
)

ax.set_title("Annual CO₂ Change Decomposed: Energy Growth vs. Grid Decarbonisation",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Change in Annual CO₂ Emissions vs. Base Year (kt CO₂)", fontsize=12)
ax.tick_params(labelsize=10)
ax.set_xticks(years_plot)
ax.set_xticklabels([str(y) for y in years_plot], rotation=45, ha="right")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DECOMP, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[Saved] Figure → {FIG_DECOMP}")

# ── Figure 4: Annual CO₂ emissions + carbon liability (dual axis) ──────────
# Note: SCC = $51/tonne sourced from REAL FINAL FILES/config.py
#       (FinancialAssumptions.scc_usd_per_ton)
annual_tonnes = annual["annual_tonnes_central"]
annual_usd_M  = annual_tonnes * SCC_USD_PER_TONNE / 1e6   # millions USD

fig, ax1 = plt.subplots(figsize=(12, 6))
color_em  = "steelblue"
color_usd = "darkorange"

ax1.plot(annual.index, annual_tonnes / 1e3, color=color_em, linewidth=2.0,
         marker="o", markersize=5, label="CO₂ Emissions (kt CO₂/year)")
ax1.fill_between(annual.index,
                 annual["annual_tonnes_lower_95"] / 1e3,
                 annual["annual_tonnes_upper_95"] / 1e3,
                 color=color_em, alpha=0.15, label="95% CI (emissions)")
ax1.set_xlabel("Year", fontsize=12)
ax1.set_ylabel("Annual CO₂ Emissions (kt CO₂/year)", fontsize=12, color=color_em)
ax1.tick_params(axis="y", labelcolor=color_em, labelsize=10)
ax1.tick_params(axis="x", labelsize=10)

ax2 = ax1.twinx()
ax2.plot(annual.index, annual_usd_M, color=color_usd, linewidth=2.0,
         marker="s", markersize=5, linestyle="--", label=f"Carbon Liability @SCC ${SCC_USD_PER_TONNE}/t ($M)")
ax2.fill_between(annual.index,
                 annual["annual_tonnes_lower_95"] * SCC_USD_PER_TONNE / 1e6,
                 annual["annual_tonnes_upper_95"] * SCC_USD_PER_TONNE / 1e6,
                 color=color_usd, alpha=0.10)
ax2.set_ylabel(f"Carbon Liability at SCC=${SCC_USD_PER_TONNE}/tonne ($M USD)", fontsize=12,
               color=color_usd)
ax2.tick_params(axis="y", labelcolor=color_usd, labelsize=10)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

ax1.set_title(
    f"Datacenter CO₂ Emissions & Carbon Liability (SCC = ${SCC_USD_PER_TONNE}/tonne)",
    fontsize=14, fontweight="bold",
)
ax1.set_xticks(annual.index)
ax1.set_xticklabels([str(y) for y in annual.index], rotation=45, ha="right")
ax1.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_COST, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[Saved] Figure → {FIG_COST}")

# ── Figure 5: SARIMAX forecast vs. energy growth scenarios ────────────────
# Load growth scenarios from posthoc sensitivity if they exist
scenarios_df = None
for analysis_dir in [
    SCENARIOS_PATH,
    "REAL FINAL FILES/outputs/analysis",
]:
    for run_dir in sorted([d for d in os.listdir(analysis_dir)
                            if os.path.isdir(os.path.join(analysis_dir, d))],
                           reverse=True) if os.path.exists(analysis_dir) else []:
        candidate = os.path.join(analysis_dir, run_dir, "energy_forecast_scenarios.csv")
        if os.path.exists(candidate):
            try:
                scenarios_df = pd.read_csv(candidate)
                print(f"\n[Scenarios] Loaded: {candidate}")
                print(f"  Columns: {list(scenarios_df.columns)}")
            except Exception:
                pass
            break
    if scenarios_df is not None:
        break

fig, ax = plt.subplots(figsize=(12, 6))

# SARIMAX forecast (central line + CI)
ax.plot(annual.index, annual_tonnes / 1e3, color="steelblue", linewidth=2.5,
        marker="o", markersize=5, zorder=5, label="SARIMAX forecast (central estimate)")
ax.fill_between(annual.index,
                annual["annual_tonnes_lower_95"] / 1e3,
                annual["annual_tonnes_upper_95"] / 1e3,
                color="steelblue", alpha=0.2, label="SARIMAX 95% CI")

# Overlay growth scenarios if available
SCENARIO_COLORS = {"Conservative": "forestgreen", "Moderate": "darkorange", "Aggressive": "firebrick"}
if scenarios_df is not None:
    # The scenarios CSV may have columns like year, scenario,
    # annual_energy_GWh, etc.; adapt to what's present
    year_col = next((c for c in scenarios_df.columns if "year" in c.lower()), None)
    scenario_col = next((c for c in scenarios_df.columns
                         if "scenario" in c.lower() or "growth" in c.lower()), None)
    energy_col = next((c for c in scenarios_df.columns
                       if "gwh" in c.lower() or "energy" in c.lower()), None)
    ci_mean = ci_fc["forecast"].resample("YS").mean()

    if year_col and scenario_col and energy_col:
        for scenario, color in SCENARIO_COLORS.items():
            mask = scenarios_df[scenario_col].str.contains(scenario, case=False, na=False)
            if mask.sum() == 0:
                continue
            sub = scenarios_df[mask][[year_col, energy_col]].copy()
            sub[year_col] = sub[year_col].astype(int)
            # Derive emissions: energy GWh × forecast CI at that year
            em_t = []
            for _, row in sub.iterrows():
                yr = row[year_col]
                ci_val = float(ci_mean.get(pd.Timestamp(f"{yr}-01-01"), ci_fc["forecast"].mean()))
                em_t.append(float(row[energy_col]) * ci_val / 1e3)   # kt CO₂
            ax.plot(sub[year_col], em_t, color=color, linewidth=1.8,
                    linestyle="dashed", alpha=0.8, label=f"{scenario} scenario")
    else:
        # Build simple growth scenarios from SARIMAX base year
        base_annual_E_GWh = float(fc_ann_eu.iloc[0])
        base_ci_val       = float(fc_ann_ci.iloc[0])
        fc_years          = pd.to_datetime(common_years).year

        for label, rate, color in [("Conservative (5%/yr)", 0.05, "forestgreen"),
                                    ("Moderate (15%/yr)",    0.15, "darkorange"),
                                    ("Aggressive (30%/yr)",  0.30, "firebrick")]:
            sc_emissions = []
            for i, yr in enumerate(fc_years):
                sc_E  = base_annual_E_GWh * (1 + rate) ** i
                sc_CI = float(fc_ann_ci.iloc[i]) if i < len(fc_ann_ci) else base_ci_val
                sc_emissions.append(sc_E * sc_CI / 1e3)   # kt CO₂
            ax.plot(fc_years, sc_emissions, color=color, linewidth=1.5,
                    linestyle="dashed", alpha=0.8, label=label)
        print("[Scenarios] energy_forecast_scenarios.csv not found — "
              "plotted three growth-rate scenarios derived from SARIMAX base year.")
else:
    # Build simple growth scenarios from SARIMAX anyway
    base_annual_E_GWh = float(fc_ann_eu.iloc[0])
    fc_years          = pd.to_datetime(common_years).year

    for label, rate, color in [("Conservative (5%/yr)", 0.05, "forestgreen"),
                                ("Moderate (15%/yr)",    0.15, "darkorange"),
                                ("Aggressive (30%/yr)",  0.30, "firebrick")]:
        sc_emissions = []
        for i, yr in enumerate(fc_years):
            sc_E  = base_annual_E_GWh * (1 + rate) ** i
            sc_CI = float(fc_ann_ci.iloc[i]) if i < len(fc_ann_ci) else float(fc_ann_ci.mean())
            sc_emissions.append(sc_E * sc_CI / 1e3)   # kt CO₂
        ax.plot(fc_years, sc_emissions, color=color, linewidth=1.5,
                linestyle="dashed", alpha=0.8, label=label)
    print("[Scenarios] energy_forecast_scenarios.csv not found — "
          "plotted three growth-rate scenarios from SARIMAX base year.")

ax.set_title("SARIMAX Emissions Forecast vs. Growth Scenarios (2025–2035)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Annual CO₂ Emissions (kt CO₂/year)", fontsize=12)
ax.tick_params(labelsize=10)
ax.set_xticks(annual.index)
ax.set_xticklabels([str(y) for y in annual.index], rotation=45, ha="right")
ax.legend(fontsize=9, loc="upper left")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_VS_SCENARIOS, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[Saved] Figure → {FIG_VS_SCENARIOS}")

# ═══════════════════════════════════════════════════════════════════════════
# 9. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("OUTPUTS WRITTEN")
print("  CSVs :")
print(f"    {EMISSIONS_CSV}")
print("  Figures :")
print(f"    {FIG_FORECAST}")
print(f"    {FIG_ANNUAL_BAR}")
print(f"    {FIG_DECOMP}")
print(f"    {FIG_COST}")
print(f"    {FIG_VS_SCENARIOS}")
print("=" * 65)
