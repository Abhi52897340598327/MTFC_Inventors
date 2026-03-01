"""
Base Forecasting Engine – Energy, CO₂, AI Growth & Grid Stress
================================================================
Integrated 10-year forecasting model that projects four coupled
time-series under multiple scenarios:

  1. **Energy Consumption** (MWh/yr) – data-centre + grid-level
  2. **CO₂ Emissions** (tonnes/yr) – Scope 2 from grid CI trajectory
  3. **AI Datacenter Capital Spending** ($B/yr) – calibrated to US Census
     monthly-spending data; feeds directly into CBA as investment driver
  4. **Grid Stress Index** (0-100) – composite of demand-to-capacity ratio,
     peak breach probability, and renewable penetration shortfall

Data Sources (real, in ``REAL FINAL DATA SOURCES/``):
  - monthly-spending-data-center-us.csv          (US DC CapEx monthly)
  - virginia_yearly_energy_consumption_bbtu.csv   (VA total energy)
  - virginia_total_carbon_emissions_energyproduction_annual.csv  (VA CO₂)
  - hrl_load_metered_combined_cleaned.csv         (hourly grid load)
  - energy_by_source_annual_grid_comp.csv         (fuel mix)

Synthetic placeholders are used where real forward data is unavailable.
Replace with actuals when obtained.

Outputs
-------
CSV:
  - forecast_energy_consumption.csv
  - forecast_co2_emissions.csv
  - forecast_ai_spending.csv
  - forecast_grid_stress.csv
  - forecast_combined_dashboard.csv

Figures:
  - figures/forecast_energy_consumption.png
  - figures/forecast_co2_emissions.png
  - figures/forecast_ai_spending.png
  - figures/forecast_grid_stress.png
  - figures/forecast_dashboard.png
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from config import (DC_PARAMS, GRID, FINANCE, GROWTH_SCENARIOS,
                    OUTPUT_DIR, FIGURE_DIR, PLOT)

# ── Data paths ───────────────────────────────────────────────────────────
_DATA = Path(__file__).resolve().parent.parent / "REAL FINAL DATA SOURCES"

# ── Constants ────────────────────────────────────────────────────────────
_BASE_YEAR = 2025                     # Last historical year
_HORIZON   = FINANCE["horizon_years"] # 10 years
_YEARS     = np.arange(_BASE_YEAR + 1, _BASE_YEAR + _HORIZON + 1)

# Grid capacity ceiling (MW) — rough PJM-DOM zone estimate
_GRID_CAPACITY_MW = 22_000

# Renewable penetration target by 2035 (Virginia Clean Economy Act)
_RENEW_TARGET_2035 = 0.30

# AI CapEx CAGR scenarios (calibrated from Census spending data 2014-2025)
_AI_SPEND = {
    "Conservative": {"base_annual_B": 22.0, "cagr": 0.12},
    "Moderate":     {"base_annual_B": 22.0, "cagr": 0.22},
    "Aggressive":   {"base_annual_B": 22.0, "cagr": 0.35},
}


# ═════════════════════════════════════════════════════════════════════════
#  1.  LOAD & CALIBRATE HISTORICAL DATA
# ═════════════════════════════════════════════════════════════════════════

def _load_historical_spending() -> pd.DataFrame:
    """Load US datacenter monthly spending → annual totals."""
    fp = _DATA / "monthly-spending-data-center-us.csv"
    if fp.exists():
        df = pd.read_csv(fp)
        col = [c for c in df.columns if "spending" in c.lower()
               or c.startswith("Monthly")][0]
        df["year"] = pd.to_datetime(df["Day"]).dt.year
        annual = df.groupby("year")[col].sum().reset_index()
        annual.columns = ["year", "spending_usd"]
        annual["spending_B"] = annual["spending_usd"] / 1e9
        return annual
    return pd.DataFrame({"year": [2024], "spending_B": [22.0]})


def _load_historical_energy() -> pd.DataFrame:
    """Load Virginia yearly energy consumption (Billion BTU)."""
    fp = _DATA / "virginia_yearly_energy_consumption_bbtu.csv"
    if fp.exists():
        df = pd.read_csv(fp)
        df.columns = ["year", "energy_bbtu"]
        return df
    return pd.DataFrame()


def _load_historical_co2() -> pd.DataFrame:
    """Load Virginia total CO₂ emissions (MMT)."""
    fp = _DATA / "virginia_total_carbon_emissions_energyproduction_annual.csv"
    if fp.exists():
        df = pd.read_csv(fp)
        return df[["Year", "Total_co2_mmt"]].rename(
            columns={"Year": "year", "Total_co2_mmt": "co2_mmt"})
    return pd.DataFrame()


def _load_grid_load() -> pd.DataFrame:
    """Load hourly grid load for peak/average stats."""
    fp = _DATA / "hrl_load_metered_combined_cleaned.csv"
    if fp.exists():
        df = pd.read_csv(fp)
        df["datetime"] = pd.to_datetime(df["datetime_beginning_ept"],
                                        format="mixed", dayfirst=False)
        df["year"] = df["datetime"].dt.year
        annual = df.groupby("year")["mw"].agg(
            mean_mw="mean", peak_mw="max", std_mw="std"
        ).reset_index()
        return annual
    return pd.DataFrame()


def _load_fuel_mix() -> pd.DataFrame:
    """Load energy-by-source for renewable penetration calc."""
    fp = _DATA / "energy_by_source_annual_grid_comp.csv"
    if fp.exists():
        df = pd.read_csv(fp)
        df["total"] = (df.get("Coal_Total_Consumption_Billion_BTU", 0)
                       + df.get("Natural_Gas_Total_Consumption_Billion_BTU", 0)
                       + df.get("Petroleum_Total_Consumption_Billion_BTU", 0)
                       + df.get("Nuclear_Total_Consumption_Billion_BTU", 0)
                       + df.get("Renewable_Total_Consumption_Billion_BTU", 0))
        df["renew_share"] = (
            df.get("Renewable_Total_Consumption_Billion_BTU", 0)
            + df.get("Nuclear_Total_Consumption_Billion_BTU", 0)
        ) / df["total"].replace(0, np.nan)
        return df[["Year", "renew_share"]].rename(columns={"Year": "year"})
    return pd.DataFrame()


# ═════════════════════════════════════════════════════════════════════════
#  2.  ENERGY CONSUMPTION FORECAST
# ═════════════════════════════════════════════════════════════════════════

def forecast_energy() -> pd.DataFrame:
    """10-year DC energy consumption forecast under 3 growth scenarios.

    Uses the physics model:
        IT_MW(yr) = base × (1 + growth)^yr
        PUE(yr)   = max(PUE₀ × (1 - pue_imp)^yr, 1.05)
        Total_MW  = IT_MW × PUE
        Annual_MWh = Total_MW × 8760
    """
    base_it = DC_PARAMS["it_capacity_mw"]
    pue0 = DC_PARAMS["pue_baseline"]
    rows = []
    for name, scen in GROWTH_SCENARIOS.items():
        g = scen["growth_rate"]
        pi = scen["pue_improvement"]
        for i, yr in enumerate(_YEARS, 1):
            it_mw = base_it * (1 + g) ** i
            pue = max(pue0 * (1 - pi) ** i, 1.05)
            total_mw = it_mw * pue
            annual_mwh = total_mw * 8760
            annual_gwh = annual_mwh / 1000
            rows.append({
                "year": yr,
                "scenario": name,
                "it_capacity_mw": round(it_mw, 1),
                "pue": round(pue, 4),
                "total_power_mw": round(total_mw, 1),
                "annual_energy_mwh": round(annual_mwh, 0),
                "annual_energy_gwh": round(annual_gwh, 1),
            })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════
#  3.  CO₂ EMISSIONS FORECAST
# ═════════════════════════════════════════════════════════════════════════

def forecast_co2() -> pd.DataFrame:
    """10-year CO₂ emissions forecast.

    Physics:
        CI(yr)  = CI₀ × (1 - ci_improvement)^yr   [grid decarbonises]
        Emissions = Total_MW × CI × 8760 / 1000    [tonnes/yr]
    """
    base_it = DC_PARAMS["it_capacity_mw"]
    pue0 = DC_PARAMS["pue_baseline"]
    ci0 = GRID["carbon_intensity_mean"]   # 345 kg/MWh
    scc = FINANCE["scc_usd_per_ton"]
    rows = []
    for name, scen in GROWTH_SCENARIOS.items():
        g = scen["growth_rate"]
        pi = scen["pue_improvement"]
        ci_imp = scen["carbon_intensity_improvement"]
        for i, yr in enumerate(_YEARS, 1):
            it_mw = base_it * (1 + g) ** i
            pue = max(pue0 * (1 - pi) ** i, 1.05)
            total_mw = it_mw * pue
            ci = ci0 * (1 - ci_imp) ** i
            annual_tons = total_mw * ci * 8760 / 1000
            carbon_cost = annual_tons * scc
            rows.append({
                "year": yr,
                "scenario": name,
                "carbon_intensity_kg_mwh": round(ci, 1),
                "annual_emissions_tons": round(annual_tons, 0),
                "annual_emissions_kt": round(annual_tons / 1000, 1),
                "carbon_cost_usd": round(carbon_cost, 0),
                "carbon_cost_M": round(carbon_cost / 1e6, 2),
            })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════
#  4.  AI DATACENTER CAPITAL SPENDING FORECAST
# ═════════════════════════════════════════════════════════════════════════

def forecast_ai_spending() -> pd.DataFrame:
    """Project US AI datacenter CapEx under 3 growth scenarios.

    Calibration: 2024 US datacenter construction spending ≈ $22B/yr
    (Census Bureau data).  CAGR scenarios: 12% / 22% / 35%.

    These numbers feed into the CBA as the *investment cost driver*:
    more spending → more capacity → more energy → more emissions.
    """
    hist = _load_historical_spending()
    rows = []
    for name, params in _AI_SPEND.items():
        base = params["base_annual_B"]
        cagr = params["cagr"]
        for i, yr in enumerate(_YEARS, 1):
            spend_B = base * (1 + cagr) ** i
            # Derive implied new capacity (rough: $15M/MW nameplate)
            new_mw = (spend_B * 1e9) * 0.30 / 15e6   # 30% → new IT capacity
            rows.append({
                "year": yr,
                "scenario": name,
                "ai_capex_B": round(spend_B, 2),
                "ai_capex_cumul_B": round(base * ((1 + cagr) ** (i + 1) - (1 + cagr))
                                          / cagr, 2),
                "implied_new_capacity_mw": round(new_mw, 0),
                "cagr_pct": round(cagr * 100, 1),
            })

    # Attach historical actuals for calibration
    hist_rows = []
    for _, h in hist.iterrows():
        if h["year"] >= 2019:
            hist_rows.append({
                "year": int(h["year"]),
                "scenario": "Historical",
                "ai_capex_B": round(h["spending_B"], 2),
                "ai_capex_cumul_B": 0,
                "implied_new_capacity_mw": 0,
                "cagr_pct": 0,
            })
    return pd.concat([pd.DataFrame(hist_rows), pd.DataFrame(rows)],
                     ignore_index=True)


# ═════════════════════════════════════════════════════════════════════════
#  5.  GRID STRESS INDEX FORECAST
# ═════════════════════════════════════════════════════════════════════════

def forecast_grid_stress() -> pd.DataFrame:
    """Composite Grid Stress Index (0-100) combining three sub-indices:

    1. **Demand-to-Capacity Ratio** (40% weight)
       Peak DC demand / grid zone capacity → 0-100 where 100 = at limit.

    2. **Peak Breach Probability** (35% weight)
       P(peak > contract) increases with growth.  Modelled as logistic
       function of demand/capacity ratio.

    3. **Renewable Shortfall** (25% weight)
       Renewable share vs VCEA 2035 target (30%).  The wider the gap,
       the higher the stress from carbon-intensive peakers.

    Grid stress > 70 = "critical" threshold requiring intervention.
    """
    base_it = DC_PARAMS["it_capacity_mw"]
    pue0 = DC_PARAMS["pue_baseline"]
    # Last known renewable share ~ 15% (2023 VA data)
    renew_now = 0.15
    renew_target = _RENEW_TARGET_2035

    rows = []
    for name, scen in GROWTH_SCENARIOS.items():
        g = scen["growth_rate"]
        pi = scen["pue_improvement"]
        for i, yr in enumerate(_YEARS, 1):
            it_mw = base_it * (1 + g) ** i
            pue = max(pue0 * (1 - pi) ** i, 1.05)
            total_mw = it_mw * pue
            peak_mw = total_mw * 1.15   # 15% peaking factor

            # Sub-index 1: Demand-to-Capacity (0-100)
            d2c_ratio = peak_mw / _GRID_CAPACITY_MW
            d2c_index = min(d2c_ratio * 100, 100)

            # Sub-index 2: Peak Breach Probability → logistic(d2c)
            # P(breach) = 1 / (1 + exp(-k*(d2c - 0.5)))
            k = 12
            p_breach = 1 / (1 + np.exp(-k * (d2c_ratio - 0.5)))
            breach_index = p_breach * 100

            # Sub-index 3: Renewable Shortfall (0-100)
            # Linear ramp from current to target
            yr_from_now = yr - _BASE_YEAR
            renew_share = renew_now + (renew_target - renew_now) * yr_from_now / 10
            shortfall = max(renew_target - renew_share, 0) / renew_target
            renew_index = shortfall * 100

            # Weighted composite
            stress = (0.40 * d2c_index
                      + 0.35 * breach_index
                      + 0.25 * renew_index)

            rows.append({
                "year": yr,
                "scenario": name,
                "total_power_mw": round(total_mw, 1),
                "peak_demand_mw": round(peak_mw, 1),
                "demand_capacity_ratio": round(d2c_ratio, 4),
                "peak_breach_prob": round(p_breach, 4),
                "renewable_share": round(renew_share, 4),
                "stress_demand_component": round(d2c_index, 1),
                "stress_breach_component": round(breach_index, 1),
                "stress_renew_component": round(renew_index, 1),
                "grid_stress_index": round(stress, 1),
                "stress_level": (
                    "Critical" if stress > 70 else
                    "Elevated" if stress > 45 else
                    "Normal"
                ),
            })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════
#  6.  COMBINED DASHBOARD
# ═════════════════════════════════════════════════════════════════════════

def build_dashboard(energy_df: pd.DataFrame,
                    co2_df: pd.DataFrame,
                    spend_df: pd.DataFrame,
                    stress_df: pd.DataFrame) -> pd.DataFrame:
    """Merge all forecasts into one wide table (one row per year×scenario)."""
    # Filter spend to non-historical
    spend_proj = spend_df[spend_df["scenario"] != "Historical"].copy()

    base = energy_df[["year", "scenario", "annual_energy_gwh",
                       "total_power_mw"]].copy()
    co2_cols = co2_df[["year", "scenario", "annual_emissions_kt",
                        "carbon_cost_M"]].copy()
    spend_cols = spend_proj[["year", "scenario", "ai_capex_B"]].copy()
    stress_cols = stress_df[["year", "scenario", "grid_stress_index",
                              "stress_level"]].copy()

    dash = base.merge(co2_cols, on=["year", "scenario"])
    dash = dash.merge(spend_cols, on=["year", "scenario"])
    dash = dash.merge(stress_cols, on=["year", "scenario"])
    return dash


# ═════════════════════════════════════════════════════════════════════════
#  7.  FIGURES
# ═════════════════════════════════════════════════════════════════════════

_COLORS = {"Conservative": "#27ae60", "Moderate": "#2980b9",
           "Aggressive": "#e74c3c", "Historical": "#7f8c8d"}


def _style_ax(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_energy(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 6))
    for name in GROWTH_SCENARIOS:
        sub = df[df["scenario"] == name]
        ax.plot(sub["year"], sub["annual_energy_gwh"],
                "o-", lw=2.5, ms=6, color=_COLORS[name], label=name)
    _style_ax(ax, "Year", "Annual Energy (GWh)",
              "Datacenter Energy Consumption Forecast (10yr)")
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "forecast_energy_consumption.png",
                dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close()


def plot_co2(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 6))
    for name in GROWTH_SCENARIOS:
        sub = df[df["scenario"] == name]
        ax.plot(sub["year"], sub["annual_emissions_kt"],
                "s-", lw=2.5, ms=6, color=_COLORS[name], label=name)
    _style_ax(ax, "Year", "Annual CO₂ Emissions (kt)",
              "Datacenter CO₂ Emissions Forecast (10yr)")
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "forecast_co2_emissions.png",
                dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close()


def plot_ai_spending(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 6))
    # Historical
    hist = df[df["scenario"] == "Historical"]
    if not hist.empty:
        ax.bar(hist["year"], hist["ai_capex_B"], color="#bdc3c7",
               width=0.6, label="Historical (US Census)", zorder=3)
    # Projections
    for name in _AI_SPEND:
        sub = df[df["scenario"] == name]
        ax.plot(sub["year"], sub["ai_capex_B"],
                "D--", lw=2, ms=6, color=_COLORS[name], label=f"{name} Forecast")
    _style_ax(ax, "Year", "Annual AI Datacenter CapEx ($B)",
              "US AI Datacenter Capital Spending Forecast")
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "forecast_ai_spending.png",
                dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close()


def plot_grid_stress(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 6))
    for name in GROWTH_SCENARIOS:
        sub = df[df["scenario"] == name]
        ax.plot(sub["year"], sub["grid_stress_index"],
                "^-", lw=2.5, ms=7, color=_COLORS[name], label=name)
    # Critical threshold
    ax.axhline(70, color="#c0392b", ls="--", lw=2, alpha=0.7)
    ax.text(_YEARS[-1] + 0.2, 71, "Critical (70)", color="#c0392b",
            fontsize=10, va="bottom")
    ax.axhline(45, color="#f39c12", ls=":", lw=1.5, alpha=0.7)
    ax.text(_YEARS[-1] + 0.2, 46, "Elevated (45)", color="#f39c12",
            fontsize=10, va="bottom")
    ax.set_ylim(0, 100)
    _style_ax(ax, "Year", "Grid Stress Index (0-100)",
              "Grid Stress Index Forecast – Virginia / PJM-DOM Zone")
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "forecast_grid_stress.png",
                dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close()


def plot_dashboard(dash: pd.DataFrame):
    """2×2 panel: energy, CO₂, AI spend, grid stress."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    for name in GROWTH_SCENARIOS:
        sub = dash[dash["scenario"] == name]
        c = _COLORS[name]
        axes[0, 0].plot(sub["year"], sub["annual_energy_gwh"],
                        "o-", lw=2, ms=5, color=c, label=name)
        axes[0, 1].plot(sub["year"], sub["annual_emissions_kt"],
                        "s-", lw=2, ms=5, color=c, label=name)
        axes[1, 0].plot(sub["year"], sub["ai_capex_B"],
                        "D-", lw=2, ms=5, color=c, label=name)
        axes[1, 1].plot(sub["year"], sub["grid_stress_index"],
                        "^-", lw=2, ms=5, color=c, label=name)

    axes[0, 0].set_ylabel("GWh/yr"); axes[0, 0].set_title("Energy Consumption")
    axes[0, 1].set_ylabel("kt CO₂/yr"); axes[0, 1].set_title("CO₂ Emissions")
    axes[1, 0].set_ylabel("$B/yr"); axes[1, 0].set_title("AI DC CapEx")
    axes[1, 1].set_ylabel("Stress (0-100)"); axes[1, 1].set_title("Grid Stress")
    axes[1, 1].axhline(70, color="#c0392b", ls="--", lw=1.5, alpha=0.7)

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_xlabel("Year", fontsize=10)

    plt.suptitle("Integrated Forecast Dashboard – 10-Year Horizon",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "forecast_dashboard.png",
                dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close()


# ═════════════════════════════════════════════════════════════════════════
#  8.  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════

def run():
    print("  ▶ Running base forecasting engine …")

    energy_df  = forecast_energy()
    co2_df     = forecast_co2()
    spend_df   = forecast_ai_spending()
    stress_df  = forecast_grid_stress()
    dash_df    = build_dashboard(energy_df, co2_df, spend_df, stress_df)

    # Save CSVs
    energy_df.to_csv(OUTPUT_DIR / "forecast_energy_consumption.csv", index=False)
    co2_df.to_csv(OUTPUT_DIR / "forecast_co2_emissions.csv", index=False)
    spend_df.to_csv(OUTPUT_DIR / "forecast_ai_spending.csv", index=False)
    stress_df.to_csv(OUTPUT_DIR / "forecast_grid_stress.csv", index=False)
    dash_df.to_csv(OUTPUT_DIR / "forecast_combined_dashboard.csv", index=False)

    # Figures
    plot_energy(energy_df)
    plot_co2(co2_df)
    plot_ai_spending(spend_df)
    plot_grid_stress(stress_df)
    plot_dashboard(dash_df)

    # Console summary
    print("    ✓ Energy consumption forecast (3 scenarios × 10 yr)")
    for name in GROWTH_SCENARIOS:
        sub = energy_df[energy_df["scenario"] == name]
        y10 = sub[sub["year"] == _YEARS[-1]]["annual_energy_gwh"].values[0]
        print(f"      {name:15s}  Year-10 energy: {y10:,.0f} GWh")

    print("    ✓ CO₂ emissions forecast")
    for name in GROWTH_SCENARIOS:
        sub = co2_df[co2_df["scenario"] == name]
        y10 = sub[sub["year"] == _YEARS[-1]]["annual_emissions_kt"].values[0]
        print(f"      {name:15s}  Year-10 CO₂: {y10:,.0f} kt")

    print("    ✓ AI datacenter CapEx forecast")
    for name in _AI_SPEND:
        sub = spend_df[(spend_df["scenario"] == name)
                       & (spend_df["year"] == _YEARS[-1])]
        y10 = sub["ai_capex_B"].values[0]
        print(f"      {name:15s}  Year-10 CapEx: ${y10:,.1f}B")

    print("    ✓ Grid stress index forecast")
    for name in GROWTH_SCENARIOS:
        sub = stress_df[stress_df["scenario"] == name]
        y10 = sub[sub["year"] == _YEARS[-1]]
        si = y10["grid_stress_index"].values[0]
        sl = y10["stress_level"].values[0]
        print(f"      {name:15s}  Year-10 stress: {si:.1f} ({sl})")

    print(f"    ✓ Combined dashboard ({len(dash_df)} rows)")
    return energy_df, co2_df, spend_df, stress_df, dash_df


if __name__ == "__main__":
    run()
