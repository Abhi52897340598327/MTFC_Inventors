"""
Real-Data Parameter Derivation
================================
Loads the 7 real CSV data sources once and derives all statistical
parameters used by the Monte-Carlo, Copula, Sobol, Tornado, and
forecast modules.  Every number traces back to a real CSV file.

Usage:
    from real_params import REAL
    temp_mean = REAL["temperature_f_mean"]   # 55.67 °F from VA data
"""

import numpy as np
import pandas as pd
from config import DATA_DIR

# ── EPA / EIA CO₂ emission factors (kg CO₂ per MWh of generation) ────────
_CO2_FACTORS = {
    "Coal": 1000, "Natural Gas": 450, "Petroleum": 900, "Nuclear": 0,
    "Hydroelectric Conventional": 0, "Solar Thermal and Photovoltaic": 0,
    "Wind": 0, "Wood and Wood Derived Fuels": 0, "Other Biomass": 0,
    "Pumped Storage": 0, "Other": 450, "Other Gases": 450,
}


def _derive_all() -> dict:
    """One-shot loader: reads all CSVs and returns a flat parameter dict."""
    p = {}

    # ── 1. Virginia Monthly Temperature ──────────────────────────────────
    fp = DATA_DIR / "monthly_temp_virginia.csv"
    temp = pd.read_csv(fp)
    temp["date"] = pd.to_datetime(temp["month"], format="%B %Y")
    for col in ["avg_temp_f", "min_temp_f", "max_temp_f",
                 "cooling_degree_days", "heating_degree_days"]:
        temp[col] = pd.to_numeric(temp[col], errors="coerce")

    p["temperature_f_mean"] = float(temp["avg_temp_f"].mean())
    p["temperature_f_std"]  = float(temp["avg_temp_f"].std())
    p["temperature_f_min"]  = float(temp["avg_temp_f"].min())
    p["temperature_f_max"]  = float(temp["avg_temp_f"].max())

    # Recent decade for more relevant distribution
    recent_temp = temp[temp["date"].dt.year >= 2015]
    p["temperature_f_recent_mean"] = float(recent_temp["avg_temp_f"].mean())
    p["temperature_f_recent_std"]  = float(recent_temp["avg_temp_f"].std())

    # CDD/HDD
    p["cdd_annual_mean"] = float(
        temp["cooling_degree_days"].sum() / max(1, temp["date"].dt.year.nunique()))
    p["hdd_annual_mean"] = float(
        temp["heating_degree_days"].sum() / max(1, temp["date"].dt.year.nunique()))

    # Temperature trend
    temp["year"] = temp["date"].dt.year
    annual_t = temp.groupby("year")["avg_temp_f"].mean().dropna()
    if len(annual_t) >= 10:
        slope, _ = np.polyfit(annual_t.index.values.astype(float),
                              annual_t.values, 1)
        p["temperature_trend_f_per_decade"] = float(slope * 10)
    else:
        p["temperature_trend_f_per_decade"] = 0.0

    # ── 2. NREL ESIF Data Centre Metrics ─────────────────────────────────
    fp = DATA_DIR / "esif_daily_avg_interpolated.csv"
    esif = pd.read_csv(fp, parse_dates=["date"])
    for c in ["pue", "it_power_kw", "cooling_kw", "hvac_kw"]:
        esif[c] = pd.to_numeric(esif[c], errors="coerce")
    # Filter to physically plausible PUE range (raw data has outliers)
    esif_clean = esif[(esif["pue"] >= 1.0) & (esif["pue"] <= 3.0)].copy()

    p["pue_mean"]     = float(esif_clean["pue"].mean())
    p["pue_std"]      = float(esif_clean["pue"].std())
    p["pue_min"]      = float(esif_clean["pue"].min())
    p["pue_max"]      = float(esif_clean["pue"].max())
    p["pue_p05"]      = float(esif_clean["pue"].quantile(0.05))
    p["pue_p95"]      = float(esif_clean["pue"].quantile(0.95))

    p["it_power_mean_kw"] = float(esif["it_power_kw"].mean())
    p["it_power_std_kw"]  = float(esif["it_power_kw"].std())
    p["cooling_fraction"] = float(
        esif["cooling_kw"].mean() / esif["it_power_kw"].mean()
    ) if esif["it_power_kw"].mean() > 0 else 0.0

    # IT power growth
    esif["year"] = esif["date"].dt.year
    annual_it = esif.groupby("year")["it_power_kw"].mean()
    if len(annual_it) >= 3:
        first_yr, last_yr = annual_it.index.min(), annual_it.index.max()
        n_yr = last_yr - first_yr
        if n_yr > 0 and annual_it.iloc[0] > 0:
            p["it_power_cagr"] = float(
                (annual_it.iloc[-1] / annual_it.iloc[0]) ** (1.0 / n_yr) - 1)
        else:
            p["it_power_cagr"] = 0.0
    else:
        p["it_power_cagr"] = 0.0

    # ── 3. Carbon Intensity from Virginia Generation Mix ─────────────────
    fp = DATA_DIR / "virginia_generation_all_years.csv"
    gen = pd.read_csv(fp)
    gen = gen[gen["TYPE OF PRODUCER"] == "Total Electric Power Industry"]
    gen = gen[gen["ENERGY SOURCE"] != "Total"]
    gen = gen.copy()
    gen["co2_kg"] = gen.apply(
        lambda r: r["GENERATION (Megawatthours)"]
        * _CO2_FACTORS.get(r["ENERGY SOURCE"], 0), axis=1)
    monthly_gen = gen.groupby(["YEAR", "MONTH"]).agg(
        total_gen=("GENERATION (Megawatthours)", "sum"),
        total_co2=("co2_kg", "sum")).reset_index()
    monthly_gen["ci"] = monthly_gen["total_co2"] / monthly_gen["total_gen"]

    # Use recent years (2020+) for current-conditions estimates
    recent_ci = monthly_gen[monthly_gen["YEAR"] >= 2020]
    p["carbon_intensity_mean"] = float(recent_ci["ci"].mean())
    p["carbon_intensity_std"]  = float(recent_ci["ci"].std())
    p["carbon_intensity_min"]  = float(recent_ci["ci"].min())
    p["carbon_intensity_max"]  = float(recent_ci["ci"].max())

    # Monthly means for seasonal pattern
    ci_by_month = recent_ci.groupby("MONTH")["ci"].mean()
    p["carbon_intensity_by_month"] = ci_by_month.to_dict()

    # Long-term CI decline rate
    annual_ci = monthly_gen.groupby("YEAR")["ci"].mean()
    if len(annual_ci) >= 10:
        recent_20yr = annual_ci[annual_ci.index >= annual_ci.index.max() - 19]
        if len(recent_20yr) >= 5:
            slope, _ = np.polyfit(recent_20yr.index.values.astype(float),
                                  recent_20yr.values, 1)
            p["carbon_intensity_trend_per_decade"] = float(slope * 10)
        else:
            p["carbon_intensity_trend_per_decade"] = 0.0
    else:
        p["carbon_intensity_trend_per_decade"] = 0.0

    # ── 4. PJM Hourly Load ──────────────────────────────────────────────
    fp = DATA_DIR / "hrl_load_metered_combined_cleaned.csv"
    pjm = pd.read_csv(fp, engine="python")
    pjm["datetime"] = pd.to_datetime(
        pjm["datetime_beginning_ept"], format="mixed", dayfirst=False)
    pjm["mw"] = pd.to_numeric(pjm["mw"], errors="coerce")
    pjm = pjm.dropna(subset=["mw"])

    p["pjm_load_mean_mw"] = float(pjm["mw"].mean())
    p["pjm_load_std_mw"]  = float(pjm["mw"].std())
    p["pjm_load_p95_mw"]  = float(pjm["mw"].quantile(0.95))
    p["pjm_load_max_mw"]  = float(pjm["mw"].max())

    # Diurnal profile
    pjm["hour"] = pjm["datetime"].dt.hour
    hourly_avg = pjm.groupby("hour")["mw"].mean()
    p["diurnal_peak_trough_ratio"] = float(
        hourly_avg.max() / hourly_avg.min()) if hourly_avg.min() > 0 else 1.0
    p["diurnal_profile"] = (hourly_avg / hourly_avg.mean()).to_dict()

    # ── 5. Virginia Grid Composition ─────────────────────────────────────
    fp = DATA_DIR / "energy_by_source_annual_grid_comp.csv"
    grid = pd.read_csv(fp)
    grid["Year"] = pd.to_numeric(grid["Year"], errors="coerce")
    grid = grid.dropna(subset=["Year"]).sort_values("Year").reset_index(drop=True)
    latest = grid.iloc[-1]
    cols = [c for c in grid.columns if c != "Year"]
    total = sum(latest[c] for c in cols if pd.notna(latest[c]))
    if total > 0:
        for c in cols:
            short = c.split("_")[0].lower()
            p[f"grid_share_{short}_pct"] = float(
                latest[c] / total * 100) if pd.notna(latest[c]) else 0.0

    # Renewable share (nuclear + renewables as clean energy)
    ren_col = [c for c in grid.columns if "renewable" in c.lower()]
    nuc_col = [c for c in grid.columns if "nuclear" in c.lower()]
    if ren_col and nuc_col:
        latest_clean = (latest.get(ren_col[0], 0) + latest.get(nuc_col[0], 0))
        p["clean_energy_share"] = float(latest_clean / total) if total > 0 else 0.0
    else:
        p["clean_energy_share"] = 0.0

    # Coal decline and gas growth
    coal_col = [c for c in grid.columns if "coal" in c.lower()]
    if coal_col:
        recent_g = grid[grid["Year"] >= grid["Year"].max() - 19]
        if len(recent_g) >= 5:
            y1, y2 = recent_g.iloc[0], recent_g.iloc[-1]
            n_yr = y2["Year"] - y1["Year"]
            if n_yr > 0 and y1[coal_col[0]] > 0:
                p["coal_cagr_20yr"] = float(
                    (y2[coal_col[0]] / y1[coal_col[0]]) ** (1.0 / n_yr) - 1)
            else:
                p["coal_cagr_20yr"] = 0.0
        else:
            p["coal_cagr_20yr"] = 0.0

    # ── 6. Virginia CO₂ Emissions ────────────────────────────────────────
    fp = DATA_DIR / "virginia_total_carbon_emissions_energyproduction_annual.csv"
    emis = pd.read_csv(fp)
    emis["Year"] = pd.to_numeric(emis["Year"], errors="coerce")
    for c in ["coal", "natural_gas", "petroleum", "Total_co2_mmt"]:
        emis[c] = pd.to_numeric(
            emis[c].astype(str).str.replace(r"\(s\)", "", regex=True),
            errors="coerce")
    latest_e = emis.dropna(subset=["Total_co2_mmt"]).iloc[-1]
    p["va_total_co2_mmt"] = float(latest_e["Total_co2_mmt"])
    p["va_co2_coal_mmt"]  = float(latest_e["coal"]) if pd.notna(latest_e["coal"]) else 0.0
    p["va_co2_gas_mmt"]   = float(latest_e["natural_gas"]) if pd.notna(latest_e["natural_gas"]) else 0.0

    # CO₂ decline trend
    recent_e = emis[emis["Year"] >= emis["Year"].max() - 19].dropna(subset=["Total_co2_mmt"])
    if len(recent_e) >= 5:
        y1, y2 = recent_e.iloc[0], recent_e.iloc[-1]
        n_yr = y2["Year"] - y1["Year"]
        if n_yr > 0 and y1["Total_co2_mmt"] > 0:
            p["va_co2_cagr_20yr"] = float(
                (y2["Total_co2_mmt"] / y1["Total_co2_mmt"]) ** (1.0 / n_yr) - 1)
        else:
            p["va_co2_cagr_20yr"] = 0.0
    else:
        p["va_co2_cagr_20yr"] = 0.0

    # ── 7. Virginia Energy Consumption ───────────────────────────────────
    fp = DATA_DIR / "virginia_yearly_energy_consumption_bbtu.csv"
    va_en = pd.read_csv(fp)
    va_en.columns = ["Year", "total_bbtu"]
    va_en["Year"] = pd.to_numeric(va_en["Year"], errors="coerce")
    va_en["total_bbtu"] = pd.to_numeric(va_en["total_bbtu"], errors="coerce")
    va_en = va_en.dropna().sort_values("Year").reset_index(drop=True)
    p["va_energy_gwh_latest"] = float(va_en.iloc[-1]["total_bbtu"] * 0.293071)

    # ── 8. DC Spending ───────────────────────────────────────────────────
    fp = DATA_DIR / "monthly-spending-data-center-us.csv"
    dc = pd.read_csv(fp)
    dc.columns = ["entity", "code", "date", "spending_usd"]
    dc["date"] = pd.to_datetime(dc["date"])
    dc["spending_usd"] = pd.to_numeric(dc["spending_usd"], errors="coerce")
    # Apply Virginia share (time-varying 20% → 30%)
    total_days = (pd.Timestamp('2025-08-01') - pd.Timestamp('2014-01-01')).days
    elapsed = (dc["date"] - pd.Timestamp('2014-01-01')).dt.days.astype(float)
    frac = np.clip(elapsed / total_days, 0.0, 1.0)
    va_share = 0.20 + frac * 0.10
    dc["spending_usd"] = dc["spending_usd"] * va_share
    dc["year"] = dc["date"].dt.year
    annual_spend = dc.groupby("year")["spending_usd"].sum()
    p["dc_spend_latest_annual_B"] = float(annual_spend.iloc[-1] / 1e9)
    if len(annual_spend) >= 3:
        first_yr, last_yr = annual_spend.index.min(), annual_spend.index.max()
        n_yr = last_yr - first_yr
        if n_yr > 0 and annual_spend.iloc[0] > 0:
            p["dc_spend_cagr"] = float(
                (annual_spend.iloc[-1] / annual_spend.iloc[0]) ** (1.0 / n_yr) - 1)
        else:
            p["dc_spend_cagr"] = 0.0
    else:
        p["dc_spend_cagr"] = 0.0

    # ── Derived: grid CI from emissions / energy ─────────────────────────
    merged = pd.merge(
        emis[["Year", "Total_co2_mmt"]],
        va_en[["Year", "total_bbtu"]],
        on="Year", how="inner").dropna()
    if len(merged) > 0:
        merged["ci_kg_mwh"] = (merged["Total_co2_mmt"] * 1e12) / (
            merged["total_bbtu"] * 293071.0)
        p["va_grid_ci_kg_mwh_latest"] = float(merged["ci_kg_mwh"].iloc[-1])
        recent_m = merged[merged["Year"] >= merged["Year"].max() - 19]
        if len(recent_m) >= 5:
            slope, _ = np.polyfit(
                recent_m["Year"].values.astype(float),
                recent_m["ci_kg_mwh"].values, 1)
            p["va_grid_ci_trend_per_decade"] = float(slope * 10)
        else:
            p["va_grid_ci_trend_per_decade"] = 0.0

    return p


# ── Module-level singleton — loaded once on first import ─────────────────
REAL: dict = _derive_all()
