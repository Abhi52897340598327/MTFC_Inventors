"""
DATA INGESTION — Synthetic / Placeholder Data Generator
=========================================================
Generates plausible FAKE data for all 6 data sources so the full CBA
pipeline (scenarios → analysis → figures) can run without any real CSVs.

★ SWAP THIS FILE for data_ingestion_real.py once the model is finalized. ★
★ PJM load data has been REMOVED — not used in this analysis. ★

The public API (ingest_all → dict with "params" + 6 DataFrames) is
identical to the real-data version, so every downstream module works
unchanged.

Data Sources (6):
    1. monthly_temp_virginia.csv              → VA temperature trend
    2. esif_daily_avg_interpolated.csv        → NREL ESIF PUE metrics
    3. energy_by_source_annual_grid_comp.csv  → Virginia grid composition
    4. virginia_total_carbon_emissions_*.csv  → Virginia CO₂ emissions
    5. virginia_yearly_energy_consumption.csv → Virginia total energy
    6. monthly-spending-data-center-us.csv    → US DC construction spending

Outputs:
    outputs/data_driven_parameters.csv — flat key-value table of synthetic params
    Returns: dict  (used programmatically by downstream modules)
"""

import pathlib, warnings
import numpy as np
import pandas as pd
from config import ROOT, OUTPUT_DIR

warnings.filterwarnings("ignore", category=FutureWarning)

# Reproducible
RNG = np.random.default_rng(42)


# ═══════════════════════════════════════════════════════════════════════════
#  SYNTHETIC DATA GENERATORS  (one per "source file")
# ═══════════════════════════════════════════════════════════════════════════

# ── 1. Virginia Monthly Temperature ─────────────────────────────────────
def _synth_temperature():
    """~594 rows, Aug 1976 – Jan 2026, with warming trend."""
    dates = pd.date_range("1976-08-01", "2026-01-01", freq="MS")
    n = len(dates)
    month = dates.month
    year = dates.year

    # Seasonal cycle + warming trend
    base = 55.0
    seasonal = 22 * np.sin(2 * np.pi * (month - 4) / 12)   # peaks in July
    trend    = 0.015 * (year - 1976)                          # ~0.15°F/decade
    noise    = RNG.normal(0, 2.5, n)
    avg_temp = base + seasonal + trend + noise

    cdd = np.where(avg_temp > 65, (avg_temp - 65) * 30, 0).astype(float)
    hdd = np.where(avg_temp < 65, (65 - avg_temp) * 30, 0).astype(float)

    df = pd.DataFrame({
        "month": dates.strftime("%B %Y"),
        "avg_temp_f": np.round(avg_temp, 1),
        "min_temp_f": np.round(avg_temp - 11, 1),
        "max_temp_f": np.round(avg_temp + 11, 1),
        "precipitation": np.round(RNG.uniform(1.5, 5.0, n), 2),
        "heating_degree_days": np.round(hdd, 0),
        "cooling_degree_days": np.round(cdd, 0),
    })
    df["date"] = dates
    return df


def _temp_params(df):
    params = {}
    params["va_avg_temp_f_mean"] = float(df["avg_temp_f"].mean())
    params["va_avg_temp_f_std"]  = float(df["avg_temp_f"].std())
    params["va_cdd_annual_mean"] = float(df["cooling_degree_days"].sum() / max(1, df["date"].dt.year.nunique()))
    params["va_hdd_annual_mean"] = float(df["heating_degree_days"].sum() / max(1, df["date"].dt.year.nunique()))
    df["year"] = df["date"].dt.year
    annual = df.groupby("year")["avg_temp_f"].mean().dropna()
    if len(annual) >= 10:
        x = annual.index.values.astype(float)
        slope, _ = np.polyfit(x, annual.values, 1)
        params["va_temp_trend_f_per_decade"] = float(slope * 10)
    else:
        params["va_temp_trend_f_per_decade"] = 0.0
    cdd_annual = df.groupby("year")["cooling_degree_days"].sum().dropna()
    if len(cdd_annual) >= 10:
        x = cdd_annual.index.values.astype(float)
        slope, _ = np.polyfit(x, cdd_annual.values, 1)
        params["va_cdd_trend_per_decade"] = float(slope * 10)
    else:
        params["va_cdd_trend_per_decade"] = 0.0
    recent = df[df["year"] >= df["year"].max() - 4]
    summer = recent[recent["date"].dt.month.isin([6, 7, 8])]
    params["va_recent_summer_cdd_monthly_avg"] = float(summer["cooling_degree_days"].mean()) if len(summer) > 0 else 0.0
    return params


# ── 3. ESIF Daily Data-Centre Metrics ───────────────────────────────────
def _synth_esif():
    """~3500 rows, Nov 2015 – Aug 2025, with PUE trending up slightly."""
    dates = pd.date_range("2015-11-09", "2025-08-28", freq="D")
    n = len(dates)
    year = dates.year

    it_power = 750 + 170 * (year - 2015) + RNG.normal(0, 40, n)
    it_power = np.clip(it_power, 400, 3000)
    pue_base = 1.03 + 0.004 * (year - 2015) + RNG.normal(0, 0.008, n)
    pue = np.clip(pue_base, 1.01, 1.12)
    cooling = it_power * (pue - 1) * 0.6

    df = pd.DataFrame({
        "date": dates,
        "cooling_kw": np.round(cooling, 2),
        "ere": np.round(RNG.uniform(0.78, 1.1, n), 3),
        "hvac_kw": np.round(cooling * 0.8, 2),
        "it_power_kw": np.round(it_power, 2),
        "plug_and_light_kw": np.round(RNG.uniform(2.5, 3.2, n), 2),
        "pue": np.round(pue, 3),
        "pump_kw": np.round(RNG.uniform(5, 50, n), 2),
    })
    return df


def _esif_params(df):
    params = {}
    params["esif_pue_mean"]         = float(df["pue"].mean())
    params["esif_pue_std"]          = float(df["pue"].std())
    params["esif_pue_p05"]          = float(df["pue"].quantile(0.05))
    params["esif_pue_p95"]          = float(df["pue"].quantile(0.95))
    params["esif_it_power_mean_kw"] = float(df["it_power_kw"].mean())
    params["esif_cooling_mean_kw"]  = float(df["cooling_kw"].mean())
    params["esif_cooling_fraction"] = float(df["cooling_kw"].mean() / df["it_power_kw"].mean()) if df["it_power_kw"].mean() > 0 else 0.0
    df["year"] = df["date"].dt.year
    annual_pue = df.groupby("year")["pue"].mean()
    if len(annual_pue) >= 3:
        x = annual_pue.index.values.astype(float)
        slope, _ = np.polyfit(x, annual_pue.values, 1)
        params["esif_pue_trend_per_year"] = float(slope)
    else:
        params["esif_pue_trend_per_year"] = 0.0
    annual_it = df.groupby("year")["it_power_kw"].mean()
    if len(annual_it) >= 3:
        first_yr, last_yr = annual_it.index.min(), annual_it.index.max()
        n = last_yr - first_yr
        params["esif_it_power_cagr"] = float((annual_it.iloc[-1] / annual_it.iloc[0]) ** (1.0 / n) - 1) if n > 0 and annual_it.iloc[0] > 0 else 0.0
    else:
        params["esif_it_power_cagr"] = 0.0
    return params


# ── 4. Virginia Grid Composition ────────────────────────────────────────
def _synth_grid_comp():
    """64 rows, 1960–2023, with coal declining + gas/renewables rising."""
    years = np.arange(1960, 2024)
    n = len(years)
    t = years - 1960

    coal      = 320_000 * np.exp(-0.005 * t) * np.clip(1 - 0.008 * t, 0.1, 1)
    natgas    = 70_000  * np.exp(0.035 * t)
    petroleum = 440_000 * np.exp(-0.002 * t)
    nuclear   = np.where(years >= 1970, 300_000 * (1 - np.exp(-0.08 * (t - 10))), 0.0)
    renewable = 60_000  * np.exp(0.018 * t)

    # Add noise
    for arr in [coal, natgas, petroleum, nuclear, renewable]:
        arr += RNG.normal(0, np.abs(arr) * 0.03 + 1)
        arr[:] = np.clip(arr, 0, None)

    df = pd.DataFrame({
        "Year": years,
        "Coal_Total_Consumption_Billion_BTU": np.round(coal, 0),
        "Natural_Gas_Total_Consumption_Billion_BTU": np.round(natgas, 0),
        "Petroleum_Total_Consumption_Billion_BTU": np.round(petroleum, 0),
        "Nuclear_Total_Consumption_Billion_BTU": np.round(nuclear, 0),
        "Renewable_Total_Consumption_Billion_BTU": np.round(renewable, 0),
    })
    return df


def _grid_comp_params(df):
    params = {}
    latest = df.iloc[-1]
    cols = [c for c in df.columns if c != "Year"]
    total = sum(latest[c] for c in cols if pd.notna(latest[c]))
    if total > 0:
        for c in cols:
            short = c.split("_")[0].lower()
            params[f"grid_share_{short}_pct"] = float(latest[c] / total * 100) if pd.notna(latest[c]) else 0.0
    params["grid_comp_latest_year"] = int(latest["Year"])
    coal_col = [c for c in df.columns if "coal" in c.lower()][0]
    recent = df[df["Year"] >= df["Year"].max() - 19].copy()
    if len(recent) >= 5:
        y1, y2 = recent.iloc[0], recent.iloc[-1]
        n = y2["Year"] - y1["Year"]
        params["coal_consumption_cagr_20yr"] = float((y2[coal_col] / y1[coal_col]) ** (1.0 / n) - 1) if n > 0 and y1[coal_col] > 0 else 0.0
    else:
        params["coal_consumption_cagr_20yr"] = 0.0
    ng_col = [c for c in df.columns if "natural" in c.lower() or "gas" in c.lower()][0]
    if len(recent) >= 5:
        y1, y2 = recent.iloc[0], recent.iloc[-1]
        n = y2["Year"] - y1["Year"]
        params["natgas_consumption_cagr_20yr"] = float((y2[ng_col] / y1[ng_col]) ** (1.0 / n) - 1) if n > 0 and y1[ng_col] > 0 else 0.0
    ren_col = [c for c in df.columns if "renewable" in c.lower()][0]
    if len(recent) >= 5:
        y1, y2 = recent.iloc[0], recent.iloc[-1]
        n = y2["Year"] - y1["Year"]
        params["renewable_consumption_cagr_20yr"] = float((y2[ren_col] / y1[ren_col]) ** (1.0 / n) - 1) if n > 0 and y1[ren_col] > 0 else 0.0
    return params


# ── 5. Virginia CO₂ Emissions ───────────────────────────────────────────
def _synth_emissions():
    """64 rows, 1960–2023, total CO₂ in MMT."""
    years = np.arange(1960, 2024)
    t = years - 1960

    coal_co2   = 16 * np.exp(-0.04 * t) + RNG.normal(0, 0.3, len(years))
    natgas_co2 = 0.1 + 0.32 * t + RNG.normal(0, 0.4, len(years))
    petro_co2  = 0.1 + 0.001 * t + RNG.normal(0, 0.05, len(years))

    coal_co2   = np.clip(coal_co2, 0.5, 30)
    natgas_co2 = np.clip(natgas_co2, 0.1, 25)
    petro_co2  = np.clip(petro_co2, 0.01, 2)
    total      = coal_co2 + natgas_co2 + petro_co2

    df = pd.DataFrame({
        "Year": years,
        "coal": np.round(coal_co2, 1),
        "natural_gas": np.round(natgas_co2, 1),
        "petroleum": np.round(petro_co2, 1),
        "Total_co2_mmt": np.round(total, 1),
    })
    return df


def _emissions_params(df):
    params = {}
    latest = df.dropna(subset=["Total_co2_mmt"]).iloc[-1]
    params["va_total_co2_mmt_latest"] = float(latest["Total_co2_mmt"])
    params["va_co2_latest_year"]      = int(latest["Year"])
    params["va_co2_coal_mmt_latest"]  = float(latest["coal"]) if pd.notna(latest["coal"]) else 0.0
    params["va_co2_natgas_mmt_latest"] = float(latest["natural_gas"]) if pd.notna(latest["natural_gas"]) else 0.0
    recent = df[df["Year"] >= df["Year"].max() - 19].dropna(subset=["Total_co2_mmt"])
    if len(recent) >= 5:
        y1, y2 = recent.iloc[0], recent.iloc[-1]
        n = y2["Year"] - y1["Year"]
        params["va_co2_cagr_20yr"] = float((y2["Total_co2_mmt"] / y1["Total_co2_mmt"]) ** (1.0 / n) - 1) if n > 0 and y1["Total_co2_mmt"] > 0 else 0.0
    else:
        params["va_co2_cagr_20yr"] = 0.0
    return params


# ── 6. Virginia Yearly Energy Consumption ───────────────────────────────
def _synth_va_energy():
    """64 rows, 1960–2023, total energy in Billion BTU."""
    years = np.arange(1960, 2024)
    t = years - 1960

    energy = 830_000 + 26_000 * t + RNG.normal(0, 30_000, len(years))
    energy = np.clip(energy, 700_000, 2_600_000)

    df = pd.DataFrame({
        "Year": years,
        "total_bbtu": np.round(energy, 0),
    })
    return df


def _va_energy_params(df):
    params = {}
    params["va_energy_bbtu_latest"] = float(df.iloc[-1]["total_bbtu"])
    params["va_energy_latest_year"] = int(df.iloc[-1]["Year"])
    params["va_energy_gwh_latest"]  = float(df.iloc[-1]["total_bbtu"] * 0.293071)
    recent = df[df["Year"] >= df["Year"].max() - 19]
    if len(recent) >= 5:
        y1, y2 = recent.iloc[0], recent.iloc[-1]
        n = y2["Year"] - y1["Year"]
        params["va_energy_cagr_20yr"] = float((y2["total_bbtu"] / y1["total_bbtu"]) ** (1.0 / n) - 1) if n > 0 and y1["total_bbtu"] > 0 else 0.0
    else:
        params["va_energy_cagr_20yr"] = 0.0
    n_full = df.iloc[-1]["Year"] - df.iloc[0]["Year"]
    if n_full > 0 and df.iloc[0]["total_bbtu"] > 0:
        params["va_energy_cagr_full"] = float((df.iloc[-1]["total_bbtu"] / df.iloc[0]["total_bbtu"]) ** (1.0 / n_full) - 1)
    else:
        params["va_energy_cagr_full"] = 0.0
    return params


# ── 7. US DC Construction Spending ──────────────────────────────────────
def _synth_dc_spending():
    """~140 rows, Jan 2014 – Aug 2025, exponential growth from ~$150M→$2.5B/mo."""
    dates = pd.date_range("2014-01-01", "2025-08-01", freq="MS")
    n = len(dates)
    t = np.arange(n)

    # Exponential: ~150M growing to ~2.5B in ~11.5 years
    spending = 150e6 * np.exp(0.024 * t) + RNG.normal(0, 20e6, n)
    spending = np.clip(spending, 100e6, 3.5e9)

    df = pd.DataFrame({
        "entity": "United States",
        "code": "USA",
        "date": dates,
        "spending_usd": np.round(spending, 0),
    })
    return df


def _dc_spending_params(df):
    params = {}
    params["dc_spend_first_date"]        = str(df["date"].iloc[0].date())
    params["dc_spend_last_date"]         = str(df["date"].iloc[-1].date())
    params["dc_spend_first_usd_monthly"] = float(df["spending_usd"].iloc[0])
    params["dc_spend_last_usd_monthly"]  = float(df["spending_usd"].iloc[-1])
    params["dc_spend_growth_factor"]     = float(df["spending_usd"].iloc[-1] / df["spending_usd"].iloc[0]) if df["spending_usd"].iloc[0] > 0 else 0.0
    df["year"] = df["date"].dt.year
    annual = df.groupby("year")["spending_usd"].sum()
    if len(annual) >= 3:
        first_yr, last_yr = annual.index.min(), annual.index.max()
        n = last_yr - first_yr
        params["dc_spend_cagr"] = float((annual.iloc[-1] / annual.iloc[0]) ** (1.0 / n) - 1) if n > 0 and annual.iloc[0] > 0 else 0.0
    else:
        params["dc_spend_cagr"] = 0.0
    if len(annual) >= 6:
        early = annual.iloc[:3]
        late  = annual.iloc[-3:]
        params["dc_spend_early_cagr"] = float((early.iloc[-1] / early.iloc[0]) ** 0.5 - 1) if early.iloc[0] > 0 else 0
        params["dc_spend_late_cagr"]  = float((late.iloc[-1] / late.iloc[0]) ** 0.5 - 1) if late.iloc[0] > 0 else 0
    else:
        params["dc_spend_early_cagr"] = 0.0
        params["dc_spend_late_cagr"]  = 0.0
    # R² of exponential fit
    dc_clean = df.dropna(subset=["spending_usd"])
    dc_clean = dc_clean[dc_clean["spending_usd"] > 0]
    if len(dc_clean) >= 10:
        x = (dc_clean["date"] - dc_clean["date"].iloc[0]).dt.days.values.astype(float)
        y = np.log(dc_clean["spending_usd"].values)
        coeffs = np.polyfit(x, y, 1)
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        params["dc_spend_exponential_r2"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    else:
        params["dc_spend_exponential_r2"] = 0.0
    return params


# ── Derived / Cross-Source ──────────────────────────────────────────────
def _derived_params(emissions_df, energy_df):
    params = {}
    merged = pd.merge(
        emissions_df[["Year", "Total_co2_mmt"]],
        energy_df[["Year", "total_bbtu"]],
        on="Year", how="inner"
    ).dropna()
    if len(merged) > 0:
        merged["ci_gco2_kwh"] = (merged["Total_co2_mmt"] * 1e12) / (merged["total_bbtu"] * 293071.0)
        latest = merged.iloc[-1]
        params["va_grid_ci_gco2_kwh_latest"]  = float(latest["ci_gco2_kwh"])
        params["va_grid_ci_kgco2_mwh_latest"] = float(latest["ci_gco2_kwh"])
        recent = merged[merged["Year"] >= merged["Year"].max() - 19]
        if len(recent) >= 5:
            x = recent["Year"].values.astype(float)
            slope, _ = np.polyfit(x, recent["ci_gco2_kwh"].values, 1)
            params["va_grid_ci_trend_per_decade"] = float(slope * 10)
        else:
            params["va_grid_ci_trend_per_decade"] = 0.0
    return params


# ═══════════════════════════════════════════════════════════════════════════
#  PUBLIC API  (identical signature to real-data version)
# ═══════════════════════════════════════════════════════════════════════════

def ingest_all():
    """
    Master function: generates synthetic data for all 6 sources,
    derives parameters, saves CSV, returns dict.
    """
    print("=" * 72)
    print("  DATA INGESTION — Generating SYNTHETIC Placeholder Data")
    print("  ★ Replace with data_ingestion_real.py once model is finalized ★")
    print("=" * 72)

    all_params = {}

    # 1. Temperature
    print("\n[1/6] Virginia Monthly Temperature (synthetic) …")
    temp = _synth_temperature()
    all_params.update(_temp_params(temp))
    print(f"      {len(temp):,} rows  |  Mean={all_params['va_avg_temp_f_mean']:.1f}°F  |  Trend={all_params['va_temp_trend_f_per_decade']:+.2f}°F/decade")

    # 2. ESIF
    print("[2/6] NREL ESIF Daily Metrics (synthetic) …")
    esif = _synth_esif()
    all_params.update(_esif_params(esif))
    print(f"      {len(esif):,} rows  |  PUE Mean={all_params['esif_pue_mean']:.3f}  |  IT CAGR={all_params['esif_it_power_cagr']:.1%}")

    # 3. Grid Composition
    print("[3/6] Virginia Grid Composition (synthetic) …")
    grid = _synth_grid_comp()
    all_params.update(_grid_comp_params(grid))
    print(f"      {len(grid):,} rows  |  Coal CAGR(20yr)={all_params.get('coal_consumption_cagr_20yr',0):.1%}  |  Renewable CAGR={all_params.get('renewable_consumption_cagr_20yr',0):.1%}")

    # 4. Emissions
    print("[4/6] Virginia CO₂ Emissions (synthetic) …")
    emis = _synth_emissions()
    all_params.update(_emissions_params(emis))
    print(f"      {len(emis):,} rows  |  Latest={all_params['va_total_co2_mmt_latest']:.1f} MMT")

    # 5. Energy Consumption
    print("[5/6] Virginia Energy Consumption (synthetic) …")
    va_en = _synth_va_energy()
    all_params.update(_va_energy_params(va_en))
    print(f"      {len(va_en):,} rows  |  Latest={all_params['va_energy_gwh_latest']:,.0f} GWh  |  CAGR(20yr)={all_params['va_energy_cagr_20yr']:.2%}")

    # 6. DC Spending
    print("[6/6] US Data-Centre Construction Spending (synthetic) …")
    dc_sp = _synth_dc_spending()
    all_params.update(_dc_spending_params(dc_sp))
    print(f"      {len(dc_sp):,} rows  |  CAGR={all_params['dc_spend_cagr']:.1%}  |  R²={all_params['dc_spend_exponential_r2']:.3f}")

    # Derived
    print("\n[+] Computing derived parameters (implied grid CI) …")
    all_params.update(_derived_params(emis, va_en))
    ci = all_params.get("va_grid_ci_kgco2_mwh_latest", 0)
    print(f"      VA Grid CI (latest) = {ci:.1f} kg CO₂/MWh")

    # Save
    out = pd.DataFrame(list(all_params.items()), columns=["parameter", "value"])
    out.to_csv(OUTPUT_DIR / "data_driven_parameters.csv", index=False)
    print(f"\n✓ Saved {len(out)} parameters → outputs/data_driven_parameters.csv")

    return {
        "params": all_params,
        "temperature": temp,
        "esif": esif,
        "grid_comp": grid,
        "emissions": emis,
        "va_energy": va_en,
        "dc_spending": dc_sp,
    }


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = ingest_all()
    print("\n── All Derived Parameters ──")
    for k, v in sorted(data["params"].items()):
        print(f"  {k:45s} = {v}")
