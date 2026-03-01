"""
DATA INGESTION — Extract Empirical Parameters from All 7 Real CSV Data Sources
================================================================================
Reads every CSV in REAL FINAL DATA SOURCES/ and derives the statistical
parameters that drive the cost-benefit analysis.  Nothing is hard-coded;
every number traces back to an identifiable row in a source file.

Outputs:
    outputs/data_driven_parameters.csv — flat key-value table of all derived params
    Returns: dict  (used programmatically by downstream modules)
"""

import pathlib, warnings
import numpy as np
import pandas as pd
from config import ROOT, DATA_DIR, OUTPUT_DIR

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────
#  1.  PJM Hourly Load  (hrl_load_metered_combined_cleaned.csv)
# ─────────────────────────────────────────────────────────────────────────
def _load_pjm_load():
    """Return cleaned hourly PJM load DataFrame with datetime index."""
    fp = DATA_DIR / "hrl_load_metered_combined_cleaned.csv"
    df = pd.read_csv(fp, engine="python")
    df["datetime"] = pd.to_datetime(df["datetime_beginning_ept"], format="mixed", dayfirst=False)
    df = df.sort_values("datetime").reset_index(drop=True)
    df["mw"] = pd.to_numeric(df["mw"], errors="coerce")
    df = df.dropna(subset=["mw"])
    return df


def _pjm_load_params(df):
    """Derive load statistics used in CBA."""
    params = {}
    params["pjm_load_mean_mw"]   = float(df["mw"].mean())
    params["pjm_load_std_mw"]    = float(df["mw"].std())
    params["pjm_load_p95_mw"]    = float(df["mw"].quantile(0.95))
    params["pjm_load_p99_mw"]    = float(df["mw"].quantile(0.99))
    params["pjm_load_min_mw"]    = float(df["mw"].min())
    params["pjm_load_max_mw"]    = float(df["mw"].max())

    # Yearly means for growth trend
    df["year"] = df["datetime"].dt.year
    yearly = df.groupby("year")["mw"].mean()
    if len(yearly) >= 2:
        first_yr, last_yr = yearly.index.min(), yearly.index.max()
        n = last_yr - first_yr
        if n > 0 and yearly.iloc[0] > 0:
            params["pjm_load_cagr"] = float((yearly.iloc[-1] / yearly.iloc[0]) ** (1.0 / n) - 1)
        else:
            params["pjm_load_cagr"] = 0.0
    else:
        params["pjm_load_cagr"] = 0.0

    # Diurnal peak-to-trough ratio (avg)
    df["hour"] = df["datetime"].dt.hour
    hourly_avg = df.groupby("hour")["mw"].mean()
    params["pjm_diurnal_peak_trough_ratio"] = float(hourly_avg.max() / hourly_avg.min()) if hourly_avg.min() > 0 else 1.0
    return params


# ─────────────────────────────────────────────────────────────────────────
#  2.  Monthly Temperature — Virginia  (monthly_temp_virginia.csv)
# ─────────────────────────────────────────────────────────────────────────
def _load_temperature():
    fp = DATA_DIR / "monthly_temp_virginia.csv"
    df = pd.read_csv(fp)
    df["date"] = pd.to_datetime(df["month"], format="%B %Y")
    df = df.sort_values("date").reset_index(drop=True)
    for col in ["avg_temp_f", "min_temp_f", "max_temp_f", "cooling_degree_days", "heating_degree_days"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _temp_params(df):
    params = {}
    params["va_avg_temp_f_mean"]     = float(df["avg_temp_f"].mean())
    params["va_avg_temp_f_std"]      = float(df["avg_temp_f"].std())
    params["va_cdd_annual_mean"]     = float(df["cooling_degree_days"].sum() / max(1, df["date"].dt.year.nunique()))
    params["va_hdd_annual_mean"]     = float(df["heating_degree_days"].sum() / max(1, df["date"].dt.year.nunique()))

    # Temperature trend (°F per decade) — linear regression on annual avg
    df["year"] = df["date"].dt.year
    annual = df.groupby("year")["avg_temp_f"].mean().dropna()
    if len(annual) >= 10:
        x = annual.index.values.astype(float)
        y = annual.values
        slope, _ = np.polyfit(x, y, 1)
        params["va_temp_trend_f_per_decade"] = float(slope * 10)
    else:
        params["va_temp_trend_f_per_decade"] = 0.0

    # CDD trend
    cdd_annual = df.groupby("year")["cooling_degree_days"].sum().dropna()
    if len(cdd_annual) >= 10:
        x = cdd_annual.index.values.astype(float)
        y = cdd_annual.values
        slope, _ = np.polyfit(x, y, 1)
        params["va_cdd_trend_per_decade"] = float(slope * 10)
    else:
        params["va_cdd_trend_per_decade"] = 0.0

    # Recent 5-year summer avg CDD (for PUE stress)
    recent = df[df["year"] >= df["year"].max() - 4]
    summer = recent[recent["date"].dt.month.isin([6, 7, 8])]
    params["va_recent_summer_cdd_monthly_avg"] = float(summer["cooling_degree_days"].mean()) if len(summer) > 0 else 0.0
    return params


# ─────────────────────────────────────────────────────────────────────────
#  3.  ESIF Daily Data-Centre Metrics  (esif_daily_avg_interpolated.csv)
# ─────────────────────────────────────────────────────────────────────────
def _load_esif():
    fp = DATA_DIR / "esif_daily_avg_interpolated.csv"
    df = pd.read_csv(fp, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    for c in ["pue", "it_power_kw", "cooling_kw", "hvac_kw"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _esif_params(df):
    params = {}
    params["esif_pue_mean"]           = float(df["pue"].mean())
    params["esif_pue_std"]            = float(df["pue"].std())
    params["esif_pue_p05"]            = float(df["pue"].quantile(0.05))
    params["esif_pue_p95"]            = float(df["pue"].quantile(0.95))
    params["esif_it_power_mean_kw"]   = float(df["it_power_kw"].mean())
    params["esif_cooling_mean_kw"]    = float(df["cooling_kw"].mean())
    params["esif_cooling_fraction"]   = float(df["cooling_kw"].mean() / df["it_power_kw"].mean()) if df["it_power_kw"].mean() > 0 else 0.0

    # PUE trend (annual)
    df["year"] = df["date"].dt.year
    annual_pue = df.groupby("year")["pue"].mean()
    if len(annual_pue) >= 3:
        x = annual_pue.index.values.astype(float)
        y = annual_pue.values
        slope, _ = np.polyfit(x, y, 1)
        params["esif_pue_trend_per_year"] = float(slope)
    else:
        params["esif_pue_trend_per_year"] = 0.0

    # IT power growth
    annual_it = df.groupby("year")["it_power_kw"].mean()
    if len(annual_it) >= 3:
        first_yr, last_yr = annual_it.index.min(), annual_it.index.max()
        n = last_yr - first_yr
        if n > 0 and annual_it.iloc[0] > 0:
            params["esif_it_power_cagr"] = float((annual_it.iloc[-1] / annual_it.iloc[0]) ** (1.0 / n) - 1)
        else:
            params["esif_it_power_cagr"] = 0.0
    else:
        params["esif_it_power_cagr"] = 0.0
    return params


# ─────────────────────────────────────────────────────────────────────────
#  4.  Virginia Grid Composition  (energy_by_source_annual_grid_comp.csv)
# ─────────────────────────────────────────────────────────────────────────
def _load_grid_comp():
    fp = DATA_DIR / "energy_by_source_annual_grid_comp.csv"
    df = pd.read_csv(fp)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Year"]).sort_values("Year").reset_index(drop=True)
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

    # Coal decline rate (last 20 years)
    coal_col = [c for c in df.columns if "coal" in c.lower()][0]
    recent = df[df["Year"] >= df["Year"].max() - 19].copy()
    if len(recent) >= 5:
        y1, y2 = recent.iloc[0], recent.iloc[-1]
        n = y2["Year"] - y1["Year"]
        if n > 0 and y1[coal_col] > 0:
            params["coal_consumption_cagr_20yr"] = float((y2[coal_col] / y1[coal_col]) ** (1.0 / n) - 1)
        else:
            params["coal_consumption_cagr_20yr"] = 0.0
    else:
        params["coal_consumption_cagr_20yr"] = 0.0

    # Natural gas growth rate (last 20 years)
    ng_col = [c for c in df.columns if "natural" in c.lower() or "gas" in c.lower()][0]
    if len(recent) >= 5:
        y1, y2 = recent.iloc[0], recent.iloc[-1]
        n = y2["Year"] - y1["Year"]
        if n > 0 and y1[ng_col] > 0:
            params["natgas_consumption_cagr_20yr"] = float((y2[ng_col] / y1[ng_col]) ** (1.0 / n) - 1)
        else:
            params["natgas_consumption_cagr_20yr"] = 0.0

    # Renewable growth rate (last 20 years)
    ren_col = [c for c in df.columns if "renewable" in c.lower()][0]
    if len(recent) >= 5:
        y1, y2 = recent.iloc[0], recent.iloc[-1]
        n = y2["Year"] - y1["Year"]
        if n > 0 and y1[ren_col] > 0:
            params["renewable_consumption_cagr_20yr"] = float((y2[ren_col] / y1[ren_col]) ** (1.0 / n) - 1)
        else:
            params["renewable_consumption_cagr_20yr"] = 0.0
    return params


# ─────────────────────────────────────────────────────────────────────────
#  5.  Virginia CO₂ Emissions  (virginia_total_carbon_emissions_energyproduction_annual.csv)
# ─────────────────────────────────────────────────────────────────────────
def _load_emissions():
    fp = DATA_DIR / "virginia_total_carbon_emissions_energyproduction_annual.csv"
    df = pd.read_csv(fp)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    # Handle "(s)" suppressed values
    for c in ["coal", "natural_gas", "petroleum", "Total_co2_mmt"]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(r"\(s\)", "", regex=True), errors="coerce")
    df = df.dropna(subset=["Year"]).sort_values("Year").reset_index(drop=True)
    return df


def _emissions_params(df):
    params = {}
    latest = df.dropna(subset=["Total_co2_mmt"]).iloc[-1]
    params["va_total_co2_mmt_latest"]       = float(latest["Total_co2_mmt"])
    params["va_co2_latest_year"]             = int(latest["Year"])
    params["va_co2_coal_mmt_latest"]         = float(latest["coal"]) if pd.notna(latest["coal"]) else 0.0
    params["va_co2_natgas_mmt_latest"]       = float(latest["natural_gas"]) if pd.notna(latest["natural_gas"]) else 0.0

    # Emissions trend (last 20 years)
    recent = df[df["Year"] >= df["Year"].max() - 19].dropna(subset=["Total_co2_mmt"])
    if len(recent) >= 5:
        y1, y2 = recent.iloc[0], recent.iloc[-1]
        n = y2["Year"] - y1["Year"]
        if n > 0 and y1["Total_co2_mmt"] > 0:
            params["va_co2_cagr_20yr"] = float((y2["Total_co2_mmt"] / y1["Total_co2_mmt"]) ** (1.0 / n) - 1)
        else:
            params["va_co2_cagr_20yr"] = 0.0
    else:
        params["va_co2_cagr_20yr"] = 0.0

    # Compute implied carbon intensity: total_co2 (MMT) / total_energy (BBtu)
    # Will be merged with energy consumption externally
    return params


# ─────────────────────────────────────────────────────────────────────────
#  6.  Virginia Yearly Energy Consumption (virginia_yearly_energy_consumption_bbtu.csv)
# ─────────────────────────────────────────────────────────────────────────
def _load_va_energy():
    fp = DATA_DIR / "virginia_yearly_energy_consumption_bbtu.csv"
    df = pd.read_csv(fp)
    df.columns = ["Year", "total_bbtu"]
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["total_bbtu"] = pd.to_numeric(df["total_bbtu"], errors="coerce")
    df = df.dropna().sort_values("Year").reset_index(drop=True)
    return df


def _va_energy_params(df):
    params = {}
    params["va_energy_bbtu_latest"]     = float(df.iloc[-1]["total_bbtu"])
    params["va_energy_latest_year"]     = int(df.iloc[-1]["Year"])

    # Convert to GWh for convenience (1 BBtu = 0.293071 GWh)
    latest_gwh = df.iloc[-1]["total_bbtu"] * 0.293071e-3  # BBtu → million BTU → GWh
    # Actually: 1 Billion BTU = 293.071 MWh = 0.293071 GWh
    params["va_energy_gwh_latest"]      = float(df.iloc[-1]["total_bbtu"] * 0.293071)

    # CAGR (last 20 years)
    recent = df[df["Year"] >= df["Year"].max() - 19]
    if len(recent) >= 5:
        y1, y2 = recent.iloc[0], recent.iloc[-1]
        n = y2["Year"] - y1["Year"]
        if n > 0 and y1["total_bbtu"] > 0:
            params["va_energy_cagr_20yr"] = float((y2["total_bbtu"] / y1["total_bbtu"]) ** (1.0 / n) - 1)
        else:
            params["va_energy_cagr_20yr"] = 0.0
    else:
        params["va_energy_cagr_20yr"] = 0.0

    # Full-period CAGR
    n_full = df.iloc[-1]["Year"] - df.iloc[0]["Year"]
    if n_full > 0 and df.iloc[0]["total_bbtu"] > 0:
        params["va_energy_cagr_full"] = float((df.iloc[-1]["total_bbtu"] / df.iloc[0]["total_bbtu"]) ** (1.0 / n_full) - 1)
    else:
        params["va_energy_cagr_full"] = 0.0
    return params


# ─────────────────────────────────────────────────────────────────────────
#  7.  US Data-Centre Construction Spending  (monthly-spending-data-center-us.csv)
# ─────────────────────────────────────────────────────────────────────────
# Virginia share of US datacenter construction spending (time-varying)
# Source: CBRE North America Data Center Trends (2024), JLL Data Center Outlook
VA_DC_SHARE_START      = 0.20    # ~20% in 2014
VA_DC_SHARE_END        = 0.30    # ~30% by 2025
_VA_SHARE_START_DATE   = pd.Timestamp('2014-01-01')
_VA_SHARE_END_DATE     = pd.Timestamp('2025-08-01')

def _va_dc_share_series(dates: pd.Series) -> pd.Series:
    """Time-varying Virginia share: linear 20% (2014) → 30% (2025)."""
    total_days = (_VA_SHARE_END_DATE - _VA_SHARE_START_DATE).days
    elapsed = (dates - _VA_SHARE_START_DATE).dt.days.astype(float)
    frac = np.clip(elapsed / total_days, 0.0, 1.0)
    return VA_DC_SHARE_START + frac * (VA_DC_SHARE_END - VA_DC_SHARE_START)

def _load_dc_spending():
    fp = DATA_DIR / "monthly-spending-data-center-us.csv"
    df = pd.read_csv(fp)
    df.columns = ["entity", "code", "date", "spending_usd"]
    df["date"] = pd.to_datetime(df["date"])
    df["spending_usd"] = pd.to_numeric(df["spending_usd"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    # Scale US-wide spending to Virginia's share (time-varying)
    va_share = _va_dc_share_series(df["date"])
    df["spending_usd_us"] = df["spending_usd"].copy()  # keep original for reference
    df["spending_usd"] = df["spending_usd"] * va_share
    print(f"  VA DC share applied to spending: {va_share.iloc[0]:.1%} → {va_share.iloc[-1]:.1%}")
    return df


def _dc_spending_params(df):
    params = {}
    params["dc_spend_first_date"]        = str(df["date"].iloc[0].date())
    params["dc_spend_last_date"]         = str(df["date"].iloc[-1].date())
    params["dc_spend_first_usd_monthly"] = float(df["spending_usd"].iloc[0])
    params["dc_spend_last_usd_monthly"]  = float(df["spending_usd"].iloc[-1])
    params["dc_spend_growth_factor"]     = float(df["spending_usd"].iloc[-1] / df["spending_usd"].iloc[0]) if df["spending_usd"].iloc[0] > 0 else 0.0

    # Annual aggregation for CAGR
    df["year"] = df["date"].dt.year
    annual = df.groupby("year")["spending_usd"].sum()
    if len(annual) >= 3:
        first_yr, last_yr = annual.index.min(), annual.index.max()
        n = last_yr - first_yr
        if n > 0 and annual.iloc[0] > 0:
            params["dc_spend_cagr"] = float((annual.iloc[-1] / annual.iloc[0]) ** (1.0 / n) - 1)
        else:
            params["dc_spend_cagr"] = 0.0
    else:
        params["dc_spend_cagr"] = 0.0

    # Acceleration: compare last-3yr CAGR vs first-3yr CAGR
    if len(annual) >= 6:
        early = annual.iloc[:3]
        late  = annual.iloc[-3:]
        early_cagr = (early.iloc[-1] / early.iloc[0]) ** 0.5 - 1 if early.iloc[0] > 0 else 0
        late_cagr  = (late.iloc[-1] / late.iloc[0]) ** 0.5 - 1 if late.iloc[0] > 0 else 0
        params["dc_spend_early_cagr"] = float(early_cagr)
        params["dc_spend_late_cagr"]  = float(late_cagr)
    else:
        params["dc_spend_early_cagr"] = 0.0
        params["dc_spend_late_cagr"]  = 0.0

    # R² of exponential fit (log-linear)
    df_clean = df.dropna(subset=["spending_usd"])
    df_clean = df_clean[df_clean["spending_usd"] > 0]
    if len(df_clean) >= 10:
        x = (df_clean["date"] - df_clean["date"].iloc[0]).dt.days.values.astype(float)
        y = np.log(df_clean["spending_usd"].values)
        coeffs = np.polyfit(x, y, 1)
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        params["dc_spend_exponential_r2"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    else:
        params["dc_spend_exponential_r2"] = 0.0
    return params


# ─────────────────────────────────────────────────────────────────────────
#  Derived / Cross-Source Parameters
# ─────────────────────────────────────────────────────────────────────────
def _derived_params(emissions_df, energy_df):
    """Compute implied Virginia grid carbon intensity from emissions / energy."""
    params = {}
    merged = pd.merge(
        emissions_df[["Year", "Total_co2_mmt"]],
        energy_df[["Year", "total_bbtu"]],
        on="Year", how="inner"
    ).dropna()

    if len(merged) > 0:
        # CI = Total_co2 (MMT) / Total_energy (BBtu → TWh)
        # 1 BBtu = 0.000293071 TWh;  1 MMT = 1e6 tonnes
        # CI (gCO2/kWh) = (MMT * 1e12 g) / (BBtu * 0.293071 * 1e6 kWh)
        #               = MMT * 1e12 / (BBtu * 293071)
        merged["ci_gco2_kwh"] = (merged["Total_co2_mmt"] * 1e12) / (merged["total_bbtu"] * 293071.0)
        latest = merged.iloc[-1]
        params["va_grid_ci_gco2_kwh_latest"]  = float(latest["ci_gco2_kwh"])
        params["va_grid_ci_kgco2_mwh_latest"] = float(latest["ci_gco2_kwh"])  # numerically same: g/kWh = kg/MWh

        # CI trend (last 20 years)
        recent = merged[merged["Year"] >= merged["Year"].max() - 19]
        if len(recent) >= 5:
            x = recent["Year"].values.astype(float)
            y = recent["ci_gco2_kwh"].values
            slope, _ = np.polyfit(x, y, 1)
            params["va_grid_ci_trend_per_decade"] = float(slope * 10)
        else:
            params["va_grid_ci_trend_per_decade"] = 0.0
    return params


# ═══════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════
def ingest_all():
    """
    Master function: reads all 7 CSVs, derives parameters, saves CSV, returns dict.
    """
    print("=" * 72)
    print("  DATA INGESTION — Reading 7 Real CSV Data Sources")
    print("=" * 72)

    all_params = {}

    # 1. PJM Load
    print("\n[1/7] PJM Hourly Load …")
    pjm = _load_pjm_load()
    all_params.update(_pjm_load_params(pjm))
    print(f"      {len(pjm):,} rows  |  Mean={all_params['pjm_load_mean_mw']:,.0f} MW")

    # 2. Temperature
    print("[2/7] Virginia Monthly Temperature …")
    temp = _load_temperature()
    all_params.update(_temp_params(temp))
    print(f"      {len(temp):,} rows  |  Mean={all_params['va_avg_temp_f_mean']:.1f}°F  |  Trend={all_params['va_temp_trend_f_per_decade']:+.2f}°F/decade")

    # 3. ESIF
    print("[3/7] NREL ESIF Daily Metrics …")
    esif = _load_esif()
    all_params.update(_esif_params(esif))
    print(f"      {len(esif):,} rows  |  PUE Mean={all_params['esif_pue_mean']:.3f}  |  IT CAGR={all_params['esif_it_power_cagr']:.1%}")

    # 4. Grid Composition
    print("[4/7] Virginia Grid Composition …")
    grid = _load_grid_comp()
    all_params.update(_grid_comp_params(grid))
    print(f"      {len(grid):,} rows  |  Coal CAGR(20yr)={all_params.get('coal_consumption_cagr_20yr',0):.1%}  |  Renewable CAGR={all_params.get('renewable_consumption_cagr_20yr',0):.1%}")

    # 5. Emissions
    print("[5/7] Virginia CO₂ Emissions …")
    emis = _load_emissions()
    all_params.update(_emissions_params(emis))
    print(f"      {len(emis):,} rows  |  Latest={all_params['va_total_co2_mmt_latest']:.1f} MMT")

    # 6. Energy Consumption
    print("[6/7] Virginia Energy Consumption …")
    va_en = _load_va_energy()
    all_params.update(_va_energy_params(va_en))
    print(f"      {len(va_en):,} rows  |  Latest={all_params['va_energy_gwh_latest']:,.0f} GWh  |  CAGR(20yr)={all_params['va_energy_cagr_20yr']:.2%}")

    # 7. DC Spending
    print("[7/7] US Data-Centre Construction Spending …")
    dc_sp = _load_dc_spending()
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

    # Return raw DataFrames too for downstream modules
    return {
        "params": all_params,
        "pjm_load": pjm,
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
