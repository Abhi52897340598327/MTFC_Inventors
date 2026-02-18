"""
MTFC Virginia Datacenter Energy Forecasting — Carbon Emissions Analysis
=========================================================================
Calculate hourly carbon footprint, annual emissions trends, fuel-mix
projections, and grid-decarbonization impact analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as cfg
from utils import log, save_fig, save_csv, COLORS
import data_loader


# ── Hourly Emissions ────────────────────────────────────────────────────────

def calculate_hourly_emissions(hourly_power: np.ndarray,
                               carbon_intensity_df: pd.DataFrame) -> np.ndarray:
    """
    Compute hourly CO₂ emissions = power (MWh) × emissions intensity (kg/MWh).
    Returns emissions in metric tons per hour.
    """
    # Align lengths
    n = min(len(hourly_power), len(carbon_intensity_df))

    if "carbon_intensity" in carbon_intensity_df.columns:
        ei = carbon_intensity_df["carbon_intensity"].values[:n]
    else:
        # Compute from fuel generation columns
        ei = compute_hourly_ei(carbon_intensity_df)[:n]

    emissions_kg = hourly_power[:n] * ei  # MW × kg/MWh = kg/h (hourly data)
    emissions_tons = emissions_kg / 1000  # metric tons per hour
    return emissions_tons


def compute_hourly_ei(carbon_df: pd.DataFrame) -> np.ndarray:
    """
    Compute hourly emissions intensity from fuel-specific generation columns.
    Uses EPA emissions factors.
    """
    fuel_map = {
        "coal":         ["net_generation_(mw)_from_coal"],
        "natural_gas":  ["net_generation_(mw)_from_natural_gas"],
        "petroleum":    ["net_generation_(mw)_from_petroleum"],
        "nuclear":      ["net_generation_(mw)_from_nuclear"],
        "solar":        ["net_generation_(mw)_from_solar"],
        "wind":         ["net_generation_(mw)_from_wind"],
    }

    total_gen = np.zeros(len(carbon_df))
    weighted_emissions = np.zeros(len(carbon_df))

    for fuel, col_names in fuel_map.items():
        for col in col_names:
            if col in carbon_df.columns:
                gen = carbon_df[col].fillna(0).values.astype(float)
                gen = np.maximum(gen, 0)  # no negative generation
                total_gen += gen
                weighted_emissions += gen * cfg.EMISSIONS_FACTORS.get(fuel, 0)

    # Handle other/remaining columns
    other_cols = [c for c in carbon_df.columns
                  if "net_generation" in c and "other" in c.lower()]
    for col in other_cols:
        gen = carbon_df[col].fillna(0).values.astype(float)
        gen = np.maximum(gen, 0)
        total_gen += gen
        weighted_emissions += gen * cfg.EMISSIONS_FACTORS["other"]

    # Emissions intensity = weighted emissions / total generation
    ei = np.zeros(len(carbon_df))
    mask = total_gen > 0
    ei[mask] = weighted_emissions[mask] / total_gen[mask]
    ei[~mask] = 400.0  # fallback

    return ei


# ── Annual Trends ───────────────────────────────────────────────────────────

def compute_annual_emissions_trend(co2_df: pd.DataFrame) -> pd.DataFrame:
    """Extract annual emissions trend from Virginia CO₂ data."""
    if "year" not in co2_df.columns and "period" in co2_df.columns:
        co2_df["year"] = pd.to_datetime(co2_df["period"], errors="coerce").dt.year

    # Filter for electric power sector if available
    if "sectorid" in co2_df.columns:
        ec = co2_df[co2_df["sectorid"].str.strip().str.upper() == "EC"]
        if len(ec) == 0:
            ec = co2_df  # fallback
    else:
        ec = co2_df

    if "value" in ec.columns:
        trend = ec.groupby("year")["value"].sum().reset_index()
        trend.columns = ["year", "co2_million_metric_tons"]
    else:
        trend = pd.DataFrame({"year": [], "co2_million_metric_tons": []})

    log.info(f"Annual CO₂ trend: {len(trend)} years")
    return trend


def project_emissions_intensity(base_year: int = 2019,
                                 base_ei: float = 400.0,
                                 target_years: list = None) -> dict:
    """Project future emissions intensity using declining rate."""
    if target_years is None:
        target_years = cfg.FORECAST_YEARS
    projections = {}
    for year in target_years:
        delta = year - base_year
        projections[year] = base_ei * (1 - cfg.GRID_EI_ANNUAL_DECLINE) ** delta
    return projections


# ── Visualisations ──────────────────────────────────────────────────────────

def plot_emissions_analysis(hourly_emissions: np.ndarray,
                            co2_trend: pd.DataFrame,
                            ei_projections: dict):
    """Generate carbon emissions analysis figures."""

    # 1. Monthly emissions from hourly data (use actual month boundaries)
    fig, ax = plt.subplots(figsize=(12, 5))
    n = len(hourly_emissions)
    # Create month array for 8760 hours: Jan=744h, Feb=672h, ... Dec=744h
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_labels = []
    for m_idx, days in enumerate(days_per_month, 1):
        month_labels.extend([m_idx] * (days * 24))
    month_arr = np.array(month_labels[:n])
    monthly = []
    for m in range(1, 13):
        mask = month_arr == m
        monthly.append(hourly_emissions[:len(mask)][mask[:len(hourly_emissions)]].sum())
    ax.bar(range(1, 13), monthly, color="#d62728", edgecolor="white")
    ax.set_title("Monthly CO₂ Emissions (2019 Baseline)")
    ax.set_xlabel("Month")
    ax.set_ylabel("CO₂ (metric tons)")
    ax.set_xticks(range(1, 13))
    save_fig(fig, "carbon_monthly_emissions")

    # 2. Historical CO₂ trend
    if len(co2_trend) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(co2_trend["year"], co2_trend["co2_million_metric_tons"],
                "o-", color="#1f77b4", lw=2, markersize=8)
        ax.set_title("Virginia Electric Power CO₂ Emissions Trend")
        ax.set_xlabel("Year")
        ax.set_ylabel("CO₂ (Million Metric Tons)")
        save_fig(fig, "carbon_historical_trend")

    # 3. Projected Emissions Intensity
    fig, ax = plt.subplots(figsize=(10, 5))
    years = sorted(ei_projections.keys())
    vals = [ei_projections[y] for y in years]
    ax.plot(years, vals, "s-", color="#2ca02c", lw=2, markersize=8)
    ax.set_title("Projected Grid Emissions Intensity (2024-2030)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Emissions Intensity (kg CO₂/MWh)")
    ax.fill_between(years, vals, alpha=0.15, color="#2ca02c")
    save_fig(fig, "carbon_ei_projection")


def run_carbon_analysis(hourly_power: np.ndarray,
                        carbon_intensity_df: pd.DataFrame,
                        co2_df: pd.DataFrame) -> dict:
    """Master carbon emissions analysis function."""
    # Hourly emissions
    hourly_emissions = calculate_hourly_emissions(hourly_power, carbon_intensity_df)
    total_emissions = hourly_emissions.sum()
    log.info(f"Total 2019 emissions: {total_emissions:,.0f} metric tons CO₂")

    # Annual trend
    co2_trend = compute_annual_emissions_trend(co2_df)

    # EI projections
    ei_proj = project_emissions_intensity()

    # Plots
    plot_emissions_analysis(hourly_emissions, co2_trend, ei_proj)

    # Per-year projected datacenter emissions
    annual_dc_emissions = {}
    for year in cfg.FORECAST_YEARS:
        ei = ei_proj[year]
        annual_mwh = hourly_power.sum()  # assuming same power; sensitivity varies this
        annual_dc_emissions[year] = annual_mwh * ei / 1000
        log.info(f"  {year}: EI={ei:.0f} kg/MWh → {annual_dc_emissions[year]:,.0f} tons CO₂")

    emissions_df = pd.DataFrame([
        {"year": y, "ei_kg_per_mwh": ei_proj[y],
         "dc_co2_metric_tons": annual_dc_emissions[y]}
        for y in cfg.FORECAST_YEARS
    ])
    save_csv(emissions_df, "carbon_annual_projections")

    return {
        "hourly_emissions": hourly_emissions,
        "total_2019_tons": total_emissions,
        "co2_trend": co2_trend,
        "ei_projections": ei_proj,
        "annual_dc_emissions": annual_dc_emissions,
    }
