"""
MTFC Sensitivity Analysis — Global Configuration & Financial Assumptions
=========================================================================
Central repository for all constants, file paths, and actuarial parameters
used across the Monte-Carlo, Sobol, Copula, and CBA modules.

ALL grid/carbon and temperature parameters are derived from real CSV data
sources via real_params.py.  Nothing is hardcoded.
"""

import os, pathlib

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "REAL FINAL DATA SOURCES"
ARCHIVE_DATA = ROOT.parent / "_archive" / "ACTUAL_FINAL_MODEL_SUBMISSION" / "paper_assets" / "data"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = ROOT / "figures"

for d in [OUTPUT_DIR, FIGURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Real-data derived parameters (loaded once from CSVs) ────────────────
from real_params import REAL as _R

# ── Data-Centre Physical Model ───────────────────────────────────────────
# PUE baseline and IT power derived from NREL ESIF real data;
# CPU utilisation from Google cluster traces (mean 45%, std 12%).
DC_PARAMS = dict(
    it_capacity_mw        = 30.0,                      # Nameplate IT capacity (MW)
    idle_power_fraction    = 0.40,                      # Fraction of peak consumed at idle
    pue_baseline           = round(_R["pue_mean"], 4),  # From ESIF real data (1.04)
    pue_temp_coef          = 0.008,                     # PUE increase per °F above 65
    pue_cpu_coef           = 0.0003,                    # PUE increase per % CPU utilisation
    cpu_utilization_mean   = 45.0,                      # Average CPU utilisation (%)
    cpu_utilization_std    = 12.0,                      # Std dev of CPU utilisation (%)
    n_servers              = 10_000,                    # Server count
    server_tdp_kw          = 0.5,                       # Thermal design power per server (kW)
)

# ── Grid / Carbon — ALL from real VA generation mix data ─────────────────
GRID = dict(
    carbon_intensity_mean  = round(_R["carbon_intensity_mean"], 1),  # From VA generation 2020+
    carbon_intensity_std   = round(_R["carbon_intensity_std"], 1),   # From VA generation 2020+
    carbon_intensity_min   = round(_R["carbon_intensity_min"], 1),   # Observed monthly min
    carbon_intensity_max   = round(_R["carbon_intensity_max"], 1),   # Observed monthly max
)

# ── Temperature — from real Virginia monthly temperature data ────────────
TEMPERATURE = dict(
    mean_f  = round(_R["temperature_f_mean"], 2),          # 55.67 °F
    std_f   = round(_R["temperature_f_std"], 2),            # 14.48 °F
    min_f   = round(_R["temperature_f_min"], 1),            # 22.8 °F
    max_f   = round(_R["temperature_f_max"], 1),            # 78.7 °F
    recent_mean_f = round(_R["temperature_f_recent_mean"], 2),  # 56.89 (2015+)
    recent_std_f  = round(_R["temperature_f_recent_std"], 2),   # 14.26 (2015+)
    trend_per_decade = round(_R["temperature_trend_f_per_decade"], 4),
)

# ── Financial / Actuarial Assumptions ────────────────────────────────────
FINANCE = dict(
    scc_usd_per_ton                = 190.0,     # EPA Social Cost of Carbon
    carbon_price_low_usd_per_ton   = 95.0,      # Low scenario
    carbon_price_high_usd_per_ton  = 300.0,     # Tail-risk scenario
    energy_price_usd_per_mwh       = 72.0,      # PJM average wholesale
    demand_charge_usd_per_kw_yr    = 180.0,     # Annual demand charge
    discount_rate                  = 0.08,       # For NPV calculations
    horizon_years                  = 10,         # CBA horizon
    contract_peak_mw               = 40.0,       # Contracted peak capacity
    peak_breach_hours              = 120.0,      # Hours exceeding contract
    peak_breach_penalty_usd_per_mwh = 450.0,    # Penalty rate
    electricity_volatility_markup  = 0.05,       # 5 % volatility premium
    cooling_maintenance_annual     = 2_500_000,  # Annual cooling CapEx
)

# ── Monte-Carlo Settings ─────────────────────────────────────────────────
MC = dict(
    n_simulations   = 50_000,
    seed            = 42,
    confidence_levels = [0.90, 0.95, 0.99],
)

# ── Sobol Settings ───────────────────────────────────────────────────────
SOBOL = dict(
    n_samples = 4096,   # Must be power of 2 for Saltelli sampling
    seed      = 42,
)

# ── Scenario Growth Rates (energy forecast) ──────────────────────────────
GROWTH_SCENARIOS = {
    "Conservative": dict(growth_rate=0.05, pue_improvement=0.01,  carbon_intensity_improvement=0.02),
    "Moderate":     dict(growth_rate=0.15, pue_improvement=0.015, carbon_intensity_improvement=0.03),
    "Aggressive":   dict(growth_rate=0.30, pue_improvement=0.005, carbon_intensity_improvement=0.01),
}

# ── Mitigation Levers ───────────────────────────────────────────────────
# energy_saving_pct  = % of *electricity* cost avoided (only PUE lever does this)
# emission_reduction_pct = % of *carbon liability* avoided (SCC-scaled)
MITIGATION_LEVERS = {
    "Dynamic Workload Shifting":  dict(emission_reduction_pct=11.5, energy_saving_pct=0.0,  capex_usd=600_000),
    "PUE Optimization (Cooling)": dict(emission_reduction_pct=18.8, energy_saving_pct=10.0, capex_usd=14_000_000),
    "Cleaner Grid Contracts":     dict(emission_reduction_pct=22.4, energy_saving_pct=0.0,  capex_usd=200_000),
    "Combined Portfolio":         dict(emission_reduction_pct=41.9, energy_saving_pct=10.0, capex_usd=15_200_000),
}

# ── Plotting Style ───────────────────────────────────────────────────────
PLOT = dict(
    figsize_wide   = (14, 6),
    figsize_square = (10, 8),
    figsize_heatmap = (14, 10),
    dpi            = 200,
    style          = "seaborn-v0_8-whitegrid",
    palette        = "RdYlGn_r",
)
