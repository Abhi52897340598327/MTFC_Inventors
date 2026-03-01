"""
MTFC Data Preparation Module
Prepares all time series for SARIMA/SARIMAX modeling.

Temporal Alignment: 2015-01-01 to 2025-08-01 (128 months)

Key Design Decisions:
- Energy & grid mix use actual monthly EIA generation data (real seasonality)
- PUE is clipped to [1.0, 3.0] (physical constraint: PUE >= 1.0)
- AI proxy is normalized to 2015-01 = 1.0 (historical baseline)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Directory Configuration
BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / 'REAL FINAL DATA SOURCES'
OUTPUT_DIR = BASE_DIR.parent / 'REAL FINAL FILES' / 'prepared_data'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Constants
START_DATE = '2015-01-01'
END_DATE = '2025-08-01'
HISTORICAL_PUE_ESTIMATE = 1.50  # For months before PUE data availability
EXPECTED_MONTHS = len(pd.date_range(START_DATE, END_DATE, freq='MS'))  # 128

# Virginia share of US datacenter construction spending
# Source: CBRE North America Data Center Trends (2024), JLL Data Center Outlook
# Virginia (NoVA / Loudoun County) hosts ~25-30% of US datacenter capacity.
# Time-varying: ~20% in 2014 growing to ~30% by 2025 reflecting VA's increasing dominance.
VA_DC_SHARE_START = 0.20   # 2014 share
VA_DC_SHARE_END   = 0.30   # 2025 share
VA_DC_SHARE_START_DATE = pd.Timestamp('2014-01-01')
VA_DC_SHARE_END_DATE   = pd.Timestamp('2025-08-01')

# Source mapping: EIA energy source names -> our 4 grid categories
SOURCE_MAP = {
    'Coal': 'coal',
    'Natural Gas': 'gas',
    'Nuclear': 'nuclear',
    'Hydroelectric Conventional': 'renewable',
    'Solar Thermal and Photovoltaic': 'renewable',
    'Wind': 'renewable',
    'Wood and Wood Derived Fuels': 'renewable',
    'Other Biomass': 'renewable',
}


def _load_eia_generation() -> pd.DataFrame:
    """
    Load and parse the EIA monthly generation dataset.

    Returns DataFrame with columns:
        date, energy_source, generation_mwh
    filtered to 'Total Electric Power Industry' only.
    """
    df = pd.read_csv(INPUT_DIR / 'virginia_generation_all_years.csv')
    df = df[df['TYPE OF PRODUCER'] == 'Total Electric Power Industry'].copy()
    df['date'] = pd.to_datetime(
        df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str).str.zfill(2) + '-01'
    )
    return df


def prepare_monthly_energy() -> pd.DataFrame:
    """
    Extract actual monthly electricity generation from EIA state data.

    Uses REAL monthly generation (not interpolated from annual), providing
    genuine seasonal patterns for SARIMAX modeling.

    Source: virginia_generation_all_years.csv (EIA Form 923 / 860M)
    Filter: Total Electric Power Industry, 'Total' source
            (net generation including pumped-storage deduction)
    Units:  GWh/month

    Output: monthly_energy_consumption.csv
    """
    df = _load_eia_generation()

    # 'Total' row = net generation across all sources (includes pumped-storage deduction)
    total = df[df['ENERGY SOURCE'] == 'Total'].copy()
    total = total.sort_values('date')

    # Convert MWh -> GWh
    total['electricity_gwh'] = total['GENERATION (Megawatthours)'] / 1000.0

    # Filter to analysis period
    mask = (total['date'] >= START_DATE) & (total['date'] <= END_DATE)
    total = total[mask]

    result = total[['date', 'electricity_gwh']].reset_index(drop=True)

    # Verify output
    assert len(result) == EXPECTED_MONTHS, f"Expected {EXPECTED_MONTHS} months, got {len(result)}"
    assert result['electricity_gwh'].isnull().sum() == 0, "Null electricity values"
    assert (result['electricity_gwh'] > 0).all(), "Negative or zero generation"

    result.to_csv(OUTPUT_DIR / 'monthly_energy_consumption.csv', index=False)

    print(f"  Energy: {len(result)} months (actual monthly EIA generation)")
    print(f"    Range: {result['electricity_gwh'].min():,.0f} to "
          f"{result['electricity_gwh'].max():,.0f} GWh/month")
    print(f"    Mean: {result['electricity_gwh'].mean():,.0f} GWh/month")
    print(f"    Annual: {result['electricity_gwh'].mean() * 12:,.0f} GWh/year")

    return result


def prepare_monthly_grid_mix() -> pd.DataFrame:
    """
    Calculate monthly grid composition from actual generation by source.

    Uses real monthly EIA generation data, giving genuine seasonal variation
    in grid composition (e.g., more gas in summer, more nuclear share in
    shoulder months).

    Source categories:
        Coal:      Coal
        Gas:       Natural Gas
        Nuclear:   Nuclear
        Renewable: Hydro + Solar + Wind + Wood + Other Biomass
    Excluded:  Petroleum, Pumped Storage, Other, Other Gases

    Output: monthly_grid_mix.csv
    """
    df = _load_eia_generation()

    # Keep only sources that map to our 4 categories
    relevant = df[df['ENERGY SOURCE'].isin(SOURCE_MAP.keys())].copy()
    relevant['category'] = relevant['ENERGY SOURCE'].map(SOURCE_MAP)

    # Sum MWh by (date, category)
    grouped = (
        relevant
        .groupby(['date', 'category'])['GENERATION (Megawatthours)']
        .sum()
        .unstack(fill_value=0)
    )

    # Ensure all 4 categories exist (Wind may be missing in early years)
    for cat in ['coal', 'gas', 'nuclear', 'renewable']:
        if cat not in grouped.columns:
            grouped[cat] = 0

    # Calculate total and percentages
    total_gen = grouped[['coal', 'gas', 'nuclear', 'renewable']].sum(axis=1)

    pct_cols = ['coal_pct', 'gas_pct', 'nuclear_pct', 'renewable_pct']
    grouped['coal_pct'] = grouped['coal'] / total_gen * 100
    grouped['gas_pct'] = grouped['gas'] / total_gen * 100
    grouped['nuclear_pct'] = grouped['nuclear'] / total_gen * 100
    grouped['renewable_pct'] = grouped['renewable'] / total_gen * 100

    # Filter to analysis period
    mask = (grouped.index >= START_DATE) & (grouped.index <= END_DATE)
    monthly = grouped.loc[mask, pct_cols].copy()

    result = monthly.reset_index().rename(columns={'index': 'date'})

    # Verify output
    assert len(result) == EXPECTED_MONTHS, f"Expected {EXPECTED_MONTHS} months, got {len(result)}"
    row_sums = result[pct_cols].sum(axis=1)
    assert np.allclose(row_sums, 100.0, atol=0.01), (
        f"Grid mix does not sum to 100%: {row_sums.min():.2f}-{row_sums.max():.2f}"
    )

    result.to_csv(OUTPUT_DIR / 'monthly_grid_mix.csv', index=False)

    print(f"  Grid Mix: {len(result)} months (actual monthly EIA by source)")
    for col in pct_cols:
        print(f"    {col}: {result[col].iloc[0]:.2f}% -> {result[col].iloc[-1]:.2f}%")

    return result


def prepare_monthly_pue() -> pd.DataFrame:
    """
    Aggregate daily PUE data to monthly averages.

    Mathematical Process:
    1. Clip to physical bounds [1.0, 3.0] (PUE >= 1.0 by definition)
    2. Calculate monthly mean
    3. Fill missing months (2015-01 to 2015-10) with historical estimate 1.50

    Input: esif_daily_avg_interpolated.csv
    Output: monthly_pue.csv
    """
    df = pd.read_csv(INPUT_DIR / 'esif_daily_avg_interpolated.csv')
    df['date'] = pd.to_datetime(df['date'])

    # PUE must be >= 1.0 by definition (total facility power / IT equipment power)
    df['pue'] = df['pue'].clip(lower=1.0, upper=3.0)

    # Monthly mean via resample
    df = df.set_index('date')
    monthly_pue = df['pue'].resample('MS').mean().reset_index()
    monthly_pue.columns = ['date', 'pue']

    # Create complete timeline
    all_months = pd.date_range(START_DATE, END_DATE, freq='MS')
    full_timeline = pd.DataFrame({'date': all_months})

    # Merge and fill missing months
    result = full_timeline.merge(monthly_pue, on='date', how='left')
    result['pue'] = result['pue'].fillna(HISTORICAL_PUE_ESTIMATE)

    # Verify output
    assert len(result) == EXPECTED_MONTHS, f"Expected {EXPECTED_MONTHS} months, got {len(result)}"
    assert result['pue'].isnull().sum() == 0, "Null PUE values detected"
    assert result['pue'].min() >= 1.0, f"Invalid PUE < 1.0: {result['pue'].min()}"

    result.to_csv(OUTPUT_DIR / 'monthly_pue.csv', index=False)

    print(f"  PUE: {len(result)} months, range {result['pue'].min():.3f} to "
          f"{result['pue'].max():.3f}")

    return result


def _va_dc_share(dates: pd.Series) -> pd.Series:
    """Compute time-varying Virginia datacenter share of US spending.

    Linearly interpolates from VA_DC_SHARE_START (2014) to VA_DC_SHARE_END (2025).
    Clipped to [VA_DC_SHARE_START, VA_DC_SHARE_END] outside the range.
    """
    total_days = (VA_DC_SHARE_END_DATE - VA_DC_SHARE_START_DATE).days
    elapsed = (dates - VA_DC_SHARE_START_DATE).dt.days.astype(float)
    frac = np.clip(elapsed / total_days, 0.0, 1.0)
    return VA_DC_SHARE_START + frac * (VA_DC_SHARE_END - VA_DC_SHARE_START)


def prepare_monthly_ai_proxy() -> pd.DataFrame:
    """
    Prepare datacenter spending as AI growth proxy (Virginia-adjusted).

    Mathematical Process:
    1. Load US monthly spending data
    2. Apply time-varying Virginia share (20% in 2014 → 30% in 2025)
    3. Filter to analysis period
    4. Normalize to baseline (2015-01 = 1.0)

    Assumption: Virginia's share of US DC construction grows linearly
    from ~20% to ~30% over 2014-2025 (CBRE, JLL industry data).

    Input: monthly-spending-data-center-us.csv
    Output: monthly_ai_proxy.csv
    """
    df = pd.read_csv(INPUT_DIR / 'monthly-spending-data-center-us.csv')

    # Rename columns for consistency
    df = df.rename(columns={
        'Day': 'date',
        'Monthly spending on data center construction in the United States': 'ai_proxy'
    })

    df['date'] = pd.to_datetime(df['date'])

    # Apply time-varying Virginia share of US spending
    va_share = _va_dc_share(df['date'])
    df['ai_proxy'] = df['ai_proxy'] * va_share
    print(f"  VA DC share applied: {va_share.iloc[0]:.1%} (start) → {va_share.iloc[-1]:.1%} (end)")

    # Filter to analysis period
    df = df[(df['date'] >= START_DATE) & (df['date'] <= END_DATE)].copy()

    # Normalize to baseline (2015-01 = 1.0)
    baseline_value = float(df.loc[df['date'] == START_DATE, 'ai_proxy'].values[0])
    df['ai_proxy'] = df['ai_proxy'] / baseline_value

    result = df[['date', 'ai_proxy']].reset_index(drop=True)

    # Verify output
    assert len(result) == EXPECTED_MONTHS, f"Expected {EXPECTED_MONTHS} months, got {len(result)}"
    assert result['ai_proxy'].isnull().sum() == 0, "Null values detected"
    assert np.isclose(result['ai_proxy'].iloc[0], 1.0, atol=1e-6), "Baseline not 1.0"

    result.to_csv(OUTPUT_DIR / 'monthly_ai_proxy.csv', index=False)

    print(f"  AI Proxy: {len(result)} months, growth 1.00 -> "
          f"{result['ai_proxy'].iloc[-1]:.2f}")

    return result


def prepare_monthly_temperature() -> pd.DataFrame:
    """
    Extract and prepare monthly temperature data.

    Input: monthly_temp_virginia.csv
    Output: monthly_temperature.csv
    """
    df = pd.read_csv(INPUT_DIR / 'monthly_temp_virginia.csv')

    # Parse month column (format: "January 2026")
    df['date'] = pd.to_datetime(df['month'], format='%B %Y')

    # Extract temperature column
    df = df.rename(columns={'avg_temp_f': 'temperature'})

    # Filter to analysis period
    df = df[(df['date'] >= START_DATE) & (df['date'] <= END_DATE)]

    result = df[['date', 'temperature']].copy()
    result = result.sort_values('date').reset_index(drop=True)

    # Verify output
    assert len(result) == EXPECTED_MONTHS, f"Expected {EXPECTED_MONTHS} months, got {len(result)}"
    assert result['temperature'].isnull().sum() == 0, "Null temperature values"

    result.to_csv(OUTPUT_DIR / 'monthly_temperature.csv', index=False)

    print(f"  Temperature: {len(result)} months, range "
          f"{result['temperature'].min():.1f}F to {result['temperature'].max():.1f}F")

    return result


def run_data_preparation() -> tuple:
    """
    Execute all data preparation steps in sequence.

    Output Files:
    - monthly_energy_consumption.csv
    - monthly_grid_mix.csv
    - monthly_pue.csv
    - monthly_ai_proxy.csv
    - monthly_temperature.csv
    """
    print("=" * 70)
    print(" DATA PREPARATION PHASE ".center(70))
    print(" Preparing Monthly Time Series from EIA Generation Data ".center(70))
    print("=" * 70)
    print(f"\nAnalysis Period: {START_DATE} to {END_DATE} ({EXPECTED_MONTHS} months)")
    print(f"Output Directory: {OUTPUT_DIR}\n")

    energy = prepare_monthly_energy()
    grid = prepare_monthly_grid_mix()
    pue = prepare_monthly_pue()
    ai = prepare_monthly_ai_proxy()
    temp = prepare_monthly_temperature()

    print("\n" + "=" * 70)
    print(" DATA PREPARATION COMPLETE ".center(70))
    print("=" * 70)
    print(f"\n  All files saved to: {OUTPUT_DIR}")
    print(f"  All time series aligned: {EXPECTED_MONTHS} months ({START_DATE[:7]} to {END_DATE[:7]})")
    print("  Ready for time series modeling\n")

    return energy, grid, pue, ai, temp


if __name__ == '__main__':
    run_data_preparation()
