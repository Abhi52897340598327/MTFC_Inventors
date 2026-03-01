"""
MTFC Data Preparation Module
Converts annual energy data to monthly frequency and prepares all time series
for SARIMA/SARIMAX modeling.

Temporal Alignment: 2015-01-01 to 2023-12-31 (108 months)

Key Design Decisions:
- Annual BBTU values are divided by 12 to get monthly consumption
- Grid percentages are interpolated directly (already in ratio form)
- PUE is clipped to [1.0, 3.0] (physical constraint: PUE >= 1.0)
- AI proxy is normalized to 2015-01 = 1.0 (historical baseline)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Directory Configuration
BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / 'REAL FINAL DATA SOURCES'
OUTPUT_DIR = BASE_DIR / 'REAL FINAL FILES' / 'prepared_data'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Constants
BBTU_TO_GWH = 0.293071
START_DATE = '2015-01-01'
END_DATE = '2023-12-31'
HISTORICAL_PUE_ESTIMATE = 1.50  # For months before PUE data availability


def _annual_to_monthly(annual_df: pd.DataFrame, value_cols: list,
                       divide_by_12: bool = True) -> pd.DataFrame:
    """
    Convert annual data to monthly via linear interpolation.

    Places each annual value at Jan 1 of that year, resamples to monthly start
    frequency, then linearly interpolates between annual anchor points.

    Parameters:
        annual_df: DataFrame with 'Year' column and value columns
        value_cols: Column names to interpolate
        divide_by_12: If True, divide by 12 (annual totals -> monthly amounts)

    Returns:
        DataFrame with DatetimeIndex at monthly frequency
    """
    df = annual_df[['Year'] + value_cols].copy()
    df['date'] = pd.to_datetime(df['Year'].astype(int), format='%Y')
    df = df.set_index('date')[value_cols]

    # Create full monthly range covering all years (Jan of first year to Dec of last year)
    first_year = int(annual_df['Year'].min())
    last_year = int(annual_df['Year'].max())
    full_range = pd.date_range(f'{first_year}-01-01', f'{last_year}-12-01', freq='MS')

    # Reindex to full monthly range, interpolate, then forward-fill last year
    monthly = df.reindex(full_range).interpolate(method='linear').ffill()

    if divide_by_12:
        monthly = monthly / 12.0

    return monthly


def prepare_monthly_energy() -> pd.DataFrame:
    """
    Convert annual energy data to monthly and separate total vs electricity.

    Mathematical Process:
    1. Total Energy: All fuel types for all sectors (annual BBTU)
    2. Grid Electricity: Coal + Gas + Nuclear + Renewable (annual BBTU)
    3. Linear interpolation to monthly, then divide by 12
    4. Convert units: E_GWh = E_BBTU * 0.293071

    Input: virginia_yearly_energy_consumption_bbtu.csv
           energy_by_source_annual_grid_comp.csv
    Output: monthly_energy_consumption.csv (with both total and electricity)
    """
    # Load total energy
    total_energy_df = pd.read_csv(INPUT_DIR / 'virginia_yearly_energy_consumption_bbtu.csv')
    total_energy_df = total_energy_df.rename(
        columns={'Total Energy Consumption (Billion BTU)': 'Total_BBTU'}
    )

    # Load grid sources
    grid_df = pd.read_csv(INPUT_DIR / 'energy_by_source_annual_grid_comp.csv')

    # Calculate electricity-only consumption (exclude petroleum - used for transport)
    grid_df['Electricity_BBTU'] = (
        grid_df['Coal_Total_Consumption_Billion_BTU']
        + grid_df['Natural_Gas_Total_Consumption_Billion_BTU']
        + grid_df['Nuclear_Total_Consumption_Billion_BTU']
        + grid_df['Renewable_Total_Consumption_Billion_BTU']
    )

    # Merge on Year
    df = total_energy_df.merge(grid_df[['Year', 'Electricity_BBTU']], on='Year')

    # Interpolate to monthly and divide by 12 (annual -> monthly)
    monthly = _annual_to_monthly(df, ['Total_BBTU', 'Electricity_BBTU'], divide_by_12=True)

    # Filter to analysis period
    monthly = monthly.loc[START_DATE:END_DATE]

    # Convert BBTU to GWh
    monthly['total_energy_gwh'] = monthly['Total_BBTU'] * BBTU_TO_GWH
    monthly['electricity_gwh'] = monthly['Electricity_BBTU'] * BBTU_TO_GWH

    # Build output
    result = monthly[['total_energy_gwh', 'electricity_gwh']].copy()
    result = result.reset_index().rename(columns={'index': 'date'})

    # Verify output
    assert len(result) == 108, f"Expected 108 months, got {len(result)}"
    assert result['total_energy_gwh'].isnull().sum() == 0, "Null total energy values"
    assert result['electricity_gwh'].isnull().sum() == 0, "Null electricity values"

    result.to_csv(OUTPUT_DIR / 'monthly_energy_consumption.csv', index=False)

    print(f"  Energy: {len(result)} months")
    print(f"    Total Energy: {result['total_energy_gwh'].min():.1f} to "
          f"{result['total_energy_gwh'].max():.1f} GWh/month")
    print(f"    Electricity: {result['electricity_gwh'].min():.1f} to "
          f"{result['electricity_gwh'].max():.1f} GWh/month")
    print(f"    Electricity Share: "
          f"{(result['electricity_gwh'].mean() / result['total_energy_gwh'].mean() * 100):.1f}%")

    return result


def prepare_monthly_grid_mix() -> pd.DataFrame:
    """
    Convert annual grid composition to monthly percentages.

    Mathematical Process:
    1. Combine petroleum into natural gas (small contribution)
    2. Calculate annual percentages
    3. Linear interpolation to monthly (NO divide by 12 - these are ratios)
    4. Normalize each row to sum to 100%

    Input: energy_by_source_annual_grid_comp.csv
    Output: monthly_grid_mix.csv
    """
    df = pd.read_csv(INPUT_DIR / 'energy_by_source_annual_grid_comp.csv')

    # Combine petroleum with natural gas
    df['Gas_Combined'] = (
        df['Natural_Gas_Total_Consumption_Billion_BTU']
        + df['Petroleum_Total_Consumption_Billion_BTU']
    )

    # Calculate total consumption
    df['Total'] = (
        df['Coal_Total_Consumption_Billion_BTU']
        + df['Gas_Combined']
        + df['Nuclear_Total_Consumption_Billion_BTU']
        + df['Renewable_Total_Consumption_Billion_BTU']
    )

    # Calculate percentages (annual)
    df['coal_pct'] = df['Coal_Total_Consumption_Billion_BTU'] / df['Total'] * 100
    df['gas_pct'] = df['Gas_Combined'] / df['Total'] * 100
    df['nuclear_pct'] = df['Nuclear_Total_Consumption_Billion_BTU'] / df['Total'] * 100
    df['renewable_pct'] = df['Renewable_Total_Consumption_Billion_BTU'] / df['Total'] * 100

    pct_cols = ['coal_pct', 'gas_pct', 'nuclear_pct', 'renewable_pct']

    # Interpolate to monthly (percentages, NOT divided by 12)
    monthly = _annual_to_monthly(df, pct_cols, divide_by_12=False)

    # Filter to analysis period
    monthly = monthly.loc[START_DATE:END_DATE]

    # Normalize to ensure sum = 100% (compositional constraint)
    row_sums = monthly[pct_cols].sum(axis=1)
    monthly[pct_cols] = monthly[pct_cols].div(row_sums, axis=0) * 100

    result = monthly[pct_cols].reset_index().rename(columns={'index': 'date'})

    # Verify output
    assert len(result) == 108, f"Expected 108 months, got {len(result)}"
    row_sum_check = result[pct_cols].sum(axis=1)
    assert np.allclose(row_sum_check, 100.0, atol=1e-6), "Grid mix does not sum to 100%"

    result.to_csv(OUTPUT_DIR / 'monthly_grid_mix.csv', index=False)

    print(f"  Grid Mix: {len(result)} months, composition verified (sum=100%)")
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
    assert len(result) == 108, f"Expected 108 months, got {len(result)}"
    assert result['pue'].isnull().sum() == 0, "Null PUE values detected"
    assert result['pue'].min() >= 1.0, f"Invalid PUE < 1.0: {result['pue'].min()}"

    result.to_csv(OUTPUT_DIR / 'monthly_pue.csv', index=False)

    print(f"  PUE: {len(result)} months, range {result['pue'].min():.3f} to "
          f"{result['pue'].max():.3f}")

    return result


def prepare_monthly_ai_proxy() -> pd.DataFrame:
    """
    Prepare datacenter spending as AI growth proxy.

    Mathematical Process:
    1. Load monthly spending data
    2. Filter to analysis period
    3. Normalize to baseline (2015-01 = 1.0)

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

    # Filter to analysis period
    df = df[(df['date'] >= START_DATE) & (df['date'] <= END_DATE)].copy()

    # Normalize to baseline (2015-01 = 1.0)
    baseline_value = float(df.loc[df['date'] == START_DATE, 'ai_proxy'].values[0])
    df['ai_proxy'] = df['ai_proxy'] / baseline_value

    result = df[['date', 'ai_proxy']].reset_index(drop=True)

    # Verify output
    assert len(result) == 108, f"Expected 108 months, got {len(result)}"
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
    assert len(result) == 108, f"Expected 108 months, got {len(result)}"
    assert result['temperature'].isnull().sum() == 0, "Null temperature values"

    result.to_csv(OUTPUT_DIR / 'monthly_temperature.csv', index=False)

    print(f"  Temperature: {len(result)} months, range "
          f"{result['temperature'].min():.1f}F to {result['temperature'].max():.1f}F")

    return result


def run_data_preparation() -> tuple:
    """
    Execute all data preparation steps in sequence.

    Output Files:
    - monthly_energy_consumption.csv (108 rows)
    - monthly_grid_mix.csv (108 rows)
    - monthly_pue.csv (108 rows)
    - monthly_ai_proxy.csv (108 rows)
    - monthly_temperature.csv (108 rows)
    """
    print("=" * 70)
    print(" DATA PREPARATION PHASE ".center(70))
    print(" Converting Annual to Monthly & Aligning Time Series ".center(70))
    print("=" * 70)
    print(f"\nAnalysis Period: {START_DATE} to {END_DATE} (108 months)")
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
    print("  All time series aligned: 108 months (2015-01 to 2023-12)")
    print("  Ready for time series modeling\n")

    return energy, grid, pue, ai, temp


if __name__ == '__main__':
    run_data_preparation()
