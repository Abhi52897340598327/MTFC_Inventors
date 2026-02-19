"""
MTFC Virginia Datacenter Energy Forecasting — PJM Grid Carbon Intensity Analysis
==================================================================================
Downloads EIA-930 grid balance data, calculates hourly carbon intensity for PJM,
and creates visualizations of the grid's emissions profile.

Data Source: EIA Hourly Electric Grid Monitor
https://www.eia.gov/electricity/gridmonitor/
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

# ── Output Directories ──────────────────────────────────────────────────────
DATA_DIR = os.path.join(cfg.PROJECT_DIR, "Data_Sources")
FIGURE_DIR = cfg.FIGURE_DIR
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


def download_eia_grid_data():
    """Download EIA-930 grid balance data for PJM 2019."""
    print("=" * 70)
    print("DOWNLOADING EIA GRID DATA (PJM 2019)")
    print("=" * 70)
    
    urls = [
        "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2019_Jan_Jun.csv",
        "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2019_Jul_Dec.csv"
    ]
    
    print("\n⬇️ Downloading EIA Grid Data (PJM 2019)...")
    dfs = []
    for i, url in enumerate(urls):
        print(f"  Downloading file {i+1}/2: {url.split('/')[-1]}")
        try:
            chunk = pd.read_csv(url, low_memory=False)
            # Filter immediately for PJM
            chunk = chunk[chunk['Balancing Authority'] == 'PJM'].copy()
            dfs.append(chunk)
            print(f"    ✓ Got {len(chunk)} PJM records")
        except Exception as e:
            print(f"    ✗ Error: {e}")
            return None
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"\n✅ Loaded {len(df)} total hourly records.")
    
    return df


def process_carbon_intensity(df):
    """Calculate carbon intensity from generation mix."""
    print("\n" + "=" * 70)
    print("PROCESSING CARBON INTENSITY")
    print("=" * 70)
    
    # Strip whitespace from all column names
    df.columns = df.columns.str.strip()
    
    # Emission factors (kg CO2 per MWh)
    fuel_map = {
        'Coal': 980,
        'Natural Gas': 420,
        'Petroleum': 800,
        'Nuclear': 0,
        'Hydro': 0,
        'Solar': 0,
        'Wind': 0,
        'Other': 700
    }
    
    # Dynamically find the actual column names in the dataframe
    found_cols = []
    df['total_gen_mw'] = 0
    df['total_emissions_kg'] = 0
    
    print("\n🛠️ Mapping Columns & Calculating Emissions...")
    
    for fuel, factor in fuel_map.items():
        # Find column that contains "Net Generation" AND the Fuel Name
        matches = [c for c in df.columns if 'Net Generation' in c and fuel in c]
        
        if matches:
            col_name = matches[0]
            found_cols.append(col_name)
            
            # Clean data (remove commas, handle NaNs)
            df[col_name] = pd.to_numeric(
                df[col_name].astype(str).str.replace(',', ''), 
                errors='coerce'
            ).fillna(0)
            df[col_name] = df[col_name].clip(lower=0)  # No negative generation
            
            # Add to totals
            df['total_gen_mw'] += df[col_name]
            df['total_emissions_kg'] += df[col_name] * factor
            print(f"   ✓ Mapped '{fuel}' to column: '{col_name[:50]}...' (Factor: {factor})")
        else:
            print(f"   ⚠️ Warning: Could not find column for '{fuel}'")
    
    # Clean timestamp
    def clean_timestamp(date_str, hour_str):
        try:
            hr = int(hour_str)
            dt = pd.to_datetime(date_str, format='%m/%d/%Y')
            if hr == 24:
                return dt + pd.Timedelta(days=1)
            return dt + pd.Timedelta(hours=hr)
        except:
            return pd.NaT
    
    print("\n🕐 Processing timestamps...")
    df['timestamp'] = df.apply(
        lambda x: clean_timestamp(x['Data Date'], x['Hour Number']), 
        axis=1
    )
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Calculate intensity (avoid division by zero)
    df = df[df['total_gen_mw'] > 0].copy()
    df['carbon_intensity'] = df['total_emissions_kg'] / df['total_gen_mw']
    
    print(f"\n✅ Processed {len(df)} valid records with carbon intensity.")
    
    return df, found_cols


def save_datasets(df, found_cols):
    """Save processed datasets to Data_Sources folder."""
    print("\n" + "=" * 70)
    print("SAVING DATASETS")
    print("=" * 70)
    
    # 1. Save full processed dataset
    cols_to_save = ['timestamp', 'total_gen_mw', 'total_emissions_kg', 'carbon_intensity'] + found_cols
    cols_to_save = [c for c in cols_to_save if c in df.columns]
    
    full_path = os.path.join(DATA_DIR, "pjm_carbon_intensity_2019_hourly.csv")
    df[cols_to_save].to_csv(full_path, index=False)
    print(f"✓ Saved: pjm_carbon_intensity_2019_hourly.csv ({len(df)} rows)")
    
    # 2. Save daily aggregated version
    df_daily = df.set_index('timestamp').resample('D').agg({
        'total_gen_mw': 'mean',
        'total_emissions_kg': 'sum',
        'carbon_intensity': 'mean'
    }).reset_index()
    
    daily_path = os.path.join(DATA_DIR, "pjm_carbon_intensity_2019_daily.csv")
    df_daily.to_csv(daily_path, index=False)
    print(f"✓ Saved: pjm_carbon_intensity_2019_daily.csv ({len(df_daily)} rows)")
    
    # 3. Save weekly generation mix
    df_weekly = df.set_index('timestamp')[found_cols].resample('W').mean().reset_index()
    
    weekly_path = os.path.join(DATA_DIR, "pjm_generation_mix_2019_weekly.csv")
    df_weekly.to_csv(weekly_path, index=False)
    print(f"✓ Saved: pjm_generation_mix_2019_weekly.csv ({len(df_weekly)} rows)")
    
    return df_daily, df_weekly


def create_visualizations(df, found_cols):
    """Create and save carbon intensity visualizations."""
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    
    # Figure 1: Carbon Intensity Timeline + Generation Mix
    fig = plt.figure(figsize=(14, 10))
    
    # Graph 1: Intensity Timeline
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df['timestamp'], df['carbon_intensity'], color='#333333', linewidth=0.5, alpha=0.7)
    
    # Add rolling average
    df['intensity_rolling'] = df['carbon_intensity'].rolling(24*7).mean()
    ax1.plot(df['timestamp'], df['intensity_rolling'], color='red', linewidth=2, 
             label=f"7-Day Average")
    
    ax1.axhline(df['carbon_intensity'].mean(), color='blue', linestyle='--', linewidth=2,
                label=f"Annual Mean: {df['carbon_intensity'].mean():.0f} kg/MWh")
    
    ax1.set_title('PJM Grid Carbon Intensity (2019) - The Emissions Risk Timeline', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Carbon Intensity (kg CO₂ / MWh)', fontsize=12)
    ax1.set_xlabel('')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add annotations for peak periods
    max_idx = df['carbon_intensity'].idxmax()
    ax1.annotate(f"Peak: {df.loc[max_idx, 'carbon_intensity']:.0f}",
                xy=(df.loc[max_idx, 'timestamp'], df.loc[max_idx, 'carbon_intensity']),
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, color='red')
    
    # Graph 2: Stacked Generation Mix
    ax2 = plt.subplot(2, 1, 2)
    
    # Resample to weekly for readability
    df_weekly = df.set_index('timestamp')[found_cols].resample('W').mean()
    
    # Clean up column labels for legend
    clean_labels = []
    for c in found_cols:
        # Extract fuel type from column name like "Net Generation (MW) from Coal"
        if ' - ' in c:
            label = c.split(' - ')[1].replace(' (MW)', '').strip()
        elif 'from ' in c:
            label = c.split('from ')[-1].replace(' (MW)', '').strip()
        else:
            label = c
        clean_labels.append(label)
    
    # Color palette for fuels
    fuel_colors = {
        'Coal': '#4a4a4a',
        'Natural Gas': '#3498db', 
        'Nuclear': '#9b59b6',
        'Hydro': '#1abc9c',
        'Wind': '#2ecc71',
        'Solar': '#f1c40f',
        'Petroleum': '#e74c3c',
        'Other': '#95a5a6'
    }
    
    colors = []
    for label in clean_labels:
        matched = False
        for fuel, color in fuel_colors.items():
            if fuel.lower() in label.lower():
                colors.append(color)
                matched = True
                break
        if not matched:
            colors.append('#95a5a6')
    
    ax2.stackplot(df_weekly.index, df_weekly.T, labels=clean_labels, colors=colors, alpha=0.8)
    ax2.set_title('Weekly Generation Mix (The Source of Emissions)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Generation (MW)', fontsize=12)
    ax2.set_xlabel('')
    ax2.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(FIGURE_DIR, 'pjm_carbon_intensity_2019.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: pjm_carbon_intensity_2019.png")
    
    # Figure 2: Monthly Carbon Intensity Boxplot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df['month'] = df['timestamp'].dt.month
    df['month_name'] = df['timestamp'].dt.strftime('%b')
    
    monthly_data = [df[df['month'] == m]['carbon_intensity'].values for m in range(1, 13)]
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    bp = ax.boxplot(monthly_data, labels=months, patch_artist=True)
    
    # Color by intensity
    monthly_means = [np.mean(d) for d in monthly_data]
    norm = plt.Normalize(min(monthly_means), max(monthly_means))
    cmap = plt.cm.RdYlGn_r
    
    for patch, mean in zip(bp['boxes'], monthly_means):
        patch.set_facecolor(cmap(norm(mean)))
        patch.set_alpha(0.7)
    
    ax.axhline(df['carbon_intensity'].mean(), color='blue', linestyle='--', 
               label=f"Annual Mean: {df['carbon_intensity'].mean():.0f}")
    
    ax.set_title('Monthly Carbon Intensity Distribution (PJM 2019)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Carbon Intensity (kg CO₂ / MWh)', fontsize=12)
    ax.set_xlabel('Month', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, 'pjm_carbon_intensity_monthly.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: pjm_carbon_intensity_monthly.png")
    
    # Figure 3: Hourly Pattern Heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    df['hour'] = df['timestamp'].dt.hour
    
    heatmap_data = df.pivot_table(
        values='carbon_intensity',
        index='hour',
        columns='month',
        aggfunc='mean'
    )
    
    import seaborn as sns
    sns.heatmap(heatmap_data, cmap='RdYlGn_r', annot=True, fmt='.0f',
                cbar_kws={'label': 'Avg Carbon Intensity (kg/MWh)'}, ax=ax)
    
    ax.set_xticklabels(months)
    ax.set_ylabel('Hour of Day', fontsize=12)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_title('Carbon Intensity by Hour and Month\n(When to Schedule Workloads for Lower Emissions)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, 'pjm_carbon_intensity_heatmap.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: pjm_carbon_intensity_heatmap.png")


def print_summary(df):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("📊 CARBON INTENSITY SUMMARY STATISTICS")
    print("=" * 70)
    
    print(f"\nPJM Grid (2019) - {len(df):,} hourly observations")
    print("-" * 50)
    print(f"Average Intensity:  {df['carbon_intensity'].mean():.2f} kg CO₂/MWh")
    print(f"Median Intensity:   {df['carbon_intensity'].median():.2f} kg CO₂/MWh")
    print(f"Std Deviation:      {df['carbon_intensity'].std():.2f} kg CO₂/MWh")
    print(f"Min Intensity:      {df['carbon_intensity'].min():.2f} kg CO₂/MWh")
    print(f"Max Intensity:      {df['carbon_intensity'].max():.2f} kg CO₂/MWh")
    
    print(f"\nTotal Generation:   {df['total_gen_mw'].sum()/1e6:.2f} TWh")
    print(f"Total Emissions:    {df['total_emissions_kg'].sum()/1e9:.2f} Million metric tons CO₂")
    
    # Best/worst hours for scheduling
    df['hour'] = df['timestamp'].dt.hour
    hourly_avg = df.groupby('hour')['carbon_intensity'].mean()
    
    print(f"\nBest hours for low-carbon workloads:")
    for hr in hourly_avg.nsmallest(3).index:
        print(f"  {hr:02d}:00 - {hourly_avg[hr]:.0f} kg/MWh")
    
    print(f"\nWorst hours (highest emissions):")
    for hr in hourly_avg.nlargest(3).index:
        print(f"  {hr:02d}:00 - {hourly_avg[hr]:.0f} kg/MWh")


def main():
    """Run the full PJM carbon intensity analysis."""
    print("\n" + "=" * 70)
    print("MTFC PJM GRID CARBON INTENSITY ANALYSIS")
    print("=" * 70)
    
    # Download data
    df = download_eia_grid_data()
    if df is None:
        print("❌ Failed to download data. Exiting.")
        return
    
    # Process carbon intensity
    df, found_cols = process_carbon_intensity(df)
    
    # Save datasets
    save_datasets(df, found_cols)
    
    # Create visualizations
    create_visualizations(df, found_cols)
    
    # Print summary
    print_summary(df)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nDatasets saved to: {DATA_DIR}")
    print(f"Figures saved to:  {FIGURE_DIR}")


if __name__ == "__main__":
    main()
