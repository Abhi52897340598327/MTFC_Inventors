"""
MTFC Virginia Datacenter Energy Forecasting — Actuarial Risk Analysis
======================================================================
Analyze temperature-driven cooling risk using actuarial methodology:
  - Timeline visualization of thermal stress periods
  - Probability distribution of cooling penalty hours
  - Quantification of "free cooling" loss zones
  
The Free Cooling Limit (15°C / 59°F) is the threshold above which
mechanical cooling must engage, significantly increasing energy costs.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

# ── Output Configuration ────────────────────────────────────────────────────
FIGURE_DIR = cfg.FIGURE_DIR
os.makedirs(FIGURE_DIR, exist_ok=True)


def load_temperature_data():
    """Load Ashburn, VA temperature data."""
    # Try multiple possible paths
    possible_paths = [
        os.path.join(cfg.PROJECT_DIR, "Data_Sources", "ashburn_va_temperature_2019.csv"),
        os.path.join(cfg.DATA_DIR, "ashburn_va_temperature_2019_cleaned.csv"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"✓ Loaded temperature data from: {os.path.basename(path)}")
            print(f"  Shape: {df.shape}")
            return df
    
    raise FileNotFoundError(f"Temperature data not found in: {possible_paths}")


def generate_actuarial_risk_visualization(df_weather, output_dir=FIGURE_DIR):
    """
    Generate actuarial risk visualizations for datacenter cooling costs.
    
    The "Free Cooling Limit" at 15°C (59°F) represents the temperature
    above which economizer/free-air cooling is no longer effective and
    mechanical chillers must engage, multiplying energy costs.
    """
    print("\n" + "=" * 70)
    print("ACTUARIAL RISK VISUALIZATION")
    print("=" * 70)
    
    # Ensure proper column names and types
    if 'timestamp' not in df_weather.columns:
        # Try to find timestamp column
        ts_cols = [c for c in df_weather.columns if 'time' in c.lower() or 'date' in c.lower()]
        if ts_cols:
            df_weather.rename(columns={ts_cols[0]: 'timestamp'}, inplace=True)
    
    df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'])
    
    # Handle temperature column naming
    if 'temperature_c' not in df_weather.columns:
        if 'temperature_f' in df_weather.columns:
            df_weather['temperature_c'] = (df_weather['temperature_f'] - 32) * 5/9
        else:
            temp_cols = [c for c in df_weather.columns if 'temp' in c.lower()]
            if temp_cols:
                df_weather['temperature_c'] = df_weather[temp_cols[0]]
    
    # Sort by timestamp
    df_weather = df_weather.sort_values('timestamp').reset_index(drop=True)
    
    # =========================================================================
    # FIGURE 1: Combined Timeline and Distribution
    # =========================================================================
    print("\nGenerating Actuarial Risk Visualizations...")
    
    fig = plt.figure(figsize=(14, 10))
    
    # --- Subplot 1: The Timeline (The "Risk Rises" View) ---
    ax1 = plt.subplot(2, 1, 1)
    
    # Plot raw data as a faint background
    ax1.plot(df_weather['timestamp'], df_weather['temperature_c'], 
             color='#1f77b4', linewidth=0.5, alpha=0.4, label='Hourly Temperature')
    
    # Add a 7-day rolling average for a clear trend
    df_weather['temp_rolling'] = df_weather['temperature_c'].rolling(window=24*7, min_periods=1).mean()
    ax1.plot(df_weather['timestamp'], df_weather['temp_rolling'], 
             color='black', linewidth=2, label='7-Day Moving Average')
    
    # The Actuarial Threshold (15°C / 59°F - When Free Cooling Stops)
    FREE_COOLING_LIMIT_C = 15
    ax1.axhline(FREE_COOLING_LIMIT_C, color='red', linestyle='--', linewidth=2, 
                label=f'Free Cooling Limit ({FREE_COOLING_LIMIT_C}°C / {FREE_COOLING_LIMIT_C * 9/5 + 32:.0f}°F)')
    
    # Fill the "Danger Zone" where cooling costs multiply
    ax1.fill_between(df_weather['timestamp'], FREE_COOLING_LIMIT_C, df_weather['temperature_c'],
                     where=(df_weather['temperature_c'] > FREE_COOLING_LIMIT_C), 
                     color='red', alpha=0.3, label='Cooling Penalty Zone')
    
    ax1.set_title('Ashburn, VA Hourly Temperatures (2019) - The Thermodynamic Stress Timeline', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Ambient Temperature (°C)', fontsize=12)
    ax1.set_xlabel('')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add secondary y-axis for Fahrenheit
    ax1_f = ax1.twinx()
    ax1_f.set_ylim(ax1.get_ylim()[0] * 9/5 + 32, ax1.get_ylim()[1] * 9/5 + 32)
    ax1_f.set_ylabel('Temperature (°F)', fontsize=10)
    
    # --- Subplot 2: The Distribution (The "Probability of Loss" View) ---
    ax2 = plt.subplot(2, 1, 2)
    
    # Create histogram
    sns.histplot(df_weather['temperature_c'].dropna(), bins=50, color='skyblue', 
                 edgecolor='black', ax=ax2)
    
    # The Actuarial Threshold (15°C)
    ax2.axvline(FREE_COOLING_LIMIT_C, color='red', linestyle='--', linewidth=2, 
                label=f'Free Cooling Limit ({FREE_COOLING_LIMIT_C}°C)')
    
    # Calculate exact probabilities (The Actuarial Math)
    penalty_hours = len(df_weather[df_weather['temperature_c'] > FREE_COOLING_LIMIT_C])
    total_hours = len(df_weather)
    percent_penalty = penalty_hours / total_hours if total_hours > 0 else 0
    
    # Highlight the Right Tail (The Expensive Hours)
    counts, bin_edges = np.histogram(df_weather['temperature_c'].dropna(), bins=50)
    max_count = counts.max()
    
    # Add text box with actuarial statistics
    ax2.text(20, max_count * 0.6, 
             f'ACTUARIAL LOSS ZONE\n{penalty_hours:,} hours\n({percent_penalty:.1%} of the year)',
             color='red', fontweight='bold', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', pad=10))
    
    ax2.set_title('Temperature Probability Distribution (Hours per Year)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Ambient Temperature (°C)', fontsize=12)
    ax2.set_ylabel('Frequency (Number of Hours)', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'actuarial_risk_temperature.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: actuarial_risk_temperature.png")
    
    # =========================================================================
    # FIGURE 2: Monthly Risk Breakdown
    # =========================================================================
    df_weather['month'] = df_weather['timestamp'].dt.month
    df_weather['month_name'] = df_weather['timestamp'].dt.strftime('%b')
    
    # Calculate penalty hours by month
    monthly_stats = df_weather.groupby('month').agg({
        'temperature_c': ['mean', 'max', 'min', 'count'],
    }).round(2)
    monthly_stats.columns = ['avg_temp', 'max_temp', 'min_temp', 'total_hours']
    
    penalty_by_month = df_weather[df_weather['temperature_c'] > FREE_COOLING_LIMIT_C].groupby('month').size()
    monthly_stats['penalty_hours'] = penalty_by_month.reindex(monthly_stats.index, fill_value=0)
    monthly_stats['penalty_percent'] = (monthly_stats['penalty_hours'] / monthly_stats['total_hours'] * 100).round(1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Stacked bar of normal vs penalty hours
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    normal_hours = monthly_stats['total_hours'] - monthly_stats['penalty_hours']
    
    axes[0].bar(months, normal_hours, color='#2ecc71', label='Free Cooling Hours', edgecolor='black')
    axes[0].bar(months, monthly_stats['penalty_hours'], bottom=normal_hours, 
                color='#e74c3c', label='Mechanical Cooling Required', edgecolor='black')
    axes[0].set_ylabel('Hours per Month', fontsize=12)
    axes[0].set_xlabel('Month', fontsize=12)
    axes[0].set_title('Monthly Breakdown: Free Cooling vs Mechanical Cooling', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Right: Temperature range by month
    monthly_temps = df_weather.groupby(['month', 'month_name'])['temperature_c'].agg(['mean', 'min', 'max']).reset_index()
    monthly_temps = monthly_temps.sort_values('month')
    
    axes[1].fill_between(months, monthly_stats['min_temp'], monthly_stats['max_temp'], 
                         alpha=0.3, color='blue', label='Temperature Range')
    axes[1].plot(months, monthly_stats['avg_temp'], 'o-', color='blue', linewidth=2, 
                 markersize=8, label='Monthly Average')
    axes[1].axhline(FREE_COOLING_LIMIT_C, color='red', linestyle='--', linewidth=2, 
                    label=f'Free Cooling Limit ({FREE_COOLING_LIMIT_C}°C)')
    axes[1].set_ylabel('Temperature (°C)', fontsize=12)
    axes[1].set_xlabel('Month', fontsize=12)
    axes[1].set_title('Monthly Temperature Profile with Risk Threshold', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'actuarial_risk_monthly_breakdown.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: actuarial_risk_monthly_breakdown.png")
    
    # =========================================================================
    # FIGURE 3: Hourly Heatmap (Risk by Hour and Month)
    # =========================================================================
    df_weather['hour'] = df_weather['timestamp'].dt.hour
    
    # Create pivot table of penalty probability
    df_weather['is_penalty'] = (df_weather['temperature_c'] > FREE_COOLING_LIMIT_C).astype(int)
    heatmap_data = df_weather.pivot_table(
        values='is_penalty', 
        index='hour', 
        columns='month', 
        aggfunc='mean'
    ) * 100  # Convert to percentage
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(heatmap_data, cmap='RdYlGn_r', annot=True, fmt='.0f', 
                cbar_kws={'label': 'Probability of Mechanical Cooling (%)'}, ax=ax)
    
    ax.set_xticklabels(months)
    ax.set_ylabel('Hour of Day', fontsize=12)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_title('Cooling Risk Heatmap: Probability of Exceeding Free Cooling Limit\n(% of hours requiring mechanical cooling)', 
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'actuarial_risk_heatmap.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: actuarial_risk_heatmap.png")
    
    # =========================================================================
    # Print Summary Statistics
    # =========================================================================
    print("\n" + "-" * 50)
    print("ACTUARIAL RISK SUMMARY")
    print("-" * 50)
    print(f"Total hours analyzed:        {total_hours:,}")
    print(f"Free cooling hours:          {total_hours - penalty_hours:,} ({(1-percent_penalty):.1%})")
    print(f"Mechanical cooling hours:    {penalty_hours:,} ({percent_penalty:.1%})")
    print(f"Free cooling limit:          {FREE_COOLING_LIMIT_C}°C / {FREE_COOLING_LIMIT_C * 9/5 + 32:.0f}°F")
    print(f"\nPeak risk months: June, July, August")
    print(f"  - June penalty hours:   {monthly_stats.loc[6, 'penalty_hours']:.0f} ({monthly_stats.loc[6, 'penalty_percent']:.1f}%)")
    print(f"  - July penalty hours:   {monthly_stats.loc[7, 'penalty_hours']:.0f} ({monthly_stats.loc[7, 'penalty_percent']:.1f}%)")
    print(f"  - August penalty hours: {monthly_stats.loc[8, 'penalty_hours']:.0f} ({monthly_stats.loc[8, 'penalty_percent']:.1f}%)")
    
    # Estimate cost impact
    print(f"\n--- ESTIMATED COST IMPACT ---")
    avg_load_mw = 80  # Assume 80% of 100MW capacity
    pue_free_cooling = 1.15
    pue_mechanical = 1.45
    electricity_rate = 0.08  # $/kWh
    
    # Cost with free cooling
    free_cooling_cost = (total_hours - penalty_hours) * avg_load_mw * pue_free_cooling * 1000 * electricity_rate
    # Cost with mechanical cooling
    mechanical_cost = penalty_hours * avg_load_mw * pue_mechanical * 1000 * electricity_rate
    # If all hours were free cooling
    baseline_cost = total_hours * avg_load_mw * pue_free_cooling * 1000 * electricity_rate
    
    actual_cost = free_cooling_cost + mechanical_cost
    penalty_cost = actual_cost - baseline_cost
    
    print(f"Assumed IT load:             {avg_load_mw} MW")
    print(f"PUE (free cooling):          {pue_free_cooling}")
    print(f"PUE (mechanical cooling):    {pue_mechanical}")
    print(f"Electricity rate:            ${electricity_rate}/kWh")
    print(f"\nAnnual cooling penalty cost: ${penalty_cost:,.0f}")
    print(f"  (Additional cost vs. year-round free cooling)")
    
    return monthly_stats


def main():
    """Run the actuarial risk analysis."""
    print("\n" + "=" * 70)
    print("MTFC ACTUARIAL RISK ANALYSIS")
    print("=" * 70)
    print(f"Output directory: {FIGURE_DIR}")
    
    # Load temperature data
    df_weather = load_temperature_data()
    
    # Generate visualizations
    monthly_stats = generate_actuarial_risk_visualization(df_weather)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nFigures saved to: {FIGURE_DIR}")
    print("  - actuarial_risk_temperature.png")
    print("  - actuarial_risk_monthly_breakdown.png")
    print("  - actuarial_risk_heatmap.png")


if __name__ == "__main__":
    main()
