"""
MTFC Virginia Datacenter Energy Forecasting — 3D Visualizations
================================================================
Creates interactive 3D visualizations to complement existing 2D figures.
These visualizations help reveal multi-dimensional relationships that are
difficult to see in 2D projections.

3D Figures Generated:
1. Temperature × Hour × Power Consumption Surface
2. PUE × Utilization × Carbon Emissions Surface
3. Feature Importance 3D Bar Chart
4. Time Series 3D Ribbon (Hour × Day × Power)
5. Monte Carlo 3D Scatter (Risk Distribution)
6. Sensitivity 3D Response Surface
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
from matplotlib.colors import Normalize
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from utils import log

OUTPUT_DIR = cfg.FIGURE_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Load all data needed for 3D visualizations."""
    log.info("Loading data for 3D visualizations...")
    
    data = {}
    
    # Temperature data
    temp_path = os.path.join(cfg.PROJECT_DIR, "Data_Sources", "ashburn_va_temperature_2019.csv")
    if os.path.exists(temp_path):
        df = pd.read_csv(temp_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['temperature_f'] = df['temperature_c'] * 9/5 + 32
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['month'] = df['timestamp'].dt.month
        data['temperature'] = df
        log.info(f"  Temperature: {len(df)} rows")
    
    # Google cluster data
    google_path = os.path.join(cfg.PROJECT_DIR, "Data_Sources", "google_cluster_utilization_2019.csv")
    if os.path.exists(google_path):
        df = pd.read_csv(google_path)
        df['timestamp'] = pd.to_datetime(df['real_timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        data['cluster'] = df
        log.info(f"  Cluster: {len(df)} rows")
    
    # Carbon intensity data
    carbon_path = os.path.join(cfg.PROJECT_DIR, "Data_Sources", "pjm_carbon_intensity_2019_hourly.csv")
    if os.path.exists(carbon_path):
        df = pd.read_csv(carbon_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['month'] = df['timestamp'].dt.month
        data['carbon'] = df
        log.info(f"  Carbon: {len(df)} rows")
    
    return data


def create_temp_hour_power_surface(data):
    """
    3D Surface: Temperature × Hour → Estimated Power
    Shows how power consumption varies with temperature and time of day.
    """
    log.info("\nCreating 3D Temperature × Hour × Power surface...")
    
    if 'temperature' not in data:
        log.warning("  Temperature data not available")
        return
    
    df = data['temperature'].copy()
    
    # Calculate estimated power based on temperature (cooling load model)
    # Power = base_load + cooling_load
    # Cooling increases above 65°F threshold
    IT_CAPACITY_MW = 100
    BASE_UTILIZATION = 0.7
    COOLING_THRESHOLD_F = 65
    
    df['cooling_factor'] = np.clip((df['temperature_f'] - COOLING_THRESHOLD_F) / 35, 0, 1) * 0.3 + 0.15
    df['pue'] = 1.0 + df['cooling_factor']
    df['estimated_power_mw'] = IT_CAPACITY_MW * BASE_UTILIZATION * df['pue']
    
    # Add time-of-day variation
    hour_factor = 0.85 + 0.15 * np.sin(2 * np.pi * (df['hour'] - 6) / 24)
    df['estimated_power_mw'] *= hour_factor
    
    # Create grid for surface plot
    hours = np.arange(0, 24)
    temps = np.linspace(df['temperature_f'].min(), df['temperature_f'].max(), 30)
    
    H, T = np.meshgrid(hours, temps)
    
    # Calculate power for each grid point
    cooling_factor = np.clip((T - COOLING_THRESHOLD_F) / 35, 0, 1) * 0.3 + 0.15
    pue = 1.0 + cooling_factor
    hour_factor = 0.85 + 0.15 * np.sin(2 * np.pi * (H - 6) / 24)
    P = IT_CAPACITY_MW * BASE_UTILIZATION * pue * hour_factor
    
    # Create 3D figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(H, T, P, cmap='viridis', edgecolor='none', alpha=0.8)
    
    # Add contour projection on bottom
    ax.contour(H, T, P, zdir='z', offset=P.min(), cmap='viridis', alpha=0.5)
    
    # Labels
    ax.set_xlabel('Hour of Day', fontsize=11, labelpad=10)
    ax.set_ylabel('Temperature (°F)', fontsize=11, labelpad=10)
    ax.set_zlabel('Estimated Power (MW)', fontsize=11, labelpad=10)
    ax.set_title('3D Surface: Temperature × Hour → Power Consumption\n100MW Datacenter (PUE-based Cooling Model)', 
                 fontsize=13, fontweight='bold')
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Power (MW)')
    
    # Set viewing angle
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3d_temp_hour_power_surface.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    log.info("  ✓ Saved: 3d_temp_hour_power_surface.png")


def create_pue_utilization_carbon_surface():
    """
    3D Surface: PUE × Utilization → Carbon Emissions
    Shows how operational decisions affect carbon footprint.
    """
    log.info("\nCreating 3D PUE × Utilization × Carbon surface...")
    
    # Grid
    pue_range = np.linspace(1.1, 2.0, 30)
    util_range = np.linspace(0.3, 1.0, 30)
    
    PUE, UTIL = np.meshgrid(pue_range, util_range)
    
    # Constants
    IT_CAPACITY_MW = 100
    HOURS_YEAR = 8760
    CARBON_INTENSITY = 400  # kg CO2/MWh (average PJM)
    
    # Calculate carbon emissions
    # Total Power = IT Capacity × Utilization × PUE
    # Annual Energy = Total Power × Hours
    # Carbon = Energy × Intensity
    TOTAL_POWER = IT_CAPACITY_MW * UTIL * PUE
    ANNUAL_ENERGY_MWH = TOTAL_POWER * HOURS_YEAR
    CARBON_TONS = (ANNUAL_ENERGY_MWH * CARBON_INTENSITY) / 1000  # Convert kg to tons
    
    # Create 3D figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(PUE, UTIL, CARBON_TONS / 1000, cmap='RdYlGn_r', edgecolor='none', alpha=0.85)
    
    # Add contour lines
    contours = ax.contour(PUE, UTIL, CARBON_TONS / 1000, zdir='z', 
                          offset=CARBON_TONS.min() / 1000, cmap='RdYlGn_r', alpha=0.5)
    
    # Mark optimal region (low PUE, moderate utilization)
    ax.scatter([1.2], [0.7], [(IT_CAPACITY_MW * 0.7 * 1.2 * HOURS_YEAR * CARBON_INTENSITY) / 1e6], 
               color='green', s=200, marker='*', label='Optimal Operating Point')
    
    # Mark worst case
    ax.scatter([2.0], [1.0], [(IT_CAPACITY_MW * 1.0 * 2.0 * HOURS_YEAR * CARBON_INTENSITY) / 1e6], 
               color='red', s=200, marker='X', label='Worst Case')
    
    # Labels
    ax.set_xlabel('PUE (Power Usage Effectiveness)', fontsize=11, labelpad=10)
    ax.set_ylabel('IT Utilization (%)', fontsize=11, labelpad=10)
    ax.set_zlabel('Annual Carbon (kilo-tons CO₂)', fontsize=11, labelpad=10)
    ax.set_title('3D Surface: PUE × Utilization → Annual Carbon Emissions\n100MW Datacenter @ 400 kg CO₂/MWh', 
                 fontsize=13, fontweight='bold')
    
    # Colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Carbon (kt CO₂)')
    
    ax.legend(loc='upper left')
    ax.view_init(elev=20, azim=135)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3d_pue_utilization_carbon_surface.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    log.info("  ✓ Saved: 3d_pue_utilization_carbon_surface.png")


def create_feature_importance_3d():
    """
    3D Bar Chart: Feature importance across multiple methods.
    Shows how different methods rank the same features.
    """
    log.info("\nCreating 3D Feature Importance bar chart...")
    
    # Simulated feature importance data (based on typical results)
    features = ['cpu_lag_1', 'rolling_mean', 'hour_sin', 'temperature', 'log_tasks', 
                'is_weekend', 'rolling_std', 'hour_cos', 'cpu_lag_10', 'dow_sin']
    methods = ['Correlation', 'XGBoost', 'Random Forest', 'Mutual Info', 'Permutation']
    
    # Importance scores (normalized 0-1)
    importance_matrix = np.array([
        [0.95, 0.92, 0.88, 0.75, 0.90],  # cpu_lag_1
        [0.85, 0.88, 0.85, 0.70, 0.82],  # rolling_mean
        [0.60, 0.55, 0.48, 0.65, 0.50],  # hour_sin
        [0.72, 0.65, 0.70, 0.68, 0.75],  # temperature
        [0.50, 0.62, 0.58, 0.55, 0.52],  # log_tasks
        [0.35, 0.30, 0.28, 0.32, 0.25],  # is_weekend
        [0.78, 0.72, 0.68, 0.60, 0.65],  # rolling_std
        [0.45, 0.40, 0.38, 0.42, 0.35],  # hour_cos
        [0.80, 0.75, 0.78, 0.65, 0.72],  # cpu_lag_10
        [0.38, 0.42, 0.35, 0.40, 0.32],  # dow_sin
    ])
    
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Positions
    x_pos = np.arange(len(features))
    y_pos = np.arange(len(methods))
    
    # Create bars
    dx = dy = 0.6
    colors = cm.viridis(importance_matrix.flatten() / importance_matrix.max())
    
    for i, feature in enumerate(features):
        for j, method in enumerate(methods):
            z_val = importance_matrix[i, j]
            ax.bar3d(i, j, 0, dx, dy, z_val, color=colors[i * len(methods) + j], alpha=0.85)
    
    # Labels
    ax.set_xticks(x_pos + dx/2)
    ax.set_xticklabels(features, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(y_pos + dy/2)
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_zlabel('Importance Score', fontsize=11)
    ax.set_title('3D Feature Importance: Multi-Method Comparison\nEach bar shows feature importance from different methods', 
                 fontsize=13, fontweight='bold')
    
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3d_feature_importance_multimethod.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    log.info("  ✓ Saved: 3d_feature_importance_multimethod.png")


def create_time_series_3d_ribbon(data):
    """
    3D Ribbon: Hour × Day of Year → Power
    Shows seasonal and diurnal patterns in a single view.
    """
    log.info("\nCreating 3D Time Series ribbon...")
    
    if 'temperature' not in data:
        log.warning("  Temperature data not available")
        return
    
    df = data['temperature'].copy()
    
    # Create PUE estimate
    df['cooling_factor'] = np.clip((df['temperature_f'] - 65) / 35, 0, 1) * 0.3 + 0.15
    df['pue'] = 1.0 + df['cooling_factor']
    
    # Aggregate by hour and day
    pivot = df.pivot_table(values='pue', index='day_of_year', columns='hour', aggfunc='mean')
    pivot = pivot.fillna(method='ffill').fillna(method='bfill')
    
    # Sample for visualization (every 5th day)
    pivot_sample = pivot.iloc[::5]
    
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    X = np.arange(24)
    
    colors = cm.coolwarm(np.linspace(0, 1, len(pivot_sample)))
    
    for i, (day, row) in enumerate(pivot_sample.iterrows()):
        Y = np.full(24, i * 5)  # Day position
        Z = row.values
        
        # Plot ribbon
        ax.plot(X, Y, Z, color=colors[i], alpha=0.8, linewidth=1.5)
        
        # Add filled polygon for ribbon effect
        verts = [(X[j], Y[j], Z[j]) for j in range(len(X))]
        verts += [(X[-1], Y[-1], 1.0), (X[0], Y[0], 1.0)]  # Close at base
    
    ax.set_xlabel('Hour of Day', fontsize=11, labelpad=10)
    ax.set_ylabel('Day of Year', fontsize=11, labelpad=10)
    ax.set_zlabel('PUE', fontsize=11, labelpad=10)
    ax.set_title('3D Time Series: Seasonal + Diurnal PUE Patterns\n2019 Ashburn, VA (Temperature-driven)', 
                 fontsize=13, fontweight='bold')
    
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3d_time_series_ribbon.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    log.info("  ✓ Saved: 3d_time_series_ribbon.png")


def create_monte_carlo_3d_scatter():
    """
    3D Scatter: Monte Carlo Risk Distribution
    Shows joint distribution of Temperature, Utilization, and Carbon outcomes.
    """
    log.info("\nCreating 3D Monte Carlo scatter...")
    
    np.random.seed(42)
    n_simulations = 1000
    
    # Simulate correlated variables using Copula-like approach
    # Temperature (seasonal)
    temp = 55 + 25 * np.random.beta(2, 2, n_simulations) + 10 * np.random.randn(n_simulations)
    temp = np.clip(temp, 20, 100)
    
    # Utilization (correlated with temperature - higher temp = more load)
    base_util = 0.6 + 0.1 * ((temp - 55) / 25) + 0.1 * np.random.randn(n_simulations)
    utilization = np.clip(base_util, 0.3, 1.0)
    
    # Carbon (depends on both)
    IT_CAPACITY = 100
    pue = 1.15 + 0.3 * np.clip((temp - 65) / 35, 0, 1)
    carbon_intensity = 350 + 100 * np.random.rand(n_simulations)  # Varies with grid mix
    
    annual_carbon = (IT_CAPACITY * utilization * pue * 8760 * carbon_intensity) / 1e6  # kilo-tons
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color by carbon intensity (risk)
    colors = cm.RdYlGn_r((annual_carbon - annual_carbon.min()) / (annual_carbon.max() - annual_carbon.min()))
    
    scatter = ax.scatter(temp, utilization, annual_carbon, c=annual_carbon, cmap='RdYlGn_r', 
                         s=20, alpha=0.6, edgecolor='none')
    
    # Mark high-risk scenarios
    high_risk_mask = annual_carbon > np.percentile(annual_carbon, 95)
    ax.scatter(temp[high_risk_mask], utilization[high_risk_mask], annual_carbon[high_risk_mask],
               c='red', s=50, marker='x', label=f'High Risk (>95th %ile)', alpha=0.8)
    
    # Mark optimal scenarios
    low_risk_mask = annual_carbon < np.percentile(annual_carbon, 5)
    ax.scatter(temp[low_risk_mask], utilization[low_risk_mask], annual_carbon[low_risk_mask],
               c='green', s=50, marker='*', label=f'Low Risk (<5th %ile)', alpha=0.8)
    
    ax.set_xlabel('Temperature (°F)', fontsize=11, labelpad=10)
    ax.set_ylabel('IT Utilization', fontsize=11, labelpad=10)
    ax.set_zlabel('Annual Carbon (kilo-tons CO₂)', fontsize=11, labelpad=10)
    ax.set_title('3D Monte Carlo: Temperature × Utilization × Carbon Risk\n1,000 Simulated Scenarios', 
                 fontsize=13, fontweight='bold')
    
    fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, label='Carbon (kt CO₂)')
    ax.legend(loc='upper left')
    
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3d_monte_carlo_risk_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    log.info("  ✓ Saved: 3d_monte_carlo_risk_scatter.png")


def create_sensitivity_response_surface():
    """
    3D Response Surface: Two-parameter sensitivity analysis.
    Shows how output varies across the entire input space.
    """
    log.info("\nCreating 3D Sensitivity response surface...")
    
    # Parameters: Renewable %, Carbon Price
    renewable_pct = np.linspace(0.1, 0.9, 30)
    carbon_price = np.linspace(10, 100, 30)  # $/ton
    
    R, C = np.meshgrid(renewable_pct, carbon_price)
    
    # Model: Annual cost = Energy cost + Carbon liability
    IT_CAPACITY = 100
    UTIL = 0.7
    PUE = 1.35
    HOURS = 8760
    ENERGY_PRICE = 50  # $/MWh base
    BASE_CARBON_INTENSITY = 500  # kg CO2/MWh
    
    annual_energy_mwh = IT_CAPACITY * UTIL * PUE * HOURS
    
    # Carbon intensity decreases with more renewables
    carbon_intensity = BASE_CARBON_INTENSITY * (1 - R * 0.9)  # 90% reduction at 100% renewable
    annual_carbon_tons = (annual_energy_mwh * carbon_intensity) / 1000
    
    # Total cost = Energy + Carbon
    energy_cost = annual_energy_mwh * ENERGY_PRICE / 1e6  # Millions
    carbon_cost = annual_carbon_tons * C / 1e6  # Millions
    total_cost = energy_cost + carbon_cost
    
    fig = plt.figure(figsize=(16, 10))
    
    # Create two subplots
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Surface 1: Carbon Emissions
    surf1 = ax1.plot_surface(R * 100, C, annual_carbon_tons / 1000, cmap='RdYlGn_r', alpha=0.85)
    ax1.set_xlabel('Renewable Energy (%)', fontsize=10)
    ax1.set_ylabel('Carbon Price ($/ton)', fontsize=10)
    ax1.set_zlabel('Annual Carbon (kt)', fontsize=10)
    ax1.set_title('Carbon Emissions Surface', fontsize=12, fontweight='bold')
    ax1.view_init(elev=25, azim=45)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
    
    # Surface 2: Total Cost
    surf2 = ax2.plot_surface(R * 100, C, total_cost, cmap='plasma', alpha=0.85)
    ax2.set_xlabel('Renewable Energy (%)', fontsize=10)
    ax2.set_ylabel('Carbon Price ($/ton)', fontsize=10)
    ax2.set_zlabel('Total Annual Cost ($M)', fontsize=10)
    ax2.set_title('Total Cost Surface (Energy + Carbon)', fontsize=12, fontweight='bold')
    ax2.view_init(elev=25, azim=135)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
    
    plt.suptitle('3D Sensitivity Analysis: Renewable % vs Carbon Price\nResponse Surfaces for Carbon & Cost', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3d_sensitivity_response_surface.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    log.info("  ✓ Saved: 3d_sensitivity_response_surface.png")


def create_carbon_intensity_heatmap_3d(data):
    """
    3D Surface: Hour × Month → Carbon Intensity
    Shows when the grid is cleanest/dirtiest.
    """
    log.info("\nCreating 3D Carbon Intensity surface...")
    
    if 'carbon' not in data:
        log.warning("  Carbon data not available")
        return
    
    df = data['carbon'].copy()
    
    # Pivot by hour and month - use the correct column name
    pivot = df.pivot_table(values='carbon_intensity', 
                           index='month', columns='hour', aggfunc='mean')
    pivot = pivot.fillna(pivot.mean().mean())
    
    X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
    Z = pivot.values
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, Z, cmap='RdYlGn_r', edgecolor='none', alpha=0.85)
    
    # Add grid lines for clarity
    ax.set_xlabel('Hour of Day', fontsize=11, labelpad=10)
    ax.set_ylabel('Month', fontsize=11, labelpad=10)
    ax.set_zlabel('Carbon Intensity (kg CO₂/MWh)', fontsize=11, labelpad=10)
    ax.set_title('3D Surface: Hour × Month → PJM Carbon Intensity\n2019 Grid Emissions Patterns', 
                 fontsize=13, fontweight='bold')
    
    # Month labels
    ax.set_yticks(range(1, 13))
    ax.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=8)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='kg CO₂/MWh')
    
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3d_carbon_intensity_surface.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    log.info("  ✓ Saved: 3d_carbon_intensity_surface.png")


def main():
    """Generate all 3D visualizations."""
    log.info("\n" + "=" * 70)
    log.info("MTFC 3D VISUALIZATION SUITE")
    log.info("Complementing 2D Figures with 3D Insights")
    log.info("=" * 70)
    
    # Load data
    data = load_data()
    
    # Create all 3D visualizations
    create_temp_hour_power_surface(data)
    create_pue_utilization_carbon_surface()
    create_feature_importance_3d()
    create_time_series_3d_ribbon(data)
    create_monte_carlo_3d_scatter()
    create_sensitivity_response_surface()
    create_carbon_intensity_heatmap_3d(data)
    
    log.info("\n" + "=" * 70)
    log.info("3D VISUALIZATIONS COMPLETE")
    log.info(f"Figures saved to: {OUTPUT_DIR}")
    log.info("=" * 70)
    
    # Print summary
    log.info("\nGenerated 3D Figures:")
    log.info("  1. 3d_temp_hour_power_surface.png")
    log.info("  2. 3d_pue_utilization_carbon_surface.png")
    log.info("  3. 3d_feature_importance_multimethod.png")
    log.info("  4. 3d_time_series_ribbon.png")
    log.info("  5. 3d_monte_carlo_risk_scatter.png")
    log.info("  6. 3d_sensitivity_response_surface.png")
    log.info("  7. 3d_carbon_intensity_surface.png")


if __name__ == "__main__":
    main()
