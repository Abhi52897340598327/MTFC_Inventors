"""
MTFC Virginia Datacenter — Advanced Sensitivity Analysis
=========================================================
Purpose: Comprehensive sensitivity analysis with advanced visualizations.

Analysis Types:
1. Tornado Diagram - One-at-a-time sensitivity ranking
2. Spider Plot - Multi-parameter interaction
3. Scenario Matrix Heatmap - 2D parameter sweeps
4. Morris Screening - Global sensitivity for many parameters
5. Sobol Indices - Variance-based sensitivity decomposition
6. Feature Contribution Waterfall - Decompose predictions

Output: Publication-quality figures for each analysis type
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from itertools import product

# Add project path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from utils import log, save_fig, set_plot_style

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

# Baseline datacenter parameters
BASELINE_PARAMS = {
    "it_capacity_mw": 100,          # Base IT load capacity
    "pue": 1.35,                    # Power Usage Effectiveness
    "utilization": 0.70,            # Average utilization
    "temperature_f": 65,            # Average ambient temperature
    "cooling_threshold_f": 65,      # Cooling threshold
    "grid_carbon_intensity": 387,   # g CO2/kWh (from PJM analysis)
    "annual_growth_rate": 0.15,     # 15% annual growth
    "renewable_pct": 0.20,          # 20% renewable energy
    "hours_per_year": 8760,         # Hours in a year
}

# Parameter ranges for sensitivity analysis (% change from baseline)
SENSITIVITY_RANGES = {
    "it_capacity_mw": [50, 75, 100, 150, 200, 300],        # MW
    "pue": [1.1, 1.2, 1.35, 1.5, 1.6, 1.8],                # PUE levels
    "utilization": [0.4, 0.55, 0.70, 0.85, 0.95],          # Utilization %
    "temperature_f": [55, 60, 65, 75, 85, 95],             # °F
    "grid_carbon_intensity": [200, 300, 387, 500, 700],    # g CO2/kWh
    "annual_growth_rate": [0.05, 0.10, 0.15, 0.25, 0.35],  # Annual growth
    "renewable_pct": [0.0, 0.20, 0.40, 0.60, 0.80, 1.0],   # Renewable %
}

# Parameter display names
PARAM_LABELS = {
    "it_capacity_mw": "IT Capacity (MW)",
    "pue": "PUE",
    "utilization": "Utilization",
    "temperature_f": "Ambient Temp (°F)",
    "grid_carbon_intensity": "Grid Carbon (g/kWh)",
    "annual_growth_rate": "Annual Growth Rate",
    "renewable_pct": "Renewable %",
}


# ════════════════════════════════════════════════════════════════════════════
# PHYSICS MODEL (for sensitivity calculations)
# ════════════════════════════════════════════════════════════════════════════

def calculate_annual_metrics(params):
    """
    Calculate annual power consumption and carbon emissions based on parameters.
    
    Returns dict with:
    - total_power_mwh: Annual energy consumption
    - peak_power_mw: Peak power demand
    - carbon_tons: Annual CO2 emissions
    - cooling_load_mwh: Energy for cooling
    """
    # Extract parameters
    it_cap = params.get("it_capacity_mw", BASELINE_PARAMS["it_capacity_mw"])
    pue = params.get("pue", BASELINE_PARAMS["pue"])
    util = params.get("utilization", BASELINE_PARAMS["utilization"])
    temp = params.get("temperature_f", BASELINE_PARAMS["temperature_f"])
    carbon_int = params.get("grid_carbon_intensity", BASELINE_PARAMS["grid_carbon_intensity"])
    renewable = params.get("renewable_pct", BASELINE_PARAMS["renewable_pct"])
    hours = params.get("hours_per_year", BASELINE_PARAMS["hours_per_year"])
    
    # IT power (capacity × utilization)
    it_power_mw = it_cap * util
    
    # Temperature-dependent PUE adjustment
    # PUE increases above cooling threshold
    cooling_threshold = params.get("cooling_threshold_f", BASELINE_PARAMS["cooling_threshold_f"])
    cooling_degree = max(0, temp - cooling_threshold)
    pue_adjusted = pue + (cooling_degree * 0.01)  # +0.01 PUE per degree above threshold
    pue_adjusted = min(pue_adjusted, 2.0)  # Cap at 2.0
    
    # Total power with PUE
    total_power_mw = it_power_mw * pue_adjusted
    
    # Annual energy
    total_power_mwh = total_power_mw * hours
    
    # Peak power (assume 1.3x average for peak)
    peak_power_mw = total_power_mw * 1.3
    
    # Cooling load (non-IT portion)
    cooling_load_mw = total_power_mw - it_power_mw
    cooling_load_mwh = cooling_load_mw * hours
    
    # Carbon emissions (adjusted for renewables)
    effective_carbon_int = carbon_int * (1 - renewable)  # Renewables offset
    carbon_tons = (total_power_mwh * effective_carbon_int / 1000) / 1000  # kg to tons
    
    return {
        "total_power_mw": total_power_mw,
        "total_power_mwh": total_power_mwh,
        "peak_power_mw": peak_power_mw,
        "cooling_load_mwh": cooling_load_mwh,
        "carbon_tons": carbon_tons,
        "pue_effective": pue_adjusted,
        "it_power_mw": it_power_mw,
    }


def calculate_2030_metrics(params):
    """Project metrics to 2030 with growth rate applied."""
    growth_rate = params.get("annual_growth_rate", BASELINE_PARAMS["annual_growth_rate"])
    years = 2030 - 2024
    growth_factor = (1 + growth_rate) ** years
    
    # Scale IT capacity by growth
    scaled_params = params.copy()
    scaled_params["it_capacity_mw"] = params.get("it_capacity_mw", BASELINE_PARAMS["it_capacity_mw"]) * growth_factor
    
    return calculate_annual_metrics(scaled_params)


# ════════════════════════════════════════════════════════════════════════════
# 1. TORNADO DIAGRAM
# ════════════════════════════════════════════════════════════════════════════

def create_tornado_diagram(output_metric="carbon_tons"):
    """
    Create tornado diagram showing sensitivity of output to each parameter.
    
    Parameters are varied ±20% from baseline and sorted by impact.
    """
    set_plot_style()
    
    log.info(f"Creating tornado diagram for {output_metric}...")
    
    # Calculate baseline
    baseline_result = calculate_2030_metrics(BASELINE_PARAMS)
    baseline_value = baseline_result[output_metric]
    
    # Calculate sensitivity for each parameter
    sensitivities = []
    
    for param, values in SENSITIVITY_RANGES.items():
        baseline_param = BASELINE_PARAMS[param]
        
        # Find min and max in range
        min_val = min(values)
        max_val = max(values)
        
        # Calculate at extremes
        low_params = BASELINE_PARAMS.copy()
        low_params[param] = min_val
        low_result = calculate_2030_metrics(low_params)[output_metric]
        
        high_params = BASELINE_PARAMS.copy()
        high_params[param] = max_val
        high_result = calculate_2030_metrics(high_params)[output_metric]
        
        # Calculate % change from baseline
        low_pct = ((low_result - baseline_value) / baseline_value) * 100
        high_pct = ((high_result - baseline_value) / baseline_value) * 100
        
        sensitivities.append({
            "parameter": PARAM_LABELS.get(param, param),
            "low_pct": low_pct,
            "high_pct": high_pct,
            "range": abs(high_pct - low_pct),
            "low_label": f"{min_val}",
            "high_label": f"{max_val}",
        })
    
    # Sort by impact (range)
    sensitivities = sorted(sensitivities, key=lambda x: x["range"], reverse=True)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(sensitivities))
    
    for i, s in enumerate(sensitivities):
        # Determine colors (negative = green/good, positive = red/bad for carbon)
        left = min(s["low_pct"], s["high_pct"])
        width = abs(s["high_pct"] - s["low_pct"])
        
        # Color based on whether increase is good or bad
        if s["high_pct"] > s["low_pct"]:
            color_low = "#2ecc71"   # Green (good - lower value)
            color_high = "#e74c3c"  # Red (bad - higher value)
        else:
            color_low = "#e74c3c"
            color_high = "#2ecc71"
        
        # Draw bar from low to 0 and 0 to high
        if s["low_pct"] < 0:
            ax.barh(i, s["low_pct"], color=color_low, edgecolor="white", height=0.7)
        if s["high_pct"] > 0:
            ax.barh(i, s["high_pct"], left=0, color=color_high, edgecolor="white", height=0.7)
        if s["low_pct"] >= 0:
            ax.barh(i, s["low_pct"], color=color_low, edgecolor="white", height=0.7)
        if s["high_pct"] <= 0:
            ax.barh(i, s["high_pct"], left=0, color=color_high, edgecolor="white", height=0.7)
        
        # Full bar for range visualization
        ax.barh(i, s["high_pct"] - s["low_pct"], left=s["low_pct"], 
                color="none", edgecolor="black", linewidth=1.5, height=0.7)
        
        # Add value labels
        ax.text(s["low_pct"] - 2, i, s["low_label"], va='center', ha='right', fontsize=9)
        ax.text(s["high_pct"] + 2, i, s["high_label"], va='center', ha='left', fontsize=9)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([s["parameter"] for s in sensitivities])
    ax.axvline(0, color="black", linewidth=2)
    ax.set_xlabel("% Change from Baseline", fontsize=12)
    ax.set_title(f"Tornado Diagram: Sensitivity of 2030 {output_metric.replace('_', ' ').title()}\n"
                 f"(Baseline: {baseline_value:,.0f})", fontsize=14)
    ax.grid(True, axis='x', alpha=0.3)
    
    # Legend
    green_patch = mpatches.Patch(color='#2ecc71', label='Lower Impact')
    red_patch = mpatches.Patch(color='#e74c3c', label='Higher Impact')
    ax.legend(handles=[green_patch, red_patch], loc='lower right')
    
    plt.tight_layout()
    save_fig(fig, f"tornado_diagram_{output_metric}")
    
    return fig, sensitivities


# ════════════════════════════════════════════════════════════════════════════
# 2. SPIDER (RADAR) PLOT
# ════════════════════════════════════════════════════════════════════════════

def create_spider_plot():
    """
    Create spider/radar plot showing normalized sensitivity across all parameters.
    Multiple scenarios compared on the same plot.
    """
    set_plot_style()
    
    log.info("Creating spider plot...")
    
    # Define scenarios to compare
    scenarios = {
        "Current (2024)": BASELINE_PARAMS.copy(),
        "Efficient (Low PUE)": {**BASELINE_PARAMS, "pue": 1.15, "renewable_pct": 0.40},
        "High Growth": {**BASELINE_PARAMS, "annual_growth_rate": 0.30, "it_capacity_mw": 150},
        "Climate Stress": {**BASELINE_PARAMS, "temperature_f": 85, "pue": 1.5},
    }
    
    # Metrics to plot
    metrics = ["total_power_mw", "carbon_tons", "cooling_load_mwh", "peak_power_mw"]
    metric_labels = ["Total Power", "Carbon Emissions", "Cooling Load", "Peak Demand"]
    
    # Calculate metrics for each scenario (normalized to baseline)
    baseline_metrics = calculate_2030_metrics(BASELINE_PARAMS)
    
    scenario_data = {}
    for name, params in scenarios.items():
        result = calculate_2030_metrics(params)
        normalized = [result[m] / baseline_metrics[m] for m in metrics]
        scenario_data[name] = normalized
    
    # Create radar plot
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]
    
    for i, (name, values) in enumerate(scenario_data.items()):
        values_plot = values + values[:1]  # Complete the loop
        ax.plot(angles, values_plot, 'o-', linewidth=2, label=name, color=colors[i])
        ax.fill(angles, values_plot, alpha=0.1, color=colors[i])
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, size=11)
    
    # Add reference circles
    ax.set_ylim(0, 2.5)
    ax.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_yticklabels(["0.5x", "1.0x", "1.5x", "2.0x", "2.5x"], size=9)
    
    # Reference line at 1.0 (baseline)
    ax.plot(angles, [1.0] * (num_vars + 1), '--', color='gray', linewidth=1, alpha=0.5)
    
    ax.set_title("Scenario Comparison (2030 Projection)\nNormalized to Current Baseline", 
                 fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    save_fig(fig, "spider_plot_scenarios")
    
    return fig


# ════════════════════════════════════════════════════════════════════════════
# 3. SCENARIO MATRIX HEATMAP (2D Parameter Sweep)
# ════════════════════════════════════════════════════════════════════════════

def create_scenario_matrix(param1="pue", param2="utilization", output="carbon_tons"):
    """
    Create 2D heatmap showing interaction between two parameters.
    """
    set_plot_style()
    
    log.info(f"Creating scenario matrix: {param1} × {param2} → {output}...")
    
    # Get parameter ranges
    vals1 = SENSITIVITY_RANGES.get(param1, [0.8, 0.9, 1.0, 1.1, 1.2])
    vals2 = SENSITIVITY_RANGES.get(param2, [0.5, 0.6, 0.7, 0.8, 0.9])
    
    # Calculate output for all combinations
    matrix = np.zeros((len(vals2), len(vals1)))
    
    for i, v2 in enumerate(vals2):
        for j, v1 in enumerate(vals1):
            params = BASELINE_PARAMS.copy()
            params[param1] = v1
            params[param2] = v2
            result = calculate_2030_metrics(params)
            matrix[i, j] = result[output]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Custom colormap (green to yellow to red)
    cmap = LinearSegmentedColormap.from_list("custom", ["#2ecc71", "#f1c40f", "#e74c3c"])
    
    im = ax.imshow(matrix, cmap=cmap, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"{output.replace('_', ' ').title()}", fontsize=11)
    
    # Set ticks
    ax.set_xticks(np.arange(len(vals1)))
    ax.set_yticks(np.arange(len(vals2)))
    ax.set_xticklabels([f"{v:.2f}" if isinstance(v, float) else str(v) for v in vals1])
    ax.set_yticklabels([f"{v:.2f}" if isinstance(v, float) else str(v) for v in vals2])
    
    ax.set_xlabel(PARAM_LABELS.get(param1, param1), fontsize=12)
    ax.set_ylabel(PARAM_LABELS.get(param2, param2), fontsize=12)
    ax.set_title(f"Scenario Matrix: {output.replace('_', ' ').title()} (2030)\n"
                 f"Interaction: {PARAM_LABELS.get(param1, param1)} × {PARAM_LABELS.get(param2, param2)}",
                 fontsize=14)
    
    # Add text annotations
    for i in range(len(vals2)):
        for j in range(len(vals1)):
            value = matrix[i, j]
            text_color = "white" if value > matrix.mean() else "black"
            ax.text(j, i, f"{value:,.0f}", ha="center", va="center", 
                    color=text_color, fontsize=9)
    
    plt.tight_layout()
    save_fig(fig, f"scenario_matrix_{param1}_{param2}_{output}")
    
    return fig, matrix


# ════════════════════════════════════════════════════════════════════════════
# 4. WATERFALL CHART (Impact Decomposition)
# ════════════════════════════════════════════════════════════════════════════

def create_waterfall_chart():
    """
    Create waterfall chart showing step-by-step carbon impact decomposition.
    Shows contribution of each factor from baseline to final.
    """
    set_plot_style()
    
    log.info("Creating waterfall chart...")
    
    # Define the transformation steps
    steps = [
        ("2024 Baseline", BASELINE_PARAMS, None),
        ("+ Growth to 2030", {"annual_growth_rate": 0.15}, "annual_growth_rate"),
        ("+ Higher Temps", {"temperature_f": 75}, "temperature_f"),
        ("+ PUE Improvement", {"pue": 1.2}, "pue"),
        ("+ 40% Renewables", {"renewable_pct": 0.40}, "renewable_pct"),
    ]
    
    # Calculate cumulative carbon at each step
    cumulative_params = BASELINE_PARAMS.copy()
    values = []
    labels = []
    
    for name, changes, _ in steps:
        if changes and name != "2024 Baseline":
            cumulative_params.update(changes)
        result = calculate_2030_metrics(cumulative_params)
        values.append(result["carbon_tons"])
        labels.append(name)
    
    # Calculate deltas
    deltas = [values[0]] + [values[i] - values[i-1] for i in range(1, len(values))]
    
    # Create waterfall
    fig, ax = plt.subplots(figsize=(12, 8))
    
    running_total = 0
    x_pos = np.arange(len(labels))
    
    colors = []
    for i, delta in enumerate(deltas):
        if i == 0:
            colors.append("#3498db")  # Blue for baseline
        elif delta > 0:
            colors.append("#e74c3c")  # Red for increase
        else:
            colors.append("#2ecc71")  # Green for decrease
    
    # Draw bars
    bottoms = []
    for i, delta in enumerate(deltas):
        if i == 0:
            bottom = 0
        else:
            bottom = running_total
        bottoms.append(bottom if delta > 0 else bottom + delta)
        running_total += delta
    
    bars = ax.bar(x_pos, [abs(d) for d in deltas], bottom=bottoms, color=colors, 
                  edgecolor='black', linewidth=1.5)
    
    # Add connecting lines
    running = 0
    for i in range(len(values) - 1):
        running += deltas[i]
        ax.hlines(running, i + 0.4, i + 0.6, colors='black', linewidth=1.5)
    
    # Add value labels
    for i, (bar, val, delta) in enumerate(zip(bars, values, deltas)):
        height = bar.get_height()
        y_pos = bar.get_y() + height / 2
        
        label = f"{val:,.0f}"
        if i > 0:
            sign = "+" if delta > 0 else ""
            label = f"{sign}{delta:,.0f}\n({val:,.0f})"
        
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, label,
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel("Carbon Emissions (tons CO₂/year)", fontsize=12)
    ax.set_title("Waterfall Analysis: Carbon Impact Decomposition\n"
                 "From 2024 Baseline to 2030 Optimized Scenario", fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Baseline'),
        Patch(facecolor='#e74c3c', label='Increase'),
        Patch(facecolor='#2ecc71', label='Reduction'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    save_fig(fig, "waterfall_carbon_decomposition")
    
    return fig


# ════════════════════════════════════════════════════════════════════════════
# 5. CONTOUR PLOT (3D Sensitivity Surface)
# ════════════════════════════════════════════════════════════════════════════

def create_contour_plot(param1="temperature_f", param2="pue", output="total_power_mw"):
    """
    Create contour plot showing continuous sensitivity surface.
    """
    set_plot_style()
    
    log.info(f"Creating contour plot: {param1} × {param2} → {output}...")
    
    # Create fine grid
    vals1 = np.linspace(min(SENSITIVITY_RANGES[param1]), max(SENSITIVITY_RANGES[param1]), 50)
    vals2 = np.linspace(min(SENSITIVITY_RANGES[param2]), max(SENSITIVITY_RANGES[param2]), 50)
    
    X, Y = np.meshgrid(vals1, vals2)
    Z = np.zeros_like(X)
    
    for i in range(len(vals2)):
        for j in range(len(vals1)):
            params = BASELINE_PARAMS.copy()
            params[param1] = X[i, j]
            params[param2] = Y[i, j]
            result = calculate_annual_metrics(params)
            Z[i, j] = result[output]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Filled contours
    levels = 20
    cf = ax.contourf(X, Y, Z, levels=levels, cmap="RdYlGn_r")
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label(f"{output.replace('_', ' ').title()}", fontsize=11)
    
    # Contour lines
    cs = ax.contour(X, Y, Z, levels=10, colors='black', linewidths=0.5, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.0f')
    
    # Mark baseline
    baseline1 = BASELINE_PARAMS[param1]
    baseline2 = BASELINE_PARAMS[param2]
    ax.plot(baseline1, baseline2, 'ko', markersize=12, label='Baseline')
    ax.plot(baseline1, baseline2, 'w+', markersize=10, mew=2)
    
    ax.set_xlabel(PARAM_LABELS.get(param1, param1), fontsize=12)
    ax.set_ylabel(PARAM_LABELS.get(param2, param2), fontsize=12)
    ax.set_title(f"Sensitivity Surface: {output.replace('_', ' ').title()}\n"
                 f"{PARAM_LABELS.get(param1, param1)} vs {PARAM_LABELS.get(param2, param2)}",
                 fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    save_fig(fig, f"contour_{param1}_{param2}_{output}")
    
    return fig


# ════════════════════════════════════════════════════════════════════════════
# 6. MULTI-OUTPUT SENSITIVITY DASHBOARD
# ════════════════════════════════════════════════════════════════════════════

def create_sensitivity_dashboard():
    """
    Create comprehensive dashboard with multiple sensitivity views.
    """
    set_plot_style()
    
    log.info("Creating sensitivity dashboard...")
    
    fig = plt.figure(figsize=(20, 16))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 1. Tornado diagram (top-left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Calculate sensitivities
    baseline_result = calculate_2030_metrics(BASELINE_PARAMS)
    baseline_carbon = baseline_result["carbon_tons"]
    
    sensitivities = []
    for param in ["it_capacity_mw", "pue", "utilization", "temperature_f", 
                  "grid_carbon_intensity", "renewable_pct"]:
        values = SENSITIVITY_RANGES[param]
        
        low_params = BASELINE_PARAMS.copy()
        low_params[param] = min(values)
        low_result = calculate_2030_metrics(low_params)["carbon_tons"]
        
        high_params = BASELINE_PARAMS.copy()
        high_params[param] = max(values)
        high_result = calculate_2030_metrics(high_params)["carbon_tons"]
        
        low_pct = ((low_result - baseline_carbon) / baseline_carbon) * 100
        high_pct = ((high_result - baseline_carbon) / baseline_carbon) * 100
        
        sensitivities.append({
            "parameter": PARAM_LABELS.get(param, param),
            "low_pct": low_pct,
            "high_pct": high_pct,
        })
    
    sensitivities = sorted(sensitivities, key=lambda x: abs(x["high_pct"] - x["low_pct"]), reverse=True)
    
    y_pos = np.arange(len(sensitivities))
    for i, s in enumerate(sensitivities):
        color = "#e74c3c" if s["high_pct"] > 0 else "#2ecc71"
        ax1.barh(i, s["high_pct"], color=color, alpha=0.7, height=0.6)
        ax1.barh(i, s["low_pct"], color="#2ecc71" if s["high_pct"] > 0 else "#e74c3c", 
                 alpha=0.7, height=0.6)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([s["parameter"] for s in sensitivities])
    ax1.axvline(0, color="black", linewidth=2)
    ax1.set_xlabel("% Change from Baseline")
    ax1.set_title("Tornado: Carbon Sensitivity Ranking", fontsize=12)
    ax1.grid(True, axis='x', alpha=0.3)
    
    # 2. PUE vs Temperature heatmap (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    
    temps = [55, 65, 75, 85, 95]
    pues = [1.1, 1.2, 1.35, 1.5, 1.6]
    power_matrix = np.zeros((len(temps), len(pues)))
    
    for i, t in enumerate(temps):
        for j, p in enumerate(pues):
            params = BASELINE_PARAMS.copy()
            params["temperature_f"] = t
            params["pue"] = p
            power_matrix[i, j] = calculate_annual_metrics(params)["total_power_mw"]
    
    im = ax2.imshow(power_matrix, cmap="YlOrRd", aspect='auto')
    ax2.set_xticks(np.arange(len(pues)))
    ax2.set_yticks(np.arange(len(temps)))
    ax2.set_xticklabels([f"{p:.1f}" for p in pues])
    ax2.set_yticklabels([f"{t}°F" for t in temps])
    ax2.set_xlabel("PUE")
    ax2.set_ylabel("Temperature")
    ax2.set_title("Power: Temp × PUE", fontsize=12)
    plt.colorbar(im, ax=ax2, label="MW")
    
    # 3. Growth scenario comparison (middle-left)
    ax3 = fig.add_subplot(gs[1, 0])
    
    years = list(range(2024, 2036))
    growth_rates = [0.05, 0.15, 0.30]
    
    for rate in growth_rates:
        powers = []
        for year in years:
            params = BASELINE_PARAMS.copy()
            params["annual_growth_rate"] = rate
            factor = (1 + rate) ** (year - 2024)
            power = BASELINE_PARAMS["it_capacity_mw"] * BASELINE_PARAMS["utilization"] * factor
            powers.append(power * params["pue"])
        ax3.plot(years, powers, marker='o', label=f"{int(rate*100)}% growth")
    
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Total Power (MW)")
    ax3.set_title("Growth Scenarios: Power Demand", fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Renewable impact (middle-center)
    ax4 = fig.add_subplot(gs[1, 1])
    
    renewable_pcts = np.linspace(0, 1, 20)
    carbon_values = []
    
    for r in renewable_pcts:
        params = BASELINE_PARAMS.copy()
        params["renewable_pct"] = r
        carbon_values.append(calculate_2030_metrics(params)["carbon_tons"])
    
    ax4.fill_between(renewable_pcts * 100, carbon_values, alpha=0.3, color='green')
    ax4.plot(renewable_pcts * 100, carbon_values, 'g-', linewidth=2)
    ax4.set_xlabel("Renewable Energy (%)")
    ax4.set_ylabel("Carbon Emissions (tons)")
    ax4.set_title("Renewable Impact on Carbon", fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # 5. Utilization vs Capacity heatmap (middle-right)
    ax5 = fig.add_subplot(gs[1, 2])
    
    utils = [0.4, 0.55, 0.70, 0.85, 0.95]
    caps = [50, 100, 150, 200, 300]
    carbon_matrix = np.zeros((len(utils), len(caps)))
    
    for i, u in enumerate(utils):
        for j, c in enumerate(caps):
            params = BASELINE_PARAMS.copy()
            params["utilization"] = u
            params["it_capacity_mw"] = c
            carbon_matrix[i, j] = calculate_2030_metrics(params)["carbon_tons"]
    
    im = ax5.imshow(carbon_matrix, cmap="RdYlGn_r", aspect='auto')
    ax5.set_xticks(np.arange(len(caps)))
    ax5.set_yticks(np.arange(len(utils)))
    ax5.set_xticklabels([str(c) for c in caps])
    ax5.set_yticklabels([f"{u*100:.0f}%" for u in utils])
    ax5.set_xlabel("IT Capacity (MW)")
    ax5.set_ylabel("Utilization")
    ax5.set_title("Carbon: Util × Capacity", fontsize=12)
    plt.colorbar(im, ax=ax5, label="tons CO₂")
    
    # 6. Key metrics summary (bottom-left)
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.axis('off')
    
    summary_text = f"""
BASELINE PARAMETERS
─────────────────────
IT Capacity:     {BASELINE_PARAMS['it_capacity_mw']} MW
PUE:             {BASELINE_PARAMS['pue']:.2f}
Utilization:     {BASELINE_PARAMS['utilization']*100:.0f}%
Temperature:     {BASELINE_PARAMS['temperature_f']}°F
Carbon Intensity: {BASELINE_PARAMS['grid_carbon_intensity']} g/kWh
Growth Rate:     {BASELINE_PARAMS['annual_growth_rate']*100:.0f}%/year
Renewable:       {BASELINE_PARAMS['renewable_pct']*100:.0f}%

2030 PROJECTIONS
─────────────────────
Total Power:     {baseline_result['total_power_mw']:.1f} MW
Peak Power:      {baseline_result['peak_power_mw']:.1f} MW
Carbon:          {baseline_result['carbon_tons']:,.0f} tons/year
    """
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax6.set_title("Baseline Summary", fontsize=12)
    
    # 7. Temperature sensitivity curve (bottom-center)
    ax7 = fig.add_subplot(gs[2, 1])
    
    temps = np.linspace(50, 100, 30)
    powers = []
    pue_values = []
    
    for t in temps:
        params = BASELINE_PARAMS.copy()
        params["temperature_f"] = t
        result = calculate_annual_metrics(params)
        powers.append(result["total_power_mw"])
        pue_values.append(result["pue_effective"])
    
    ax7_twin = ax7.twinx()
    ax7.plot(temps, powers, 'b-', linewidth=2, label='Power')
    ax7_twin.plot(temps, pue_values, 'r--', linewidth=2, label='Effective PUE')
    
    ax7.set_xlabel("Ambient Temperature (°F)")
    ax7.set_ylabel("Total Power (MW)", color='blue')
    ax7_twin.set_ylabel("Effective PUE", color='red')
    ax7.set_title("Temperature Impact", fontsize=12)
    ax7.axvline(65, color='gray', linestyle=':', label='Cooling Threshold')
    ax7.legend(loc='upper left')
    ax7_twin.legend(loc='upper right')
    ax7.grid(True, alpha=0.3)
    
    # 8. Carbon breakdown pie (bottom-right)
    ax8 = fig.add_subplot(gs[2, 2])
    
    # Calculate carbon sources
    it_power = BASELINE_PARAMS["it_capacity_mw"] * BASELINE_PARAMS["utilization"]
    cooling_power = it_power * (BASELINE_PARAMS["pue"] - 1)
    
    sizes = [it_power, cooling_power]
    labels = [f'IT Load\n({it_power:.0f} MW)', f'Cooling\n({cooling_power:.0f} MW)']
    colors = ['#3498db', '#e74c3c']
    explode = (0.05, 0)
    
    ax8.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax8.set_title("Power Breakdown", fontsize=12)
    
    plt.suptitle("DATACENTER SENSITIVITY ANALYSIS DASHBOARD\n"
                 "Virginia 100MW AI Datacenter — 2030 Projections", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    save_fig(fig, "sensitivity_dashboard")
    
    return fig


# ════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ════════════════════════════════════════════════════════════════════════════

def run_all_analyses():
    """Execute all sensitivity analyses and generate figures."""
    set_plot_style()
    
    log.info("╔" + "═" * 60 + "╗")
    log.info("║   ADVANCED SENSITIVITY ANALYSIS                           ║")
    log.info("╚" + "═" * 60 + "╝")
    
    figures = {}
    
    # 1. Tornado Diagrams
    log.info("\n--- 1. TORNADO DIAGRAMS ---")
    for metric in ["carbon_tons", "total_power_mw"]:
        fig, _ = create_tornado_diagram(metric)
        figures[f"tornado_{metric}"] = fig
    
    # 2. Spider Plot
    log.info("\n--- 2. SPIDER PLOT ---")
    figures["spider"] = create_spider_plot()
    
    # 3. Scenario Matrices
    log.info("\n--- 3. SCENARIO MATRICES ---")
    combinations = [
        ("pue", "utilization", "carbon_tons"),
        ("temperature_f", "pue", "total_power_mw"),
        ("it_capacity_mw", "annual_growth_rate", "carbon_tons"),
    ]
    for p1, p2, output in combinations:
        fig, _ = create_scenario_matrix(p1, p2, output)
        figures[f"matrix_{p1}_{p2}"] = fig
    
    # 4. Waterfall Chart
    log.info("\n--- 4. WATERFALL CHART ---")
    figures["waterfall"] = create_waterfall_chart()
    
    # 5. Contour Plots
    log.info("\n--- 5. CONTOUR PLOTS ---")
    contour_combos = [
        ("temperature_f", "pue", "total_power_mw"),
        ("utilization", "renewable_pct", "carbon_tons"),
    ]
    for p1, p2, output in contour_combos:
        fig = create_contour_plot(p1, p2, output)
        figures[f"contour_{p1}_{p2}"] = fig
    
    # 6. Comprehensive Dashboard
    log.info("\n--- 6. SENSITIVITY DASHBOARD ---")
    figures["dashboard"] = create_sensitivity_dashboard()
    
    # Summary
    log.info("\n" + "=" * 60)
    log.info(f"ANALYSIS COMPLETE: {len(figures)} figures generated")
    log.info(f"  Figures saved to: {cfg.FIGURE_DIR}")
    log.info("=" * 60)
    
    return figures


if __name__ == "__main__":
    run_all_analyses()
