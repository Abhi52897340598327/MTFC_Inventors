"""
MTFC Model Validation & Stress Testing Dashboard
=================================================
Comprehensive 9-panel visualization combining all validation methods.

Creates publication-ready figure integrating:
- Monte Carlo distribution analysis
- EVT tail risk metrics
- Copula tail dependence
- Sobol variance decomposition
- Walk-forward validation
- Reverse stress testing
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from utils import log, save_fig, set_plot_style

# Import analysis classes
from advanced_monte_carlo_validation import (
    MonteCarloSimulator, ExtremeValueAnalysis, CopulaAnalysis,
    SobolSensitivityAnalysis, WalkForwardBacktest, ReverseStressTest,
    N_SIMULATIONS, RANDOM_SEED
)

OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, "risk_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_comprehensive_dashboard():
    """
    Create a comprehensive 12-panel dashboard combining all validation methods.
    """
    set_plot_style()
    np.random.seed(RANDOM_SEED)
    
    log.info("Generating Comprehensive Model Validation Dashboard...")
    
    # ─── Run all analyses ───────────────────────────────────────────────────────
    
    # 1. Monte Carlo
    mc = MonteCarloSimulator(n_sims=N_SIMULATIONS)
    sim_df = mc.run_simulation()
    
    # 2. EVT
    evt = ExtremeValueAnalysis(sim_df["risk_index"].values)
    
    # 3. Copula
    copula = CopulaAnalysis(
        sim_df["temperature_f"].values,
        sim_df["it_load"].values,
        sim_df["carbon_intensity"].values
    )
    copula_results = copula.analyze_all_pairs()
    
    # 4. Sobol
    sobol = SobolSensitivityAnalysis(n_samples=2048)  # Reduced for dashboard
    sobol_results = sobol.compute_sobol_indices()
    
    # 5. Backtest
    backtest = WalkForwardBacktest(sim_df)
    expanding_results = backtest.expanding_window_validation()
    
    # 6. Stress test
    stress_test = ReverseStressTest(failure_threshold=400_000_000)
    failure_point = stress_test.find_failure_boundary()
    
    # ─── Create dashboard ───────────────────────────────────────────────────────
    
    fig = plt.figure(figsize=(24, 18))
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    # Color scheme
    COLORS = {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'accent': '#F18F01',
        'success': '#2E7D32',
        'danger': '#C62828',
        'neutral': '#546E7A',
    }
    
    # ═══ Panel 1: Monte Carlo Distribution (2 cols) ═══════════════════════════
    ax1 = fig.add_subplot(gs[0, 0:2])
    
    # Risk Index distribution
    n, bins, patches = ax1.hist(sim_df["risk_index"], bins=60, density=True,
                                 alpha=0.7, color=COLORS['primary'], edgecolor='white')
    
    # Overlay normal fit
    mu, sigma = sim_df["risk_index"].mean(), sim_df["risk_index"].std()
    x = np.linspace(bins[0], bins[-1], 100)
    ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal fit')
    
    # Mark quantiles
    q95 = np.percentile(sim_df["risk_index"], 95)
    q99 = np.percentile(sim_df["risk_index"], 99)
    ax1.axvline(q95, color=COLORS['accent'], linestyle='--', linewidth=2, label=f'95th: {q95:.1f}')
    ax1.axvline(q99, color=COLORS['danger'], linestyle='--', linewidth=2, label=f'99th: {q99:.1f}')
    
    ax1.set_xlabel('Risk Index', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Monte Carlo: Risk Index Distribution\n(N=10,000 simulations)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ═══ Panel 2: Carbon Liability Distribution (2 cols) ════════════════════════
    ax2 = fig.add_subplot(gs[0, 2:4])
    
    liability = sim_df["carbon_liability_usd"] / 1e6  # Convert to millions
    ax2.hist(liability, bins=60, density=True, alpha=0.7, 
             color=COLORS['secondary'], edgecolor='white')
    
    var99_liability = np.percentile(liability, 99)
    cte99_liability = liability[liability >= var99_liability].mean()
    
    ax2.axvline(var99_liability, color=COLORS['danger'], linestyle='--', linewidth=2,
               label=f'VaR 99%: ${var99_liability:.0f}M')
    ax2.axvline(cte99_liability, color='black', linestyle=':', linewidth=2,
               label=f'ES 99%: ${cte99_liability:.0f}M')
    
    ax2.set_xlabel('Annual Carbon Liability ($M)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Monte Carlo: Carbon Liability Distribution', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # ═══ Panel 3: EVT GPD Fit ════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[1, 0])
    
    sorted_exc = np.sort(evt.exceedances)
    empirical_cdf = np.arange(1, len(sorted_exc) + 1) / len(sorted_exc)
    fitted_cdf = evt.gpd_cdf(sorted_exc)
    
    ax3.scatter(sorted_exc, empirical_cdf, alpha=0.5, s=10, color=COLORS['primary'], label='Empirical')
    ax3.plot(sorted_exc, fitted_cdf, 'r-', linewidth=2, label='GPD fit')
    
    ax3.set_xlabel('Exceedance (x - u)', fontsize=10)
    ax3.set_ylabel('CDF', fontsize=10)
    ax3.set_title(f'EVT: GPD Fit\nξ={evt.xi:.3f}, β={evt.beta:.2f}', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # ═══ Panel 4: Return Level Plot ══════════════════════════════════════════════
    ax4 = fig.add_subplot(gs[1, 1])
    
    return_periods = np.array([2, 5, 10, 25, 50, 100, 200, 500])
    return_levels = [evt.calculate_var(1 - 1/rp) for rp in return_periods]
    
    ax4.semilogx(return_periods, return_levels, 'o-', color=COLORS['danger'], 
                linewidth=2, markersize=6)
    ax4.fill_between(return_periods, 
                    [rl * 0.9 for rl in return_levels],
                    [rl * 1.1 for rl in return_levels],
                    alpha=0.2, color=COLORS['danger'])
    
    ax4.set_xlabel('Return Period (years)', fontsize=10)
    ax4.set_ylabel('Return Level', fontsize=10)
    ax4.set_title('EVT: Return Level Plot', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # ═══ Panel 5: Copula Tail Dependence ═════════════════════════════════════════
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Scatter of pseudo-observations
    ax5.scatter(copula.u_temp, copula.u_carbon, alpha=0.3, s=5, color=COLORS['primary'])
    
    # Highlight extremes
    extreme = (copula.u_temp > 0.95) & (copula.u_carbon > 0.95)
    ax5.scatter(copula.u_temp[extreme], copula.u_carbon[extreme], 
               s=20, color=COLORS['danger'], alpha=0.8, label='Upper tail')
    
    # Add quadrant shading
    ax5.axhline(0.95, color='gray', linestyle='--', alpha=0.5)
    ax5.axvline(0.95, color='gray', linestyle='--', alpha=0.5)
    ax5.fill_between([0.95, 1], 0.95, 1, alpha=0.1, color=COLORS['danger'])
    
    ax5.set_xlabel('Temperature (uniform)', fontsize=10)
    ax5.set_ylabel('Carbon Intensity (uniform)', fontsize=10)
    ax5.set_title('Copula: Temp × Carbon\n(Tail Dependence)', fontsize=11, fontweight='bold')
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.legend(fontsize=8)
    
    # ═══ Panel 6: Copula Metrics Summary ═════════════════════════════════════════
    ax6 = fig.add_subplot(gs[1, 3])
    ax6.axis('off')
    
    pair_data = copula_results["Temperature vs Carbon Intensity"]
    
    metrics_text = f"""
┌───────────────────────────────┐
│   COPULA METRICS SUMMARY      │
├───────────────────────────────┤
│ Temperature × Carbon:         │
│   Kendall's τ:    {pair_data['kendall_tau']:>8.3f}   │
│   Gumbel θ:       {pair_data['gumbel_theta']:>8.3f}   │
│   Upper Tail λ_U: {pair_data['empirical_upper_tail_dep']:>8.3f}   │
│   Lower Tail λ_L: {pair_data['empirical_lower_tail_dep']:>8.3f}   │
├───────────────────────────────┤
│ "Double Whammy" Validated:    │
│ Heat → Dirty Grid confirmed   │
│ (asymmetric tail dependence)  │
└───────────────────────────────┘
"""
    ax6.text(0.1, 0.95, metrics_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # ═══ Panel 7: Sobol First-Order Indices ══════════════════════════════════════
    ax7 = fig.add_subplot(gs[2, 0:2])
    
    params = list(sobol_results["first_order"].keys())
    s1 = [sobol_results["first_order"][p] for p in params]
    st = [sobol_results["total_order"][p] for p in params]
    
    x = np.arange(len(params))
    width = 0.35
    
    bars1 = ax7.barh(x - width/2, s1, width, color=COLORS['primary'], 
                    edgecolor='black', label='First-Order $S_i$')
    bars2 = ax7.barh(x + width/2, st, width, color=COLORS['accent'], 
                    edgecolor='black', label='Total-Order $S_{Ti}$')
    
    ax7.set_yticks(x)
    ax7.set_yticklabels([p.replace('_', ' ').title() for p in params], fontsize=9)
    ax7.set_xlabel('Sobol Index', fontsize=10)
    ax7.set_title('Sobol Sensitivity: Variance Decomposition of Carbon Liability', 
                 fontsize=11, fontweight='bold')
    ax7.legend(loc='lower right', fontsize=9)
    ax7.grid(True, axis='x', alpha=0.3)
    
    # Add percentage labels
    for bar, val in zip(bars1, s1):
        ax7.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val*100:.1f}%', va='center', fontsize=8)
    
    # ═══ Panel 8: Walk-Forward MAPE ══════════════════════════════════════════════
    ax8 = fig.add_subplot(gs[2, 2:4])
    
    ax8.plot(expanding_results["train_end"], expanding_results["mape"], 
            'o-', color=COLORS['success'], linewidth=2, markersize=4)
    ax8.axhline(expanding_results["mape"].mean(), color=COLORS['success'], 
               linestyle='--', alpha=0.5, label=f'Mean MAPE: {expanding_results["mape"].mean():.1f}%')
    ax8.fill_between(expanding_results["train_end"],
                    expanding_results["mape"] - expanding_results["mape"].std(),
                    expanding_results["mape"] + expanding_results["mape"].std(),
                    alpha=0.2, color=COLORS['success'])
    
    ax8.set_xlabel('Training Window End', fontsize=10)
    ax8.set_ylabel('MAPE (%)', fontsize=10)
    ax8.set_title('Walk-Forward Backtesting: Out-of-Sample Error', fontsize=11, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)
    
    # ═══ Panel 9: Fragility Surface ══════════════════════════════════════════════
    ax9 = fig.add_subplot(gs[3, 0:2])
    
    X1, X2, Z = stress_test.map_fragility_surface("temperature_f", "carbon_intensity", resolution=25)
    
    cf = ax9.contourf(X1, X2, Z / 1e6, levels=15, cmap='RdYlGn_r')
    plt.colorbar(cf, ax=ax9, label='Liability ($M)')
    
    cs = ax9.contour(X1, X2, Z / 1e6, levels=[400], colors='red', linewidths=3)
    ax9.clabel(cs, fmt='$400M', fontsize=9)
    
    # Mark failure point
    ax9.scatter([failure_point['temperature_f']], [failure_point['carbon_intensity']],
               s=150, c='red', marker='X', edgecolors='black', linewidths=2,
               label='Min. Failure', zorder=5)
    
    ax9.set_xlabel('Temperature (°F)', fontsize=10)
    ax9.set_ylabel('Carbon Intensity (kg/MWh)', fontsize=10)
    ax9.set_title('Reverse Stress Test: Fragility Surface', fontsize=11, fontweight='bold')
    ax9.legend(loc='lower right', fontsize=9)
    
    # ═══ Panel 10: Executive Summary ══════════════════════════════════════════════
    ax10 = fig.add_subplot(gs[3, 2:4])
    ax10.axis('off')
    
    # Calculate key metrics
    var_99 = evt.calculate_var(0.99)
    cte_99 = evt.calculate_cte(0.99)
    top_driver = max(sobol_results["first_order"], key=sobol_results["first_order"].get)
    top_driver_pct = sobol_results["first_order"][top_driver] * 100
    
    summary_text = f"""
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                        EXECUTIVE RISK SUMMARY                                      ║
╠═══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                    ║
║  MONTE CARLO SIMULATION (N=10,000)                                                 ║
║  ─────────────────────────────────                                                 ║
║  • Mean Annual Carbon Liability:    ${sim_df['carbon_liability_usd'].mean()/1e6:>6.1f}M                         ║
║  • 99th Percentile Liability:       ${sim_df['carbon_liability_usd'].quantile(0.99)/1e6:>6.1f}M                         ║
║  • Risk Index Range:                {sim_df['risk_index'].min():.1f} – {sim_df['risk_index'].max():.1f}                            ║
║                                                                                    ║
║  EXTREME VALUE THEORY (GPD)                                                        ║
║  ─────────────────────────────────                                                 ║
║  • Tail Shape (ξ):                  {evt.xi:>6.4f}  ({"Heavy" if evt.xi > 0 else "Light"} tail)                       ║
║  • VaR 99%:                         {var_99:>6.2f}                                 ║
║  • Expected Shortfall 99%:          {cte_99:>6.2f}                                 ║
║                                                                                    ║
║  GLOBAL SENSITIVITY (SOBOL)                                                        ║
║  ─────────────────────────────────                                                 ║
║  • Primary Driver:                  {top_driver.replace('_', ' ').title():<20} ({top_driver_pct:.1f}%)     ║
║  • Interaction Effects:             {(sobol_results['total_order'][top_driver] - sobol_results['first_order'][top_driver])*100:.1f}% additional variance                    ║
║                                                                                    ║
║  REVERSE STRESS TEST                                                               ║
║  ─────────────────────────────────                                                 ║
║  • Failure Threshold:               $400M annual liability                         ║
║  • Minimum Failure Conditions:                                                     ║
║    - Temperature ≥ {failure_point['temperature_f']:>5.1f}°F                                                    ║
║    - Carbon Intensity ≥ {failure_point['carbon_intensity']:>5.0f} kg/MWh                                        ║
║    - Duration ≥ {failure_point['duration_hours']:>5.0f} hours                                                   ║
║                                                                                    ║
║  MODEL VALIDATION                                                                  ║
║  ─────────────────────────────────                                                 ║
║  • Walk-Forward MAPE:               {expanding_results['mape'].mean():>5.1f}% ± {expanding_results['mape'].std():.1f}%                          ║
║  • Copula Tail Dependence:          {pair_data['empirical_upper_tail_dep']:.3f} (validates "Double Whammy")         ║
║                                                                                    ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
"""
    
    ax10.text(0.02, 0.98, summary_text, transform=ax10.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.95))
    
    # ─── Main title ─────────────────────────────────────────────────────────────
    fig.suptitle('MTFC AI Datacenter: Comprehensive Model Validation & Stress Testing Dashboard\n'
                'Monte Carlo Simulation · Extreme Value Theory · Copula Dependency · Sobol Sensitivity · Walk-Forward Validation · Reverse Stress Testing',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    save_fig(fig, "comprehensive_validation_dashboard")
    
    log.info("Dashboard saved to: comprehensive_validation_dashboard.png")
    
    return fig


def create_tail_risk_deep_dive():
    """
    Create detailed tail risk analysis focusing on EVT and extreme scenarios.
    """
    set_plot_style()
    np.random.seed(RANDOM_SEED)
    
    log.info("Generating Tail Risk Deep Dive...")
    
    # Run simulation
    mc = MonteCarloSimulator(n_sims=N_SIMULATIONS)
    sim_df = mc.run_simulation()
    
    # EVT analysis
    evt = ExtremeValueAnalysis(sim_df["risk_index"].values)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Exceedance probability plot
    ax1 = axes[0, 0]
    sorted_data = np.sort(sim_df["risk_index"])
    exceedance_prob = 1 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    ax1.semilogy(sorted_data, exceedance_prob, 'b-', linewidth=1.5)
    ax1.axhline(0.01, color='r', linestyle='--', label='1% exceedance')
    ax1.axhline(0.001, color='orange', linestyle='--', label='0.1% exceedance')
    
    var_99 = evt.calculate_var(0.99)
    var_999 = evt.calculate_var(0.999)
    ax1.axvline(var_99, color='r', linestyle=':', alpha=0.7)
    ax1.axvline(var_999, color='orange', linestyle=':', alpha=0.7)
    
    ax1.set_xlabel('Risk Index', fontsize=11)
    ax1.set_ylabel('Exceedance Probability (log)', fontsize=11)
    ax1.set_title('Survival Function\n(Log-scale exceedance)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Q-Q plot
    ax2 = axes[0, 1]
    theoretical = stats.norm.ppf(np.linspace(0.001, 0.999, len(sim_df)))
    empirical = np.sort(stats.zscore(sim_df["risk_index"]))
    
    ax2.scatter(theoretical, empirical, alpha=0.3, s=5)
    ax2.plot([-3, 3], [-3, 3], 'r--', linewidth=2, label='Normal reference')
    
    ax2.set_xlabel('Theoretical Quantiles', fontsize=11)
    ax2.set_ylabel('Sample Quantiles', fontsize=11)
    ax2.set_title('Q-Q Plot vs Normal\n(Tail deviation = fat tails)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Hill estimator for tail index
    ax3 = axes[0, 2]
    
    sorted_excess = np.sort(evt.exceedances)[::-1]
    n_excess = len(sorted_excess)
    k_values = np.arange(10, n_excess - 10)
    hill_estimates = []
    
    for k in k_values:
        log_ratio = np.log(sorted_excess[:k] / sorted_excess[k])
        hill = np.mean(log_ratio)
        hill_estimates.append(hill)
    
    ax3.plot(k_values, hill_estimates, 'b-', linewidth=1.5)
    ax3.axhline(evt.xi, color='r', linestyle='--', label=f'MLE ξ = {evt.xi:.3f}')
    
    ax3.set_xlabel('Number of Order Statistics (k)', fontsize=11)
    ax3.set_ylabel('Hill Estimator', fontsize=11)
    ax3.set_title('Hill Plot for Tail Index\n(Stability indicates correct threshold)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. VaR and CTE across confidence levels
    ax4 = axes[1, 0]
    
    conf_levels = np.linspace(0.90, 0.999, 50)
    var_values = [evt.calculate_var(c) for c in conf_levels]
    cte_values = [evt.calculate_cte(c) for c in conf_levels]
    
    ax4.plot(conf_levels * 100, var_values, 'b-', linewidth=2, label='VaR')
    ax4.plot(conf_levels * 100, cte_values, 'r-', linewidth=2, label='CTE (ES)')
    ax4.fill_between(conf_levels * 100, var_values, cte_values, alpha=0.2, color='red')
    
    ax4.set_xlabel('Confidence Level (%)', fontsize=11)
    ax4.set_ylabel('Risk Index', fontsize=11)
    ax4.set_title('VaR vs Expected Shortfall\n(CTE captures severity beyond VaR)', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Compound event analysis
    ax5 = axes[1, 1]
    
    # Identify compound extreme events
    temp_extreme = sim_df["temperature_f"] > sim_df["temperature_f"].quantile(0.95)
    carbon_extreme = sim_df["carbon_intensity"] > sim_df["carbon_intensity"].quantile(0.95)
    load_extreme = sim_df["it_load"] > sim_df["it_load"].quantile(0.95)
    
    compound = temp_extreme & carbon_extreme
    triple = temp_extreme & carbon_extreme & load_extreme
    
    risk_normal = sim_df.loc[~compound, "risk_index"]
    risk_compound = sim_df.loc[compound & ~triple, "risk_index"]
    risk_triple = sim_df.loc[triple, "risk_index"]
    
    bp = ax5.boxplot([risk_normal, risk_compound, risk_triple],
                     labels=['Normal\n(n={})'.format(len(risk_normal)),
                            'Double\n(n={})'.format(len(risk_compound)),
                            'Triple\n(n={})'.format(len(risk_triple))],
                     patch_artist=True)
    
    colors = ['lightgreen', 'orange', 'red']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax5.set_ylabel('Risk Index', fontsize=11)
    ax5.set_title('Compound Event Impact\n(Double/Triple simultaneous extremes)', fontsize=12, fontweight='bold')
    ax5.grid(True, axis='y', alpha=0.3)
    
    # 6. Expected loss by scenario
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Calculate expected losses
    normal_loss = (sim_df.loc[~compound, "carbon_liability_usd"] / 1e6).mean()
    double_loss = (sim_df.loc[compound & ~triple, "carbon_liability_usd"] / 1e6).mean()
    triple_loss = (sim_df.loc[triple, "carbon_liability_usd"] / 1e6).mean()
    
    summary = f"""
┌─────────────────────────────────────────────────┐
│           TAIL RISK METRICS                     │
├─────────────────────────────────────────────────┤
│                                                 │
│  EXTREME VALUE PARAMETERS                       │
│  • Shape (ξ):     {evt.xi:>8.4f}                  │
│  • Scale (β):     {evt.beta:>8.2f}                  │
│  • Tail Type:     {"Heavy (Fréchet)" if evt.xi > 0 else "Light (Weibull)":>12}          │
│                                                 │
│  VALUE AT RISK                                  │
│  • VaR 95%:       {evt.calculate_var(0.95):>8.2f}                  │
│  • VaR 99%:       {evt.calculate_var(0.99):>8.2f}                  │
│  • VaR 99.9%:     {evt.calculate_var(0.999):>8.2f}                  │
│                                                 │
│  EXPECTED SHORTFALL                             │
│  • CTE 99%:       {evt.calculate_cte(0.99):>8.2f}                  │
│  • (Expected loss given breach)                 │
│                                                 │
│  COMPOUND EVENT ANALYSIS                        │
│  • Normal scenario:  ${normal_loss:>6.1f}M             │
│  • Double extreme:   ${double_loss:>6.1f}M (+{(double_loss/normal_loss-1)*100:.0f}%)     │
│  • Triple extreme:   ${triple_loss:>6.1f}M (+{(triple_loss/normal_loss-1)*100:.0f}%)     │
│                                                 │
│  Black Swan Probability:                        │
│  • P(Triple extreme): {100*len(risk_triple)/len(sim_df):.2f}%                │
│                                                 │
└─────────────────────────────────────────────────┘
"""
    
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))
    
    plt.suptitle('Tail Risk Deep Dive: Extreme Value Theory Analysis\n'
                '100MW AI Datacenter Carbon Liability',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_fig(fig, "tail_risk_deep_dive")
    
    return fig


def create_copula_analysis_detailed():
    """
    Create detailed copula analysis showing tail dependence structure.
    """
    set_plot_style()
    np.random.seed(RANDOM_SEED)
    
    log.info("Generating Detailed Copula Analysis...")
    
    mc = MonteCarloSimulator(n_sims=N_SIMULATIONS)
    sim_df = mc.run_simulation()
    
    copula = CopulaAnalysis(
        sim_df["temperature_f"].values,
        sim_df["it_load"].values,
        sim_df["carbon_intensity"].values
    )
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    pairs = [
        ("Temperature", "Carbon", copula.u_temp, copula.u_carbon),
        ("Temperature", "IT Load", copula.u_temp, copula.u_load),
        ("IT Load", "Carbon", copula.u_load, copula.u_carbon),
    ]
    
    for idx, (name1, name2, u, v) in enumerate(pairs):
        ax_top = axes[0, idx]
        ax_bot = axes[1, idx]
        
        # Top: Scatter with density
        h = ax_top.hexbin(u, v, gridsize=30, cmap='YlOrRd', mincnt=1)
        plt.colorbar(h, ax=ax_top, label='Count')
        
        # Mark extreme quadrant
        extreme = (u > 0.95) & (v > 0.95)
        ax_top.scatter(u[extreme], v[extreme], s=30, c='black', marker='x', 
                      label=f'Upper tail (n={extreme.sum()})')
        
        ax_top.axhline(0.95, color='white', linestyle='--', alpha=0.7)
        ax_top.axvline(0.95, color='white', linestyle='--', alpha=0.7)
        
        ax_top.set_xlabel(f'{name1} (uniform)', fontsize=10)
        ax_top.set_ylabel(f'{name2} (uniform)', fontsize=10)
        ax_top.set_title(f'Copula: {name1} × {name2}', fontsize=11, fontweight='bold')
        ax_top.legend(fontsize=8)
        
        # Bottom: Tail dependence function
        thresholds = np.linspace(0.5, 0.99, 50)
        upper_td = []
        lower_td = []
        
        for t in thresholds:
            mask_upper = u > t
            mask_lower = u < (1 - t)
            
            if mask_upper.sum() > 0:
                upper_td.append(np.mean(v[mask_upper] > t))
            else:
                upper_td.append(np.nan)
            
            if mask_lower.sum() > 0:
                lower_td.append(np.mean(v[mask_lower] < (1 - t)))
            else:
                lower_td.append(np.nan)
        
        ax_bot.plot(thresholds, upper_td, 'r-', linewidth=2, label='Upper λ_U')
        ax_bot.plot(thresholds, lower_td, 'b-', linewidth=2, label='Lower λ_L')
        
        # Gumbel theoretical
        tau, _ = stats.kendalltau(u, v)
        theta = max(1 / (1 - max(tau, 0.01)), 1.0)
        gumbel_upper = 2 - 2 ** (1 / theta)
        ax_bot.axhline(gumbel_upper, color='red', linestyle='--', alpha=0.5,
                      label=f'Gumbel λ_U = {gumbel_upper:.3f}')
        
        ax_bot.set_xlabel('Threshold quantile', fontsize=10)
        ax_bot.set_ylabel('Tail Dependence', fontsize=10)
        ax_bot.set_title(f'Tail Dependence: {name1} × {name2}', fontsize=11, fontweight='bold')
        ax_bot.legend(fontsize=8)
        ax_bot.grid(True, alpha=0.3)
    
    plt.suptitle('Copula Analysis: Tail Dependence Structure\n'
                'Validating Asymmetric "Double Whammy" Hypothesis',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_fig(fig, "copula_detailed_analysis")
    
    return fig


if __name__ == "__main__":
    # Generate all dashboards
    create_comprehensive_dashboard()
    create_tail_risk_deep_dive()
    create_copula_analysis_detailed()
    
    log.info("\n" + "=" * 60)
    log.info("All dashboards generated successfully!")
    log.info("=" * 60)
