#!/usr/bin/env python3
"""
MTFC Paper Figures — Publication-Quality Academic Format
========================================================
Generates 10 standalone figures for the MTFC paper.

Figures:
  01 — Scenario comparison  (3-yr / 5-yr / 8-yr growth half-life)
  02 — Historical + forecast electricity with 95% CI
  03 — Cumulative datacenter CO₂ emissions
  04 — DC share milestone timeline
  05 — SARIMA model diagnostic panel  (ACF, PACF, Q-Q, residuals)
  06 — AI growth multiplier: historical fit + forecast
  07 — Carbon intensity change decomposition  (waterfall)
  08 — Seasonal electricity generation heatmap
  09 — Parameter sensitivity tornado chart
  10 — MTFC forecast vs. published industry projections

Style:
  - Serif fonts (Times New Roman / DejaVu Serif)
  - Colorblind-safe palette  (Wong 2011, Nature Methods)
  - 300 DPI, print-ready, APA / Nature formatting conventions
  - Spines on left + bottom only; subtle grid

Usage:  python generate_paper_figures.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.optimize import curve_fit
from scipy import stats as sp_stats
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════
BASE_DIR   = Path(__file__).parent.parent
PREP_DIR   = BASE_DIR / 'REAL FINAL FILES' / 'prepared_data'
FCAST_DIR  = BASE_DIR / 'REAL FINAL FILES' / 'model_forecasts'
OUTPUT_DIR = BASE_DIR / 'REAL FINAL FILES' / 'paper_figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# COLORBLIND-SAFE PALETTE  (Wong 2011, Nature Methods)
# ══════════════════════════════════════════════════════════════
C = {
    'blue':    '#0072B2',
    'orange':  '#E69F00',
    'green':   '#009E73',
    'red':     '#D55E00',
    'purple':  '#CC79A7',
    'sky':     '#56B4E9',
    'yellow':  '#F0E442',
    'black':   '#000000',
    'gray':    '#7F7F7F',
    'ltgray':  '#BFBFBF',
}

# ══════════════════════════════════════════════════════════════
# ACADEMIC STYLE
# ══════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif'],
    'font.size':          10,
    'axes.titlesize':     11,
    'axes.titleweight':   'bold',
    'axes.labelsize':     10,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    8.5,
    'legend.framealpha':  0.92,
    'legend.edgecolor':   '#cccccc',
    'legend.fancybox':    False,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.15,
    'savefig.facecolor':  'white',
    'axes.linewidth':     0.7,
    'axes.grid':          True,
    'grid.alpha':         0.22,
    'grid.linewidth':     0.4,
    'grid.linestyle':     '--',
    'lines.linewidth':    1.5,
    'mathtext.fontset':   'cm',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})

# ══════════════════════════════════════════════════════════════
# CONSTANTS  (must match pipeline)
# ══════════════════════════════════════════════════════════════
DATACENTER_BASELINE_SHARE = 0.25
GROWTH_HALFLIFE_YEARS     = 5.0
MIN_MONTHLY_GROWTH        = 0.003
FORECAST_PERIODS          = 160
EMISSION_FACTORS = {
    'coal_pct':      2.23,   # lb CO₂/kWh  (EPA eGRID 2022)
    'gas_pct':       0.91,
    'nuclear_pct':   0.01,
    'renewable_pct': 0.03,
}
KWH_PER_GWH = 1_000_000
LB_PER_TON  = 2_000


# ══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════
def _exp_growth(t, a, b):
    """Pure exponential: y = a · exp(b · t)."""
    return a * np.exp(b * t)


def _compute_multiplier(b_fit, halflife_yr, min_growth=MIN_MONTHLY_GROWTH,
                         periods=FORECAST_PERIODS):
    """Compute normalised AI multiplier trajectory for a given half-life."""
    decay = np.log(2) / (halflife_yr * 12)
    vals = [1.0]
    for i in range(periods):
        rate = min_growth + (b_fit - min_growth) * np.exp(-decay * i)
        vals.append(vals[-1] * np.exp(rate))
    arr = np.array(vals[1:])
    return arr / arr[0]


def _load_hist_energy():
    df = pd.read_csv(PREP_DIR / 'monthly_energy_consumption.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df


def _load_fcast_energy():
    df = pd.read_csv(FCAST_DIR / 'forecast_energy.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df


def _load_integrated():
    df = pd.read_csv(FCAST_DIR / 'forecast_integrated.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df


def _load_ai_proxy():
    df = pd.read_csv(PREP_DIR / 'monthly_ai_proxy.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df


def _load_ai_fcast():
    df = pd.read_csv(FCAST_DIR / 'forecast_ai_multiplier.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df


def _fit_exponential():
    """Fit y = a·exp(b·t) to historical AI proxy.  Returns (a, b, pcov, df)."""
    ai = _load_ai_proxy()
    t = np.arange(len(ai))
    y = ai['ai_proxy'].values
    popt, pcov = curve_fit(_exp_growth, t, y, p0=[1.0, 0.02], maxfev=10000)
    return popt[0], popt[1], pcov, ai


def _savefig(fig, name):
    fig.savefig(OUTPUT_DIR / name, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  \u2713 {name}')


# ══════════════════════════════════════════════════════════════
#  FIGURE 01 — Scenario Comparison Fan Chart
# ══════════════════════════════════════════════════════════════
def fig01_scenario_comparison():
    """DC share trajectories under 3 growth-rate decay assumptions."""
    _, b_fit, _, _ = _fit_exponential()
    fcst = _load_fcast_energy()
    dates = fcst['date']

    scenarios = [
        (3.0, 'Aggressive (3-yr half-life)', C['red'],    '--', 1.3),
        (5.0, 'Base case (5-yr half-life)',   C['blue'],   '-',  2.0),
        (8.0, 'Conservative (8-yr half-life)',C['green'],  '--', 1.3),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    trajectories = {}

    for hl, label, color, ls, lw in scenarios:
        mult = _compute_multiplier(b_fit, hl)
        dc_share = DATACENTER_BASELINE_SHARE * mult * 100
        trajectories[hl] = dc_share
        ax.plot(dates, dc_share, color=color, linestyle=ls,
                linewidth=lw, label=label)

    # Shading between aggressive and conservative
    ax.fill_between(dates, trajectories[3.0], trajectories[8.0],
                    alpha=0.08, color=C['gray'], label='Scenario range')

    # Reference thresholds
    for yval, txt in [(100, '100 % of grid'), (50, '50 % of grid')]:
        ax.axhline(yval, color=C['ltgray'], linestyle=':', linewidth=1.0)
        ax.text(dates.iloc[-1], yval + 2, txt, fontsize=7.5,
                ha='right', va='bottom', color=C['gray'])

    # End-point annotations
    for hl in [3.0, 5.0, 8.0]:
        end_val = trajectories[hl][-1]
        ax.text(dates.iloc[-1] + pd.Timedelta(days=30), end_val,
                f'{end_val:.0f} %', fontsize=8, va='center',
                color=scenarios[[s[0] for s in scenarios].index(hl)][2])

    ax.set_xlabel('Year')
    ax.set_ylabel('Datacenter Share of Grid Generation (%)')
    ax.set_title('Figure 1.  DC Grid Share Under Alternative Growth Scenarios')
    ax.legend(loc='upper left', framealpha=0.92)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_ylim(bottom=0)

    _savefig(fig, 'fig01_scenario_comparison.png')


# ══════════════════════════════════════════════════════════════
#  FIGURE 02 — Historical + Forecast Energy with 95 % CI
# ══════════════════════════════════════════════════════════════
def fig02_energy_forecast():
    """Combined 2015-2038 electricity generation with prediction interval."""
    hist = _load_hist_energy()
    fcst = _load_fcast_energy()

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(hist['date'], hist['electricity_gwh'],
            color=C['black'], linewidth=1.3, label='Historical (EIA)')

    ax.plot(fcst['date'], fcst['electricity_gwh'],
            color=C['blue'], linewidth=1.8,
            label=r'SARIMA(1,1,1)(1,1,1)$_{12}$ forecast')

    if 'electricity_gwh_lower' in fcst.columns:
        ax.fill_between(fcst['date'],
                        fcst['electricity_gwh_lower'],
                        fcst['electricity_gwh_upper'],
                        alpha=0.15, color=C['blue'],
                        label='95 % prediction interval')

    # Forecast boundary
    boundary = hist['date'].iloc[-1]
    ax.axvline(boundary, color=C['gray'], linestyle='--', linewidth=0.8, alpha=0.5)
    ax.text(boundary + pd.Timedelta(days=60),
            ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02,
            'Forecast \u2192', fontsize=8, color=C['gray'])

    ax.set_xlabel('Year')
    ax.set_ylabel('Electricity Generation (GWh / month)')
    ax.set_title('Figure 2.  Virginia Monthly Electricity Generation (2015\u20132038)')
    ax.legend(loc='upper left')
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    _savefig(fig, 'fig02_energy_forecast.png')


# ══════════════════════════════════════════════════════════════
#  FIGURE 03 — Cumulative Datacenter CO₂
# ══════════════════════════════════════════════════════════════
def fig03_cumulative_co2():
    """Running total of datacenter CO₂ emissions by source."""
    intg = _load_integrated()

    # Cumulative in million metric tons
    for src in ['coal', 'gas', 'nuclear', 'renewable']:
        intg[f'cum_{src}'] = intg[f'co2_{src}_tons'].cumsum() / 1e6
    intg['cum_total'] = intg['co2_total_tons'].cumsum() / 1e6

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Stacked area — order matters (coal bottom → renewable top)
    ax.fill_between(intg['date'], 0, intg['cum_coal'],
                    color='#8B4513', alpha=0.80, label='Coal')
    ax.fill_between(intg['date'], intg['cum_coal'],
                    intg['cum_coal'] + intg['cum_gas'],
                    color=C['orange'], alpha=0.80, label='Natural Gas')
    ax.fill_between(intg['date'],
                    intg['cum_coal'] + intg['cum_gas'],
                    intg['cum_coal'] + intg['cum_gas'] + intg['cum_nuclear'],
                    color=C['blue'], alpha=0.80, label='Nuclear')
    ax.fill_between(intg['date'],
                    intg['cum_coal'] + intg['cum_gas'] + intg['cum_nuclear'],
                    intg['cum_total'],
                    color=C['green'], alpha=0.80, label='Renewable')

    # Total outline
    ax.plot(intg['date'], intg['cum_total'],
            color=C['black'], linewidth=1.8, linestyle='-')

    # Annotate final value
    final = intg['cum_total'].iloc[-1]
    ax.annotate(f'{final:,.0f} Mt CO\u2082',
                xy=(intg['date'].iloc[-1], final),
                xytext=(-90, 8), textcoords='offset points',
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C['black'], lw=1.0))

    ax.set_xlabel('Year')
    ax.set_ylabel('Cumulative CO\u2082 Emissions (Million Metric Tons)')
    ax.set_title('Figure 3.  Cumulative Datacenter CO\u2082 Emissions (2025\u20132038)')
    ax.legend(loc='upper left')
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.set_ylim(bottom=0)

    _savefig(fig, 'fig03_cumulative_co2.png')


# ══════════════════════════════════════════════════════════════
#  FIGURE 04 — DC Share Milestone Timeline
# ══════════════════════════════════════════════════════════════
def fig04_threshold_timeline():
    """Horizontal bar chart showing when DC share crosses key thresholds."""
    intg = _load_integrated()
    start = pd.Timestamp(intg['date'].iloc[0])

    thresholds = [50, 75, 100]
    colors_bar = [C['orange'], C['red'], '#8B0000']
    results = []

    for thresh in thresholds:
        mask = intg['dc_share_pct'] >= thresh
        if mask.any():
            cross_date = pd.Timestamp(intg.loc[mask.idxmax(), 'date'])
            months = (cross_date.year - start.year) * 12 + cross_date.month - start.month
            results.append((thresh, cross_date, months))

    fig, ax = plt.subplots(figsize=(7, 3.0))

    y_pos = np.arange(len(results))
    for i, (thresh, date, months) in enumerate(results):
        bar = ax.barh(i, months, height=0.5, color=colors_bar[i],
                      edgecolor='white', linewidth=0.6, alpha=0.85)
        date_str = date.strftime('%b %Y')
        ax.text(months + 1.5, i, f'{date_str}  ({months} mo)',
                va='center', fontsize=9, fontweight='bold', color=colors_bar[i])

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'{r[0]} %' for r in results])
    ax.set_xlabel(f'Months from Forecast Start ({start.strftime("%b %Y")})')
    ax.set_title('Figure 4.  Time to Datacenter Grid-Share Milestones')
    ax.invert_yaxis()
    ax.set_xlim(right=ax.get_xlim()[1] * 1.30)

    _savefig(fig, 'fig04_threshold_timeline.png')


# ══════════════════════════════════════════════════════════════
#  FIGURE 05 — SARIMA Model Diagnostic Panel
# ══════════════════════════════════════════════════════════════
def fig05_model_diagnostics():
    """Four-panel residual diagnostics for the energy SARIMA model."""
    hist = _load_hist_energy()
    data = hist.set_index('date')[['electricity_gwh']]
    data.index.freq = 'MS'

    # Refit SARIMA
    model = SARIMAX(data['electricity_gwh'],
                    order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False, maxiter=200, method='lbfgs')
    resid = results.resid.dropna()

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6))

    # (a) Standardised residual time series
    ax = axes[0, 0]
    std_resid = resid / resid.std()
    ax.plot(std_resid.index, std_resid.values, color=C['blue'], linewidth=0.7)
    ax.axhline(0, color=C['gray'], linewidth=0.6)
    ax.axhline(2, color=C['red'], linewidth=0.5, linestyle=':', alpha=0.5)
    ax.axhline(-2, color=C['red'], linewidth=0.5, linestyle=':', alpha=0.5)
    ax.set_title('(a)  Standardised Residuals', fontsize=10)
    ax.set_ylabel('$\\sigma$')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # (b) ACF
    ax = axes[0, 1]
    plot_acf(resid, ax=ax, lags=36, alpha=0.05,
             color=C['blue'], vlines_kwargs={'colors': C['blue'], 'linewidths': 0.7})
    ax.set_title('(b)  Autocorrelation (ACF)', fontsize=10)

    # (c) PACF
    ax = axes[1, 0]
    plot_pacf(resid, ax=ax, lags=36, alpha=0.05, method='ywm',
              color=C['blue'], vlines_kwargs={'colors': C['blue'], 'linewidths': 0.7})
    ax.set_title('(c)  Partial Autocorrelation (PACF)', fontsize=10)

    # (d) Normal Q-Q
    ax = axes[1, 1]
    (osm, osr), (slope, intercept, _) = sp_stats.probplot(resid, dist='norm')
    ax.scatter(osm, osr, s=12, color=C['blue'], alpha=0.6, edgecolors='none')
    qq_line_x = np.array([osm.min(), osm.max()])
    ax.plot(qq_line_x, slope * qq_line_x + intercept,
            color=C['red'], linewidth=1.2, linestyle='--')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    ax.set_title('(d)  Normal Q\u2013Q Plot', fontsize=10)

    # Ljung-Box statistic
    lb = acorr_ljungbox(resid, lags=[12], return_df=True)
    lb_q = lb['lb_stat'].values[0]
    lb_p = lb['lb_pvalue'].values[0]
    fig.text(0.50, -0.01,
             f'Ljung\u2013Box Q(12) = {lb_q:.2f},  p = {lb_p:.4f}    |    '
             f'AIC = {results.aic:.1f}    |    BIC = {results.bic:.1f}',
             ha='center', fontsize=8.5, style='italic')

    fig.suptitle(r'Figure 5.  SARIMA(1,1,1)(1,1,1)$_{12}$ Diagnostic Panel',
                 fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()
    _savefig(fig, 'fig05_model_diagnostics.png')


# ══════════════════════════════════════════════════════════════
#  FIGURE 06 — AI Multiplier: Historical Fit + Forecast
# ══════════════════════════════════════════════════════════════
def fig06_ai_multiplier():
    """Historical DC spending proxy, exponential fit, and forecast with CI."""
    a_fit, b_fit, pcov, ai_hist = _fit_exponential()
    ai_fcst = _load_ai_fcast()

    # Fitted curve over historical
    t_hist = np.arange(len(ai_hist))
    y_fitted = _exp_growth(t_hist, a_fit, b_fit)
    y_actual = ai_hist['ai_proxy'].values

    ss_res = np.sum((y_actual - y_fitted) ** 2)
    ss_tot = np.sum((y_actual - y_actual.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    # Scale forecast to historical axis (forecast starts at 1.0 = last_fitted)
    scale = y_fitted[-1]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.scatter(ai_hist['date'], ai_hist['ai_proxy'],
               s=10, color=C['black'], alpha=0.45, zorder=3,
               label='Observed (VA DC Construction Spending, est.)')

    ax.plot(ai_hist['date'], y_fitted,
            color=C['red'], linewidth=1.4, linestyle='--',
            label=f'Exponential fit  ($R^2 = {r2:.3f}$)')

    ax.plot(ai_fcst['date'], ai_fcst['ai_multiplier'] * scale,
            color=C['blue'], linewidth=1.8,
            label='Forecast (decaying growth rate)')

    if 'ai_multiplier_lower' in ai_fcst.columns:
        ax.fill_between(ai_fcst['date'],
                        ai_fcst['ai_multiplier_lower'] * scale,
                        ai_fcst['ai_multiplier_upper'] * scale,
                        alpha=0.13, color=C['blue'], label='95 % CI')

    ax.axvline(ai_hist['date'].iloc[-1], color=C['gray'],
               linestyle='--', linewidth=0.7, alpha=0.4)

    ax.set_xlabel('Year')
    ax.set_ylabel('AI Proxy Index (2015-01 = 1.0)')
    ax.set_title('Figure 6.  AI Growth Proxy (Virginia est.) \u2014 Exponential Fit and Forecast')
    ax.legend(loc='upper left', fontsize=8)
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_ylim(bottom=0)

    _savefig(fig, 'fig06_ai_multiplier.png')


# ══════════════════════════════════════════════════════════════
#  FIGURE 07 — Carbon Intensity Decomposition (Waterfall)
# ══════════════════════════════════════════════════════════════
def fig07_carbon_decomposition():
    """Waterfall showing each source's contribution to CI change 2026 → 2038."""
    intg = _load_integrated()

    y2026 = intg[intg['date'].dt.year == 2026].iloc[-1]
    y2038 = intg[intg['date'].dt.year == 2038].iloc[-1]

    ci_start = y2026['carbon_intensity']

    # Contribution of each source: Δpct / 100 × emission_factor
    src_order = ['coal_pct', 'gas_pct', 'nuclear_pct', 'renewable_pct']
    src_labels = ['Coal\nRetirement', 'Natural Gas\nShift',
                  'Nuclear\nChange', 'Renewable\nGrowth']
    deltas = []
    for src in src_order:
        d = (y2038[src] - y2026[src]) / 100.0 * EMISSION_FACTORS[src]
        deltas.append(d)

    # Running total
    running = [ci_start]
    for d in deltas:
        running.append(running[-1] + d)

    categories = ['2026\nBaseline'] + src_labels + ['2038\nForecast']
    n = len(categories)
    x = np.arange(n)
    w = 0.55

    # Bar parameters
    bottoms = []
    heights = []
    colors_list = []

    # First bar (total)
    bottoms.append(0)
    heights.append(ci_start)
    colors_list.append(C['blue'])

    # Change bars
    for i, d in enumerate(deltas):
        if d >= 0:
            bottoms.append(running[i])
            heights.append(d)
            colors_list.append(C['red'])
        else:
            bottoms.append(running[i + 1])
            heights.append(-d)
            colors_list.append(C['green'])

    # Last bar (total)
    bottoms.append(0)
    heights.append(running[-1])
    colors_list.append(C['blue'])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(x, heights, bottom=bottoms, width=w,
                  color=colors_list, edgecolor='white', linewidth=0.6)

    # Connectors
    for i in range(n - 1):
        y_conn = running[min(i, len(running) - 1)]
        ax.plot([x[i] + w / 2, x[i + 1] - w / 2],
                [y_conn, y_conn],
                color=C['gray'], linewidth=0.5, linestyle='-')

    # Value labels
    for i in range(n):
        top = bottoms[i] + heights[i]
        if i == 0 or i == n - 1:
            ax.text(x[i], top + 0.004, f'{heights[i]:.4f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        else:
            sign = '+' if deltas[i - 1] >= 0 else ''
            ax.text(x[i], top + 0.004, f'{sign}{deltas[i - 1]:.4f}',
                    ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8.5)
    ax.set_ylabel('Carbon Intensity (lb CO\u2082 / kWh)')
    ax.set_title('Figure 7.  Carbon Intensity Change Decomposition (2026 \u2192 2038)')

    # Zoom y-axis to show detail
    y_lo = min(bottoms) - 0.02
    y_hi = max(b + h for b, h in zip(bottoms, heights)) + 0.03
    ax.set_ylim(max(0, y_lo), y_hi)

    # Legend (manual)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=C['green'], label='Decrease (reduces CI)'),
        Patch(facecolor=C['red'],   label='Increase (raises CI)'),
        Patch(facecolor=C['blue'],  label='Total'),
    ], loc='upper right', fontsize=8)

    _savefig(fig, 'fig07_carbon_decomposition.png')


# ══════════════════════════════════════════════════════════════
#  FIGURE 08 — Seasonal Electricity Generation Heatmap
# ══════════════════════════════════════════════════════════════
def fig08_seasonal_heatmap():
    """Month × Year heatmap of Virginia electricity generation."""
    hist = _load_hist_energy()
    hist['year']  = hist['date'].dt.year
    hist['month'] = hist['date'].dt.month

    pivot = hist.pivot(index='year', columns='month', values='electricity_gwh')

    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig, ax = plt.subplots(figsize=(7, 4.0))

    masked = np.ma.masked_invalid(pivot.values)
    im = ax.imshow(masked, cmap='YlOrRd', aspect='auto',
                   interpolation='nearest')

    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels, fontsize=8.5)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(int), fontsize=8.5)
    ax.set_xlabel('Month')
    ax.set_ylabel('Year')
    ax.set_title('Figure 8.  Monthly Electricity Generation Patterns (GWh)')

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cbar.set_label('GWh / month', fontsize=9)

    # Annotate each cell with value
    for i in range(masked.shape[0]):
        for j in range(masked.shape[1]):
            val = masked[i, j]
            if val is not np.ma.masked and not np.isnan(val):
                text_color = 'white' if val > masked.max() * 0.65 else 'black'
                ax.text(j, i, f'{val:,.0f}', ha='center', va='center',
                        fontsize=6, color=text_color)

    fig.tight_layout()
    _savefig(fig, 'fig08_seasonal_heatmap.png')


# ══════════════════════════════════════════════════════════════
#  FIGURE 09 — Parameter Sensitivity Tornado
# ══════════════════════════════════════════════════════════════
def fig09_sensitivity_tornado():
    """One-at-a-time sensitivity of cumulative CO₂ to key parameters."""
    intg = _load_integrated()
    _, b_fit, _, _ = _fit_exponential()

    base_cum = intg['co2_total_tons'].sum() / 1e6  # Mt

    # ─── Sensitivity definitions ───
    scenarios = []  # (label, low_cum, high_cum, low_val_str, high_val_str)

    # 1. DC Baseline Share  (±20 %)
    lo = base_cum * 0.20 / 0.25
    hi = base_cum * 0.30 / 0.25
    scenarios.append(('DC Baseline Share\n(0.20 vs 0.30)', lo, hi, '0.20', '0.30'))

    # 2. Growth Half-life  (3 yr vs 8 yr)
    for hl, store_idx in [(3.0, 'lo'), (8.0, 'hi')]:
        mult = _compute_multiplier(b_fit, hl)
        new_dc = intg['electricity_gwh'].values * DATACENTER_BASELINE_SHARE * mult
        co2_t = new_dc * intg['carbon_intensity'].values * KWH_PER_GWH / LB_PER_TON
        val = co2_t.sum() / 1e6
        if store_idx == 'lo':
            hl_vals = [val]
        else:
            hl_vals.append(val)
    # 3yr gives MORE growth → higher CO2; 8yr gives less
    scenarios.append(('Growth Half-life\n(8 yr vs 3 yr)',
                      min(hl_vals), max(hl_vals), '8 yr', '3 yr'))

    # 3. Min Monthly Growth  (±50 %)
    mmg_vals = []
    for mmg in [0.0015, 0.0045]:
        mult = _compute_multiplier(b_fit, GROWTH_HALFLIFE_YEARS, min_growth=mmg)
        new_dc = intg['electricity_gwh'].values * DATACENTER_BASELINE_SHARE * mult
        co2_t = new_dc * intg['carbon_intensity'].values * KWH_PER_GWH / LB_PER_TON
        mmg_vals.append(co2_t.sum() / 1e6)
    scenarios.append(('Growth-Rate Floor\n(1.8 % vs 5.5 % yr)',
                      min(mmg_vals), max(mmg_vals), '1.8 %', '5.5 %'))

    # 4. Gas Emission Factor  (±20 %)
    gef_vals = []
    for gef in [0.73, 1.09]:
        new_ci = (intg['coal_pct'] / 100 * 2.23 +
                  intg['gas_pct'] / 100 * gef +
                  intg['nuclear_pct'] / 100 * 0.01 +
                  intg['renewable_pct'] / 100 * 0.03)
        new_ci = np.maximum(new_ci, 0.03)
        co2_t = intg['dc_energy_gwh'] * new_ci * KWH_PER_GWH / LB_PER_TON
        gef_vals.append(co2_t.sum() / 1e6)
    scenarios.append(('Gas Emission Factor\n(0.73 vs 1.09 lb/kWh)',
                      min(gef_vals), max(gef_vals), '0.73', '1.09'))

    # Sort by total span (largest impact at top)
    scenarios.sort(key=lambda s: abs(s[2] - s[1]), reverse=True)

    fig, ax = plt.subplots(figsize=(7, 4.0))

    y_pos = np.arange(len(scenarios))
    for i, (label, lo, hi, lo_str, hi_str) in enumerate(scenarios):
        # Low bar (extends left from base)
        ax.barh(i, lo - base_cum, left=base_cum, height=0.45,
                color=C['sky'], edgecolor='white', linewidth=0.5)
        # High bar (extends right from base)
        ax.barh(i, hi - base_cum, left=base_cum, height=0.45,
                color=C['red'], alpha=0.65, edgecolor='white', linewidth=0.5)
        # Annotations
        ax.text(lo - 3, i, f'{lo:.0f}', ha='right', va='center',
                fontsize=7.5, color=C['blue'])
        ax.text(hi + 3, i, f'{hi:.0f}', ha='left', va='center',
                fontsize=7.5, color=C['red'])

    ax.axvline(base_cum, color=C['black'], linewidth=1.5, zorder=5)
    ax.text(base_cum, len(scenarios) - 0.1,
            f'Base: {base_cum:.0f} Mt', ha='center', va='bottom',
            fontsize=8.5, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([s[0] for s in scenarios], fontsize=8.5)
    ax.set_xlabel('Cumulative CO\u2082 Emissions 2025\u20132038 (Million Metric Tons)')
    ax.set_title('Figure 9.  Parameter Sensitivity \u2014 Cumulative CO\u2082')
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=C['sky'],  label='Lower bound'),
        Patch(facecolor=C['red'], alpha=0.65, label='Upper bound'),
    ], loc='lower right', fontsize=8)

    _savefig(fig, 'fig09_sensitivity_tornado.png')


# ══════════════════════════════════════════════════════════════
#  FIGURE 10 — MTFC Forecast vs. Industry Projections
# ══════════════════════════════════════════════════════════════
def fig10_industry_comparison():
    """Our AI multiplier forecast against published industry estimates."""
    ai_fcst = _load_ai_fcast()

    fig, ax = plt.subplots(figsize=(7, 5))

    # ── Our forecast ──
    ax.plot(ai_fcst['date'], ai_fcst['ai_multiplier'],
            color=C['blue'], linewidth=2.2, label='MTFC Base Case', zorder=4)

    if 'ai_multiplier_lower' in ai_fcst.columns:
        ax.fill_between(ai_fcst['date'],
                        ai_fcst['ai_multiplier_lower'],
                        ai_fcst['ai_multiplier_upper'],
                        alpha=0.12, color=C['blue'], label='95 % CI')

    # ── Industry projections ──
    # Each: (label, date, multiplier, color, marker)
    # Multipliers are approximate DC-demand growth factors from each source's
    # respective baseline; direct comparison requires caution (see note).
    refs = [
        ('Dominion IRP\n(Virginia, to 2038)', pd.Timestamp('2038-06-01'), 1.85,
         C['green'], 's'),
        ('IEA WEO High\n(Global, to 2030)',   pd.Timestamp('2030-06-01'), 2.10,
         C['orange'], '^'),
        ('Goldman Sachs\n(U.S., to 2030)',     pd.Timestamp('2030-01-01'), 2.60,
         C['red'], 'D'),
        ('McKinsey\n(U.S., to 2030)',          pd.Timestamp('2029-06-01'), 2.00,
         C['purple'], 'o'),
    ]

    for label, date, mult, color, marker in refs:
        ax.scatter(date, mult, color=color, marker=marker, s=90, zorder=5,
                   edgecolors='white', linewidth=0.8, label=label)

    ax.set_xlabel('Year')
    ax.set_ylabel('AI Demand Multiplier  (Forecast Start = 1.0\u00d7)')
    ax.set_title('Figure 10.  MTFC Forecast vs. Published Industry Projections')
    ax.legend(loc='upper left', fontsize=7.5, ncol=1, framealpha=0.92)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_ylim(bottom=0)

    # Source note
    fig.text(0.50, -0.03,
             'Note: Industry projections have varying baselines and scopes '
             '(Virginia / U.S. / Global).  Direct comparison is approximate.',
             ha='center', fontsize=7, style='italic', color=C['gray'])

    _savefig(fig, 'fig10_industry_comparison.png')


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def generate_all():
    """Generate all 10 publication-quality figures."""
    import time
    t0 = time.time()
    print('Generating MTFC paper figures ...')
    print(f'  Output: {OUTPUT_DIR}\n')

    fig01_scenario_comparison()
    fig02_energy_forecast()
    fig03_cumulative_co2()
    fig04_threshold_timeline()
    fig05_model_diagnostics()
    fig06_ai_multiplier()
    fig07_carbon_decomposition()
    fig08_seasonal_heatmap()
    fig09_sensitivity_tornado()
    fig10_industry_comparison()

    elapsed = time.time() - t0
    print(f'\n\u2713 All 10 figures generated in {elapsed:.1f} s')
    print(f'  {OUTPUT_DIR}')


if __name__ == '__main__':
    generate_all()
