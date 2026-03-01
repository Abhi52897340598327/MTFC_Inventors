#!/usr/bin/env python3
"""
MTFC Correlation Matrices — Publication-Quality Heatmaps
========================================================
Generates correlation matrices for every model in the MTFC pipeline.

Outputs (saved to REAL FINAL FILES/paper_figures/):
  corr_model1_sarima_residual_lags.png   — Residual lag-correlation for energy SARIMA
  corr_model2_grid_mix_cross_fuel.png    — Historical vs forecast cross-fuel correlations
  corr_model3_ai_growth_features.png     — AI spending proxy feature correlations
  corr_model4_integrated_outputs.png     — Full 14-variable integration output matrix
  corr_cross_model_residuals.png         — Cross-model SARIMA residual independence test

Usage:  python visualizations/plot_correlation_matrices.py
"""

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — safe on headless / CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
#  PATHS  —  resolved relative to this file so it works from
#            any working directory.
# ══════════════════════════════════════════════════════════════
_HERE      = Path(__file__).resolve().parent
BASE_DIR   = _HERE.parent                                          # MTFC_Inventors/
PREP_DIR   = BASE_DIR / 'REAL FINAL FILES' / 'prepared_data'
FCAST_DIR  = BASE_DIR / 'REAL FINAL FILES' / 'model_forecasts'
OUTPUT_DIR = BASE_DIR / 'REAL FINAL FILES' / 'paper_figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Source data for Model 3 (raw spending CSV)
SRC_DIR = BASE_DIR / 'REAL_FINAL_MODEL_SUBMISSION_MTFC_INVENTORS' / 'REAL FINAL DATA SOURCES'

# ══════════════════════════════════════════════════════════════
#  STYLE
# ══════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif'],
    'font.size':        10,
    'axes.titlesize':   12,
    'axes.titleweight': 'bold',
    'axes.labelsize':   10,
    'figure.dpi':       300,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'savefig.facecolor': 'white',
})
CMAP = 'RdBu_r'


# ══════════════════════════════════════════════════════════════
#  HELPER — annotated heatmap with dynamic text colour
# ══════════════════════════════════════════════════════════════
def _heatmap(corr: pd.DataFrame, ax, *, fmt: str = '.2f', annot_size: int = 9):
    """Draw a correlation heatmap with numeric annotations."""
    sns.heatmap(
        corr, annot=True, fmt=fmt, cmap=CMAP,
        vmin=-1, vmax=1, square=True, linewidths=0.5,
        ax=ax, cbar_kws={'label': 'Pearson r', 'shrink': 0.8},
        annot_kws={'size': annot_size},
    )


def _savefig(fig, name: str):
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  ✓ {name}')


# ══════════════════════════════════════════════════════════════
#  SAFE LOADERS  —  fail fast with clear messages
# ══════════════════════════════════════════════════════════════
def _load_csv(directory: Path, filename: str,
              required_cols: list[str] | None = None) -> pd.DataFrame:
    """Load a CSV with validation.  Raises FileNotFoundError / KeyError early."""
    fp = directory / filename
    if not fp.exists():
        raise FileNotFoundError(f'Required file missing: {fp}')
    df = pd.read_csv(fp)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    if required_cols:
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise KeyError(f'{filename} is missing columns: {missing}')
    return df


# ══════════════════════════════════════════════════════════════
#  MODEL 1 — SARIMA Residual Lag-Correlation Matrix
# ══════════════════════════════════════════════════════════════
def fig_model1_correlation():
    """
    Fit SARIMA(1,1,1)(1,1,1)₁₂ on historical energy, extract residuals,
    then construct a lag-correlation matrix (lags 0–12).

    Purpose: Verify that residuals are approximately white noise
    (off-diagonal values should be near zero if the model is adequate).
    """
    print('[Model 1] Residual lag-correlation matrix …')

    energy = _load_csv(PREP_DIR, 'monthly_energy_consumption.csv',
                       required_cols=['date', 'electricity_gwh'])
    data = energy.set_index('date')[['electricity_gwh']]
    data.index.freq = 'MS'

    # Fit SARIMA
    model = SARIMAX(data['electricity_gwh'],
                    order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False, maxiter=200, method='lbfgs')
    resid = results.resid.dropna().values

    # Build lag matrix — each column is residuals shifted by k steps
    MAX_LAG = 12
    n = len(resid)
    lag_dict = {}
    for k in range(MAX_LAG + 1):
        # Align all lag vectors to the same length (n - MAX_LAG)
        lag_dict[f'Lag {k}'] = resid[MAX_LAG - k : n - k] if k > 0 \
                               else resid[MAX_LAG:]
    lag_df = pd.DataFrame(lag_dict)

    # Sanity: all columns same length
    assert lag_df.shape[0] == n - MAX_LAG, 'Lag alignment error'

    corr = lag_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    _heatmap(corr, ax, annot_size=8)
    ax.set_title('Model 1 — SARIMA(1,1,1)(1,1,1)$_{12}$ Residual\n'
                 'Lag-Correlation Matrix (Lags 0–12)')
    fig.tight_layout()
    _savefig(fig, 'corr_model1_sarima_residual_lags.png')


# ══════════════════════════════════════════════════════════════
#  MODEL 2 — Cross-Fuel Correlation (Historical + Forecast)
# ══════════════════════════════════════════════════════════════
def fig_model2_correlation():
    """
    Side-by-side heatmaps: historical (2015-2025) and forecast (2025-2038)
    cross-fuel correlations for coal, gas, nuclear, renewable.

    Purpose: Show how inter-fuel relationships evolve from history
    into the forecast (e.g., coal-gas substitution, renewable independence).
    """
    print('[Model 2] Cross-fuel correlation matrices …')

    FUELS = ['coal_pct', 'gas_pct', 'nuclear_pct', 'renewable_pct']
    LABELS = {'coal_pct': 'Coal %', 'gas_pct': 'Gas %',
              'nuclear_pct': 'Nuclear %', 'renewable_pct': 'Renewable %'}

    # ── Historical grid mix ──────────────────────────────────
    hist = _load_csv(PREP_DIR, 'monthly_grid_mix.csv', required_cols=FUELS)

    # ── Forecast grid mix  (merge 4 individual files) ────────
    fc_frames = []
    for fuel in FUELS:
        source_name = fuel.replace('_pct', '')               # e.g. 'coal'
        fname = f'forecast_grid_{source_name}.csv'
        df = _load_csv(FCAST_DIR, fname, required_cols=['date', fuel])
        fc_frames.append(df[['date', fuel]])

    fc = fc_frames[0]
    for frame in fc_frames[1:]:
        fc = fc.merge(frame, on='date', how='outer')
    fc = fc.sort_values('date').reset_index(drop=True)

    # ── Plot ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    corr_hist = hist[FUELS].rename(columns=LABELS).corr()
    _heatmap(corr_hist, axes[0])
    axes[0].set_title('Historical Grid Mix\n(2015–2025)')

    corr_fc = fc[FUELS].rename(columns=LABELS).corr()
    _heatmap(corr_fc, axes[1])
    axes[1].set_title('Forecast Grid Mix\n(2025–2038)')

    fig.suptitle('Model 2 — SARIMA / SARIMAX Grid Mix:\n'
                 'Cross-Fuel Pearson Correlation',
                 fontsize=13, fontweight='bold', y=1.04)
    fig.tight_layout()
    _savefig(fig, 'corr_model2_grid_mix_cross_fuel.png')


# ══════════════════════════════════════════════════════════════
#  MODEL 3 — AI Growth Proxy Feature Correlations
# ══════════════════════════════════════════════════════════════
def fig_model3_correlation():
    """
    Feature correlation for the exponential-fit AI growth model:
    Time Index, VA DC Spending ($M), Log Spending, Exponential Fit,
    and Fit Residual.

    Purpose: Validate that the exponential model captures the trend
    (high correlation with log-spending, near-zero residual–trend r).
    """
    print('[Model 3] AI growth proxy feature correlations …')

    # ── Load the prepared (normalised) AI proxy ──────────────
    ai = _load_csv(PREP_DIR, 'monthly_ai_proxy.csv',
                   required_cols=['date', 'ai_proxy'])

    t = np.arange(len(ai))
    y = ai['ai_proxy'].values

    # Exponential fit  y = a · exp(b · t)
    def _exp(t, a, b):
        return a * np.exp(b * t)

    popt, _ = curve_fit(_exp, t, y, p0=[1.0, 0.02], maxfev=10_000)
    y_fit = _exp(t, *popt)

    features = pd.DataFrame({
        'Time Index':     t,
        'AI Proxy':       y,
        'Log(AI Proxy)':  np.log(np.maximum(y, 1e-12)),   # guard against log(0)
        'Exp. Fit':       y_fit,
        'Residual':       y - y_fit,
    })

    corr = features.corr()

    fig, ax = plt.subplots(figsize=(8, 6.5))
    _heatmap(corr, ax)
    ax.set_title('Model 3 — AI Growth Proxy:\n'
                 'Feature Correlation Matrix (Virginia est.)')
    fig.tight_layout()
    _savefig(fig, 'corr_model3_ai_growth_features.png')


# ══════════════════════════════════════════════════════════════
#  MODEL 4 — Full Integrated-Output Correlation Matrix
# ══════════════════════════════════════════════════════════════
def fig_model4_correlation():
    """
    Pearson r for every pair of the 15 numeric columns in
    forecast_integrated.csv.

    Purpose: Reveal structural relationships (e.g., DC energy ↔ CO₂,
    coal ↔ carbon intensity, AI multiplier ↔ dc_share).
    """
    print('[Model 4] Integrated output correlation matrix …')

    df = _load_csv(FCAST_DIR, 'forecast_integrated.csv')

    RENAME = {
        'electricity_gwh':       'Electricity\n(GWh)',
        'dc_energy_baseline_gwh':'DC Baseline\n(GWh)',
        'dc_energy_gwh':         'DC Energy\n(GWh)',
        'ai_multiplier':         'AI\nMultiplier',
        'coal_pct':              'Coal %',
        'gas_pct':               'Gas %',
        'nuclear_pct':           'Nuclear %',
        'renewable_pct':         'Renewable %',
        'carbon_intensity':      'Carbon\nIntensity',
        'co2_coal_tons':         'CO₂ Coal\n(tons)',
        'co2_gas_tons':          'CO₂ Gas\n(tons)',
        'co2_nuclear_tons':      'CO₂ Nuclear\n(tons)',
        'co2_renewable_tons':    'CO₂ Renewable\n(tons)',
        'co2_total_tons':        'CO₂ Total\n(tons)',
        'dc_share_pct':          'DC Share\n(%)',
    }

    # Keep only columns that actually exist (future-proof)
    avail = {k: v for k, v in RENAME.items() if k in df.columns}
    plot_df = df[list(avail.keys())].rename(columns=avail)

    corr = plot_df.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    _heatmap(corr, ax, annot_size=7)
    ax.set_title('Model 4 — Integrated Forecast:\n'
                 'Full Output Correlation Matrix (2025–2038)',
                 fontsize=13)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    fig.tight_layout()
    _savefig(fig, 'corr_model4_integrated_outputs.png')


# ══════════════════════════════════════════════════════════════
#  BONUS — Cross-Model Residual Independence Test
# ══════════════════════════════════════════════════════════════
def fig_cross_model_residuals():
    """
    Fit separate SARIMAs for energy + each fuel source, collect their
    residuals, and compute the cross-model residual correlation matrix.

    Purpose: Verify that the models' unexplained components are
    approximately independent (low off-diagonal r).
    """
    print('[Cross-Model] Residual independence matrix …')

    # ── Model 1: energy residuals ────────────────────────────
    energy = _load_csv(PREP_DIR, 'monthly_energy_consumption.csv',
                       required_cols=['date', 'electricity_gwh'])
    e_data = energy.set_index('date')[['electricity_gwh']]
    e_data.index.freq = 'MS'

    m1 = SARIMAX(e_data['electricity_gwh'],
                 order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                 enforce_stationarity=False, enforce_invertibility=False)
    r1 = m1.fit(disp=False, maxiter=200, method='lbfgs').resid.dropna()

    # ── Model 2: grid-mix residuals per fuel ─────────────────
    grid = _load_csv(PREP_DIR, 'monthly_grid_mix.csv',
                     required_cols=['date', 'coal_pct', 'gas_pct',
                                    'nuclear_pct', 'renewable_pct'])
    grid = grid.set_index('date')
    grid.index.freq = 'MS'

    FUEL_LABELS = {'coal_pct': 'Coal', 'gas_pct': 'Gas',
                   'nuclear_pct': 'Nuclear', 'renewable_pct': 'Renewable'}

    # For renewable we use SARIMAX with ai_proxy exogenous (matching Model 2 actual spec)
    ai = _load_csv(PREP_DIR, 'monthly_ai_proxy.csv',
                   required_cols=['date', 'ai_proxy'])
    ai = ai.set_index('date')
    ai.index.freq = 'MS'

    resid_dict: dict[str, pd.Series] = {'Energy': r1}

    for fuel, label in FUEL_LABELS.items():
        y = grid[fuel].dropna()
        try:
            if fuel == 'renewable_pct':
                # SARIMAX matching Model 2 specification
                exog = ai.reindex(y.index)[['ai_proxy']]
                m = SARIMAX(y, exog=exog,
                            order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            else:
                m = SARIMAX(y,
                            order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            resid_dict[label] = m.fit(disp=False, maxiter=200,
                                      method='lbfgs').resid.dropna()
        except Exception as exc:
            print(f'  ⚠ {fuel} fit failed ({exc}), skipping.')

    # Align all residual series to the same date range
    resid_df = pd.DataFrame(resid_dict)
    resid_df = resid_df.dropna()

    if resid_df.shape[1] < 2:
        print('  ⚠ Too few successful fits — skipping cross-model matrix.')
        return

    corr = resid_df.corr()

    fig, ax = plt.subplots(figsize=(8, 6.5))
    _heatmap(corr, ax)
    ax.set_title('Cross-Model SARIMA Residual Correlation Matrix\n'
                 '(Energy × Grid-Mix Fuels, 2015–2025)')
    fig.tight_layout()
    _savefig(fig, 'corr_cross_model_residuals.png')


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def generate_all():
    import time
    t0 = time.time()

    print('=' * 60)
    print('  Generating MTFC Correlation Matrices')
    print(f'  Output: {OUTPUT_DIR}')
    print('=' * 60 + '\n')

    fig_model1_correlation()
    fig_model2_correlation()
    fig_model3_correlation()
    fig_model4_correlation()
    fig_cross_model_residuals()

    elapsed = time.time() - t0
    print(f'\n✓ All 5 correlation matrices generated in {elapsed:.1f} s')
    print(f'  {OUTPUT_DIR}')


if __name__ == '__main__':
    generate_all()
