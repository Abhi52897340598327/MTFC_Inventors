"""
Copula Tail-Dependency Analysis
================================
Fits bivariate copulas (Gumbel / Clayton) to key variable pairs,
computes parametric upper- and lower-tail dependence coefficients,
and generates tail-dependence curves.

Outputs
-------
- copula_tail_dependence.csv
- copula_tail_curves.csv
- figures/copula_tail_dependence.png
- figures/copula_scatter_matrix.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from config import MC, OUTPUT_DIR, FIGURE_DIR, PLOT, DC_PARAMS, GRID

np.random.seed(MC["seed"])


# ── Pair definitions ─────────────────────────────────────────────────────
PAIRS = [
    ("temperature",    "energy_demand"),
    ("temperature",    "carbon_intensity"),
    ("carbon_intensity", "emissions"),
    ("cpu_utilization", "energy_demand"),
    ("pue",            "total_cost"),
]


# ── Generate correlated draws from the physical model ────────────────────
def _build_dataset(n: int = 20000) -> pd.DataFrame:
    temp_f = np.random.normal(60, 15, n)
    temp_f = np.clip(temp_f, 10, 110)
    cpu = np.random.normal(DC_PARAMS["cpu_utilization_mean"],
                           DC_PARAMS["cpu_utilization_std"], n)
    cpu = np.clip(cpu, 5, 100)
    delta_t = np.maximum(temp_f - 65, 0)
    pue = (DC_PARAMS["pue_baseline"]
           + DC_PARAMS["pue_temp_coef"] * delta_t
           + DC_PARAMS["pue_cpu_coef"] * cpu)
    idle = DC_PARAMS["idle_power_fraction"]
    it_mw = DC_PARAMS["it_capacity_mw"] * (idle + (1 - idle) * cpu / 100)
    total_mw = it_mw * pue
    energy_mwh = total_mw * 8760
    ci = np.random.normal(GRID["carbon_intensity_mean"],
                          GRID["carbon_intensity_std"], n)
    ci = np.clip(ci, GRID["carbon_intensity_min"], GRID["carbon_intensity_max"])
    emissions = total_mw * ci * 8760 / 1000
    total_cost = energy_mwh * 72 + emissions * 190
    return pd.DataFrame({
        "temperature": temp_f,
        "cpu_utilization": cpu,
        "pue": pue,
        "energy_demand": energy_mwh,
        "carbon_intensity": ci,
        "emissions": emissions,
        "total_cost": total_cost,
    })


# ── Tail-dependence computation ──────────────────────────────────────────
def _empirical_tail_dependence(u, v, thresholds):
    """Compute empirical upper and lower tail dependence at each threshold."""
    upper, lower = [], []
    for t in thresholds:
        # Upper tail:  P(V > t | U > t)
        mask_u = u > t
        if mask_u.sum() > 10:
            upper.append((v[mask_u] > t).mean())
        else:
            upper.append(np.nan)
        # Lower tail:  P(V < 1-t | U < 1-t)
        lt = 1 - t
        mask_l = u < lt
        if mask_l.sum() > 10:
            lower.append((v[mask_l] < lt).mean())
        else:
            lower.append(np.nan)
    return np.array(upper), np.array(lower)


def _rank_transform(x):
    """Convert to pseudo-observations on (0, 1)."""
    n = len(x)
    return stats.rankdata(x) / (n + 1)


def compute_copula_metrics(df: pd.DataFrame):
    thresholds = np.linspace(0.80, 0.995, 50)

    dep_rows = []
    curve_rows = []

    for (var_a, var_b) in PAIRS:
        a, b = df[var_a].values, df[var_b].values
        u, v = _rank_transform(a), _rank_transform(b)

        upper, lower = _empirical_tail_dependence(u, v, thresholds)

        # Kendall tau → parametric copula fit
        tau, _ = stats.kendalltau(a, b)
        tau = np.clip(tau, -0.99, 0.99)

        # Gumbel parameter  (only valid for tau >= 0)
        if tau > 0:
            theta_gumbel = 1 / (1 - tau)
            upper_param = 2 - 2 ** (1 / theta_gumbel)
        else:
            theta_gumbel = 1.0
            upper_param = 0.0

        # Clayton parameter  (only valid for tau > 0)
        if tau > 0:
            theta_clayton = 2 * tau / (1 - tau)
            lower_param = 2 ** (-1 / theta_clayton)
        else:
            theta_clayton = 0.0
            lower_param = 0.0

        pair_label = f"{var_a}_vs_{var_b}"
        dep_rows.append({
            "pair": pair_label,
            "kendall_tau": round(tau, 4),
            "gumbel_theta": round(theta_gumbel, 4),
            "clayton_theta": round(theta_clayton, 4),
            "upper_tail": round(upper_param, 4),
            "lower_tail": round(lower_param, 4),
            "spearman_rho": round(stats.spearmanr(a, b).correlation, 4),
        })

        for t, up, lo in zip(thresholds, upper, lower):
            curve_rows.append({
                "pair": pair_label,
                "threshold": round(t, 4),
                "upper_tail_empirical": round(up, 4) if not np.isnan(up) else None,
                "lower_tail_empirical": round(lo, 4) if not np.isnan(lo) else None,
            })

    dep_df = pd.DataFrame(dep_rows)
    curve_df = pd.DataFrame(curve_rows)
    return dep_df, curve_df


# ── Figures ──────────────────────────────────────────────────────────────
def plot_tail_dependence(dep_df: pd.DataFrame, curve_df: pd.DataFrame):
    n_pairs = len(PAIRS)
    fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 5), squeeze=False)
    axes = axes.ravel()

    for i, (var_a, var_b) in enumerate(PAIRS):
        ax = axes[i]
        pair_label = f"{var_a}_vs_{var_b}"
        sub = curve_df[curve_df["pair"] == pair_label]
        ax.plot(sub["threshold"], sub["upper_tail_empirical"],
                color="#e74c3c", lw=2, label="Upper tail λ_U")
        ax.plot(sub["threshold"], sub["lower_tail_empirical"],
                color="#2980b9", lw=2, label="Lower tail λ_L")

        # Parametric reference
        row = dep_df[dep_df["pair"] == pair_label].iloc[0]
        ax.axhline(row["upper_tail"], color="#e74c3c", ls="--", alpha=0.5)
        ax.axhline(row["lower_tail"], color="#2980b9", ls="--", alpha=0.5)

        ax.set_title(pair_label.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Threshold quantile", fontsize=10)
        ax.set_ylabel("Tail dependence", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "copula_tail_dependence.png", dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close()


def plot_scatter_matrix(df: pd.DataFrame):
    cols = ["temperature", "cpu_utilization", "pue",
            "energy_demand", "carbon_intensity", "emissions"]
    sub = df[cols].sample(min(4000, len(df)), random_state=42)
    fig, axes = plt.subplots(len(cols), len(cols),
                             figsize=(16, 16))
    for i, ci_col in enumerate(cols):
        for j, cj_col in enumerate(cols):
            ax = axes[i][j]
            if i == j:
                ax.hist(sub[ci_col], bins=40, color="#2b8c4e", alpha=0.7, edgecolor="white", linewidth=0.3)
            else:
                ax.scatter(sub[cj_col], sub[ci_col], s=2, alpha=0.25, color="#2980b9")
            if j == 0:
                ax.set_ylabel(ci_col.replace("_", "\n"), fontsize=7)
            if i == len(cols) - 1:
                ax.set_xlabel(cj_col.replace("_", "\n"), fontsize=7)
            ax.tick_params(labelsize=6)
    fig.suptitle("Copula Scatter Matrix – Key DC Variables", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "copula_scatter_matrix.png", dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close()


# ── Entry Point ──────────────────────────────────────────────────────────
def run():
    print("  ▶ Running copula tail-dependency analysis …")
    df = _build_dataset()
    dep_df, curve_df = compute_copula_metrics(df)

    dep_df.to_csv(OUTPUT_DIR / "copula_tail_dependence.csv", index=False)
    curve_df.to_csv(OUTPUT_DIR / "copula_tail_curves.csv", index=False)

    plot_tail_dependence(dep_df, curve_df)
    plot_scatter_matrix(df)

    print(f"    ✓ {len(PAIRS)} pairs analysed")
    for _, r in dep_df.iterrows():
        print(f"      {r['pair']:30s}  τ={r['kendall_tau']:+.3f}  λ_U={r['upper_tail']:.3f}  λ_L={r['lower_tail']:.3f}")
    return dep_df, curve_df


if __name__ == "__main__":
    run()
