"""
Sobol Sensitivity Indices (Variance-Based Global SA)
=====================================================
Uses Saltelli's extension of the Sobol' sequence to estimate
first-order (S1) and total-effect (ST) indices for both
energy consumption and emissions outputs.

Parameters sampled
------------------
1. idle_power_fraction    [0.30 – 0.50]
2. cpu_utilization        [20 – 80]
3. temperature_f          [20 – 100]
4. pue_baseline           [1.15 – 1.50]
5. carbon_intensity       [280 – 420]

Outputs
-------
- sobol_indices.csv          (emissions target)
- sobol_indices_energy.csv   (energy target)
- figures/sobol_indices.png
- figures/sobol_indices_energy.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import DC_PARAMS, GRID, SOBOL, OUTPUT_DIR, FIGURE_DIR, PLOT

np.random.seed(SOBOL["seed"])


# ── Parameter space ──────────────────────────────────────────────────────
PARAMS = [
    ("idle_power_fraction", 0.30, 0.50),
    ("cpu_utilization",     20.0, 80.0),
    ("temperature_f",       20.0, 100.0),
    ("pue_baseline",        1.15, 1.50),
    ("carbon_intensity",    280,  420),
]
K = len(PARAMS)


# ── Saltelli sampling ───────────────────────────────────────────────────
def _saltelli_sample(n: int) -> np.ndarray:
    """Generate (n * (2K + 2)) × K matrix via Saltelli scheme."""
    # Two independent Sobol-like quasi-random bases
    A = np.random.uniform(0, 1, (n, K))
    B = np.random.uniform(0, 1, (n, K))

    samples = [A, B]
    for i in range(K):
        AB_i = A.copy()
        AB_i[:, i] = B[:, i]
        samples.append(AB_i)
    for i in range(K):
        BA_i = B.copy()
        BA_i[:, i] = A[:, i]
        samples.append(BA_i)

    X = np.vstack(samples)  # shape (n*(2K+2), K)
    # Scale to physical bounds
    for j, (_, lo, hi) in enumerate(PARAMS):
        X[:, j] = lo + X[:, j] * (hi - lo)
    return X, n


# ── Model evaluation ────────────────────────────────────────────────────
def _evaluate(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (emissions_vector, energy_vector) for each row of X."""
    idle   = X[:, 0]
    cpu    = X[:, 1]
    temp_f = X[:, 2]
    pue_b  = X[:, 3]
    ci     = X[:, 4]

    it_mw = DC_PARAMS["it_capacity_mw"] * (idle + (1 - idle) * cpu / 100)
    delta_t = np.maximum(temp_f - 65, 0)
    pue = pue_b + DC_PARAMS["pue_temp_coef"] * delta_t + DC_PARAMS["pue_cpu_coef"] * cpu
    total_mw = it_mw * pue
    energy_mwh = total_mw * 8760
    emissions_tons = total_mw * ci * 8760 / 1000

    return emissions_tons, energy_mwh


# ── Sobol index estimation ──────────────────────────────────────────────
def _sobol_indices(Y_all: np.ndarray, n: int) -> pd.DataFrame:
    """Estimate S1 and ST from Saltelli output matrix."""
    Y_A  = Y_all[:n]
    Y_B  = Y_all[n:2*n]
    Y_AB = [Y_all[(2 + i)*n : (3 + i)*n] for i in range(K)]
    Y_BA = [Y_all[(2 + K + i)*n : (3 + K + i)*n] for i in range(K)]

    f0_sq = Y_A.mean() * Y_B.mean()
    var_y = np.var(np.concatenate([Y_A, Y_B]))

    rows = []
    for i in range(K):
        # S1:  Jansen estimator
        s1_num = np.mean(Y_B * (Y_AB[i] - Y_A))
        s1 = s1_num / var_y if var_y > 0 else 0

        # ST:  Jansen estimator
        st_num = 0.5 * np.mean((Y_A - Y_AB[i]) ** 2)
        st = st_num / var_y if var_y > 0 else 0

        # Clamp to [0, 1]
        s1 = float(np.clip(s1, 0, 1))
        st = float(np.clip(st, 0, 1))

        rows.append({
            "parameter": PARAMS[i][0],
            "S1": round(s1, 4),
            "ST": round(st, 4),
            "S1_rank": 0,
            "ST_rank": 0,
        })

    df = pd.DataFrame(rows)
    df["S1_rank"] = df["S1"].rank(ascending=False).astype(int)
    df["ST_rank"] = df["ST"].rank(ascending=False).astype(int)
    return df


# ── Figures ──────────────────────────────────────────────────────────────
def _plot_sobol(df: pd.DataFrame, target_name: str, fname: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    w = 0.35

    bars1 = ax.bar(x - w/2, df["S1"], w, label="First-order (S₁)",
                   color="#2980b9", edgecolor="white")
    bars2 = ax.bar(x + w/2, df["ST"], w, label="Total-effect (S_T)",
                   color="#e74c3c", edgecolor="white")

    for b in bars1:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f"{b.get_height():.3f}", ha="center", fontsize=9)
    for b in bars2:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f"{b.get_height():.3f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in df["parameter"]],
                       fontsize=10)
    ax.set_ylabel("Sobol Index", fontsize=12)
    ax.set_title(f"Sobol Sensitivity Indices – {target_name}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, min(df[["S1", "ST"]].max().max() * 1.3, 1.05))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / fname, dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close()


# ── Entry Point ──────────────────────────────────────────────────────────
def run():
    print("  ▶ Running Sobol sensitivity analysis …")
    n = SOBOL["n_samples"]
    X, n_base = _saltelli_sample(n)
    Y_em, Y_en = _evaluate(X)

    sobol_em = _sobol_indices(Y_em, n_base)
    sobol_en = _sobol_indices(Y_en, n_base)

    sobol_em.to_csv(OUTPUT_DIR / "sobol_indices.csv", index=False)
    sobol_en.to_csv(OUTPUT_DIR / "sobol_indices_energy.csv", index=False)

    _plot_sobol(sobol_em, "Emissions (t CO₂/yr)", "sobol_indices.png")
    _plot_sobol(sobol_en, "Energy (MWh/yr)", "sobol_indices_energy.png")

    print("    ✓ Sobol indices (emissions):")
    for _, r in sobol_em.iterrows():
        print(f"      {r['parameter']:25s}  S1={r['S1']:.4f}  ST={r['ST']:.4f}")
    return sobol_em, sobol_en


if __name__ == "__main__":
    run()
