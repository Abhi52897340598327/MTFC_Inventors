"""Stage 4 (Physics): Total Facility Power = IT Power × PUE.

Formula:
    Total Power (MW) = IT Power (MW) * PUE

Inputs:
    it_power_mw : array-like, IT power in MW (output of Stage 2)
    pue         : array-like, Power Usage Effectiveness (output of Stage 3)

Optional: linear calibration coefficients can be applied to correct systematic
bias between the physics estimate and observed power traces.  When calibration
is disabled the identity transform (slope=1, intercept=0) is used.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def compute_total_power(
    it_power_mw: np.ndarray,
    pue: np.ndarray,
) -> np.ndarray:
    """Compute total facility power as IT power multiplied by PUE.

    Parameters
    ----------
    it_power_mw : array-like
        IT power in MW (Stage 2 output).
    pue : array-like
        Power Usage Effectiveness values (Stage 3 output).

    Returns
    -------
    np.ndarray
        Total facility power in MW, same shape as inputs.
    """
    it_arr = np.asarray(it_power_mw, dtype=float)
    pue_arr = np.asarray(pue, dtype=float)
    return it_arr * pue_arr


def apply_calibration(
    pred_it: np.ndarray,
    pred_total: np.ndarray,
    calibration: Dict[str, float],
    facility_mw: float,
    max_pue: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply optional linear calibration to IT power and total power estimates.

    When calibration is disabled (identity), the inputs are returned unchanged.

    Parameters
    ----------
    pred_it : array-like
        Raw predicted IT power in MW.
    pred_total : array-like
        Raw predicted total power in MW.
    calibration : dict
        Dictionary with keys ``it_power_slope``, ``it_power_intercept``,
        ``total_power_slope``, ``total_power_intercept``.
    facility_mw : float
        Total installed IT capacity in MW (used for upper clip).
    max_pue : float
        Maximum PUE (used for upper clip: facility_mw * max_pue).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Calibrated ``(pred_it, pred_total)`` arrays.
    """
    it_arr = np.asarray(pred_it, dtype=float)
    total_arr = np.asarray(pred_total, dtype=float)

    it_cal = (
        calibration.get("it_power_slope", 1.0) * it_arr
        + calibration.get("it_power_intercept", 0.0)
    )
    total_cal = (
        calibration.get("total_power_slope", 1.0) * total_arr
        + calibration.get("total_power_intercept", 0.0)
    )

    upper = float(facility_mw) * float(max_pue)
    it_cal = np.clip(it_cal, 0.0, upper)
    total_cal = np.clip(total_cal, 0.0, upper)
    return it_cal, total_cal


# ---------------------------------------------------------------------------
# Convenience wrapper that accepts a PipelineConfig object directly
# ---------------------------------------------------------------------------

def compute_total_power_from_config(
    it_power_mw: np.ndarray,
    pue: np.ndarray,
    calibration: Optional[Dict[str, float]] = None,
    config=None,
    use_calibration: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute total power and optionally apply calibration.

    Parameters
    ----------
    it_power_mw : array-like
        IT power in MW (Stage 2 output).
    pue : array-like
        PUE values (Stage 3 output).
    calibration : dict or None
        Calibration coefficients.  If ``None`` or ``use_calibration`` is
        ``False``, the identity transform is used.
    config : PipelineConfig or None
        Pipeline config (needed when ``use_calibration`` is True).
    use_calibration : bool
        Whether to apply the calibration transform.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(it_power_cal, total_power_cal)`` in MW.
    """
    total_raw = compute_total_power(it_power_mw, pue)
    it_raw = np.asarray(it_power_mw, dtype=float)

    if use_calibration and calibration is not None and config is not None:
        return apply_calibration(
            it_raw,
            total_raw,
            calibration=calibration,
            facility_mw=float(config.facility_mw),
            max_pue=float(config.max_pue),
        )
    return it_raw, total_raw


# ---------------------------------------------------------------------------
# Stand-alone demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 4 – Total Power physics model")
    parser.add_argument("--facility-mw", type=float, default=100.0)
    parser.add_argument("--max-pue", type=float, default=2.0)
    args = parser.parse_args()

    sample_it = np.array([30.0, 47.5, 65.0, 82.5, 100.0])
    sample_pue = np.array([1.40, 1.45, 1.50, 1.58, 1.65])

    total = compute_total_power(sample_it, sample_pue)

    print(f"facility_mw={args.facility_mw}  max_pue={args.max_pue}")
    print(f"{'IT Power (MW)':>14}  {'PUE':>6}  {'Total Power (MW)':>16}")
    print("-" * 42)
    for it, pue, tp in zip(sample_it, sample_pue, total):
        print(f"{it:>14.2f}  {pue:>6.3f}  {tp:>16.4f}")
