"""Stage 4 physics: Total facility power and optional linear calibration.

Functions
---------
compute_total_power(it_power, pue)
    Total power (MW) = IT power × PUE.

apply_calibration(pred_it, pred_total, calibration, facility_mw, max_pue)
    Apply a pre-fitted linear calibration to IT-power and total-power arrays.
    Returns (calibrated_it, calibrated_total), both clamped to physically
    plausible ranges.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def compute_total_power(
    it_power: np.ndarray,
    pue: np.ndarray,
) -> np.ndarray:
    """Compute total facility power (MW) as IT power × PUE.

    Parameters
    ----------
    it_power:
        IT power in MW.
    pue:
        Power Usage Effectiveness (dimensionless, >= 1).

    Returns
    -------
    np.ndarray
        Total facility power in MW.
    """
    return np.asarray(it_power, dtype=float) * np.asarray(pue, dtype=float)


def apply_calibration(
    pred_it: np.ndarray,
    pred_total: np.ndarray,
    calibration: Dict[str, object],
    facility_mw: float,
    max_pue: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a fitted linear calibration to physics IT-power and total-power arrays.

    The calibration dictionary must contain at minimum:
        ``it_power_slope``, ``it_power_intercept``,
        ``total_power_slope``, ``total_power_intercept``.

    Parameters
    ----------
    pred_it:
        Raw physics IT-power predictions (MW).
    pred_total:
        Raw physics total-power predictions (MW).
    calibration:
        Dict of linear coefficients produced during training.
    facility_mw:
        Nameplate facility capacity (MW) — used for clamping.
    max_pue:
        Maximum physically plausible PUE — used for upper-bound clamping.

    Returns
    -------
    (cal_it, cal_total) : Tuple[np.ndarray, np.ndarray]
        Calibrated IT power and total power in MW.
    """
    it = np.asarray(pred_it, dtype=float)
    tot = np.asarray(pred_total, dtype=float)

    it_slope = float(calibration.get("it_power_slope", 1.0))
    it_intercept = float(calibration.get("it_power_intercept", 0.0))
    tot_slope = float(calibration.get("total_power_slope", 1.0))
    tot_intercept = float(calibration.get("total_power_intercept", 0.0))

    cal_it = np.clip(it_slope * it + it_intercept, 0.0, float(facility_mw))
    cal_total = np.clip(
        tot_slope * tot + tot_intercept, 0.0, float(facility_mw) * float(max_pue)
    )
    return cal_it, cal_total
