"""Stage 3 physics: Power Usage Effectiveness (PUE) from CPU utilization and temperature.

Formula:
    PUE = base_pue
        + pue_temp_coef  * max(0, temp_f - cooling_threshold_f)
        + pue_cpu_coef   * cpu_util
        + pue_dewpoint_coef * max(0, dew_point_f - dew_point_threshold_f)  [optional]
        - pue_wind_coef  * wind_speed_mps                                   [optional]

Result is clamped to [base_pue, max_pue].
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from config import PipelineConfig


def compute_pue_from_config(
    cpu_util: np.ndarray,
    temp_f: np.ndarray,
    config: PipelineConfig,
    dew_point_f: Optional[np.ndarray] = None,
    wind_speed_mps: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute PUE from CPU utilization, ambient temperature, and pipeline config.

    Parameters
    ----------
    cpu_util:
        CPU utilization in [0, 1].
    temp_f:
        Ambient temperature in degrees Fahrenheit.
    config:
        Pipeline configuration supplying PUE physics coefficients.
    dew_point_f:
        Optional dew-point temperature in degrees Fahrenheit.
    wind_speed_mps:
        Optional wind speed in metres per second (higher wind → better cooling
        → lower PUE).

    Returns
    -------
    np.ndarray
        PUE values, clamped to [config.base_pue, config.max_pue].
    """
    cpu = np.asarray(cpu_util, dtype=float)
    temp = np.asarray(temp_f, dtype=float)

    pue = (
        config.base_pue
        + config.pue_temp_coef * np.maximum(0.0, temp - config.cooling_threshold_f)
        + config.pue_cpu_coef * cpu
    )

    if dew_point_f is not None:
        dp = np.asarray(dew_point_f, dtype=float)
        pue = pue + config.pue_dewpoint_coef * np.maximum(0.0, dp - config.dew_point_threshold_f)

    if wind_speed_mps is not None:
        wind = np.asarray(wind_speed_mps, dtype=float)
        pue = pue - config.pue_wind_coef * wind

    return np.clip(pue, config.base_pue, config.max_pue)
