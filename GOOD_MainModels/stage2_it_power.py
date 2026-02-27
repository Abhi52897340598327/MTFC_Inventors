"""Stage 2 physics: IT power from CPU utilization.

Formula (Barroso & Hölzle, 2007):
    IT_power_MW = facility_mw * (idle_power_fraction + (1 - idle_power_fraction) * cpu_util)

Where cpu_util is in [0, 1].
"""

from __future__ import annotations

import numpy as np

from config import PipelineConfig


def compute_it_power_from_config(
    cpu_util: np.ndarray,
    config: PipelineConfig,
) -> np.ndarray:
    """Compute IT power (MW) from CPU utilization fraction and pipeline config.

    Parameters
    ----------
    cpu_util:
        Array of CPU utilization values in [0, 1].
    config:
        Pipeline configuration supplying ``facility_mw`` and
        ``idle_power_fraction``.

    Returns
    -------
    np.ndarray
        IT power in MW, same shape as ``cpu_util``.
    """
    cpu = np.asarray(cpu_util, dtype=float)
    idle_frac = float(config.idle_power_fraction)
    facility_mw = float(config.facility_mw)
    return facility_mw * (idle_frac + (1.0 - idle_frac) * cpu)
