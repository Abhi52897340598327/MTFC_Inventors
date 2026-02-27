"""Stage 6 physics: Carbon emissions from total power and carbon intensity.

Formula:
    emissions = total_power_MW * carbon_intensity

The units of ``emissions`` match those of ``carbon_intensity``.  For the PJM
grid the carbon intensity is supplied in kg CO₂ / MWh, so the result is in
kg CO₂ per hour (equivalently, kg CO₂ for a one-hour window).
"""

from __future__ import annotations

import numpy as np


def compute_emissions(
    total_power_mw: np.ndarray,
    carbon_intensity: np.ndarray,
) -> np.ndarray:
    """Compute carbon emissions as total power × carbon intensity.

    Parameters
    ----------
    total_power_mw:
        Total facility power in MW.
    carbon_intensity:
        Grid carbon intensity (e.g. kg CO₂ / MWh).

    Returns
    -------
    np.ndarray
        Emissions in the unit implied by ``carbon_intensity`` multiplied by MW
        (e.g. kg CO₂ / h when carbon_intensity is in kg CO₂ / MWh).
    """
    return np.asarray(total_power_mw, dtype=float) * np.asarray(carbon_intensity, dtype=float)
