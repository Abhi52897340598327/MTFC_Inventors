"""Stage 6 (Physics): Carbon Emissions = Total Power × Carbon Intensity.

Formula:
    Emissions (kg CO₂e/h) = Total Power (MW) * Carbon Intensity (kg CO₂e/MWh)

Inputs:
    total_power_mw    : array-like, total facility power in MW (output of Stage 4)
    carbon_intensity  : array-like, grid carbon intensity in kg CO₂e/MWh
                        (predicted by Stage 5)

Note:
    The product of MW and kg/MWh yields kg CO₂e per hour (assuming power is
    measured as an instantaneous MW reading over a 1-hour interval).
"""

from __future__ import annotations

import numpy as np


def compute_emissions(
    total_power_mw: np.ndarray,
    carbon_intensity: np.ndarray,
) -> np.ndarray:
    """Compute carbon emissions from total facility power and grid carbon intensity.

    Parameters
    ----------
    total_power_mw : array-like
        Total facility power in MW (Stage 4 output).
    carbon_intensity : array-like
        Grid carbon intensity in kg CO₂e per MWh (Stage 5 prediction).

    Returns
    -------
    np.ndarray
        Carbon emissions in kg CO₂e per hour, same shape as inputs.
    """
    power_arr = np.asarray(total_power_mw, dtype=float)
    ci_arr = np.asarray(carbon_intensity, dtype=float)
    return power_arr * ci_arr


# ---------------------------------------------------------------------------
# Convenience wrapper — no config needed (pure multiplication)
# ---------------------------------------------------------------------------

def compute_emissions_from_stages(
    total_power_mw: np.ndarray,
    carbon_intensity: np.ndarray,
) -> np.ndarray:
    """Alias for :func:`compute_emissions` for explicit stage-chaining use.

    Parameters
    ----------
    total_power_mw : array-like
        Total facility power in MW.
    carbon_intensity : array-like
        Grid carbon intensity in kg CO₂e/MWh.

    Returns
    -------
    np.ndarray
        Carbon emissions in kg CO₂e/h.
    """
    return compute_emissions(total_power_mw, carbon_intensity)


# ---------------------------------------------------------------------------
# Stand-alone demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 6 – Carbon Emissions physics model")
    parser.parse_args()

    sample_power = np.array([42.0, 67.5, 97.5, 120.0, 160.0])
    sample_ci = np.array([200.0, 350.0, 450.0, 300.0, 150.0])

    emissions = compute_emissions(sample_power, sample_ci)

    print(f"{'Total Power (MW)':>16}  {'Carbon Intensity (kg/MWh)':>24}  {'Emissions (kg CO2e/h)':>21}")
    print("-" * 66)
    for pw, ci, em in zip(sample_power, sample_ci, emissions):
        print(f"{pw:>16.2f}  {ci:>24.1f}  {em:>21.2f}")
