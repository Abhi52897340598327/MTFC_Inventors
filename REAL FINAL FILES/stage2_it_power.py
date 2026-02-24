"""Stage 2 (Physics): IT Power from predicted CPU utilization.

Formula:
    IT Power (MW) = facility_mw * (idle_power_fraction + (1 - idle_power_fraction) * cpu_util)

Inputs:
    cpu_util  : array-like, predicted CPU utilization in [0, 1]

Config parameters used:
    facility_mw         : total installed IT capacity (MW)
    idle_power_fraction : fraction of facility_mw drawn at 0% CPU
"""

from __future__ import annotations

import numpy as np


def compute_it_power(
    cpu_util: np.ndarray,
    facility_mw: float,
    idle_power_fraction: float,
) -> np.ndarray:
    """Compute IT power (MW) from CPU utilization.

    Parameters
    ----------
    cpu_util : array-like
        Predicted CPU utilization values in [0, 1].
    facility_mw : float
        Total installed IT capacity in MW.
    idle_power_fraction : float
        Fraction of ``facility_mw`` consumed at 0 % CPU (idle baseline).

    Returns
    -------
    np.ndarray
        IT power in MW, same shape as ``cpu_util``.
    """
    cpu_arr = np.asarray(cpu_util, dtype=float)
    return facility_mw * (idle_power_fraction + (1.0 - idle_power_fraction) * cpu_arr)


# ---------------------------------------------------------------------------
# Convenience wrapper that accepts a PipelineConfig object directly
# ---------------------------------------------------------------------------

def compute_it_power_from_config(cpu_util: np.ndarray, config) -> np.ndarray:
    """Compute IT power using parameters from a :class:`PipelineConfig`.

    Parameters
    ----------
    cpu_util : array-like
        Predicted CPU utilization values in [0, 1].
    config : PipelineConfig
        Pipeline configuration object with ``facility_mw`` and
        ``idle_power_fraction`` attributes.

    Returns
    -------
    np.ndarray
        IT power in MW.
    """
    return compute_it_power(
        cpu_util,
        facility_mw=float(config.facility_mw),
        idle_power_fraction=float(config.idle_power_fraction),
    )


# ---------------------------------------------------------------------------
# Stand-alone demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 2 – IT Power physics model")
    parser.add_argument("--facility-mw", type=float, default=100.0)
    parser.add_argument("--idle-fraction", type=float, default=0.30)
    args = parser.parse_args()

    sample_cpu = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    it_power = compute_it_power(sample_cpu, args.facility_mw, args.idle_fraction)

    print(f"facility_mw={args.facility_mw}  idle_fraction={args.idle_fraction}")
    print(f"{'CPU Util':>10}  {'IT Power (MW)':>14}")
    print("-" * 28)
    for cpu, pw in zip(sample_cpu, it_power):
        print(f"{cpu:>10.2f}  {pw:>14.4f}")
