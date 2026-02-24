"""Stage 3 (Physics): Power Usage Effectiveness (PUE) from CPU utilization and temperature.

Formula:
    PUE = base_pue
          + pue_temp_coef  * max(0, temp_f - cooling_threshold_f)
          + pue_cpu_coef   * cpu_util
          [+ pue_dewpoint_coef * max(0, dew_point_f - dew_point_threshold_f)]   (optional)
          [- pue_wind_coef    * max(0, wind_speed_mps)]                          (optional)
    PUE = clip(PUE, base_pue, max_pue)

Inputs:
    cpu_util      : array-like, predicted CPU utilization in [0, 1]
    temp_f        : array-like, ambient temperature in degrees Fahrenheit
    dew_point_f   : array-like or None, dew-point temperature in °F (optional)
    wind_speed_mps: array-like or None, wind speed in m/s (optional)

Config parameters used:
    base_pue               : PUE at 0% CPU and reference temperature
    max_pue                : hard upper clip for PUE
    pue_temp_coef          : sensitivity of PUE to temperature excess
    pue_cpu_coef           : sensitivity of PUE to CPU load
    pue_dewpoint_coef      : sensitivity of PUE to dew-point excess
    pue_wind_coef          : cooling benefit of wind (reduces PUE)
    cooling_threshold_f    : temperature (°F) above which extra cooling load kicks in
    dew_point_threshold_f  : dew-point (°F) above which humidity penalty kicks in
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def compute_pue(
    cpu_util: np.ndarray,
    temp_f: np.ndarray,
    base_pue: float,
    max_pue: float,
    pue_temp_coef: float,
    pue_cpu_coef: float,
    cooling_threshold_f: float,
    dew_point_f: Optional[np.ndarray] = None,
    dew_point_threshold_f: float = 55.0,
    pue_dewpoint_coef: float = 0.0,
    wind_speed_mps: Optional[np.ndarray] = None,
    pue_wind_coef: float = 0.0,
) -> np.ndarray:
    """Compute PUE from CPU utilization and environmental conditions.

    Parameters
    ----------
    cpu_util : array-like
        Predicted CPU utilization values in [0, 1].
    temp_f : array-like
        Ambient temperature in degrees Fahrenheit at time t+h.
    base_pue : float
        Baseline PUE at zero CPU load and reference temperature.
    max_pue : float
        Hard upper clip for PUE.
    pue_temp_coef : float
        Increase in PUE per degree Fahrenheit above ``cooling_threshold_f``.
    pue_cpu_coef : float
        Increase in PUE per unit of CPU utilization.
    cooling_threshold_f : float
        Temperature (°F) above which additional cooling load is incurred.
    dew_point_f : array-like or None
        Dew-point temperature in °F (optional).
    dew_point_threshold_f : float
        Dew-point (°F) above which humidity penalty is applied.
    pue_dewpoint_coef : float
        Increase in PUE per degree above ``dew_point_threshold_f``.
    wind_speed_mps : array-like or None
        Wind speed in m/s (optional).  Higher wind reduces PUE.
    pue_wind_coef : float
        Decrease in PUE per m/s of wind speed.

    Returns
    -------
    np.ndarray
        PUE values, clipped to [base_pue, max_pue], same shape as ``cpu_util``.
    """
    cpu_arr = np.asarray(cpu_util, dtype=float)
    temp_arr = np.asarray(temp_f, dtype=float)

    temp_above = np.maximum(0.0, temp_arr - cooling_threshold_f)
    pue = base_pue + pue_temp_coef * temp_above + pue_cpu_coef * cpu_arr

    if dew_point_f is not None:
        dew_arr = np.asarray(dew_point_f, dtype=float)
        dew_above = np.maximum(0.0, dew_arr - dew_point_threshold_f)
        pue = pue + pue_dewpoint_coef * dew_above

    if wind_speed_mps is not None:
        wind_arr = np.asarray(wind_speed_mps, dtype=float)
        pue = pue - pue_wind_coef * np.maximum(0.0, wind_arr)

    return np.clip(pue, base_pue, max_pue)


# ---------------------------------------------------------------------------
# Convenience wrapper that accepts a PipelineConfig object directly
# ---------------------------------------------------------------------------

def compute_pue_from_config(
    cpu_util: np.ndarray,
    temp_f: np.ndarray,
    config,
    dew_point_f: Optional[np.ndarray] = None,
    wind_speed_mps: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute PUE using parameters from a :class:`PipelineConfig`.

    Parameters
    ----------
    cpu_util : array-like
        Predicted CPU utilization values in [0, 1].
    temp_f : array-like
        Ambient temperature in degrees Fahrenheit at time t+h.
    config : PipelineConfig
        Pipeline configuration object.
    dew_point_f : array-like or None
        Optional dew-point temperature in °F.
    wind_speed_mps : array-like or None
        Optional wind speed in m/s.

    Returns
    -------
    np.ndarray
        PUE values clipped to [base_pue, max_pue].
    """
    return compute_pue(
        cpu_util=cpu_util,
        temp_f=temp_f,
        base_pue=float(config.base_pue),
        max_pue=float(config.max_pue),
        pue_temp_coef=float(config.pue_temp_coef),
        pue_cpu_coef=float(config.pue_cpu_coef),
        cooling_threshold_f=float(config.cooling_threshold_f),
        dew_point_f=dew_point_f,
        dew_point_threshold_f=float(config.dew_point_threshold_f),
        pue_dewpoint_coef=float(config.pue_dewpoint_coef),
        wind_speed_mps=wind_speed_mps,
        pue_wind_coef=float(config.pue_wind_coef),
    )


# ---------------------------------------------------------------------------
# Stand-alone demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 3 – PUE physics model")
    parser.add_argument("--base-pue", type=float, default=1.4)
    parser.add_argument("--max-pue", type=float, default=2.0)
    parser.add_argument("--pue-temp-coef", type=float, default=0.005)
    parser.add_argument("--pue-cpu-coef", type=float, default=0.1)
    parser.add_argument("--cooling-threshold-f", type=float, default=65.0)
    args = parser.parse_args()

    sample_cpu = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    sample_temp = np.array([50.0, 65.0, 75.0, 85.0, 95.0])

    pue = compute_pue(
        sample_cpu,
        sample_temp,
        base_pue=args.base_pue,
        max_pue=args.max_pue,
        pue_temp_coef=args.pue_temp_coef,
        pue_cpu_coef=args.pue_cpu_coef,
        cooling_threshold_f=args.cooling_threshold_f,
    )

    print(f"base_pue={args.base_pue}  max_pue={args.max_pue}  "
          f"pue_temp_coef={args.pue_temp_coef}  pue_cpu_coef={args.pue_cpu_coef}  "
          f"cooling_threshold_f={args.cooling_threshold_f}")
    print(f"{'CPU Util':>10}  {'Temp (°F)':>10}  {'PUE':>8}")
    print("-" * 34)
    for cpu, temp, p in zip(sample_cpu, sample_temp, pue):
        print(f"{cpu:>10.2f}  {temp:>10.1f}  {p:>8.4f}")
