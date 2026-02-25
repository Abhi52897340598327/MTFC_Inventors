"""Schema and validation contracts for REAL FINAL FILES pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "cpu": ["real_timestamp", "avg_cpu_utilization", "num_tasks_sampled"],
    "temperature": ["timestamp", "temperature_c"],
    "carbon_intensity": ["timestamp", "carbon_intensity_kg_per_mwh"],
}

OPTIONAL_REQUIRED_COLUMNS = {
    "grid_exog": [
        "timestamp",
        "demand_mwh",
        "demand_forecast_mwh",
        "demand_forecast_error_mwh",
        "interchange_mwh",
        "net_generation_mwh",
        "fuel_col_share",
        "fuel_ng_share",
        "fuel_nuc_share",
        "fuel_wnd_share",
        "fuel_sun_share",
        "fuel_wat_share",
        "fuel_oth_share",
    ],
    "weather_exog": ["dew_point_c", "wind_speed_mps"],
    "power_optional": ["measured_power_util", "production_power_util"],
    "cluster_plus_power_optional": ["avg_cpu_utilization", "num_tasks_sampled", "measured_power_util", "production_power_util"],
    "pjm_hourly_demand_optional": ["datetime_utc", "demand_mwh"],
}


@dataclass
class StageMetrics:
    """Per-stage metric row for reports."""

    stage: str
    split: str
    model: str
    mae: float
    rmse: float
    mape: float
    smape: float
    r2: float
    ci_mae_low: float = float("nan")
    ci_mae_high: float = float("nan")
    ci_rmse_low: float = float("nan")
    ci_rmse_high: float = float("nan")
    ci_mape_low: float = float("nan")
    ci_mape_high: float = float("nan")
    ci_smape_low: float = float("nan")
    ci_smape_high: float = float("nan")
    ci_r2_low: float = float("nan")
    ci_r2_high: float = float("nan")
    defensible: bool | None = None
    uplift_vs_persistence_rmse: float | None = None


@dataclass
class RunArtifacts:
    """Output locations for one pipeline run."""

    run_id: str
    result_dir: str
    figure_dir: str
    model_dir: str
    files: Dict[str, str] = field(default_factory=dict)


def validate_required_columns(df: pd.DataFrame, dataset_key: str) -> None:
    """Validate required column presence for a source dataset."""
    expected = REQUIRED_COLUMNS.get(dataset_key)
    if expected is None:
        raise KeyError(f"Unknown dataset_key '{dataset_key}'.")

    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset '{dataset_key}' is missing required columns: {missing}. "
            f"Expected columns include: {expected}"
        )


def validate_numeric_ranges(df: pd.DataFrame) -> None:
    """Validate core value ranges for cleaned merged data."""
    if "avg_cpu_utilization" in df.columns:
        invalid = df[(df["avg_cpu_utilization"] < 0) | (df["avg_cpu_utilization"] > 1)]
        if not invalid.empty:
            raise ValueError(
                "avg_cpu_utilization must be in [0, 1]. "
                f"Found {len(invalid)} invalid rows."
            )

    if "temperature_c" in df.columns:
        invalid = df[(df["temperature_c"] < -80) | (df["temperature_c"] > 80)]
        if not invalid.empty:
            raise ValueError(
                "temperature_c outside plausible range [-80, 80]. "
                f"Found {len(invalid)} invalid rows."
            )

    if "carbon_intensity" in df.columns:
        invalid = df[(df["carbon_intensity"] <= 0) | (df["carbon_intensity"] > 2500)]
        if not invalid.empty:
            raise ValueError(
                "carbon_intensity must be in (0, 2500]. "
                f"Found {len(invalid)} invalid rows."
            )


def validate_monotonic_timestamp(df: pd.DataFrame, timestamp_col: str = "timestamp") -> None:
    """Validate timestamp monotonicity and duplicate handling."""
    if timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp column '{timestamp_col}'.")

    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        raise TypeError(f"Column '{timestamp_col}' must be datetime-like.")

    if not df[timestamp_col].is_monotonic_increasing:
        raise ValueError(f"Column '{timestamp_col}' must be sorted ascending.")

    duplicate_count = int(df[timestamp_col].duplicated().sum())
    if duplicate_count > 0:
        raise ValueError(f"Found {duplicate_count} duplicate timestamps in '{timestamp_col}'.")


def validate_merged_columns(df: pd.DataFrame) -> None:
    """Validate merged dataset columns required by the modeling pipeline."""
    needed = [
        "timestamp",
        "avg_cpu_utilization",
        "num_tasks_sampled",
        "temperature_c",
        "carbon_intensity",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Merged data missing required columns: {missing}")


def validate_optional_columns(df: pd.DataFrame, dataset_key: str) -> None:
    """Validate required column contracts for optional datasets."""
    if dataset_key not in OPTIONAL_REQUIRED_COLUMNS:
        raise KeyError(f"Unknown optional dataset_key '{dataset_key}'.")

    if dataset_key == "grid_exog":
        expected = OPTIONAL_REQUIRED_COLUMNS["grid_exog"]
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(
                f"Optional dataset 'grid_exog' is missing required columns: {missing}. "
                f"Expected columns include: {expected}"
            )
        return

    if dataset_key == "weather_exog":
        has_ts = ("timestamp" in df.columns) or ("DATE" in df.columns)
        if not has_ts:
            raise ValueError("Optional dataset 'weather_exog' must include 'timestamp' or 'DATE'.")
        missing = [c for c in OPTIONAL_REQUIRED_COLUMNS["weather_exog"] if c not in df.columns]
        if missing:
            raise ValueError(
                f"Optional dataset 'weather_exog' is missing required columns: {missing}. "
                "Required when weather exogenous mode is enabled."
            )
        return

    if dataset_key == "power_optional":
        has_ts = ("timestamp" in df.columns) or ("real_timestamp" in df.columns)
        if not has_ts:
            raise ValueError("Optional dataset 'power_optional' must include 'timestamp' or 'real_timestamp'.")
        has_power_signal = any(c in df.columns for c in OPTIONAL_REQUIRED_COLUMNS["power_optional"])
        if not has_power_signal:
            raise ValueError(
                "Optional dataset 'power_optional' must include at least one of "
                f"{OPTIONAL_REQUIRED_COLUMNS['power_optional']}."
            )
        return

    if dataset_key == "cluster_plus_power_optional":
        has_ts = ("timestamp" in df.columns) or ("real_timestamp" in df.columns)
        if not has_ts:
            raise ValueError(
                "Optional dataset 'cluster_plus_power_optional' must include 'timestamp' or 'real_timestamp'."
            )
        missing = [c for c in OPTIONAL_REQUIRED_COLUMNS["cluster_plus_power_optional"] if c not in df.columns]
        if missing:
            raise ValueError(
                "Optional dataset 'cluster_plus_power_optional' is missing required columns: "
                f"{missing}."
            )
        return

    if dataset_key == "pjm_hourly_demand_optional":
        missing = [c for c in OPTIONAL_REQUIRED_COLUMNS["pjm_hourly_demand_optional"] if c not in df.columns]
        if missing:
            raise ValueError(
                f"Optional dataset 'pjm_hourly_demand_optional' is missing required columns: {missing}."
            )
        return


def validate_future_exog_frame(
    df: pd.DataFrame,
    required_columns: Optional[Sequence[str]] = None,
) -> None:
    """Validate future exogenous frame for iterative prediction."""
    required = list(required_columns) if required_columns is not None else ["timestamp", "temperature_c"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "future_exog_df is missing required columns. "
            f"Missing: {missing}"
        )

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise TypeError("future_exog_df['timestamp'] must be datetime-like.")

    if df["timestamp"].duplicated().any():
        raise ValueError("future_exog_df contains duplicate timestamps.")

    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("future_exog_df timestamps must be monotonic increasing.")

    non_ts_required = [c for c in required if c != "timestamp"]
    if non_ts_required:
        null_counts = {c: int(df[c].isna().sum()) for c in non_ts_required if df[c].isna().any()}
        if null_counts:
            raise ValueError(
                "future_exog_df contains nulls in required columns. "
                f"Null counts: {null_counts}"
            )


__all__ = [
    "REQUIRED_COLUMNS",
    "OPTIONAL_REQUIRED_COLUMNS",
    "RunArtifacts",
    "StageMetrics",
    "validate_required_columns",
    "validate_optional_columns",
    "validate_numeric_ranges",
    "validate_monotonic_timestamp",
    "validate_merged_columns",
    "validate_future_exog_frame",
]
