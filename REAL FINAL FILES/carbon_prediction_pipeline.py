"""Leak-free, OOF-evaluated 6-stage carbon forecasting pipeline.

Stages:
1) Stage 1 (ML): CPU utilization forecast (t+h)
2) Stage 2 (Physics): IT power from predicted CPU
3) Stage 3 (Physics): PUE from predicted CPU + future temperature
4) Stage 4 (Physics): Total power = IT power * PUE
5) Stage 5 (ML): Carbon intensity forecast (t+h)
6) Stage 6 (Physics): Emissions = Total power * predicted carbon intensity
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PipelineConfig, make_config
from random_forest_model import fit as fit_cpu_model
from random_forest_model import predict as predict_cpu_model
from schema import (
    RunArtifacts,
    StageMetrics,
    validate_future_exog_frame,
    validate_merged_columns,
    validate_monotonic_timestamp,
    validate_numeric_ranges,
    validate_optional_columns,
    validate_required_columns,
)
from utils import (
    assert_no_bfill,
    assert_strictly_past_features,
    bootstrap_ci_for_metrics,
    calc_metrics,
    create_run_id,
    ensure_dir,
    get_logger,
    relative_rmse_improvement,
    save_dataframe,
    save_figure,
    save_json,
    save_pickle,
)
from xgboost_model import fit as fit_ci_model
from xgboost_model import predict as predict_ci_model


class CarbonForecastPipeline:
    """Rigorous forecasting pipeline with strict leakage controls."""

    def __init__(
        self,
        config: PipelineConfig,
        use_grid_exog: Optional[bool] = None,
        use_weather_exog: Optional[bool] = None,
        use_power_calibration: Optional[bool] = None,
    ):
        self.config = config
        self.log = get_logger("CarbonPipeline")

        self.run_id: str = create_run_id("carbon")
        self.fitted: bool = False

        self.horizon_steps: int = config.forecast_horizon_hours
        self.seasonal_period_steps: int = 24

        self.models: Dict[str, Any] = {}
        self.best_params: Dict[str, Dict[str, Any]] = {}

        self.feature_cols: Dict[str, List[str]] = {}
        self.feature_lag_map: Dict[str, Dict[str, int]] = {}
        self.zero_lag_allow: Dict[str, set] = {}

        self.cv_metrics_df: Optional[pd.DataFrame] = None
        self.train_context_df: Optional[pd.DataFrame] = None
        self.context_rows: int = 64
        self.latest_report: Optional[Dict[str, Any]] = None
        self.coverage_report_df: pd.DataFrame = pd.DataFrame()
        self.stage5_feature_importance_df: pd.DataFrame = pd.DataFrame()
        self.physics_calibration_df: pd.DataFrame = pd.DataFrame()
        self.physics_calibration: Dict[str, float] = {
            "it_power_slope": 1.0,
            "it_power_intercept": 0.0,
            "total_power_slope": 1.0,
            "total_power_intercept": 0.0,
        }
        self.calibration_valid: bool = True
        self.calibration_reason: str = "disabled_identity"
        self.selected_variant: str = "baseline_core"
        self.using_optional_features: bool = False
        self.required_future_exog_columns: List[str] = ["timestamp", "temperature_c"]
        self.sanity_checks: Dict[str, Any] = {}

        self.use_grid_exog_request = use_grid_exog
        self.use_weather_exog_request = use_weather_exog
        self.use_power_calibration_request = use_power_calibration
        self.optional_dataset_usage: Dict[str, bool] = {
            "grid_exog": False,
            "weather_exog": False,
            "power_optional": False,
            "cluster_plus_power_optional": False,
            "pjm_hourly_demand_optional": False,
            "power_calibration": False,
        }

        self.leakage_checks = {
            "no_bfill_in_source": False,
            "strict_past_stage1": False,
            "strict_past_stage5": False,
        }

    # ------------------------------------------------------------------
    # Data loading and validation
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_timestamp(series: pd.Series, column_name: str) -> pd.Series:
        parsed = pd.to_datetime(series, utc=True, errors="coerce")
        if parsed.isna().any():
            bad = int(parsed.isna().sum())
            raise ValueError(f"Column '{column_name}' has {bad} unparseable timestamps.")
        return parsed.dt.tz_convert("UTC").dt.tz_localize(None)

    def _ensure_no_bfill(self) -> None:
        source_text = Path(__file__).read_text(encoding="utf-8")
        assert_no_bfill(source_text)
        self.leakage_checks["no_bfill_in_source"] = True

    def _resolve_optional_usage(self, key: str, requested: Optional[bool]) -> bool:
        if key not in self.config.datasets:
            if requested is True:
                raise FileNotFoundError(
                    f"Optional dataset '{key}' is enabled but missing from PipelineConfig.datasets."
                )
            return False
        path = self.config.dataset_path(key)
        exists = path.exists()
        if requested is None:
            return exists
        if requested and not exists:
            raise FileNotFoundError(
                f"Optional dataset '{key}' is enabled but file is missing: {path}"
            )
        return bool(requested and exists)

    @staticmethod
    def _coverage_row(
        source: str,
        path: str,
        exists: bool,
        frame: Optional[pd.DataFrame],
        timestamp_col: str,
        core_start: pd.Timestamp,
        core_end: pd.Timestamp,
        core_rows: int,
    ) -> Dict[str, Any]:
        if (not exists) or frame is None or frame.empty or timestamp_col not in frame.columns:
            return {
                "source": source,
                "path": path,
                "exists": bool(exists),
                "rows_total": 0,
                "start_timestamp": "",
                "end_timestamp": "",
                "overlap_rows": 0,
                "overlap_ratio_vs_core": 0.0,
            }

        ts = pd.to_datetime(frame[timestamp_col], errors="coerce")
        ts = ts.dropna().sort_values()
        if ts.empty:
            return {
                "source": source,
                "path": path,
                "exists": bool(exists),
                "rows_total": int(len(frame)),
                "start_timestamp": "",
                "end_timestamp": "",
                "overlap_rows": 0,
                "overlap_ratio_vs_core": 0.0,
            }

        overlap = int(((ts >= core_start) & (ts <= core_end)).sum())
        overlap_ratio = float(overlap / max(core_rows, 1))
        return {
            "source": source,
            "path": path,
            "exists": bool(exists),
            "rows_total": int(len(frame)),
            "start_timestamp": str(ts.min()),
            "end_timestamp": str(ts.max()),
            "overlap_rows": overlap,
            "overlap_ratio_vs_core": overlap_ratio,
        }

    @staticmethod
    def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        ts = df["timestamp"]
        hour = ts.dt.hour
        dow = ts.dt.dayofweek

        df = df.copy()
        df["hour"] = hour
        df["day_of_week"] = dow
        df["is_weekend"] = (dow >= 5).astype(int)
        df["is_business_hour"] = ((hour >= 8) & (hour <= 18)).astype(int)
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        return df

    def _infer_sampling_and_horizon(self, df: pd.DataFrame) -> None:
        deltas = df["timestamp"].diff().dropna().dt.total_seconds().values
        if len(deltas) == 0:
            self.horizon_steps = 1
            self.seasonal_period_steps = 24
            return

        median_seconds = float(np.median(deltas))
        if median_seconds <= 0:
            median_seconds = 3600.0

        steps_per_hour = max(1, int(round(3600.0 / median_seconds)))
        proposed_h = self.config.forecast_horizon_hours * steps_per_hour

        # Keep horizon feasible with available data. If too large, degrade explicitly to 1-step.
        if proposed_h >= len(df) // 4:
            self.log.warning(
                "Requested horizon (%s steps) is too large for current sample size (%s). "
                "Falling back to 1-step forecasting.",
                proposed_h,
                len(df),
            )
            self.horizon_steps = 1
        else:
            self.horizon_steps = max(1, proposed_h)

        # Seasonal period uses one-day cadence when possible, capped for sparse short datasets.
        steps_per_day = max(2, int(round(86400.0 / median_seconds)))
        self.seasonal_period_steps = min(max(steps_per_day, 2), 24)

    def load_data(self) -> pd.DataFrame:
        """Load, validate, merge, and harden dataset inputs."""
        self._ensure_no_bfill()
        self.config.validate_required_files()

        cpu_path = self.config.dataset_path("cpu")
        temp_path = self.config.dataset_path("temperature")
        carbon_path = self.config.dataset_path("carbon_intensity")
        grid_path = self.config.dataset_path("grid_exog") if "grid_exog" in self.config.datasets else None
        weather_path = self.config.dataset_path("weather_exog") if "weather_exog" in self.config.datasets else None
        power_path = self.config.dataset_path("power_optional") if "power_optional" in self.config.datasets else None
        cluster_plus_path = (
            self.config.dataset_path("cluster_plus_power_optional")
            if "cluster_plus_power_optional" in self.config.datasets
            else None
        )
        pjm_demand_path = (
            self.config.dataset_path("pjm_hourly_demand_optional")
            if "pjm_hourly_demand_optional" in self.config.datasets
            else None
        )

        use_grid_exog = self._resolve_optional_usage("grid_exog", self.use_grid_exog_request)
        use_weather_exog = self._resolve_optional_usage("weather_exog", self.use_weather_exog_request)
        use_power_optional = self._resolve_optional_usage("power_optional", self.use_power_calibration_request)
        use_cluster_plus = self._resolve_optional_usage("cluster_plus_power_optional", None)
        use_pjm_demand = self._resolve_optional_usage("pjm_hourly_demand_optional", None)
        use_power_calibration = bool(use_power_optional)
        self.optional_dataset_usage = {
            "grid_exog": bool(use_grid_exog),
            "weather_exog": bool(use_weather_exog),
            "power_optional": bool(use_power_optional),
            "cluster_plus_power_optional": bool(use_cluster_plus),
            "pjm_hourly_demand_optional": bool(use_pjm_demand),
            "power_calibration": bool(use_power_calibration),
        }
        self.using_optional_features = bool(
            use_grid_exog or use_weather_exog or use_power_optional or use_cluster_plus or use_pjm_demand
        )

        self.log.info("Loading datasets from Data_Sources...")
        self.log.info("CPU: %s", cpu_path)
        self.log.info("Temperature: %s", temp_path)
        self.log.info("Carbon intensity: %s", carbon_path)
        if grid_path is not None:
            self.log.info("Grid exog (%s): %s", "enabled" if use_grid_exog else "disabled", grid_path)
        if weather_path is not None:
            self.log.info("Weather exog (%s): %s", "enabled" if use_weather_exog else "disabled", weather_path)
        if power_path is not None:
            self.log.info("Power optional (%s): %s", "enabled" if use_power_optional else "disabled", power_path)
        if cluster_plus_path is not None:
            self.log.info(
                "Cluster+Power optional (%s): %s",
                "enabled" if use_cluster_plus else "disabled",
                cluster_plus_path,
            )
        if pjm_demand_path is not None:
            self.log.info(
                "PJM hourly demand optional (%s): %s",
                "enabled" if use_pjm_demand else "disabled",
                pjm_demand_path,
            )

        cpu_df = pd.read_csv(cpu_path)
        temp_df = pd.read_csv(temp_path)
        carbon_df = pd.read_csv(carbon_path)

        validate_required_columns(cpu_df, "cpu")
        validate_required_columns(temp_df, "temperature")
        validate_required_columns(carbon_df, "carbon_intensity")

        cpu_df = cpu_df.copy()
        temp_df = temp_df.copy()
        carbon_df = carbon_df.copy()

        cpu_df["timestamp"] = self._parse_timestamp(cpu_df["real_timestamp"], "real_timestamp")
        temp_df["timestamp"] = self._parse_timestamp(temp_df["timestamp"], "timestamp")
        carbon_df["timestamp"] = self._parse_timestamp(carbon_df["timestamp"], "timestamp")

        # Deterministic aggregation for duplicate timestamps.
        cpu_df = (
            cpu_df.groupby("timestamp", as_index=False)
            .agg(
                avg_cpu_utilization=("avg_cpu_utilization", "mean"),
                num_tasks_sampled=("num_tasks_sampled", "sum"),
            )
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # External datasets are hourly; aggregate by floored hour deterministically.
        temp_df["hour_key"] = temp_df["timestamp"].dt.floor("h")
        carbon_df["hour_key"] = carbon_df["timestamp"].dt.floor("h")

        temp_hourly = (
            temp_df.groupby("hour_key", as_index=False)
            .agg(temperature_c=("temperature_c", "mean"))
            .rename(columns={"hour_key": "hour_key"})
        )
        carbon_hourly = (
            carbon_df.groupby("hour_key", as_index=False)
            .agg(carbon_intensity=("carbon_intensity_kg_per_mwh", "mean"))
            .rename(columns={"hour_key": "hour_key"})
        )

        cpu_df["hour_key"] = cpu_df["timestamp"].dt.floor("h")

        merged = cpu_df.merge(temp_hourly, on="hour_key", how="left")
        merged = merged.merge(carbon_hourly, on="hour_key", how="left")

        # Optional weather exogenous dataset.
        weather_df: Optional[pd.DataFrame] = None
        weather_hourly: Optional[pd.DataFrame] = None
        if use_weather_exog and weather_path is not None:
            weather_df = pd.read_csv(weather_path).copy()
            validate_optional_columns(weather_df, "weather_exog")
            if "timestamp" in weather_df.columns:
                weather_df["timestamp"] = self._parse_timestamp(weather_df["timestamp"], "timestamp")
            elif "DATE" in weather_df.columns:
                weather_df["timestamp"] = self._parse_timestamp(weather_df["DATE"], "DATE")
            else:
                raise ValueError("weather_exog dataset must include 'timestamp' or 'DATE'.")

            weather_df["hour_key"] = weather_df["timestamp"].dt.floor("h")
            weather_agg_map = {
                "dew_point_c": "mean",
                "wind_speed_mps": "mean",
                "sea_level_pressure_hpa": "mean",
                "visibility_m": "mean",
            }
            available_weather_cols = [c for c in weather_agg_map if c in weather_df.columns]
            if available_weather_cols:
                weather_hourly = (
                    weather_df.groupby("hour_key", as_index=False)
                    .agg({c: weather_agg_map[c] for c in available_weather_cols})
                    .rename(columns={"hour_key": "hour_key"})
                )
                merged = merged.merge(weather_hourly, on="hour_key", how="left")

        # Optional grid exogenous dataset for Stage 5.
        grid_df: Optional[pd.DataFrame] = None
        grid_hourly: Optional[pd.DataFrame] = None
        if use_grid_exog and grid_path is not None:
            grid_df = pd.read_csv(grid_path).copy()
            validate_optional_columns(grid_df, "grid_exog")
            if "timestamp" in grid_df.columns:
                grid_df["timestamp"] = self._parse_timestamp(grid_df["timestamp"], "timestamp")
            else:
                raise ValueError("grid_exog dataset must include 'timestamp'.")

            grid_df["hour_key"] = grid_df["timestamp"].dt.floor("h")
            grid_cols = [
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
            ]
            available_grid_cols = [c for c in grid_cols if c in grid_df.columns]
            if available_grid_cols:
                grid_hourly = (
                    grid_df.groupby("hour_key", as_index=False)
                    .agg({c: "mean" for c in available_grid_cols})
                    .rename(columns={"hour_key": "hour_key"})
                )
                merged = merged.merge(grid_hourly, on="hour_key", how="left")

        # Optional real power traces for Stage 1 features and physics calibration.
        power_df: Optional[pd.DataFrame] = None
        power_hourly: Optional[pd.DataFrame] = None
        if use_power_optional and power_path is not None:
            power_df = pd.read_csv(power_path).copy()
            validate_optional_columns(power_df, "power_optional")
            if "real_timestamp" in power_df.columns:
                power_df["timestamp"] = self._parse_timestamp(power_df["real_timestamp"], "real_timestamp")
            elif "timestamp" in power_df.columns:
                power_df["timestamp"] = self._parse_timestamp(power_df["timestamp"], "timestamp")
            else:
                raise ValueError("power_optional dataset must include 'real_timestamp' or 'timestamp'.")

            power_df["hour_key"] = power_df["timestamp"].dt.floor("h")
            power_cols = [c for c in ["measured_power_util", "production_power_util"] if c in power_df.columns]
            if power_cols:
                power_hourly = (
                    power_df.groupby("hour_key", as_index=False)
                    .agg({c: "mean" for c in power_cols})
                    .rename(columns={"hour_key": "hour_key"})
                )
                merged = merged.merge(power_hourly, on="hour_key", how="left")
                if "measured_power_util" in merged.columns:
                    merged["observed_power_util"] = merged["measured_power_util"]
                elif "production_power_util" in merged.columns:
                    merged["observed_power_util"] = merged["production_power_util"]

        # Optional joined cluster+power trace (can supply richer power/cpu covariates).
        cluster_plus_df: Optional[pd.DataFrame] = None
        if use_cluster_plus and cluster_plus_path is not None:
            cluster_plus_df = pd.read_csv(cluster_plus_path).copy()
            validate_optional_columns(cluster_plus_df, "cluster_plus_power_optional")
            if "real_timestamp" in cluster_plus_df.columns:
                cluster_plus_df["timestamp"] = self._parse_timestamp(cluster_plus_df["real_timestamp"], "real_timestamp")
            else:
                cluster_plus_df["timestamp"] = self._parse_timestamp(cluster_plus_df["timestamp"], "timestamp")
            cluster_plus_df["hour_key"] = cluster_plus_df["timestamp"].dt.floor("h")
            plus_cols = [
                c
                for c in ["measured_power_util", "production_power_util", "sample_count"]
                if c in cluster_plus_df.columns
            ]
            if plus_cols:
                plus_hourly = (
                    cluster_plus_df.groupby("hour_key", as_index=False)
                    .agg({c: "mean" for c in plus_cols})
                    .rename(columns={"hour_key": "hour_key"})
                )
                rename_map = {
                    "measured_power_util": "measured_power_util_plus",
                    "production_power_util": "production_power_util_plus",
                    "sample_count": "sample_count_plus",
                }
                plus_hourly = plus_hourly.rename(columns=rename_map)
                merged = merged.merge(plus_hourly, on="hour_key", how="left")
                if "observed_power_util" not in merged.columns:
                    if "measured_power_util_plus" in merged.columns:
                        merged["observed_power_util"] = merged["measured_power_util_plus"]
                    elif "production_power_util_plus" in merged.columns:
                        merged["observed_power_util"] = merged["production_power_util_plus"]

        # Optional demand-only PJM dataset (independent signal from the wide exog table).
        pjm_demand_df: Optional[pd.DataFrame] = None
        if use_pjm_demand and pjm_demand_path is not None:
            pjm_demand_df = pd.read_csv(pjm_demand_path).copy()
            validate_optional_columns(pjm_demand_df, "pjm_hourly_demand_optional")
            pjm_demand_df["timestamp"] = self._parse_timestamp(pjm_demand_df["datetime_utc"], "datetime_utc")
            pjm_demand_df["hour_key"] = pjm_demand_df["timestamp"].dt.floor("h")
            demand_hourly = (
                pjm_demand_df.groupby("hour_key", as_index=False)
                .agg(demand_mwh_raw=("demand_mwh", "mean"))
                .rename(columns={"hour_key": "hour_key"})
            )
            merged = merged.merge(demand_hourly, on="hour_key", how="left")

        merged = merged.drop(columns=["hour_key"])

        merged = merged.sort_values("timestamp").reset_index(drop=True)

        # Missing policy: forward-fill exogenous only within bounded gaps.
        exog_ffill_cols = ["temperature_c", "num_tasks_sampled"]
        for col in [
            "dew_point_c",
            "wind_speed_mps",
            "sea_level_pressure_hpa",
            "visibility_m",
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
            "observed_power_util",
            "measured_power_util_plus",
            "production_power_util_plus",
            "sample_count_plus",
            "demand_mwh_raw",
        ]:
            if col in merged.columns:
                exog_ffill_cols.append(col)

        for col in exog_ffill_cols:
            merged[col] = merged[col].ffill(limit=self.config.max_ffill_gap_hours)

        # Never backward fill targets.
        required_targets = ["avg_cpu_utilization", "carbon_intensity"]
        for col in required_targets:
            missing_ratio = float(merged[col].isna().mean())
            if missing_ratio > self.config.max_missing_ratio:
                raise ValueError(
                    f"Column '{col}' missing ratio {missing_ratio:.3f} exceeds "
                    f"max_missing_ratio={self.config.max_missing_ratio:.3f}."
                )

        merged = merged.dropna(subset=["avg_cpu_utilization", "carbon_intensity", "temperature_c"])

        merged["temperature_f"] = merged["temperature_c"] * 9.0 / 5.0 + 32.0
        if "dew_point_c" in merged.columns:
            merged["dew_point_f"] = merged["dew_point_c"] * 9.0 / 5.0 + 32.0
        merged["log_num_tasks"] = np.log1p(merged["num_tasks_sampled"].clip(lower=0))

        merged = self._add_time_features(merged)

        validate_merged_columns(merged)
        validate_monotonic_timestamp(merged, "timestamp")
        validate_numeric_ranges(merged)

        self._infer_sampling_and_horizon(merged)
        self.log.info(
            "Prepared %s rows. Horizon steps=%s Seasonal period=%s",
            len(merged),
            self.horizon_steps,
            self.seasonal_period_steps,
        )

        # Coverage report for all source datasets against core CPU window.
        core_start = cpu_df["timestamp"].min()
        core_end = cpu_df["timestamp"].max()
        core_rows = len(cpu_df)
        coverage_rows = [
            self._coverage_row(
                source="cpu",
                path=str(cpu_path),
                exists=True,
                frame=cpu_df[["timestamp"]].copy(),
                timestamp_col="timestamp",
                core_start=core_start,
                core_end=core_end,
                core_rows=core_rows,
            ),
            self._coverage_row(
                source="temperature",
                path=str(temp_path),
                exists=True,
                frame=temp_df[["timestamp"]].copy(),
                timestamp_col="timestamp",
                core_start=core_start,
                core_end=core_end,
                core_rows=core_rows,
            ),
            self._coverage_row(
                source="carbon_intensity",
                path=str(carbon_path),
                exists=True,
                frame=carbon_df[["timestamp"]].copy(),
                timestamp_col="timestamp",
                core_start=core_start,
                core_end=core_end,
                core_rows=core_rows,
            ),
            self._coverage_row(
                source="grid_exog",
                path=str(grid_path) if grid_path is not None else "",
                exists=bool(use_grid_exog and grid_path is not None and grid_path.exists()),
                frame=grid_df[["timestamp"]].copy() if grid_df is not None and "timestamp" in grid_df.columns else None,
                timestamp_col="timestamp",
                core_start=core_start,
                core_end=core_end,
                core_rows=core_rows,
            ),
            self._coverage_row(
                source="weather_exog",
                path=str(weather_path) if weather_path is not None else "",
                exists=bool(use_weather_exog and weather_path is not None and weather_path.exists()),
                frame=weather_df[["timestamp"]].copy() if weather_df is not None and "timestamp" in weather_df.columns else None,
                timestamp_col="timestamp",
                core_start=core_start,
                core_end=core_end,
                core_rows=core_rows,
            ),
            self._coverage_row(
                source="power_optional",
                path=str(power_path) if power_path is not None else "",
                exists=bool(use_power_optional and power_path is not None and power_path.exists()),
                frame=power_df[["timestamp"]].copy() if power_df is not None and "timestamp" in power_df.columns else None,
                timestamp_col="timestamp",
                core_start=core_start,
                core_end=core_end,
                core_rows=core_rows,
            ),
            self._coverage_row(
                source="cluster_plus_power_optional",
                path=str(cluster_plus_path) if cluster_plus_path is not None else "",
                exists=bool(use_cluster_plus and cluster_plus_path is not None and cluster_plus_path.exists()),
                frame=cluster_plus_df[["timestamp"]].copy()
                if cluster_plus_df is not None and "timestamp" in cluster_plus_df.columns
                else None,
                timestamp_col="timestamp",
                core_start=core_start,
                core_end=core_end,
                core_rows=core_rows,
            ),
            self._coverage_row(
                source="pjm_hourly_demand_optional",
                path=str(pjm_demand_path) if pjm_demand_path is not None else "",
                exists=bool(use_pjm_demand and pjm_demand_path is not None and pjm_demand_path.exists()),
                frame=pjm_demand_df[["timestamp"]].copy()
                if pjm_demand_df is not None and "timestamp" in pjm_demand_df.columns
                else None,
                timestamp_col="timestamp",
                core_start=core_start,
                core_end=core_end,
                core_rows=core_rows,
            ),
        ]
        self.coverage_report_df = pd.DataFrame(coverage_rows)
        if not self.coverage_report_df.empty:
            self.coverage_report_df["used_in_model"] = self.coverage_report_df["source"].map(
                {
                    "cpu": True,
                    "temperature": True,
                    "carbon_intensity": True,
                    "grid_exog": bool(use_grid_exog),
                    "weather_exog": bool(use_weather_exog),
                    "power_optional": bool(use_power_optional),
                    "cluster_plus_power_optional": bool(use_cluster_plus),
                    "pjm_hourly_demand_optional": bool(use_pjm_demand),
                }
            ).fillna(False)

        return merged

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def _build_cpu_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict[str, int], set]:
        dfx = df.copy()
        lag_map: Dict[str, int] = {}
        allow_zero: set = set()

        for lag in [1, 2, 3, 6, 12, 24, 48]:
            col = f"cpu_lag_{lag}"
            dfx[col] = dfx["avg_cpu_utilization"].shift(lag)
            lag_map[col] = lag

        shifted_cpu = dfx["avg_cpu_utilization"].shift(1)
        for window in [3, 6, 12, 24]:
            col_mean = f"cpu_roll_mean_{window}"
            col_std = f"cpu_roll_std_{window}"
            dfx[col_mean] = shifted_cpu.rolling(window, min_periods=window).mean()
            dfx[col_std] = shifted_cpu.rolling(window, min_periods=window).std()
            lag_map[col_mean] = 1
            lag_map[col_std] = 1

        dfx["cpu_ewm_12"] = shifted_cpu.ewm(span=12, adjust=False, min_periods=12).mean()
        lag_map["cpu_ewm_12"] = 1

        for lag in [1, 24]:
            col = f"tasks_lag_{lag}"
            dfx[col] = dfx["num_tasks_sampled"].shift(lag)
            lag_map[col] = lag

        if self.optional_dataset_usage.get("power_optional", False) and "observed_power_util" in dfx.columns:
            dfx["power_lag_1"] = dfx["observed_power_util"].shift(1)
            dfx["power_lag_3"] = dfx["observed_power_util"].shift(3)
            dfx["power_roll_mean_6"] = dfx["observed_power_util"].shift(1).rolling(6, min_periods=6).mean()
            lag_map["power_lag_1"] = 1
            lag_map["power_lag_3"] = 3
            lag_map["power_roll_mean_6"] = 1
        for base in ["measured_power_util_plus", "production_power_util_plus", "sample_count_plus"]:
            if base in dfx.columns:
                dfx[f"{base}_lag_1"] = dfx[base].shift(1)
                dfx[f"{base}_lag_3"] = dfx[base].shift(3)
                dfx[f"{base}_roll_mean_6"] = dfx[base].shift(1).rolling(6, min_periods=6).mean()
                lag_map[f"{base}_lag_1"] = 1
                lag_map[f"{base}_lag_3"] = 3
                lag_map[f"{base}_roll_mean_6"] = 1

        dfx["temperature_f_lag1"] = dfx["temperature_f"].shift(1)
        lag_map["temperature_f_lag1"] = 1

        time_cols = [
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "is_weekend",
            "is_business_hour",
        ]
        for col in time_cols:
            lag_map[col] = 0
            allow_zero.add(col)

        feature_cols = [
            c
            for c in [
                "cpu_lag_1",
                "cpu_lag_2",
                "cpu_lag_3",
                "cpu_lag_6",
                "cpu_lag_12",
                "cpu_lag_24",
                "cpu_lag_48",
                "cpu_roll_mean_3",
                "cpu_roll_mean_6",
                "cpu_roll_mean_12",
                "cpu_roll_mean_24",
                "cpu_roll_std_3",
                "cpu_roll_std_6",
                "cpu_roll_std_12",
                "cpu_roll_std_24",
                "cpu_ewm_12",
                "tasks_lag_1",
                "tasks_lag_24",
                "power_lag_1",
                "power_lag_3",
                "power_roll_mean_6",
                "measured_power_util_plus_lag_1",
                "measured_power_util_plus_lag_3",
                "measured_power_util_plus_roll_mean_6",
                "production_power_util_plus_lag_1",
                "production_power_util_plus_lag_3",
                "production_power_util_plus_roll_mean_6",
                "sample_count_plus_lag_1",
                "sample_count_plus_lag_3",
                "sample_count_plus_roll_mean_6",
                "temperature_f_lag1",
                "hour_sin",
                "hour_cos",
                "dow_sin",
                "dow_cos",
                "is_weekend",
                "is_business_hour",
            ]
            if c in dfx.columns
        ]

        return dfx, feature_cols, lag_map, allow_zero

    def _build_carbon_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict[str, int], set]:
        dfx = df.copy()
        lag_map: Dict[str, int] = {}
        allow_zero: set = set()

        for lag in [1, 2, 3, 6, 12, 24, 48]:
            col = f"ci_lag_{lag}"
            dfx[col] = dfx["carbon_intensity"].shift(lag)
            lag_map[col] = lag

        shifted_ci = dfx["carbon_intensity"].shift(1)
        for window in [3, 6, 12, 24]:
            col_mean = f"ci_roll_mean_{window}"
            col_std = f"ci_roll_std_{window}"
            dfx[col_mean] = shifted_ci.rolling(window, min_periods=window).mean()
            dfx[col_std] = shifted_ci.rolling(window, min_periods=window).std()
            lag_map[col_mean] = 1
            lag_map[col_std] = 1

        dfx["ci_ewm_12"] = shifted_ci.ewm(span=12, adjust=False, min_periods=12).mean()
        lag_map["ci_ewm_12"] = 1

        dfx["temperature_f_lag1_for_ci"] = dfx["temperature_f"].shift(1)
        lag_map["temperature_f_lag1_for_ci"] = 1

        grid_exog_cols = [
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
        ]
        if self.optional_dataset_usage.get("grid_exog", False):
            for base_col in [c for c in grid_exog_cols if c in dfx.columns]:
                for lag in [1, 3, 24]:
                    feat = f"{base_col}_lag_{lag}"
                    dfx[feat] = dfx[base_col].shift(lag)
                    lag_map[feat] = lag
                roll_feat = f"{base_col}_roll_mean_6"
                dfx[roll_feat] = dfx[base_col].shift(1).rolling(6, min_periods=6).mean()
                lag_map[roll_feat] = 1
        if "demand_mwh_raw" in dfx.columns:
            for lag in [1, 3, 24]:
                feat = f"demand_mwh_raw_lag_{lag}"
                dfx[feat] = dfx["demand_mwh_raw"].shift(lag)
                lag_map[feat] = lag
            dfx["demand_mwh_raw_roll_mean_6"] = dfx["demand_mwh_raw"].shift(1).rolling(6, min_periods=6).mean()
            lag_map["demand_mwh_raw_roll_mean_6"] = 1

        time_cols = [
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "is_weekend",
            "is_business_hour",
        ]
        for col in time_cols:
            lag_map[col] = 0
            allow_zero.add(col)

        feature_cols = [
            c
            for c in [
                "ci_lag_1",
                "ci_lag_2",
                "ci_lag_3",
                "ci_lag_6",
                "ci_lag_12",
                "ci_lag_24",
                "ci_lag_48",
                "ci_roll_mean_3",
                "ci_roll_mean_6",
                "ci_roll_mean_12",
                "ci_roll_mean_24",
                "ci_roll_std_3",
                "ci_roll_std_6",
                "ci_roll_std_12",
                "ci_roll_std_24",
                "ci_ewm_12",
                "temperature_f_lag1_for_ci",
                "demand_mwh_lag_1",
                "demand_mwh_lag_3",
                "demand_mwh_lag_24",
                "demand_mwh_roll_mean_6",
                "demand_mwh_raw_lag_1",
                "demand_mwh_raw_lag_3",
                "demand_mwh_raw_lag_24",
                "demand_mwh_raw_roll_mean_6",
                "demand_forecast_mwh_lag_1",
                "demand_forecast_mwh_lag_3",
                "demand_forecast_mwh_lag_24",
                "demand_forecast_mwh_roll_mean_6",
                "demand_forecast_error_mwh_lag_1",
                "demand_forecast_error_mwh_lag_3",
                "demand_forecast_error_mwh_lag_24",
                "demand_forecast_error_mwh_roll_mean_6",
                "interchange_mwh_lag_1",
                "interchange_mwh_lag_3",
                "interchange_mwh_lag_24",
                "interchange_mwh_roll_mean_6",
                "net_generation_mwh_lag_1",
                "net_generation_mwh_lag_3",
                "net_generation_mwh_lag_24",
                "net_generation_mwh_roll_mean_6",
                "fuel_col_share_lag_1",
                "fuel_col_share_lag_3",
                "fuel_col_share_lag_24",
                "fuel_col_share_roll_mean_6",
                "fuel_ng_share_lag_1",
                "fuel_ng_share_lag_3",
                "fuel_ng_share_lag_24",
                "fuel_ng_share_roll_mean_6",
                "fuel_nuc_share_lag_1",
                "fuel_nuc_share_lag_3",
                "fuel_nuc_share_lag_24",
                "fuel_nuc_share_roll_mean_6",
                "fuel_wnd_share_lag_1",
                "fuel_wnd_share_lag_3",
                "fuel_wnd_share_lag_24",
                "fuel_wnd_share_roll_mean_6",
                "fuel_sun_share_lag_1",
                "fuel_sun_share_lag_3",
                "fuel_sun_share_lag_24",
                "fuel_sun_share_roll_mean_6",
                "fuel_wat_share_lag_1",
                "fuel_wat_share_lag_3",
                "fuel_wat_share_lag_24",
                "fuel_wat_share_roll_mean_6",
                "fuel_oth_share_lag_1",
                "fuel_oth_share_lag_3",
                "fuel_oth_share_lag_24",
                "fuel_oth_share_roll_mean_6",
                "hour_sin",
                "hour_cos",
                "dow_sin",
                "dow_cos",
                "is_weekend",
                "is_business_hour",
            ]
            if c in dfx.columns
        ]

        return dfx, feature_cols, lag_map, allow_zero

    def _build_supervised_views(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        dfx, cpu_features, cpu_lag_map, cpu_allow_zero = self._build_cpu_features(df)
        dfx, ci_features, ci_lag_map, ci_allow_zero = self._build_carbon_features(dfx)

        horizon = self.horizon_steps
        seasonal_shift = max(1, self.seasonal_period_steps - horizon)

        dfx["target_cpu"] = dfx["avg_cpu_utilization"].shift(-horizon)
        dfx["target_ci"] = dfx["carbon_intensity"].shift(-horizon)
        dfx["temperature_f_t_plus_h"] = dfx["temperature_f"].shift(-horizon)
        if self.optional_dataset_usage.get("weather_exog", False) and "dew_point_f" in dfx.columns:
            dfx["dew_point_f_t_plus_h"] = dfx["dew_point_f"].shift(-horizon)
        if self.optional_dataset_usage.get("weather_exog", False) and "wind_speed_mps" in dfx.columns:
            dfx["wind_speed_mps_t_plus_h"] = dfx["wind_speed_mps"].shift(-horizon)
        if "observed_power_util" in dfx.columns:
            dfx["observed_power_util_t_plus_h"] = dfx["observed_power_util"].shift(-horizon)
            dfx["observed_total_power_mw_t_plus_h"] = dfx["observed_power_util_t_plus_h"] * float(self.config.facility_mw)

        dfx["baseline_cpu_persistence"] = dfx["avg_cpu_utilization"]
        dfx["baseline_ci_persistence"] = dfx["carbon_intensity"]

        dfx["baseline_cpu_seasonal"] = dfx["avg_cpu_utilization"].shift(seasonal_shift)
        dfx["baseline_ci_seasonal"] = dfx["carbon_intensity"].shift(seasonal_shift)

        cpu_cols = [
            "timestamp",
            "target_cpu",
            "baseline_cpu_persistence",
            "baseline_cpu_seasonal",
            "temperature_f_t_plus_h",
            "dew_point_f_t_plus_h",
            "wind_speed_mps_t_plus_h",
            "observed_power_util_t_plus_h",
            "observed_total_power_mw_t_plus_h",
        ] + cpu_features
        ci_cols = ["timestamp", "target_ci", "baseline_ci_persistence", "baseline_ci_seasonal"] + ci_features

        cpu_cols = [c for c in cpu_cols if c in dfx.columns]
        cpu_view = dfx[cpu_cols].dropna(subset=cpu_features + ["target_cpu"]).copy()
        ci_view = dfx[ci_cols].dropna(subset=ci_features + ["target_ci"]).copy()

        self.feature_cols["stage1_cpu"] = cpu_features
        self.feature_cols["stage5_ci"] = ci_features
        self.feature_lag_map["stage1_cpu"] = cpu_lag_map
        self.feature_lag_map["stage5_ci"] = ci_lag_map
        self.zero_lag_allow["stage1_cpu"] = cpu_allow_zero
        self.zero_lag_allow["stage5_ci"] = ci_allow_zero

        assert_strictly_past_features(cpu_lag_map, allow_zero_for=cpu_allow_zero)
        assert_strictly_past_features(ci_lag_map, allow_zero_for=ci_allow_zero)
        self.leakage_checks["strict_past_stage1"] = True
        self.leakage_checks["strict_past_stage5"] = True

        target_cols = [
            "timestamp",
            "target_cpu",
            "target_ci",
            "temperature_f_t_plus_h",
            "dew_point_f_t_plus_h",
            "wind_speed_mps_t_plus_h",
            "observed_power_util_t_plus_h",
            "observed_total_power_mw_t_plus_h",
            "baseline_cpu_persistence",
            "baseline_ci_persistence",
            "baseline_cpu_seasonal",
            "baseline_ci_seasonal",
        ]
        merged_targets = dfx[[c for c in target_cols if c in dfx.columns]].copy()

        return cpu_view, ci_view, merged_targets

    # ------------------------------------------------------------------
    # Physics stages
    # ------------------------------------------------------------------
    def _physics_it_power(self, cpu_util: np.ndarray | pd.Series) -> np.ndarray:
        cpu_util_arr = np.asarray(cpu_util, dtype=float)
        return self.config.facility_mw * (
            self.config.idle_power_fraction + (1.0 - self.config.idle_power_fraction) * cpu_util_arr
        )

    def _physics_pue(
        self,
        cpu_util: np.ndarray | pd.Series,
        temp_f: np.ndarray | pd.Series,
        dew_point_f: Optional[np.ndarray | pd.Series] = None,
        wind_speed_mps: Optional[np.ndarray | pd.Series] = None,
    ) -> np.ndarray:
        cpu_util_arr = np.asarray(cpu_util, dtype=float)
        temp_f_arr = np.asarray(temp_f, dtype=float)
        temp_above = np.maximum(0.0, temp_f_arr - self.config.cooling_threshold_f)
        pue = self.config.base_pue + self.config.pue_temp_coef * temp_above + self.config.pue_cpu_coef * cpu_util_arr
        if dew_point_f is not None:
            dew_arr = np.asarray(dew_point_f, dtype=float)
            dew_above = np.maximum(0.0, dew_arr - self.config.dew_point_threshold_f)
            pue = pue + self.config.pue_dewpoint_coef * dew_above
        if wind_speed_mps is not None:
            wind_arr = np.asarray(wind_speed_mps, dtype=float)
            pue = pue - self.config.pue_wind_coef * np.maximum(0.0, wind_arr)
        return np.clip(pue, self.config.base_pue, self.config.max_pue)

    def _set_identity_calibration(self, reason: str, calibration_valid: bool) -> None:
        self.physics_calibration = {
            "it_power_slope": 1.0,
            "it_power_intercept": 0.0,
            "total_power_slope": 1.0,
            "total_power_intercept": 0.0,
            "calibration_valid": bool(calibration_valid),
            "calibration_reason": str(reason),
        }
        self.calibration_valid = bool(calibration_valid)
        self.calibration_reason = str(reason)
        self.physics_calibration_df = pd.DataFrame([self.physics_calibration])

    @staticmethod
    def _resolve_stage6_target_mode(
        requested_mode: Optional[str],
        fallback_mode: str,
    ) -> str:
        mode = requested_mode or fallback_mode
        valid = {"physics", "observed_power_if_available"}
        if mode not in valid:
            raise ValueError(f"Invalid stage6_target_mode='{mode}'. Expected one of {sorted(valid)}.")
        return mode

    @staticmethod
    def _build_required_future_exog_columns(feature_cols_stage5: List[str]) -> List[str]:
        required = {"timestamp", "temperature_c"}
        for feat in feature_cols_stage5:
            if feat.startswith("ci_lag_") or feat.startswith("ci_roll_mean_") or feat.startswith("ci_roll_std_"):
                continue
            if feat in {
                "ci_ewm_12",
                "temperature_f_lag1_for_ci",
                "hour_sin",
                "hour_cos",
                "dow_sin",
                "dow_cos",
                "is_weekend",
                "is_business_hour",
            }:
                continue
            if "_lag_" in feat:
                required.add(feat.rsplit("_lag_", 1)[0])
            elif "_roll_mean_" in feat:
                required.add(feat.rsplit("_roll_mean_", 1)[0])
        return sorted(required)

    @staticmethod
    def _compute_sanity_checks(
        merged: pd.DataFrame,
        required_future_exog_columns: List[str],
    ) -> Dict[str, Any]:
        pred_cols = ["pred_cpu", "pred_ci", "pred_it_power", "pred_pue", "pred_total_power", "pred_emissions"]
        pred_cols = [c for c in pred_cols if c in merged.columns]

        nan_counts = {c: int(merged[c].isna().sum()) for c in pred_cols}
        inf_counts = {c: int(np.isinf(merged[c].to_numpy(dtype=float)).sum()) for c in pred_cols}
        negative_counts = {
            c: int((merged[c].to_numpy(dtype=float) < 0).sum())
            for c in ["pred_it_power", "pred_pue", "pred_total_power", "pred_emissions"]
            if c in merged.columns
        }

        residual_total = float(
            np.max(
                np.abs(
                    merged["pred_total_power"].to_numpy(dtype=float)
                    - merged["pred_it_power"].to_numpy(dtype=float) * merged["pred_pue"].to_numpy(dtype=float)
                )
            )
        )
        residual_emissions = float(
            np.max(
                np.abs(
                    merged["pred_emissions"].to_numpy(dtype=float)
                    - merged["pred_total_power"].to_numpy(dtype=float) * merged["pred_ci"].to_numpy(dtype=float)
                )
            )
        )

        passed = bool(
            all(v == 0 for v in nan_counts.values())
            and all(v == 0 for v in inf_counts.values())
            and all(v == 0 for v in negative_counts.values())
            and residual_total < 1e-8
            and residual_emissions < 1e-8
        )

        return {
            "nan_counts": nan_counts,
            "inf_counts": inf_counts,
            "negative_counts": negative_counts,
            "physics_identity_max_abs_error": {
                "total_power_vs_it_times_pue": residual_total,
                "emissions_vs_total_times_ci": residual_emissions,
            },
            "required_future_exog_columns": list(required_future_exog_columns),
            "predict_feature_contract_ready": bool(len(required_future_exog_columns) >= 2),
            "passed": passed,
        }

    @staticmethod
    def _fit_linear_calibration(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        x_arr = np.asarray(x, dtype=float).ravel()
        y_arr = np.asarray(y, dtype=float).ravel()
        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        x_arr = x_arr[mask]
        y_arr = y_arr[mask]
        if len(x_arr) < 20:
            return 1.0, 0.0
        x_design = np.column_stack([x_arr, np.ones(len(x_arr))])
        coef, *_ = np.linalg.lstsq(x_design, y_arr, rcond=None)
        slope = float(coef[0])
        intercept = float(coef[1])
        if not np.isfinite(slope):
            slope = 1.0
        if not np.isfinite(intercept):
            intercept = 0.0
        return slope, intercept

    def _fit_physics_calibration(self, cpu_view: pd.DataFrame, cpu_oof: np.ndarray) -> None:
        if not self.optional_dataset_usage.get("power_calibration", False):
            self._set_identity_calibration("disabled_identity", calibration_valid=True)
            return

        if "observed_total_power_mw_t_plus_h" not in cpu_view.columns:
            self._set_identity_calibration("missing_observed_power_target", calibration_valid=False)
            return

        mask = np.isfinite(cpu_oof) & np.isfinite(cpu_view["observed_total_power_mw_t_plus_h"].to_numpy(dtype=float))
        if not mask.any():
            self._set_identity_calibration("no_finite_overlap_for_calibration", calibration_valid=False)
            return

        pred_cpu = cpu_oof[mask]
        temp_f = cpu_view.loc[mask, "temperature_f_t_plus_h"].to_numpy(dtype=float)
        dew_f = (
            cpu_view.loc[mask, "dew_point_f_t_plus_h"].to_numpy(dtype=float)
            if "dew_point_f_t_plus_h" in cpu_view.columns
            else None
        )
        wind = (
            cpu_view.loc[mask, "wind_speed_mps_t_plus_h"].to_numpy(dtype=float)
            if "wind_speed_mps_t_plus_h" in cpu_view.columns
            else None
        )
        y_total = cpu_view.loc[mask, "observed_total_power_mw_t_plus_h"].to_numpy(dtype=float)

        pred_it = self._physics_it_power(pred_cpu)
        pred_pue = self._physics_pue(pred_cpu, temp_f, dew_f, wind)
        pred_total = pred_it * pred_pue

        q = float(self.config.power_calibration_trim_quantile)
        if 0.0 < q < 0.5 and len(pred_total) > 50:
            lo = np.nanquantile(y_total, q)
            hi = np.nanquantile(y_total, 1.0 - q)
            keep = np.isfinite(y_total) & (y_total >= lo) & (y_total <= hi)
            pred_it = pred_it[keep]
            pred_total = pred_total[keep]
            y_total = y_total[keep]
        if len(y_total) < 20:
            self._set_identity_calibration("insufficient_trimmed_samples", calibration_valid=False)
            return

        it_slope, it_intercept = self._fit_linear_calibration(pred_it, y_total)
        total_slope, total_intercept = self._fit_linear_calibration(pred_total, y_total)
        invalid_fit = (
            (not np.isfinite(it_slope))
            or (not np.isfinite(total_slope))
            or (it_slope < 0.0)
            or (total_slope < 0.0)
        )
        if invalid_fit:
            self._set_identity_calibration("nonphysical_negative_slope", calibration_valid=False)
            return

        self.physics_calibration = {
            "it_power_slope": float(it_slope),
            "it_power_intercept": float(it_intercept) if np.isfinite(it_intercept) else 0.0,
            "total_power_slope": float(total_slope),
            "total_power_intercept": float(total_intercept) if np.isfinite(total_intercept) else 0.0,
            "calibration_valid": True,
            "calibration_reason": "fitted",
        }
        self.calibration_valid = True
        self.calibration_reason = "fitted"
        self.physics_calibration_df = pd.DataFrame([self.physics_calibration])

    def _apply_physics_calibration(self, pred_it: np.ndarray, pred_total: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.optional_dataset_usage.get("power_calibration", False):
            return pred_it, pred_total

        it = (
            self.physics_calibration.get("it_power_slope", 1.0) * np.asarray(pred_it, dtype=float)
            + self.physics_calibration.get("it_power_intercept", 0.0)
        )
        total = (
            self.physics_calibration.get("total_power_slope", 1.0) * np.asarray(pred_total, dtype=float)
            + self.physics_calibration.get("total_power_intercept", 0.0)
        )
        it = np.clip(it, 0.0, self.config.facility_mw * self.config.max_pue)
        total = np.clip(total, 0.0, self.config.facility_mw * self.config.max_pue)
        return it, total

    # ------------------------------------------------------------------
    # Training and CV
    # ------------------------------------------------------------------
    @staticmethod
    def split_train_holdout(df: pd.DataFrame, train_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        split_idx = int(len(df) * train_ratio)
        if split_idx < 50 or (len(df) - split_idx) < 20:
            raise ValueError(
                f"Insufficient rows ({len(df)}) for robust split at ratio={train_ratio}. "
                "Need at least 50 train and 20 holdout rows."
            )
        train_df = df.iloc[:split_idx].copy()
        holdout_df = df.iloc[split_idx:].copy()
        return train_df, holdout_df

    def _select_params_and_train(
        self,
        frame: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        stage_name: str,
        param_grid: List[Dict[str, Any]],
        fit_fn,
        predict_fn,
    ) -> Tuple[Any, Dict[str, Any], pd.DataFrame, np.ndarray]:
        if len(frame) < (self.config.cv_splits + 1) * 20:
            raise ValueError(
                f"Not enough samples ({len(frame)}) for {self.config.cv_splits}-fold TimeSeriesSplit in {stage_name}."
            )

        tscv = TimeSeriesSplit(n_splits=self.config.cv_splits)
        best_params: Optional[Dict[str, Any]] = None
        best_rmse = float("inf")
        cv_rows: List[Dict[str, Any]] = []

        # Support both ParameterGrid-style dicts of lists and direct candidate dicts.
        candidates: List[Dict[str, Any]] = []
        for grid in param_grid:
            if not isinstance(grid, dict):
                raise TypeError(f"Invalid parameter grid entry for {stage_name}: {type(grid).__name__}")
            is_grid = True
            for value in grid.values():
                if isinstance(value, str) or not isinstance(value, (list, tuple, np.ndarray)):
                    is_grid = False
                    break
            if is_grid:
                for params in ParameterGrid([grid]):
                    candidates.append(dict(params))
            else:
                candidates.append(dict(grid))
        if not candidates:
            raise ValueError(f"No parameter candidates available for {stage_name}.")

        for param_idx, params in enumerate(candidates, start=1):
            fold_rmses: List[float] = []

            for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(frame), start=1):
                tr = frame.iloc[tr_idx]
                val = frame.iloc[val_idx]

                X_train = tr[feature_cols].to_numpy(dtype=float)
                y_train = tr[target_col].to_numpy(dtype=float)
                X_val = val[feature_cols].to_numpy(dtype=float)
                y_val = val[target_col].to_numpy(dtype=float)

                model, _ = fit_fn(X_train, y_train, X_val, y_val, params=params, save_artifacts=False)
                y_val_pred = predict_fn(model, X_val)
                metrics = calc_metrics(y_val, y_val_pred)
                fold_rmses.append(metrics["RMSE"])

                cv_rows.append(
                    {
                        "stage": stage_name,
                        "param_idx": param_idx,
                        "params": json.dumps(params, sort_keys=True),
                        "fold": fold_idx,
                        **metrics,
                    }
                )

            mean_rmse = float(np.mean(fold_rmses))
            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_params = dict(params)

        if best_params is None:
            raise RuntimeError(f"Parameter search failed for stage {stage_name}.")

        oof_pred = np.full(len(frame), np.nan, dtype=float)
        for tr_idx, val_idx in tscv.split(frame):
            tr = frame.iloc[tr_idx]
            val = frame.iloc[val_idx]

            X_train = tr[feature_cols].to_numpy(dtype=float)
            y_train = tr[target_col].to_numpy(dtype=float)
            X_val = val[feature_cols].to_numpy(dtype=float)
            y_val = val[target_col].to_numpy(dtype=float)

            model, _ = fit_fn(X_train, y_train, X_val, y_val, params=best_params, save_artifacts=False)
            oof_pred[val_idx] = predict_fn(model, X_val)

        val_size = max(24, int(0.1 * len(frame)))
        if len(frame) - val_size < 20:
            val_size = max(10, int(0.2 * len(frame)))

        train_main = frame.iloc[:-val_size]
        val_main = frame.iloc[-val_size:]

        X_train_main = train_main[feature_cols].to_numpy(dtype=float)
        y_train_main = train_main[target_col].to_numpy(dtype=float)
        X_val_main = val_main[feature_cols].to_numpy(dtype=float)
        y_val_main = val_main[target_col].to_numpy(dtype=float)

        final_model, _ = fit_fn(
            X_train_main,
            y_train_main,
            X_val_main,
            y_val_main,
            params=best_params,
            save_artifacts=False,
        )

        cv_df = pd.DataFrame(cv_rows)
        return final_model, best_params, cv_df, oof_pred

    def fit(self, df_train: pd.DataFrame) -> "CarbonForecastPipeline":
        """Fit Stage 1 and Stage 5 models using rolling CV and OOF diagnostics."""
        self.log.info("Fitting pipeline on training data...")

        df_train = df_train.copy().sort_values("timestamp").reset_index(drop=True)
        validate_monotonic_timestamp(df_train, "timestamp")
        validate_merged_columns(df_train)

        cpu_view, ci_view, _ = self._build_supervised_views(df_train)

        self.log.info("Training samples: Stage1=%s Stage5=%s", len(cpu_view), len(ci_view))

        cpu_model, cpu_params, cpu_cv_df, cpu_oof = self._select_params_and_train(
            cpu_view,
            self.feature_cols["stage1_cpu"],
            "target_cpu",
            "stage1_cpu",
            self.config.cpu_param_grid,
            fit_cpu_model,
            predict_cpu_model,
        )

        ci_model, ci_params, ci_cv_df, ci_oof = self._select_params_and_train(
            ci_view,
            self.feature_cols["stage5_ci"],
            "target_ci",
            "stage5_ci",
            self.config.carbon_param_grid,
            fit_ci_model,
            predict_ci_model,
        )

        self.models["stage1_cpu"] = cpu_model
        self.models["stage5_ci"] = ci_model
        self.best_params["stage1_cpu"] = cpu_params
        self.best_params["stage5_ci"] = ci_params

        self.cv_metrics_df = pd.concat([cpu_cv_df, ci_cv_df], ignore_index=True)

        max_lag = max(
            max(v for v in self.feature_lag_map["stage1_cpu"].values()),
            max(v for v in self.feature_lag_map["stage5_ci"].values()),
            self.seasonal_period_steps,
        )
        self.context_rows = max_lag + self.horizon_steps + 5
        self.train_context_df = df_train.tail(self.context_rows).copy()

        self.fitted = True

        cpu_oof_mask = np.isfinite(cpu_oof)
        ci_oof_mask = np.isfinite(ci_oof)
        if cpu_oof_mask.any():
            cpu_oof_metrics = calc_metrics(cpu_view.loc[cpu_oof_mask, "target_cpu"], cpu_oof[cpu_oof_mask])
            self.log.info("Stage1 OOF RMSE=%.6f R2=%.6f", cpu_oof_metrics["RMSE"], cpu_oof_metrics["R2"])
        if ci_oof_mask.any():
            ci_oof_metrics = calc_metrics(ci_view.loc[ci_oof_mask, "target_ci"], ci_oof[ci_oof_mask])
            self.log.info("Stage5 OOF RMSE=%.6f R2=%.6f", ci_oof_metrics["RMSE"], ci_oof_metrics["R2"])

        self._fit_physics_calibration(cpu_view, cpu_oof)
        if hasattr(self.models["stage5_ci"], "feature_importances_"):
            importances = np.asarray(getattr(self.models["stage5_ci"], "feature_importances_"), dtype=float)
            self.stage5_feature_importance_df = pd.DataFrame(
                {
                    "feature": self.feature_cols["stage5_ci"],
                    "importance": importances,
                }
            ).sort_values("importance", ascending=False).reset_index(drop=True)
        else:
            self.stage5_feature_importance_df = pd.DataFrame(columns=["feature", "importance"])

        self.required_future_exog_columns = self._build_required_future_exog_columns(
            self.feature_cols.get("stage5_ci", [])
        )

        return self

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _prepare_eval_views(self, df_holdout: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.train_context_df is None:
            raise RuntimeError("train_context_df missing. Fit the pipeline first.")

        holdout_sorted = df_holdout.copy().sort_values("timestamp").reset_index(drop=True)
        validate_monotonic_timestamp(holdout_sorted, "timestamp")

        combined = pd.concat([self.train_context_df, holdout_sorted], ignore_index=True)
        combined = combined.sort_values("timestamp").reset_index(drop=True)

        cpu_view, ci_view, _ = self._build_supervised_views(combined)

        holdout_ts = set(holdout_sorted["timestamp"])
        cpu_eval = cpu_view[cpu_view["timestamp"].isin(holdout_ts)].copy()
        ci_eval = ci_view[ci_view["timestamp"].isin(holdout_ts)].copy()

        if cpu_eval.empty or ci_eval.empty:
            raise ValueError(
                "No valid evaluation rows found after feature engineering. "
                "Check horizon, lag windows, and holdout length."
            )

        return cpu_eval, ci_eval

    def _stage_metric_row(
        self,
        stage: str,
        split: str,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        defensible: Optional[bool] = None,
        uplift_vs_persistence: Optional[float] = None,
    ) -> StageMetrics:
        metrics = calc_metrics(y_true, y_pred)
        ci = bootstrap_ci_for_metrics(
            y_true,
            y_pred,
            n_boot=self.config.bootstrap_iterations,
            block_size=self.config.bootstrap_block_size,
            alpha=self.config.bootstrap_alpha,
            seed=self.config.random_seed,
        )

        return StageMetrics(
            stage=stage,
            split=split,
            model=model_name,
            mae=metrics["MAE"],
            rmse=metrics["RMSE"],
            mape=metrics["MAPE"],
            smape=metrics["sMAPE"],
            r2=metrics["R2"],
            ci_mae_low=ci["MAE"][0],
            ci_mae_high=ci["MAE"][1],
            ci_rmse_low=ci["RMSE"][0],
            ci_rmse_high=ci["RMSE"][1],
            ci_mape_low=ci["MAPE"][0],
            ci_mape_high=ci["MAPE"][1],
            ci_smape_low=ci["sMAPE"][0],
            ci_smape_high=ci["sMAPE"][1],
            ci_r2_low=ci["R2"][0],
            ci_r2_high=ci["R2"][1],
            defensible=defensible,
            uplift_vs_persistence_rmse=uplift_vs_persistence,
        )

    def evaluate(
        self,
        df_holdout: pd.DataFrame,
        stage6_target_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate fitted pipeline on untouched holdout data."""
        if not self.fitted:
            raise RuntimeError("Pipeline must be fitted before evaluation.")

        cpu_eval, ci_eval = self._prepare_eval_views(df_holdout)

        X_cpu = cpu_eval[self.feature_cols["stage1_cpu"]].to_numpy(dtype=float)
        pred_cpu = predict_cpu_model(self.models["stage1_cpu"], X_cpu)

        X_ci = ci_eval[self.feature_cols["stage5_ci"]].to_numpy(dtype=float)
        pred_ci = predict_ci_model(self.models["stage5_ci"], X_ci)

        stage1_keep_cols = [
            "timestamp",
            "target_cpu",
            "baseline_cpu_persistence",
            "baseline_cpu_seasonal",
            "temperature_f_t_plus_h",
            "dew_point_f_t_plus_h",
            "wind_speed_mps_t_plus_h",
            "observed_total_power_mw_t_plus_h",
        ]
        stage1_df = cpu_eval[[c for c in stage1_keep_cols if c in cpu_eval.columns]].copy()
        stage1_df["pred_cpu"] = pred_cpu

        stage5_df = ci_eval[["timestamp", "target_ci", "baseline_ci_persistence", "baseline_ci_seasonal"]].copy()
        stage5_df["pred_ci"] = pred_ci

        merged = stage1_df.merge(stage5_df, on="timestamp", how="inner")
        merged = merged.sort_values("timestamp").reset_index(drop=True)

        dew = (
            merged["dew_point_f_t_plus_h"]
            if self.optional_dataset_usage.get("weather_exog", False) and "dew_point_f_t_plus_h" in merged.columns
            else None
        )
        wind = (
            merged["wind_speed_mps_t_plus_h"]
            if self.optional_dataset_usage.get("weather_exog", False) and "wind_speed_mps_t_plus_h" in merged.columns
            else None
        )

        merged["pred_it_power_raw"] = self._physics_it_power(merged["pred_cpu"])
        merged["actual_it_power"] = self._physics_it_power(merged["target_cpu"])

        merged["pred_pue"] = self._physics_pue(merged["pred_cpu"], merged["temperature_f_t_plus_h"], dew, wind)
        merged["actual_pue"] = self._physics_pue(merged["target_cpu"], merged["temperature_f_t_plus_h"], dew, wind)

        merged["pred_total_power_raw"] = merged["pred_it_power_raw"] * merged["pred_pue"]
        cal_it, cal_total = self._apply_physics_calibration(
            merged["pred_it_power_raw"].to_numpy(dtype=float),
            merged["pred_total_power_raw"].to_numpy(dtype=float),
        )
        merged["pred_it_power"] = cal_it
        merged["pred_total_power"] = cal_total
        merged["actual_total_power_physics"] = merged["actual_it_power"] * merged["actual_pue"]
        observed_total = np.full(len(merged), np.nan, dtype=float)
        if "observed_total_power_mw_t_plus_h" in merged.columns:
            observed_total = merged["observed_total_power_mw_t_plus_h"].to_numpy(dtype=float)
        merged["actual_total_power_observed"] = observed_total

        resolved_stage6_mode = self._resolve_stage6_target_mode(
            stage6_target_mode,
            self.config.stage6_target_mode_default,
        )
        if resolved_stage6_mode == "physics":
            merged["actual_total_power"] = merged["actual_total_power_physics"]
        else:
            merged["actual_total_power"] = merged["actual_total_power_physics"]
            obs_mask = np.isfinite(merged["actual_total_power_observed"].to_numpy(dtype=float))
            merged.loc[obs_mask, "actual_total_power"] = merged.loc[obs_mask, "actual_total_power_observed"]
        merged["pred_emissions"] = merged["pred_total_power"] * merged["pred_ci"]
        merged["actual_emissions"] = merged["actual_total_power"] * merged["target_ci"]

        # Baselines: stage-specific and propagated to emissions.
        merged["baseline_it_power_persistence"] = self._physics_it_power(merged["baseline_cpu_persistence"])
        merged["baseline_pue_persistence"] = self._physics_pue(
            merged["baseline_cpu_persistence"], merged["temperature_f_t_plus_h"], dew, wind
        )
        merged["baseline_total_power_persistence"] = (
            merged["baseline_it_power_persistence"] * merged["baseline_pue_persistence"]
        )
        _, baseline_total_cal = self._apply_physics_calibration(
            merged["baseline_it_power_persistence"].to_numpy(dtype=float),
            merged["baseline_total_power_persistence"].to_numpy(dtype=float),
        )
        merged["baseline_total_power_persistence"] = baseline_total_cal
        merged["baseline_emissions_persistence"] = (
            merged["baseline_total_power_persistence"] * merged["baseline_ci_persistence"]
        )

        merged["baseline_it_power_seasonal"] = self._physics_it_power(merged["baseline_cpu_seasonal"])
        merged["baseline_pue_seasonal"] = self._physics_pue(
            merged["baseline_cpu_seasonal"], merged["temperature_f_t_plus_h"], dew, wind
        )
        merged["baseline_total_power_seasonal"] = (
            merged["baseline_it_power_seasonal"] * merged["baseline_pue_seasonal"]
        )
        _, baseline_total_seasonal_cal = self._apply_physics_calibration(
            merged["baseline_it_power_seasonal"].to_numpy(dtype=float),
            merged["baseline_total_power_seasonal"].to_numpy(dtype=float),
        )
        merged["baseline_total_power_seasonal"] = baseline_total_seasonal_cal
        merged["baseline_emissions_seasonal"] = (
            merged["baseline_total_power_seasonal"] * merged["baseline_ci_seasonal"]
        )

        metric_rows: List[StageMetrics] = []
        baseline_rows: List[Dict[str, Any]] = []

        # Stage 1
        stage1_model_metrics = calc_metrics(merged["target_cpu"], merged["pred_cpu"])
        stage1_persist_metrics = calc_metrics(merged["target_cpu"], merged["baseline_cpu_persistence"])
        stage1_uplift = relative_rmse_improvement(stage1_persist_metrics["RMSE"], stage1_model_metrics["RMSE"])
        stage1_defensible = bool(np.isfinite(stage1_uplift) and stage1_uplift >= self.config.min_relative_rmse_improvement)

        metric_rows.append(
            self._stage_metric_row(
                "Stage1_CPU",
                "holdout",
                "RandomForest",
                merged["target_cpu"].to_numpy(),
                merged["pred_cpu"].to_numpy(),
                defensible=stage1_defensible,
                uplift_vs_persistence=stage1_uplift,
            )
        )

        # Stage 2/3/4 deterministic physics
        metric_rows.append(
            self._stage_metric_row(
                "Stage2_ITPower",
                "holdout",
                "Physics",
                merged["actual_it_power"].to_numpy(),
                merged["pred_it_power"].to_numpy(),
            )
        )
        metric_rows.append(
            self._stage_metric_row(
                "Stage3_PUE",
                "holdout",
                "Physics",
                merged["actual_pue"].to_numpy(),
                merged["pred_pue"].to_numpy(),
            )
        )
        metric_rows.append(
            self._stage_metric_row(
                "Stage4_TotalPower",
                "holdout",
                "Physics",
                merged["actual_total_power"].to_numpy(),
                merged["pred_total_power"].to_numpy(),
            )
        )

        # Stage 5
        stage5_model_metrics = calc_metrics(merged["target_ci"], merged["pred_ci"])
        stage5_persist_metrics = calc_metrics(merged["target_ci"], merged["baseline_ci_persistence"])
        stage5_uplift = relative_rmse_improvement(stage5_persist_metrics["RMSE"], stage5_model_metrics["RMSE"])
        stage5_defensible = bool(np.isfinite(stage5_uplift) and stage5_uplift >= self.config.min_relative_rmse_improvement)

        metric_rows.append(
            self._stage_metric_row(
                "Stage5_CarbonIntensity",
                "holdout",
                "XGBoost",
                merged["target_ci"].to_numpy(),
                merged["pred_ci"].to_numpy(),
                defensible=stage5_defensible,
                uplift_vs_persistence=stage5_uplift,
            )
        )

        # Stage 6
        stage6_model_metrics = calc_metrics(merged["actual_emissions"], merged["pred_emissions"])
        stage6_persist_metrics = calc_metrics(merged["actual_emissions"], merged["baseline_emissions_persistence"])
        stage6_uplift = relative_rmse_improvement(stage6_persist_metrics["RMSE"], stage6_model_metrics["RMSE"])
        stage6_defensible = bool(np.isfinite(stage6_uplift) and stage6_uplift >= self.config.min_relative_rmse_improvement)

        metric_rows.append(
            self._stage_metric_row(
                "Stage6_Emissions",
                "holdout",
                "PhysicsFromPredictions",
                merged["actual_emissions"].to_numpy(),
                merged["pred_emissions"].to_numpy(),
                defensible=stage6_defensible,
                uplift_vs_persistence=stage6_uplift,
            )
        )

        # Baseline comparisons.
        def _safe_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
            mask = np.isfinite(y_true.to_numpy(dtype=float)) & np.isfinite(y_pred.to_numpy(dtype=float))
            if not mask.any():
                return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "sMAPE": np.nan, "R2": np.nan}
            return calc_metrics(y_true.to_numpy(dtype=float)[mask], y_pred.to_numpy(dtype=float)[mask])

        baseline_rows.append(
            {
                "stage": "Stage1_CPU",
                "model_rmse": stage1_model_metrics["RMSE"],
                "persistence_rmse": stage1_persist_metrics["RMSE"],
                "seasonal_rmse": _safe_metrics(merged["target_cpu"], merged["baseline_cpu_seasonal"])["RMSE"],
                "uplift_vs_persistence": stage1_uplift,
                "defensible": stage1_defensible,
            }
        )
        baseline_rows.append(
            {
                "stage": "Stage5_CarbonIntensity",
                "model_rmse": stage5_model_metrics["RMSE"],
                "persistence_rmse": stage5_persist_metrics["RMSE"],
                "seasonal_rmse": _safe_metrics(merged["target_ci"], merged["baseline_ci_seasonal"])["RMSE"],
                "uplift_vs_persistence": stage5_uplift,
                "defensible": stage5_defensible,
            }
        )
        baseline_rows.append(
            {
                "stage": "Stage6_Emissions",
                "model_rmse": stage6_model_metrics["RMSE"],
                "persistence_rmse": stage6_persist_metrics["RMSE"],
                "seasonal_rmse": _safe_metrics(merged["actual_emissions"], merged["baseline_emissions_seasonal"])["RMSE"],
                "uplift_vs_persistence": stage6_uplift,
                "defensible": stage6_defensible,
            }
        )

        metrics_summary_df = pd.DataFrame([asdict(row) for row in metric_rows])
        baseline_comparison_df = pd.DataFrame(baseline_rows)

        stage1_predictions = merged[
            [
                "timestamp",
                "target_cpu",
                "pred_cpu",
                "baseline_cpu_persistence",
                "baseline_cpu_seasonal",
            ]
        ].rename(columns={"target_cpu": "actual_cpu"})

        stage5_predictions = merged[
            [
                "timestamp",
                "target_ci",
                "pred_ci",
                "baseline_ci_persistence",
                "baseline_ci_seasonal",
            ]
        ].rename(columns={"target_ci": "actual_ci"})

        stage6_cols = [
            "timestamp",
            "actual_emissions",
            "pred_emissions",
            "baseline_emissions_persistence",
            "baseline_emissions_seasonal",
            "actual_total_power",
            "actual_total_power_physics",
            "actual_total_power_observed",
            "pred_it_power",
            "pred_pue",
            "pred_total_power",
            "pred_ci",
            "temperature_f_t_plus_h",
            "observed_total_power_mw_t_plus_h",
        ]
        stage6_predictions = merged[[c for c in stage6_cols if c in merged.columns]].rename(
            columns={
                "pred_ci": "pred_carbon_intensity",
                "observed_total_power_mw_t_plus_h": "observed_total_power_mw",
                "actual_total_power_observed": "actual_total_power_observed_mw",
            }
        )

        self.sanity_checks = self._compute_sanity_checks(merged, self.required_future_exog_columns)

        metadata = {
            "run_id": self.run_id,
            "horizon_steps": self.horizon_steps,
            "seasonal_period_steps": self.seasonal_period_steps,
            "train_rows_context": len(self.train_context_df) if self.train_context_df is not None else 0,
            "holdout_rows_evaluated": len(merged),
            "best_params": self.best_params,
            "leakage_checks": self.leakage_checks,
            "optional_dataset_usage": self.optional_dataset_usage,
            "using_optional_features": self.using_optional_features,
            "selected_variant": self.selected_variant,
            "feature_set": {
                "stage1_cpu": list(self.feature_cols.get("stage1_cpu", [])),
                "stage5_ci": list(self.feature_cols.get("stage5_ci", [])),
            },
            "stage6_target_mode": resolved_stage6_mode,
            "stage6_target_mode_used_for_variant_gate": resolved_stage6_mode,
            "coverage_rows": int(len(self.coverage_report_df)),
            "physics_calibration": dict(self.physics_calibration),
            "calibration_valid": bool(self.calibration_valid),
            "calibration_reason": self.calibration_reason,
            "required_future_exog_columns": list(self.required_future_exog_columns),
            "sanity_checks_passed": bool(self.sanity_checks.get("passed", False)),
        }

        report = {
            "metrics_summary_df": metrics_summary_df,
            "metrics_cv_df": self.cv_metrics_df.copy() if self.cv_metrics_df is not None else pd.DataFrame(),
            "baseline_comparison_df": baseline_comparison_df,
            "stage1_predictions": stage1_predictions,
            "stage5_predictions": stage5_predictions,
            "stage6_predictions": stage6_predictions,
            "stage5_feature_importance_df": self.stage5_feature_importance_df.copy(),
            "stage1_physics_calibration_df": self.physics_calibration_df.copy(),
            "data_coverage_report_df": self.coverage_report_df.copy(),
            "sanity_checks": dict(self.sanity_checks),
            "metadata": metadata,
        }

        self.latest_report = report
        return report

    # ------------------------------------------------------------------
    # Iterative forecasting API
    # ------------------------------------------------------------------
    @staticmethod
    def _time_feature_dict(ts: pd.Timestamp) -> Dict[str, float]:
        hour = ts.hour
        dow = ts.dayofweek
        return {
            "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
            "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
            "dow_sin": float(np.sin(2 * np.pi * dow / 7)),
            "dow_cos": float(np.cos(2 * np.pi * dow / 7)),
            "is_weekend": float(int(dow >= 5)),
            "is_business_hour": float(int(8 <= hour <= 18)),
        }

    @staticmethod
    def _lag_from_name(name: str, prefix: str) -> Optional[int]:
        token = f"{prefix}_lag_"
        if name.startswith(token):
            return int(name.split("_")[-1])
        return None

    @staticmethod
    def _roll_window_from_name(name: str, prefix: str, kind: str) -> Optional[int]:
        token = f"{prefix}_roll_{kind}_"
        if name.startswith(token):
            return int(name.split("_")[-1])
        return None

    def _build_point_cpu_features(
        self,
        timestamp: pd.Timestamp,
        cpu_hist: List[float],
        task_hist: List[float],
        temp_hist_f: List[float],
        power_util_hist: Optional[List[float]] = None,
        cpu_exog_hist: Optional[Dict[str, List[float]]] = None,
    ) -> np.ndarray:
        values: Dict[str, float] = {}
        time_feats = self._time_feature_dict(timestamp)
        cpu_exog_hist = cpu_exog_hist or {}

        for name in self.feature_cols["stage1_cpu"]:
            lag = self._lag_from_name(name, "cpu")
            if lag is not None:
                values[name] = float(cpu_hist[-lag])
                continue

            roll_mean_w = self._roll_window_from_name(name, "cpu", "mean")
            if roll_mean_w is not None:
                values[name] = float(np.mean(cpu_hist[-roll_mean_w:]))
                continue

            roll_std_w = self._roll_window_from_name(name, "cpu", "std")
            if roll_std_w is not None:
                values[name] = float(np.std(cpu_hist[-roll_std_w:], ddof=1))
                continue

            if name == "cpu_ewm_12":
                values[name] = float(pd.Series(cpu_hist).ewm(span=12, adjust=False, min_periods=1).mean().iloc[-1])
                continue

            if name.startswith("tasks_lag_"):
                lag_t = int(name.split("_")[-1])
                values[name] = float(task_hist[-lag_t])
                continue

            if name.startswith("power_lag_"):
                if power_util_hist is None:
                    raise ValueError("power_util_hist is required for power lag features.")
                lag_p = int(name.split("_")[-1])
                values[name] = float(power_util_hist[-lag_p])
                continue

            if name.startswith("power_roll_mean_"):
                if power_util_hist is None:
                    raise ValueError("power_util_hist is required for power rolling features.")
                roll_p = int(name.split("_")[-1])
                values[name] = float(np.mean(power_util_hist[-roll_p:]))
                continue

            if name == "temperature_f_lag1":
                values[name] = float(temp_hist_f[-1])
                continue

            if "_lag_" in name:
                base_name, lag_token = name.rsplit("_lag_", 1)
                if base_name in cpu_exog_hist:
                    lag_x = int(lag_token)
                    values[name] = float(cpu_exog_hist[base_name][-lag_x])
                    continue

            if "_roll_mean_" in name:
                base_name, win_token = name.rsplit("_roll_mean_", 1)
                if base_name in cpu_exog_hist:
                    roll_w = int(win_token)
                    values[name] = float(np.mean(cpu_exog_hist[base_name][-roll_w:]))
                    continue

            if name in time_feats:
                values[name] = time_feats[name]
                continue

            raise KeyError(f"Unhandled CPU feature: {name}")

        vector = np.array([values[c] for c in self.feature_cols["stage1_cpu"]], dtype=float)
        if np.isnan(vector).any():
            raise ValueError("NaN in point CPU features during prediction.")
        return vector.reshape(1, -1)

    def _build_point_ci_features(
        self,
        timestamp: pd.Timestamp,
        ci_hist: List[float],
        temp_hist_f: List[float],
        exog_hist: Optional[Dict[str, List[float]]] = None,
    ) -> np.ndarray:
        values: Dict[str, float] = {}
        time_feats = self._time_feature_dict(timestamp)
        exog_hist = exog_hist or {}

        for name in self.feature_cols["stage5_ci"]:
            lag = self._lag_from_name(name, "ci")
            if lag is not None:
                values[name] = float(ci_hist[-lag])
                continue

            roll_mean_w = self._roll_window_from_name(name, "ci", "mean")
            if roll_mean_w is not None:
                values[name] = float(np.mean(ci_hist[-roll_mean_w:]))
                continue

            roll_std_w = self._roll_window_from_name(name, "ci", "std")
            if roll_std_w is not None:
                values[name] = float(np.std(ci_hist[-roll_std_w:], ddof=1))
                continue

            if name == "ci_ewm_12":
                values[name] = float(pd.Series(ci_hist).ewm(span=12, adjust=False, min_periods=1).mean().iloc[-1])
                continue

            if name == "temperature_f_lag1_for_ci":
                values[name] = float(temp_hist_f[-1])
                continue

            if name in time_feats:
                values[name] = time_feats[name]
                continue

            if "_lag_" in name:
                base_name, lag_token = name.rsplit("_lag_", 1)
                if base_name in exog_hist:
                    lag_x = int(lag_token)
                    values[name] = float(exog_hist[base_name][-lag_x])
                    continue

            if "_roll_mean_" in name:
                base_name, win_token = name.rsplit("_roll_mean_", 1)
                if base_name in exog_hist:
                    roll_w = int(win_token)
                    values[name] = float(np.mean(exog_hist[base_name][-roll_w:]))
                    continue

            raise KeyError(f"Unhandled CI feature: {name}")

        vector = np.array([values[c] for c in self.feature_cols["stage5_ci"]], dtype=float)
        if np.isnan(vector).any():
            raise ValueError("NaN in point carbon features during prediction.")
        return vector.reshape(1, -1)

    def predict(
        self,
        df_history: pd.DataFrame,
        future_exog_df: pd.DataFrame,
        horizon_hours: int = 1,
    ) -> pd.DataFrame:
        """Forecast recursively using history + required future exogenous inputs."""
        if not self.fitted:
            raise RuntimeError("Pipeline must be fitted before calling predict.")

        history = df_history.copy()
        if "timestamp" not in history.columns:
            raise ValueError("df_history must include timestamp.")

        if "avg_cpu_utilization" not in history.columns or "carbon_intensity" not in history.columns:
            raise ValueError("df_history must include avg_cpu_utilization and carbon_intensity.")

        if "temperature_c" not in history.columns:
            raise ValueError("df_history must include temperature_c.")

        if "num_tasks_sampled" not in history.columns:
            history["num_tasks_sampled"] = 0.0

        history["timestamp"] = self._parse_timestamp(history["timestamp"], "timestamp")
        history = history.sort_values("timestamp").reset_index(drop=True)

        future = future_exog_df.copy()
        future["timestamp"] = self._parse_timestamp(future["timestamp"], "timestamp")
        future = future.sort_values("timestamp").reset_index(drop=True)

        required_future_cols = list(self.required_future_exog_columns or ["timestamp", "temperature_c"])
        validate_future_exog_frame(future, required_columns=required_future_cols)

        horizon = int(horizon_hours)
        if horizon < 1:
            raise ValueError("horizon_hours must be >= 1")
        if len(future) < horizon:
            raise ValueError(
                f"future_exog_df has {len(future)} rows but horizon_hours={horizon}. "
                "Provide at least horizon_hours rows of future exogenous data."
            )

        max_required_lag = max(
            max(v for v in self.feature_lag_map["stage1_cpu"].values()),
            max(v for v in self.feature_lag_map["stage5_ci"].values()),
        )
        if len(history) < max_required_lag:
            raise ValueError(
                f"df_history has {len(history)} rows but at least {max_required_lag} rows are required "
                "for lag features."
            )

        cpu_hist = history["avg_cpu_utilization"].astype(float).tolist()
        ci_hist = history["carbon_intensity"].astype(float).tolist()
        task_hist = history["num_tasks_sampled"].astype(float).tolist()
        temp_hist_f = (history["temperature_c"].astype(float) * 9.0 / 5.0 + 32.0).tolist()
        if "observed_power_util" in history.columns:
            power_util_hist = history["observed_power_util"].astype(float).ffill().fillna(0.0).tolist()
        elif "measured_power_util" in history.columns:
            power_util_hist = history["measured_power_util"].astype(float).ffill().fillna(0.0).tolist()
        elif "production_power_util" in history.columns:
            power_util_hist = history["production_power_util"].astype(float).ffill().fillna(0.0).tolist()
        else:
            power_util_hist = (
                history["avg_cpu_utilization"].astype(float).clip(lower=0.0, upper=1.0).tolist()
            )

        cpu_exog_bases: set[str] = set()
        for feat in self.feature_cols.get("stage1_cpu", []):
            if feat.startswith("cpu_lag_") or feat.startswith("cpu_roll_mean_") or feat.startswith("cpu_roll_std_"):
                continue
            if feat in {
                "cpu_ewm_12",
                "temperature_f_lag1",
                "hour_sin",
                "hour_cos",
                "dow_sin",
                "dow_cos",
                "is_weekend",
                "is_business_hour",
            }:
                continue
            if feat.startswith("tasks_lag_") or feat.startswith("power_lag_") or feat.startswith("power_roll_mean_"):
                continue
            if "_lag_" in feat:
                cpu_exog_bases.add(feat.rsplit("_lag_", 1)[0])
            elif "_roll_mean_" in feat:
                cpu_exog_bases.add(feat.rsplit("_roll_mean_", 1)[0])

        cpu_exog_hist: Dict[str, List[float]] = {}
        for base in sorted(cpu_exog_bases):
            if base not in history.columns:
                raise ValueError(
                    "df_history is missing required Stage 1 exogenous columns. "
                    f"Missing in history: {base}"
                )
            hist_vals = history[base].astype(float).ffill().tolist()
            if pd.Series(hist_vals).isna().any():
                raise ValueError(
                    "df_history contains NaN values in required Stage 1 exogenous columns. "
                    f"Column: {base}"
                )
            cpu_exog_hist[base] = hist_vals

        ci_exog_bases: set[str] = set()
        for feat in self.feature_cols.get("stage5_ci", []):
            if feat.startswith("ci_lag_") or feat.startswith("ci_roll_mean_") or feat.startswith("ci_roll_std_"):
                continue
            if feat in {
                "ci_ewm_12",
                "temperature_f_lag1_for_ci",
                "hour_sin",
                "hour_cos",
                "dow_sin",
                "dow_cos",
                "is_weekend",
                "is_business_hour",
            }:
                continue
            if "_lag_" in feat:
                ci_exog_bases.add(feat.rsplit("_lag_", 1)[0])
            elif "_roll_mean_" in feat:
                ci_exog_bases.add(feat.rsplit("_roll_mean_", 1)[0])

        exog_hist: Dict[str, List[float]] = {}
        for base in sorted(ci_exog_bases):
            if base not in history.columns:
                raise ValueError(
                    "df_history is missing required exogenous columns for trained Stage 5 model. "
                    f"Missing in history: {base}"
                )
            hist_vals = history[base].astype(float).ffill().tolist()
            if pd.Series(hist_vals).isna().any():
                raise ValueError(
                    "df_history contains NaN values in required Stage 5 exogenous columns. "
                    f"Column: {base}"
                )
            exog_hist[base] = hist_vals

        rows: List[Dict[str, Any]] = []
        future_slice = future.iloc[:horizon].copy()

        if "num_tasks_sampled" not in future_slice.columns:
            future_slice["num_tasks_sampled"] = np.nan

        for step, row in enumerate(future_slice.itertuples(index=False), start=1):
            ts = row.timestamp
            temp_c = float(row.temperature_c)
            temp_f = float(temp_c * 9.0 / 5.0 + 32.0)

            future_tasks = float(row.num_tasks_sampled) if np.isfinite(row.num_tasks_sampled) else float(task_hist[-1])

            x_cpu = self._build_point_cpu_features(
                ts,
                cpu_hist,
                task_hist,
                temp_hist_f,
                power_util_hist=power_util_hist,
                cpu_exog_hist=cpu_exog_hist,
            )
            x_ci = self._build_point_ci_features(ts, ci_hist, temp_hist_f, exog_hist=exog_hist)

            pred_cpu = float(predict_cpu_model(self.models["stage1_cpu"], x_cpu)[0])
            pred_ci = float(predict_ci_model(self.models["stage5_ci"], x_ci)[0])

            pred_it = float(self._physics_it_power(np.array([pred_cpu]))[0])
            dew_f = None
            wind_v = None
            if hasattr(row, "dew_point_c") and np.isfinite(getattr(row, "dew_point_c")):
                dew_f = float(getattr(row, "dew_point_c") * 9.0 / 5.0 + 32.0)
            if hasattr(row, "wind_speed_mps") and np.isfinite(getattr(row, "wind_speed_mps")):
                wind_v = float(getattr(row, "wind_speed_mps"))
            pred_pue = float(
                self._physics_pue(
                    np.array([pred_cpu]),
                    np.array([temp_f]),
                    np.array([dew_f]) if dew_f is not None else None,
                    np.array([wind_v]) if wind_v is not None else None,
                )[0]
            )
            pred_total_raw = float(pred_it * pred_pue)
            pred_it_arr, pred_total_arr = self._apply_physics_calibration(
                np.array([pred_it], dtype=float),
                np.array([pred_total_raw], dtype=float),
            )
            pred_it = float(pred_it_arr[0])
            pred_total = float(pred_total_arr[0])
            pred_emissions = float(pred_total * pred_ci)

            rows.append(
                {
                    "timestamp": ts,
                    "horizon_step": step,
                    "pred_cpu": pred_cpu,
                    "pred_carbon_intensity": pred_ci,
                    "pred_it_power": pred_it,
                    "pred_pue": pred_pue,
                    "pred_total_power": pred_total,
                    "pred_emissions": pred_emissions,
                }
            )

            cpu_hist.append(pred_cpu)
            ci_hist.append(pred_ci)
            task_hist.append(future_tasks)
            temp_hist_f.append(temp_f)
            power_util_hist.append(float(pred_total / max(self.config.facility_mw, 1e-6)))
            for base in cpu_exog_bases:
                if base in future_slice.columns:
                    raw_value = getattr(row, base)
                    if not np.isfinite(raw_value):
                        raise ValueError(
                            "future_exog_df contains non-finite values in required Stage 1 exogenous columns. "
                            f"Column: {base}, timestamp: {ts}"
                        )
                    current = float(raw_value)
                else:
                    current = float(power_util_hist[-1]) if "power_util" in base else float(cpu_exog_hist[base][-1])
                cpu_exog_hist[base].append(current)
            for base in ci_exog_bases:
                raw_value = getattr(row, base)
                if not np.isfinite(raw_value):
                    raise ValueError(
                        "future_exog_df contains non-finite values in required exogenous columns for Stage 5. "
                        f"Column: {base}, timestamp: {ts}"
                    )
                current = float(raw_value)
                exog_hist[base].append(current)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Artifact writing and plots
    # ------------------------------------------------------------------
    @staticmethod
    def _plot_actual_vs_pred(df: pd.DataFrame, actual_col: str, pred_col: str, title: str, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df["timestamp"], df[actual_col], label="Actual", color="#1f77b4", linewidth=1.2)
        ax.plot(df["timestamp"], df[pred_col], label="Predicted", color="#d62728", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(alpha=0.3)
        save_figure(fig, path)

    @staticmethod
    def _plot_residual_hist(df: pd.DataFrame, actual_col: str, pred_col: str, title: str, path: Path) -> None:
        residual = df[actual_col] - df[pred_col]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(residual, bins=40, color="#9467bd", edgecolor="white", alpha=0.9)
        ax.axvline(0.0, color="red", linestyle="--", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Residual")
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.3)
        save_figure(fig, path)

    @staticmethod
    def _plot_calibration(df: pd.DataFrame, actual_col: str, pred_col: str, title: str, path: Path) -> None:
        work = df[[actual_col, pred_col]].copy()
        work = work.replace([np.inf, -np.inf], np.nan).dropna()
        if work.empty:
            return

        work["bin"] = pd.qcut(work[pred_col], q=min(10, len(work)), duplicates="drop")
        grouped = work.groupby("bin", observed=False).agg(actual=(actual_col, "mean"), pred=(pred_col, "mean"))

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(grouped["actual"], grouped["pred"], color="#2ca02c", s=50)
        mn = min(grouped["actual"].min(), grouped["pred"].min())
        mx = max(grouped["actual"].max(), grouped["pred"].max())
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=1.2)
        ax.set_xlabel("Mean Actual by Bin")
        ax.set_ylabel("Mean Predicted by Bin")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        save_figure(fig, path)

    @staticmethod
    def _plot_pipeline_diagram(metrics_df: pd.DataFrame, path: Path) -> None:
        stage_to_r2 = {}
        for _, row in metrics_df.iterrows():
            stage_to_r2[row["stage"]] = row["r2"]

        labels = [
            "Stage1_CPU",
            "Stage2_ITPower",
            "Stage3_PUE",
            "Stage4_TotalPower",
            "Stage5_CarbonIntensity",
            "Stage6_Emissions",
        ]

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.axis("off")

        y_positions = np.linspace(0.85, 0.15, len(labels))
        for y, label in zip(y_positions, labels):
            r2 = stage_to_r2.get(label, float("nan"))
            text = f"{label}\nR2={r2:.4f}" if np.isfinite(r2) else f"{label}\nR2=NA"
            rect = plt.Rectangle((0.2, y - 0.05), 0.6, 0.09, facecolor="#e8f1fa", edgecolor="black")
            ax.add_patch(rect)
            ax.text(0.5, y - 0.005, text, ha="center", va="center", fontsize=10, fontweight="bold")

        for y1, y2 in zip(y_positions[:-1], y_positions[1:]):
            ax.annotate(
                "",
                xy=(0.5, y2 + 0.04),
                xytext=(0.5, y1 - 0.05),
                arrowprops=dict(arrowstyle="->", linewidth=1.5),
            )

        ax.set_title("6-Stage Pipeline (Run-Computed Holdout Metrics)", fontsize=14, fontweight="bold")
        save_figure(fig, path)

    def _build_model_card(self, report: Dict[str, Any]) -> str:
        metrics_df = report["metrics_summary_df"]
        baseline_df = report["baseline_comparison_df"]
        metadata = report["metadata"]

        headline = metrics_df[metrics_df["stage"].isin(["Stage1_CPU", "Stage5_CarbonIntensity", "Stage6_Emissions"])]

        lines = []
        lines.append("# Model Card")
        lines.append("")
        lines.append(f"Run ID: {metadata['run_id']}")
        lines.append(f"Horizon steps: {metadata['horizon_steps']}")
        lines.append(f"Seasonal period steps: {metadata['seasonal_period_steps']}")
        lines.append(f"Selected variant: {metadata.get('selected_variant', 'baseline_core')}")
        lines.append(f"Stage6 target mode: {metadata.get('stage6_target_mode', 'unknown')}")
        lines.append(f"Calibration valid: {metadata.get('calibration_valid', False)}")
        lines.append("")
        negative_uplift_stages: List[str] = []
        if not baseline_df.empty and "uplift_vs_persistence" in baseline_df.columns:
            for _, row in baseline_df.iterrows():
                uplift_val = float(row["uplift_vs_persistence"])
                if np.isfinite(uplift_val) and uplift_val < 0.0:
                    negative_uplift_stages.append(str(row["stage"]))
        sanity_failed = not bool(metadata.get("sanity_checks_passed", False))
        if negative_uplift_stages or sanity_failed:
            lines.append("## Red Flags")
            if negative_uplift_stages:
                lines.append(
                    "- NOT DEFENSIBLE FOR FINAL CLAIMS: negative uplift vs persistence in "
                    + ", ".join(negative_uplift_stages)
                )
            if sanity_failed:
                lines.append("- Sanity checks failed; see sanity_checks.json for details.")
            lines.append("")
        lines.append("## Methodology")
        lines.append("- Stage 1 and Stage 5 are ML forecasters with rolling time-series CV.")
        lines.append("- Stages 2, 3, 4, and 6 are deterministic physics transformations.")
        lines.append("- Holdout metrics are computed on a strict final chronological split.")
        lines.append("")
        lines.append("## Leakage Guards")
        for key, passed in self.leakage_checks.items():
            lines.append(f"- {key}: {'PASS' if passed else 'FAIL'}")
        lines.append("")
        lines.append("## Optional Data Usage")
        optional_usage = metadata.get("optional_dataset_usage", {})
        if optional_usage:
            for key, used in optional_usage.items():
                lines.append(f"- {key}: {'ENABLED' if used else 'DISABLED'}")
        else:
            lines.append("- none")
        lines.append("")
        lines.append("## Headline Holdout Metrics")
        if headline.empty:
            lines.append("No headline metrics available.")
        else:
            for _, row in headline.iterrows():
                lines.append(
                    f"- {row['stage']}: RMSE={row['rmse']:.6f}, R2={row['r2']:.6f}, "
                    f"Defensible={row['defensible']}"
                )
        lines.append("")
        lines.append("## Baseline Comparison")
        if baseline_df.empty:
            lines.append("No baseline comparison available.")
        else:
            for _, row in baseline_df.iterrows():
                lines.append(
                    f"- {row['stage']}: model_rmse={row['model_rmse']:.6f}, "
                    f"persistence_rmse={row['persistence_rmse']:.6f}, "
                    f"uplift={row['uplift_vs_persistence']:.4f}"
                )
        lines.append("")
        lines.append("## Limitations")
        lines.append("- Accuracy depends on data quality and coverage of CPU + carbon traces.")
        lines.append("- If horizon fallback occurred due short sampling horizon feasibility, results are one-step forecasts.")

        return "\n".join(lines)

    def save_artifacts(self, report: Dict[str, Any]) -> RunArtifacts:
        """Persist reports, predictions, plots, models, and model card for one run."""
        result_dir = ensure_dir(self.config.results_dir / self.run_id)
        figure_dir = ensure_dir(self.config.figure_dir / self.run_id)
        model_dir = ensure_dir(self.config.model_dir / self.run_id)

        files: Dict[str, str] = {}

        files["metrics_summary"] = str(save_dataframe(report["metrics_summary_df"], result_dir / "metrics_summary.csv"))
        files["metrics_cv_folds"] = str(save_dataframe(report["metrics_cv_df"], result_dir / "metrics_cv_folds.csv"))
        files["stage1_predictions"] = str(
            save_dataframe(report["stage1_predictions"], result_dir / "stage1_cpu_holdout_predictions.csv")
        )
        files["stage5_predictions"] = str(
            save_dataframe(report["stage5_predictions"], result_dir / "stage5_carbon_holdout_predictions.csv")
        )
        files["stage6_predictions"] = str(
            save_dataframe(report["stage6_predictions"], result_dir / "stage6_emissions_holdout_predictions.csv")
        )
        files["baseline_comparison"] = str(
            save_dataframe(report["baseline_comparison_df"], result_dir / "baseline_comparison.csv")
        )
        if "variant_comparison_df" in report:
            files["variant_comparison"] = str(
                save_dataframe(report["variant_comparison_df"], result_dir / "variant_comparison.csv")
            )
        files["stage5_feature_importance"] = str(
            save_dataframe(report.get("stage5_feature_importance_df", pd.DataFrame()), result_dir / "stage5_feature_importance.csv")
        )
        files["stage1_physics_calibration"] = str(
            save_dataframe(report.get("stage1_physics_calibration_df", pd.DataFrame()), result_dir / "stage1_physics_calibration.csv")
        )
        files["data_coverage_report"] = str(
            save_dataframe(report.get("data_coverage_report_df", pd.DataFrame()), result_dir / "data_coverage_report.csv")
        )
        files["sanity_checks"] = str(
            save_json(report.get("sanity_checks", {}), result_dir / "sanity_checks.json")
        )
        files["run_metadata"] = str(save_json(report["metadata"], result_dir / "run_metadata.json"))

        files["stage1_model"] = str(save_pickle(self.models["stage1_cpu"], model_dir / "stage1_cpu_model.pkl"))
        files["stage5_model"] = str(save_pickle(self.models["stage5_ci"], model_dir / "stage5_carbon_model.pkl"))

        # Figures: actual vs pred
        self._plot_actual_vs_pred(
            report["stage1_predictions"],
            "actual_cpu",
            "pred_cpu",
            "Stage 1 CPU Holdout - Actual vs Predicted",
            figure_dir / "stage1_actual_vs_pred.png",
        )
        files["fig_stage1_actual_vs_pred"] = str(figure_dir / "stage1_actual_vs_pred.png")

        self._plot_actual_vs_pred(
            report["stage5_predictions"],
            "actual_ci",
            "pred_ci",
            "Stage 5 Carbon Intensity Holdout - Actual vs Predicted",
            figure_dir / "stage5_actual_vs_pred.png",
        )
        files["fig_stage5_actual_vs_pred"] = str(figure_dir / "stage5_actual_vs_pred.png")

        self._plot_actual_vs_pred(
            report["stage6_predictions"],
            "actual_emissions",
            "pred_emissions",
            "Stage 6 Emissions Holdout - Actual vs Predicted",
            figure_dir / "stage6_actual_vs_pred.png",
        )
        files["fig_stage6_actual_vs_pred"] = str(figure_dir / "stage6_actual_vs_pred.png")

        # Figures: residual diagnostics
        self._plot_residual_hist(
            report["stage1_predictions"],
            "actual_cpu",
            "pred_cpu",
            "Stage 1 CPU Residual Diagnostics",
            figure_dir / "stage1_residuals.png",
        )
        files["fig_stage1_residuals"] = str(figure_dir / "stage1_residuals.png")

        self._plot_residual_hist(
            report["stage5_predictions"],
            "actual_ci",
            "pred_ci",
            "Stage 5 Carbon Residual Diagnostics",
            figure_dir / "stage5_residuals.png",
        )
        files["fig_stage5_residuals"] = str(figure_dir / "stage5_residuals.png")

        self._plot_residual_hist(
            report["stage6_predictions"],
            "actual_emissions",
            "pred_emissions",
            "Stage 6 Emissions Residual Diagnostics",
            figure_dir / "stage6_residuals.png",
        )
        files["fig_stage6_residuals"] = str(figure_dir / "stage6_residuals.png")

        # Figures: calibration/reliability
        self._plot_calibration(
            report["stage1_predictions"],
            "actual_cpu",
            "pred_cpu",
            "Stage 1 CPU Reliability",
            figure_dir / "stage1_reliability.png",
        )
        files["fig_stage1_reliability"] = str(figure_dir / "stage1_reliability.png")

        self._plot_calibration(
            report["stage5_predictions"],
            "actual_ci",
            "pred_ci",
            "Stage 5 Carbon Reliability",
            figure_dir / "stage5_reliability.png",
        )
        files["fig_stage5_reliability"] = str(figure_dir / "stage5_reliability.png")

        self._plot_calibration(
            report["stage6_predictions"],
            "actual_emissions",
            "pred_emissions",
            "Stage 6 Emissions Reliability",
            figure_dir / "stage6_reliability.png",
        )
        files["fig_stage6_reliability"] = str(figure_dir / "stage6_reliability.png")

        # Dynamic pipeline diagram
        self._plot_pipeline_diagram(report["metrics_summary_df"], figure_dir / "pipeline_diagram_dynamic.png")
        files["fig_pipeline_diagram"] = str(figure_dir / "pipeline_diagram_dynamic.png")

        model_card = self._build_model_card(report)
        model_card_path = result_dir / "model_card.md"
        model_card_path.write_text(model_card, encoding="utf-8")
        files["model_card"] = str(model_card_path)

        artifacts = RunArtifacts(
            run_id=self.run_id,
            result_dir=str(result_dir),
            figure_dir=str(figure_dir),
            model_dir=str(model_dir),
            files=files,
        )
        return artifacts


# ----------------------------------------------------------------------
# CLI helpers
# ----------------------------------------------------------------------
def run_train_eval(pipeline: CarbonForecastPipeline) -> Tuple[CarbonForecastPipeline, Dict[str, Any], RunArtifacts]:
    data = pipeline.load_data()
    train_df, holdout_df = pipeline.split_train_holdout(data, pipeline.config.train_ratio)
    gate_mode = pipeline.config.stage6_variant_gate_target_mode

    pipeline.fit(train_df)
    enhanced_report = pipeline.evaluate(holdout_df, stage6_target_mode=gate_mode)
    enhanced_report["metadata"]["selected_variant"] = "exog_enhanced" if pipeline.using_optional_features else "baseline_core"
    enhanced_report["metadata"]["stage6_target_mode_used_for_variant_gate"] = gate_mode

    selected_pipeline = pipeline
    selected_report = enhanced_report

    if pipeline.using_optional_features:
        core_pipeline = CarbonForecastPipeline(
            pipeline.config,
            use_grid_exog=False,
            use_weather_exog=False,
            use_power_calibration=False,
        )
        core_pipeline.run_id = pipeline.run_id
        core_pipeline.horizon_steps = pipeline.horizon_steps
        core_pipeline.seasonal_period_steps = pipeline.seasonal_period_steps
        core_pipeline.coverage_report_df = pipeline.coverage_report_df.copy()
        if not core_pipeline.coverage_report_df.empty and "source" in core_pipeline.coverage_report_df.columns:
            core_pipeline.coverage_report_df["used_in_model"] = core_pipeline.coverage_report_df["source"].map(
                {
                    "cpu": True,
                    "temperature": True,
                    "carbon_intensity": True,
                    "grid_exog": False,
                    "weather_exog": False,
                    "power_optional": False,
                    "cluster_plus_power_optional": False,
                    "pjm_hourly_demand_optional": False,
                }
            ).fillna(False)
        core_pipeline.leakage_checks["no_bfill_in_source"] = pipeline.leakage_checks.get("no_bfill_in_source", False)
        core_pipeline.fit(train_df)
        core_report = core_pipeline.evaluate(holdout_df, stage6_target_mode=gate_mode)
        core_report["metadata"]["selected_variant"] = "baseline_core"
        core_report["metadata"]["stage6_target_mode_used_for_variant_gate"] = gate_mode

        def _stage_rmse(rep: Dict[str, Any], stage: str) -> float:
            row = rep["metrics_summary_df"].loc[rep["metrics_summary_df"]["stage"] == stage, "rmse"]
            if row.empty:
                return float("nan")
            return float(row.iloc[0])

        rmse_core_s5 = _stage_rmse(core_report, "Stage5_CarbonIntensity")
        rmse_core_s6 = _stage_rmse(core_report, "Stage6_Emissions")
        rmse_enh_s5 = _stage_rmse(enhanced_report, "Stage5_CarbonIntensity")
        rmse_enh_s6 = _stage_rmse(enhanced_report, "Stage6_Emissions")

        uplift_s5 = relative_rmse_improvement(rmse_core_s5, rmse_enh_s5)
        uplift_s6 = relative_rmse_improvement(rmse_core_s6, rmse_enh_s6)
        passes_leakage = all(bool(v) for v in pipeline.leakage_checks.values())
        enhanced_promoted = bool(
            np.isfinite(uplift_s5)
            and np.isfinite(uplift_s6)
            and uplift_s5 >= pipeline.config.min_relative_rmse_improvement
            and uplift_s6 >= -float(pipeline.config.max_stage6_rmse_degradation)
            and passes_leakage
        )

        variant_comparison = pd.DataFrame(
            [
                {
                    "variant": "baseline_core",
                    "stage5_rmse": rmse_core_s5,
                    "stage6_rmse": rmse_core_s6,
                },
                {
                    "variant": "exog_enhanced",
                    "stage5_rmse": rmse_enh_s5,
                    "stage6_rmse": rmse_enh_s6,
                },
            ]
        )

        enhanced_report["metadata"]["variant_comparison"] = {
            "baseline_core_stage5_rmse": rmse_core_s5,
            "baseline_core_stage6_rmse": rmse_core_s6,
            "exog_enhanced_stage5_rmse": rmse_enh_s5,
            "exog_enhanced_stage6_rmse": rmse_enh_s6,
            "uplift_stage5_vs_core": uplift_s5,
            "uplift_stage6_vs_core": uplift_s6,
            "enhanced_promoted": enhanced_promoted,
            "passes_leakage_checks": passes_leakage,
            "stage6_target_mode_used_for_variant_gate": gate_mode,
        }
        enhanced_report["variant_comparison_df"] = variant_comparison

        core_report["metadata"]["variant_comparison"] = enhanced_report["metadata"]["variant_comparison"]
        core_report["variant_comparison_df"] = variant_comparison

        if enhanced_promoted:
            selected_pipeline = pipeline
            selected_pipeline.selected_variant = "exog_enhanced"
            selected_report = enhanced_report
            selected_report["metadata"]["selected_variant"] = "exog_enhanced"
        else:
            selected_pipeline = core_pipeline
            selected_pipeline.selected_variant = "baseline_core"
            selected_report = core_report
            selected_report["metadata"]["selected_variant"] = "baseline_core"

    selected_report["metadata"]["stage6_target_mode_used_for_variant_gate"] = gate_mode
    artifacts = selected_pipeline.save_artifacts(selected_report)

    headline = selected_report["metrics_summary_df"][
        selected_report["metrics_summary_df"]["stage"].isin(["Stage1_CPU", "Stage5_CarbonIntensity", "Stage6_Emissions"])
    ]
    for _, row in headline.iterrows():
        selected_pipeline.log.info(
            "%-24s RMSE=%10.6f R2=%8.6f Defensible=%s",
            row["stage"],
            row["rmse"],
            row["r2"],
            row["defensible"],
        )

    selected_pipeline.log.info("Selected variant: %s", selected_report["metadata"].get("selected_variant", "baseline_core"))
    selected_pipeline.log.info("Artifacts written to: %s", artifacts.result_dir)
    return selected_pipeline, selected_report, artifacts


def run_validate_inputs(pipeline: CarbonForecastPipeline) -> None:
    data = pipeline.load_data()
    cpu_view, ci_view, _ = pipeline._build_supervised_views(data)
    pipeline.log.info("Input validation complete. Rows=%s CPU-supervised=%s CI-supervised=%s", len(data), len(cpu_view), len(ci_view))


def run_predict_mode(pipeline: CarbonForecastPipeline, horizon: int) -> pd.DataFrame:
    data = pipeline.load_data()
    train_df, holdout_df = pipeline.split_train_holdout(data, pipeline.config.train_ratio)
    pipeline.fit(train_df)

    required_cols = list(dict.fromkeys((pipeline.required_future_exog_columns or ["timestamp", "temperature_c"]) + ["num_tasks_sampled"]))
    future_exog = holdout_df[[c for c in required_cols if c in holdout_df.columns]].head(horizon).copy()
    pred_df = pipeline.predict(train_df, future_exog, horizon_hours=horizon)

    result_dir = ensure_dir(pipeline.config.results_dir / pipeline.run_id)
    save_dataframe(pred_df, result_dir / "predict_mode_output.csv")
    pipeline.log.info("Saved predict output: %s", result_dir / "predict_mode_output.csv")

    return pred_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Leak-free carbon forecasting pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        default="train_eval",
        choices=["train_eval", "predict", "validate_inputs"],
        help="Execution mode",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="Prediction horizon for --mode predict",
    )
    parser.add_argument(
        "--use-grid-exog",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="Use grid exogenous dataset if available",
    )
    parser.add_argument(
        "--use-weather-exog",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="Use weather exogenous dataset if available",
    )
    parser.add_argument(
        "--use-power-calibration",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="Use optional power data for Stage1 features and physics calibration",
    )
    return parser


def _parse_optional_bool(raw: str) -> Optional[bool]:
    value = str(raw).strip().lower()
    if value == "auto":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"Unsupported boolean flag value: {raw}")


def main() -> None:
    args = build_arg_parser().parse_args()
    config = make_config()
    pipeline = CarbonForecastPipeline(
        config,
        use_grid_exog=_parse_optional_bool(args.use_grid_exog),
        use_weather_exog=_parse_optional_bool(args.use_weather_exog),
        use_power_calibration=_parse_optional_bool(args.use_power_calibration),
    )

    if args.mode == "train_eval":
        run_train_eval(pipeline)
    elif args.mode == "predict":
        run_predict_mode(pipeline, horizon=args.horizon)
    elif args.mode == "validate_inputs":
        run_validate_inputs(pipeline)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
