"""Configuration for the REAL FINAL FILES carbon forecasting pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class PipelineConfig:
    """Central configuration for paths, data, model params, and evaluation."""

    base_dir: Path | None = None
    project_dir: Path | None = None
    data_dir: Path | None = None
    output_dir: Path | None = None

    datasets: Dict[str, str] = field(
        default_factory=lambda: {
            "cpu": "cleaned/google_cluster_utilization_2019_cellb_hourly_cleaned.csv",
            "temperature": "cleaned/ashburn_va_temperature_2019_cleaned.csv",
            "carbon_intensity": "cleaned/pjm_grid_carbon_intensity_2019_full_cleaned.csv",
            "grid_exog": "cleaned/pjm_exogenous_hourly_2019_2024_cleaned.csv",
            "weather_exog": "cleaned/noaa_global_hourly_dulles_2019_2024_cleaned.csv",
            "power_optional": "cleaned/google_power_utilization_2019_cellb_hourly.csv",
            "cluster_plus_power_optional": "cleaned/google_cluster_plus_power_2019_cellb_hourly.csv",
            "pjm_hourly_demand_optional": "cleaned/pjm_hourly_demand_2019_2024_cleaned.csv",
        }
    )
    required_dataset_keys: List[str] = field(
        default_factory=lambda: ["cpu", "temperature", "carbon_intensity"]
    )

    random_seed: int = 42
    forecast_horizon_hours: int = 1
    train_ratio: float = 0.85
    cv_splits: int = 5

    max_ffill_gap_hours: int = 3
    max_missing_ratio: float = 0.05

    min_relative_rmse_improvement: float = 0.01
    max_stage6_rmse_degradation: float = 0.0
    stage6_target_mode_default: str = "observed_power_if_available"
    stage6_variant_gate_target_mode: str = "observed_power_if_available"

    bootstrap_iterations: int = 400
    bootstrap_block_size: int = 24
    bootstrap_alpha: float = 0.05

    cpu_param_grid: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 2, "max_features": "sqrt"},
            {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 2, "max_features": "sqrt"},
            {"n_estimators": 400, "max_depth": 16, "min_samples_leaf": 1, "max_features": "sqrt"},
        ]
    )

    carbon_param_grid: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
            },
            {
                "n_estimators": 500,
                "max_depth": 5,
                "learning_rate": 0.03,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_alpha": 0.01,
                "reg_lambda": 1.0,
            },
            {
                "n_estimators": 700,
                "max_depth": 6,
                "learning_rate": 0.02,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_alpha": 0.01,
                "reg_lambda": 2.0,
            },
        ]
    )

    facility_mw: float = 100.0
    idle_power_fraction: float = 0.30
    cooling_threshold_f: float = 65.0
    base_pue: float = 1.10
    max_pue: float = 2.00
    pue_temp_coef: float = 0.012
    pue_cpu_coef: float = 0.050
    dew_point_threshold_f: float = 60.0
    pue_dewpoint_coef: float = 0.001
    pue_wind_coef: float = 0.002
    power_calibration_trim_quantile: float = 0.01

    def __post_init__(self) -> None:
        if self.base_dir is None:
            self.base_dir = Path(__file__).resolve().parent
        if self.project_dir is None:
            self.project_dir = self.base_dir.parent
        if self.data_dir is None:
            self.data_dir = self.project_dir / "Data_Sources"
        if self.output_dir is None:
            self.output_dir = self.base_dir / "outputs"

        self.figure_dir = self.output_dir / "figures"
        self.model_dir = self.output_dir / "models"
        self.results_dir = self.output_dir / "results"

        for directory in [self.output_dir, self.figure_dir, self.model_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        if not (0.0 < self.train_ratio < 1.0):
            raise ValueError("train_ratio must be between 0 and 1.")
        if self.forecast_horizon_hours < 1:
            raise ValueError("forecast_horizon_hours must be >= 1.")
        if self.cv_splits < 2:
            raise ValueError("cv_splits must be >= 2.")
        valid_stage6_modes = {"physics", "observed_power_if_available"}
        if self.stage6_target_mode_default not in valid_stage6_modes:
            raise ValueError(
                f"stage6_target_mode_default must be one of {sorted(valid_stage6_modes)}."
            )
        if self.stage6_variant_gate_target_mode not in valid_stage6_modes:
            raise ValueError(
                f"stage6_variant_gate_target_mode must be one of {sorted(valid_stage6_modes)}."
            )

    def dataset_path(self, key: str) -> Path:
        if key not in self.datasets:
            valid = ", ".join(sorted(self.datasets.keys()))
            raise KeyError(f"Unknown dataset key '{key}'. Valid keys: {valid}")
        return self.data_dir / self.datasets[key]

    def validate_required_files(self) -> None:
        missing = []
        for key in self.required_dataset_keys:
            if key not in self.datasets:
                missing.append((key, Path(f"<missing dataset map entry for key '{key}'>")))
                continue
            path = self.dataset_path(key)
            if not path.exists():
                missing.append((key, path))

        if missing:
            lines = ["Missing required dataset files:"]
            for key, path in missing:
                lines.append(f"- {key}: {path}")
            lines.append(
                "\nAction: provide these exact files under Data_Sources or update PipelineConfig.datasets "
                "to valid files."
            )
            raise FileNotFoundError("\n".join(lines))

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["base_dir"] = str(self.base_dir)
        data["project_dir"] = str(self.project_dir)
        data["data_dir"] = str(self.data_dir)
        data["output_dir"] = str(self.output_dir)
        data["figure_dir"] = str(self.figure_dir)
        data["model_dir"] = str(self.model_dir)
        data["results_dir"] = str(self.results_dir)
        return data


def make_config() -> PipelineConfig:
    """Factory for default pipeline config."""
    return PipelineConfig()
