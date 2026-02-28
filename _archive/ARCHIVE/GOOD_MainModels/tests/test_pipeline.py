from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carbon_prediction_pipeline import CarbonForecastPipeline, run_train_eval
from config import PipelineConfig
from schema import validate_required_columns


def _make_synthetic_environment(tmp_path: Path, include_optional: bool = False) -> PipelineConfig:
    data_root = tmp_path / "Data_Sources"
    cleaned = data_root / "cleaned"
    cleaned.mkdir(parents=True, exist_ok=True)

    n = 500
    ts = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")

    rng = np.random.default_rng(123)
    cpu = 0.55 + 0.15 * np.sin(np.arange(n) * 2 * np.pi / 24) + rng.normal(0, 0.02, n)
    cpu = np.clip(cpu, 0.05, 0.98)
    tasks = 1000 + 120 * np.sin(np.arange(n) * 2 * np.pi / 12) + rng.normal(0, 25, n)
    tasks = np.clip(tasks, 10, None)

    temp_c = 12 + 10 * np.sin(np.arange(n) * 2 * np.pi / (24 * 7)) + rng.normal(0, 0.8, n)
    carbon = 350 + 25 * np.sin(np.arange(n) * 2 * np.pi / 24) + rng.normal(0, 8, n)
    carbon = np.clip(carbon, 150, 900)

    cpu_df = pd.DataFrame(
        {
            "real_timestamp": ts.strftime("%Y-%m-%d %H:%M:%S+00:00"),
            "hour_of_day": ts.hour,
            "avg_cpu_utilization": cpu,
            "num_tasks_sampled": tasks,
        }
    )
    temp_df = pd.DataFrame({"timestamp": ts.tz_convert("UTC").tz_localize(None), "temperature_c": temp_c})
    carbon_df = pd.DataFrame(
        {"timestamp": ts.tz_convert("UTC").tz_localize(None), "carbon_intensity_kg_per_mwh": carbon}
    )

    cpu_df.to_csv(cleaned / "google_cluster_utilization_2019_cleaned.csv", index=False)
    temp_df.to_csv(cleaned / "ashburn_va_temperature_2019_cleaned.csv", index=False)
    carbon_df.to_csv(cleaned / "pjm_grid_carbon_intensity_2019_full_cleaned.csv", index=False)

    datasets = {
        "cpu": "cleaned/google_cluster_utilization_2019_cleaned.csv",
        "temperature": "cleaned/ashburn_va_temperature_2019_cleaned.csv",
        "carbon_intensity": "cleaned/pjm_grid_carbon_intensity_2019_full_cleaned.csv",
    }
    if include_optional:
        demand = 10000 + 600 * np.sin(np.arange(n) * 2 * np.pi / 24) + rng.normal(0, 50, n)
        fuel_ng = np.clip(0.45 + 0.08 * np.sin(np.arange(n) * 2 * np.pi / 48), 0.05, 0.85)
        fuel_col = np.clip(0.18 + 0.03 * np.cos(np.arange(n) * 2 * np.pi / 72), 0.01, 0.45)
        fuel_nuc = np.clip(0.22 + 0.02 * np.sin(np.arange(n) * 2 * np.pi / 96), 0.05, 0.45)
        fuel_wnd = np.clip(0.07 + 0.02 * np.cos(np.arange(n) * 2 * np.pi / 36), 0.0, 0.25)
        fuel_sun = np.clip(0.04 + 0.015 * np.maximum(0, np.sin(np.arange(n) * 2 * np.pi / 24)), 0.0, 0.2)
        fuel_wat = np.clip(0.03 + 0.01 * np.sin(np.arange(n) * 2 * np.pi / 60), 0.0, 0.2)
        fuel_oth = np.clip(1.0 - (fuel_ng + fuel_col + fuel_nuc + fuel_wnd + fuel_sun + fuel_wat), 0.0, 0.4)

        grid_df = pd.DataFrame(
            {
                "timestamp": ts.tz_convert("UTC").tz_localize(None),
                "demand_mwh": demand,
                "demand_forecast_mwh": demand + rng.normal(0, 70, n),
                "demand_forecast_error_mwh": rng.normal(0, 40, n),
                "interchange_mwh": rng.normal(0, 120, n),
                "net_generation_mwh": demand + rng.normal(0, 90, n),
                "fuel_col_share": fuel_col,
                "fuel_ng_share": fuel_ng,
                "fuel_nuc_share": fuel_nuc,
                "fuel_wnd_share": fuel_wnd,
                "fuel_sun_share": fuel_sun,
                "fuel_wat_share": fuel_wat,
                "fuel_oth_share": fuel_oth,
            }
        )
        weather_df = pd.DataFrame(
            {
                "timestamp": ts.tz_convert("UTC").tz_localize(None),
                "dew_point_c": temp_c - 3.0 + rng.normal(0, 0.5, n),
                "wind_speed_mps": np.clip(3.5 + rng.normal(0, 1.0, n), 0.1, None),
            }
        )
        power_util = np.clip(0.30 + 0.70 * cpu + rng.normal(0, 0.01, n), 0.05, 1.0)
        power_df = pd.DataFrame(
            {
                "real_timestamp": ts.strftime("%Y-%m-%d %H:%M:%S+00:00"),
                "measured_power_util": power_util,
            }
        )

        grid_df.to_csv(cleaned / "grid_exog_cleaned.csv", index=False)
        weather_df.to_csv(cleaned / "weather_exog_cleaned.csv", index=False)
        power_df.to_csv(cleaned / "power_optional_cleaned.csv", index=False)
        datasets.update(
            {
                "grid_exog": "cleaned/grid_exog_cleaned.csv",
                "weather_exog": "cleaned/weather_exog_cleaned.csv",
                "power_optional": "cleaned/power_optional_cleaned.csv",
            }
        )

    base_dir = tmp_path / "REAL FINAL FILES"
    output_dir = base_dir / "outputs"
    base_dir.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig(
        base_dir=base_dir,
        project_dir=tmp_path,
        data_dir=data_root,
        output_dir=output_dir,
        datasets=datasets,
        cv_splits=3,
        cpu_param_grid=[{"n_estimators": 120, "max_depth": 8, "min_samples_leaf": 2, "max_features": "sqrt"}],
        carbon_param_grid=[
            {
                "n_estimators": 120,
                "max_depth": 4,
                "learning_rate": 0.06,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
            }
        ],
        bootstrap_iterations=80,
    )
    return cfg


@pytest.fixture()
def pipeline_env(tmp_path: Path):
    cfg = _make_synthetic_environment(tmp_path)
    pipe = CarbonForecastPipeline(cfg)
    return cfg, pipe


@pytest.fixture()
def pipeline_env_optional(tmp_path: Path):
    cfg = _make_synthetic_environment(tmp_path, include_optional=True)
    pipe = CarbonForecastPipeline(
        cfg,
        use_grid_exog=True,
        use_weather_exog=True,
        use_power_calibration=True,
    )
    return cfg, pipe


def test_no_backfill_in_source():
    source = (ROOT / "carbon_prediction_pipeline.py").read_text(encoding="utf-8")
    assert ".bfill(" not in source
    assert "fillna(method='bfill')" not in source
    assert 'fillna(method="bfill")' not in source


def test_feature_engineering_uses_past_values_only(pipeline_env):
    _, pipe = pipeline_env
    df = pipe.load_data()
    cpu_view, ci_view, _ = pipe._build_supervised_views(df)
    assert not cpu_view.empty and not ci_view.empty

    row = cpu_view.iloc[0]
    ts = row["timestamp"]
    idx = int(df.index[df["timestamp"] == ts][0])

    assert row["cpu_lag_1"] == pytest.approx(df.loc[idx - 1, "avg_cpu_utilization"])
    expected_roll = df["avg_cpu_utilization"].shift(1).rolling(3, min_periods=3).mean().iloc[idx]
    assert row["cpu_roll_mean_3"] == pytest.approx(expected_roll)

    row_ci = ci_view.iloc[0]
    ts_ci = row_ci["timestamp"]
    idx_ci = int(df.index[df["timestamp"] == ts_ci][0])
    assert row_ci["ci_lag_1"] == pytest.approx(df.loc[idx_ci - 1, "carbon_intensity"])


def test_timestamp_integrity_and_schema_checks(pipeline_env):
    _, pipe = pipeline_env
    df = pipe.load_data()
    assert df["timestamp"].is_monotonic_increasing
    assert not df["timestamp"].duplicated().any()
    assert df["timestamp"].dt.tz is None

    with pytest.raises(ValueError):
        validate_required_columns(pd.DataFrame({"a": [1]}), "cpu")


def test_physics_consistency_in_evaluation(pipeline_env):
    _, pipe = pipeline_env
    df = pipe.load_data()
    train_df, holdout_df = pipe.split_train_holdout(df, pipe.config.train_ratio)
    pipe.fit(train_df)
    report = pipe.evaluate(holdout_df)

    stage6 = report["stage6_predictions"]
    assert np.allclose(stage6["pred_total_power"], stage6["pred_it_power"] * stage6["pred_pue"], atol=1e-8)
    assert np.allclose(
        stage6["pred_emissions"],
        stage6["pred_total_power"] * stage6["pred_carbon_intensity"],
        atol=1e-8,
    )


def test_integration_artifacts_and_reproducibility(pipeline_env):
    _, pipe1 = pipeline_env
    df1 = pipe1.load_data()
    tr1, ho1 = pipe1.split_train_holdout(df1, pipe1.config.train_ratio)
    pipe1.fit(tr1)
    rep1 = pipe1.evaluate(ho1)
    artifacts = pipe1.save_artifacts(rep1)

    required = [
        "metrics_summary",
        "metrics_cv_folds",
        "stage1_predictions",
        "stage5_predictions",
        "stage6_predictions",
        "baseline_comparison",
        "sanity_checks",
        "run_metadata",
        "model_card",
    ]
    for key in required:
        assert key in artifacts.files
        assert Path(artifacts.files[key]).exists()
    sanity_payload = json.loads(Path(artifacts.files["sanity_checks"]).read_text(encoding="utf-8"))
    assert "physics_identity_max_abs_error" in sanity_payload
    assert "passed" in sanity_payload

    # Reproducibility check with same seed and data.
    cfg2 = pipe1.config
    pipe2 = CarbonForecastPipeline(cfg2)
    df2 = pipe2.load_data()
    tr2, ho2 = pipe2.split_train_holdout(df2, pipe2.config.train_ratio)
    pipe2.fit(tr2)
    rep2 = pipe2.evaluate(ho2)

    rmse1 = float(rep1["metrics_summary_df"].loc[rep1["metrics_summary_df"]["stage"] == "Stage6_Emissions", "rmse"].iloc[0])
    rmse2 = float(rep2["metrics_summary_df"].loc[rep2["metrics_summary_df"]["stage"] == "Stage6_Emissions", "rmse"].iloc[0])
    assert abs(rmse1 - rmse2) < 1e-9


def test_predict_requires_only_history_and_future_exog(pipeline_env):
    _, pipe = pipeline_env
    df = pipe.load_data()
    train_df, holdout_df = pipe.split_train_holdout(df, pipe.config.train_ratio)
    pipe.fit(train_df)

    history = train_df[["timestamp", "avg_cpu_utilization", "carbon_intensity", "temperature_c", "num_tasks_sampled"]].copy()
    future_exog = holdout_df[["timestamp", "temperature_c", "num_tasks_sampled"]].head(12).copy()
    preds = pipe.predict(history, future_exog, horizon_hours=12)

    assert len(preds) == 12
    for col in ["pred_cpu", "pred_carbon_intensity", "pred_it_power", "pred_pue", "pred_total_power", "pred_emissions"]:
        assert col in preds.columns


def test_variant_stage6_target_equivalence_when_gate_mode_matches(pipeline_env_optional):
    cfg, enhanced_pipe = pipeline_env_optional
    data = enhanced_pipe.load_data()
    train_df, holdout_df = enhanced_pipe.split_train_holdout(data, enhanced_pipe.config.train_ratio)
    enhanced_pipe.fit(train_df)
    enhanced_report = enhanced_pipe.evaluate(holdout_df, stage6_target_mode=cfg.stage6_variant_gate_target_mode)

    core_pipe = CarbonForecastPipeline(
        cfg,
        use_grid_exog=False,
        use_weather_exog=False,
        use_power_calibration=False,
    )
    core_pipe.fit(train_df)
    core_report = core_pipe.evaluate(holdout_df, stage6_target_mode=cfg.stage6_variant_gate_target_mode)

    lhs = enhanced_report["stage6_predictions"][["timestamp", "actual_emissions"]].rename(
        columns={"actual_emissions": "actual_enh"}
    )
    rhs = core_report["stage6_predictions"][["timestamp", "actual_emissions"]].rename(
        columns={"actual_emissions": "actual_core"}
    )
    merged = lhs.merge(rhs, on="timestamp", how="inner")
    assert not merged.empty
    assert np.allclose(merged["actual_enh"], merged["actual_core"], atol=1e-10)


def test_calibration_negative_slope_falls_back_to_identity(pipeline_env):
    _, pipe = pipeline_env
    pipe.optional_dataset_usage["power_calibration"] = True
    n = 120
    cpu_oof = np.linspace(0.1, 0.9, n)
    cpu_view = pd.DataFrame(
        {
            "temperature_f_t_plus_h": np.full(n, 70.0),
            "observed_total_power_mw_t_plus_h": np.linspace(200.0, 50.0, n),
        }
    )
    pipe._fit_physics_calibration(cpu_view, cpu_oof)

    assert pipe.calibration_valid is False
    assert pipe.physics_calibration["it_power_slope"] == pytest.approx(1.0)
    assert pipe.physics_calibration["total_power_slope"] == pytest.approx(1.0)
    assert pipe.physics_calibration["calibration_reason"] == "nonphysical_negative_slope"


def test_predict_requires_future_exog_when_stage5_uses_exog(pipeline_env_optional):
    _, pipe = pipeline_env_optional
    data = pipe.load_data()
    train_df, holdout_df = pipe.split_train_holdout(data, pipe.config.train_ratio)
    pipe.fit(train_df)
    history = train_df.copy()
    required = list(pipe.required_future_exog_columns)
    assert "demand_mwh" in required

    bad_future = holdout_df[["timestamp", "temperature_c"]].head(12).copy()
    with pytest.raises(ValueError):
        pipe.predict(history, bad_future, horizon_hours=12)


def test_coverage_usage_matches_selected_variant_metadata(pipeline_env_optional):
    _, pipe = pipeline_env_optional
    selected_pipe, selected_report, _ = run_train_eval(pipe)
    metadata = selected_report["metadata"]
    coverage = selected_report["data_coverage_report_df"].copy()
    assert not coverage.empty

    source_to_expected = {
        "cpu": True,
        "temperature": True,
        "carbon_intensity": True,
        "grid_exog": bool(metadata["optional_dataset_usage"]["grid_exog"]),
        "weather_exog": bool(metadata["optional_dataset_usage"]["weather_exog"]),
        "power_optional": bool(metadata["optional_dataset_usage"]["power_optional"]),
    }
    got = dict(zip(coverage["source"], coverage["used_in_model"]))
    for source, expected in source_to_expected.items():
        assert bool(got[source]) == expected
    assert metadata["stage6_target_mode_used_for_variant_gate"] == selected_pipe.config.stage6_variant_gate_target_mode
