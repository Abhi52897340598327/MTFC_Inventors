"""Build additional cleaned datasets from available raw sources."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CLEANED = ROOT / "cleaned"
EXTERNAL = ROOT / "external_downloads"


def _to_utc_hour(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    return ts.dt.floor("h")


def build_pjm_exogenous() -> Path:
    demand = pd.read_csv(ROOT / "pjm_hourly_demand_2019_2024_eia.csv")
    forecast = pd.read_csv(ROOT / "pjm_demand_forecast_2019_2024_eia.csv")
    interchange = pd.read_csv(ROOT / "pjm_interchange_2019_2024_eia.csv")
    net_gen = pd.read_csv(ROOT / "pjm_net_generation_2019_2024_eia.csv")
    by_fuel = pd.read_csv(ROOT / "pjm_generation_by_fuel_2019_2024_eia.csv")

    demand_h = pd.DataFrame(
        {
            "timestamp": _to_utc_hour(demand["datetime_utc"]),
            "demand_mwh": pd.to_numeric(demand["demand_mwh"], errors="coerce"),
        }
    ).dropna(subset=["timestamp"])
    demand_h = demand_h.groupby("timestamp", as_index=False).agg({"demand_mwh": "mean"})

    forecast_h = pd.DataFrame(
        {
            "timestamp": _to_utc_hour(forecast["period"]),
            "demand_forecast_mwh": pd.to_numeric(forecast["value"], errors="coerce"),
        }
    ).dropna(subset=["timestamp"])
    forecast_h = forecast_h.groupby("timestamp", as_index=False).agg({"demand_forecast_mwh": "mean"})

    interchange_h = pd.DataFrame(
        {
            "timestamp": _to_utc_hour(interchange["period"]),
            "interchange_mwh": pd.to_numeric(interchange["value"], errors="coerce"),
        }
    ).dropna(subset=["timestamp"])
    interchange_h = interchange_h.groupby("timestamp", as_index=False).agg({"interchange_mwh": "mean"})

    net_gen_h = pd.DataFrame(
        {
            "timestamp": _to_utc_hour(net_gen["period"]),
            "net_generation_mwh": pd.to_numeric(net_gen["value"], errors="coerce"),
        }
    ).dropna(subset=["timestamp"])
    net_gen_h = net_gen_h.groupby("timestamp", as_index=False).agg({"net_generation_mwh": "mean"})

    fuel = pd.DataFrame(
        {
            "timestamp": _to_utc_hour(by_fuel["period"]),
            "fueltype": by_fuel["fueltype"].astype(str),
            "fuel_mwh": pd.to_numeric(by_fuel["value"], errors="coerce"),
        }
    ).dropna(subset=["timestamp"])
    fuel_wide = (
        fuel.pivot_table(index="timestamp", columns="fueltype", values="fuel_mwh", aggfunc="sum")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    fuel_cols = [c for c in fuel_wide.columns if c != "timestamp"]
    rename = {c: f"fuel_{str(c).lower()}_mwh" for c in fuel_cols}
    fuel_wide = fuel_wide.rename(columns=rename)
    fuel_value_cols = [c for c in fuel_wide.columns if c != "timestamp"]
    fuel_wide["fuel_total_mwh"] = fuel_wide[fuel_value_cols].sum(axis=1, skipna=True)
    for col in fuel_value_cols:
        share_col = col.replace("_mwh", "_share")
        fuel_wide[share_col] = fuel_wide[col] / fuel_wide["fuel_total_mwh"]

    merged = demand_h.merge(forecast_h, on="timestamp", how="outer")
    merged = merged.merge(interchange_h, on="timestamp", how="outer")
    merged = merged.merge(net_gen_h, on="timestamp", how="outer")
    merged = merged.merge(fuel_wide, on="timestamp", how="outer")
    merged["demand_forecast_error_mwh"] = merged["demand_mwh"] - merged["demand_forecast_mwh"]
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    out = CLEANED / "pjm_exogenous_hourly_2019_2024_cleaned.csv"
    merged.to_csv(out, index=False)
    return out


def _split_field(series: pd.Series, idx: int, scale: float = 1.0) -> pd.Series:
    tokens = series.astype(str).str.split(",", expand=True)
    vals = pd.to_numeric(tokens[idx], errors="coerce")
    vals = vals.mask(vals >= 9999)
    return vals / scale


def iter_noaa_global_files() -> Iterable[Path]:
    for p in sorted(EXTERNAL.glob("noaa_global_hourly_72403093738_*.csv")):
        yield p


def build_noaa_global_hourly_cleaned() -> Path:
    frames = []
    for p in iter_noaa_global_files():
        df = pd.read_csv(p, low_memory=False)
        if "DATE" not in df.columns:
            continue
        out = pd.DataFrame()
        out["timestamp"] = _to_utc_hour(df["DATE"])
        out["temperature_c"] = _split_field(df.get("TMP", pd.Series(dtype=str)), 0, scale=10.0)
        out["dew_point_c"] = _split_field(df.get("DEW", pd.Series(dtype=str)), 0, scale=10.0)
        out["sea_level_pressure_hpa"] = _split_field(df.get("SLP", pd.Series(dtype=str)), 0, scale=10.0)
        out["wind_speed_mps"] = _split_field(df.get("WND", pd.Series(dtype=str)), 3, scale=10.0)
        out["visibility_m"] = _split_field(df.get("VIS", pd.Series(dtype=str)), 0, scale=1.0)
        frames.append(out.dropna(subset=["timestamp"]))

    if not frames:
        raise FileNotFoundError("No NOAA global hourly files found in external_downloads.")

    all_rows = pd.concat(frames, ignore_index=True)
    agg = (
        all_rows.groupby("timestamp", as_index=False)
        .agg(
            temperature_c=("temperature_c", "mean"),
            dew_point_c=("dew_point_c", "mean"),
            sea_level_pressure_hpa=("sea_level_pressure_hpa", "mean"),
            wind_speed_mps=("wind_speed_mps", "mean"),
            visibility_m=("visibility_m", "mean"),
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    out = CLEANED / "noaa_global_hourly_dulles_2019_2024_cleaned.csv"
    agg.to_csv(out, index=False)
    return out


def main() -> None:
    CLEANED.mkdir(parents=True, exist_ok=True)
    pjm_out = build_pjm_exogenous()
    weather_out = build_noaa_global_hourly_cleaned()
    print(f"Wrote: {pjm_out}")
    print(f"Wrote: {weather_out}")


if __name__ == "__main__":
    main()

