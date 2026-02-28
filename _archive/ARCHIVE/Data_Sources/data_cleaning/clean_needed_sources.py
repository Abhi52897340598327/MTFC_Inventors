#!/usr/bin/env python
"""
Clean every CSV currently labeled as "needs cleaning".

Outputs:
- cleaned CSVs in Data_Sources/data_cleaning/cleaned_outputs/<relative_source_path>.csv
- summary report in Data_Sources/data_cleaning/reports/cleaning_summary.csv
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_SOURCES = ROOT / "Data_Sources"
OUT_ROOT = DATA_SOURCES / "data_cleaning" / "cleaned_outputs"
REPORT_ROOT = DATA_SOURCES / "data_cleaning" / "reports"


USABLE_NOW = {
    "Data_Sources/ashburn_va_temperature_2019.csv",
    "Data_Sources/cleaned/ashburn_va_temperature_2019_cleaned.csv",
    "Data_Sources/cleaned/google_cluster_plus_power_2019_cellb_hourly.csv",
    "Data_Sources/cleaned/google_cluster_utilization_2019_cellb_hourly_cleaned.csv",
    "Data_Sources/cleaned/google_power_utilization_2019_cellb_hourly.csv",
    "Data_Sources/cleaned/noaa_global_hourly_dulles_2019_2024_cleaned.csv",
    "Data_Sources/cleaned/pjm_exogenous_hourly_2019_2024_cleaned.csv",
    "Data_Sources/cleaned/pjm_grid_carbon_intensity_2019_full_cleaned.csv",
    "Data_Sources/cleaned/pjm_hourly_demand_2019_2024_cleaned.csv",
    "Data_Sources/external_downloads/google_cluster_cpu_cellb_hourly_2019.csv",
    "Data_Sources/external_downloads/google_power_cellb_hourly_2019.csv",
    "Data_Sources/google_cluster_utilization_2019.csv",
    "Data_Sources/gpu-price-performance.csv",
    "Data_Sources/hardware-and-energy-cost-to-train-notable-ai-systems.csv",
    "Data_Sources/model_ready/final_training_table_2019_hourly.csv",
    "Data_Sources/model_ready/final_training_table_2019_hourly_with_power.csv",
    "Data_Sources/model_ready/joined_cpu_power_optional.csv",
    "Data_Sources/model_ready/power_optional_feature.csv",
    "Data_Sources/model_ready/stage1_cpu_primary.csv",
    "Data_Sources/model_ready/stage3_temperature_primary.csv",
    "Data_Sources/model_ready/stage3_weather_exog_optional.csv",
    "Data_Sources/model_ready/stage5_carbon_primary.csv",
    "Data_Sources/model_ready/stage5_grid_exog_optional.csv",
    "Data_Sources/monthly-spending-data-center-us.csv",
    "Data_Sources/noaa_dulles_daily_2015_2024.csv",
    "Data_Sources/noaa_gsod_dulles_2019.csv",
    "Data_Sources/pjm_demand_forecast_2019_2024_eia.csv",
    "Data_Sources/pjm_generation_by_fuel_2019_2024_eia.csv",
    "Data_Sources/pjm_hourly_demand_2019_2024_eia.csv",
    "Data_Sources/pjm_interchange_2019_2024_eia.csv",
    "Data_Sources/pjm_net_generation_2019_2024_eia.csv",
    "Data_Sources/share_companies_using_ai_owid_raw.csv",
}

EXCLUDE = {
    "Data_Sources/_archive/google_power_cellb_hourly_idx_2019.csv",
    "Data_Sources/cleaned/semisynthetic_datacenter_power_2015_2024.csv",
    "Data_Sources/epa_temperature_loudoun_2019.csv",
    "Data_Sources/model_ready/synthetic_reference_only.csv",
    "Data_Sources/reports/data_inventory.csv",
    "Data_Sources/reports/data_quality_flags.csv",
    "Data_Sources/reports/model_data_catalog.csv",
}


def to_rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def safe_print(msg: str) -> None:
    print(msg, flush=True)


def read_csv_robust(path: Path) -> tuple[pd.DataFrame, str]:
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "utf-16", "latin1"):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            return df, enc
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise RuntimeError(f"Failed to read {path} with robust encodings: {last_err}")


def strip_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype(str).str.strip()
            out[c] = out[c].replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return out


def clean_col_name(name: str, idx: int) -> str:
    s = "" if name is None else str(name).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_")
    if not s or s.lower().startswith("unnamed"):
        s = f"col_{idx}"
    return s.lower()


def uniquify_columns(cols: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for c in cols:
        n = seen.get(c, 0)
        if n == 0:
            out.append(c)
        else:
            out.append(f"{c}_{n}")
        seen[c] = n + 1
    return out


def convert_numeric_and_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        series = out[c]
        if series.dtype != object:
            continue
        sample = series.dropna()
        if sample.empty:
            continue

        # numeric conversion threshold
        num = pd.to_numeric(sample.astype(str).str.replace(",", "", regex=False), errors="coerce")
        num_ratio = num.notna().mean()
        if num_ratio >= 0.80 and len(sample) >= 8:
            out[c] = pd.to_numeric(out[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
            continue

        # datetime conversion threshold (only date-like names)
        if re.search(r"(date|time|period|timestamp|year|month|day)", c, flags=re.I):
            dt = pd.to_datetime(sample, errors="coerce", utc=False)
            dt_ratio = dt.notna().mean()
            if dt_ratio >= 0.70:
                out[c] = pd.to_datetime(out[c], errors="coerce", utc=False)
    return out


def generic_clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = strip_object_columns(out)
    out = out.dropna(axis=1, how="all")
    out = out.dropna(axis=0, how="all")
    out.columns = uniquify_columns([clean_col_name(c, i) for i, c in enumerate(out.columns)])
    out = convert_numeric_and_dates(out)
    out = out.drop_duplicates()
    return out.reset_index(drop=True)


def clean_wide_net_imports(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    base_cols = list(out.columns[:3])
    out = out.rename(columns={base_cols[0]: "iso_code", base_cols[1]: "entity", base_cols[2]: "series"})
    value_cols = [c for c in out.columns if c not in {"iso_code", "entity", "series"}]
    out = out.melt(id_vars=["iso_code", "entity", "series"], value_vars=value_cols, var_name="period_raw", value_name="value")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"]).copy()
    out["period"] = pd.to_datetime(out["period_raw"], errors="coerce")
    # Try annual parse if monthly parse failed.
    bad = out["period"].isna()
    if bad.any():
        out.loc[bad, "period"] = pd.to_datetime(out.loc[bad, "period_raw"], format="%Y", errors="coerce")
    out["year"] = out["period"].dt.year
    out["month"] = out["period"].dt.month
    out = out.sort_values(["iso_code", "series", "period"]).reset_index(drop=True)
    return out[["iso_code", "entity", "series", "period_raw", "period", "year", "month", "value"]]


def detect_header_row(raw: pd.DataFrame, max_scan: int = 40) -> int:
    best_idx = 0
    best_score = -1.0
    lim = min(max_scan, len(raw))
    for i in range(lim):
        row = raw.iloc[i].astype(str)
        vals = [v.strip() for v in row.tolist() if v and v.strip() and v.strip().lower() != "nan"]
        if not vals:
            continue
        meaningful = [v for v in vals if not v.lower().startswith("unnamed")]
        year_tokens = sum(bool(re.search(r"(19|20)\d{2}", v)) for v in vals)
        score = len(meaningful) + (2.0 * year_tokens)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def clean_excel_export_style(path: Path) -> pd.DataFrame:
    # Read raw without header so we can detect true table header rows.
    raw = None
    for enc in ("utf-8-sig", "utf-8", "utf-16", "latin1"):
        try:
            raw = pd.read_csv(path, header=None, encoding=enc, low_memory=False, dtype=str)
            break
        except Exception:  # noqa: BLE001
            continue
    if raw is None:
        raise RuntimeError(f"Could not read {path} as raw export table.")

    raw = raw.replace(r"^\s*$", np.nan, regex=True)
    raw = raw.dropna(axis=1, how="all")
    raw = raw.dropna(axis=0, how="all")
    if raw.empty:
        return pd.DataFrame()

    h_idx = detect_header_row(raw)
    header = [clean_col_name(v, i) for i, v in enumerate(raw.iloc[h_idx].tolist())]
    header = uniquify_columns(header)
    data = raw.iloc[h_idx + 1 :].copy()
    data.columns = header
    data = data.dropna(axis=0, how="all")
    data = generic_clean(data)
    return data


def parse_noaa_code_numeric(series: pd.Series, scale: float = 1.0, missing_sentinels: set[int] | None = None) -> pd.Series:
    first = series.astype(str).str.split(",", n=1).str[0]
    vals = pd.to_numeric(first, errors="coerce")
    if missing_sentinels:
        for sentinel in missing_sentinels:
            vals = vals.mask(vals == sentinel, np.nan)
    return vals / scale


def clean_noaa_global_hourly(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df.get("DATE"), errors="coerce")
    out["station"] = df.get("STATION")
    out["temperature_c"] = parse_noaa_code_numeric(df.get("TMP", pd.Series(dtype=object)), scale=10.0, missing_sentinels={9999, 99999})
    out["dew_point_c"] = parse_noaa_code_numeric(df.get("DEW", pd.Series(dtype=object)), scale=10.0, missing_sentinels={9999, 99999})
    out["sea_level_pressure_hpa"] = parse_noaa_code_numeric(df.get("SLP", pd.Series(dtype=object)), scale=10.0, missing_sentinels={99999, 9999})
    # WND format: dir,dir_qc,type,speed,speed_qc ; speed usually tenths of m/s.
    wnd_speed_raw = df.get("WND", pd.Series(dtype=object)).astype(str).str.split(",").str[3]
    wnd_speed = pd.to_numeric(wnd_speed_raw, errors="coerce").mask(pd.to_numeric(wnd_speed_raw, errors="coerce") == 9999, np.nan) / 10.0
    out["wind_speed_mps"] = wnd_speed
    out["visibility_m"] = parse_noaa_code_numeric(df.get("VIS", pd.Series(dtype=object)), scale=1.0, missing_sentinels={999999})
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates()
    return out.reset_index(drop=True)


def clean_ashburn_2024(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    tidy = (
        out.pivot_table(index=["date", "station"], columns="datatype", values="value", aggfunc="mean")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    # Preserve old style too for traceability
    tidy = tidy.sort_values("date").reset_index(drop=True)
    return tidy


def clean_ai_models(df: pd.DataFrame) -> pd.DataFrame:
    out = generic_clean(df)
    if "publication_date" in out.columns:
        out["publication_date"] = pd.to_datetime(out["publication_date"], errors="coerce")
    for c in out.columns:
        if re.search(r"(compute|cost|power|parameter|time_hours|training_time|chip_hours|citation)", c, flags=re.I):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def clean_hour_index_cpu(df: pd.DataFrame) -> pd.DataFrame:
    out = generic_clean(df)
    if "hour_index" in out.columns:
        out["hour_index"] = pd.to_numeric(out["hour_index"], errors="coerce")
        out["timestamp_utc_assumed"] = pd.Timestamp("2019-05-01T00:00:00Z") + pd.to_timedelta(out["hour_index"], unit="h")
    return out


def clean_virginia_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = generic_clean(df)
    if "period" in out.columns:
        out["period"] = pd.to_datetime(out["period"], errors="coerce")
    return out


def choose_strategy(rel_path: str) -> tuple[str, Callable[[pd.DataFrame], pd.DataFrame] | None]:
    lower = rel_path.lower()
    name = Path(rel_path).name.lower()
    if "data-including-net-imports" in name:
        return "wide_to_long_timeseries", clean_wide_net_imports
    if "sep tables for va__" in name or "eia923_schedules_2_3_4_5_m_11_2025_21jan2026__" in name:
        return "excel_export_table_cleanup", None  # handled with path-level reader
    if "noaa_global_hourly_72403093738_" in name:
        return "decode_noaa_global_hourly", clean_noaa_global_hourly
    if name == "ashburn_va_temperature_2024.csv" or name == "ashburn_va_temperature_2024_cleaned.csv":
        return "pivot_station_daily_temperature", clean_ashburn_2024
    if name in {"all_ai_models.csv", "frontier_ai_models.csv", "large_scale_ai_models.csv", "notable_ai_models.csv"}:
        return "ai_models_schema_cleanup", clean_ai_models
    if name == "google_cluster_cpu_cellb_hourly_idx_2019.csv":
        return "hour_index_to_timestamp", clean_hour_index_cpu
    if "virginia_" in name and ("generation" in name or "consumption" in name or "co2" in name):
        return "metadata_standardization", clean_virginia_metadata
    return "generic_cleanup", generic_clean


def clean_one(path: Path) -> dict:
    rel = to_rel(path)
    strategy, fn = choose_strategy(rel)
    status = "cleaned"
    notes = ""
    encoding = ""
    rows_in = cols_in = rows_out = cols_out = 0

    try:
        if strategy == "excel_export_table_cleanup":
            df_out = clean_excel_export_style(path)
            # For stats we still read with robust parser.
            df_in, encoding = read_csv_robust(path)
        else:
            df_in, encoding = read_csv_robust(path)
            if fn is None:
                df_out = generic_clean(df_in)
            else:
                df_out = fn(df_in)

        rows_in, cols_in = df_in.shape
        rows_out, cols_out = df_out.shape

        if rows_out == 0 or cols_out == 0:
            status = "cleaned_but_limited"
            notes = "No substantive data after cleaning."
        elif cols_out <= 3 and not any(re.search(r"(value|generation|sales|price|demand|netgen|carbon|emission)", c, flags=re.I) for c in df_out.columns):
            status = "cleaned_but_limited"
            notes = "Likely metadata-only structure after cleanup."

        out_path = OUT_ROOT / Path(rel).relative_to("Data_Sources")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_path, index=False)

        return {
            "source_file": rel,
            "output_file": str(out_path.relative_to(ROOT)).replace("\\", "/"),
            "strategy": strategy,
            "status": status,
            "notes": notes,
            "encoding": encoding,
            "rows_in": rows_in,
            "cols_in": cols_in,
            "rows_out": rows_out,
            "cols_out": cols_out,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "source_file": rel,
            "output_file": "",
            "strategy": strategy,
            "status": "failed",
            "notes": str(exc),
            "encoding": encoding,
            "rows_in": rows_in,
            "cols_in": cols_in,
            "rows_out": rows_out,
            "cols_out": cols_out,
        }


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    all_csvs = sorted(DATA_SOURCES.rglob("*.csv"))
    needs = []
    for p in all_csvs:
        rel = to_rel(p)
        if rel.startswith("Data_Sources/data_cleaning/"):
            continue
        if rel in USABLE_NOW:
            continue
        if rel in EXCLUDE:
            continue
        needs.append(p)

    safe_print(f"Found {len(needs)} needs-cleaning files.")
    results = []
    for idx, p in enumerate(needs, start=1):
        safe_print(f"[{idx}/{len(needs)}] Cleaning {to_rel(p)}")
        results.append(clean_one(p))

    report = pd.DataFrame(results).sort_values(["status", "source_file"]).reset_index(drop=True)
    report_path = REPORT_ROOT / "cleaning_summary.csv"
    report.to_csv(report_path, index=False)

    counts = report["status"].value_counts(dropna=False).to_dict()
    safe_print(f"Done. Status counts: {counts}")
    safe_print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

