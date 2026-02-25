"""Audit and organize Data_Sources assets.

Creates:
- Data_Sources/reports/data_inventory.csv
- Data_Sources/reports/data_quality_flags.csv
- Data_Sources/reports/model_data_catalog.csv
- Data_Sources/reports/DATA_AUDIT_SUMMARY.md
- Data_Sources/model_ready/* (non-destructive copies of key tables)
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
MODEL_READY_DIR = ROOT / "model_ready"
SKIP_DIRS = {"reports", "model_ready", "_archive", "scripts"}


@dataclass
class FileProfile:
    rel_path: str
    kind: str
    size_bytes: int
    encoding: str
    rows: int
    cols: int
    columns: str
    timestamp_columns: str
    ts_min: str
    ts_max: str
    missing_ratio_sample: float
    quality_flags: str
    notes: str


def iter_files(base: Path) -> Iterable[Path]:
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(base)
        if rel.parts and rel.parts[0] in SKIP_DIRS:
            continue
        yield p


def rel_posix(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def detect_csv_encoding(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            with path.open("r", encoding=enc, newline="") as fh:
                fh.readline()
            return enc
        except UnicodeDecodeError:
            continue
    return "unknown"


def safe_line_count(path: Path, encoding: str) -> int:
    if encoding == "unknown":
        return -1
    count = 0
    with path.open("r", encoding=encoding, errors="replace", newline="") as fh:
        for _ in fh:
            count += 1
    return max(count - 1, 0)


def summarize_timestamps(df: pd.DataFrame) -> tuple[str, str, str]:
    ts_cols = [c for c in df.columns if any(k in c.lower() for k in ("time", "date", "timestamp"))]
    if not ts_cols:
        return "", "", ""

    ts_min = ""
    ts_max = ""
    for col in ts_cols:
        s = pd.to_datetime(df[col], errors="coerce", utc=True)
        s = s.dropna()
        if s.empty:
            continue
        cmin = s.min().isoformat()
        cmax = s.max().isoformat()
        if not ts_min or cmin < ts_min:
            ts_min = cmin
        if not ts_max or cmax > ts_max:
            ts_max = cmax
    return ",".join(ts_cols), ts_min, ts_max


def profile_csv(path: Path) -> FileProfile:
    flags: list[str] = []
    notes: list[str] = []
    encoding = detect_csv_encoding(path)
    size = path.stat().st_size
    if size == 0:
        flags.append("empty_file")
        return FileProfile(
            rel_path=rel_posix(path),
            kind="csv",
            size_bytes=size,
            encoding=encoding,
            rows=0,
            cols=0,
            columns="",
            timestamp_columns="",
            ts_min="",
            ts_max="",
            missing_ratio_sample=1.0,
            quality_flags="|".join(flags),
            notes="",
        )

    rows = safe_line_count(path, encoding)
    try:
        df = pd.read_csv(path, encoding=encoding, nrows=10000, low_memory=False)
    except Exception as exc:
        flags.append("read_error")
        notes.append(str(exc))
        return FileProfile(
            rel_path=rel_posix(path),
            kind="csv",
            size_bytes=size,
            encoding=encoding,
            rows=max(rows, 0),
            cols=0,
            columns="",
            timestamp_columns="",
            ts_min="",
            ts_max="",
            missing_ratio_sample=1.0,
            quality_flags="|".join(flags),
            notes="; ".join(notes),
        )

    cols = [str(c) for c in df.columns]
    ts_cols, ts_min, ts_max = summarize_timestamps(df)
    miss_ratio = float(df.isna().mean().mean()) if not df.empty else 0.0

    if rows == 0:
        flags.append("header_only")
    if rows > 0 and rows < 24:
        flags.append("very_short_series")
    if size < 1024:
        flags.append("tiny_file")
    if miss_ratio > 0.4:
        flags.append("high_missingness_sample")

    name = path.name.lower()
    if "semisynthetic" in name:
        flags.append("synthetic_data")
    if "download_test" in name:
        flags.append("test_artifact")
    if "idx" in name:
        flags.append("intermediate_export")

    return FileProfile(
        rel_path=rel_posix(path),
        kind="csv",
        size_bytes=size,
        encoding=encoding,
        rows=max(rows, 0),
        cols=len(cols),
        columns=",".join(cols),
        timestamp_columns=ts_cols,
        ts_min=ts_min,
        ts_max=ts_max,
        missing_ratio_sample=miss_ratio,
        quality_flags="|".join(flags),
        notes="; ".join(notes),
    )


def profile_xlsx(path: Path) -> FileProfile:
    flags: list[str] = []
    notes: list[str] = []
    size = path.stat().st_size
    rows = -1
    cols = -1
    columns = ""
    ts_cols = ""
    ts_min = ""
    ts_max = ""
    miss_ratio = 0.0

    try:
        xf = pd.ExcelFile(path)
        sheet_names = xf.sheet_names
        notes.append(f"sheets={','.join(sheet_names[:10])}")
        sample = pd.read_excel(path, sheet_name=sheet_names[0], nrows=5000)
        rows = len(sample)
        cols = len(sample.columns)
        columns = ",".join(str(c) for c in sample.columns)
        ts_cols, ts_min, ts_max = summarize_timestamps(sample)
        miss_ratio = float(sample.isna().mean().mean()) if not sample.empty else 0.0
    except Exception as exc:
        flags.append("xlsx_read_error")
        notes.append(str(exc))

    return FileProfile(
        rel_path=rel_posix(path),
        kind="xlsx",
        size_bytes=size,
        encoding="binary",
        rows=max(rows, 0),
        cols=max(cols, 0),
        columns=columns,
        timestamp_columns=ts_cols,
        ts_min=ts_min,
        ts_max=ts_max,
        missing_ratio_sample=miss_ratio,
        quality_flags="|".join(flags),
        notes="; ".join(notes),
    )


def profile_generic(path: Path) -> FileProfile:
    size = path.stat().st_size
    ext = path.suffix.lower().lstrip(".") or "file"
    flags: list[str] = []
    notes: list[str] = []
    name = path.name.lower()
    if size == 0:
        flags.append("empty_file")
    if "download_test" in name:
        flags.append("test_artifact")
    if ext in {"md", "txt", "json", "html", "ipynb"}:
        notes.append("documentation_or_metadata")
    if ext in {"gz", "pdf"}:
        notes.append("binary_asset")

    return FileProfile(
        rel_path=rel_posix(path),
        kind=ext,
        size_bytes=size,
        encoding="n/a",
        rows=0,
        cols=0,
        columns="",
        timestamp_columns="",
        ts_min="",
        ts_max="",
        missing_ratio_sample=0.0,
        quality_flags="|".join(flags),
        notes="; ".join(notes),
    )


def pick_model_inputs(inv: pd.DataFrame) -> list[dict]:
    picks: list[dict] = []
    rows = inv.set_index("rel_path")

    def add(role: str, path: str, rationale: str) -> None:
        exists = path in rows.index
        picks.append(
            {
                "role": role,
                "path": path,
                "exists": bool(exists),
                "rows": int(rows.loc[path, "rows"]) if exists else 0,
                "quality_flags": str(rows.loc[path, "quality_flags"]) if exists else "missing",
                "rationale": rationale,
            }
        )

    add(
        "stage1_cpu_primary",
        "cleaned/google_cluster_utilization_2019_cellb_hourly_cleaned.csv",
        "Full hourly May-2019 CPU trace extracted from Google cluster data (cell b).",
    )
    add(
        "stage3_temperature_primary",
        "cleaned/ashburn_va_temperature_2019_cleaned.csv",
        "Hourly local temperature aligned to 2019 timeline for PUE physics stage.",
    )
    add(
        "stage5_carbon_primary",
        "cleaned/pjm_grid_carbon_intensity_2019_full_cleaned.csv",
        "Hourly PJM carbon intensity for carbon forecast target.",
    )
    add(
        "power_optional_feature",
        "cleaned/google_power_utilization_2019_cellb_hourly.csv",
        "Real power-utilization trace can be used for physics calibration and auxiliary features.",
    )
    add(
        "joined_cpu_power_optional",
        "cleaned/google_cluster_plus_power_2019_cellb_hourly.csv",
        "CPU-power joined view for learning CPU->power mapping and diagnostics.",
    )
    add(
        "stage5_grid_exog_optional",
        "cleaned/pjm_exogenous_hourly_2019_2024_cleaned.csv",
        "Hourly PJM demand, forecast, interchange, and fuel mix for carbon-intensity exogenous features.",
    )
    add(
        "stage3_weather_exog_optional",
        "cleaned/noaa_global_hourly_dulles_2019_2024_cleaned.csv",
        "Richer hourly weather signals (dew point, pressure, wind, visibility).",
    )
    add(
        "synthetic_reference_only",
        "cleaned/semisynthetic_datacenter_power_2015_2024.csv",
        "Keep for stress testing and ablation, not for headline results.",
    )
    return picks


def copy_model_ready(catalog: pd.DataFrame) -> list[str]:
    MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for _, row in catalog.iterrows():
        if not bool(row["exists"]):
            continue
        src = ROOT / str(row["path"])
        dst_name = f"{row['role']}.csv"
        dst = MODEL_READY_DIR / dst_name
        if src.suffix.lower() == ".csv":
            shutil.copy2(src, dst)
            copied.append(str(dst.relative_to(ROOT)))
    return copied


def _normalize_hourly(
    df: pd.DataFrame,
    timestamp_col: str,
    rename: dict[str, str],
    agg: dict[str, str],
) -> pd.DataFrame:
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work[timestamp_col], errors="coerce", utc=True)
    work = work.dropna(subset=["timestamp"])
    work["timestamp"] = work["timestamp"].dt.floor("h")
    work = work.rename(columns=rename)
    cols = ["timestamp"] + [c for c in agg.keys() if c in work.columns]
    work = work[cols]
    out = work.groupby("timestamp", as_index=False).agg(agg)
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def build_training_tables(catalog: pd.DataFrame) -> dict[str, int]:
    role_to_path = {str(r["role"]): ROOT / str(r["path"]) for _, r in catalog.iterrows() if bool(r["exists"])}
    if not {"stage1_cpu_primary", "stage3_temperature_primary", "stage5_carbon_primary"}.issubset(role_to_path):
        return {}

    cpu = pd.read_csv(role_to_path["stage1_cpu_primary"])
    temp = pd.read_csv(role_to_path["stage3_temperature_primary"])
    ci = pd.read_csv(role_to_path["stage5_carbon_primary"])

    cpu_h = _normalize_hourly(
        cpu,
        "real_timestamp",
        rename={"avg_cpu_utilization": "avg_cpu_utilization", "num_tasks_sampled": "num_tasks_sampled"},
        agg={"avg_cpu_utilization": "mean", "num_tasks_sampled": "sum"},
    )
    temp_h = _normalize_hourly(
        temp,
        "timestamp",
        rename={"temperature_c": "temperature_c"},
        agg={"temperature_c": "mean"},
    )
    ci_h = _normalize_hourly(
        ci,
        "timestamp",
        rename={"carbon_intensity_kg_per_mwh": "carbon_intensity_kg_per_mwh"},
        agg={"carbon_intensity_kg_per_mwh": "mean"},
    )

    merged = cpu_h.merge(temp_h, on="timestamp", how="inner")
    merged = merged.merge(ci_h, on="timestamp", how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    merged.to_csv(MODEL_READY_DIR / "final_training_table_2019_hourly.csv", index=False)

    with_power = merged.copy()
    if "power_optional_feature" in role_to_path:
        power = pd.read_csv(role_to_path["power_optional_feature"])
        power_h = _normalize_hourly(
            power,
            "real_timestamp",
            rename={
                "measured_power_util": "measured_power_util",
                "production_power_util": "production_power_util",
                "sample_count": "power_sample_count",
            },
            agg={
                "measured_power_util": "mean",
                "production_power_util": "mean",
                "power_sample_count": "sum",
            },
        )
        with_power = with_power.merge(power_h, on="timestamp", how="left")
    with_power.to_csv(MODEL_READY_DIR / "final_training_table_2019_hourly_with_power.csv", index=False)

    return {
        "cpu_rows_hourly": len(cpu_h),
        "temp_rows_hourly": len(temp_h),
        "ci_rows_hourly": len(ci_h),
        "merged_rows": len(merged),
        "merged_with_power_rows": len(with_power),
    }


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    profiles: list[FileProfile] = []
    for path in iter_files(ROOT):
        suffix = path.suffix.lower()
        if suffix == ".csv":
            profiles.append(profile_csv(path))
        elif suffix == ".xlsx":
            profiles.append(profile_xlsx(path))
        else:
            profiles.append(profile_generic(path))

    inv = pd.DataFrame(asdict(p) for p in profiles).sort_values("rel_path").reset_index(drop=True)
    inv_path = REPORTS_DIR / "data_inventory.csv"
    inv.to_csv(inv_path, index=False, quoting=csv.QUOTE_MINIMAL)

    flagged = inv[inv["quality_flags"].astype(str) != ""].copy()
    flagged_path = REPORTS_DIR / "data_quality_flags.csv"
    flagged.to_csv(flagged_path, index=False, quoting=csv.QUOTE_MINIMAL)

    catalog = pd.DataFrame(pick_model_inputs(inv))
    catalog_path = REPORTS_DIR / "model_data_catalog.csv"
    catalog.to_csv(catalog_path, index=False, quoting=csv.QUOTE_MINIMAL)

    copied = copy_model_ready(catalog)
    merged_stats = build_training_tables(catalog)

    ext_counts = inv["kind"].value_counts().to_dict()
    csv_total = int((inv["kind"] == "csv").sum())
    csv_flagged = int(((inv["kind"] == "csv") & (inv["quality_flags"] != "")).sum())

    lines = [
        "# Data Audit Summary",
        "",
        f"- Scanned files: {len(inv)}",
        f"- CSV files: {csv_total}",
        f"- CSV files with quality flags: {csv_flagged}",
        f"- File types: {json.dumps(ext_counts, sort_keys=True)}",
        "",
        "## Key Model Data Picks",
    ]
    for _, row in catalog.iterrows():
        lines.append(
            f"- {row['role']}: `{row['path']}` "
            f"(exists={row['exists']}, rows={row['rows']}, flags={row['quality_flags']})"
        )

    lines.extend(
        [
            "",
            "## Copied To model_ready",
        ]
    )
    if copied:
        for p in copied:
            lines.append(f"- `{p}`")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Unified Training Tables",
        ]
    )
    if merged_stats:
        lines.append(f"- `model_ready/final_training_table_2019_hourly.csv` rows={merged_stats['merged_rows']}")
        lines.append(
            "- `model_ready/final_training_table_2019_hourly_with_power.csv` "
            f"rows={merged_stats['merged_with_power_rows']}"
        )
    else:
        lines.append("- not built (missing core source files)")

    lines.extend(
        [
            "",
            "## Suggested Cleanup Actions",
            "- Exclude files flagged `synthetic_data` from headline model training/evaluation.",
            "- Exclude files flagged `test_artifact` and `intermediate_export` from production ingestion.",
            "- Prefer UTF-8 files under `cleaned/` or `model_ready/` for pipeline reads.",
        ]
    )

    summary_path = REPORTS_DIR / "DATA_AUDIT_SUMMARY.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {inv_path}")
    print(f"Wrote: {flagged_path}")
    print(f"Wrote: {catalog_path}")
    print(f"Wrote: {summary_path}")
    print(f"Copied model_ready files: {len(copied)}")


if __name__ == "__main__":
    main()
