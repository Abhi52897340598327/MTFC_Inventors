#!/usr/bin/env python3
"""
Generate a reference feature-correlation matrix from Data_Sources.

This script avoids third-party dependencies so it can run in minimal
environments. It builds a monthly feature table from available energy/grid/
weather datasets, then computes pairwise Pearson correlations.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class StatsAgg:
    count: int = 0
    total: float = 0.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def add(self, value: float) -> None:
        if value is None or math.isnan(value):
            return
        self.count += 1
        self.total += value
        self.min_value = value if self.min_value is None else min(self.min_value, value)
        self.max_value = value if self.max_value is None else max(self.max_value, value)

    def mean(self) -> Optional[float]:
        if self.count == 0:
            return None
        return self.total / self.count


def safe_float(raw: str) -> Optional[float]:
    if raw is None:
        return None
    s = str(raw).strip().replace(",", "")
    if s == "" or s.lower() in {"na", "nan", "none"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def month_key_from_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m")


def sanitize_name(name: str) -> str:
    out = []
    prev_underscore = False
    for ch in name.lower():
        if ch.isalnum():
            out.append(ch)
            prev_underscore = False
        elif not prev_underscore:
            out.append("_")
            prev_underscore = True
    cleaned = "".join(out).strip("_")
    return cleaned or "feature"


def parse_monthly_table_from_hourly(
    path: Path,
    timestamp_col: str,
    timestamp_format: str,
    value_col: str,
    feature_prefix: str,
) -> Dict[str, Dict[str, float]]:
    by_month = defaultdict(lambda: {"sum": StatsAgg(), "mean": StatsAgg(), "max": StatsAgg(), "min": StatsAgg()})
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_ts = row.get(timestamp_col, "")
            raw_val = row.get(value_col, "")
            value = safe_float(raw_val)
            if value is None:
                continue
            try:
                dt = datetime.strptime(raw_ts, timestamp_format)
            except ValueError:
                continue
            mk = month_key_from_dt(dt)
            by_month[mk]["sum"].add(value)
            by_month[mk]["mean"].add(value)
            by_month[mk]["max"].add(value)
            by_month[mk]["min"].add(value)

    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    for mk, stats in by_month.items():
        out[mk][f"{feature_prefix}_sum"] = stats["sum"].total
        out[mk][f"{feature_prefix}_mean"] = stats["mean"].mean()
        out[mk][f"{feature_prefix}_max"] = stats["max"].max_value
        out[mk][f"{feature_prefix}_min"] = stats["min"].min_value
        out[mk][f"{feature_prefix}_count"] = float(stats["mean"].count)
    return out


def merge_feature_rows(
    target: Dict[str, Dict[str, float]],
    source: Dict[str, Dict[str, float]],
) -> None:
    for mk, feat_map in source.items():
        target[mk].update(feat_map)


def load_hrl_metered_monthly(root: Path) -> Dict[str, Dict[str, float]]:
    files = [
        "hrl_load_metered (5).csv",
        "hrl_load_metered (4).csv",
        "hrl_load_metered (3).csv",
        "hrl_load_metered (2).csv",
        "hrl_load_metered (1).csv",
        "hrl_load_metered.csv",
    ]
    seen = set()
    by_month = defaultdict(lambda: {"sum": StatsAgg(), "mean": StatsAgg(), "max": StatsAgg(), "min": StatsAgg()})
    for name in files:
        path = root / name
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw_ts = row.get("datetime_beginning_utc", "")
                raw_val = row.get("mw", "")
                if not raw_ts or raw_ts in seen:
                    continue
                seen.add(raw_ts)
                value = safe_float(raw_val)
                if value is None:
                    continue
                try:
                    dt = datetime.strptime(raw_ts, "%m/%d/%Y %I:%M:%S %p")
                except ValueError:
                    continue
                mk = month_key_from_dt(dt)
                by_month[mk]["sum"].add(value)
                by_month[mk]["mean"].add(value)
                by_month[mk]["max"].add(value)
                by_month[mk]["min"].add(value)

    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    for mk, stats in by_month.items():
        out[mk]["hrl_rto_load_mw_sum"] = stats["sum"].total
        out[mk]["hrl_rto_load_mw_mean"] = stats["mean"].mean()
        out[mk]["hrl_rto_load_mw_max"] = stats["max"].max_value
        out[mk]["hrl_rto_load_mw_min"] = stats["min"].min_value
        out[mk]["hrl_rto_load_obs_count"] = float(stats["mean"].count)
    return out


def load_noaa_daily_monthly(path: Path) -> Dict[str, Dict[str, float]]:
    cols_mean = [
        ("avg_temp_f", "noaa_avg_temp_f_mean"),
        ("max_temp_f", "noaa_max_temp_f_mean"),
        ("min_temp_f", "noaa_min_temp_f_mean"),
        ("avg_relative_humidity_pct", "noaa_avg_relative_humidity_pct_mean"),
        ("avg_wind_speed_mph", "noaa_avg_wind_speed_mph_mean"),
        ("max_wind_speed_2min_mph", "noaa_max_wind_2min_mph_mean"),
        ("max_wind_speed_5sec_mph", "noaa_max_wind_5sec_mph_mean"),
    ]
    cols_sum = [
        ("precipitation_inches", "noaa_precipitation_inches_sum"),
        ("snowfall_inches", "noaa_snowfall_inches_sum"),
        ("snow_depth_inches", "noaa_snow_depth_inches_sum"),
    ]
    month_aggs: Dict[str, Dict[str, StatsAgg]] = defaultdict(lambda: defaultdict(StatsAgg))
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_date = row.get("date", "")
            try:
                dt = datetime.strptime(raw_date, "%Y-%m-%d")
            except ValueError:
                continue
            mk = month_key_from_dt(dt)

            for src_col, out_col in cols_mean:
                value = safe_float(row.get(src_col))
                if value is not None:
                    month_aggs[mk][out_col].add(value)

            for src_col, out_col in cols_sum:
                value = safe_float(row.get(src_col))
                if value is not None:
                    month_aggs[mk][out_col].add(value)

            avg_temp = safe_float(row.get("avg_temp_f"))
            if avg_temp is not None:
                cdd = max(0.0, avg_temp - 65.0)
                month_aggs[mk]["noaa_cdd65_f_day_sum"].add(cdd)

    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    for mk, agg_map in month_aggs.items():
        for out_col, agg in agg_map.items():
            if out_col.endswith("_sum"):
                out[mk][out_col] = agg.total
            else:
                out[mk][out_col] = agg.mean()
        out[mk]["noaa_daily_obs_count"] = float(agg_map["noaa_avg_temp_f_mean"].count)
    return out


def load_ashburn_2019_hourly_monthly(path: Path) -> Dict[str, Dict[str, float]]:
    by_month = defaultdict(StatsAgg)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_ts = row.get("timestamp", "")
            temp_c = safe_float(row.get("temperature_c"))
            if temp_c is None:
                continue
            try:
                dt = datetime.strptime(raw_ts, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            mk = month_key_from_dt(dt)
            by_month[mk].add(temp_c)

    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    for mk, agg in by_month.items():
        out[mk]["ashburn_2019_temp_c_mean"] = agg.mean()
        out[mk]["ashburn_2019_temp_c_max"] = agg.max_value
        out[mk]["ashburn_2019_temp_c_min"] = agg.min_value
        out[mk]["ashburn_2019_temp_obs_count"] = float(agg.count)
    return out


def load_ashburn_2024_daily_monthly(path: Path) -> Dict[str, Dict[str, float]]:
    by_month = defaultdict(lambda: defaultdict(StatsAgg))
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_date = row.get("date", "")
            dtype = (row.get("datatype") or "").strip().upper()
            value = safe_float(row.get("value"))
            if value is None or dtype not in {"TAVG", "TMAX", "TMIN"}:
                continue
            try:
                dt = datetime.strptime(raw_date[:10], "%Y-%m-%d")
            except ValueError:
                continue
            mk = month_key_from_dt(dt)
            by_month[mk][dtype].add(value)

    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    for mk, dtype_map in by_month.items():
        for dtype, agg in dtype_map.items():
            out[mk][f"ashburn_2024_{dtype.lower()}_c_mean"] = agg.mean()
            out[mk][f"ashburn_2024_{dtype.lower()}_c_max"] = agg.max_value
            out[mk][f"ashburn_2024_{dtype.lower()}_c_min"] = agg.min_value
            out[mk][f"ashburn_2024_{dtype.lower()}_obs_count"] = float(agg.count)
    return out


def load_google_cluster_monthly(path: Path) -> Dict[str, Dict[str, float]]:
    cols = [
        ("avg_cpu_utilization", "google_cluster_avg_cpu_utilization"),
        ("num_tasks_sampled", "google_cluster_num_tasks_sampled"),
        ("hour_of_day", "google_cluster_hour_of_day"),
    ]
    by_month = defaultdict(lambda: defaultdict(StatsAgg))
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_ts = row.get("real_timestamp", "")
            try:
                dt = datetime.strptime(raw_ts[:19], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            mk = month_key_from_dt(dt)
            for src_col, name in cols:
                value = safe_float(row.get(src_col))
                if value is not None:
                    by_month[mk][name].add(value)

    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    for mk, agg_map in by_month.items():
        for name, agg in agg_map.items():
            out[mk][f"{name}_mean"] = agg.mean()
            out[mk][f"{name}_max"] = agg.max_value
            out[mk][f"{name}_min"] = agg.min_value
        out[mk]["google_cluster_obs_count"] = float(agg_map["google_cluster_avg_cpu_utilization"].count)
    return out


def load_monthly_spending(path: Path) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (row.get("Entity") or "").strip() != "United States":
                continue
            raw_day = row.get("Day", "")
            value = safe_float(row.get("Monthly spending on data center construction in the United States"))
            if value is None:
                continue
            try:
                dt = datetime.strptime(raw_day, "%Y-%m-%d")
            except ValueError:
                continue
            mk = month_key_from_dt(dt)
            out[mk]["us_data_center_construction_spending_usd"] = value
    return out


def load_owid_va_energy_annual_as_monthly(path: Path) -> Dict[str, Dict[str, float]]:
    keep_types = {
        "total",
        "fossil",
        "lowcarbon",
        "nuclear",
        "renewables-including-hydro",
        "renewables-except-hydro",
        "wind-and-solar",
    }
    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        years = header[3:]
        for row in reader:
            if len(row) < 4:
                continue
            geo_code = row[0].strip()
            series_type = row[2].strip()
            if geo_code != "US-VA" or series_type not in keep_types:
                continue
            feature = f"owid_va_{sanitize_name(series_type)}_twh_yearly"
            values = row[3:]
            for year, raw_val in zip(years, values):
                value = safe_float(raw_val)
                if value is None:
                    continue
                for month in range(1, 13):
                    mk = f"{year}-{month:02d}"
                    out[mk][feature] = value
    return out


def load_pjm_generation_by_fuel(path: Path) -> Dict[str, Dict[str, float]]:
    month_fuel_agg = defaultdict(lambda: defaultdict(StatsAgg))
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_ts = row.get("period", "")
            fuel = sanitize_name(row.get("fueltype", "unknown"))
            value = safe_float(row.get("value"))
            if value is None:
                continue
            try:
                dt = datetime.strptime(raw_ts, "%Y-%m-%dT%H")
            except ValueError:
                continue
            mk = month_key_from_dt(dt)
            month_fuel_agg[mk][fuel].add(value)

    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    for mk, fuel_map in month_fuel_agg.items():
        for fuel, agg in fuel_map.items():
            out[mk][f"pjm_gen_fuel_{fuel}_mwh_sum"] = agg.total
            out[mk][f"pjm_gen_fuel_{fuel}_mwh_mean"] = agg.mean()
        out[mk]["pjm_gen_fuel_obs_count"] = float(sum(a.count for a in fuel_map.values()))
    return out


def load_noaa_gsod_2019(path: Path) -> Dict[str, Dict[str, float]]:
    cols = [
        ("temp", "noaa_gsod_temp_f_mean"),
        ("dewp", "noaa_gsod_dewpoint_f_mean"),
        ("max", "noaa_gsod_max_temp_f_mean"),
        ("min", "noaa_gsod_min_temp_f_mean"),
        ("prcp", "noaa_gsod_precip_in_sum"),
        ("sndp", "noaa_gsod_snow_depth_in_sum"),
    ]
    by_month = defaultdict(lambda: defaultdict(StatsAgg))
    with path.open(newline="", encoding="utf-16") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                year = int((row.get("year") or "").strip())
                month = int((row.get("mo") or "").strip())
                day = int((row.get("da") or "").strip())
                dt = datetime(year, month, day)
            except (TypeError, ValueError):
                continue
            mk = month_key_from_dt(dt)
            for src, out_name in cols:
                value = safe_float(row.get(src))
                if value is None:
                    continue
                if src == "sndp" and value == 999.9:
                    continue
                by_month[mk][out_name].add(value)

    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    for mk, agg_map in by_month.items():
        for out_name, agg in agg_map.items():
            if out_name.endswith("_sum"):
                out[mk][out_name] = agg.total
            else:
                out[mk][out_name] = agg.mean()
        out[mk]["noaa_gsod_daily_obs_count"] = float(agg_map["noaa_gsod_temp_f_mean"].count)
    return out


def pearson_corr(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for x, y in zip(xs, ys):
        dx = x - mean_x
        dy = y - mean_y
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy
    if den_x <= 0.0 or den_y <= 0.0:
        return None
    return num / math.sqrt(den_x * den_y)


def compute_correlation_matrix(
    rows_by_month: Dict[str, Dict[str, float]],
    min_overlap: int = 24,
) -> Tuple[List[str], Dict[Tuple[str, str], Optional[float]], Dict[Tuple[str, str], int]]:
    features = sorted(
        {
            feature
            for feat_map in rows_by_month.values()
            for feature, value in feat_map.items()
            if value is not None and not math.isnan(value)
        }
    )

    matrix: Dict[Tuple[str, str], Optional[float]] = {}
    overlap: Dict[Tuple[str, str], int] = {}
    for i, fi in enumerate(features):
        for j, fj in enumerate(features):
            if j < i:
                matrix[(fi, fj)] = matrix[(fj, fi)]
                overlap[(fi, fj)] = overlap[(fj, fi)]
                continue
            xs: List[float] = []
            ys: List[float] = []
            for mk in rows_by_month:
                vi = rows_by_month[mk].get(fi)
                vj = rows_by_month[mk].get(fj)
                if vi is None or vj is None:
                    continue
                if math.isnan(vi) or math.isnan(vj):
                    continue
                xs.append(vi)
                ys.append(vj)
            overlap[(fi, fj)] = len(xs)
            if fi == fj:
                matrix[(fi, fj)] = 1.0 if len(xs) > 1 else None
            elif len(xs) < min_overlap:
                matrix[(fi, fj)] = None
            else:
                matrix[(fi, fj)] = pearson_corr(xs, ys)
    return features, matrix, overlap


def write_feature_table(path: Path, rows_by_month: Dict[str, Dict[str, float]]) -> None:
    months = sorted(rows_by_month.keys())
    features = sorted({k for row in rows_by_month.values() for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["month"] + features)
        for mk in months:
            row = [mk]
            feat_map = rows_by_month[mk]
            for f in features:
                v = feat_map.get(f)
                row.append("" if v is None else f"{v:.10g}")
            writer.writerow(row)


def write_corr_matrix_csv(
    path: Path,
    features: List[str],
    matrix: Dict[Tuple[str, str], Optional[float]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature"] + features)
        for fi in features:
            row = [fi]
            for fj in features:
                corr = matrix.get((fi, fj))
                row.append("" if corr is None else f"{corr:.10g}")
            writer.writerow(row)


def write_overlap_matrix_csv(
    path: Path,
    features: List[str],
    overlap: Dict[Tuple[str, str], int],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature"] + features)
        for fi in features:
            row = [fi]
            for fj in features:
                row.append(str(overlap.get((fi, fj), 0)))
            writer.writerow(row)


def write_importance_proxy(
    path: Path,
    features: List[str],
    matrix: Dict[Tuple[str, str], Optional[float]],
) -> None:
    rows = []
    for fi in features:
        vals = []
        for fj in features:
            if fi == fj:
                continue
            corr = matrix.get((fi, fj))
            if corr is None:
                continue
            vals.append(abs(corr))
        if not vals:
            rows.append((fi, "", "", 0))
            continue
        rows.append((fi, sum(vals) / len(vals), max(vals), len(vals)))
    rows.sort(key=lambda x: (x[1] if x[1] != "" else -1), reverse=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "mean_abs_corr", "max_abs_corr", "valid_pair_count"])
        for feature, mean_abs, max_abs, count in rows:
            writer.writerow(
                [
                    feature,
                    "" if mean_abs == "" else f"{mean_abs:.10g}",
                    "" if max_abs == "" else f"{max_abs:.10g}",
                    count,
                ]
            )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "Data_Sources"
    out_dir = repo_root / "data" / "analysis_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_by_month: Dict[str, Dict[str, float]] = defaultdict(dict)

    # Core demand and weather features.
    merge_feature_rows(rows_by_month, load_hrl_metered_monthly(data_dir))
    merge_feature_rows(
        rows_by_month,
        parse_monthly_table_from_hourly(
            data_dir / "pjm_hourly_demand_2019_2024_eia.csv",
            timestamp_col="datetime_utc",
            timestamp_format="%Y-%m-%dT%H",
            value_col="demand_mwh",
            feature_prefix="pjm_hourly_demand_mwh",
        ),
    )
    merge_feature_rows(
        rows_by_month,
        parse_monthly_table_from_hourly(
            data_dir / "pjm_net_generation_2019_2024_eia.csv",
            timestamp_col="period",
            timestamp_format="%Y-%m-%dT%H",
            value_col="value",
            feature_prefix="pjm_net_generation_mwh",
        ),
    )
    merge_feature_rows(
        rows_by_month,
        parse_monthly_table_from_hourly(
            data_dir / "pjm_demand_forecast_2019_2024_eia.csv",
            timestamp_col="period",
            timestamp_format="%Y-%m-%dT%H",
            value_col="value",
            feature_prefix="pjm_day_ahead_demand_forecast_mwh",
        ),
    )
    merge_feature_rows(
        rows_by_month,
        parse_monthly_table_from_hourly(
            data_dir / "pjm_interchange_2019_2024_eia.csv",
            timestamp_col="period",
            timestamp_format="%Y-%m-%dT%H",
            value_col="value",
            feature_prefix="pjm_total_interchange_mwh",
        ),
    )
    merge_feature_rows(rows_by_month, load_pjm_generation_by_fuel(data_dir / "pjm_generation_by_fuel_2019_2024_eia.csv"))
    merge_feature_rows(rows_by_month, load_noaa_daily_monthly(data_dir / "noaa_dulles_daily_2015_2024.csv"))

    # Additional reference features where available.
    ashburn_2019 = data_dir / "ashburn_va_temperature_2019.csv"
    if ashburn_2019.exists():
        merge_feature_rows(rows_by_month, load_ashburn_2019_hourly_monthly(ashburn_2019))
    ashburn_2024 = data_dir / "ashburn_va_temperature_2024.csv"
    if ashburn_2024.exists():
        merge_feature_rows(rows_by_month, load_ashburn_2024_daily_monthly(ashburn_2024))
    cluster = data_dir / "google_cluster_utilization_2019.csv"
    if cluster.exists():
        merge_feature_rows(rows_by_month, load_google_cluster_monthly(cluster))
    spending = data_dir / "monthly-spending-data-center-us.csv"
    if spending.exists():
        merge_feature_rows(rows_by_month, load_monthly_spending(spending))
    owid_va = data_dir / "data-including-net-imports.csv"
    if owid_va.exists():
        merge_feature_rows(rows_by_month, load_owid_va_energy_annual_as_monthly(owid_va))
    gsod = data_dir / "noaa_gsod_dulles_2019.csv"
    if gsod.exists():
        merge_feature_rows(rows_by_month, load_noaa_gsod_2019(gsod))

    features, matrix, overlap = compute_correlation_matrix(rows_by_month, min_overlap=24)

    feature_table_path = out_dir / "all_data_sources_monthly_feature_table.csv"
    corr_path = out_dir / "all_data_sources_feature_correlation_matrix.csv"
    overlap_path = out_dir / "all_data_sources_feature_correlation_overlap_counts.csv"
    importance_path = out_dir / "all_data_sources_feature_importance_proxy.csv"

    write_feature_table(feature_table_path, rows_by_month)
    write_corr_matrix_csv(corr_path, features, matrix)
    write_overlap_matrix_csv(overlap_path, features, overlap)
    write_importance_proxy(importance_path, features, matrix)

    total_months = len(rows_by_month)
    print(f"Wrote feature table: {feature_table_path}")
    print(f"Wrote correlation matrix: {corr_path}")
    print(f"Wrote overlap matrix: {overlap_path}")
    print(f"Wrote importance proxy: {importance_path}")
    print(f"Months covered: {total_months}")
    print(f"Features included: {len(features)}")


if __name__ == "__main__":
    main()
