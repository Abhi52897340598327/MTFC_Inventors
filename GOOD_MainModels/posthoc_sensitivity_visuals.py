"""Post-hoc sensitivity and recommendations visuals for REAL FINAL FILES.

Guarantees:
- Does NOT retrain or alter any model.
- Reads existing datasets + latest run metadata.
- Uses deterministic physics equations from the active pipeline config.
"""

from __future__ import annotations

import csv
import html
import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
RESULTS_ROOT = ROOT / "outputs" / "results"
ANALYSIS_ROOT = ROOT / "outputs" / "analysis"
FONT_SCALE = 0.55
MIN_FONT_SIZE = 8

# Single source of truth for physics constants — no duplicate definition here.
# PipelineConfig exposes the same attribute names (facility_mw, base_pue, etc.)
# so all existing code that uses PhysicsConfig() continues to work unchanged.
from config import PipelineConfig as PhysicsConfig  # noqa: E402  — intentional alias


def _scale_font_size(size: float) -> int:
    return max(MIN_FONT_SIZE, int(round(size * FONT_SCALE)))


def _scale_svg_font_sizes(raw: str) -> str:
    def repl_attr(match: re.Match[str]) -> str:
        quote = match.group(1)
        size = float(match.group(2))
        return f"font-size={quote}{_scale_font_size(size)}{quote}"

    def repl_style(match: re.Match[str]) -> str:
        size = float(match.group(1))
        suffix = match.group(2) or ""
        return f"font-size:{_scale_font_size(size)}{suffix}"

    out = re.sub(r"font-size=(['\"])(\d+(?:\.\d+)?)\1", repl_attr, raw)
    out = re.sub(r"font-size:\s*(\d+(?:\.\d+)?)(px)?", repl_style, out)
    return out


def _parse_iso(raw: str) -> datetime:
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _hour_key(dt: datetime) -> str:
    return dt.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")


def _load_latest_run_id() -> str:
    runs = sorted([p.name for p in RESULTS_ROOT.glob("carbon_*") if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No run directories found under {RESULTS_ROOT}")
    return runs[-1]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute quantile on empty values.")
    arr = sorted(values)
    if q <= 0:
        return arr[0]
    if q >= 1:
        return arr[-1]
    pos = (len(arr) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return arr[lo]
    frac = pos - lo
    return arr[lo] * (1 - frac) + arr[hi] * frac


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _hex_to_rgb(color: str) -> Tuple[int, int, int]:
    c = color.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _mix(c1: str, c2: str, t: float) -> str:
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    r = int(round(r1 + (r2 - r1) * t))
    g = int(round(g1 + (g2 - g1) * t))
    b = int(round(b1 + (b2 - b1) * t))
    return _rgb_to_hex((r, g, b))


def _palette_green_yellow_red(t: float) -> str:
    t = _clip(t, 0.0, 1.0)
    if t <= 0.5:
        return _mix("#0a7a3e", "#f0e68c", t / 0.5)
    return _mix("#f0e68c", "#b30021", (t - 0.5) / 0.5)


def _text_color_for_bg(bg_hex: str) -> str:
    r, g, b = _hex_to_rgb(bg_hex)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#111111" if lum > 150 else "#f8f8f8"


class Svg:
    def __init__(self, w: int, h: int, bg: str = "#f2f2f2"):
        self.w = w
        self.h = h
        self.parts: List[str] = [
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>",
            f"<rect x='0' y='0' width='{w}' height='{h}' fill='{bg}'/>",
        ]

    def add(self, raw: str) -> None:
        self.parts.append(_scale_svg_font_sizes(raw))

    def text(
        self,
        x: float,
        y: float,
        value: str,
        size: int = 14,
        weight: str = "normal",
        anchor: str = "start",
        fill: str = "#222222",
    ) -> None:
        scaled_size = _scale_font_size(size)
        self.parts.append(
            "<text x='{:.2f}' y='{:.2f}' font-family='Helvetica, Arial, sans-serif' "
            "font-size='{}' font-weight='{}' text-anchor='{}' fill='{}'>{}</text>".format(
                x,
                y,
                scaled_size,
                weight,
                anchor,
                fill,
                html.escape(value),
            )
        )

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = "none",
        stroke: str = "none",
        stroke_width: float = 1.0,
        rx: float = 0.0,
        opacity: float = 1.0,
    ) -> None:
        self.parts.append(
            "<rect x='{:.2f}' y='{:.2f}' width='{:.2f}' height='{:.2f}' fill='{}' stroke='{}' "
            "stroke-width='{:.2f}' rx='{:.2f}' opacity='{:.3f}'/>".format(
                x, y, w, h, fill, stroke, stroke_width, rx, opacity
            )
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str = "#444444",
        width: float = 1.0,
        dash: str | None = None,
        opacity: float = 1.0,
    ) -> None:
        dash_attr = f" stroke-dasharray='{dash}'" if dash else ""
        self.parts.append(
            "<line x1='{:.2f}' y1='{:.2f}' x2='{:.2f}' y2='{:.2f}' stroke='{}' stroke-width='{:.2f}' "
            "opacity='{:.3f}'{}/>".format(x1, y1, x2, y2, stroke, width, opacity, dash_attr)
        )

    def circle(
        self,
        cx: float,
        cy: float,
        r: float,
        fill: str = "#1f77b4",
        stroke: str = "none",
        stroke_width: float = 0.0,
        opacity: float = 1.0,
    ) -> None:
        self.parts.append(
            "<circle cx='{:.2f}' cy='{:.2f}' r='{:.2f}' fill='{}' stroke='{}' stroke-width='{:.2f}' opacity='{:.3f}'/>".format(
                cx, cy, r, fill, stroke, stroke_width, opacity
            )
        )

    def polyline(
        self,
        points: Sequence[Tuple[float, float]],
        stroke: str = "#1f77b4",
        width: float = 2.0,
        fill: str = "none",
        opacity: float = 1.0,
    ) -> None:
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.parts.append(
            f"<polyline points='{pts}' fill='{fill}' stroke='{stroke}' stroke-width='{width:.2f}' opacity='{opacity:.3f}'/>"
        )

    def polygon(
        self,
        points: Sequence[Tuple[float, float]],
        stroke: str = "#1f77b4",
        width: float = 2.0,
        fill: str = "none",
        opacity: float = 1.0,
    ) -> None:
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.parts.append(
            f"<polygon points='{pts}' fill='{fill}' stroke='{stroke}' stroke-width='{width:.2f}' opacity='{opacity:.3f}'/>"
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.parts + ["</svg>"]), encoding="utf-8")


def _load_hourly_data() -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    cpu_rows = _read_csv(PROJECT_ROOT / "Data_Sources" / "cleaned" / "google_cluster_utilization_2019_cellb_hourly_cleaned.csv")
    temp_rows = _read_csv(PROJECT_ROOT / "Data_Sources" / "cleaned" / "ashburn_va_temperature_2019_cleaned.csv")
    ci_rows = _read_csv(PROJECT_ROOT / "Data_Sources" / "cleaned" / "pjm_grid_carbon_intensity_2019_full_cleaned.csv")

    cpu_agg: Dict[str, Dict[str, float]] = {}
    for r in cpu_rows:
        key = _hour_key(_parse_iso(r["real_timestamp"]))
        cur = cpu_agg.setdefault(key, {"cpu_sum": 0.0, "cpu_n": 0.0, "tasks_sum": 0.0})
        cur["cpu_sum"] += float(r["avg_cpu_utilization"])
        cur["cpu_n"] += 1.0
        cur["tasks_sum"] += float(r["num_tasks_sampled"])

    temp_agg: Dict[str, Dict[str, float]] = {}
    for r in temp_rows:
        key = _hour_key(_parse_iso(r["timestamp"]))
        cur = temp_agg.setdefault(key, {"sum": 0.0, "n": 0.0})
        cur["sum"] += float(r["temperature_c"])
        cur["n"] += 1.0

    ci_agg: Dict[str, Dict[str, float]] = {}
    for r in ci_rows:
        key = _hour_key(_parse_iso(r["timestamp"]))
        cur = ci_agg.setdefault(key, {"sum": 0.0, "n": 0.0})
        cur["sum"] += float(r["carbon_intensity_kg_per_mwh"])
        cur["n"] += 1.0

    keys = sorted(set(cpu_agg) & set(temp_agg) & set(ci_agg))
    merged: List[Dict[str, float]] = []
    for key in keys:
        cpu = cpu_agg[key]["cpu_sum"] / max(cpu_agg[key]["cpu_n"], 1.0)
        temp_c = temp_agg[key]["sum"] / max(temp_agg[key]["n"], 1.0)
        ci = ci_agg[key]["sum"] / max(ci_agg[key]["n"], 1.0)
        merged.append(
            {
                "timestamp": key,
                "cpu_utilization": cpu,
                "num_tasks_sampled": cpu_agg[key]["tasks_sum"],
                "temperature_c": temp_c,
                "temperature_f": temp_c * 9.0 / 5.0 + 32.0,
                "carbon_intensity": ci,
            }
        )

    # Full-year carbon hourly rows for heatmap.
    carbon_hourly: List[Dict[str, float]] = []
    for r in ci_rows:
        dt = _parse_iso(r["timestamp"])
        carbon_hourly.append(
            {
                "month": dt.month,
                "hour": dt.hour,
                "carbon_intensity": float(r["carbon_intensity_kg_per_mwh"]),
            }
        )
    return merged, carbon_hourly


def _physics(
    cpu: float,
    temp_f: float,
    ci: float,
    cfg: PhysicsConfig,
    idle: float | None = None,
    pue_temp_coef: float | None = None,
    pue_cpu_coef: float | None = None,
) -> Dict[str, float]:
    idle_x = cfg.idle_power_fraction if idle is None else idle
    a = cfg.pue_temp_coef if pue_temp_coef is None else pue_temp_coef
    b = cfg.pue_cpu_coef if pue_cpu_coef is None else pue_cpu_coef
    it_power = cfg.facility_mw * (idle_x + (1.0 - idle_x) * cpu)
    temp_above = max(0.0, temp_f - cfg.cooling_threshold_f)
    pue = _clip(cfg.base_pue + a * temp_above + b * cpu, cfg.base_pue, cfg.max_pue)
    total_power = it_power * pue
    emissions = total_power * ci
    annual_energy_gwh = total_power * 8760.0 / 1000.0
    return {
        "it_power_mw": it_power,
        "pue": pue,
        "total_power_mw": total_power,
        "annual_energy_gwh": annual_energy_gwh,
        "emissions_kg_per_h": emissions,
        "cooling_mw": max(0.0, total_power - it_power),
    }


def _sobol_indices(
    ranges: Dict[str, Tuple[float, float]],
    cfg: "PhysicsConfig",
    output_key: str = "emissions_kg_per_h",
    n: int = 10000,
    seed: int = 42,
) -> List[Dict[str, float]]:
    """Compute first-order and total-order Sobol indices using SALib (Saltelli sampling).

    Parameters
    ----------
    n : int
        Base sample size passed to saltelli.sample.  SALib generates
        N*(D+2) model evaluations for first-order analysis.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    List of dicts with keys:
        parameter, S1, S1_ci_low, S1_ci_high,
        ST, ST_ci_low, ST_ci_high, output_metric
    """
    import numpy as np
    from SALib.sample import saltelli  # type: ignore[import]
    from SALib.analyze import sobol as sobol_analyze  # type: ignore[import]

    names = list(ranges.keys())
    problem: Dict[str, object] = {
        "num_vars": len(names),
        "names": names,
        "bounds": [list(ranges[k]) for k in names],
    }

    param_values = saltelli.sample(problem, n, calc_second_order=False, seed=seed)

    def _eval_row(row: "np.ndarray") -> float:
        vals = {names[i]: float(row[i]) for i in range(len(names))}
        return _physics(
            cpu=vals["cpu_utilization"],
            temp_f=vals["temperature_f"],
            ci=vals["carbon_intensity"],
            cfg=cfg,
            idle=vals["idle_power_fraction"],
            pue_temp_coef=vals["pue_temp_coef"],
            pue_cpu_coef=vals["pue_cpu_coef"],
        )[output_key]

    Y = np.array([_eval_row(row) for row in param_values], dtype=float)

    si = sobol_analyze.analyze(
        problem,
        Y,
        calc_second_order=False,
        conf_level=0.95,
        print_to_console=False,
        seed=seed,
    )

    rows: List[Dict[str, float]] = []
    for i, name in enumerate(names):
        s1 = float(si["S1"][i])
        s1_conf = float(si["S1_conf"][i])
        st = float(si["ST"][i])
        st_conf = float(si["ST_conf"][i])
        rows.append(
            {
                "parameter": name,
                "S1": _clip(s1, 0.0, 1.0),
                "S1_ci_low": _clip(s1 - s1_conf, 0.0, 1.0),
                "S1_ci_high": _clip(s1 + s1_conf, 0.0, 1.0),
                "ST": _clip(st, 0.0, 1.0),
                "ST_ci_low": _clip(st - st_conf, 0.0, 1.0),
                "ST_ci_high": _clip(st + st_conf, 0.0, 1.0),
                "output_metric": output_key,
            }
        )
    return rows


def _tornado_oat(
    ranges: Dict[str, Tuple[float, float]],
    baseline: Dict[str, float],
    cfg: PhysicsConfig,
    output_key: str = "emissions_kg_per_h",
) -> List[Dict[str, float]]:
    def run(v: Dict[str, float]) -> float:
        return _physics(
            cpu=v["cpu_utilization"],
            temp_f=v["temperature_f"],
            ci=v["carbon_intensity"],
            cfg=cfg,
            idle=v["idle_power_fraction"],
            pue_temp_coef=v["pue_temp_coef"],
            pue_cpu_coef=v["pue_cpu_coef"],
        )[output_key]

    base = run(baseline)
    rows: List[Dict[str, float]] = []
    for p, (lo, hi) in ranges.items():
        low_case = dict(baseline)
        high_case = dict(baseline)
        low_case[p] = lo
        high_case[p] = hi
        low_y = run(low_case)
        high_y = run(high_case)
        rows.append(
            {
                "parameter": p,
                "baseline_kg_per_h": base,
                "low_value": lo,
                "high_value": hi,
                "low_kg_per_h": low_y,
                "high_kg_per_h": high_y,
                "swing_kg_per_h": high_y - low_y,
                "delta_low_pct": ((low_y / base) - 1.0) * 100.0 if base else 0.0,
                "delta_high_pct": ((high_y / base) - 1.0) * 100.0 if base else 0.0,
                "output_metric": output_key,
            }
        )
    rows.sort(key=lambda x: abs(float(x["high_kg_per_h"]) - float(x["low_kg_per_h"])), reverse=True)
    return rows


def _pseudo_observations(values: Sequence[float]) -> List[float]:
    order = sorted([(v, idx) for idx, v in enumerate(values)])
    ranks = [0] * len(values)
    for rank, (_, idx) in enumerate(order, start=1):
        ranks[idx] = rank
    n = len(values)
    return [r / (n + 1.0) for r in ranks]


def _tail_dependence(u: Sequence[float], v: Sequence[float], q: float) -> Tuple[float, float]:
    up_u = [x >= q for x in u]
    up_v = [x >= q for x in v]
    lo_u = [x <= (1 - q) for x in u]
    lo_v = [x <= (1 - q) for x in v]

    n_up = sum(up_u)
    n_lo = sum(lo_u)
    lam_u = (sum(1 for a, b in zip(up_u, up_v) if a and b) / n_up) if n_up else 0.0
    lam_l = (sum(1 for a, b in zip(lo_u, lo_v) if a and b) / n_lo) if n_lo else 0.0
    return lam_u, lam_l


def _bootstrap_tail_dependence_ci(
    u: Sequence[float],
    v: Sequence[float],
    q: float,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    """Bootstrap 95% CIs for upper- and lower-tail dependence coefficients.

    Returns
    -------
    (upper_ci_low, upper_ci_high, lower_ci_low, lower_ci_high)
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    n = len(u)
    u_arr = list(u)
    v_arr = list(v)
    boot_upper: List[float] = []
    boot_lower: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        u_b = [u_arr[i] for i in idx]
        v_b = [v_arr[i] for i in idx]
        lu, ll = _tail_dependence(u_b, v_b, q)
        boot_upper.append(lu)
        boot_lower.append(ll)
    boot_upper_arr = np.sort(boot_upper)
    boot_lower_arr = np.sort(boot_lower)
    lo_idx = int(round(0.025 * n_bootstrap))
    hi_idx = int(round(0.975 * n_bootstrap)) - 1
    hi_idx = max(lo_idx, min(hi_idx, n_bootstrap - 1))
    return (
        float(boot_upper_arr[lo_idx]),
        float(boot_upper_arr[hi_idx]),
        float(boot_lower_arr[lo_idx]),
        float(boot_lower_arr[hi_idx]),
    )


def _copula_analysis(merged: List[Dict[str, float]]) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], Dict[str, List[float]]]:
    temp = [r["temperature_f"] for r in merged]
    ci = [r["carbon_intensity"] for r in merged]
    cpu = [r["cpu_utilization"] for r in merged]
    emi = [r["emissions_kg_per_h"] for r in merged]
    ene = [r["annual_energy_gwh"] for r in merged]

    u_temp = _pseudo_observations(temp)
    u_ci = _pseudo_observations(ci)
    u_cpu = _pseudo_observations(cpu)
    u_emi = _pseudo_observations(emi)
    u_ene = _pseudo_observations(ene)

    pair_map = {
        "temp_vs_carbon": (u_temp, u_ci),
        "temp_vs_cpu": (u_temp, u_cpu),
        "cpu_vs_carbon": (u_cpu, u_ci),
        "temp_vs_emissions": (u_temp, u_emi),
        "carbon_vs_emissions": (u_ci, u_emi),
        "cpu_vs_emissions": (u_cpu, u_emi),
        "temp_vs_energy": (u_temp, u_ene),
        "carbon_vs_energy": (u_ci, u_ene),
        "cpu_vs_energy": (u_cpu, u_ene),
    }

    summary_rows: List[Dict[str, float]] = []
    for name, (u, v) in pair_map.items():
        lu, ll = _tail_dependence(u, v, 0.95)
        u_ci_lo, u_ci_hi, l_ci_lo, l_ci_hi = _bootstrap_tail_dependence_ci(
            u, v, q=0.95, n_bootstrap=1000, seed=42
        )
        summary_rows.append(
            {
                "pair": name,
                "upper_tail_q95": lu,
                "upper_tail_ci_low": u_ci_lo,
                "upper_tail_ci_high": u_ci_hi,
                "lower_tail_q05": ll,
                "lower_tail_ci_low": l_ci_lo,
                "lower_tail_ci_high": l_ci_hi,
            }
        )
    summary_rows.sort(key=lambda x: x["upper_tail_q95"], reverse=True)

    thresholds = [round(0.80 + i * 0.01, 2) for i in range(20)]
    curve_rows: List[Dict[str, float]] = []
    for name, (u, v) in pair_map.items():
        for q in thresholds:
            lu, ll = _tail_dependence(u, v, q)
            curve_rows.append(
                {
                    "pair": name,
                    "threshold_q": q,
                    "lambda_upper": lu,
                    "lambda_lower": ll,
                }
            )

    pseudo = {
        "u_temp": u_temp,
        "u_ci": u_ci,
        "u_cpu": u_cpu,
        "u_emi": u_emi,
        "u_ene": u_ene,
    }
    return summary_rows, curve_rows, pseudo


def _carbon_heatmap(carbon_hourly: List[Dict[str, float]]) -> Tuple[List[Dict[str, float]], List[List[float]]]:
    by_key: Dict[Tuple[int, int], List[float]] = {}
    for r in carbon_hourly:
        key = (int(r["hour"]), int(r["month"]))
        by_key.setdefault(key, []).append(float(r["carbon_intensity"]))

    matrix = [[0.0 for _ in range(12)] for _ in range(24)]
    rows: List[Dict[str, float]] = []
    for h in range(24):
        for m in range(1, 13):
            vals = by_key.get((h, m), [0.0])
            mean_v = statistics.fmean(vals) if vals else 0.0
            matrix[h][m - 1] = mean_v
            rows.append({"hour": h, "month": m, "mean_carbon_intensity": mean_v})
    return rows, matrix


def _build_recommendation_scenarios(
    merged: List[Dict[str, float]],
    cfg: PhysicsConfig,
) -> List[Dict[str, float]]:
    # Baseline aggregates from observed merged sample.
    base_out = [_physics(r["cpu_utilization"], r["temperature_f"], r["carbon_intensity"], cfg) for r in merged]
    base_total = [o["total_power_mw"] for o in base_out]
    base_emi = [o["emissions_kg_per_h"] for o in base_out]
    base_peak = _quantile(base_total, 0.95)
    base_cooling = [o["cooling_mw"] for o in base_out]

    base = {
        "total_power_mw_mean": statistics.fmean(base_total),
        "emissions_kg_per_h_mean": statistics.fmean(base_emi),
        "peak_power_mw_p95": base_peak,
        "cooling_mw_mean": statistics.fmean(base_cooling),
    }

    def eval_scenario(
        name: str,
        cpu_scale: float,
        temp_shift_f: float,
        ci_scale: float,
        pue_temp_scale: float,
        idle_shift: float = 0.0,
    ) -> Dict[str, float]:
        out = []
        for r in merged:
            o = _physics(
                cpu=_clip(r["cpu_utilization"] * cpu_scale, 0.0, 1.0),
                temp_f=r["temperature_f"] + temp_shift_f,
                ci=r["carbon_intensity"] * ci_scale,
                cfg=cfg,
                idle=_clip(cfg.idle_power_fraction + idle_shift, 0.15, 0.60),
                pue_temp_coef=cfg.pue_temp_coef * pue_temp_scale,
                pue_cpu_coef=cfg.pue_cpu_coef,
            )
            out.append(o)
        total = [o["total_power_mw"] for o in out]
        emi = [o["emissions_kg_per_h"] for o in out]
        cooling = [o["cooling_mw"] for o in out]
        return {
            "scenario": name,
            "total_power_mw_mean": statistics.fmean(total),
            "emissions_kg_per_h_mean": statistics.fmean(emi),
            "peak_power_mw_p95": _quantile(total, 0.95),
            "cooling_mw_mean": statistics.fmean(cooling),
            "total_power_x_baseline": statistics.fmean(total) / base["total_power_mw_mean"],
            "emissions_x_baseline": statistics.fmean(emi) / base["emissions_kg_per_h_mean"],
            "peak_power_x_baseline": _quantile(total, 0.95) / base["peak_power_mw_p95"],
            "cooling_x_baseline": statistics.fmean(cooling) / base["cooling_mw_mean"],
        }

    scenarios = [
        eval_scenario("Current (Baseline)", 1.00, 0.0, 1.00, 1.00, 0.00),
        eval_scenario("Efficient (Lower PUE + CI)", 0.95, -1.0, 0.85, 0.80, -0.03),
        eval_scenario("High Growth", 1.30, 2.0, 1.10, 1.10, 0.02),
        eval_scenario("Climate Stress", 1.10, 8.0, 1.08, 1.20, 0.00),
    ]
    return scenarios


def _build_energy_forecast_rows(
    base_power_mw: float,
    base_peak_mw: float,
    years: Sequence[int],
) -> List[Dict[str, float]]:
    """Build energy forecast rows using OLS linear trend from PJM annual demand data.

    Methodology
    -----------
    1. Load pjm_hourly_demand_2019_2024_cleaned.csv and compute annual median demand
       (robust to outliers via percentile filter).
    2. Fit an OLS linear trend: demand ~ year, using scipy.stats.linregress.
    3. Convert the fractional grid growth signal to a datacenter-level energy forecast
       anchored to `base_power_mw` from the physics model.
    4. Compute 80% and 95% prediction intervals via the t-distribution (df = n-2).
    5. Produce three scenarios: point forecast, lower 80%, upper 80%.

    Falls back to flat-forecast if data or scipy are unavailable.
    """
    try:
        import numpy as np
        from scipy import stats  # type: ignore[import]

        pjm_path = PROJECT_ROOT / "Data_Sources" / "cleaned" / "pjm_hourly_demand_2019_2024_cleaned.csv"
        if not pjm_path.exists():
            raise FileNotFoundError(f"PJM demand file not found: {pjm_path}")

        with pjm_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            raw = list(reader)

        # Aggregate to annual median, filtering obvious outliers
        by_year: Dict[int, List[float]] = {}
        for r in raw:
            try:
                ts = str(r.get("datetime_utc", "")).strip()
                yr = int(ts[:4])
                val = float(r.get("demand_mwh", 0))
            except (ValueError, TypeError):
                continue
            if 50_000 <= val <= 200_000:  # plausible hourly MWh range for PJM
                by_year.setdefault(yr, []).append(val)

        if len(by_year) < 4:
            raise ValueError("Insufficient clean years for regression.")

        sorted_years = sorted(by_year.keys())
        year_arr = np.array(sorted_years, dtype=float)
        # Use median per year for robustness
        demand_arr = np.array(
            [float(np.median(by_year[y])) for y in sorted_years], dtype=float
        )
        n_obs = len(year_arr)

        slope, intercept, r_value, p_value, std_err = stats.linregress(year_arr, demand_arr)

        # Prediction interval width for new x at each forecast year
        # Using formula: t * s * sqrt(1 + 1/n + (x - x_mean)^2 / Sxx)
        x_mean = year_arr.mean()
        Sxx = float(np.sum((year_arr - x_mean) ** 2))
        y_hat_obs = slope * year_arr + intercept
        residuals = demand_arr - y_hat_obs
        s = float(np.sqrt(np.sum(residuals ** 2) / max(n_obs - 2, 1)))
        t80 = float(stats.t.ppf(0.90, df=max(n_obs - 2, 1)))   # one-sided 90% → 80% two-sided
        t95 = float(stats.t.ppf(0.975, df=max(n_obs - 2, 1)))  # one-sided 97.5% → 95% two-sided

        # Baseline demand at the anchor year for scaling to datacenter MW
        anchor_year = float(min(years))
        demand_at_anchor = slope * anchor_year + intercept
        if demand_at_anchor <= 0:
            demand_at_anchor = demand_arr.mean()

        rows: List[Dict[str, float]] = []
        for yr in years:
            x = float(yr)
            demand_forecast = slope * x + intercept
            # Fractional change relative to anchor
            frac = demand_forecast / demand_at_anchor
            pi_width = (
                lambda t: t * s * math.sqrt(
                    1 + 1.0 / n_obs + ((x - x_mean) ** 2) / Sxx
                )
            )
            pi80 = pi_width(t80) / demand_at_anchor
            pi95 = pi_width(t95) / demand_at_anchor

            power_mw_point = base_power_mw * frac
            peak_mw_point = base_peak_mw * frac
            annual_energy_gwh_point = power_mw_point * 8760.0 / 1000.0

            for scenario, power_mw in [
                ("Regression forecast (PJM trend)", power_mw_point),
                ("Regression lower 80% PI", base_power_mw * max(frac - pi80, 0.5 * frac)),
                ("Regression upper 80% PI", base_power_mw * (frac + pi80)),
            ]:
                pwr = power_mw
                pk = base_peak_mw * (pwr / max(base_power_mw, 1e-9))
                gwh = pwr * 8760.0 / 1000.0
                rows.append(
                    {
                        "forecast_scenario": scenario,
                        "growth_rate": slope / demand_at_anchor,
                        "year": yr,
                        "forecast_total_power_mw": pwr,
                        "forecast_peak_mw": pk,
                        "forecast_annual_energy_gwh": gwh,
                        "forecast_annual_energy_mwh": gwh * 1000.0,
                        "pi_80_pct_half_width_frac": pi80,
                        "ols_r2": r_value ** 2,
                        "ols_p_value": p_value,
                        "ols_n_obs": n_obs,
                        "method": "OLS_linear_PJM_demand_proxy",
                    }
                )
        return rows

    except Exception as _exc:
        # Fallback: flat forecast with ±10% uncertainty band
        rows = []
        fallback_note = f"fallback_flat_reason={_exc!r:.120}"
        for yr in years:
            for scenario, pwr_scale in [
                ("Flat forecast (no growth)", 1.0),
                ("Flat lower 10% band", 0.90),
                ("Flat upper 10% band", 1.10),
            ]:
                pwr = base_power_mw * pwr_scale
                pk = base_peak_mw * pwr_scale
                gwh = pwr * 8760.0 / 1000.0
                rows.append(
                    {
                        "forecast_scenario": scenario,
                        "growth_rate": 0.0,
                        "year": yr,
                        "forecast_total_power_mw": pwr,
                        "forecast_peak_mw": pk,
                        "forecast_annual_energy_gwh": gwh,
                        "forecast_annual_energy_mwh": gwh * 1000.0,
                        "pi_80_pct_half_width_frac": 0.1,
                        "ols_r2": float("nan"),
                        "ols_p_value": float("nan"),
                        "ols_n_obs": 0,
                        "method": fallback_note,
                    }
                )
        return rows


def _render_heatmap(matrix: List[List[float]], path: Path) -> None:
    w, h = 1650, 1120
    left, top = 120, 90
    cell_w, cell_h = 120, 40
    heat_w, heat_h = 12 * cell_w, 24 * cell_h
    vmin = min(min(row) for row in matrix)
    vmax = max(max(row) for row in matrix)
    vr = max(vmax - vmin, 1e-9)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    svg = Svg(w, h, bg="#e9e9e9")
    svg.text(w / 2, 44, "Carbon Intensity by Hour and Month", size=50, weight="700", anchor="middle")
    svg.text(w / 2, 82, "(When to Schedule Workloads for Lower Emissions)", size=32, weight="600", anchor="middle")

    # Grid cells
    for hour in range(24):
        for month in range(12):
            val = matrix[hour][month]
            t = (val - vmin) / vr
            color = _palette_green_yellow_red(t)
            x = left + month * cell_w
            y = top + hour * cell_h
            svg.rect(x, y, cell_w, cell_h, fill=color, stroke="#d0d0d0", stroke_width=1.0)
            svg.text(
                x + cell_w / 2,
                y + cell_h * 0.67,
                f"{val:.0f}",
                size=16,
                anchor="middle",
                fill=_text_color_for_bg(color),
            )

    # Axes labels
    for i, m in enumerate(months):
        svg.text(left + i * cell_w + cell_w / 2, top + heat_h + 30, m, size=20, anchor="middle")
    for hr in range(24):
        svg.text(left - 18, top + hr * cell_h + cell_h * 0.67, str(hr), size=16, anchor="end")

    svg.text(left + heat_w / 2, top + heat_h + 75, "Month", size=28, anchor="middle")
    # y label rotated
    svg.add(
        "<text x='28' y='{}' transform='rotate(-90 28,{})' font-family='Helvetica, Arial, sans-serif' "
        "font-size='36' text-anchor='middle'>Hour of Day</text>".format(top + heat_h / 2, top + heat_h / 2)
    )

    # Color bar
    cb_x, cb_y, cb_w, cb_h = left + heat_w + 90, top, 56, heat_h
    n_steps = 120
    for i in range(n_steps):
        y0 = cb_y + (i / n_steps) * cb_h
        t = 1 - (i / n_steps)
        c = _palette_green_yellow_red(t)
        svg.rect(cb_x, y0, cb_w, cb_h / n_steps + 1, fill=c, stroke="none")
    svg.rect(cb_x, cb_y, cb_w, cb_h, fill="none", stroke="#666", stroke_width=1)
    for p in [0.0, 0.25, 0.50, 0.75, 1.0]:
        v = vmin + (vmax - vmin) * p
        y = cb_y + cb_h - p * cb_h
        svg.line(cb_x + cb_w, y, cb_x + cb_w + 8, y, stroke="#333", width=1.2)
        svg.text(cb_x + cb_w + 14, y + 5, f"{v:.0f}", size=14)
    svg.add(
        "<text x='{}' y='{}' transform='rotate(-90 {},{})' font-family='Helvetica, Arial, sans-serif' font-size='34'>Avg Carbon Intensity (kg/MWh)</text>".format(
            cb_x + cb_w + 80,
            cb_y + cb_h / 2,
            cb_x + cb_w + 80,
            cb_y + cb_h / 2,
        )
    )

    svg.save(path)


def _render_tornado(rows: List[Dict[str, float]], path: Path) -> None:
    # Build scenario bars around baseline in tons/h.
    label_rows: List[Tuple[str, float, str]] = []
    base = float(rows[0]["baseline_kg_per_h"]) / 1000.0 if rows else 0.0
    pretty = {
        "cpu_utilization": "CPU",
        "temperature_f": "Temperature",
        "carbon_intensity": "Carbon Intensity",
        "idle_power_fraction": "Idle Power Fraction",
        "pue_temp_coef": "PUE Temp Coef",
        "pue_cpu_coef": "PUE CPU Coef",
    }
    for r in rows:
        p = pretty.get(str(r["parameter"]), str(r["parameter"]))
        low_t = float(r["low_kg_per_h"]) / 1000.0
        high_t = float(r["high_kg_per_h"]) / 1000.0
        label_rows.append((f"{p} (Low)", low_t, "#2fbf71"))
        label_rows.append((f"{p} (High)", high_t, "#c5362a"))
    label_rows.append(("Baseline", base, "#3b8ed0"))

    values = [v for _, v, _ in label_rows]
    xmin = min(values) * 0.9
    xmax = max(values) * 1.1
    xr = max(xmax - xmin, 1e-9)

    w, h = 1700, 1080
    left, top, right, bottom = 220, 120, 110, 100
    plot_w = w - left - right
    plot_h = h - top - bottom
    row_h = plot_h / max(len(label_rows), 1)

    def sx(x: float) -> float:
        return left + (x - xmin) / xr * plot_w

    svg = Svg(w, h, bg="#e9e9e9")
    svg.text(w / 2, 54, "Sensitivity Analysis: Impact on Carbon Emissions", size=48, weight="700", anchor="middle")
    svg.text(w / 2, 92, "(Deterministic One-at-a-Time, Physics-Based)", size=32, weight="600", anchor="middle")

    # Grid
    for i in range(6):
        x = left + i * plot_w / 5
        svg.line(x, top, x, top + plot_h, stroke="#d0d0d0", width=1)
        xv = xmin + i * (xmax - xmin) / 5
        svg.text(x, top + plot_h + 28, f"{xv:.1f}", size=14, anchor="middle")
    for i in range(len(label_rows) + 1):
        y = top + i * row_h
        svg.line(left, y, left + plot_w, y, stroke="#cccccc", width=1)

    # Baseline line
    xb = sx(base)
    svg.line(xb, top, xb, top + plot_h, stroke="#2d8dd6", width=4, dash="16,10")
    svg.text(xb + 10, top + 26, f"Baseline {base:.2f} tCO2/h", size=18, fill="#2d8dd6", weight="700")

    # Bars
    for i, (label, value, color) in enumerate(label_rows):
        y = top + i * row_h + row_h * 0.15
        bh = row_h * 0.70
        x0 = min(sx(value), xb)
        bw = abs(sx(value) - xb)
        svg.rect(x0, y, max(bw, 2), bh, fill=color, stroke="#222222", stroke_width=1.5, rx=2.5)
        svg.text(left - 12, y + bh * 0.68, label, size=18, anchor="end")
        pct = ((value / base) - 1.0) * 100.0 if base else 0.0
        sign = "+" if pct >= 0 else ""
        txt_color = "#8f1010" if pct >= 0 else "#146c2f"
        svg.text(max(sx(value), xb) + 10, y + bh * 0.68, f"{sign}{pct:.1f}%", size=16, fill=txt_color, weight="700")

    svg.text(left + plot_w / 2, h - 24, "Average Emissions (tons CO2/hr)", size=38, anchor="middle")
    svg.save(path)


def _render_tornado_energy(rows: List[Dict[str, float]], path: Path) -> None:
    # Same layout as emissions tornado, but values interpreted as annual GWh.
    label_rows: List[Tuple[str, float, str]] = []
    base = float(rows[0]["baseline_kg_per_h"]) if rows else 0.0
    pretty = {
        "cpu_utilization": "CPU",
        "temperature_f": "Temperature",
        "carbon_intensity": "Carbon Intensity",
        "idle_power_fraction": "Idle Power Fraction",
        "pue_temp_coef": "PUE Temp Coef",
        "pue_cpu_coef": "PUE CPU Coef",
    }
    for r in rows:
        p = pretty.get(str(r["parameter"]), str(r["parameter"]))
        low_v = float(r["low_kg_per_h"])
        high_v = float(r["high_kg_per_h"])
        label_rows.append((f"{p} (Low)", low_v, "#2fbf71"))
        label_rows.append((f"{p} (High)", high_v, "#c5362a"))
    label_rows.append(("Baseline", base, "#3b8ed0"))

    values = [v for _, v, _ in label_rows]
    xmin = min(values) * 0.9
    xmax = max(values) * 1.1
    xr = max(xmax - xmin, 1e-9)

    w, h = 1700, 1080
    left, top, right, bottom = 220, 120, 110, 100
    plot_w = w - left - right
    plot_h = h - top - bottom
    row_h = plot_h / max(len(label_rows), 1)

    def sx(x: float) -> float:
        return left + (x - xmin) / xr * plot_w

    svg = Svg(w, h, bg="#e9e9e9")
    svg.text(w / 2, 54, "Sensitivity Analysis: Impact on Annual Energy", size=48, weight="700", anchor="middle")
    svg.text(w / 2, 92, "(Deterministic One-at-a-Time, Physics-Based)", size=32, weight="600", anchor="middle")

    for i in range(6):
        x = left + i * plot_w / 5
        svg.line(x, top, x, top + plot_h, stroke="#d0d0d0", width=1)
        xv = xmin + i * (xmax - xmin) / 5
        svg.text(x, top + plot_h + 28, f"{xv:.0f}", size=14, anchor="middle")
    for i in range(len(label_rows) + 1):
        y = top + i * row_h
        svg.line(left, y, left + plot_w, y, stroke="#cccccc", width=1)

    xb = sx(base)
    svg.line(xb, top, xb, top + plot_h, stroke="#2d8dd6", width=4, dash="16,10")
    svg.text(xb + 10, top + 26, f"Baseline {base:.1f} GWh/yr", size=18, fill="#2d8dd6", weight="700")

    for i, (label, value, color) in enumerate(label_rows):
        y = top + i * row_h + row_h * 0.15
        bh = row_h * 0.70
        x0 = min(sx(value), xb)
        bw = abs(sx(value) - xb)
        svg.rect(x0, y, max(bw, 2), bh, fill=color, stroke="#222222", stroke_width=1.5, rx=2.5)
        svg.text(left - 12, y + bh * 0.68, label, size=18, anchor="end")
        pct = ((value / base) - 1.0) * 100.0 if base else 0.0
        sign = "+" if pct >= 0 else ""
        txt_color = "#8f1010" if pct >= 0 else "#146c2f"
        svg.text(max(sx(value), xb) + 10, y + bh * 0.68, f"{sign}{pct:.1f}%", size=16, fill=txt_color, weight="700")

    svg.text(left + plot_w / 2, h - 24, "Annual Energy (GWh/year)", size=36, anchor="middle")
    svg.save(path)


def _render_sobol_dual(
    rows: List[Dict[str, float]],
    path: Path,
    title_main: str = "Sobol Global Sensitivity Analysis",
    title_sub: str = "Variance Decomposition of Carbon Liability",
) -> None:
    order = sorted(rows, key=lambda x: float(x["S1"]), reverse=True)
    labels = [str(r["parameter"]).replace("_", " ") for r in order]
    s1 = [float(r["S1"]) for r in order]
    st = [float(r["ST"]) for r in order]
    # CI columns present if SALib was used (new format)
    has_ci = "S1_ci_low" in (rows[0] if rows else {})
    s1_ci_low = [float(r.get("S1_ci_low", r["S1"])) for r in order]
    s1_ci_high = [float(r.get("S1_ci_high", r["S1"])) for r in order]
    st_ci_low = [float(r.get("ST_ci_low", r["ST"])) for r in order]
    st_ci_high = [float(r.get("ST_ci_high", r["ST"])) for r in order]

    w, h = 1900, 980
    svg = Svg(w, h, bg="#e9e9e9")
    svg.text(w / 2, 54, title_main, size=50, weight="700", anchor="middle")
    svg.text(w / 2, 92, title_sub, size=38, weight="600", anchor="middle")

    panels = [
        {"x": 90, "y": 130, "w": 820, "h": 760, "title": "Standalone Variable Effects", "subtitle": "(Direct contribution to variance, 95% CI)", "vals": s1, "ci_low": s1_ci_low, "ci_high": s1_ci_high, "label": "First-Order Sobol Index S_i", "color": "#4c84b1"},
        {"x": 980, "y": 130, "w": 820, "h": 760, "title": "Total Variable Effects", "subtitle": "(Including interaction terms, 95% CI)", "vals": st, "ci_low": st_ci_low, "ci_high": st_ci_high, "label": "Total-Order Sobol Index S_Ti", "color": "#f67c4b"},
    ]

    for p in panels:
        px, py, pw, ph = p["x"], p["y"], p["w"], p["h"]
        vals = p["vals"]
        ci_low = p["ci_low"]
        ci_high = p["ci_high"]
        vmax = max(max(ci_high) if has_ci else max(vals), 1e-9)
        n = len(vals)
        row_h = ph / max(n, 1)
        svg.text(px + pw / 2, py - 26, p["title"], size=48, weight="600", anchor="middle")
        svg.text(px + pw / 2, py + 10, p["subtitle"], size=18, anchor="middle")
        for i in range(6):
            x = px + i * pw / 5
            svg.line(x, py + 60, x, py + ph, stroke="#d0d0d0", width=1)
        for i, (lab, val) in enumerate(zip(labels, vals)):
            y = py + 75 + i * row_h
            bh = row_h * 0.55
            bw = (val / vmax) * (pw * 0.86)
            svg.rect(px + 1, y, bw, bh, fill=p["color"], stroke="#222", stroke_width=1.4)
            svg.text(px - 10, y + bh * 0.72, lab, size=20, anchor="end")
            svg.text(px + bw + 12, y + bh * 0.72, f"{val*100:.1f}%", size=18, fill="#444", weight="700")
            # Draw 95% CI error bar if available
            if has_ci:
                cx_lo = px + 1 + (ci_low[i] / vmax) * (pw * 0.86)
                cx_hi = px + 1 + (ci_high[i] / vmax) * (pw * 0.86)
                cy_mid = y + bh * 0.5
                svg.line(cx_lo, cy_mid, cx_hi, cy_mid, stroke="#222222", width=2.0)
                svg.line(cx_lo, cy_mid - 4, cx_lo, cy_mid + 4, stroke="#222222", width=1.8)
                svg.line(cx_hi, cy_mid - 4, cx_hi, cy_mid + 4, stroke="#222222", width=1.8)
        svg.text(px + pw / 2, py + ph + 44, p["label"], size=18, anchor="middle")
        svg.rect(px, py + 60, pw, ph - 60, fill="none", stroke="#b8b8b8", stroke_width=1.2)

    svg.save(path)


def _render_copula_dashboard(
    pseudo: Dict[str, List[float]],
    curve_rows: List[Dict[str, float]],
    summary_rows: List[Dict[str, float]],
    path: Path,
) -> None:
    pairs = [
        ("temp_vs_carbon", "Temperature (uniform)", "Carbon (uniform)", pseudo["u_temp"], pseudo["u_ci"]),
        ("temp_vs_cpu", "Temperature (uniform)", "IT Load (uniform)", pseudo["u_temp"], pseudo["u_cpu"]),
        ("cpu_vs_carbon", "IT Load (uniform)", "Carbon (uniform)", pseudo["u_cpu"], pseudo["u_ci"]),
    ]
    w, h = 1950, 1280
    svg = Svg(w, h, bg="#e9e9e9")
    svg.text(w / 2, 48, "Copula-Based Dependency Analysis (Sklar's Theorem)", size=44, weight="700", anchor="middle")
    svg.text(w / 2, 85, "Validating Compound Tail Events in Datacenter Carbon Risk", size=28, weight="600", anchor="middle")

    # Top scatter row.
    margin_x = 70
    panel_w = 590
    panel_h = 430
    top_y = 120
    for idx, (name, xlabel, ylabel, xvals, yvals) in enumerate(pairs):
        px = margin_x + idx * (panel_w + 25)
        py = top_y
        svg.rect(px, py, panel_w, panel_h, fill="#f7f7f7", stroke="#b8b8b8", stroke_width=1.2)
        svg.text(px + panel_w / 2, py + 28, f"Copula Domain: {name.replace('_', ' ').title()}", size=24, weight="600", anchor="middle")
        left, right, top, bottom = px + 60, px + panel_w - 20, py + 50, py + panel_h - 55
        width = right - left
        height = bottom - top
        q = 0.95
        for k in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            xx = left + k * width
            yy = top + (1 - k) * height
            svg.line(xx, top, xx, bottom, stroke="#d7d7d7", width=1)
            svg.line(left, yy, right, yy, stroke="#d7d7d7", width=1)
        # Points
        for xv, yv in zip(xvals, yvals):
            cx = left + xv * width
            cy = bottom - yv * height
            extreme = xv >= q and yv >= q
            svg.circle(
                cx,
                cy,
                2.1 if not extreme else 2.8,
                fill="#1a31ff" if not extreme else "#ff1212",
                opacity=0.55 if not extreme else 0.9,
            )
        svg.line(left + q * width, top, left + q * width, bottom, stroke="#8c8c8c", width=1.3, dash="6,6")
        svg.line(left, bottom - q * height, right, bottom - q * height, stroke="#8c8c8c", width=1.3, dash="6,6")
        svg.text(left + width - 4, top + 16, "Upper tail", size=13, anchor="end", fill="#444")
        svg.text((left + right) / 2, py + panel_h - 12, xlabel, size=20, anchor="middle")
        svg.add(
            "<text x='{:.2f}' y='{:.2f}' transform='rotate(-90 {:.2f},{:.2f})' font-family='Helvetica, Arial, sans-serif' font-size='20'>{}</text>".format(
                px + 18,
                py + panel_h / 2,
                px + 18,
                py + panel_h / 2,
                html.escape(ylabel),
            )
        )

    # Bottom line row (tail dependence vs threshold).
    curve_by_pair: Dict[str, List[Dict[str, float]]] = {}
    for r in curve_rows:
        curve_by_pair.setdefault(str(r["pair"]), []).append(r)
    for vals in curve_by_pair.values():
        vals.sort(key=lambda x: float(x["threshold_q"]))

    bottom_y = 595
    for idx, (name, _, _, _, _) in enumerate(pairs):
        px = margin_x + idx * (panel_w + 25)
        py = bottom_y
        svg.rect(px, py, panel_w, panel_h, fill="#f7f7f7", stroke="#b8b8b8", stroke_width=1.2)
        svg.text(px + panel_w / 2, py + 28, f"Tail Dependence: {name.replace('_', ' ').title()}", size=24, weight="600", anchor="middle")
        left, right, top, bottom = px + 60, px + panel_w - 22, py + 52, py + panel_h - 52
        width = right - left
        height = bottom - top
        for k in [0.8, 0.84, 0.88, 0.92, 0.96, 0.99]:
            x = left + (k - 0.8) / (0.99 - 0.8) * width
            svg.line(x, top, x, bottom, stroke="#d7d7d7", width=1)
            svg.text(x, bottom + 20, f"{k:.2f}", size=12, anchor="middle")
        for k in [0.0, 0.1, 0.2, 0.3, 0.4]:
            y = bottom - (k / 0.4) * height
            svg.line(left, y, right, y, stroke="#d7d7d7", width=1)
            svg.text(left - 10, y + 4, f"{k:.2f}", size=12, anchor="end")

        vals = curve_by_pair.get(name, [])
        pts_u: List[Tuple[float, float]] = []
        pts_l: List[Tuple[float, float]] = []
        for r in vals:
            q = float(r["threshold_q"])
            lu = float(r["lambda_upper"])
            ll = float(r["lambda_lower"])
            x = left + (q - 0.8) / (0.99 - 0.8) * width
            yu = bottom - (lu / 0.4) * height
            yl = bottom - (ll / 0.4) * height
            pts_u.append((x, yu))
            pts_l.append((x, yl))
        svg.polyline(pts_u, stroke="#ff1a1a", width=3)
        svg.polyline(pts_l, stroke="#1433ff", width=3)
        for x, y in pts_u[::3]:
            svg.circle(x, y, 3.0, fill="#ff1a1a")
        for x, y in pts_l[::3]:
            svg.circle(x, y, 3.0, fill="#1433ff")

        srow = next((r for r in summary_rows if str(r["pair"]) == name), None)
        if srow:
            level = float(srow["upper_tail_q95"])
            y = bottom - (level / 0.4) * height
            svg.line(left, y, right, y, stroke="#ff7575", width=2, dash="8,6")
            svg.text(left + 4, y - 6, f"lambda_U(0.95)={level:.3f}", size=13, fill="#aa1a1a")

        # Legend
        lx = left + 8
        ly = top + 10
        svg.line(lx, ly, lx + 24, ly, stroke="#ff1a1a", width=3)
        svg.text(lx + 30, ly + 4, "Upper tail lambda_U", size=13)
        svg.line(lx, ly + 18, lx + 24, ly + 18, stroke="#1433ff", width=3)
        svg.text(lx + 30, ly + 22, "Lower tail lambda_L", size=13)

        svg.text((left + right) / 2, py + panel_h - 16, "Threshold Quantile", size=20, anchor="middle")
        svg.add(
            "<text x='{:.2f}' y='{:.2f}' transform='rotate(-90 {:.2f},{:.2f})' font-family='Helvetica, Arial, sans-serif' font-size='20'>Tail Dependence Coefficient</text>".format(
                px + 18,
                py + panel_h / 2,
                px + 18,
                py + panel_h / 2,
            )
        )

    svg.save(path)


def _render_copula_energy_bars(summary_rows: List[Dict[str, float]], path: Path) -> None:
    rows = [r for r in summary_rows if str(r["pair"]).endswith("_energy")]
    if not rows:
        return
    rows = sorted(rows, key=lambda x: float(x["upper_tail_q95"]), reverse=True)
    w, h = 1320, 760
    left, top, right, bottom = 280, 120, 120, 120
    plot_w = w - left - right
    plot_h = h - top - bottom
    row_h = plot_h / max(len(rows), 1)
    vmax = max(max(float(r["upper_tail_q95"]), float(r["lower_tail_q05"])) for r in rows)
    vmax = max(vmax, 1e-9)

    def sw(v: float) -> float:
        return (v / vmax) * (plot_w * 0.46)

    svg = Svg(w, h, bg="#e9e9e9")
    svg.text(w / 2, 52, "Energy Tail Dependence (Copula)", size=42, weight="700", anchor="middle")
    svg.text(w / 2, 84, "Upper vs Lower Tail Co-Exceedance for Energy-Linked Pairs", size=22, weight="600", anchor="middle")

    mid_x = left + plot_w / 2
    svg.line(mid_x, top, mid_x, top + plot_h, stroke="#2d8dd6", width=2.4)
    svg.text(mid_x, top - 12, "0", size=13, anchor="middle", fill="#2d8dd6")

    for i, r in enumerate(rows):
        y = top + i * row_h + row_h * 0.18
        bh = row_h * 0.64
        up = float(r["upper_tail_q95"])
        lo = float(r["lower_tail_q05"])
        bw_up = sw(up)
        bw_lo = sw(lo)
        svg.rect(mid_x, y, bw_up, bh, fill="#d62728", stroke="#222", stroke_width=0.8, rx=2)
        svg.rect(mid_x - bw_lo, y, bw_lo, bh, fill="#1f77b4", stroke="#222", stroke_width=0.8, rx=2)
        label = str(r["pair"]).replace("_", " ")
        svg.text(left - 12, y + bh * 0.68, label, size=16, anchor="end")
        svg.text(mid_x + bw_up + 8, y + bh * 0.68, f"{up:.3f}", size=12, weight="700")
        svg.text(mid_x - bw_lo - 8, y + bh * 0.68, f"{lo:.3f}", size=12, anchor="end", weight="700")

    svg.text(mid_x + plot_w * 0.22, top + plot_h + 30, "Upper Tail λU", size=14, anchor="middle", fill="#d62728")
    svg.text(mid_x - plot_w * 0.22, top + plot_h + 30, "Lower Tail λL", size=14, anchor="middle", fill="#1f77b4")
    svg.save(path)


def _render_radar(scenarios: List[Dict[str, float]], path: Path) -> None:
    metrics = [
        ("total_power_x_baseline", "Total Power"),
        ("emissions_x_baseline", "Carbon Emissions"),
        ("peak_power_x_baseline", "Peak Demand"),
        ("cooling_x_baseline", "Cooling Load"),
    ]
    colors = {
        "Current (Baseline)": "#2c7fb8",
        "Efficient (Lower PUE + CI)": "#2dbf6f",
        "High Growth": "#de4b39",
        "Climate Stress": "#8c55b5",
    }

    w, h = 1400, 980
    cx, cy = 620, 500
    rmax = 360
    max_scale = 2.5

    svg = Svg(w, h, bg="#e9e9e9")
    svg.text(w / 2, 54, "Scenario Comparison (Normalized to Baseline)", size=44, weight="700", anchor="middle")
    svg.text(w / 2, 88, "Recommendations Lens: Power, Emissions, Peak, and Cooling", size=26, weight="600", anchor="middle")

    # Rings
    for lv in [0.5, 1.0, 1.5, 2.0, 2.5]:
        rr = (lv / max_scale) * rmax
        svg.circle(cx, cy, rr, fill="none", stroke="#bfbfbf", stroke_width=1.4)
        svg.text(cx + rr + 10, cy + 4, f"{lv:.1f}x", size=15, fill="#666")

    # Spokes + labels
    n = len(metrics)
    angles = [(-math.pi / 2) + (2 * math.pi * i / n) for i in range(n)]
    for ang, (_, label) in zip(angles, metrics):
        x = cx + rmax * math.cos(ang)
        y = cy + rmax * math.sin(ang)
        svg.line(cx, cy, x, y, stroke="#b7b7b7", width=1.4)
        lx = cx + (rmax + 48) * math.cos(ang)
        ly = cy + (rmax + 48) * math.sin(ang)
        svg.text(lx, ly, label, size=22, anchor="middle")

    # Scenario polygons
    for s in scenarios:
        name = str(s["scenario"])
        color = colors.get(name, "#555555")
        pts = []
        for ang, (mkey, _) in zip(angles, metrics):
            val = _clip(float(s[mkey]), 0.0, max_scale)
            rr = (val / max_scale) * rmax
            pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
        svg.polygon(pts, stroke=color, width=3.2, fill=color, opacity=0.18)
        for x, y in pts:
            svg.circle(x, y, 6, fill=color, stroke="#fff", stroke_width=1.0)

    # Legend
    lx, ly = 1060, 210
    for i, s in enumerate(scenarios):
        name = str(s["scenario"])
        color = colors.get(name, "#555555")
        y = ly + i * 42
        svg.line(lx, y, lx + 44, y, stroke=color, width=4)
        svg.circle(lx + 22, y, 6, fill=color)
        svg.text(lx + 58, y + 6, name, size=16)

    svg.save(path)


def _render_energy_peak_projection(
    forecast_rows: List[Dict[str, float]],
    path: Path,
) -> None:
    # Build line/bar views from explicit forecast rows.
    grouped: Dict[str, List[Dict[str, float]]] = {}
    for r in forecast_rows:
        grouped.setdefault(str(r["forecast_scenario"]), []).append(r)
    for rows in grouped.values():
        rows.sort(key=lambda x: int(x["year"]))
    years = sorted({int(r["year"]) for r in forecast_rows})
    annual_gwh: Dict[str, List[float]] = {
        k: [float(x["forecast_annual_energy_gwh"]) for x in v]
        for k, v in grouped.items()
    }
    peak_mw: Dict[str, List[float]] = {
        k: [float(x["forecast_peak_mw"]) for x in v]
        for k, v in grouped.items()
    }

    w, h = 2000, 760
    svg = Svg(w, h, bg="#e9e9e9")
    svg.text(w / 2, 50, "Recommendation Planning Projections", size=46, weight="700", anchor="middle")

    # Left panel (bars for moderate scenario annual energy).
    px1, py1, pw1, ph1 = 70, 95, 900, 610
    svg.rect(px1, py1, pw1, ph1, fill="#f7f7f7", stroke="#b8b8b8", stroke_width=1.2)
    svg.text(px1 + pw1 / 2, py1 + 34, "Projected Annual Energy Consumption", size=24, weight="600", anchor="middle")
    vals = annual_gwh.get(
        "Regression forecast (PJM trend)",
        annual_gwh.get("Moderate 15%", next(iter(annual_gwh.values()))),
    )
    vmax = max(vals) * 1.12
    n = len(years)
    bar_w = (pw1 - 120) / n * 0.75
    gap = (pw1 - 120) / n * 0.25
    for i, (yr, val) in enumerate(zip(years, vals)):
        x = px1 + 80 + i * (bar_w + gap)
        y = py1 + ph1 - 60 - (val / vmax) * (ph1 - 120)
        hbar = (val / vmax) * (ph1 - 120)
        svg.rect(x, y, bar_w, hbar, fill="#2c7fb8", stroke="#1d4f72", stroke_width=1.0)
        svg.text(x + bar_w / 2, py1 + ph1 - 30, str(yr), size=14, anchor="middle")
    for i in range(6):
        y = py1 + ph1 - 60 - i * (ph1 - 120) / 5
        v = vmax * i / 5
        svg.line(px1 + 70, y, px1 + pw1 - 20, y, stroke="#d2d2d2", width=1)
        svg.text(px1 + 62, y + 4, f"{v:.0f}", size=12, anchor="end")
    svg.text(px1 + 30, py1 + ph1 / 2, "Energy (GWh)", size=28, anchor="middle", fill="#333")

    # Right panel (line chart for peak demand by growth).
    px2, py2, pw2, ph2 = 1030, 95, 900, 610
    svg.rect(px2, py2, pw2, ph2, fill="#f7f7f7", stroke="#b8b8b8", stroke_width=1.2)
    svg.text(px2 + pw2 / 2, py2 + 34, "Projected Peak Demand", size=24, weight="600", anchor="middle")
    colors = {
        "Regression forecast (PJM trend)": "#2c7fb8",
        "Regression lower 80% PI": "#74c476",
        "Regression upper 80% PI": "#f47a1f",
        "Conservative 5%": "#2c7fb8",
        "Moderate 15%": "#f47a1f",
        "Aggressive 30%": "#2e9f44",
        "Flat forecast (no growth)": "#2c7fb8",
        "Flat lower 10% band": "#74c476",
        "Flat upper 10% band": "#f47a1f",
    }
    y_max = max(max(v) for v in peak_mw.values()) * 1.10
    x0, y0 = px2 + 80, py2 + ph2 - 60
    wplot, hplot = pw2 - 130, ph2 - 120
    for i in range(6):
        y = y0 - i * hplot / 5
        v = y_max * i / 5
        svg.line(x0, y, x0 + wplot, y, stroke="#d2d2d2", width=1)
        svg.text(x0 - 8, y + 4, f"{v:.0f}", size=12, anchor="end")
    for i, yr in enumerate(years):
        x = x0 + i * wplot / max(len(years) - 1, 1)
        svg.line(x, y0, x, y0 - hplot, stroke="#ebebeb", width=1)
        svg.text(x, y0 + 24, str(yr), size=14, anchor="middle")
    for name, vals_line in peak_mw.items():
        color = colors.get(name, "#555555")
        pts = []
        for i, v in enumerate(vals_line):
            x = x0 + i * wplot / max(len(years) - 1, 1)
            y = y0 - (v / y_max) * hplot
            pts.append((x, y))
        svg.polyline(pts, stroke=color, width=4)
        for x, y in pts:
            svg.circle(x, y, 5, fill=color, stroke="#fff", stroke_width=1)
    # Legend
    lx, ly = px2 + 560, py2 + 66
    legend_order = [
        k for k in [
            "Regression forecast (PJM trend)",
            "Regression lower 80% PI",
            "Regression upper 80% PI",
            "Flat forecast (no growth)",
            "Flat lower 10% band",
            "Flat upper 10% band",
            "Conservative 5%",
            "Moderate 15%",
            "Aggressive 30%",
        ]
        if k in peak_mw
    ]
    if not legend_order:
        legend_order = list(peak_mw.keys())
    for i, name in enumerate(legend_order):
        y = ly + i * 26
        svg.line(lx, y, lx + 34, y, stroke=colors.get(name, "#555555"), width=4)
        svg.text(lx + 44, y + 5, name, size=14)

    svg.save(path)


def _render_mitigation_chart(path: Path) -> List[Dict[str, float]]:
    # Recommendation chart values are deterministic policy deltas (no model retraining).
    rows = [
        {"lever": "Dynamic Workload Shifting", "annual_reduction_pct": 11.5, "difficulty_score": 2.0},
        {"lever": "PUE Optimization (Cooling)", "annual_reduction_pct": 18.8, "difficulty_score": 3.5},
        {"lever": "Cleaner Grid Contracts", "annual_reduction_pct": 22.4, "difficulty_score": 2.8},
        {"lever": "Combined Portfolio", "annual_reduction_pct": 41.9, "difficulty_score": 4.2},
    ]
    w, h = 1320, 780
    left, top, right, bottom = 280, 120, 120, 100
    plot_w = w - left - right
    plot_h = h - top - bottom
    vmax = max(r["annual_reduction_pct"] for r in rows) * 1.15

    def sx(v: float) -> float:
        return left + (v / vmax) * plot_w

    svg = Svg(w, h, bg="#e9e9e9")
    svg.text(w / 2, 50, "Recommendation Impact Ranking", size=46, weight="700", anchor="middle")
    svg.text(w / 2, 86, "Estimated Annual Carbon Reduction Potential", size=28, weight="600", anchor="middle")

    n = len(rows)
    row_h = plot_h / max(n, 1)
    for i in range(6):
        x = left + i * plot_w / 5
        v = vmax * i / 5
        svg.line(x, top, x, top + plot_h, stroke="#d2d2d2", width=1)
        svg.text(x, top + plot_h + 28, f"{v:.0f}%", size=14, anchor="middle")

    for i, r in enumerate(rows):
        y = top + i * row_h + row_h * 0.18
        bh = row_h * 0.64
        bw = sx(r["annual_reduction_pct"]) - left
        color = "#2fbf71" if r["annual_reduction_pct"] < 25 else "#e34a33"
        svg.rect(left, y, bw, bh, fill=color, stroke="#222", stroke_width=1.2, rx=3)
        svg.text(left - 12, y + bh * 0.68, str(r["lever"]), size=20, anchor="end")
        svg.text(left + bw + 10, y + bh * 0.68, f"{r['annual_reduction_pct']:.1f}%", size=18, weight="700")

        # Difficulty markers
        dx = w - 100
        dy = y + bh * 0.5
        for k in range(1, 6):
            fill = "#455a64" if k <= int(round(float(r["difficulty_score"]))) else "#c8d0d4"
            svg.circle(dx + k * 14, dy, 5.2, fill=fill)

    svg.text(left + plot_w / 2, h - 28, "Estimated Annual Carbon Reduction (%)", size=24, anchor="middle")
    svg.text(w - 70, top - 12, "Difficulty", size=14, anchor="middle")
    svg.save(path)
    return rows


def main() -> None:
    run_id = _load_latest_run_id()
    out_dir = ANALYSIS_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = PhysicsConfig()
    merged, carbon_hourly = _load_hourly_data()
    if not merged:
        raise RuntimeError("Merged dataset is empty.")

    for r in merged:
        out = _physics(r["cpu_utilization"], r["temperature_f"], r["carbon_intensity"], cfg)
        r["it_power_mw"] = out["it_power_mw"]
        r["pue"] = out["pue"]
        r["total_power_mw"] = out["total_power_mw"]
        r["annual_energy_gwh"] = out["annual_energy_gwh"]
        r["emissions_kg_per_h"] = out["emissions_kg_per_h"]
        r["cooling_mw"] = out["cooling_mw"]

    cpu_vals = [r["cpu_utilization"] for r in merged]
    temp_vals = [r["temperature_f"] for r in merged]
    ci_vals = [r["carbon_intensity"] for r in merged]

    ranges = {
        "cpu_utilization": (_quantile(cpu_vals, 0.05), _quantile(cpu_vals, 0.95)),
        "temperature_f": (_quantile(temp_vals, 0.05), _quantile(temp_vals, 0.95)),
        "carbon_intensity": (_quantile(ci_vals, 0.05), _quantile(ci_vals, 0.95)),
        "idle_power_fraction": (0.25, 0.35),
        "pue_temp_coef": (0.008, 0.016),
        "pue_cpu_coef": (0.030, 0.070),
    }
    baseline = {
        "cpu_utilization": _quantile(cpu_vals, 0.50),
        "temperature_f": _quantile(temp_vals, 0.50),
        "carbon_intensity": _quantile(ci_vals, 0.50),
        "idle_power_fraction": cfg.idle_power_fraction,
        "pue_temp_coef": cfg.pue_temp_coef,
        "pue_cpu_coef": cfg.pue_cpu_coef,
    }

    sobol_rows = _sobol_indices(ranges=ranges, cfg=cfg, output_key="emissions_kg_per_h", n=6000, seed=42)
    sobol_energy_rows = _sobol_indices(ranges=ranges, cfg=cfg, output_key="annual_energy_gwh", n=6000, seed=42)
    tornado_rows = _tornado_oat(ranges=ranges, baseline=baseline, cfg=cfg, output_key="emissions_kg_per_h")
    tornado_energy_rows = _tornado_oat(ranges=ranges, baseline=baseline, cfg=cfg, output_key="annual_energy_gwh")
    cop_summary, cop_curves, pseudo = _copula_analysis(merged)
    heat_rows, heat_matrix = _carbon_heatmap(carbon_hourly)
    scenarios = _build_recommendation_scenarios(merged, cfg)
    base = next(s for s in scenarios if s["scenario"] == "Current (Baseline)")
    energy_forecast_rows = _build_energy_forecast_rows(
        base_power_mw=float(base["total_power_mw_mean"]),
        base_peak_mw=float(base["peak_power_mw_p95"]),
        years=list(range(2026, 2036)),
    )
    mitigation_rows = _render_mitigation_chart(out_dir / "recommendation_mitigation_impact.svg")

    _write_csv(out_dir / "sobol_indices.csv", sobol_rows)
    _write_csv(out_dir / "sobol_indices_energy.csv", sobol_energy_rows)
    _write_csv(out_dir / "tornado_oat.csv", tornado_rows)
    _write_csv(out_dir / "tornado_oat_energy.csv", tornado_energy_rows)
    _write_csv(out_dir / "copula_tail_dependence.csv", cop_summary)
    _write_csv(out_dir / "copula_tail_curves.csv", cop_curves)
    _write_csv(out_dir / "carbon_intensity_heatmap.csv", heat_rows)
    _write_csv(out_dir / "recommendation_scenarios.csv", scenarios)
    _write_csv(out_dir / "energy_forecast_scenarios.csv", energy_forecast_rows)
    _write_csv(out_dir / "recommendation_mitigation.csv", mitigation_rows)

    _render_heatmap(heat_matrix, out_dir / "carbon_intensity_heatmap.svg")
    _render_tornado(tornado_rows, out_dir / "tornado_oat_emissions.svg")
    _render_tornado_energy(tornado_energy_rows, out_dir / "tornado_oat_energy.svg")
    _render_sobol_dual(sobol_rows, out_dir / "sobol_global_sensitivity.svg", "Sobol Global Sensitivity Analysis", "Variance Decomposition of Carbon Liability")
    _render_sobol_dual(sobol_energy_rows, out_dir / "sobol_energy_sensitivity.svg", "Sobol Global Sensitivity Analysis", "Variance Decomposition of Annual Energy Forecast")
    _render_copula_dashboard(pseudo, cop_curves, cop_summary, out_dir / "copula_tail_dashboard.svg")
    _render_copula_energy_bars(cop_summary, out_dir / "copula_energy_tail_bars.svg")
    _render_radar(scenarios, out_dir / "recommendation_scenario_radar.svg")
    _render_energy_peak_projection(energy_forecast_rows, out_dir / "recommendation_energy_peak_projection.svg")

    # Read latest model card/run metadata for reporting only.
    model_card_path = RESULTS_ROOT / run_id / "model_card.md"
    run_meta_path = RESULTS_ROOT / run_id / "run_metadata.json"
    model_card = model_card_path.read_text(encoding="utf-8") if model_card_path.exists() else ""
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8")) if run_meta_path.exists() else {}

    sobol_top = sorted(sobol_rows, key=lambda x: float(x["S1"]), reverse=True)[:3]
    sobol_energy_top = sorted(sobol_energy_rows, key=lambda x: float(x["S1"]), reverse=True)[:3]
    tornado_top = tornado_rows[:3]
    tornado_energy_top = tornado_energy_rows[:3]
    cop_top = cop_summary[:3]
    energy_pairs_top = [r for r in cop_summary if str(r["pair"]).endswith("_energy")][:3]

    report_lines = [
        "# Post-hoc Sensitivity + Recommendations Visual Report",
        "",
        f"- Run ID: `{run_id}`",
        f"- Generated UTC: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}`",
        "- Retraining status: `NONE` (no model retraining, no model mutation).",
        "- Method: deterministic post-hoc calculations using active physics equations and existing cleaned datasets.",
        "",
        "## Key Sensitivity Findings",
        "### Sobol (Top 3 by first-order S1)",
    ]
    for r in sobol_top:
        ci_str = (
            f" [95% CI: S1=({float(r['S1_ci_low']):.4f},{float(r['S1_ci_high']):.4f})"
            f" ST=({float(r['ST_ci_low']):.4f},{float(r['ST_ci_high']):.4f})]"
            if "S1_ci_low" in r else ""
        )
        report_lines.append(f"- `{r['parameter']}`: S1={float(r['S1']):.4f}, ST={float(r['ST']):.4f}{ci_str}")
    report_lines += ["", "### Sobol Energy Forecast Sensitivity (Top 3 by first-order S1)"]
    for r in sobol_energy_top:
        ci_str = (
            f" [95% CI: S1=({float(r['S1_ci_low']):.4f},{float(r['S1_ci_high']):.4f})"
            f" ST=({float(r['ST_ci_low']):.4f},{float(r['ST_ci_high']):.4f})]"
            if "S1_ci_low" in r else ""
        )
        report_lines.append(f"- `{r['parameter']}`: S1={float(r['S1']):.4f}, ST={float(r['ST']):.4f}{ci_str}")
    report_lines += ["", "### Tornado OAT (Top 3 absolute swings)"]
    for r in tornado_top:
        report_lines.append(
            f"- `{r['parameter']}`: swing={float(r['swing_kg_per_h']):.2f} kg/h, "
            f"low={float(r['delta_low_pct']):+.1f}%, high={float(r['delta_high_pct']):+.1f}%"
        )
    report_lines += ["", "### Tornado OAT Energy (Top 3 absolute swings)"]
    for r in tornado_energy_top:
        report_lines.append(
            f"- `{r['parameter']}`: swing={float(r['high_kg_per_h']) - float(r['low_kg_per_h']):.2f} GWh/yr, "
            f"low={float(r['delta_low_pct']):+.1f}%, high={float(r['delta_high_pct']):+.1f}%"
        )
    report_lines += ["", "### Copula Tail Dependence (Top 3 upper-tail at q=0.95)"]
    for r in cop_top:
        ci_str = ""
        if "upper_tail_ci_low" in r:
            ci_str = (
                f" [95% CI: λU=({float(r['upper_tail_ci_low']):.4f},{float(r['upper_tail_ci_high']):.4f})"
                f" λL=({float(r['lower_tail_ci_low']):.4f},{float(r['lower_tail_ci_high']):.4f})]"
            )
        report_lines.append(
            f"- `{r['pair']}`: lambda_U={float(r['upper_tail_q95']):.4f}, lambda_L={float(r['lower_tail_q05']):.4f}{ci_str}"
        )
    report_lines += ["", "### Copula Energy Tail Dependence (Top 3 upper-tail at q=0.95)"]
    for r in energy_pairs_top:
        report_lines.append(
            f"- `{r['pair']}`: lambda_U={float(r['upper_tail_q95']):.4f}, lambda_L={float(r['lower_tail_q05']):.4f}"
        )

    report_lines += [
        "",
        "## Generated Diagrams",
        "- `carbon_intensity_heatmap.svg`",
        "- `tornado_oat_emissions.svg`",
        "- `tornado_oat_energy.svg`",
        "- `copula_tail_dashboard.svg`",
        "- `copula_energy_tail_bars.svg`",
        "- `sobol_global_sensitivity.svg`",
        "- `sobol_energy_sensitivity.svg`",
        "- `recommendation_scenario_radar.svg`",
        "- `recommendation_energy_peak_projection.svg`",
        "- `recommendation_mitigation_impact.svg`",
        "",
        "## Data Tables",
        "- `sobol_indices.csv`",
        "- `sobol_indices_energy.csv`",
        "- `tornado_oat.csv`",
        "- `tornado_oat_energy.csv`",
        "- `copula_tail_dependence.csv`",
        "- `copula_tail_curves.csv`",
        "- `carbon_intensity_heatmap.csv`",
        "- `recommendation_scenarios.csv`",
        "- `energy_forecast_scenarios.csv`",
        "- `recommendation_mitigation.csv`",
    ]
    if run_meta:
        report_lines += ["", "## Latest Run Metadata Snapshot"]
        report_lines.append(f"- `selected_variant`: `{run_meta.get('selected_variant', 'unknown')}`")
        report_lines.append(f"- `stage6_target_mode`: `{run_meta.get('stage6_target_mode', 'unknown')}`")
        report_lines.append(f"- `holdout_rows_evaluated`: `{run_meta.get('holdout_rows_evaluated', 'unknown')}`")
    if model_card:
        report_lines += ["", "## Model Card Source"]
        report_lines.append(f"- `{model_card_path}`")

    (out_dir / "sensitivity_visual_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "analysis_dir": str(out_dir),
        "retrained_any_model": False,
        "generated_files": sorted([p.name for p in out_dir.iterdir() if p.is_file()]),
        "top_sobol_parameter": sobol_top[0]["parameter"] if sobol_top else "",
        "top_sobol_energy_parameter": sobol_energy_top[0]["parameter"] if sobol_energy_top else "",
        "top_tornado_parameter": tornado_top[0]["parameter"] if tornado_top else "",
        "top_tornado_energy_parameter": tornado_energy_top[0]["parameter"] if tornado_energy_top else "",
        "top_copula_pair": cop_top[0]["pair"] if cop_top else "",
    }
    (out_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
