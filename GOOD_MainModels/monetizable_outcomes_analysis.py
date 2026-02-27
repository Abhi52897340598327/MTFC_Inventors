"""Monetizable outcomes analysis for MTFC paper (no model retraining).

Inputs (from post-hoc sensitivity outputs):
- recommendation_scenarios.csv
- recommendation_mitigation.csv
- copula_tail_dependence.csv
- tornado_oat.csv
- energy_forecast_scenarios.csv

Outputs:
- financial_assumptions.csv
- scenario_monetization.csv
- risk_premium_breakdown.csv
- mitigation_cost_benefit.csv
- energy_forecast_costs.csv
- monetary_numbers.csv (master long-form monetary numbers)
- scenario_cost_stack.svg
- energy_forecast_costs.svg
- mitigation_npv_payback.svg
- monetizable_outcomes_dashboard.svg
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

# FinancialAssumptions lives in config.py (single source of truth for all constants).
from config import FinancialAssumptions
from typing import Dict, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
ANALYSIS_ROOT = ROOT / "outputs" / "analysis"
RESULTS_ROOT = ROOT / "outputs" / "results"
FONT_SCALE = 0.55
MIN_FONT_SIZE = 8


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


def _latest_run_id() -> str:
    runs = sorted([p.name for p in RESULTS_ROOT.glob("carbon_*") if p.is_dir()])
    if not runs:
        raise FileNotFoundError("No run directories found in outputs/results.")
    return runs[-1]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_monetary_csv(
    path: Path,
    rows: List[Dict[str, object]],
    assumptions: "FinancialAssumptions",
) -> None:
    """Write a financial-output CSV with a provenance header comment.

    The comment line (prefixed with '#') lists the active FinancialAssumptions
    constants so that any reader can verify the numbers without cross-checking
    config.py.  This satisfies the requirement that constants match between
    config.py and all monetary outputs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    provenance = (
        f"# scc_usd_per_ton={assumptions.scc_usd_per_ton}, "
        f"energy_price_usd_per_mwh={assumptions.energy_price_usd_per_mwh}, "
        f"discount_rate={assumptions.discount_rate:.0%}, "
        f"carbon_price_low={assumptions.carbon_price_low_usd_per_ton}, "
        f"carbon_price_high={assumptions.carbon_price_high_usd_per_ton}, "
        f"analysis_horizon_years={assumptions.analysis_horizon_years}\n"
    )
    if not rows:
        path.write_text(provenance, encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(provenance)
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _to_float(v: object) -> float:
    return float(v)


def _npv(discount_rate: float, cashflows: Sequence[float]) -> float:
    # cashflows indexed from t=0.
    out = 0.0
    for t, cf in enumerate(cashflows):
        out += cf / ((1.0 + discount_rate) ** t)
    return out


def _simple_payback(capex: float, annual_net: float) -> float:
    if annual_net <= 0:
        return float("inf")
    return capex / annual_net


def _fmt_money(x: float) -> str:
    return f"${x:,.0f}"


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _hex_to_rgb(c: str) -> Tuple[int, int, int]:
    s = c.lstrip("#")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _mix(c1: str, c2: str, t: float) -> str:
    t = _clamp(t, 0.0, 1.0)
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    return _rgb_to_hex(
        (
            int(round(r1 + (r2 - r1) * t)),
            int(round(g1 + (g2 - g1) * t)),
            int(round(b1 + (b2 - b1) * t)),
        )
    )


class Svg:
    def __init__(self, w: int, h: int, bg: str = "#ececec"):
        self.w = w
        self.h = h
        self.parts = [
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>",
            f"<rect x='0' y='0' width='{w}' height='{h}' fill='{bg}'/>",
        ]

    def text(self, x: float, y: float, t: str, size: int = 14, weight: str = "normal", anchor: str = "start", fill: str = "#222") -> None:
        scaled_size = _scale_font_size(size)
        esc = (
            t.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        self.parts.append(
            f"<text x='{x:.2f}' y='{y:.2f}' font-family='Helvetica, Arial, sans-serif' font-size='{scaled_size}' "
            f"font-weight='{weight}' text-anchor='{anchor}' fill='{fill}'>{esc}</text>"
        )

    def rect(self, x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", sw: float = 1.0, rx: float = 0.0, op: float = 1.0) -> None:
        self.parts.append(
            f"<rect x='{x:.2f}' y='{y:.2f}' width='{w:.2f}' height='{h:.2f}' fill='{fill}' "
            f"stroke='{stroke}' stroke-width='{sw:.2f}' rx='{rx:.2f}' opacity='{op:.3f}'/>"
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, stroke: str = "#666", sw: float = 1.0, dash: str | None = None) -> None:
        dash_attr = f" stroke-dasharray='{dash}'" if dash else ""
        self.parts.append(
            f"<line x1='{x1:.2f}' y1='{y1:.2f}' x2='{x2:.2f}' y2='{y2:.2f}' stroke='{stroke}' stroke-width='{sw:.2f}'{dash_attr}/>"
        )

    def circle(self, cx: float, cy: float, r: float, fill: str = "#333", stroke: str = "none", sw: float = 0.0) -> None:
        self.parts.append(
            f"<circle cx='{cx:.2f}' cy='{cy:.2f}' r='{r:.2f}' fill='{fill}' stroke='{stroke}' stroke-width='{sw:.2f}'/>"
        )

    def add(self, raw: str) -> None:
        self.parts.append(_scale_svg_font_sizes(raw))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.parts + ["</svg>"]), encoding="utf-8")


def _scenario_monetization(
    scenario_rows: List[Dict[str, str]],
    tail_rows: List[Dict[str, str]],
    tornado_rows: List[Dict[str, str]],
    assumptions: FinancialAssumptions,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    # Extract tail metrics.
    tail_map = {r["pair"]: _to_float(r["upper_tail_q95"]) for r in tail_rows}
    lam_temp_emi = tail_map.get("temp_vs_emissions", 0.0)
    lam_carb_emi = tail_map.get("carbon_vs_emissions", 0.0)

    # Extract temperature high-side effect from tornado.
    temp_row = next((r for r in tornado_rows if r["parameter"] == "temperature_f"), None)
    temp_high_pct = _to_float(temp_row["delta_high_pct"]) / 100.0 if temp_row else 0.0

    monetized: List[Dict[str, object]] = []
    premium_rows: List[Dict[str, object]] = []

    for r in scenario_rows:
        scenario = r["scenario"]
        total_power_mw = _to_float(r["total_power_mw_mean"])
        emissions_kg_h = _to_float(r["emissions_kg_per_h_mean"])
        peak_mw = _to_float(r["peak_power_mw_p95"])
        cooling_x = _to_float(r["cooling_x_baseline"])

        annual_energy_mwh = total_power_mw * 8760.0
        annual_emissions_tons = emissions_kg_h * 8760.0 / 1000.0

        carbon_liability = annual_emissions_tons * assumptions.scc_usd_per_ton
        carbon_liability_low = annual_emissions_tons * assumptions.carbon_price_low_usd_per_ton
        carbon_liability_high = annual_emissions_tons * assumptions.carbon_price_high_usd_per_ton
        electricity_cost = annual_energy_mwh * assumptions.energy_price_usd_per_mwh
        demand_charge_cost = peak_mw * 1000.0 * assumptions.peak_demand_charge_usd_per_kw_year

        # Tail risk premiums based on sensitivity + copula outcomes.
        extreme_joint_hours = (
            8760.0 * (1.0 - assumptions.extreme_threshold_q) * lam_temp_emi
        )
        incremental_tons_per_extreme_hour = (
            emissions_kg_h * max(temp_high_pct, 0.0) / 1000.0
        )
        tail_carbon_tons = extreme_joint_hours * incremental_tons_per_extreme_hour
        tail_carbon_premium = tail_carbon_tons * assumptions.scc_usd_per_ton

        grid_volatility_premium = (
            electricity_cost * assumptions.electricity_volatility_markup_pct * lam_carb_emi
        )

        breach_mw = max(0.0, peak_mw - assumptions.contract_peak_mw)
        breach_energy_mwh = breach_mw * assumptions.peak_breach_hours
        peak_breach_penalty = breach_energy_mwh * assumptions.peak_breach_penalty_usd_per_mwh

        cooling_wear_premium = max(0.0, cooling_x - 1.0) * assumptions.cooling_maintenance_scale_usd

        risk_premium_total = (
            tail_carbon_premium
            + grid_volatility_premium
            + peak_breach_penalty
            + cooling_wear_premium
        )

        total_annual_cost = (
            carbon_liability
            + electricity_cost
            + demand_charge_cost
            + risk_premium_total
        )
        total_annual_cost_low_carbon = (
            carbon_liability_low
            + electricity_cost
            + demand_charge_cost
            + risk_premium_total
        )
        total_annual_cost_high_carbon = (
            carbon_liability_high
            + electricity_cost
            + demand_charge_cost
            + risk_premium_total
        )

        monetized.append(
            {
                "scenario": scenario,
                "annual_energy_mwh": annual_energy_mwh,
                "annual_emissions_tons": annual_emissions_tons,
                "carbon_liability_usd": carbon_liability,
                "carbon_liability_low_usd": carbon_liability_low,
                "carbon_liability_high_usd": carbon_liability_high,
                "electricity_cost_usd": electricity_cost,
                "demand_charge_cost_usd": demand_charge_cost,
                "tail_carbon_premium_usd": tail_carbon_premium,
                "grid_volatility_premium_usd": grid_volatility_premium,
                "peak_breach_penalty_usd": peak_breach_penalty,
                "cooling_wear_premium_usd": cooling_wear_premium,
                "risk_premium_total_usd": risk_premium_total,
                "total_annual_cost_usd": total_annual_cost,
                "total_annual_cost_low_carbon_usd": total_annual_cost_low_carbon,
                "total_annual_cost_high_carbon_usd": total_annual_cost_high_carbon,
            }
        )

        premium_rows.extend(
            [
                {"scenario": scenario, "premium_component": "tail_carbon_premium_usd", "value_usd": tail_carbon_premium},
                {"scenario": scenario, "premium_component": "grid_volatility_premium_usd", "value_usd": grid_volatility_premium},
                {"scenario": scenario, "premium_component": "peak_breach_penalty_usd", "value_usd": peak_breach_penalty},
                {"scenario": scenario, "premium_component": "cooling_wear_premium_usd", "value_usd": cooling_wear_premium},
                {"scenario": scenario, "premium_component": "risk_premium_total_usd", "value_usd": risk_premium_total},
            ]
        )

    baseline = next((x for x in monetized if x["scenario"] == "Current (Baseline)"), None)
    if baseline:
        for row in monetized:
            row["delta_vs_baseline_total_cost_usd"] = row["total_annual_cost_usd"] - baseline["total_annual_cost_usd"]
            row["delta_vs_baseline_total_cost_low_carbon_usd"] = row["total_annual_cost_low_carbon_usd"] - baseline["total_annual_cost_low_carbon_usd"]
            row["delta_vs_baseline_total_cost_high_carbon_usd"] = row["total_annual_cost_high_carbon_usd"] - baseline["total_annual_cost_high_carbon_usd"]
            row["delta_vs_baseline_carbon_usd"] = row["carbon_liability_usd"] - baseline["carbon_liability_usd"]
            row["delta_vs_baseline_energy_usd"] = row["electricity_cost_usd"] - baseline["electricity_cost_usd"]
            row["delta_vs_baseline_risk_usd"] = row["risk_premium_total_usd"] - baseline["risk_premium_total_usd"]
    return monetized, premium_rows


def _energy_forecast_costs(
    forecast_rows: List[Dict[str, str]],
    assumptions: FinancialAssumptions,
    baseline_carbon_ton_per_mwh: float,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in forecast_rows:
        scenario = str(row["forecast_scenario"])
        year = int(row["year"])
        annual_energy_mwh = _to_float(row.get("forecast_annual_energy_mwh", 0.0))
        if annual_energy_mwh <= 0.0:
            annual_energy_mwh = _to_float(row["forecast_total_power_mw"]) * 8760.0
        peak_mw = _to_float(row["forecast_peak_mw"])

        electricity_cost = annual_energy_mwh * assumptions.energy_price_usd_per_mwh
        demand_charge_cost = peak_mw * 1000.0 * assumptions.peak_demand_charge_usd_per_kw_year
        annual_emissions_tons = annual_energy_mwh * baseline_carbon_ton_per_mwh
        carbon_liability = annual_emissions_tons * assumptions.scc_usd_per_ton
        carbon_liability_low = annual_emissions_tons * assumptions.carbon_price_low_usd_per_ton
        carbon_liability_high = annual_emissions_tons * assumptions.carbon_price_high_usd_per_ton

        grid_volatility_premium = electricity_cost * assumptions.electricity_volatility_markup_pct
        breach_mw = max(0.0, peak_mw - assumptions.contract_peak_mw)
        breach_energy_mwh = breach_mw * assumptions.peak_breach_hours
        peak_breach_penalty = breach_energy_mwh * assumptions.peak_breach_penalty_usd_per_mwh

        total_cost = (
            electricity_cost
            + demand_charge_cost
            + carbon_liability
            + grid_volatility_premium
            + peak_breach_penalty
        )
        total_cost_low = (
            electricity_cost
            + demand_charge_cost
            + carbon_liability_low
            + grid_volatility_premium
            + peak_breach_penalty
        )
        total_cost_high = (
            electricity_cost
            + demand_charge_cost
            + carbon_liability_high
            + grid_volatility_premium
            + peak_breach_penalty
        )

        out.append(
            {
                "forecast_scenario": scenario,
                "year": year,
                "forecast_annual_energy_mwh": annual_energy_mwh,
                "forecast_peak_mw": peak_mw,
                "forecast_annual_emissions_tons": annual_emissions_tons,
                "electricity_cost_usd": electricity_cost,
                "demand_charge_cost_usd": demand_charge_cost,
                "carbon_liability_usd": carbon_liability,
                "carbon_liability_low_usd": carbon_liability_low,
                "carbon_liability_high_usd": carbon_liability_high,
                "grid_volatility_premium_usd": grid_volatility_premium,
                "peak_breach_penalty_usd": peak_breach_penalty,
                "total_annual_cost_usd": total_cost,
                "total_annual_cost_low_carbon_usd": total_cost_low,
                "total_annual_cost_high_carbon_usd": total_cost_high,
            }
        )

    out.sort(key=lambda r: (str(r["forecast_scenario"]), int(r["year"])))
    return out


def _mitigation_cost_benefit(
    monetized_rows: List[Dict[str, object]],
    mitigation_rows: List[Dict[str, str]],
    assumptions: FinancialAssumptions,
) -> List[Dict[str, object]]:
    baseline = next((r for r in monetized_rows if r["scenario"] == "Current (Baseline)"), None)
    if baseline is None:
        raise ValueError("Baseline scenario not found.")

    baseline_emissions_tons = _to_float(baseline["annual_emissions_tons"])
    baseline_energy_mwh = _to_float(baseline["annual_energy_mwh"])
    baseline_total_cost = _to_float(baseline["total_annual_cost_usd"])
    baseline_peak_cost = _to_float(baseline["demand_charge_cost_usd"])
    baseline_risk = _to_float(baseline["risk_premium_total_usd"])

    # Assumed implementation economics per mitigation.
    # These are explicit financial assumptions, not model retraining outputs.
    implementation = {
        "Dynamic Workload Shifting": {
            "capex_usd": 600_000.0,
            "annual_opex_usd": 250_000.0,
            "energy_change_pct": -0.02,
            "peak_reduction_pct": 0.06,
            "risk_reduction_pct": 0.08,
        },
        "PUE Optimization (Cooling)": {
            "capex_usd": 14_000_000.0,
            "annual_opex_usd": 600_000.0,
            "energy_change_pct": -0.09,
            "peak_reduction_pct": 0.10,
            "risk_reduction_pct": 0.25,
        },
        "Cleaner Grid Contracts": {
            "capex_usd": 200_000.0,
            "annual_opex_usd": 120_000.0,
            "energy_change_pct": 0.00,
            "peak_reduction_pct": 0.00,
            "risk_reduction_pct": 0.12,
            "green_contract_premium_usd_per_mwh": 4.0,
        },
        "Combined Portfolio": {
            "capex_usd": 15_200_000.0,
            "annual_opex_usd": 900_000.0,
            "energy_change_pct": -0.105,
            "peak_reduction_pct": 0.13,
            "risk_reduction_pct": 0.35,
            "green_contract_premium_usd_per_mwh": 2.5,
        },
    }

    out: List[Dict[str, object]] = []

    for m in mitigation_rows:
        lever = m["lever"]
        reduction_pct = _to_float(m["annual_reduction_pct"]) / 100.0
        impl = implementation.get(lever)
        if impl is None:
            continue

        carbon_avoided_tons = baseline_emissions_tons * reduction_pct
        carbon_avoided_usd = carbon_avoided_tons * assumptions.scc_usd_per_ton
        carbon_avoided_usd_low = carbon_avoided_tons * assumptions.carbon_price_low_usd_per_ton
        carbon_avoided_usd_high = carbon_avoided_tons * assumptions.carbon_price_high_usd_per_ton

        energy_change_pct = _to_float(impl["energy_change_pct"])
        energy_delta_mwh = baseline_energy_mwh * energy_change_pct
        energy_savings_usd = -energy_delta_mwh * assumptions.energy_price_usd_per_mwh

        green_premium = _to_float(impl.get("green_contract_premium_usd_per_mwh", 0.0))
        green_premium_cost_usd = baseline_energy_mwh * green_premium

        peak_savings_usd = baseline_peak_cost * _to_float(impl["peak_reduction_pct"])
        risk_savings_usd = baseline_risk * _to_float(impl["risk_reduction_pct"])

        annual_gross_benefit = (
            carbon_avoided_usd
            + energy_savings_usd
            + peak_savings_usd
            + risk_savings_usd
        )
        annual_gross_benefit_low = (
            carbon_avoided_usd_low
            + energy_savings_usd
            + peak_savings_usd
            + risk_savings_usd
        )
        annual_gross_benefit_high = (
            carbon_avoided_usd_high
            + energy_savings_usd
            + peak_savings_usd
            + risk_savings_usd
        )

        annual_opex = _to_float(impl["annual_opex_usd"])
        annual_net_benefit = annual_gross_benefit - annual_opex - green_premium_cost_usd
        annual_net_benefit_low = annual_gross_benefit_low - annual_opex - green_premium_cost_usd
        annual_net_benefit_high = annual_gross_benefit_high - annual_opex - green_premium_cost_usd

        capex = _to_float(impl["capex_usd"])
        payback_years = _simple_payback(capex, annual_net_benefit)

        # 10-year NPV and IRR proxy through NPV only (to avoid root-finding instability in stdlib).
        cashflows = [-capex] + [annual_net_benefit] * assumptions.analysis_horizon_years
        npv_10y = _npv(assumptions.discount_rate, cashflows)
        roi_10y = (npv_10y / capex) if capex > 0 else float("inf")
        cashflows_low = [-capex] + [annual_net_benefit_low] * assumptions.analysis_horizon_years
        cashflows_high = [-capex] + [annual_net_benefit_high] * assumptions.analysis_horizon_years
        npv_10y_low = _npv(assumptions.discount_rate, cashflows_low)
        npv_10y_high = _npv(assumptions.discount_rate, cashflows_high)

        out.append(
            {
                "lever": lever,
                "annual_reduction_pct": reduction_pct * 100.0,
                "carbon_avoided_tons_per_year": carbon_avoided_tons,
                "carbon_avoided_usd_per_year": carbon_avoided_usd,
                "carbon_avoided_usd_low_per_year": carbon_avoided_usd_low,
                "carbon_avoided_usd_high_per_year": carbon_avoided_usd_high,
                "energy_delta_mwh_per_year": energy_delta_mwh,
                "energy_savings_usd_per_year": energy_savings_usd,
                "peak_savings_usd_per_year": peak_savings_usd,
                "risk_savings_usd_per_year": risk_savings_usd,
                "green_premium_cost_usd_per_year": green_premium_cost_usd,
                "annual_opex_usd": annual_opex,
                "annual_net_benefit_usd": annual_net_benefit,
                "annual_net_benefit_low_usd": annual_net_benefit_low,
                "annual_net_benefit_high_usd": annual_net_benefit_high,
                "capex_usd": capex,
                "payback_years": payback_years if math.isfinite(payback_years) else -1.0,
                "npv_10y_usd": npv_10y,
                "npv_10y_low_usd": npv_10y_low,
                "npv_10y_high_usd": npv_10y_high,
                "roi_10y_ratio": roi_10y,
            }
        )

    out.sort(key=lambda r: _to_float(r["npv_10y_usd"]), reverse=True)
    return out


def _master_monetary_csv(
    assumptions: FinancialAssumptions,
    scenario_rows: List[Dict[str, object]],
    mitigation_rows: List[Dict[str, object]],
    premium_rows: List[Dict[str, object]],
    energy_forecast_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for k, v in assumptions.__dict__.items():
        rows.append(
            {
                "table": "assumptions",
                "entity": "global",
                "metric": k,
                "value": v,
                "unit": "usd" if "usd" in k else "ratio_or_other",
                "is_monetary": "usd" in k,
            }
        )

    monetary_metrics = [
        "carbon_liability_usd",
        "carbon_liability_low_usd",
        "carbon_liability_high_usd",
        "electricity_cost_usd",
        "demand_charge_cost_usd",
        "tail_carbon_premium_usd",
        "grid_volatility_premium_usd",
        "peak_breach_penalty_usd",
        "cooling_wear_premium_usd",
        "risk_premium_total_usd",
        "total_annual_cost_usd",
        "total_annual_cost_low_carbon_usd",
        "total_annual_cost_high_carbon_usd",
        "delta_vs_baseline_total_cost_usd",
        "delta_vs_baseline_total_cost_low_carbon_usd",
        "delta_vs_baseline_total_cost_high_carbon_usd",
        "delta_vs_baseline_carbon_usd",
        "delta_vs_baseline_energy_usd",
        "delta_vs_baseline_risk_usd",
    ]
    for row in scenario_rows:
        sc = row["scenario"]
        for m in monetary_metrics:
            rows.append(
                {
                    "table": "scenario_monetization",
                    "entity": sc,
                    "metric": m,
                    "value": row.get(m, 0.0),
                    "unit": "usd",
                    "is_monetary": True,
                }
            )

    for row in premium_rows:
        rows.append(
            {
                "table": "risk_premiums",
                "entity": row["scenario"],
                "metric": row["premium_component"],
                "value": row["value_usd"],
                "unit": "usd",
                "is_monetary": True,
            }
        )

    mitigation_monetary = [
        "carbon_avoided_usd_per_year",
        "carbon_avoided_usd_low_per_year",
        "carbon_avoided_usd_high_per_year",
        "energy_savings_usd_per_year",
        "peak_savings_usd_per_year",
        "risk_savings_usd_per_year",
        "green_premium_cost_usd_per_year",
        "annual_opex_usd",
        "annual_net_benefit_usd",
        "annual_net_benefit_low_usd",
        "annual_net_benefit_high_usd",
        "capex_usd",
        "npv_10y_usd",
        "npv_10y_low_usd",
        "npv_10y_high_usd",
    ]
    for row in mitigation_rows:
        lever = row["lever"]
        for m in mitigation_monetary:
            rows.append(
                {
                    "table": "mitigation_cost_benefit",
                    "entity": lever,
                    "metric": m,
                    "value": row.get(m, 0.0),
                    "unit": "usd",
                    "is_monetary": True,
                }
            )

    energy_forecast_monetary = [
        "electricity_cost_usd",
        "demand_charge_cost_usd",
        "carbon_liability_usd",
        "carbon_liability_low_usd",
        "carbon_liability_high_usd",
        "grid_volatility_premium_usd",
        "peak_breach_penalty_usd",
        "total_annual_cost_usd",
        "total_annual_cost_low_carbon_usd",
        "total_annual_cost_high_carbon_usd",
    ]
    for row in energy_forecast_rows:
        entity = f"{row['forecast_scenario']}:{int(row['year'])}"
        for metric in energy_forecast_monetary:
            rows.append(
                {
                    "table": "energy_forecast_costs",
                    "entity": entity,
                    "metric": metric,
                    "value": row.get(metric, 0.0),
                    "unit": "usd",
                    "is_monetary": True,
                }
            )
    return rows


def _plot_scenario_cost_stack(path: Path, rows: List[Dict[str, object]]) -> None:
    scenarios = [r["scenario"] for r in rows]
    comps = [
        ("carbon_liability_usd", "#e15759"),
        ("electricity_cost_usd", "#4e79a7"),
        ("demand_charge_cost_usd", "#f28e2b"),
        ("risk_premium_total_usd", "#b07aa1"),
    ]

    # Determine scale.
    totals = [sum(_to_float(r[c]) for c, _ in comps) for r in rows]
    vmax = max(totals) * 1.15 if totals else 1.0

    w, h = 1700, 980
    left, top, right, bottom = 140, 130, 120, 130
    plot_w = w - left - right
    plot_h = h - top - bottom
    n = len(scenarios)
    bar_w = plot_w / max(n, 1) * 0.60
    gap = plot_w / max(n, 1) * 0.40

    svg = Svg(w, h, bg="#ececec")
    svg.text(w / 2, 52, "Annual Cost Stack by Scenario", size=48, weight="700", anchor="middle")
    svg.text(w / 2, 88, "Carbon + Electricity + Demand + Risk Premiums", size=28, weight="600", anchor="middle")

    for i in range(6):
        y = top + plot_h - i * plot_h / 5
        v = vmax * i / 5
        svg.line(left, y, left + plot_w, y, stroke="#cfcfcf", sw=1)
        svg.text(left - 10, y + 4, f"${v/1e6:.1f}M", size=14, anchor="end")

    for idx, row in enumerate(rows):
        x = left + idx * (bar_w + gap) + gap / 2
        y_base = top + plot_h
        for metric, color in comps:
            v = _to_float(row[metric])
            bh = (v / vmax) * plot_h
            y_base -= bh
            svg.rect(x, y_base, bar_w, bh, fill=color, stroke="#222", sw=0.8)
        total = sum(_to_float(row[c]) for c, _ in comps)
        svg.text(x + bar_w / 2, top + plot_h + 28, str(row["scenario"]), size=14, anchor="middle")
        svg.text(x + bar_w / 2, y_base - 8, f"${total/1e6:.2f}M", size=13, anchor="middle", weight="700")

    # Legend
    lx, ly = w - 420, top + 20
    for i, (metric, color) in enumerate(comps):
        y = ly + i * 28
        svg.rect(lx, y - 12, 18, 18, fill=color, stroke="#222", sw=0.8)
        svg.text(lx + 28, y + 2, metric.replace("_usd", "").replace("_", " "), size=14)

    svg.text(left + plot_w / 2, h - 24, "Scenario", size=24, anchor="middle")
    svg.text(34, top + plot_h / 2, "Annual Cost (USD)", size=24, anchor="middle")
    svg.save(path)


def _plot_mitigation_npv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    labels = [str(r["lever"]) for r in rows]
    npvs = [_to_float(r["npv_10y_usd"]) for r in rows]
    paybacks = [_to_float(r["payback_years"]) for r in rows]

    min_npv = min(min(npvs), 0.0)
    max_npv = max(max(npvs), 1.0)
    span = max(max_npv - min_npv, 1e-9)

    w, h = 1700, 960
    left, top, right, bottom = 280, 120, 120, 140
    plot_w = w - left - right
    plot_h = h - top - bottom
    row_h = plot_h / max(len(rows), 1)

    def sx(v: float) -> float:
        return left + ((v - min_npv) / span) * plot_w

    svg = Svg(w, h, bg="#ececec")
    svg.text(w / 2, 52, "Mitigation Economics (10-Year Horizon)", size=48, weight="700", anchor="middle")
    svg.text(w / 2, 88, "NPV Bars with Payback Overlay", size=28, weight="600", anchor="middle")

    for i in range(7):
        x = left + i * plot_w / 6
        v = min_npv + i * span / 6
        svg.line(x, top, x, top + plot_h, stroke="#d0d0d0", sw=1)
        svg.text(x, top + plot_h + 28, f"${v/1e6:.1f}M", size=13, anchor="middle")

    x0 = sx(0.0)
    svg.line(x0, top, x0, top + plot_h, stroke="#2f77b4", sw=2.5, dash="8,6")
    svg.text(x0 + 8, top + 18, "NPV=0", size=13, fill="#2f77b4")

    for i, (lab, npv, pb) in enumerate(zip(labels, npvs, paybacks)):
        y = top + i * row_h + row_h * 0.18
        bh = row_h * 0.62
        x_start = min(x0, sx(npv))
        bw = abs(sx(npv) - x0)
        color = "#2ca02c" if npv >= 0 else "#d62728"
        svg.rect(x_start, y, max(bw, 2), bh, fill=color, stroke="#222", sw=1.0, rx=2)
        svg.text(left - 12, y + bh * 0.68, lab, size=17, anchor="end")
        svg.text(sx(npv) + (8 if npv >= 0 else -8), y + bh * 0.68, f"${npv/1e6:.2f}M", size=13, anchor="start" if npv >= 0 else "end", weight="700")

        # Payback badge.
        badge_x = left + plot_w + 18
        badge_color = "#2ca02c" if (pb > 0 and pb <= 8) else "#ff7f0e" if pb < 99 else "#d62728"
        pb_text = f"{pb:.1f}y" if pb > 0 and pb < 99 else "N/A"
        svg.rect(badge_x, y + 5, 70, bh - 10, fill=badge_color, stroke="#222", sw=0.8, rx=5)
        svg.text(badge_x + 35, y + bh * 0.66, pb_text, size=13, anchor="middle", fill="#fff", weight="700")

    svg.text(left + plot_w / 2, h - 24, "10-year NPV (USD)", size=22, anchor="middle")
    svg.text(left + plot_w + 53, top - 10, "Payback", size=12, anchor="middle")
    svg.save(path)


def _plot_dashboard(path: Path, scen: List[Dict[str, object]], miti: List[Dict[str, object]]) -> None:
    # Compact executive dashboard with cards.
    baseline = next((r for r in scen if r["scenario"] == "Current (Baseline)"), None)
    efficient = next((r for r in scen if r["scenario"] == "Efficient (Lower PUE + CI)"), None)
    combined = next((r for r in miti if r["lever"] == "Combined Portfolio"), None)
    if baseline is None or efficient is None:
        return

    w, h = 1500, 860
    svg = Svg(w, h, bg="#ececec")
    svg.text(w / 2, 52, "Monetizable Outcomes Dashboard", size=46, weight="700", anchor="middle")
    svg.text(w / 2, 84, "Risk-to-Dollar Translation for Datacenter Carbon Strategy", size=24, weight="600", anchor="middle")

    card_w, card_h = 430, 220
    xs = [70, 535, 1000]
    y1, y2 = 130, 400

    def card(x: float, y: float, title: str, body: List[str], accent: str) -> None:
        svg.rect(x, y, card_w, card_h, fill="#f8f8f8", stroke="#b8b8b8", sw=1.2, rx=10)
        svg.rect(x, y, card_w, 8, fill=accent, stroke="none", rx=10)
        svg.text(x + 18, y + 34, title, size=22, weight="700")
        for i, line in enumerate(body):
            svg.text(x + 18, y + 70 + i * 28, line, size=18)

    card(
        xs[0],
        y1,
        "Baseline Annual Cost",
        [
            _fmt_money(_to_float(baseline["total_annual_cost_usd"])),
            f"Carbon: {_fmt_money(_to_float(baseline['carbon_liability_usd']))}",
            f"Energy: {_fmt_money(_to_float(baseline['electricity_cost_usd']))}",
            f"Risk Premium: {_fmt_money(_to_float(baseline['risk_premium_total_usd']))}",
        ],
        "#4e79a7",
    )
    card(
        xs[1],
        y1,
        "Efficient Scenario Delta",
        [
            f"Total: {_fmt_money(_to_float(efficient['delta_vs_baseline_total_cost_usd']))}",
            f"Carbon: {_fmt_money(_to_float(efficient['delta_vs_baseline_carbon_usd']))}",
            f"Energy: {_fmt_money(_to_float(efficient['delta_vs_baseline_energy_usd']))}",
            f"Risk: {_fmt_money(_to_float(efficient['delta_vs_baseline_risk_usd']))}",
        ],
        "#2ca02c",
    )

    if combined is not None:
        card(
            xs[2],
            y1,
            "Combined Portfolio",
            [
                f"NPV(10y): {_fmt_money(_to_float(combined['npv_10y_usd']))}",
                f"NPV band: {_fmt_money(_to_float(combined['npv_10y_low_usd']))} to {_fmt_money(_to_float(combined['npv_10y_high_usd']))}",
                f"Annual Net: {_fmt_money(_to_float(combined['annual_net_benefit_usd']))}",
                f"Payback: {_to_float(combined['payback_years']):.1f} years",
                f"Carbon Avoided: {_to_float(combined['carbon_avoided_tons_per_year']):,.0f} t/yr",
            ],
            "#d62728",
        )

    # Bottom strip: mitigation ranking by annual net benefit.
    svg.rect(70, y2, w - 140, 360, fill="#f8f8f8", stroke="#b8b8b8", sw=1.2, rx=10)
    svg.text(95, y2 + 38, "Mitigation Annual Net Benefit (USD/yr)", size=24, weight="700")
    if miti:
        max_v = max(_to_float(r["annual_net_benefit_usd"]) for r in miti)
        min_v = min(_to_float(r["annual_net_benefit_usd"]) for r in miti)
        span = max(max_v - min_v, 1e-9)
        left, top = 180, y2 + 70
        width, height = w - 300, 250
        n = len(miti)
        bar_h = height / max(n, 1) * 0.58
        gap = height / max(n, 1) * 0.42
        for i, row in enumerate(miti):
            y = top + i * (bar_h + gap)
            v = _to_float(row["annual_net_benefit_usd"])
            bw = (v - min_v) / span * (width - 50)
            color = _mix("#f7b6b2", "#2ca02c", (v - min_v) / span)
            svg.rect(left, y, bw, bar_h, fill=color, stroke="#333", sw=0.9, rx=3)
            svg.text(left - 12, y + bar_h * 0.7, str(row["lever"]), size=16, anchor="end")
            svg.text(left + bw + 8, y + bar_h * 0.7, _fmt_money(v), size=14, weight="700")

    svg.save(path)


def _plot_carbon_price_band(path: Path, scen: List[Dict[str, object]]) -> None:
    # Error-bar style ranges for total annual cost under low/high carbon price assumptions.
    w, h = 1500, 760
    left, top, right, bottom = 220, 130, 140, 120
    plot_w = w - left - right
    plot_h = h - top - bottom

    vals_low = [_to_float(r["total_annual_cost_low_carbon_usd"]) for r in scen]
    vals_mid = [_to_float(r["total_annual_cost_usd"]) for r in scen]
    vals_high = [_to_float(r["total_annual_cost_high_carbon_usd"]) for r in scen]
    vmin = min(vals_low) * 0.95
    vmax = max(vals_high) * 1.05
    span = max(vmax - vmin, 1e-9)

    def sx(v: float) -> float:
        return left + ((v - vmin) / span) * plot_w

    svg = Svg(w, h, bg="#ececec")
    svg.text(w / 2, 52, "Annual Cost Exposure Under Carbon Price Bands", size=44, weight="700", anchor="middle")
    svg.text(w / 2, 86, "Low/Base/High carbon liability assumptions ($95 / $190 / $300 per ton)", size=22, weight="600", anchor="middle")

    for i in range(7):
        x = left + i * plot_w / 6
        v = vmin + i * span / 6
        svg.line(x, top, x, top + plot_h, stroke="#d0d0d0", sw=1.0)
        svg.text(x, top + plot_h + 28, f"${v/1e6:.1f}M", size=13, anchor="middle")

    n = len(scen)
    row_h = plot_h / max(n, 1)
    for i, r in enumerate(scen):
        y = top + i * row_h + row_h * 0.50
        x_lo = sx(_to_float(r["total_annual_cost_low_carbon_usd"]))
        x_mid = sx(_to_float(r["total_annual_cost_usd"]))
        x_hi = sx(_to_float(r["total_annual_cost_high_carbon_usd"]))
        svg.line(x_lo, y, x_hi, y, stroke="#4e79a7", sw=4.0)
        svg.circle(x_mid, y, 6.5, fill="#d62728", stroke="#fff", sw=1.0)
        svg.circle(x_lo, y, 4.0, fill="#2ca02c")
        svg.circle(x_hi, y, 4.0, fill="#ff7f0e")
        svg.text(left - 14, y + 5, str(r["scenario"]), size=16, anchor="end")
        svg.text(x_mid + 8, y - 10, f"${_to_float(r['total_annual_cost_usd'])/1e6:.2f}M", size=12)

    lx, ly = w - 300, top + 20
    svg.line(lx, ly, lx + 40, ly, stroke="#4e79a7", sw=4)
    svg.circle(lx + 20, ly, 6.5, fill="#d62728", stroke="#fff", sw=1)
    svg.text(lx + 48, ly + 4, "Cost range with base marker", size=13)
    svg.circle(lx + 4, ly + 22, 4.0, fill="#2ca02c")
    svg.text(lx + 16, ly + 26, "Low carbon price", size=12)
    svg.circle(lx + 4, ly + 42, 4.0, fill="#ff7f0e")
    svg.text(lx + 16, ly + 46, "High carbon price", size=12)

    svg.text(left + plot_w / 2, h - 24, "Total Annual Cost (USD)", size=20, anchor="middle")
    svg.save(path)


def _plot_energy_forecast_costs(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["forecast_scenario"]), []).append(row)
    for vals in grouped.values():
        vals.sort(key=lambda r: int(r["year"]))

    all_costs = [_to_float(r["total_annual_cost_usd"]) for r in rows]
    vmin = min(all_costs) * 0.95
    vmax = max(all_costs) * 1.05
    span = max(vmax - vmin, 1e-9)
    years = sorted({int(r["year"]) for r in rows})

    w, h = 1600, 860
    left, top, right, bottom = 140, 130, 180, 120
    plot_w = w - left - right
    plot_h = h - top - bottom
    colors = {"Conservative 5%": "#2c7fb8", "Moderate 15%": "#f47a1f", "Aggressive 30%": "#2e9f44"}

    def sy(v: float) -> float:
        return top + plot_h - ((v - vmin) / span) * plot_h

    def sx(year: int) -> float:
        return left + (year - years[0]) / max((years[-1] - years[0]), 1) * plot_w

    svg = Svg(w, h, bg="#ececec")
    svg.text(w / 2, 52, "Projected Annual Cost from Energy Forecast", size=44, weight="700", anchor="middle")
    svg.text(
        w / 2,
        86,
        "Electricity + Demand + Carbon + Volatility + Peak-Breach Penalties",
        size=22,
        weight="600",
        anchor="middle",
    )

    for i in range(6):
        y = top + plot_h - i * plot_h / 5
        v = vmin + i * span / 5
        svg.line(left, y, left + plot_w, y, stroke="#d0d0d0", sw=1.0)
        svg.text(left - 10, y + 4, f"${v/1e6:.1f}M", size=13, anchor="end")
    for year in years:
        x = sx(year)
        svg.line(x, top, x, top + plot_h, stroke="#ececec", sw=1.0)
        svg.text(x, top + plot_h + 24, str(year), size=13, anchor="middle")

    for scenario, vals in grouped.items():
        color = colors.get(scenario, "#555555")
        points = [(sx(int(r["year"])), sy(_to_float(r["total_annual_cost_usd"]))) for r in vals]
        for i in range(len(points) - 1):
            svg.line(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1], stroke=color, sw=3.2)
        for x, y in points:
            svg.circle(x, y, 4.8, fill=color, stroke="#fff", sw=1.0)

    lx, ly = w - 160, top + 16
    for i, name in enumerate(["Conservative 5%", "Moderate 15%", "Aggressive 30%"]):
        if name not in grouped:
            continue
        y = ly + i * 24
        svg.line(lx - 44, y, lx - 14, y, stroke=colors.get(name, "#555555"), sw=3.2)
        svg.circle(lx - 29, y, 4.8, fill=colors.get(name, "#555555"), stroke="#fff", sw=1.0)
        svg.text(lx, y + 4, name, size=13)

    svg.text(left + plot_w / 2, h - 24, "Forecast Year", size=20, anchor="middle")
    svg.text(50, top + plot_h / 2, "Total Annual Cost (USD)", size=20, anchor="middle")
    svg.save(path)


def main() -> None:
    run_id = _latest_run_id()
    run_dir = ANALYSIS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    scenario_path = run_dir / "recommendation_scenarios.csv"
    mitigation_path = run_dir / "recommendation_mitigation.csv"
    tail_path = run_dir / "copula_tail_dependence.csv"
    tornado_path = run_dir / "tornado_oat.csv"
    forecast_path = run_dir / "energy_forecast_scenarios.csv"

    for p in [scenario_path, mitigation_path, tail_path, tornado_path, forecast_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input for monetization: {p}")

    scenario_rows = _read_csv(scenario_path)
    mitigation_rows = _read_csv(mitigation_path)
    tail_rows = _read_csv(tail_path)
    tornado_rows = _read_csv(tornado_path)
    forecast_rows = _read_csv(forecast_path)

    assumptions = FinancialAssumptions()
    scen_monetized, premium_rows = _scenario_monetization(
        scenario_rows=scenario_rows,
        tail_rows=tail_rows,
        tornado_rows=tornado_rows,
        assumptions=assumptions,
    )
    mitigation_fin = _mitigation_cost_benefit(
        monetized_rows=scen_monetized,
        mitigation_rows=mitigation_rows,
        assumptions=assumptions,
    )
    baseline_row = next((r for r in scen_monetized if r["scenario"] == "Current (Baseline)"), None)
    if baseline_row is None:
        raise ValueError("Baseline monetization row not found.")
    baseline_carbon_ton_per_mwh = (
        _to_float(baseline_row["annual_emissions_tons"]) / max(_to_float(baseline_row["annual_energy_mwh"]), 1e-9)
    )
    energy_forecast_cost_rows = _energy_forecast_costs(
        forecast_rows=forecast_rows,
        assumptions=assumptions,
        baseline_carbon_ton_per_mwh=baseline_carbon_ton_per_mwh,
    )
    master_rows = _master_monetary_csv(
        assumptions=assumptions,
        scenario_rows=scen_monetized,
        mitigation_rows=mitigation_fin,
        premium_rows=premium_rows,
        energy_forecast_rows=energy_forecast_cost_rows,
    )

    assumptions_rows = [
        {"assumption": k, "value": v}
        for k, v in assumptions.__dict__.items()
    ]

    _write_csv(run_dir / "financial_assumptions.csv", assumptions_rows)
    _write_csv(run_dir / "scenario_monetization.csv", scen_monetized)
    _write_csv(run_dir / "risk_premium_breakdown.csv", premium_rows)
    _write_csv(run_dir / "mitigation_cost_benefit.csv", mitigation_fin)
    _write_csv(run_dir / "energy_forecast_costs.csv", energy_forecast_cost_rows)
    # monetary_numbers.csv gets a provenance header so constants are traceable.
    _write_monetary_csv(run_dir / "monetary_numbers.csv", master_rows, assumptions)

    _plot_scenario_cost_stack(run_dir / "scenario_cost_stack.svg", scen_monetized)
    _plot_carbon_price_band(run_dir / "scenario_carbon_price_band.svg", scen_monetized)
    _plot_energy_forecast_costs(run_dir / "energy_forecast_costs.svg", energy_forecast_cost_rows)
    _plot_mitigation_npv(run_dir / "mitigation_npv_payback.svg", mitigation_fin)
    _plot_dashboard(run_dir / "monetizable_outcomes_dashboard.svg", scen_monetized, mitigation_fin)

    top_mit = mitigation_fin[0]["lever"] if mitigation_fin else ""
    top_npv = mitigation_fin[0]["npv_10y_usd"] if mitigation_fin else 0.0
    baseline = baseline_row
    energy_tail = max(energy_forecast_cost_rows, key=lambda r: _to_float(r["total_annual_cost_usd"])) if energy_forecast_cost_rows else None
    summary = {
        "run_id": run_id,
        "retrained_any_model": False,
        "baseline_total_annual_cost_usd": baseline["total_annual_cost_usd"] if baseline else None,
        "top_mitigation_by_npv": top_mit,
        "top_mitigation_npv_10y_usd": top_npv,
        "max_forecast_annual_cost_usd": energy_tail["total_annual_cost_usd"] if energy_tail else None,
        "max_forecast_annual_cost_scenario": energy_tail["forecast_scenario"] if energy_tail else None,
        "max_forecast_annual_cost_year": energy_tail["year"] if energy_tail else None,
        "outputs": sorted([p.name for p in run_dir.glob("*") if p.is_file()]),
    }
    (run_dir / "monetary_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
