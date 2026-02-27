"""
Run the streamlined MTFC 4-model stack end-to-end.

Order:
1) Model 1: sarimax_energy_usage.py
2) Model 2: model2_sarimax_carbon_emissions.py
3) Model 3: model3_grid_stress_analysis.py
4) Model 4: model4_monetization_cost_benefit.py

Run from repository root:
    python GOOD_MainModels/run_mtfc_four_models.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

MODEL_SCRIPTS = [
    HERE / "sarimax_energy_usage.py",
    HERE / "model2_sarimax_carbon_emissions.py",
    HERE / "model3_grid_stress_analysis.py",
    HERE / "model4_monetization_cost_benefit.py",
]


def main() -> None:
    print("=" * 72)
    print("MTFC Four-Model Run")
    print("=" * 72)

    for i, script in enumerate(MODEL_SCRIPTS, start=1):
        if not script.exists():
            raise FileNotFoundError(f"Model script not found: {script}")
        print(f"\n[{i}/4] Running {script.name}")
        subprocess.run([sys.executable, str(script)], cwd=str(REPO_ROOT), check=True)

    print("\nAll four models completed successfully.")
    print("Key outputs:")
    print("- GOOD_MainModels/energy_usage_forecast.csv")
    print("- GOOD_MainModels/carbon_emissions_forecast.csv")
    print("- GOOD_MainModels/grid_stress_annual_summary.csv")
    print("- GOOD_MainModels/annual_risk_monetization.csv")
    print("- GOOD_MainModels/recommendation_cost_benefit.csv")


if __name__ == "__main__":
    main()
