"""
Master Execution Script for MTFC AI Datacenter Impact Model
Runs entire pipeline from data preparation through visualization.
"""

import time
from pathlib import Path
import pandas as pd

# Import all modules
from models import data_preparation
from models import model_1_energy
from models import model_2_grid_mix
from models import model_3_ai_growth
from models import model_4_integration
from visualizations import plot_energy
from visualizations import plot_grid_mix
from visualizations import plot_carbon
from visualizations import plot_co2_breakdown
from visualizations import plot_grid_stress

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / 'REAL FINAL FILES'

def print_banner(phase_name):
    print("\n" + "="*70)
    print(f"  {phase_name}")
    print("="*70)

def run_pipeline():
    start_time = time.time()
    
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  MTFC AI DATACENTER CARBON IMPACT FORECASTING MODEL".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    try:
        # Phase 0: Data Preparation
        print_banner("PHASE 0: Data Preparation")
        print("Loading actual monthly EIA generation data...")
        data_preparation.prepare_monthly_energy()
        data_preparation.prepare_monthly_grid_mix()
        data_preparation.prepare_monthly_pue()
        data_preparation.prepare_monthly_ai_proxy()
        data_preparation.prepare_monthly_temperature()
        print("✓ Data preparation complete")
        
        # Phase 1: Energy Model
        print_banner("PHASE 1: Energy Generation Forecast (SARIMA)")
        print("Fitting SARIMA(1,1,1)(1,1,1)_12 (no exogenous)...")
        model_1_energy.run_energy_forecast()
        print("✓ Energy forecasting complete")
        
        # Phase 2: Grid Mix Model
        print_banner("PHASE 2: Grid Composition Forecast (SARIMA/SARIMAX)")
        print("Modeling grid source evolution:")
        print("  - Coal, Gas, Nuclear: SARIMA(1,1,1)(1,1,1)_12")
        print("  - Renewable: SARIMAX(1,1,1)(1,1,1)_12 with AI spending")
        model_2_grid_mix.run_grid_mix_forecast()
        print("✓ Grid mix forecasting complete")
        
        # Phase 3: AI Growth Model
        print_banner("PHASE 3: AI Growth Multiplier Forecast (Exponential)")
        print("Fitting exponential curve to DC spending with decaying growth rate...")
        model_3_ai_growth.run_ai_growth_forecast()
        print("✓ AI growth forecasting complete")
        
        # Phase 4: Integration & Risk Metrics
        print_banner("PHASE 4: Integration & Carbon Risk Analysis")
        print("Calculating interaction terms:")
        print("  - Carbon intensity from grid mix")
        print("  - CO2 emissions by source")
        print("  - DC share of grid generation")
        model_4_integration.run_integration()
        print("✓ Integration complete")
        
        # Phase 5: Visualizations
        print_banner("PHASE 5: Generating Visualizations")
        plot_energy.plot_energy_forecast()
        plot_grid_mix.plot_grid_mix()
        plot_carbon.plot_carbon_intensity()
        plot_co2_breakdown.plot_co2_breakdown()
        plot_grid_stress.plot_grid_stress()
        print("✓ All visualizations generated")
        
        # Phase 6: Summary Statistics
        print_banner("PHASE 6: Summary Statistics")
        
        # Load integrated forecast
        integrated = pd.read_csv(OUTPUT_DIR / 'model_forecasts' / 'forecast_integrated.csv')
        
        # Year-end values
        year_2026 = integrated[integrated['date'].str.startswith('2026')].iloc[-1]
        year_2030 = integrated[integrated['date'].str.startswith('2030')].iloc[-1]
        year_2038 = integrated[integrated['date'].str.startswith('2038')].iloc[-1]
        
        print("\n📊 KEY FORECAST METRICS:")
        print(f"\n2026 (Baseline):")
        print(f"  Grid Electricity: {year_2026['electricity_gwh']:.1f} GWh (generation)")
        print(f"  DC Share: {year_2026['dc_share_pct']:.1f}%")
        print(f"  Carbon Intensity: {year_2026['carbon_intensity']:.3f} lb CO2/kWh")
        print(f"  Renewable Mix: {year_2026['renewable_pct']:.1f}%")
        
        print(f"\n2030 (Mid-term):")
        print(f"  Grid Electricity: {year_2030['electricity_gwh']:.1f} GWh")
        print(f"  DC Share: {year_2030['dc_share_pct']:.1f}%")
        print(f"  Carbon Intensity: {year_2030['carbon_intensity']:.3f} lb CO2/kWh")
        print(f"  Renewable Mix: {year_2030['renewable_pct']:.1f}%")
        
        print(f"\n2038 (End Forecast):")
        print(f"  Grid Electricity: {year_2038['electricity_gwh']:.1f} GWh")
        print(f"  DC Share: {year_2038['dc_share_pct']:.1f}%")
        print(f"  Carbon Intensity: {year_2038['carbon_intensity']:.3f} lb CO2/kWh")
        print(f"  Renewable Mix: {year_2038['renewable_pct']:.1f}%")
        
        # Growth rates
        energy_growth = ((year_2038['electricity_gwh'] / year_2026['electricity_gwh']) - 1) * 100
        dc_share_growth = year_2038['dc_share_pct'] - year_2026['dc_share_pct']
        carbon_reduction = ((year_2026['carbon_intensity'] - year_2038['carbon_intensity']) / 
                           year_2026['carbon_intensity']) * 100
        renewable_growth = year_2038['renewable_pct'] - year_2026['renewable_pct']
        
        print(f"\n📈 2026-2038 TRENDS:")
        sign = '+' if energy_growth >= 0 else ''
        print(f"  Energy Growth: {sign}{energy_growth:.1f}%")
        print(f"  DC Share Change: +{dc_share_growth:.1f} percentage points")
        print(f"  Carbon Intensity Reduction: -{carbon_reduction:.1f}%")
        print(f"  Renewable Growth: +{renewable_growth:.1f} percentage points")
        
        # Peak values
        max_dc_share = integrated['dc_share_pct'].max()
        max_dc_share_date = integrated.loc[integrated['dc_share_pct'].idxmax(), 'date']
        max_carbon = integrated['carbon_intensity'].max()
        max_carbon_date = integrated.loc[integrated['carbon_intensity'].idxmax(), 'date']
        
        print(f"\n⚠️  PEAK RISK METRICS:")
        print(f"  Maximum DC Share: {max_dc_share:.1f}% ({max_dc_share_date})")
        print(f"  Maximum Carbon Intensity: {max_carbon:.3f} lb CO2/kWh ({max_carbon_date})")
        
        # Threshold warnings
        months_over_100 = (integrated['dc_share_pct'] > 100).sum()
        months_over_50 = ((integrated['dc_share_pct'] > 50) & 
                         (integrated['dc_share_pct'] <= 100)).sum()
        
        print(f"\n🚨 DC SHARE WARNINGS:")
        print(f"  Months DC > 100% of Grid (Exceeds Capacity): {months_over_100}")
        print(f"  Months DC 50-100% of Grid (High Stress): {months_over_50}")
        
        # Save summary
        summary_stats = {
            '2026_electricity_gwh': year_2026['electricity_gwh'],
            '2026_dc_share_pct': year_2026['dc_share_pct'],
            '2026_carbon_intensity': year_2026['carbon_intensity'],
            '2038_electricity_gwh': year_2038['electricity_gwh'],
            '2038_dc_share_pct': year_2038['dc_share_pct'],
            '2038_carbon_intensity': year_2038['carbon_intensity'],
            'energy_growth_pct': energy_growth,
            'dc_share_change_pp': dc_share_growth,
            'carbon_reduction_pct': carbon_reduction,
            'renewable_growth_pp': renewable_growth,
            'max_dc_share_pct': max_dc_share,
            'max_dc_share_date': max_dc_share_date,
            'months_dc_over_100': months_over_100,
            'months_dc_over_50': months_over_50
        }
        
        pd.DataFrame([summary_stats]).to_csv(
            OUTPUT_DIR / 'summary_statistics.csv', 
            index=False
        )
        print("\n✓ Summary statistics saved to summary_statistics.csv")
        
        # Final banner
        elapsed = time.time() - start_time
        print_banner("PIPELINE COMPLETE")
        print(f"\n⏱️  Total Execution Time: {elapsed:.1f} seconds")
        print(f"📁 Output Location: {OUTPUT_DIR}")
        print(f"\n📊 Generated Files:")
        print(f"  - 7 prepared datasets (prepared_data/)")
        print(f"  - 9 forecast files (model_forecasts/)")
        print(f"  - 5 visualizations (visualizations/)")
        print(f"  - 1 summary report (summary_statistics.csv)")
        print("\n✅ Model ready for MTFC submission!")
        print("\n" + "#"*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: Pipeline failed")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    success = run_pipeline()
    exit(0 if success else 1)
