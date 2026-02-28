# MTFC Inventors - Data Center Carbon Emissions Digital Twin

A comprehensive modeling system for predicting and analyzing carbon emissions from data centers in Northern Virginia, focusing on the relationship between weather patterns, grid operations, and environmental impact.

## Project Overview

This project develops a digital twin model for data center operations, integrating:
- Weather data (temperature, climate patterns)
- Grid demand and generation data from PJM Interconnection
- Carbon emissions from Virginia's electricity grid
- Data center power consumption patterns

## Key Features

- **Multi-source Data Integration**: Combines NOAA weather data, EIA grid data, EPA emissions data, and Google cluster utilization patterns
- **Predictive Modeling**: Machine learning models for forecasting power consumption and carbon emissions
- **Grid Stress Analysis**: Evaluates impact on electrical grid during peak demand periods
- **Sensitivity Analysis**: Assesses how different factors affect emissions

## Project Structure

```
├── FINAL MODEL/           # Core modeling code
│   ├── main.py           # Main execution script
│   ├── data_loader.py    # Data loading and preprocessing
│   ├── feature_engineering.py
│   ├── forecasting.py
│   ├── carbon_emissions.py
│   ├── grid_stress.py
│   └── models/           # ML model implementations
├── Data_Sources/         # Raw and cleaned data files
│   ├── cleaned/          # Processed datasets
│   └── *.csv            # Various data sources
├── Figures/              # Generated visualizations
├── Information_Docs/     # Documentation and guides
└── Model_Files/          # Additional model utilities
```

## Data Sources

- **NOAA**: Historical weather data for Dulles/Ashburn area
- **EIA (Energy Information Administration)**: PJM grid demand, generation, and fuel mix data
- **EPA**: Virginia CO2 emissions data
- **Google**: Cluster utilization patterns (2019 dataset)

## Requirements

- Python 3.8+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Additional dependencies in requirements.txt (to be generated)

## Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Review data sources in `Data_Sources/`
4. Run the digital twin: `python "FINAL MODEL/run_digital_twin.py"`

## Documentation

Comprehensive documentation is available in the `Information_Docs/` folder:
- `YOUR_NEXT_STEPS.txt` - Getting started guide
- `complete_data_guide.txt` - Data source documentation
- `model_summary.txt` - Model architecture overview

## Authors

MTFC Inventors Team

## License

[To be determined]

## Notes

This project uses real-world data to model environmental impact of data centers. API keys and sensitive credentials are not included in this repository.
