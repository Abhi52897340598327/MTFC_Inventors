# Literature Citation Usage Map (MTFC)

This file maps external literature citations to the exact MTFC report sections and claims they support.

| Citation | MTFC section(s) | Claim/equation supported |
|---|---|---|
| Barroso & Hoelzle (2007), *The Case for Energy-Proportional Computing* | 1. Background Information (Stage 2) | Linear IT power assumption: capacity-scaled idle-plus-dynamic utilization model. |
| Zhang et al. (2013), *High-level energy model for heterogeneous data centers* | 1. Background Information (Stages 2 and 4) | Justification for high-level utilization-based system energy modeling. |
| CloudSim `PowerModelLinear` documentation | 3. Mathematics Methodology (Stages 2-4 deterministic path) | Practical implementation precedent for linear utilization-to-power modeling. |
| Lei & Masanet (2020), *Location-specific PUE prediction* | 1. Background Information (Stage 3), 5. Recommendations | Weather-sensitive PUE behavior and climate-linked efficiency variation. |
| Google Data Centers Efficiency/PUE page | 1. Background Information, 5. Recommendations | Real-world PUE efficiency benchmarks and operational cooling context. |
| ASHRAE TC9.9 thermal guidance white paper | 1. Background Information (Stage 3), 5.4 Recommendations | Thermal/humidity operating context for temperature/dew-point driven cooling assumptions. |
| The Green Grid, *PUE and DCiE* | 1. Background Information (Stages 4 and 6) | PUE identity used to derive total power from IT power and then emissions. |
| U.S. EIA (EIA-930 variable context) | 2. Data Methodology, 4.4 Energy Forecast Integration | Meaning of demand, net generation, and interchange variables used in grid-impact outputs. |
| NERC reliability assessment process (reserve margin context) | 5.4 Recommendations | Reliability framing for stress-threshold operations and peak-risk mitigation. |
| FERC/Brattle resource adequacy report | 5.1-5.4 Recommendations | Economic and reliability motivation for load shifting/peak shaving playbooks. |

## Notes for evaluators

- These references support assumptions and interpretation logic.
- Numerical model performance, sensitivity, and forecast tables are still sourced from run artifacts in `REAL FINAL FILES/outputs/...`.
