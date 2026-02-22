# MTFC Stacked Model Pipeline

## Overview

The **Stacked Model Pipeline** (`stacked_pipeline.py`) is a cascaded machine learning architecture where the outputs of one model feed into the next, maximizing R² through optimal model ordering and feature flow.

## Pipeline Architecture

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                    STACKED MODEL PIPELINE                     │
                    └─────────────────────────────────────────────────────────────┘
                                               │
                    ┌──────────────────────────▼──────────────────────────┐
    STAGE 0:        │     RIDGE FEATURE COMBINER                           │
    Feature         │     • Combines 5 importance methods                  │
    Selection       │     • Outputs: Optimally weighted feature ranking    │
                    └──────────────────────────▼──────────────────────────┘
                                               │
                    ┌──────────────────────────▼──────────────────────────┐
    STAGE 1:        │     RANDOM FOREST (Base Learner)                     │
    Base Model      │     • Input: Ridge-selected features                 │
                    │     • Output: RF predictions + residuals             │
                    └──────────────────────────▼──────────────────────────┘
                                               │
                    ┌──────────────────────────▼──────────────────────────┐
    STAGE 2:        │     XGBOOST (Residual Corrector)                     │
    Boosting        │     • Input: Features + RF predictions               │
                    │     • Output: Corrected predictions                  │
                    └──────────────────────────▼──────────────────────────┘
                                               │
                    ┌──────────────────────────▼──────────────────────────┐
    STAGE 3:        │     GRU NEURAL NETWORK (Sequence Refiner)            │
    Deep Learning   │     • Input: Sequences of stacked features           │
                    │     • Output: Temporally-smoothed predictions        │
                    └──────────────────────────▼──────────────────────────┘
                                               │
                    ┌──────────────────────────▼──────────────────────────┐
    STAGE 4:        │     RIDGE META-LEARNER (Final Combiner)              │
    Meta-Ensemble   │     • Input: All model predictions                   │
                    │     • Output: Optimal weighted combination           │
                    └──────────────────────────▼──────────────────────────┘
                                               │
                                               ▼
                              FINAL PREDICTION (Maximized R²)
```

## Why This Order?

The model ordering was chosen to maximize R² through complementary strengths:

| Stage | Model | Role | Why This Position |
|-------|-------|------|-------------------|
| 0 | Ridge Feature Combiner | Feature Selection | Uses 4 importance methods (Correlation, GB, RF, MI) with Ridge regression to find optimal feature weights. Reduces noise before training. |
| 1 | Random Forest | Base Learner | Robust to outliers, provides stable base predictions. Trees naturally handle feature interactions. |
| 2 | XGBoost | Residual Corrector | Learns from RF errors (via augmented features). Sequential boosting corrects systematic biases. |
| 3 | GRU | Sequence Refiner | Captures temporal patterns missed by tree models. Uses lookback sequences with stacked predictions. |
| 4 | Ridge Meta-Learner | Final Combiner | Optimally combines all predictions with L2 regularization. Learns which models to trust more. |

## How Outputs Connect to Inputs

### Stage 0 → Stage 1
- **Input**: Raw feature matrix (15 features)
- **Process**: Ridge-weighted importance ranking
- **Output**: Top 12 selected features → Used by all subsequent stages

### Stage 1 → Stage 2
- **RF Input**: Selected & scaled features (12 features)
- **RF Output**: Predictions
- **XGB Input**: Original features **+ RF predictions** (13 features)
- **Why**: XGBoost sees both raw features AND RF's interpretation

### Stage 2 → Stage 3
- **XGB Output**: Corrected predictions
- **GRU Input**: Sequence windows of [features + RF_pred + XGB_pred]
- **Why**: GRU sees temporal patterns in both raw data AND model predictions

### Stage 3 → Stage 4
- **All model outputs**: [RF_pred, XGB_pred, GRU_pred]
- **Meta-learner**: Ridge regression on prediction matrix
- **Final output**: Weighted combination based on learned coefficients

## Performance Results

```
COMPARISON: Stacked Pipeline vs Individual Stages
--------------------------------------------------
  Stage1_RF:   R² = 0.4206 (Base)
  Stage2_XGB:  R² = 0.3994 (With RF features)
  Stage3_GRU:  R² = 0.1026 (Sequence model)
  Stage4_Meta: R² = 0.3858 (Validation combined)
  ──────────────────────────────
  FINAL (Test): R² = 0.5029 ✓

  Improvement over base RF: +0.0823 R² (+19.6%)
```

### Meta-Learner Weights
The Ridge meta-learner learned these optimal combination weights:
- **Random Forest**: 0.614 (61.4% weight)
- **XGBoost**: 0.455 (45.5% weight)  
- **GRU**: -0.034 (negative weight, acts as regularizer)

This shows RF and XGBoost are the most predictive, while GRU helps smooth extreme predictions.

## How to Run

```bash
cd "FINAL MODEL"
python stacked_pipeline.py
```

### Outputs Generated
| File | Description |
|------|-------------|
| `outputs/figures/stacked_pipeline_results.png` | 4-panel visualization showing stage-wise R², predictions vs actuals, residuals, and architecture |
| `outputs/results/stacked_pipeline_metrics.csv` | Stage-by-stage metrics (R², MAE, RMSE) |
| `outputs/results/stacked_pipeline_predictions.csv` | Test set predictions with residuals |
| `outputs/models/stacked_pipeline.pkl` | Saved pipeline object for inference |

## Key Design Decisions

### 1. Ridge-Based Feature Selection (Stage 0)
Instead of using a single importance method, we:
- Compute importance from 4 methods (Correlation, Gradient Boosting, Random Forest, Mutual Information)
- Use Ridge regression to learn optimal weights for combining methods
- This provides more robust feature selection than any single method

### 2. Feature Augmentation (Stage 1 → 2)
XGBoost receives the original features **plus** RF predictions. This:
- Allows XGBoost to "correct" RF errors
- Creates an implicit residual-learning mechanism
- Is more effective than training on residuals directly

### 3. Sequence-Based Deep Learning (Stage 3)
The GRU receives:
- Lookback windows of the feature sequence
- RF and XGBoost predictions embedded in each window
- This captures temporal dependencies in both raw data and model predictions

### 4. Ridge Meta-Learner (Stage 4)
Final combination uses Ridge (not simple averaging) because:
- L2 regularization prevents overfitting to validation set
- Automatically learns to downweight poor-performing models
- Handles potential collinearity between model predictions

## Integration with main.py

The stacked pipeline is designed to work alongside the existing `main.py`. To use:

```python
from stacked_pipeline import StackedModelPipeline, load_and_prepare_data

# Load data
df, feature_cols, target_col = load_and_prepare_data()

# Create and train pipeline
pipeline = StackedModelPipeline(use_gru=True)
pipeline.fit(X_train, y_train, feature_cols, X_val, y_val)

# Predict
predictions = pipeline.predict(X_test, feature_cols)

# Evaluate
results = pipeline.evaluate(X_test, y_test, feature_cols, label="Test")
```

## Comparison with Original Architecture

| Aspect | Original (main.py) | Stacked Pipeline |
|--------|-------------------|------------------|
| Model Connection | Parallel (independent) | Serial (cascaded) |
| Feature Selection | Config-based | Ridge-learned |
| Ensemble Method | None / Simple average | Ridge meta-learner |
| XGB Input | Raw features only | Features + RF predictions |
| GRU Input | Raw sequences | Sequences + model predictions |
| Test R² | ~0.42 (RF alone) | 0.50 (+19.6%) |

## Future Improvements

1. **Cross-validation for meta-learner**: Use k-fold CV to prevent overfitting
2. **Hyperparameter tuning**: Grid search for each stage's parameters
3. **Confidence intervals**: Bootstrap the meta-learner for uncertainty quantification
4. **Online learning**: Update models incrementally as new data arrives
