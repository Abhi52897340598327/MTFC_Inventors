ACTUAL_FINAL_MODEL_SUBMISSION

Winner: BAD_FINAL_MODEL_NO_USE

Comparison protocol:
- Both pipelines were trained and evaluated on the same exported dataset:
  /Users/abhiraamvenigalla/MTFC_Inventors/REAL FINAL FILES/outputs/analysis/real_vs_bad_comparison/common_dataset_same_for_both_models.csv
- Both used the same chronological split: 85% train / 15% holdout.
- Decision rule: highest Stage6 (Emissions) holdout R2, tie-break by mean holdout R2 across stages 1-6.

Key files:
- comparison/stage_metrics_comparison.csv
- comparison/overall_model_comparison.csv
- comparison/winner_decision.json
- winner_model_bundle/ (full winning model code, outputs, figures, model artifacts, and graph scripts)
