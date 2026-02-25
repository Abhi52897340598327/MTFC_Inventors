"""
MTFC Virginia Datacenter Energy Forecasting — Ridge Regression Feature Combiner
================================================================================
Combines feature importance weights from multiple methods using Ridge Regression.

This approach:
1. Collects normalized importance scores from XGBoost, Random Forest, 
   Mutual Information, Correlation, and Permutation Importance
2. Uses Ridge Regression with cross-validation to learn optimal combination weights
3. Produces a final unified feature importance ranking

This justifies feature selection by using a statistically principled method
that regularizes against overfitting to any single importance metric.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from utils import log, save_fig

OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, "feature_importance")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_prepare_data():
    """Load data and engineer features for importance analysis."""
    log.info("Loading data for Ridge Feature Combiner...")
    
    # Load Google Cluster data
    google_path = os.path.join(cfg.PROJECT_DIR, "Data_Sources", "google_cluster_utilization_2019.csv")
    df = pd.read_csv(google_path)
    df['timestamp'] = pd.to_datetime(df['real_timestamp'])
    
    # Target
    target_col = 'avg_cpu_utilization'
    
    # Feature engineering
    ts = df['timestamp']
    df['hour'] = ts.dt.hour
    df['minute'] = ts.dt.minute
    df['day_of_week'] = ts.dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_business_hour'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
    
    # Cyclical encodings
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Log transform
    if 'num_tasks_sampled' in df.columns:
        df['log_num_tasks'] = np.log1p(df['num_tasks_sampled'])
    
    # Lag features
    df['cpu_lag_1'] = df[target_col].shift(1)
    df['cpu_lag_10'] = df[target_col].shift(10)
    df['cpu_rolling_mean_10'] = df[target_col].rolling(10, min_periods=1).mean()
    df['cpu_rolling_std_10'] = df[target_col].rolling(10, min_periods=1).std().fillna(0)
    if 'num_tasks_sampled' in df.columns:
        df['tasks_lag_1'] = df['num_tasks_sampled'].shift(1)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    # Select only numeric columns that are not metadata
    feature_cols = [col for col in df.columns 
                   if col not in ['real_timestamp', 'timestamp', target_col, 'hour_of_day']
                   and df[col].dtype in ['int64', 'float64']]
    
    # Additional check: remove any columns with remaining NaN or infinite values
    valid_cols = []
    for col in feature_cols:
        if df[col].isna().sum() == 0 and np.isfinite(df[col]).all():
            valid_cols.append(col)
    feature_cols = valid_cols
    
    log.info(f"  Prepared {len(df)} samples with {len(feature_cols)} features")
    return df, feature_cols, target_col


def compute_all_importances(df, feature_cols, target_col):
    """Compute importance scores from multiple methods."""
    log.info("Computing feature importance from 5 methods...")
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Replace any remaining NaN/inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    
    importances = {}
    
    # 1. Correlation (absolute)
    log.info("  1/5: Correlation analysis...")
    correlations = df[feature_cols + [target_col]].corr()[target_col].drop(target_col)
    correlations = correlations.fillna(0)  # Fill NaN correlations with 0
    importances['correlation'] = correlations.abs()
    
    # 2. Gradient Boosting Importance
    log.info("  2/5: Gradient Boosting importance...")
    gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42)
    gb_model.fit(X, y)
    importances['gradient_boosting'] = pd.Series(gb_model.feature_importances_, index=feature_cols)
    
    # 3. Random Forest Importance
    log.info("  3/5: Random Forest importance...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X, y)
    importances['random_forest'] = pd.Series(rf_model.feature_importances_, index=feature_cols)
    
    # 4. Mutual Information
    log.info("  4/5: Mutual Information scores...")
    mi_scores = mutual_info_regression(X, y, random_state=42)
    importances['mutual_info'] = pd.Series(mi_scores, index=feature_cols)
    
    # 5. Permutation Importance
    log.info("  5/5: Permutation importance...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_perm = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    rf_perm.fit(X_train, y_train)
    perm_imp = permutation_importance(rf_perm, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    importances['permutation'] = pd.Series(perm_imp.importances_mean, index=feature_cols)
    
    return importances


def normalize_importances(importances):
    """Normalize all importance scores to [0, 1] range."""
    normalized = {}
    for method, scores in importances.items():
        # Fill NaN values with 0
        scores = scores.fillna(0)
        min_val = scores.min()
        max_val = scores.max()
        if max_val - min_val > 1e-10:
            normalized[method] = (scores - min_val) / (max_val - min_val)
        else:
            normalized[method] = scores * 0  # All zeros if no variance
        # Ensure no NaN remains
        normalized[method] = normalized[method].fillna(0)
    return normalized


def ridge_combine_importances(importances, df, feature_cols, target_col):
    """
    Use Ridge Regression to learn optimal weights for combining importance methods.
    
    The idea: We train a meta-model where each importance method provides a "vote"
    for each feature, and Ridge regression learns how to weight these votes
    based on actual predictive performance.
    """
    log.info("\n" + "=" * 60)
    log.info("RIDGE REGRESSION FEATURE COMBINER")
    log.info("=" * 60)
    
    # Create importance matrix (features × methods)
    importance_df = pd.DataFrame(importances)
    
    # Normalize to [0,1]
    normalized = normalize_importances(importances)
    norm_df = pd.DataFrame(normalized)
    
    # Fill any remaining NaN with 0
    norm_df = norm_df.fillna(0)
    
    log.info(f"\nImportance matrix shape: {norm_df.shape}")
    log.info(f"Methods: {list(norm_df.columns)}")
    
    # Method 1: Direct Ridge Regression on importance scores
    # We want to find weights w_i such that sum(w_i * importance_i) gives best ranking
    # Use actual model performance as the target
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Replace any NaN/inf values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # For each feature, compute its actual contribution via a simple model
    log.info("\nComputing actual feature contributions...")
    actual_contributions = []
    
    for i, feat in enumerate(feature_cols):
        # Single feature model R²
        from sklearn.linear_model import LinearRegression
        single_model = LinearRegression()
        X_feat = X_train[:, i:i+1]
        # Check for NaN/inf in this feature
        if np.isnan(X_feat).any() or np.isinf(X_feat).any():
            actual_contributions.append(0)
            continue
        single_model.fit(X_feat, y_train)
        score = single_model.score(X_test[:, i:i+1], y_test)
        actual_contributions.append(max(0, score))  # Clamp negatives
    
    actual_contrib = pd.Series(actual_contributions, index=feature_cols)
    actual_contrib_normalized = (actual_contrib - actual_contrib.min()) / (actual_contrib.max() - actual_contrib.min() + 1e-10)
    
    # Now use Ridge to learn optimal weights
    # X_meta: each row is a feature, columns are importance scores from each method
    # y_meta: actual contribution of that feature
    X_meta = norm_df.values
    y_meta = actual_contrib_normalized.values
    
    # Final check for NaN/inf - replace with 0
    X_meta = np.nan_to_num(X_meta, nan=0.0, posinf=0.0, neginf=0.0)
    y_meta = np.nan_to_num(y_meta, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ridge CV to find optimal alpha
    ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
    ridge.fit(X_meta, y_meta)
    
    log.info(f"\nRidge Regression Results:")
    log.info(f"  Optimal alpha: {ridge.alpha_:.3f}")
    log.info(f"  R² score: {ridge.score(X_meta, y_meta):.4f}")
    
    # Get learned weights for each method
    method_weights = pd.Series(ridge.coef_, index=norm_df.columns)
    method_weights_sum = method_weights.sum()
    if abs(method_weights_sum) > 1e-10:
        method_weights = method_weights / method_weights_sum  # Normalize to sum to 1
    
    log.info(f"\nLearned Method Weights (Ridge Regression):")
    log.info("-" * 40)
    for method, weight in method_weights.sort_values(ascending=False).items():
        log.info(f"  {method:<20s}: {weight:.4f}")
    
    # Compute combined importance using learned weights
    combined_importance = (norm_df * method_weights.values).sum(axis=1)
    combined_importance = combined_importance.sort_values(ascending=False)
    
    log.info(f"\n" + "=" * 60)
    log.info("RIDGE-COMBINED FEATURE IMPORTANCE RANKING")
    log.info("=" * 60)
    
    for i, (feat, score) in enumerate(combined_importance.head(20).items(), 1):
        log.info(f"  {i:2d}. {feat:<30s}: {score:.4f}")
    
    return combined_importance, method_weights, norm_df


def create_ridge_visualizations(combined_importance, method_weights, norm_df, feature_cols):
    """Create visualizations for the Ridge Feature Combiner."""
    log.info("\nGenerating Ridge Combiner visualizations...")
    
    # 1. Method Weights Pie Chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Pie chart of method weights
    ax = axes[0]
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12']
    weights_positive = method_weights.clip(lower=0)
    weights_positive = weights_positive / weights_positive.sum()
    
    wedges, texts, autotexts = ax.pie(
        weights_positive.values,
        labels=weights_positive.index,
        autopct='%1.1f%%',
        colors=colors[:len(weights_positive)],
        explode=[0.02] * len(weights_positive),
        shadow=True
    )
    ax.set_title('Ridge-Learned Method Weights\n(Optimal Combination)', fontsize=12, fontweight='bold')
    
    # 2. Combined Feature Importance Bar Chart
    ax = axes[1]
    top_15 = combined_importance.head(15)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_15)))
    bars = ax.barh(range(len(top_15)), top_15.values, color=colors)
    ax.set_yticks(range(len(top_15)))
    ax.set_yticklabels(top_15.index)
    ax.invert_yaxis()
    ax.set_xlabel('Ridge-Combined Importance Score', fontsize=11)
    ax.set_title('Top 15 Features\n(Ridge-Weighted Combination)', fontsize=12, fontweight='bold')
    
    # 3. Heatmap of normalized importances by method
    ax = axes[2]
    top_features = combined_importance.head(12).index
    heatmap_data = norm_df.loc[top_features]
    
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Normalized Score'},
        ax=ax
    )
    ax.set_title('Feature Scores by Method\n(Top 12 Features)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature')
    ax.set_xlabel('Importance Method')
    plt.xticks(rotation=30, ha='right')
    
    plt.tight_layout()
    save_fig(fig, 'ridge_feature_combiner')
    plt.close()
    
    # 4. Comparison: Simple Average vs Ridge-Weighted
    fig, ax = plt.subplots(figsize=(14, 10))
    
    simple_avg = norm_df.mean(axis=1).sort_values(ascending=False)
    ridge_combined = combined_importance.sort_values(ascending=False)
    
    # Take top N from each (handle case where we have fewer features)
    n_features = min(15, len(simple_avg))
    top_simple = simple_avg.head(n_features)
    
    x = np.arange(n_features)
    width = 0.35
    
    # Plot both side by side
    bars1 = ax.bar(x - width/2, top_simple.values, width, label='Simple Average', color='#3498db', alpha=0.8, edgecolor='navy')
    bars2 = ax.bar(x + width/2, [ridge_combined.get(f, 0) for f in top_simple.index], 
                   width, label='Ridge-Weighted', color='#e74c3c', alpha=0.8, edgecolor='darkred')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, rotation=90)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, rotation=90)
    
    ax.set_xticks(x)
    ax.set_xticklabels(top_simple.index, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Importance Score', fontsize=12)
    ax.set_xlabel('Feature', fontsize=12)
    ax.set_title('Simple Average vs Ridge-Weighted Feature Importance\nComparing Naive Averaging with Optimized Combination', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(top_simple.max(), ridge_combined.max()) * 1.25)  # Add headroom for labels
    
    # Add explanatory text
    ax.text(0.02, 0.98, 'Ridge Regression learns optimal weights\nto combine 5 importance methods', 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_fig(fig, 'ridge_vs_simple_average')
    plt.close()
    
    log.info("✓ Saved: ridge_feature_combiner.png")
    log.info("✓ Saved: ridge_vs_simple_average.png")


def save_ridge_results(combined_importance, method_weights, norm_df):
    """Save Ridge Combiner results to CSV."""
    # Combined ranking
    results_df = pd.DataFrame({
        'feature': combined_importance.index,
        'ridge_combined_score': combined_importance.values,
        'rank': range(1, len(combined_importance) + 1)
    })
    
    # Add individual method scores
    for method in norm_df.columns:
        results_df[f'{method}_score'] = norm_df.loc[results_df['feature'], method].values
    
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'ridge_combined_feature_ranking.csv'), index=False)
    
    # Method weights
    weights_df = pd.DataFrame({
        'method': method_weights.index,
        'weight': method_weights.values,
        'weight_pct': (method_weights.values * 100)
    })
    weights_df.to_csv(os.path.join(OUTPUT_DIR, 'ridge_method_weights.csv'), index=False)
    
    log.info("✓ Saved: ridge_combined_feature_ranking.csv")
    log.info("✓ Saved: ridge_method_weights.csv")


def main():
    """Run Ridge Regression Feature Combiner analysis."""
    log.info("\n" + "=" * 70)
    log.info("RIDGE REGRESSION FEATURE COMBINER")
    log.info("Combining Feature Importance Weights via Regularized Regression")
    log.info("=" * 70)
    
    # Load data
    df, feature_cols, target_col = load_and_prepare_data()
    
    # Compute importances from all methods
    importances = compute_all_importances(df, feature_cols, target_col)
    
    # Ridge combine
    combined_importance, method_weights, norm_df = ridge_combine_importances(
        importances, df, feature_cols, target_col
    )
    
    # Visualizations
    create_ridge_visualizations(combined_importance, method_weights, norm_df, feature_cols)
    
    # Save results
    save_ridge_results(combined_importance, method_weights, norm_df)
    
    log.info("\n" + "=" * 70)
    log.info("RIDGE FEATURE COMBINER COMPLETE")
    log.info(f"Results saved to: {OUTPUT_DIR}")
    log.info("=" * 70)
    
    return combined_importance, method_weights


if __name__ == "__main__":
    main()
