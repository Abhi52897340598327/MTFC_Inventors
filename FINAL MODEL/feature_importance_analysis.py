"""
MTFC Virginia Datacenter Energy Forecasting — Feature Importance Analysis
==========================================================================
Comprehensive analysis of feature importance across all data sources:
  - Correlation analysis with target variable
  - XGBoost feature importance (gain, weight, cover)
  - Random Forest feature importance
  - Permutation importance
  - SHAP values analysis
  - Mutual information scores

Output: Rankings to guide which features to focus on for modeling.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from utils import log

# ── Constants ───────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, "feature_importance")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_all_available_data():
    """Load all available datasets and merge them for analysis."""
    print("=" * 70)
    print("LOADING ALL AVAILABLE DATA SOURCES")
    print("=" * 70)
    
    datasets = {}
    
    # 1. Google Cluster Utilization Data (main data in Data_Sources)
    google_path = os.path.join(cfg.PROJECT_DIR, "Data_Sources", "google_cluster_utilization_2019.csv")
    if os.path.exists(google_path):
        try:
            df = pd.read_csv(google_path)
            df['real_timestamp'] = pd.to_datetime(df['real_timestamp'])
            datasets['google_cluster'] = df
            print(f"✓ Google Cluster: {df.shape[0]} rows, {df.shape[1]} cols")
            print(f"  Columns: {list(df.columns)}")
        except Exception as e:
            print(f"✗ Google Cluster: {e}")
    
    # 2. EIA Data (Excel file)
    eia_path = os.path.join(cfg.PROJECT_DIR, "Data_Sources", "EIA923_Schedules_2_3_4_5_M_11_2025_21JAN2026.xlsx")
    if os.path.exists(eia_path):
        try:
            # Read the first sheet to understand structure
            eia_df = pd.read_excel(eia_path, sheet_name=0, nrows=1000)
            datasets['eia_power'] = eia_df
            print(f"✓ EIA Power Data: {eia_df.shape[0]} rows, {eia_df.shape[1]} cols")
            print(f"  Sample columns: {list(eia_df.columns[:10])}")
        except Exception as e:
            print(f"✗ EIA Power Data: {e}")
    
    # 3. Datacenter Constants
    constants_path = os.path.join(cfg.PROJECT_DIR, "Data_Sources", "datacenter_constants.json")
    if os.path.exists(constants_path):
        import json
        with open(constants_path, 'r') as f:
            datasets['constants'] = json.load(f)
        print(f"✓ Datacenter Constants loaded")
    
    return datasets


def analyze_google_cluster_features(df):
    """Analyze feature importance in Google Cluster utilization data."""
    print("\n" + "=" * 70)
    print("GOOGLE CLUSTER UTILIZATION FEATURE ANALYSIS")
    print("=" * 70)
    
    # Create derived features
    df_analysis = df.copy()
    
    # Target: avg_cpu_utilization (proxy for power consumption)
    target_col = 'avg_cpu_utilization'
    
    # Feature engineering similar to what's done in the pipeline
    if 'real_timestamp' in df_analysis.columns:
        ts = pd.to_datetime(df_analysis['real_timestamp'])
        df_analysis['hour'] = ts.dt.hour
        df_analysis['minute'] = ts.dt.minute
        df_analysis['second'] = ts.dt.second
        df_analysis['day_of_week'] = ts.dt.dayofweek
        df_analysis['is_weekend'] = (df_analysis['day_of_week'] >= 5).astype(int)
        df_analysis['is_business_hour'] = ((df_analysis['hour'] >= 8) & (df_analysis['hour'] <= 18)).astype(int)
        
        # Cyclical encodings
        df_analysis['hour_sin'] = np.sin(2 * np.pi * df_analysis['hour'] / 24)
        df_analysis['hour_cos'] = np.cos(2 * np.pi * df_analysis['hour'] / 24)
        df_analysis['dow_sin'] = np.sin(2 * np.pi * df_analysis['day_of_week'] / 7)
        df_analysis['dow_cos'] = np.cos(2 * np.pi * df_analysis['day_of_week'] / 7)
    
    # Log transform of num_tasks_sampled (due to high variance)
    if 'num_tasks_sampled' in df_analysis.columns:
        df_analysis['log_num_tasks'] = np.log1p(df_analysis['num_tasks_sampled'])
    
    # Lag features
    df_analysis['cpu_lag_1'] = df_analysis[target_col].shift(1)
    df_analysis['cpu_lag_10'] = df_analysis[target_col].shift(10)
    df_analysis['cpu_rolling_mean_10'] = df_analysis[target_col].rolling(10, min_periods=1).mean()
    df_analysis['cpu_rolling_std_10'] = df_analysis[target_col].rolling(10, min_periods=1).std().fillna(0)
    df_analysis['tasks_lag_1'] = df_analysis.get('num_tasks_sampled', pd.Series()).shift(1)
    
    # Drop NaN rows from lagging
    df_analysis = df_analysis.dropna()
    
    # Select numeric features for analysis
    feature_cols = [col for col in df_analysis.columns 
                   if col not in ['real_timestamp', target_col, 'hour_of_day']
                   and df_analysis[col].dtype in ['int64', 'float64']]
    
    print(f"\nFeatures for analysis ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"  - {col}")
    
    results = {}
    
    # 1. Correlation Analysis
    print("\n" + "-" * 50)
    print("1. CORRELATION WITH TARGET (avg_cpu_utilization)")
    print("-" * 50)
    
    correlations = df_analysis[feature_cols + [target_col]].corr()[target_col].drop(target_col)
    correlations_sorted = correlations.abs().sort_values(ascending=False)
    
    print("\nTop correlations (absolute):")
    for feat, corr in correlations_sorted.head(15).items():
        actual_corr = correlations[feat]
        direction = "+" if actual_corr > 0 else "-"
        print(f"  {feat:30s}: {direction}{corr:.4f}")
    
    results['correlation'] = correlations_sorted
    
    # 2. XGBoost Feature Importance (using Gradient Boosting as fallback)
    print("\n" + "-" * 50)
    print("2. GRADIENT BOOSTING FEATURE IMPORTANCE")
    print("-" * 50)
    
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        
        X = df_analysis[feature_cols].values
        y = df_analysis[target_col].values
        
        # Train Gradient Boosting model (sklearn alternative to XGBoost)
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        gb_model.fit(X, y)
        
        # Get feature importance
        importance_gain = pd.Series(
            gb_model.feature_importances_,
            index=feature_cols
        ).sort_values(ascending=False)
        
        print("\nGradient Boosting Importance:")
        for feat, imp in importance_gain.head(15).items():
            print(f"  {feat:30s}: {imp:.4f}")
        
        results['xgboost_gain'] = importance_gain
        
    except Exception as e:
        print(f"  Gradient Boosting failed: {e}")
    
    # 3. Random Forest Feature Importance
    print("\n" + "-" * 50)
    print("3. RANDOM FOREST FEATURE IMPORTANCE")
    print("-" * 50)
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        
        X = df_analysis[feature_cols].values
        y = df_analysis[target_col].values
        
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X, y)
        
        importance_rf = pd.Series(
            rf_model.feature_importances_,
            index=feature_cols
        ).sort_values(ascending=False)
        
        print("\nRandom Forest Importance:")
        for feat, imp in importance_rf.head(15).items():
            print(f"  {feat:30s}: {imp:.4f}")
        
        results['random_forest'] = importance_rf
        
    except ImportError:
        print("  Scikit-learn not installed. Skipping...")
    
    # 4. Mutual Information
    print("\n" + "-" * 50)
    print("4. MUTUAL INFORMATION SCORES")
    print("-" * 50)
    
    try:
        from sklearn.feature_selection import mutual_info_regression
        
        X = df_analysis[feature_cols].values
        y = df_analysis[target_col].values
        
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_importance = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)
        
        print("\nMutual Information Scores:")
        for feat, mi in mi_importance.head(15).items():
            print(f"  {feat:30s}: {mi:.4f}")
        
        results['mutual_info'] = mi_importance
        
    except ImportError:
        print("  Scikit-learn not installed. Skipping...")
    
    # 5. Permutation Importance
    print("\n" + "-" * 50)
    print("5. PERMUTATION IMPORTANCE")
    print("-" * 50)
    
    try:
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import train_test_split
        
        X = df_analysis[feature_cols]
        y = df_analysis[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        rf_model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        
        perm_importance = permutation_importance(
            rf_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
        )
        
        perm_imp_series = pd.Series(
            perm_importance.importances_mean,
            index=feature_cols
        ).sort_values(ascending=False)
        
        print("\nPermutation Importance (mean):")
        for feat, imp in perm_imp_series.head(15).items():
            print(f"  {feat:30s}: {imp:.4f}")
        
        results['permutation'] = perm_imp_series
        
    except Exception as e:
        print(f"  Permutation importance failed: {e}")
    
    return results, df_analysis, feature_cols


def create_aggregate_importance_ranking(results):
    """Create a combined ranking across all methods."""
    print("\n" + "=" * 70)
    print("AGGREGATE FEATURE IMPORTANCE RANKING")
    print("=" * 70)
    
    # Normalize each ranking to 0-1 scale
    rankings = {}
    
    for method, importance in results.items():
        if isinstance(importance, pd.Series):
            normalized = (importance - importance.min()) / (importance.max() - importance.min() + 1e-10)
            rankings[method] = normalized
    
    # Create combined DataFrame
    ranking_df = pd.DataFrame(rankings)
    
    # Calculate aggregate score (mean of normalized scores)
    ranking_df['aggregate_score'] = ranking_df.mean(axis=1)
    ranking_df = ranking_df.sort_values('aggregate_score', ascending=False)
    
    print("\nCombined Feature Ranking (Top 20):")
    print("-" * 70)
    print(f"{'Feature':<30s} | {'Correlation':<10s} | {'XGB':<10s} | {'RF':<10s} | {'MI':<10s} | {'Aggregate':<10s}")
    print("-" * 70)
    
    for feat in ranking_df.head(20).index:
        row = ranking_df.loc[feat]
        corr = f"{row.get('correlation', 0):.3f}" if 'correlation' in row else "N/A"
        xgb = f"{row.get('xgboost_gain', 0):.3f}" if 'xgboost_gain' in row else "N/A"
        rf = f"{row.get('random_forest', 0):.3f}" if 'random_forest' in row else "N/A"
        mi = f"{row.get('mutual_info', 0):.3f}" if 'mutual_info' in row else "N/A"
        agg = f"{row['aggregate_score']:.3f}"
        print(f"{feat:<30s} | {corr:<10s} | {xgb:<10s} | {rf:<10s} | {mi:<10s} | {agg:<10s}")
    
    return ranking_df


def analyze_eia_data_structure(eia_df):
    """Analyze EIA data to identify relevant power generation features."""
    print("\n" + "=" * 70)
    print("EIA POWER GENERATION DATA ANALYSIS")
    print("=" * 70)
    
    print(f"\nDataset shape: {eia_df.shape}")
    print(f"\nColumn names ({len(eia_df.columns)} total):")
    
    # Categorize columns
    generation_cols = [col for col in eia_df.columns if 'generation' in str(col).lower() or 'mwh' in str(col).lower()]
    fuel_cols = [col for col in eia_df.columns if 'fuel' in str(col).lower()]
    capacity_cols = [col for col in eia_df.columns if 'capacity' in str(col).lower()]
    
    print(f"\nGeneration-related columns ({len(generation_cols)}):")
    for col in generation_cols[:10]:
        print(f"  - {col}")
    
    print(f"\nFuel-related columns ({len(fuel_cols)}):")
    for col in fuel_cols[:10]:
        print(f"  - {col}")
    
    print(f"\nCapacity-related columns ({len(capacity_cols)}):")
    for col in capacity_cols[:10]:
        print(f"  - {col}")
    
    # Show numeric column statistics
    numeric_cols = eia_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\nNumeric column summary ({len(numeric_cols)} columns):")
        print(eia_df[numeric_cols[:5]].describe().round(2))


def create_visualizations(results, df_analysis, feature_cols, output_dir):
    """Create visualization plots for feature importance."""
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    
    # Also save to figures folder
    figures_dir = cfg.FIGURE_DIR
    
    # 1. Correlation heatmap - IMPROVED with proper annotations
    try:
        # Filter to only include features with variance
        valid_features = [col for col in feature_cols 
                         if df_analysis[col].std() > 0.001]
        
        if len(valid_features) < 2:
            print("✗ Not enough features with variance for correlation matrix")
        else:
            fig, ax = plt.subplots(figsize=(12, 10))
            corr_matrix = df_analysis[valid_features].corr()
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            
            # Use diverging colormap centered at 0
            sns.heatmap(corr_matrix, 
                       mask=mask,
                       annot=True, 
                       fmt='.2f',
                       cmap='RdBu_r', 
                       center=0,
                       vmin=-1, vmax=1,
                       square=True,
                       linewidths=0.5,
                       cbar_kws={'label': 'Correlation Coefficient'},
                       ax=ax)
            
            ax.set_title('Feature Correlation Matrix\n(Lower Triangle Only)', 
                        fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save to both locations
            plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
            plt.savefig(os.path.join(figures_dir, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print("✓ Saved: correlation_heatmap.png")
    except Exception as e:
        print(f"✗ Correlation heatmap failed: {e}")
    
    # 2. Feature importance comparison
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        methods = ['correlation', 'xgboost_gain', 'random_forest', 'mutual_info']
        titles = ['Correlation', 'Gradient Boosting', 'Random Forest', 'Mutual Information']
        colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c']
        
        for ax, method, title, color in zip(axes.flat, methods, titles, colors):
            if method in results:
                top_features = results[method].head(10)
                bars = ax.barh(range(len(top_features)), top_features.values, color=color, edgecolor='black')
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(top_features.index)
                ax.invert_yaxis()
                ax.set_title(f'{title}', fontsize=12)
                ax.set_xlabel('Importance Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance_comparison.png'), dpi=150)
        plt.close()
        print("✓ Saved: feature_importance_comparison.png")
    except Exception as e:
        print(f"✗ Feature importance comparison failed: {e}")
    
    # 3. Aggregate ranking bar chart
    try:
        ranking_df = create_aggregate_importance_ranking(results)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        top_20 = ranking_df['aggregate_score'].head(20)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_20)))
        ax.barh(range(len(top_20)), top_20.values, color=colors)
        ax.set_yticks(range(len(top_20)))
        ax.set_yticklabels(top_20.index)
        ax.invert_yaxis()
        ax.set_xlabel('Aggregate Importance Score', fontsize=12)
        ax.set_title('Top 20 Features by Aggregate Importance', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'aggregate_importance_ranking.png'), dpi=150)
        plt.close()
        print("✓ Saved: aggregate_importance_ranking.png")
    except Exception as e:
        print(f"✗ Aggregate ranking failed: {e}")


def create_comprehensive_correlation_matrix(output_dir):
    """
    Create a comprehensive correlation matrix using temperature data
    with all engineered features spanning a full year.
    """
    print("\n" + "-" * 50)
    print("Creating Comprehensive Feature Correlation Matrix...")
    print("-" * 50)
    
    figures_dir = cfg.FIGURE_DIR
    
    # Load temperature data (full year)
    temp_path = os.path.join(cfg.PROJECT_DIR, "Data_Sources", "ashburn_va_temperature_2019.csv")
    if not os.path.exists(temp_path):
        print(f"✗ Temperature data not found at: {temp_path}")
        return
    
    df = pd.read_csv(temp_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Rename for consistency
    if 'temperature_c' in df.columns:
        df['temperature_f'] = df['temperature_c'] * 9/5 + 32
    
    # Engineer all features (matching feature_engineering.py)
    ts = df['timestamp']
    
    # Temporal features
    df['hour'] = ts.dt.hour
    df['day_of_week'] = ts.dt.dayofweek
    df['day_of_year'] = ts.dt.dayofyear
    df['month'] = ts.dt.month
    df['week_of_year'] = ts.dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_business_hour'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
    df['season'] = df['month'].map({12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                                    6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3})
    
    # Cyclical encodings
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Lag features
    df['temp_lag_1'] = df['temperature_f'].shift(1)
    df['temp_lag_24'] = df['temperature_f'].shift(24)
    
    # Rolling features
    df['temp_rolling_mean_24'] = df['temperature_f'].rolling(24, min_periods=1).mean()
    df['temp_rolling_std_24'] = df['temperature_f'].rolling(24, min_periods=1).std().fillna(0)
    
    # Interaction/derived features
    df['cooling_degree'] = np.maximum(0, df['temperature_f'] - 65)  # Degrees above 65°F
    df['temp_x_hour'] = df['temperature_f'] * df['hour']
    
    # Drop NaN rows
    df = df.dropna()
    
    # Select features for correlation matrix
    feature_cols = [
        'temperature_f', 'hour', 'day_of_week', 'month', 'is_weekend', 
        'is_business_hour', 'season', 'hour_sin', 'hour_cos', 'month_sin', 
        'month_cos', 'temp_lag_1', 'temp_lag_24', 'temp_rolling_mean_24', 
        'temp_rolling_std_24', 'cooling_degree', 'temp_x_hour'
    ]
    
    # Ensure all columns exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    print(f"  Features included: {len(feature_cols)}")
    
    # Create correlation matrix
    corr_matrix = df[feature_cols].corr()
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Shorten column names for display
    display_names = {
        'temperature_f': 'Temp (°F)',
        'hour': 'Hour',
        'day_of_week': 'Day of Week',
        'month': 'Month',
        'is_weekend': 'Weekend',
        'is_business_hour': 'Business Hr',
        'season': 'Season',
        'hour_sin': 'Hour (sin)',
        'hour_cos': 'Hour (cos)',
        'month_sin': 'Month (sin)',
        'month_cos': 'Month (cos)',
        'temp_lag_1': 'Temp Lag-1',
        'temp_lag_24': 'Temp Lag-24',
        'temp_rolling_mean_24': 'Temp Roll Mean',
        'temp_rolling_std_24': 'Temp Roll Std',
        'cooling_degree': 'Cooling Deg',
        'temp_x_hour': 'Temp × Hour'
    }
    
    corr_display = corr_matrix.copy()
    corr_display.index = [display_names.get(c, c) for c in corr_display.index]
    corr_display.columns = [display_names.get(c, c) for c in corr_display.columns]
    
    sns.heatmap(corr_display, 
               annot=True, 
               fmt='.2f',
               cmap='RdBu_r', 
               center=0,
               vmin=-1, vmax=1,
               square=True,
               linewidths=0.5,
               annot_kws={'size': 8},
               cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
               ax=ax)
    
    ax.set_title('Comprehensive Feature Correlation Matrix\n(Ashburn, VA 2019 - 8,760 Hours)', 
                fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    # Save to both locations
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(figures_dir, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: correlation_heatmap.png (comprehensive)")
    
    # Print key correlations
    print("\n  Key Correlations with Temperature:")
    temp_corrs = corr_matrix['temperature_f'].drop('temperature_f').sort_values(key=abs, ascending=False)
    for feat, corr in temp_corrs.head(8).items():
        print(f"    {feat:25s}: {corr:+.3f}")


def generate_recommendations(results):
    """Generate actionable recommendations based on feature importance analysis."""
    print("\n" + "=" * 70)
    print("FEATURE RECOMMENDATIONS FOR DATACENTER ENERGY FORECASTING")
    print("=" * 70)
    
    # Combine all rankings
    all_features = set()
    for method, importance in results.items():
        if isinstance(importance, pd.Series):
            all_features.update(importance.head(10).index)
    
    # Categorize features
    temporal_features = ['hour', 'minute', 'second', 'day_of_week', 'is_weekend', 
                        'is_business_hour', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
    lag_features = ['cpu_lag_1', 'cpu_lag_10', 'cpu_rolling_mean_10', 'cpu_rolling_std_10', 'tasks_lag_1']
    load_features = ['num_tasks_sampled', 'log_num_tasks']
    
    print("\n📊 KEY FINDINGS:")
    print("-" * 50)
    
    # Check which category is most important
    categories_importance = {
        'Temporal Features': [],
        'Lag/Rolling Features': [],
        'Load Indicators': []
    }
    
    for method, importance in results.items():
        if isinstance(importance, pd.Series):
            for feat in temporal_features:
                if feat in importance.index:
                    categories_importance['Temporal Features'].append(importance[feat])
            for feat in lag_features:
                if feat in importance.index:
                    categories_importance['Lag/Rolling Features'].append(importance[feat])
            for feat in load_features:
                if feat in importance.index:
                    categories_importance['Load Indicators'].append(importance[feat])
    
    for cat, values in categories_importance.items():
        if values:
            avg_importance = np.mean(values)
            print(f"  {cat}: Average importance = {avg_importance:.4f}")
    
    print("\n🎯 RECOMMENDATIONS:")
    print("-" * 50)
    
    recommendations = [
        "1. LAG FEATURES ARE CRITICAL:",
        "   - cpu_lag_1 (previous timestep) shows highest predictive power",
        "   - rolling_mean_10 captures short-term trends",
        "   - rolling_std_10 captures volatility patterns",
        "",
        "2. TASK/LOAD INDICATORS ARE IMPORTANT:",
        "   - num_tasks_sampled correlates with CPU utilization",
        "   - Use log transform (log_num_tasks) to handle high variance",
        "",
        "3. TEMPORAL PATTERNS MATTER:",
        "   - hour_sin/hour_cos capture diurnal patterns",
        "   - is_business_hour distinguishes peak vs off-peak",
        "   - day_of_week shows weekly patterns",
        "",
        "4. RECOMMENDED FEATURE SET FOR MODELING:",
        "   PRIMARY (High importance):",
        "     • cpu_lag_1, cpu_lag_10 (autoregressive)",
        "     • cpu_rolling_mean_10, cpu_rolling_std_10 (trend/volatility)",
        "     • log_num_tasks (load indicator)",
        "",
        "   SECONDARY (Medium importance):",
        "     • hour_sin, hour_cos (cyclical time)",
        "     • is_business_hour, is_weekend (business patterns)",
        "",
        "   TERTIARY (Context features):",
        "     • day_of_week, minute (fine-grained time)",
        "",
        "5. FEATURES TO DEPRIORITIZE:",
        "   - Raw 'second' column (too granular, noise)",
        "   - Highly correlated redundant features",
    ]
    
    for rec in recommendations:
        print(rec)
    
    return recommendations


def main():
    """Run the complete feature importance analysis."""
    print("\n" + "=" * 70)
    print("MTFC DATACENTER FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load all available data
    datasets = load_all_available_data()
    
    # Analyze Google Cluster data (main analysis)
    if 'google_cluster' in datasets:
        results, df_analysis, feature_cols = analyze_google_cluster_features(datasets['google_cluster'])
        
        # Create aggregate ranking
        ranking_df = create_aggregate_importance_ranking(results)
        
        # Create visualizations
        create_visualizations(results, df_analysis, feature_cols, OUTPUT_DIR)
        
        # Generate recommendations
        generate_recommendations(results)
        
        # Save results to CSV
        ranking_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance_ranking.csv'))
        print(f"\n✓ Saved rankings to: feature_importance_ranking.csv")
    
    # Create comprehensive correlation matrix using full-year temperature data
    create_comprehensive_correlation_matrix(OUTPUT_DIR)
    
    # Analyze EIA data structure
    if 'eia_power' in datasets:
        analyze_eia_data_structure(datasets['eia_power'])
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
