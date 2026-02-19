"""
Benchmark Alternative Models to Replace SARIMAX
================================================
Tests: Random Forest, LightGBM, CatBoost, Ridge, ElasticNet, SVR, GRU
"""

import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add module path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
import data_loader
import feature_engineering
import data_preparation

def calc_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

def benchmark_model(name, model, X_train, y_train, X_test, y_test):
    """Train and evaluate a model, return metrics and timing."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    start = time.time()
    try:
        model.fit(X_train, y_train)
        train_time = time.time() - start
        
        y_pred = model.predict(X_test)
        metrics = calc_metrics(y_test, y_pred)
        
        print(f"  ✓ Train time: {train_time:.2f}s")
        print(f"  ✓ R²: {metrics['R2']:.4f}")
        print(f"  ✓ MAE: {metrics['MAE']:.2f}")
        print(f"  ✓ MAPE: {metrics['MAPE']:.2f}%")
        
        return {"name": name, "time": train_time, **metrics, "status": "OK"}
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return {"name": name, "time": 0, "R2": -999, "MAE": 999, "MAPE": 999, "status": str(e)}

def main():
    print("="*60)
    print("MODEL BENCHMARK: Finding Best SARIMAX Alternative")
    print("="*60)
    
    # Load and prepare data
    print("\n[1/3] Loading data...")
    data_dict = data_loader.load_all()
    hourly_data = data_dict["hourly"]
    
    print("[2/3] Engineering features...")
    df_feat = feature_engineering.engineer_features(hourly_data.copy())
    
    print("[3/3] Preparing train/test split...")
    train, val, test = data_preparation.split_data(df_feat)
    
    # Combine train+val for simplicity
    train = pd.concat([train, val])
    
    target_col = cfg.TARGET_COL
    feature_cols = [c for c in df_feat.columns if c not in [target_col, 'datetime', 'timestamp']]
    feature_cols = [c for c in feature_cols if c in train.columns and c in test.columns]
    
    X_train = train[feature_cols].values
    y_train = train[target_col].values
    X_test = test[feature_cols].values
    y_test = test[target_col].values
    
    print(f"\nData ready: X_train={X_train.shape}, X_test={X_test.shape}")
    
    results = []
    
    # ─────────────────────────────────────────────────────────────
    # 1. Random Forest
    # ─────────────────────────────────────────────────────────────
    try:
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
        results.append(benchmark_model("Random Forest", rf, X_train, y_train, X_test, y_test))
    except ImportError as e:
        print(f"Random Forest: SKIPPED ({e})")
    
    # ─────────────────────────────────────────────────────────────
    # 2. LightGBM
    # ─────────────────────────────────────────────────────────────
    try:
        import lightgbm as lgb
        lgbm = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, 
                                  random_state=42, verbose=-1)
        results.append(benchmark_model("LightGBM", lgbm, X_train, y_train, X_test, y_test))
    except ImportError:
        print("\nLightGBM: NOT INSTALLED (pip install lightgbm)")
    
    # ─────────────────────────────────────────────────────────────
    # 3. CatBoost
    # ─────────────────────────────────────────────────────────────
    try:
        from catboost import CatBoostRegressor
        cat = CatBoostRegressor(iterations=200, depth=6, learning_rate=0.05, 
                                 random_state=42, verbose=0)
        results.append(benchmark_model("CatBoost", cat, X_train, y_train, X_test, y_test))
    except ImportError:
        print("\nCatBoost: NOT INSTALLED (pip install catboost)")
    
    # ─────────────────────────────────────────────────────────────
    # 4. Ridge Regression
    # ─────────────────────────────────────────────────────────────
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        ridge = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])
        results.append(benchmark_model("Ridge Regression", ridge, X_train, y_train, X_test, y_test))
    except ImportError as e:
        print(f"Ridge: SKIPPED ({e})")
    
    # ─────────────────────────────────────────────────────────────
    # 5. ElasticNet
    # ─────────────────────────────────────────────────────────────
    try:
        from sklearn.linear_model import ElasticNet
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        elastic = Pipeline([('scaler', StandardScaler()), 
                           ('elastic', ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000))])
        results.append(benchmark_model("ElasticNet", elastic, X_train, y_train, X_test, y_test))
    except ImportError as e:
        print(f"ElasticNet: SKIPPED ({e})")
    
    # ─────────────────────────────────────────────────────────────
    # 6. SVR (Support Vector Regression)
    # ─────────────────────────────────────────────────────────────
    try:
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        # SVR is slow, use subset
        n_sample = min(2000, len(X_train))
        idx = np.random.choice(len(X_train), n_sample, replace=False)
        svr = Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='rbf', C=100))])
        results.append(benchmark_model("SVR (sampled)", svr, X_train[idx], y_train[idx], X_test, y_test))
    except ImportError as e:
        print(f"SVR: SKIPPED ({e})")
    
    # ─────────────────────────────────────────────────────────────
    # 7. GRU (Gated Recurrent Unit)
    # ─────────────────────────────────────────────────────────────
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        from sklearn.preprocessing import MinMaxScaler
        
        print(f"\n{'='*60}")
        print("Testing: GRU")
        print(f"{'='*60}")
        
        start = time.time()
        
        # Prepare sequences
        lookback = 24
        scaler_X = StandardScaler()
        scaler_y = MinMaxScaler()
        
        X_train_sc = scaler_X.fit_transform(X_train)
        X_test_sc = scaler_X.transform(X_test)
        y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        def create_sequences(X, y, lookback):
            Xs, ys = [], []
            for i in range(lookback, len(X)):
                Xs.append(X[i-lookback:i])
                ys.append(y[i])
            return np.array(Xs), np.array(ys)
        
        X_train_seq, y_train_seq = create_sequences(X_train_sc, y_train_sc, lookback)
        X_test_seq, y_test_seq = create_sequences(X_test_sc, y_test[lookback:], lookback)
        
        # Build GRU
        model = tf.keras.Sequential([
            tf.keras.layers.GRU(32, return_sequences=True, input_shape=(lookback, X_train.shape[1])),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.GRU(16),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=64, 
                  validation_split=0.1, verbose=0)
        
        y_pred_sc = model.predict(X_test_seq, verbose=0).ravel()
        y_pred = scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()
        
        train_time = time.time() - start
        metrics = calc_metrics(y_test[lookback:], y_pred)
        
        print(f"  ✓ Train time: {train_time:.2f}s")
        print(f"  ✓ R²: {metrics['R2']:.4f}")
        print(f"  ✓ MAE: {metrics['MAE']:.2f}")
        print(f"  ✓ MAPE: {metrics['MAPE']:.2f}%")
        
        results.append({"name": "GRU", "time": train_time, **metrics, "status": "OK"})
        
    except Exception as e:
        print(f"\nGRU: FAILED ({e})")
    
    # ─────────────────────────────────────────────────────────────
    # RESULTS SUMMARY
    # ─────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("BENCHMARK RESULTS - SARIMAX ALTERNATIVES")
    print("="*70)
    
    # Sort by R²
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('R2', ascending=False)
    
    print(f"\n{'Model':<20} {'R²':>10} {'MAE':>10} {'MAPE':>10} {'Time (s)':>10}")
    print("-"*60)
    for _, row in results_df.iterrows():
        print(f"{row['name']:<20} {row['R2']:>10.4f} {row['MAE']:>10.2f} {row['MAPE']:>9.2f}% {row['time']:>10.2f}")
    
    # Recommendation
    best = results_df.iloc[0]
    print(f"\n{'='*70}")
    print(f"✅ RECOMMENDATION: Replace SARIMAX with {best['name']}")
    print(f"   R² = {best['R2']:.4f}, MAE = {best['MAE']:.2f}, Train time = {best['time']:.1f}s")
    print(f"{'='*70}")
    
    return results_df

if __name__ == "__main__":
    main()
