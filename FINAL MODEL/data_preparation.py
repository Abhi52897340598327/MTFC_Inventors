"""
MTFC Virginia Datacenter Energy Forecasting — Data Preparation
================================================================
Train / Validation / Test splitting, scaling, LSTM sequence creation,
and final data-quality validation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import config as cfg
from utils import log, save_pickle


# ── Splitting ───────────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame):
    """
    Chronological 70 / 15 / 15 split.

    Returns
    -------
    train, val, test : pd.DataFrame
    """
    n = len(df)
    t1 = int(n * cfg.TRAIN_RATIO)
    t2 = t1 + int(n * cfg.VAL_RATIO)

    train = df.iloc[:t1].copy()
    val   = df.iloc[t1:t2].copy()
    test  = df.iloc[t2:].copy()

    log.info(f"Split — Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


# ── Scaling ─────────────────────────────────────────────────────────────────

def fit_scaler(train: pd.DataFrame, feature_cols: list):
    """
    Fit StandardScaler on training data and return scaler + column list.
    Saves the scaler to disk for later inverse-transform.
    """
    scaler = StandardScaler()
    scaler.fit(train[feature_cols])
    save_pickle({"scaler": scaler, "feature_cols": feature_cols}, "scaler")
    log.info(f"Scaler fitted on {len(feature_cols)} features")
    return scaler


def scale_features(df: pd.DataFrame, scaler: StandardScaler,
                   feature_cols: list) -> np.ndarray:
    """Apply scaler.transform and return ndarray."""
    return scaler.transform(df[feature_cols])


# ── LSTM Sequences ──────────────────────────────────────────────────────────

def create_sequences(X: np.ndarray, y: np.ndarray, lookback: int = cfg.LSTM_LOOKBACK):
    """
    Build overlapping (X_seq, y_seq) arrays for LSTM.

    Parameters
    ----------
    X : (n_samples, n_features) scaled feature array
    y : (n_samples,) target array
    lookback : number of past steps to look back

    Returns
    -------
    X_seq : (n_sequences, lookback, n_features)
    y_seq : (n_sequences,)
    """
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i - lookback:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


# ── Feature column selectors ───────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame) -> list:
    """Return all feature columns (everything except target and timestamp)."""
    exclude = {cfg.TARGET_COL, "timestamp"}
    return [c for c in df.columns
            if c not in exclude and df[c].dtype in ("float64", "int64", "int32", "float32")]


def get_xgb_feature_cols(df: pd.DataFrame) -> list:
    """Feature columns suitable for XGBoost (includes lags, rolling, etc.)."""
    return get_feature_cols(df)


def get_forecast_feature_cols(df: pd.DataFrame) -> list:
    """Feature columns safe for multi-year forecasting (NO target-derived lags/rolling).

    These features can be constructed for future years without knowing future
    target values.  This is the feature set used for the forecast XGBoost.
    """
    exclude_prefixes = ("power_lag", "power_rolling")
    return [c for c in get_feature_cols(df)
            if not any(c.startswith(p) for p in exclude_prefixes)]


def get_sarimax_target_and_exog(df: pd.DataFrame):
    """Return (target_series, exog_df) for SARIMAX."""
    target = df[cfg.TARGET_COL]
    exog_cols = [c for c in cfg.SARIMAX_EXOG_COLS if c in df.columns]
    exog = df[exog_cols] if exog_cols else None
    return target, exog


# ── Convenience: full preparation pipeline ──────────────────────────────────

def prepare_all(df: pd.DataFrame):
    """
    Run splitting, scaling, and sequence creation.

    Returns
    -------
    dict with keys:
        train, val, test : raw DataFrames
        X_train, X_val, X_test : scaled ndarrays
        y_train, y_val, y_test : target ndarrays
        X_train_seq, y_train_seq, X_val_seq, y_val_seq,
        X_test_seq, y_test_seq : LSTM-ready sequences
        scaler : fitted StandardScaler
        feature_cols : list of feature names
    """
    train, val, test = split_data(df)

    feature_cols = get_feature_cols(df)
    scaler = fit_scaler(train, feature_cols)

    X_train = scale_features(train, scaler, feature_cols)
    X_val   = scale_features(val, scaler, feature_cols)
    X_test  = scale_features(test, scaler, feature_cols)

    y_train = train[cfg.TARGET_COL].values
    y_val   = val[cfg.TARGET_COL].values
    y_test  = test[cfg.TARGET_COL].values

    # LSTM sequences
    X_train_seq, y_train_seq = create_sequences(X_train, y_train)
    X_val_seq,   y_val_seq   = create_sequences(X_val,   y_val)
    X_test_seq,  y_test_seq  = create_sequences(X_test,  y_test)

    log.info(f"LSTM sequences — Train: {X_train_seq.shape}, "
             f"Val: {X_val_seq.shape}, Test: {X_test_seq.shape}")

    return {
        "train": train, "val": val, "test": test,
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "X_train_seq": X_train_seq, "y_train_seq": y_train_seq,
        "X_val_seq": X_val_seq, "y_val_seq": y_val_seq,
        "X_test_seq": X_test_seq, "y_test_seq": y_test_seq,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }
