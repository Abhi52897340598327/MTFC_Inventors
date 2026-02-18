"""
MTFC — LSTM Model
==================
Two-layer LSTM neural network for hourly power time-series forecasting.
Uses TensorFlow / Keras with EarlyStopping and learning-rate reduction.
Target is MinMax-scaled for improved convergence.
"""

import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF info logs

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler

import config as cfg
from utils import log, calc_metrics, save_pickle

# Reproducibility
tf.random.set_seed(cfg.RANDOM_SEED)


def build_lstm(n_features: int) -> Sequential:
    """Construct the 2-layer LSTM architecture per the modeling plan."""
    model = Sequential([
        LSTM(cfg.LSTM_UNITS_1, return_sequences=True,
             input_shape=(cfg.LSTM_LOOKBACK, n_features)),
        Dropout(cfg.LSTM_DROPOUT),

        LSTM(cfg.LSTM_UNITS_2, return_sequences=False),
        Dropout(cfg.LSTM_DROPOUT),

        Dense(cfg.LSTM_DENSE_UNITS, activation="relu"),
        Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LSTM_LR),
        loss="mse",
        metrics=["mae"],
    )
    log.info(f"LSTM built — {model.count_params():,} parameters")
    return model


def train_lstm(X_train: np.ndarray, y_train: np.ndarray,
               X_val: np.ndarray, y_val: np.ndarray) -> tuple:
    """
    Train the LSTM model with early stopping.
    Target is MinMax-scaled internally for better convergence.

    Returns
    -------
    model, target_scaler  : trained Keras model + target scaler for inverse
    """
    n_features = X_train.shape[2]
    model = build_lstm(n_features)

    # Scale the target to [0, 1] for better LSTM convergence
    target_scaler = MinMaxScaler()
    y_train_sc = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_sc = target_scaler.transform(y_val.reshape(-1, 1)).ravel()

    # Save scaler for evaluation
    save_pickle(target_scaler, "lstm_target_scaler")

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=cfg.LSTM_PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=1,
        ),
    ]

    log.info(f"Training LSTM — {cfg.LSTM_EPOCHS} max epochs, "
             f"batch_size={cfg.LSTM_BATCH_SIZE}, lookback={cfg.LSTM_LOOKBACK}")

    history = model.fit(
        X_train, y_train_sc,
        validation_data=(X_val, y_val_sc),
        epochs=cfg.LSTM_EPOCHS,
        batch_size=cfg.LSTM_BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Save model
    model_path = os.path.join(cfg.MODEL_DIR, "lstm_model.keras")
    model.save(model_path)
    log.info(f"LSTM model saved → {model_path}")

    return model, target_scaler


def predict_lstm(model, X_seq: np.ndarray,
                 target_scaler: MinMaxScaler) -> np.ndarray:
    """Run inference and inverse-scale back to MW."""
    pred_scaled = model.predict(X_seq, verbose=0).ravel()
    pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    return pred


def evaluate_lstm(model, X_test_seq: np.ndarray,
                  y_test_seq: np.ndarray,
                  target_scaler: MinMaxScaler) -> dict:
    """Evaluate LSTM on the test set (in original MW scale)."""
    y_pred = predict_lstm(model, X_test_seq, target_scaler)
    metrics = calc_metrics(y_test_seq, y_pred)
    log.info(f"LSTM test metrics: {metrics}")
    return {"y_pred": y_pred, "y_true": y_test_seq, "metrics": metrics}
