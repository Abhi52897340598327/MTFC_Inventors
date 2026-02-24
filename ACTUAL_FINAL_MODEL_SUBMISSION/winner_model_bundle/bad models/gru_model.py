"""
MTFC — GRU Model
=================
Two-layer GRU neural network for hourly power time-series forecasting.
GRU (Gated Recurrent Unit) is faster than LSTM with comparable performance.
Uses TensorFlow / Keras with learning-rate reduction.
Target is MinMax-scaled for improved convergence.
Fully deterministic training for reproducibility.
"""

import numpy as np
import os
import random

# Suppress TensorFlow/Abseil verbose logging BEFORE importing TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # FATAL only
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"    # suppress Abseil mutex warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["TF_DETERMINISTIC_OPS"] = "1"  # Enable deterministic operations

import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TF Python logging too
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler

import config as cfg
from utils import log, calc_metrics, save_pickle

# Set all random seeds for full reproducibility
SEED = cfg.RANDOM_SEED
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Ensure deterministic behavior
tf.config.experimental.enable_op_determinism()


def build_gru(n_features: int) -> Sequential:
    """Construct optimized Bidirectional GRU for R² > 0.9."""
    model = Sequential([
        # Bidirectional layer captures both past and future context
        Bidirectional(GRU(cfg.LSTM_UNITS_1, return_sequences=True),
                      input_shape=(cfg.LSTM_LOOKBACK, n_features)),
        Dropout(cfg.LSTM_DROPOUT),

        Bidirectional(GRU(cfg.LSTM_UNITS_2, return_sequences=False)),
        Dropout(cfg.LSTM_DROPOUT),

        Dense(cfg.LSTM_DENSE_UNITS, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LSTM_LR),
        loss="mse",
        metrics=["mae"],
    )
    log.info(f"Bidirectional GRU built — {model.count_params():,} parameters")
    return model


def train_gru(X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> tuple:
    """
    Train the GRU model with learning rate reduction.
    Target is MinMax-scaled internally for better convergence.

    Returns
    -------
    model, target_scaler  : trained Keras model + target scaler for inverse
    """
    n_features = X_train.shape[2]
    model = build_gru(n_features)

    # Scale the target to [0, 1] for better GRU convergence
    target_scaler = MinMaxScaler()
    y_train_sc = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_sc = target_scaler.transform(y_val.reshape(-1, 1)).ravel()

    # Save scaler for evaluation
    save_pickle(target_scaler, "gru_target_scaler")

    callbacks = [
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=1,
        ),
    ]

    log.info(f"Training GRU — {cfg.LSTM_EPOCHS} epochs, "
             f"batch_size={cfg.LSTM_BATCH_SIZE}, lookback={cfg.LSTM_LOOKBACK}")

    history = model.fit(
        X_train, y_train_sc,
        validation_data=(X_val, y_val_sc),
        epochs=cfg.LSTM_EPOCHS,
        batch_size=cfg.LSTM_BATCH_SIZE,
        callbacks=callbacks,
        shuffle=False,  # Disable shuffling for deterministic training
        verbose=1,
    )

    # Save model
    model_path = os.path.join(cfg.MODEL_DIR, "gru_model.keras")
    model.save(model_path)
    log.info(f"GRU model saved → {model_path}")

    return model, target_scaler


def predict_gru(model, X_seq: np.ndarray,
                target_scaler: MinMaxScaler) -> np.ndarray:
    """Run inference and inverse-scale back to MW."""
    pred_scaled = model.predict(X_seq, verbose=0).ravel()
    pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    return pred


def evaluate_gru(model, X_test_seq: np.ndarray,
                 y_test_seq: np.ndarray,
                 target_scaler: MinMaxScaler) -> dict:
    """Evaluate GRU on the test set (in original MW scale)."""
    y_pred = predict_gru(model, X_test_seq, target_scaler)
    metrics = calc_metrics(y_test_seq, y_pred)
    log.info(f"GRU test metrics: {metrics}")
    return {"y_pred": y_pred, "y_true": y_test_seq, "metrics": metrics}
