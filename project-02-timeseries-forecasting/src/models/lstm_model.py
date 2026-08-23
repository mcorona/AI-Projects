"""
LSTM model for one-step-ahead price forecasting.

Author: Manuel Corona
"""

from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras


def create_sequences(values: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding windows: X[i] = values[i:i+lookback], y[i] = values[i+lookback].

    Using the true historical values for every window (never the model's
    own prior predictions) makes this a one-step-ahead task, consistent
    with how ARIMA and the naive baselines are evaluated.
    """
    X, y = [], []
    for i in range(len(values) - lookback):
        X.append(values[i : i + lookback])
        y.append(values[i + lookback])
    return np.array(X), np.array(y)


def build_model(lookback: int, units: int = 32) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(lookback, 1)),
        keras.layers.LSTM(units),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def train_lstm(
    train_series: pd.Series,
    val_series: pd.Series,
    lookback: int = 30,
    units: int = 32,
    epochs: int = 50,
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple[keras.Model, MinMaxScaler]:
    """
    Scale (fit on train only -- no leakage), build sequences using a
    continuous train+val window so the first val predictions have real
    lookback context, and train with early stopping on val loss.

    Returns:
        (trained model, fitted scaler)
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_series.values.reshape(-1, 1)).flatten()

    full_series = pd.concat([train_series, val_series])
    full_scaled = scaler.transform(full_series.values.reshape(-1, 1)).flatten()

    X_train, y_train = create_sequences(train_scaled, lookback)

    # Val sequences: windows whose target falls in val_series, using true
    # lookback context that may span the train/val boundary.
    val_start = len(train_series)
    X_full, y_full = create_sequences(full_scaled, lookback)
    val_target_positions = np.arange(val_start - lookback, len(full_scaled) - lookback)
    val_target_positions = val_target_positions[val_target_positions >= 0]
    X_val, y_val = X_full[val_target_positions], y_full[val_target_positions]

    model = build_model(lookback, units)
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(
        X_train[..., np.newaxis], y_train,
        validation_data=(X_val[..., np.newaxis], y_val),
        epochs=epochs, batch_size=batch_size, callbacks=[early_stop], verbose=0,
    )
    return model, scaler


def forecast_one_step(
    model: keras.Model, scaler: MinMaxScaler, train_series: pd.Series, eval_series: pd.Series, lookback: int
) -> pd.Series:
    """
    One-step-ahead predictions for every point in eval_series, using true
    historical values (never the model's own predictions) as lookback context.
    """
    full_series = pd.concat([train_series, eval_series])
    full_scaled = scaler.transform(full_series.values.reshape(-1, 1)).flatten()

    eval_start = len(train_series)
    X, _ = create_sequences(full_scaled, lookback)
    target_positions = np.arange(eval_start - lookback, len(full_scaled) - lookback)
    target_positions = target_positions[target_positions >= 0]
    X_eval = X[target_positions][..., np.newaxis]

    preds_scaled = model.predict(X_eval, verbose=0).flatten()
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    return pd.Series(preds, index=eval_series.index)
