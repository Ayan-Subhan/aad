"""LSTM baseline: LSTM 64 -> Dense 32 -> softmax 2.

Treats the n_features columns as a length-n_features pseudo-sequence, as the
paper does. This is not a real temporal sequence -- the columns are independent
flow statistics in arbitrary order -- and that is very likely why the paper's
LSTM collapses hardest under attack (99.80% -> 4.89%). Kept for fidelity; a
genuine per-source-IP windowed sequence is the phase 8 extension.
"""

from __future__ import annotations

from tensorflow import keras

from .base import N_CLASSES, OUTPUT_ACTIVATION


def build(n_features: int, units: int, dense_units: int, dropout: float) -> keras.Model:
    inputs = keras.Input(shape=(n_features,), name="features")
    x = keras.layers.Reshape((n_features, 1), name="to_sequence")(inputs)
    x = keras.layers.LSTM(units, name="lstm")(x)
    x = keras.layers.Dense(dense_units, activation="relu", name="dense")(x)
    if dropout:
        x = keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = keras.layers.Dense(N_CLASSES, activation=OUTPUT_ACTIVATION, name="softmax")(x)
    return keras.Model(inputs, outputs, name="lstm")
