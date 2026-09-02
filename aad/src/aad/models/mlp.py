"""MLP baseline: Dense 128 -> 64 -> 32 -> softmax 2.

The paper's strongest clean model and its most fragile under attack (99.98% ->
24.95%). Structured tabular data is what a plain MLP is good at.
"""

from __future__ import annotations

from tensorflow import keras

from .base import N_CLASSES, OUTPUT_ACTIVATION


def build(n_features: int, hidden_units: list[int], activation: str, dropout: float) -> keras.Model:
    inputs = keras.Input(shape=(n_features,), name="features")
    x = inputs
    for i, units in enumerate(hidden_units):
        x = keras.layers.Dense(units, activation=activation, name=f"dense_{i}")(x)
        if dropout:
            x = keras.layers.Dropout(dropout, name=f"dropout_{i}")(x)
    outputs = keras.layers.Dense(N_CLASSES, activation=OUTPUT_ACTIVATION, name="softmax")(x)
    return keras.Model(inputs, outputs, name="mlp")
