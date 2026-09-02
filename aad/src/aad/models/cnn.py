"""1D-CNN baseline: Conv1D 64x3 -> MaxPool -> Conv1D 128x3 -> GlobalMaxPool -> Dense 64.

Reshapes (n_features,) -> (n_features, 1) inside the graph so the model's input
space matches the MLP's. Convolving across feature index has no physical meaning
on tabular flow records, but it is what the paper does, and it is the model that
survives adversarial perturbation best (99.92% -> 49.76%).
"""

from __future__ import annotations

from tensorflow import keras

from .base import N_CLASSES, OUTPUT_ACTIVATION


def build(n_features: int, conv_blocks: list[dict], dense_units: int, dropout: float) -> keras.Model:
    inputs = keras.Input(shape=(n_features,), name="features")
    x = keras.layers.Reshape((n_features, 1), name="to_sequence")(inputs)

    for i, block in enumerate(conv_blocks):
        x = keras.layers.Conv1D(
            filters=int(block["filters"]),
            kernel_size=int(block["kernel_size"]),
            activation="relu",
            padding="same",
            name=f"conv_{i}",
        )(x)
        if block.get("pool"):
            x = keras.layers.MaxPooling1D(pool_size=int(block["pool"]), name=f"pool_{i}")(x)

    x = keras.layers.GlobalMaxPooling1D(name="global_pool")(x)
    x = keras.layers.Dense(dense_units, activation="relu", name="dense")(x)
    if dropout:
        x = keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = keras.layers.Dense(N_CLASSES, activation=OUTPUT_ACTIVATION, name="softmax")(x)
    return keras.Model(inputs, outputs, name="cnn")
