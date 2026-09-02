"""Stratified 70/15/15 split and train-only MinMax scaling."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

log = logging.getLogger(__name__)


def stratified_split(
    df: pd.DataFrame,
    train: float,
    val: float,
    test: float,
    stratify_col: str,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Two-stage stratified split. Stratifying on the 6-way label keeps the
    53-row SQL Injection class present in all three partitions."""
    total = train + val + test
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")

    strata = df[stratify_col]
    counts = strata.value_counts()
    # A three-way stratified split needs >=1 row in train and >=1 in each of
    # val/test, i.e. the held-out remainder must hold at least two rows.
    min_rows = int(np.ceil(2 / (val + test)))
    if counts.min() < min_rows:
        offenders = counts[counts < min_rows].to_dict()
        raise ValueError(
            f"a {train}/{val}/{test} split needs at least {min_rows} rows per class; "
            f"too few in: {offenders}. Merge the class, drop it, or stratify on the "
            "binary label instead."
        )

    df_train, df_rest = train_test_split(
        df, train_size=train, stratify=strata, random_state=seed, shuffle=True
    )
    # val vs test within the held-out remainder
    rel_val = val / (val + test)
    df_val, df_test = train_test_split(
        df_rest,
        train_size=rel_val,
        stratify=df_rest[stratify_col],
        random_state=seed,
        shuffle=True,
    )

    parts = {
        "train": df_train.reset_index(drop=True),
        "val": df_val.reset_index(drop=True),
        "test": df_test.reset_index(drop=True),
    }
    for name, part in parts.items():
        log.info("%s: %d rows (%.2f%%)", name, len(part), 100 * len(part) / len(df))
    return parts


def fit_scaler(
    train_df: pd.DataFrame, feature_cols: list[str], feature_range: tuple[float, float]
) -> MinMaxScaler:
    """Fit on train ONLY. Fitting on the full frame is the single most common
    source of inflated numbers in this literature."""
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit(train_df[feature_cols].to_numpy(dtype=np.float64, copy=False))
    return scaler


def transform(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler: MinMaxScaler,
    clip: bool,
    feature_range: tuple[float, float],
) -> tuple[np.ndarray, dict]:
    """Scale and optionally clip back into range.

    val/test can hold values beyond the train min/max, which would land outside
    [0,1]; ART's attacks are configured with ``clip_values=(0,1)`` and expect the
    input domain to honour that, so clip and report how much was clipped.
    """
    X = scaler.transform(df[feature_cols].to_numpy(dtype=np.float64, copy=False))
    lo, hi = feature_range
    out_of_range = int(((X < lo) | (X > hi)).sum())
    if clip:
        X = np.clip(X, lo, hi)
    stats = {
        "cells": int(X.size),
        "cells_out_of_range_before_clip": out_of_range,
        "fraction_clipped": (out_of_range / X.size) if X.size else 0.0,
    }
    return X.astype(np.float32), stats
