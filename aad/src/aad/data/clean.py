"""Row-level cleaning and label normalisation."""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_SLUG = re.compile(r"[^a-z0-9]+")


def slugify_label(raw: str) -> str:
    """'Brute Force -Web' -> 'bruteforceweb'. Absorbs every spacing variant."""
    return _SLUG.sub("", str(raw).strip().lower())


def normalise_labels(
    df: pd.DataFrame,
    label_column: str,
    label_map: dict[str, dict],
) -> tuple[pd.DataFrame, dict]:
    """Map raw label strings onto canonical name / multiclass code / binary target."""
    raw = df[label_column].astype(str)
    slugs = raw.map(slugify_label)

    unknown = sorted(set(slugs.unique()) - set(label_map))
    if unknown:
        raise ValueError(
            f"unmapped label slug(s) {unknown}; add them to config/data.yaml:label_map"
        )

    df = df.drop(columns=[label_column])
    df["label_name"] = slugs.map(lambda s: label_map[s]["name"]).astype("category")
    df["label_multiclass"] = slugs.map(lambda s: label_map[s]["code"]).astype(np.int8)
    df["label_binary"] = slugs.map(lambda s: label_map[s]["binary"]).astype(np.int8)

    observed = {label_map[s]["name"]: int((slugs == s).sum()) for s in sorted(set(slugs))}
    log.info("label distribution after normalisation: %s", observed)
    return df, observed


def clean_rows(
    df: pd.DataFrame,
    feature_cols: list[str],
    replace_inf_with_nan: bool = True,
    drop_nan_rows: bool = True,
    drop_duplicates: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Resolve inf/NaN and exact duplicates, reporting the cost of each step."""
    report: dict = {"rows_in": len(df)}

    if replace_inf_with_nan:
        block = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
        inf_mask = ~np.isfinite(block) & ~np.isnan(block)
        n_inf = int(inf_mask.sum())
        if n_inf:
            per_col = pd.Series(inf_mask.sum(axis=0), index=feature_cols)
            report["inf_cells"] = n_inf
            report["inf_by_column"] = per_col[per_col > 0].astype(int).to_dict()
            df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
            log.info("replaced %d non-finite cell(s) with NaN: %s", n_inf, report["inf_by_column"])
        else:
            report["inf_cells"] = 0
            report["inf_by_column"] = {}

    nan_by_col = df[feature_cols].isna().sum()
    report["nan_cells"] = int(nan_by_col.sum())
    report["nan_by_column"] = nan_by_col[nan_by_col > 0].astype(int).to_dict()

    if drop_nan_rows:
        before = len(df)
        nan_rows = df[feature_cols].isna().any(axis=1)
        report["rows_dropped_nan"] = int(nan_rows.sum())
        report["nan_rows_by_class"] = (
            df.loc[nan_rows, "label_name"].value_counts().astype(int).to_dict()
            if nan_rows.any()
            else {}
        )
        df = df.loc[~nan_rows]
        log.info("dropped %d row(s) containing NaN (%d -> %d)", before - len(df), before, len(df))

    if drop_duplicates:
        before = len(df)
        dupe_subset = feature_cols + ["label_multiclass"]
        dupes = df.duplicated(subset=dupe_subset, keep="first")
        report["rows_dropped_duplicate"] = int(dupes.sum())
        report["duplicate_rows_by_class"] = (
            df.loc[dupes, "label_name"].value_counts().astype(int).to_dict()
            if dupes.any()
            else {}
        )
        df = df.loc[~dupes]
        log.info(
            "dropped %d exact duplicate row(s) (%d -> %d)", before - len(df), before, len(df)
        )

    df = df.reset_index(drop=True)
    report["rows_out"] = len(df)
    return df, report


def find_constant_columns(
    df: pd.DataFrame, feature_cols: list[str], tol: float = 0.0
) -> list[str]:
    """Columns with no variance carry no signal and no usable gradient.

    Called on the *training split only* so the feature set is not chosen with any
    knowledge of val/test.
    """
    block = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    spread = block.max(axis=0) - block.min(axis=0)
    constant = [c for c, s in zip(feature_cols, spread) if s <= tol]
    log.info("%d zero-variance feature column(s) on train: %s", len(constant), constant)
    return constant
