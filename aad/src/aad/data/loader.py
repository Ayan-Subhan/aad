"""Chunked CSV ingestion for CSE-CIC-IDS2018.

The two capture files are ~350 MB each and carry three defects that break a naive
``pd.read_csv``:

1. ``02-16`` repeats its header row partway through the file, which forces every
   column in that chunk to object dtype.
2. ``Flow Byts/s`` / ``Flow Pkts/s`` hold the literal string ``Infinity`` (and
   blanks) wherever ``Flow Duration`` is zero.
3. Label strings are inconsistently spaced across files.

This module reads in chunks, drops the embedded headers, coerces every feature to
float32, and returns one frame per file with the raw label kept as a category.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _coerce_features(chunk: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Force every feature column to float32, turning junk into NaN."""
    for col in feature_cols:
        if chunk[col].dtype != np.float32:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype(np.float32)
    return chunk


def load_file(
    path: Path,
    label_column: str = "Label",
    drop_columns: tuple[str, ...] = ("Timestamp",),
    chunksize: int = 200_000,
) -> tuple[pd.DataFrame, dict]:
    """Read one capture file into a float32 frame plus an ingestion report."""
    frames: list[pd.DataFrame] = []
    rows_read = 0
    embedded_headers = 0
    feature_cols: list[str] | None = None

    reader = pd.read_csv(path, chunksize=chunksize, low_memory=False)
    for chunk in reader:
        rows_read += len(chunk)

        # Defect 1: the repeated header row shows up as a data row whose Label
        # literally reads "Label". Label is always object dtype, so no cast needed.
        header_rows = chunk[label_column].astype(str).str.strip() == label_column
        if header_rows.any():
            embedded_headers += int(header_rows.sum())
            chunk = chunk.loc[~header_rows]

        chunk = chunk.drop(columns=[c for c in drop_columns if c in chunk.columns])

        if feature_cols is None:
            feature_cols = [c for c in chunk.columns if c != label_column]

        # Defect 2: "Infinity"/"" become inf/NaN here rather than silently object.
        chunk = _coerce_features(chunk, feature_cols)
        chunk[label_column] = chunk[label_column].astype(str).str.strip().astype("category")

        frames.append(chunk)

    df = pd.concat(frames, ignore_index=True, copy=False)
    del frames

    report = {
        "file": path.name,
        "rows_read": rows_read,
        "embedded_header_rows": embedded_headers,
        "rows_kept": len(df),
        "n_feature_columns": len(feature_cols or []),
    }
    log.info(
        "%s: %d rows read, %d embedded header row(s) removed, %d kept",
        path.name,
        rows_read,
        embedded_headers,
        len(df),
    )
    return df, report


def load_all(
    raw_dir: Path,
    filenames: list[str],
    label_column: str = "Label",
    drop_columns: tuple[str, ...] = ("Timestamp",),
    chunksize: int = 200_000,
) -> tuple[pd.DataFrame, list[dict]]:
    """Read every capture file and concatenate, verifying schemas agree."""
    frames, reports = [], []
    schema: list[str] | None = None

    for name in filenames:
        path = Path(raw_dir) / name
        if not path.exists():
            raise FileNotFoundError(f"expected raw capture file at {path}")

        df, report = load_file(path, label_column, drop_columns, chunksize)

        if schema is None:
            schema = list(df.columns)
        elif list(df.columns) != schema:
            missing = set(schema) ^ set(df.columns)
            raise ValueError(f"{name} schema differs from the first file; symmetric diff: {missing}")

        df["source_file"] = name
        frames.append(df)
        reports.append(report)

    combined = pd.concat(frames, ignore_index=True, copy=False)
    del frames
    log.info("combined: %d rows x %d columns", len(combined), combined.shape[1])
    return combined, reports
