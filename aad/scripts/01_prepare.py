"""Phase 1 - build the processed dataset for every downstream phase.

    python scripts/01_prepare.py                 # full run
    python scripts/01_prepare.py --sample-frac .02   # ~2 min smoke test

Writes:
    data/interim/combined.parquet          cleaned, pre-split, unscaled
    data/processed/{train,val,test}.npz    X float32 in [0,1], y_binary, y_multiclass
    data/processed/scaler.pkl              MinMaxScaler fitted on train only
    data/processed/feature_names.json      the exact retained feature order
    artifacts/reports/class_stats.csv|.md  Table I
    artifacts/reports/prepare_meta.json    every count this run produced
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aad.data.clean import clean_rows, find_constant_columns, normalise_labels  # noqa: E402
from aad.data.loader import load_all  # noqa: E402
from aad.data.split import fit_scaler, stratified_split, transform  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prepare")

LABEL_COLS = ["label_name", "label_multiclass", "label_binary"]
NON_FEATURE_COLS = LABEL_COLS + ["source_file"]


def build_class_table(parts: dict[str, pd.DataFrame], type_names: dict) -> pd.DataFrame:
    """Table I: per-class counts across the three splits."""
    rows = []
    all_df = pd.concat(parts.values(), ignore_index=True)
    for code, group in all_df.groupby("label_multiclass", observed=True):
        name = group["label_name"].iloc[0]
        row = {
            "paper_type": type_names.get(int(code), "-"),
            "class": name,
            "binary": "attack" if int(group["label_binary"].iloc[0]) == 1 else "benign",
            "total": len(group),
        }
        for split, part in parts.items():
            row[split] = int((part["label_multiclass"] == code).sum())
        row["pct_of_total"] = round(100 * len(group) / len(all_df), 4)
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("total", ascending=False).reset_index(drop=True)
    totals = {
        "paper_type": "",
        "class": "ALL",
        "binary": "",
        "total": len(all_df),
        **{s: len(p) for s, p in parts.items()},
        "pct_of_total": 100.0,
    }
    return pd.concat([table, pd.DataFrame([totals])], ignore_index=True)


def write_markdown_table(table: pd.DataFrame, path: Path, meta: dict) -> None:
    lines = [
        "# Table I - Statistical description of the dataset",
        "",
        f"Source: {', '.join(meta['raw_files'])}",
        f"Rows ingested: {meta['rows_ingested']:,} -> retained after cleaning: {meta['rows_after_cleaning']:,}",
        f"Features retained: {meta['n_features']} of {meta['n_features_before_constant_drop']}",
        f"Split: {meta['split_ratios']} stratified on the 6-way label, seed {meta['seed']}",
        "",
        table.to_markdown(index=False),
        "",
        "## Cleaning ledger",
        "",
        f"- Embedded header rows removed: {meta['embedded_header_rows']}",
        f"- Non-finite (Infinity) cells replaced: {meta['clean']['inf_cells']:,} "
        f"in {list(meta['clean']['inf_by_column'])}",
        f"- Rows dropped for NaN: {meta['clean'].get('rows_dropped_nan', 0):,}",
        f"- Exact duplicate rows dropped: {meta['clean'].get('rows_dropped_duplicate', 0):,}",
        f"- Zero-variance columns dropped (detected on train only): {meta['constant_columns']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "data.yaml"))
    ap.add_argument("--sample-frac", type=float, default=None,
                    help="stratified subsample of the cleaned data, for a fast smoke test")
    ap.add_argument("--no-dedup", action="store_true",
                    help="keep exact duplicate rows (paper-parity variant; leaks across splits)")
    ap.add_argument("--out-dir", default=None,
                    help="override output.processed_dir, e.g. data/processed_paperparity")
    ap.add_argument("--tag", default=None,
                    help="suffix for the report filenames so variants do not overwrite each other")
    args = ap.parse_args()

    t0 = time.time()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)

    raw_dir = ROOT / cfg["raw"]["dir"]
    interim_dir = ROOT / cfg["output"]["interim_dir"]
    processed_dir = ROOT / (args.out_dir or cfg["output"]["processed_dir"])
    reports_dir = ROOT / cfg["output"]["reports_dir"]
    for d in (interim_dir, processed_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ---- ingest -----------------------------------------------------------
    log.info("reading %d capture file(s) from %s", len(cfg["raw"]["files"]), raw_dir)
    df, load_reports = load_all(
        raw_dir=raw_dir,
        filenames=cfg["raw"]["files"],
        label_column=cfg["label_column"],
        drop_columns=tuple(cfg["drop_columns"]),
        chunksize=int(cfg["raw"]["chunksize"]),
    )
    rows_ingested = sum(r["rows_read"] for r in load_reports)
    embedded_headers = sum(r["embedded_header_rows"] for r in load_reports)

    # ---- labels -----------------------------------------------------------
    df, raw_label_counts = normalise_labels(df, cfg["label_column"], cfg["label_map"])
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    log.info("%d candidate feature columns", len(feature_cols))

    # ---- clean ------------------------------------------------------------
    df, clean_report = clean_rows(
        df,
        feature_cols,
        replace_inf_with_nan=cfg["clean"]["replace_inf_with_nan"],
        drop_nan_rows=cfg["clean"]["drop_nan_rows"],
        drop_duplicates=cfg["clean"]["drop_duplicates"] and not args.no_dedup,
    )

    if args.sample_frac:
        n_before = len(df)
        df = (
            df.groupby("label_multiclass", group_keys=False, observed=True)
            # floor of 10 so the rarest class still survives a 70/15/15 split
            .apply(lambda g: g.sample(min(len(g), max(10, int(len(g) * args.sample_frac))),
                                      random_state=seed))
            .reset_index(drop=True)
        )
        log.warning("SMOKE TEST: subsampled %d -> %d rows", n_before, len(df))

    if cfg["output"]["write_interim_parquet"] and not args.sample_frac and not args.no_dedup:
        out = interim_dir / "combined.parquet"
        df.to_parquet(out, index=False, compression="snappy")
        log.info("wrote %s (%.0f MB)", out, out.stat().st_size / 1e6)

    # ---- split ------------------------------------------------------------
    parts = stratified_split(
        df,
        train=cfg["split"]["train"],
        val=cfg["split"]["val"],
        test=cfg["split"]["test"],
        stratify_col="label_multiclass",
        seed=seed,
    )
    del df

    # ---- feature selection on train only ----------------------------------
    constant_cols = find_constant_columns(parts["train"], feature_cols)
    n_features_before = len(feature_cols)
    feature_cols = [c for c in feature_cols if c not in constant_cols]
    log.info("retained %d feature(s)", len(feature_cols))

    # ---- scale ------------------------------------------------------------
    frange = tuple(cfg["scale"]["feature_range"])
    scaler = fit_scaler(parts["train"], feature_cols, frange)

    scale_stats = {}
    for name, part in parts.items():
        X, stats = transform(part, feature_cols, scaler, cfg["scale"]["clip_transformed"], frange)
        scale_stats[name] = stats
        y_bin = part["label_binary"].to_numpy(dtype=np.int8)
        y_multi = part["label_multiclass"].to_numpy(dtype=np.int8)

        assert X.shape == (len(part), len(feature_cols)), f"{name} X shape mismatch"
        assert np.isfinite(X).all(), f"{name} contains non-finite values after scaling"
        assert X.min() >= frange[0] - 1e-6 and X.max() <= frange[1] + 1e-6, f"{name} out of range"

        np.savez_compressed(processed_dir / f"{name}.npz", X=X, y=y_bin, y_multiclass=y_multi)
        log.info(
            "%s.npz: X%s  attack=%.3f%%  clipped=%.4f%%",
            name, X.shape, 100 * y_bin.mean(), 100 * stats["fraction_clipped"],
        )

    joblib.dump(scaler, processed_dir / "scaler.pkl")
    (processed_dir / "feature_names.json").write_text(
        json.dumps(feature_cols, indent=2), encoding="utf-8"
    )

    # ---- Table I and the run ledger ---------------------------------------
    meta = {
        "seed": seed,
        "raw_files": cfg["raw"]["files"],
        "rows_ingested": rows_ingested,
        "embedded_header_rows": embedded_headers,
        "rows_after_cleaning": sum(len(p) for p in parts.values()),
        "raw_label_counts": raw_label_counts,
        "clean": clean_report,
        "n_features_before_constant_drop": n_features_before,
        "n_features": len(feature_cols),
        "constant_columns": constant_cols,
        "split_ratios": f"{cfg['split']['train']}/{cfg['split']['val']}/{cfg['split']['test']}",
        "split_sizes": {k: len(v) for k, v in parts.items()},
        "scaling": {"feature_range": list(frange), "per_split": scale_stats},
        "sample_frac": args.sample_frac,
        "deduplicated": cfg["clean"]["drop_duplicates"] and not args.no_dedup,
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    table = build_class_table(parts, {int(k): v for k, v in cfg["paper_type_names"].items()})
    tag = f"_{args.tag}" if args.tag else ""
    table.to_csv(reports_dir / f"class_stats{tag}.csv", index=False)
    write_markdown_table(table, reports_dir / f"class_stats{tag}.md", meta)
    (reports_dir / f"prepare_meta{tag}.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )

    print("\n" + table.to_string(index=False))
    print(f"\ndone in {meta['elapsed_seconds']}s -> {processed_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
