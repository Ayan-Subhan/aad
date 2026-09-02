"""Phase 2 - train the three clean baseline NIDS models.

    python scripts/02_train_baselines.py                    # all three
    python scripts/02_train_baselines.py --models mlp       # just one
    python scripts/02_train_baselines.py --data-dir data/processed_paperparity --tag paperparity

Writes:
    artifacts/models/{mlp,cnn,lstm}.keras
    artifacts/models/{mlp,cnn,lstm}_history.json
    artifacts/reports/baseline_metrics.json     full metrics incl. per-attack-type recall
    artifacts/reports/table3_nids.{csv,md}      Table III
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # mute TF's startup banner

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorflow import keras  # noqa: E402

from aad.evaluate import HEADER, binary_metrics, format_row, per_type_recall  # noqa: E402
from aad.models import cnn as cnn_mod  # noqa: E402
from aad.models import lstm as lstm_mod  # noqa: E402
from aad.models import mlp as mlp_mod  # noqa: E402
from aad.models.base import build_callbacks, compile_model, set_seeds, summarise  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("train")

BUILDERS = {"mlp": mlp_mod.build, "cnn": cnn_mod.build, "lstm": lstm_mod.build}


def load_split(data_dir: Path, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(data_dir / f"{name}.npz")
    return d["X"], d["y"].astype(np.int64), d["y_multiclass"].astype(np.int64)


def one_hot(y: np.ndarray) -> np.ndarray:
    """2-unit softmax targets to pair with CategoricalCrossentropy."""
    return keras.utils.to_categorical(y, num_classes=2).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "models.yaml"))
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--models", nargs="+", default=["mlp", "cnn", "lstm"], choices=list(BUILDERS))
    ap.add_argument("--tag", default=None, help="suffix so variant runs do not overwrite each other")
    ap.add_argument("--epochs", type=int, default=None, help="override config, for a quick check")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    tcfg = cfg["training"]
    seed = int(cfg["seed"])
    tag = f"_{args.tag}" if args.tag else ""

    data_dir = ROOT / args.data_dir
    models_dir = ROOT / cfg["output"]["models_dir"]
    reports_dir = ROOT / cfg["output"]["reports_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, _ = load_split(data_dir, "train")
    X_val, y_val, _ = load_split(data_dir, "val")
    X_test, y_test, y_test_multi = load_split(data_dir, "test")
    n_features = X_train.shape[1]
    feature_names = json.loads((data_dir / "feature_names.json").read_text(encoding="utf-8"))
    assert len(feature_names) == n_features, "feature_names.json disagrees with train.npz"

    log.info(
        "data %s | train %s  val %s  test %s | attack share %.3f%%",
        args.data_dir, X_train.shape, X_val.shape, X_test.shape, 100 * y_train.mean(),
    )

    Y_train, Y_val = one_hot(y_train), one_hot(y_val)

    class_weight = None
    if tcfg["class_weight"]:
        n0, n1 = int((y_train == 0).sum()), int((y_train == 1).sum())
        class_weight = {0: len(y_train) / (2 * n0), 1: len(y_train) / (2 * n1)}
        log.info("class_weight enabled: %s", class_weight)

    epochs = args.epochs or int(tcfg["epochs"])
    results: dict[str, dict] = {}

    for name in args.models:
        log.info("=" * 70)
        log.info("training %s", name.upper())
        set_seeds(seed)

        model = BUILDERS[name](n_features=n_features, **cfg["models"][name])
        compile_model(model, tcfg["optimizer"], float(tcfg["learning_rate"]))
        log.info("\n%s", summarise(model))

        t0 = time.time()
        history = model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=epochs,
            batch_size=int(tcfg["batch_size"]),
            callbacks=build_callbacks(tcfg),
            class_weight=class_weight,
            verbose=2,
        )
        train_seconds = round(time.time() - t0, 1)

        prob = model.predict(X_test, batch_size=4096, verbose=0)
        y_pred = prob.argmax(axis=1)

        m = binary_metrics(y_test, y_pred, y_prob=prob)
        m["per_attack_type"] = per_type_recall(y_test_multi, y_pred)
        m["train_seconds"] = train_seconds
        m["epochs_run"] = len(history.history["loss"])
        m["params"] = int(model.count_params())
        results[name] = m

        model.save(models_dir / f"{name}{tag}.keras")
        (models_dir / f"{name}{tag}_history.json").write_text(
            json.dumps({k: [float(v) for v in vals] for k, vals in history.history.items()}, indent=2),
            encoding="utf-8",
        )
        np.savez_compressed(models_dir / f"{name}{tag}_test_probs.npz", prob=prob.astype(np.float32))

        log.info(
            "%s: acc %.4f%%  F1 %.4f%%  FPR %.4f%%  loss %.5f  (%ds, %d epochs)",
            name, m["accuracy"] * 100, m["f1"] * 100, m["fpr"] * 100, m["loss"],
            train_seconds, m["epochs_run"],
        )

    # ---- Table III ---------------------------------------------------------
    rows = []
    for name, m in results.items():
        rows.append({
            "model": name.upper(),
            "accuracy_%": round(m["accuracy"] * 100, 4),
            "precision_%": round(m["precision"] * 100, 4),
            "recall_%": round(m["recall"] * 100, 4),
            "f1_%": round(m["f1"] * 100, 4),
            "val_loss": round(m["loss"], 5),
            "fpr_%": round(m["fpr"] * 100, 4),
            "fn": m["fn"], "fp": m["fp"],
            "params": m["params"],
            "epochs": m["epochs_run"],
            "train_s": m["train_seconds"],
        })
    table = pd.DataFrame(rows)
    table.to_csv(reports_dir / f"table3_nids{tag}.csv", index=False)

    md = [
        "# Table III - NIDS accuracy and loss (clean test set)",
        "",
        f"Data: `{args.data_dir}` | test rows: {len(y_test):,} "
        f"({int(y_test.sum()):,} attack, {int((y_test == 0).sum()):,} benign)",
        f"Seed {seed}, batch {tcfg['batch_size']}, Adam lr={tcfg['learning_rate']}, "
        f"early stopping on val_loss (patience {tcfg['early_stopping']['patience']}).",
        "",
        table.to_markdown(index=False),
        "",
        "## Detection rate per attack type",
        "",
    ]
    type_rows = []
    any_model = next(iter(results))
    for tname, stats in results[any_model]["per_attack_type"].items():
        row = {"class": tname, "support": stats["support"]}
        for name, m in results.items():
            row[name.upper()] = round(m["per_attack_type"][tname]["correct_rate"] * 100, 3)
        type_rows.append(row)
    type_table = pd.DataFrame(type_rows)
    md += [type_table.to_markdown(index=False), ""]
    (reports_dir / f"table3_nids{tag}.md").write_text("\n".join(md), encoding="utf-8")
    type_table.to_csv(reports_dir / f"per_type_recall{tag}.csv", index=False)

    (reports_dir / f"baseline_metrics{tag}.json").write_text(
        json.dumps({"data_dir": args.data_dir, "seed": seed, "results": results}, indent=2),
        encoding="utf-8",
    )

    print("\n" + HEADER)
    print("-" * len(HEADER))
    for name, m in results.items():
        print(format_row(name.upper(), m))
    print("\nper-attack-type detection rate (%):")
    print(type_table.to_string(index=False))

    below = [n for n, m in results.items() if m["accuracy"] < 0.995]
    if below:
        log.warning("below the 99.5%% target: %s", below)
    else:
        log.info("all models cleared the 99.5%% target")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
