"""Phase 3 - generate the 60,000 adversarial examples.

    python scripts/03_generate_ae.py                                   # all 12 cells + sweep
    python scripts/03_generate_ae.py --models mlp --attacks fgsm       # one cell
    python scripts/03_generate_ae.py --n-samples 50 --no-sweep         # smoke test
    python scripts/03_generate_ae.py --force                           # ignore cached .npz

Writes:
    artifacts/adversarial/{attack}_{model}.npz     12 files, 5,000 rows each
    artifacts/reports/adversarial_metrics.json     every number this run produced
    artifacts/reports/table4_evasion.{csv,md}      Figure 3 as a table
    artifacts/reports/eps_sweep.csv                Figure 4 as a table
    artifacts/reports/figures/fig3_evasion.png
    artifacts/reports/figures/fig4_eps_sweep.png

Resumable by design: a cell whose .npz already exists is skipped unless --force.
PGD-against-LSTM and DeepFool are expensive enough that a crash in the plotting
code must not cost the whole run.
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

import art  # noqa: E402
import tensorflow as tf  # noqa: E402

from aad.attacks.generate import (  # noqa: E402
    ATTACK_NAMES,
    EPS_ATTACKS,
    build_attack,
    perturbation_norms,
    predict_labels,
    run_attack,
    select_attack_rows,
    summarise,
)
from aad.attacks.wrappers import load_wrapped  # noqa: E402
from aad.models.base import set_seeds  # noqa: E402
from aad.plots import figure3_evasion, figure4_eps_sweep  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("attack")


def build_table(results: dict) -> pd.DataFrame:
    """Table IV - one row per (model, attack) cell."""
    rows = []
    for model, per_attack in results.items():
        for attack, m in per_attack.items():
            rows.append({
                "model": model.upper(),
                "attack": attack.upper(),
                "eps": m["params"].get("eps", "-"),
                "clean_acc_%": round(100 * m["accuracy_clean"], 3),
                "adv_acc_%": round(100 * m["accuracy_adv"], 3),
                "evasion_%": round(100 * m["evasion_rate_on_correct"], 3),
                "linf_max": round(m["linf_max"], 5),
                "l2_mean": round(m["l2_mean"], 4),
                "feats_changed": round(m["features_changed_mean"], 1),
                "seconds": round(m["elapsed_seconds"], 1),
            })
    return pd.DataFrame(rows)


def write_markdown(table: pd.DataFrame, sweep: pd.DataFrame, meta: dict, path: Path) -> None:
    md = [
        "# Table IV - evasion of the baseline NIDS under white-box attack",
        "",
        f"Source rows: {meta['n_samples']:,} true-attack flows drawn from "
        f"`{meta['data_dir']}/test.npz` with seed {meta['seed']}. The **same rows** are "
        "used for every cell, so the twelve results are directly comparable.",
        "",
        "`clean_acc` is the model's detection rate on those rows before perturbation; "
        "`adv_acc` is the same rows after. `evasion` is the fraction of the "
        "*correctly-classified* rows that the attack flipped to benign -- the honest "
        "denominator. Precision and FPR are undefined on an all-attack subset and are "
        "deliberately omitted.",
        "",
        table.to_markdown(index=False),
        "",
        "## Paper comparison",
        "",
        "Verma et al. report a single unattributed collapse figure per model. Ours is "
        "reported per attack, which is why there are twelve numbers here and three there.",
        "",
        "| Model | Paper (under attack) | This run (worst attack) |",
        "| --- | ---: | ---: |",
    ]
    paper = {"MLP": 24.95, "CNN": 49.76, "LSTM": 4.89}
    for model in ["MLP", "CNN", "LSTM"]:
        sub = table[table["model"] == model]
        worst = f"{sub['adv_acc_%'].min():.2f}%" if len(sub) else "-"
        md.append(f"| {model} | {paper[model]:.2f}% | {worst} |")

    if len(sweep):
        md += [
            "",
            "## Accuracy vs. perturbation budget",
            "",
            f"Computed on a fixed {meta['sweep_samples']:,}-row subset of the same source "
            "rows. DeepFool is absent because its `epsilon` is an overshoot multiplier, "
            "not an L-inf budget -- sweeping it would not mean anything.",
            "",
            sweep.pivot_table(
                index=["model", "attack"], columns="eps", values="accuracy_adv"
            ).mul(100).round(3).to_markdown(),
            "",
            "Detection rate (%). Accuracy must fall as eps rises; a non-monotone row is "
            "evidence of gradient masking and is itself a reportable finding.",
        ]
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "attacks.yaml"))
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--models", nargs="+", default=["mlp", "cnn", "lstm"])
    ap.add_argument("--attacks", nargs="+", default=list(ATTACK_NAMES), choices=list(ATTACK_NAMES))
    ap.add_argument("--tag", default=None, help="suffix so variant runs do not collide")
    ap.add_argument("--n-samples", type=int, default=None, help="override config, for a smoke test")
    ap.add_argument("--no-sweep", action="store_true", help="skip the eps sweep")
    ap.add_argument("--force", action="store_true", help="regenerate cells that already exist")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg["seed"])
    batch_size = int(cfg["batch_size"])
    n_samples = args.n_samples or int(cfg["n_samples"])
    tag = f"_{args.tag}" if args.tag else ""

    data_dir = ROOT / args.data_dir
    adv_dir = ROOT / cfg["output"]["adversarial_dir"]
    reports_dir = ROOT / cfg["output"]["reports_dir"]
    figures_dir = ROOT / cfg["output"]["figures_dir"]
    models_dir = ROOT / cfg["output"]["models_dir"]
    for d in (adv_dir, reports_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    set_seeds(seed)

    # ---- source rows -------------------------------------------------------
    test = np.load(data_dir / "test.npz")
    X_test, y_test = test["X"].astype(np.float32), test["y"].astype(np.int64)
    y_multi = test["y_multiclass"].astype(np.int8)
    n_features = X_test.shape[1]

    feature_names = json.loads((data_dir / "feature_names.json").read_text(encoding="utf-8"))
    if len(feature_names) != n_features:
        raise ValueError("feature_names.json disagrees with test.npz")

    idx = select_attack_rows(y_test, n_samples, seed)
    X_src, y_src, y_src_multi = X_test[idx], y_test[idx], y_multi[idx]

    # The sweep reuses a prefix of the same rows, so a sweep point and the
    # canonical run differ in eps and nothing else.
    sweep_cfg = cfg.get("sweep", {})
    run_sweep = bool(sweep_cfg.get("enabled", False)) and not args.no_sweep
    n_sweep = min(int(sweep_cfg.get("n_samples", 1000)), n_samples)

    log.info(
        "%d source rows from %s/test.npz | %d features | attack classes present: %s",
        len(idx), args.data_dir, n_features,
        dict(zip(*[a.tolist() for a in np.unique(y_src_multi, return_counts=True)])),
    )

    results: dict[str, dict] = {}
    sweep_rows: list[dict] = []

    for model_name in args.models:
        keras_model, classifier, logit_classifier = load_wrapped(
            models_dir, model_name, n_features, tag
        )
        # DeepFool navigates distances to the decision boundary, which softmax
        # compresses; it runs at reduced strength on probabilities and ART warns
        # about it. Same weights, same predictions, faithful geometry.
        clf_for = lambda a: logit_classifier if a == "deepfool" else classifier  # noqa: E731
        results[model_name] = {}

        # Clean predictions are per-model, not per-attack: compute once.
        pred_clean, _ = predict_labels(keras_model, X_src)
        log.info(
            "%s: clean detection rate on the source rows %.4f%%",
            model_name, 100 * (pred_clean == 1).mean(),
        )

        for attack_name in args.attacks:
            out_npz = adv_dir / f"{attack_name}_{model_name}{tag}.npz"
            params = dict(cfg["attacks"][attack_name])

            if out_npz.exists() and not args.force:
                log.info("%-8s x %-4s  cached, skipping (--force to regenerate)",
                         attack_name, model_name)
                cached = np.load(out_npz, allow_pickle=False)
                results[model_name][attack_name] = json.loads(str(cached["meta"]))["metrics"]
                continue

            log.info("%-8s x %-4s  %s", attack_name, model_name, params)
            attack = build_attack(attack_name, clf_for(attack_name), params, batch_size)
            X_adv, elapsed = run_attack(attack, X_src, y_src)

            pred_adv, prob_adv = predict_labels(keras_model, X_adv)
            linf, l2, n_changed = perturbation_norms(X_src, X_adv)
            m = summarise(y_src, pred_clean, pred_adv, linf, l2, n_changed)
            m["elapsed_seconds"] = elapsed
            m["params"] = params

            # Contract checks. A silent violation here invalidates every
            # downstream phase, so fail loudly instead.
            if X_adv.min() < -1e-6 or X_adv.max() > 1 + 1e-6:
                raise AssertionError(
                    f"{attack_name}/{model_name}: adversarial rows escaped [0,1] "
                    f"({X_adv.min():.4f}, {X_adv.max():.4f}) -- clip_values not applied"
                )
            if attack_name in EPS_ATTACKS and m["linf_max"] > params["eps"] + 1e-5:
                raise AssertionError(
                    f"{attack_name}/{model_name}: L-inf {m['linf_max']:.6f} exceeds "
                    f"the eps budget {params['eps']}"
                )

            meta = {
                "attack": attack_name,
                "model": model_name,
                "params": params,
                "seed": seed,
                "n_samples": len(idx),
                "data_dir": args.data_dir,
                "batch_size": batch_size,
                "versions": {"art": art.__version__, "tensorflow": tf.__version__},
                "metrics": m,
            }
            np.savez_compressed(
                out_npz,
                X_adv=X_adv,
                X_clean=X_src,
                idx=idx,
                y_true=y_src.astype(np.int8),
                y_multiclass=y_src_multi,
                pred_clean=pred_clean,
                pred_adv=pred_adv,
                prob_adv=prob_adv,
                linf=linf,
                l2=l2,
                n_changed=n_changed,
                meta=json.dumps(meta),
            )
            results[model_name][attack_name] = m
            log.info(
                "  -> %.3f%% -> %.3f%%  (evasion %.2f%% of correct, Linf %.4f, %.0f/%d feats, %.1fs)",
                100 * m["accuracy_clean"], 100 * m["accuracy_adv"],
                100 * m["evasion_rate_on_correct"], m["linf_max"],
                m["features_changed_mean"], n_features, elapsed,
            )

        # ---- eps sweep -----------------------------------------------------
        if run_sweep:
            X_sw, y_sw, pred_sw = X_src[:n_sweep], y_src[:n_sweep], pred_clean[:n_sweep]
            for attack_name in [a for a in sweep_cfg["attacks"] if a in args.attacks]:
                for eps in sweep_cfg["eps"]:
                    attack = build_attack(
                        attack_name, clf_for(attack_name), cfg["attacks"][attack_name],
                        batch_size, eps_override=float(eps),
                    )
                    X_adv, elapsed = run_attack(attack, X_sw, y_sw)
                    pred_adv, _ = predict_labels(keras_model, X_adv)
                    linf, l2, n_changed = perturbation_norms(X_sw, X_adv)
                    m = summarise(y_sw, pred_sw, pred_adv, linf, l2, n_changed)
                    sweep_rows.append({
                        "model": model_name, "attack": attack_name, "eps": float(eps),
                        "n_samples": n_sweep, "accuracy_adv": m["accuracy_adv"],
                        "evasion_rate_on_correct": m["evasion_rate_on_correct"],
                        "l2_mean": m["l2_mean"], "seconds": round(elapsed, 1),
                    })
                    log.info(
                        "  sweep %-4s eps=%-5s -> %.3f%%  (%.1fs)",
                        attack_name, eps, 100 * m["accuracy_adv"], elapsed,
                    )

    # ---- reports -----------------------------------------------------------
    table = build_table(results)
    table.to_csv(reports_dir / f"table4_evasion{tag}.csv", index=False)

    sweep = pd.DataFrame(sweep_rows)
    if len(sweep):
        sweep.to_csv(reports_dir / f"eps_sweep{tag}.csv", index=False)

    run_meta = {
        "seed": seed,
        "data_dir": args.data_dir,
        "n_samples": len(idx),
        "sweep_samples": n_sweep if run_sweep else 0,
        "n_features": n_features,
        "source_row_indices_sha": int(idx.sum()),  # cheap reproducibility fingerprint
        "versions": {"art": art.__version__, "tensorflow": tf.__version__},
        "results": results,
    }
    (reports_dir / f"adversarial_metrics{tag}.json").write_text(
        json.dumps(run_meta, indent=2), encoding="utf-8"
    )
    write_markdown(table, sweep, run_meta, reports_dir / f"table4_evasion{tag}.md")

    try:
        figure3_evasion(run_meta, figures_dir / f"fig3_evasion{tag}.png")
        if len(sweep):
            figure4_eps_sweep(sweep, figures_dir / f"fig4_eps_sweep{tag}.png")
    except Exception as exc:  # noqa: BLE001 - never lose a run to a plotting bug
        log.error("figures failed (.npz and tables are safe): %s", exc)

    print("\n" + table.to_string(index=False))

    total = sum(m["elapsed_seconds"] for pa in results.values() for m in pa.values())
    log.info("generated %d cells, %.1f min of attack time", table.shape[0], total / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
