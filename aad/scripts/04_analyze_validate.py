"""Phase 4 - mine benign traffic into rules.json, then measure the validator.

    python scripts/04_analyze_validate.py                 # mine + evaluate
    python scripts/04_analyze_validate.py --fpr-budget 0.005

Writes:
    artifacts/rules.json                           the mined rule set
    artifacts/reports/table5_validator.{csv,md}    catch-rate and FPR, separately
    artifacts/reports/validator_metrics.json

The two numbers this produces are reported apart and never blended. A validator
that catches 100% of adversarial rows while rejecting 3% of legitimate traffic
is unusable, and a single combined "accuracy" would hide that completely.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aad.defense.analyzer import calibrate_distribution, mine, save  # noqa: E402
from aad.defense.validator import FAMILIES, Validator  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("validate")


def load_split(data_dir: Path, name: str):
    d = np.load(data_dir / f"{name}.npz")
    return d["X"].astype(np.float32), d["y"].astype(np.int64), d["y_multiclass"].astype(np.int8)


def row_for(label: str, kind: str, result, n: int) -> dict:
    row = {"set": label, "kind": kind, "n": n, "combined_%": round(100 * result.reject_rate, 3)}
    for family, rate in result.family_rates().items():
        row[f"{family}_%"] = round(100 * rate, 3)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--adv-dir", default="artifacts/adversarial")
    ap.add_argument("--out", default="artifacts/rules.json")
    ap.add_argument("--reports-dir", default="artifacts/reports")
    ap.add_argument("--fpr-budget", type=float, default=0.01,
                    help="distribution family's false-positive budget on benign val")
    ap.add_argument("--range-tolerance", type=float, default=0.01)
    ap.add_argument("--dependency-percentile", type=float, default=99.9)
    args = ap.parse_args()

    data_dir = ROOT / args.data_dir
    reports_dir = ROOT / args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, _ = load_split(data_dir, "train")
    X_val, y_val, _ = load_split(data_dir, "val")
    X_test, y_test, _ = load_split(data_dir, "test")
    feature_names = json.loads((data_dir / "feature_names.json").read_text(encoding="utf-8"))

    scaler = joblib.load(data_dir / "scaler.pkl")
    data_min = np.asarray(scaler.data_min_, dtype=np.float64)
    data_range = np.asarray(scaler.data_range_, dtype=np.float64)

    # ---- mine on benign train, calibrate on benign val ---------------------
    rules = mine(
        X_train[y_train == 0], feature_names, data_min, data_range,
        range_tolerance=args.range_tolerance,
        dependency_percentile=args.dependency_percentile,
    )
    rules = calibrate_distribution(rules, X_val[y_val == 0], fpr_budget=args.fpr_budget)
    save(rules, ROOT / args.out)

    validator = Validator(rules)
    rows: list[dict] = []

    # ---- false positives, on rows the rules never saw ----------------------
    benign_test = X_test[y_test == 0]
    attack_test = X_test[y_test == 1]
    fpr = validator.check(benign_test)
    rows.append(row_for("clean benign (test)", "FPR", fpr, len(benign_test)))
    rows.append(row_for("clean attack (test)", "reject", validator.check(attack_test), len(attack_test)))

    log.info("=" * 72)
    log.info("false-positive rate on %d unseen benign test rows: %.4f%%",
             len(benign_test), 100 * fpr.reject_rate)
    for family, rate in fpr.family_rates().items():
        log.info("    %-13s %.4f%%", family, 100 * rate)

    # ---- catch rate, per adversarial artefact ------------------------------
    adv_dir = ROOT / args.adv_dir
    npz_files = sorted(adv_dir.glob("*.npz"))
    if not npz_files:
        log.warning("no .npz in %s -- run scripts/03_generate_ae.py first", adv_dir)

    per_cell: list[dict] = []
    for path in npz_files:
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        res = validator.check(d["X_adv"])
        # The number that actually matters. Every one of these adversarial rows
        # was built from a real attack flow, and real attack flows already sit
        # outside the benign envelope the rules were mined from -- so a raw
        # catch rate partly measures *maliciousness*, not *perturbation*. The
        # gate's real contribution is what it catches beyond what it already
        # caught on the same rows unperturbed.
        base = validator.check(d["X_clean"])
        row = row_for(f"{meta['attack']} x {meta['model']}", "catch", res, len(d["X_adv"]))
        row["clean_orig_%"] = round(100 * base.reject_rate, 3)
        row["attributable_%"] = round(100 * float((res.reject & ~base.reject).mean()), 3)
        row["attack"], row["model"] = meta["attack"], meta["model"]
        per_cell.append(row)
        log.info(
            "%-22s catch %.2f%% (clean originals %.2f%%, attributable %.2f%%)  [%s]",
            path.stem, 100 * res.reject_rate, 100 * base.reject_rate,
            row["attributable_%"],
            ", ".join(f"{k} {100*v:.1f}%" for k, v in res.family_rates().items()),
        )

    table = pd.DataFrame(rows + [{k: v for k, v in r.items() if k not in ("attack", "model")}
                                 for r in per_cell])
    table.to_csv(reports_dir / "table5_validator.csv", index=False)

    # ---- report ------------------------------------------------------------
    fam_cols = [f"{f}_%" for f in FAMILIES]
    md = [
        "# Table V - validator catch-rate and false-positive rate",
        "",
        f"Rules mined from {rules['provenance']['n_rows']:,} **benign training** rows. "
        f"The distribution threshold is calibrated on benign *validation* rows at a "
        f"{100*args.fpr_budget:.1f}% budget. Everything below is measured on **test**, "
        "which neither pass saw.",
        "",
        "Catch-rate and FPR are reported separately and per rule family. A validator "
        "rejecting 3% of legitimate traffic is unusable whatever its catch rate, and a "
        "single blended number would hide that.",
        "",
        table.to_markdown(index=False),
        "",
        "## Reading this",
        "",
        f"- **FPR** on clean benign test traffic is the cost of deploying the gate: "
        f"{100*fpr.reject_rate:.3f}% of legitimate flows dropped. Note the "
        f"{100*args.fpr_budget:.1f}% budget is spent by the *distribution* family alone; "
        "the four families are OR-ed, so the combined FPR is roughly their sum. Tighten "
        "`--fpr-budget` if the combined figure has to sit under a hard limit.",
        "- **reject** on clean *attack* traffic is not an error -- those are real "
        "intrusions and the EIDS behind the gate would have flagged them anyway. It is "
        "reported so the gate's behaviour is fully described.",
        "- **catch** is the fraction of adversarial rows the gate stops before any model "
        "sees them.",
        "- **clean_orig** is the gate's reject rate on the *same rows unperturbed*, and "
        "**attributable** is the fraction it catches that it would not have caught anyway. "
        "This distinction is easy to miss and it matters: every adversarial row here was "
        "built from a real attack flow, and real attack flows already fall outside the "
        "benign envelope the rules were mined from. A headline catch rate therefore "
        "partly measures *maliciousness* rather than *perturbation*. **`attributable` is "
        "the honest measure of what the gate adds as an adversarial-example detector**; "
        "read the two together or the result is overstated.",
        "",
    ]
    if per_cell:
        cell_df = pd.DataFrame(per_cell)
        dominant = max(FAMILIES, key=lambda f: cell_df[f"{f}_%"].mean())
        md += [
            f"The **{dominant}** family alone accounts for most of the catch rate "
            f"({cell_df[f'{dominant}_%'].mean():.2f}% mean across the twelve cells). That is "
            "the honest finding: the defense rests on arithmetic that a gradient attack "
            "violates incidentally, not on anything the attacker is forced to violate. An "
            "adaptive adversary that rounds its perturbation onto the integer lattice and "
            "re-derives the dependent features would pass this gate untouched -- which is "
            "what phase 8 is for.",
            "",
        ]
    (reports_dir / "table5_validator.md").write_text("\n".join(md), encoding="utf-8")

    (reports_dir / "validator_metrics.json").write_text(
        json.dumps({
            "fpr_budget": args.fpr_budget,
            "range_tolerance": args.range_tolerance,
            "dependency_percentile": args.dependency_percentile,
            "n_integer_columns": len(rules["integrality"]["indices"]),
            "integer_columns": [feature_names[i] for i in rules["integrality"]["indices"]],
            "n_dependency_rules": len(rules["dependency"]["rules"]),
            "distribution_threshold": rules["distribution"]["threshold"],
            "rows": rows + per_cell,
        }, indent=2),
        encoding="utf-8",
    )

    print("\n" + table.to_string(index=False))

    if fpr.reject_rate > 0.03:
        log.warning("FPR %.3f%% exceeds the 3%% usability line", 100 * fpr.reject_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
