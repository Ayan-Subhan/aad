"""The inbound analyzer: mine benign traffic into rules.json.

Not a model -- a statistics pass. It looks only at *benign training* flows and
writes down what normal traffic obeys. Two consequences that matter for the
write-up:

* It never sees an adversarial example, so the validator cannot overfit to the
  attacks it will face. That is the property NIDS-CBAD claims and the AAD paper
  does not clearly have.
* It never sees val or test, so the false-positive rate measured later is
  measured on rows the rules were not fitted to.

Every tolerance is *calibrated*, not guessed. For each rule we compute its
residual across benign traffic and place the threshold at a high percentile of
that distribution, which fixes the rule's own false-positive rate by
construction. A hand-picked tolerance would be tuned, in practice, until the
catch rate looked good -- which is the failure this project exists to criticise.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .validator import (
    DURATION_COL,
    ORDERING_RULES,
    PRODUCT_RULES,
    RATE_RULES,
    Validator,
    norm_name,
)

log = logging.getLogger(__name__)

# Phase 1 stores X as float32, so recovering raw units costs about
# ``range * 2^-24`` (~6e-8 relative). These constants set every tolerance a
# comfortable margin above that noise floor while staying far below the shift a
# gradient perturbation produces.
FLOAT32_REL_TOL = 1e-6  # ~16x the float32 round-trip error
INTEGER_TOL_FLOOR = 1e-6  # absolute floor for narrow-range columns

# A tolerance of 0.5 or more makes the integrality check vacuous: every real
# number is within 0.5 of some integer. Wide-range columns hit that ceiling --
# ``Flow Byts/s`` spans ~8.5e8, so its float32 noise floor alone is ~850 -- which
# means integrality is simply *unverifiable* for them at float32 precision. Such
# columns are excluded and listed in rules.json rather than silently carried as
# rules that can never fire, which would overstate how much of the feature space
# the check actually covers.
INTEGRALITY_MAX_TOL = 0.25

# Benign dependency residuals are essentially float noise, so calibrating a
# tolerance at a raw percentile fits the noise and rejects the next benign
# sample. Give the calibrated value headroom, and floor it.
DEPENDENCY_TOL_HEADROOM = 10.0
DEPENDENCY_TOL_FLOOR = 1e-6


def _index_map(feature_names: list[str]) -> dict[str, int]:
    return {norm_name(n): i for i, n in enumerate(feature_names)}


def _resolve(idx_map: dict[str, int], *names: str) -> list[int] | None:
    """Column indices for a rule, or None if any column was dropped in phase 1."""
    out = []
    for n in names:
        i = idx_map.get(norm_name(n))
        if i is None:
            return None
        out.append(i)
    return out


def find_integer_columns(
    raw: np.ndarray, feature_names: list[str], tolerance: np.ndarray
) -> list[int]:
    """Discover count-like columns empirically rather than hardcoding a list.

    Hardcoding invites a silent mismatch when the feature set changes. Deriving
    it from the data means rules.json states exactly what was found, and the
    write-up can show the discovered list is the expected one (ports, protocol,
    packet counts, header lengths, flag counts, window sizes, subflow counts).

    ``tolerance`` is per column, because the float32 round-trip error scales
    with the column's range -- see ``Validator.check_integrality``.

    Returns ``(usable, unverifiable)``: columns that are integral *and* can be
    checked meaningfully, and columns whose required tolerance is so loose the
    test could never fail.
    """
    frac = np.abs(raw - np.rint(raw))
    integral = (frac <= tolerance).all(axis=0)
    checkable = tolerance < INTEGRALITY_MAX_TOL

    idx = np.flatnonzero(integral & checkable).tolist()
    unverifiable = np.flatnonzero(integral & ~checkable).tolist()

    log.info(
        "%d/%d columns are integral and checkable: %s",
        len(idx), raw.shape[1], [feature_names[i] for i in idx],
    )
    if unverifiable:
        log.info(
            "%d further column(s) look integral but span too wide a range to verify "
            "at float32 precision; excluded: %s",
            len(unverifiable), [feature_names[i] for i in unverifiable],
        )
    return idx, unverifiable


def build_dependency_rules(feature_names: list[str]) -> list[dict]:
    """Instantiate every declared invariant whose columns survived phase 1."""
    idx_map = _index_map(feature_names)
    rules: list[dict] = []

    for lo, mid, hi in ORDERING_RULES:
        ix = _resolve(idx_map, lo, mid, hi)
        if ix:
            rules.append({
                "kind": "ordering", "name": f"{lo} <= {mid} <= {hi}",
                "columns": [lo, mid, hi], "indices": ix,
            })

    for total, count, mean in PRODUCT_RULES:
        ix = _resolve(idx_map, total, count, mean)
        if ix:
            rules.append({
                "kind": "product", "name": f"{total} ~= {count} * {mean}",
                "columns": [total, count, mean], "indices": ix,
            })

    i_dur = idx_map.get(norm_name(DURATION_COL))
    if i_dur is not None:
        for rate, numerators in RATE_RULES:
            ix = _resolve(idx_map, rate, *numerators)
            if ix:
                rules.append({
                    "kind": "rate",
                    "name": f"{rate} ~= ({' + '.join(numerators)}) / {DURATION_COL}",
                    "columns": [rate, *numerators, DURATION_COL],
                    "rate_index": ix[0], "numerator_indices": ix[1:], "duration_index": i_dur,
                })

    return rules


def mine(
    X_benign_scaled: np.ndarray,
    feature_names: list[str],
    data_min: np.ndarray,
    data_range: np.ndarray,
    range_tolerance: float = 0.01,
    dependency_percentile: float = 99.9,
) -> dict:
    """Build the rule set from benign training traffic.

    ``range_tolerance`` widens each observed [min,max] by that fraction of the
    column's span. Benign val/test rows legitimately fall slightly outside what
    train happened to contain, and a zero-tolerance range rule would charge that
    to the false-positive budget.
    """
    raw = np.asarray(X_benign_scaled, dtype=np.float64) * data_range + data_min
    n_rows, n_cols = raw.shape
    log.info("mining %d benign training rows x %d features", n_rows, n_cols)

    lo, hi = raw.min(axis=0), raw.max(axis=0)
    pad = range_tolerance * np.maximum(hi - lo, 1e-9)

    # Per column, because the cost of the float32 round-trip scales with span.
    int_tol = np.maximum(INTEGER_TOL_FLOOR, data_range * FLOAT32_REL_TOL)
    int_idx, int_unverifiable = find_integer_columns(raw, feature_names, int_tol)

    rules = {
        "provenance": {
            "source": "benign rows of data/processed/train.npz",
            "n_rows": int(n_rows),
            "range_tolerance": range_tolerance,
            "dependency_percentile": dependency_percentile,
            "integer_rel_tolerance": FLOAT32_REL_TOL,
        },
        "feature_names": feature_names,
        "scaling": {"data_min": data_min.tolist(), "data_range": data_range.tolist()},
        "range": {"min": (lo - pad).tolist(), "max": (hi + pad).tolist()},
        "integrality": {
            "indices": int_idx,
            "columns": [feature_names[i] for i in int_idx],
            "tolerance": int_tol[int_idx].tolist(),
            "max_tolerance": INTEGRALITY_MAX_TOL,
            # Integral in the data, but spanning too wide a range for float32 to
            # confirm it. Listed so the count above is not mistaken for "every
            # count-like column in the schema".
            "excluded_unverifiable": [feature_names[i] for i in int_unverifiable],
        },
        "dependency": {"rules": build_dependency_rules(feature_names)},
        # Filled by calibrate_distribution; placeholder keeps Validator loadable.
        "distribution": {
            "median": np.median(raw, axis=0).tolist(),
            "mad": _mad(raw).tolist(),
            "threshold": float("inf"),
        },
    }

    _calibrate_dependencies(rules, raw, dependency_percentile)
    return rules


def _mad(raw: np.ndarray) -> np.ndarray:
    """Median absolute deviation, scaled to be comparable to a std deviation.

    Floored so a column that is constant among benign rows does not divide by
    zero and flag every row on a rounding difference.
    """
    med = np.median(raw, axis=0)
    mad = np.median(np.abs(raw - med), axis=0) * 1.4826
    span = np.maximum(raw.max(axis=0) - raw.min(axis=0), 1e-9)
    return np.maximum(mad, 1e-3 * span)


def _calibrate_dependencies(rules: dict, raw: np.ndarray, percentile: float) -> None:
    """Set each invariant's tolerance from its own benign residual distribution.

    CICFlowMeter's own arithmetic is not exact -- it rounds, and it computes
    means over slightly different packet sets than the totals -- so a strict
    equality would reject real traffic. The percentile absorbs that measurement
    noise while leaving the tolerance far below what a gradient step produces.
    """
    v = Validator(rules)
    for rule in rules["dependency"]["rules"]:
        res = v.dependency_residual(raw, {**rule, "tolerance": 0.0})
        tol = float(np.percentile(res, percentile))
        # Headroom and a floor. Benign residuals for an invariant CICFlowMeter
        # satisfies exactly are pure float32 noise (order 1e-8), so taking the
        # percentile at face value fits the noise of *this* sample and rejects
        # the next one. Both constants stay ~5 orders of magnitude below the
        # residual a gradient perturbation produces (order 1e-1), so the
        # separation is not close.
        rule["tolerance"] = max(tol * DEPENDENCY_TOL_HEADROOM, DEPENDENCY_TOL_FLOOR)
        rule["benign_residual"] = {
            "p50": float(np.percentile(res, 50)),
            "p99": float(np.percentile(res, 99)),
            f"p{percentile}": tol,
            "max": float(res.max()),
        }
        log.info(
            "  %-52s tol=%.3e  (benign p50 %.2e, max %.2e)",
            rule["name"], rule["tolerance"], rule["benign_residual"]["p50"], res.max(),
        )


def calibrate_distribution(
    rules: dict, X_val_benign_scaled: np.ndarray, fpr_budget: float = 0.01
) -> dict:
    """Set the outlier threshold on *validation* benign traffic.

    Calibrating on the same rows the median/MAD were mined from would report an
    optimistic false-positive rate -- exactly the leak this project criticises
    elsewhere. The threshold is the (1 - budget) quantile of the benign val
    score, so the distribution family spends its 1% budget by construction.
    """
    v = Validator(rules)
    scores = v.distribution_score(v.to_raw(X_val_benign_scaled))
    threshold = float(np.percentile(scores, 100 * (1 - fpr_budget)))
    rules["distribution"]["threshold"] = threshold
    rules["distribution"]["calibration"] = {
        "source": "benign rows of data/processed/val.npz",
        "n_rows": int(len(scores)),
        "fpr_budget": fpr_budget,
        "score_p50": float(np.percentile(scores, 50)),
        "score_p99": float(np.percentile(scores, 99)),
    }
    log.info(
        "distribution threshold %.4f from %d benign val rows at a %.1f%% budget",
        threshold, len(scores), 100 * fpr_budget,
    )
    return rules


def save(rules: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rules, indent=2), encoding="utf-8")
    log.info("wrote %s", path)
    return path
