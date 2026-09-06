"""Gate 1: reject flows that are not valid network flows.

The premise of the whole defense. A gradient attack perturbs a 68-dimensional
vector of *statistics*, but those statistics were computed from real packets and
are therefore not independent. Move them freely and you get a row describing a
flow that could not exist: 7.4 packets, a minimum packet length above the
maximum, a byte rate that contradicts its own duration. The validator checks
arithmetic, not learned behaviour, so it needs no adversarial training data and
has nothing to overfit.

Four rule families, each answering a different question:

* **range**        - is every value inside what benign traffic ever showed?
* **integrality**  - are the count-like features whole numbers?
* **dependency**   - do the features still agree with each other?
* **distribution** - is the row jointly plausible, even if each value is legal?

This module owns both the rule *definitions* and the residual arithmetic.
``analyzer.py`` imports the same functions to calibrate tolerances, so the
mining pass and the checking pass can never drift apart -- if they did, every
threshold would be measuring something other than what it gates.

The checks run in **raw units**, not the scaled [0,1] space. "Whole number"
becomes "on a lattice of step 1/(max-min)" after MinMax scaling, which is
technically checkable and practically unreadable. Everything needed to invert
the scaling is stored in rules.json, so the validator is pure numpy with no
sklearn dependency at inference time.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

FAMILIES = ("range", "integrality", "dependency", "distribution")

# Ceiling on any single feature's z-score, so one near-constant column cannot
# dominate a row's outlier score. See Validator.distribution_score.
Z_CLIP = 50.0


def norm_name(s: str) -> str:
    """'TotLen Fwd Pkts' -> 'totlenfwdpkts'. Absorbs CICFlowMeter spacing drift."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# --- declarative rule definitions ------------------------------------------
#
# Named by CSV header. Any rule referencing a column that phase 1 dropped as
# constant is silently skipped, which is why these are declarations rather than
# hardcoded indices.

# min <= mean <= max, per statistic family.
ORDERING_RULES: list[tuple[str, str, str]] = [
    ("Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Max"),
    ("Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Max"),
    ("Pkt Len Min", "Pkt Len Mean", "Pkt Len Max"),
    ("Flow IAT Min", "Flow IAT Mean", "Flow IAT Max"),
    ("Fwd IAT Min", "Fwd IAT Mean", "Fwd IAT Max"),
    ("Bwd IAT Min", "Bwd IAT Mean", "Bwd IAT Max"),
    ("Active Min", "Active Mean", "Active Max"),
    ("Idle Min", "Idle Mean", "Idle Max"),
]

# total ~= count * mean
PRODUCT_RULES: list[tuple[str, str, str]] = [
    ("TotLen Fwd Pkts", "Tot Fwd Pkts", "Fwd Pkt Len Mean"),
    ("TotLen Bwd Pkts", "Tot Bwd Pkts", "Bwd Pkt Len Mean"),
]

# rate ~= (numerator terms summed) / (Flow Duration in seconds)
RATE_RULES: list[tuple[str, list[str]]] = [
    ("Flow Pkts/s", ["Tot Fwd Pkts", "Tot Bwd Pkts"]),
    ("Flow Byts/s", ["TotLen Fwd Pkts", "TotLen Bwd Pkts"]),
    ("Fwd Pkts/s", ["Tot Fwd Pkts"]),
    ("Bwd Pkts/s", ["Tot Bwd Pkts"]),
]

DURATION_COL = "Flow Duration"  # microseconds in CICFlowMeter
MICROSECONDS = 1e6


def _rel_residual(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Scale-free disagreement between two quantities.

    Absolute error is useless here: ``Flow Duration`` runs to 1e8 while
    ``Down/Up Ratio`` is single digits, so one absolute tolerance cannot serve
    both. The +1 floor keeps the ratio finite when both sides are near zero,
    which is the common case for idle/active timers.
    """
    return np.abs(actual - expected) / (np.abs(expected) + 1.0)


def ordering_residual(raw: np.ndarray, i_lo: int, i_mid: int, i_hi: int) -> np.ndarray:
    """How far the min<=mean<=max ordering is violated, in relative terms."""
    lo, mid, hi = raw[:, i_lo], raw[:, i_mid], raw[:, i_hi]
    scale = np.abs(hi) + 1.0
    return np.maximum(np.maximum(lo - mid, mid - hi), 0.0) / scale


def product_residual(raw: np.ndarray, i_total: int, i_count: int, i_mean: int) -> np.ndarray:
    return _rel_residual(raw[:, i_total], raw[:, i_count] * raw[:, i_mean])


def rate_residual(raw: np.ndarray, i_rate: int, num_idx: list[int], i_dur: int) -> np.ndarray:
    seconds = raw[:, i_dur] / MICROSECONDS
    # A zero-duration flow makes the rate undefined rather than wrong; phase 1
    # already dropped those rows (they were the Infinity cells), but guard anyway.
    safe = np.where(np.abs(seconds) < 1e-9, np.nan, seconds)
    expected = raw[:, num_idx].sum(axis=1) / safe
    res = _rel_residual(raw[:, i_rate], expected)
    return np.nan_to_num(res, nan=0.0, posinf=0.0)


@dataclass
class ValidationResult:
    """Which rows were rejected, and by which family."""

    reject: np.ndarray
    by_family: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def reject_rate(self) -> float:
        return float(self.reject.mean()) if len(self.reject) else 0.0

    def family_rates(self) -> dict[str, float]:
        return {k: float(v.mean()) for k, v in self.by_family.items()}


class Validator:
    """Stateless, vectorised checker over a mined rules.json."""

    def __init__(self, rules: dict):
        self.rules = rules
        self.feature_names: list[str] = rules["feature_names"]
        self.n_features = len(self.feature_names)

        # Inverse MinMax, stored so this class needs no scaler.pkl.
        self._data_min = np.asarray(rules["scaling"]["data_min"], dtype=np.float64)
        self._data_range = np.asarray(rules["scaling"]["data_range"], dtype=np.float64)

        r = rules["range"]
        self._lo = np.asarray(r["min"], dtype=np.float64)
        self._hi = np.asarray(r["max"], dtype=np.float64)

        self._int_idx = np.asarray(rules["integrality"]["indices"], dtype=np.int64)
        # Per-column, not a single scalar. See check_integrality.
        self._int_tol = np.asarray(rules["integrality"]["tolerance"], dtype=np.float64)

        self._deps = rules["dependency"]["rules"]

        d = rules["distribution"]
        self._median = np.asarray(d["median"], dtype=np.float64)
        self._mad = np.asarray(d["mad"], dtype=np.float64)
        self._z_threshold = float(d["threshold"])

    @classmethod
    def load(cls, path: Path) -> Validator:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    # --- unit conversion ---------------------------------------------------

    def to_raw(self, X_scaled: np.ndarray) -> np.ndarray:
        """Undo phase 1's MinMax scaling: X * (max-min) + min."""
        return np.asarray(X_scaled, dtype=np.float64) * self._data_range + self._data_min

    # --- the four families -------------------------------------------------

    def check_range(self, raw: np.ndarray) -> np.ndarray:
        return ((raw < self._lo) | (raw > self._hi)).any(axis=1)

    def check_integrality(self, raw: np.ndarray) -> np.ndarray:
        """The rule that does most of the work.

        A gradient step adds a real-valued delta to every feature it touches, so
        a count-like column lands off the integers almost surely. FGSM at
        eps=0.1 cheerfully produces 7.4 packets.

        The tolerance is **per column and proportional to that column's range**,
        not one absolute number. Phase 1 stores X as float32, so recovering raw
        units costs about ``range * 2^-24``: negligible for ``SYN Flag Cnt``
        (span 2, error 2e-8) but 7.8e-4 for ``Init Fwd Win Byts`` (span 65,535)
        and **1.43** for ``Flow Duration`` (span 1.2e8). A flat 1e-6 tolerance
        would therefore call genuine benign flows fractional and reject almost
        all legitimate traffic. Scaling with the range leaves ~16x headroom over
        float32 noise while staying ~5 orders of magnitude below the shift an
        eps=0.1 perturbation produces, so the separation is not close.
        """
        if not len(self._int_idx):
            return np.zeros(len(raw), dtype=bool)
        sub = raw[:, self._int_idx]
        return (np.abs(sub - np.rint(sub)) > self._int_tol).any(axis=1)

    def check_dependency(self, raw: np.ndarray) -> np.ndarray:
        viol = np.zeros(len(raw), dtype=bool)
        for rule in self._deps:
            viol |= self.dependency_residual(raw, rule) > rule["tolerance"]
        return viol

    def dependency_residual(self, raw: np.ndarray, rule: dict) -> np.ndarray:
        """One rule's residual. Shared with the analyzer's calibration pass."""
        kind = rule["kind"]
        if kind == "ordering":
            return ordering_residual(raw, *rule["indices"])
        if kind == "product":
            return product_residual(raw, *rule["indices"])
        if kind == "rate":
            return rate_residual(raw, rule["rate_index"], rule["numerator_indices"], rule["duration_index"])
        if kind == "nonnegative":
            idx = np.asarray(rule["indices"], dtype=np.int64)
            # Residual = how far below zero, relative to the column's own scale.
            return np.maximum(-raw[:, idx] / (np.abs(self._hi[idx]) + 1.0), 0.0).max(axis=1)
        raise ValueError(f"unknown dependency rule kind {kind!r}")

    def distribution_score(self, raw: np.ndarray) -> np.ndarray:
        """Robust per-row outlier score: the mean of clipped per-feature z-scores.

        Median/MAD rather than mean/std, because flow features are heavy-tailed
        enough that a standard deviation is dominated by the very tail it is
        meant to detect.

        Two details that took a failed run to get right:

        * **Mean, not max.** Scoring a row by its single worst feature lets one
          near-constant column decide everything. Many flow columns are almost
          always zero (the flag counts), so their MAD sits on its floor and any
          nonzero value pegs z at the cap -- for enough benign rows that the
          calibrated threshold saturates and the family never fires at all.
          Averaging is also a better match to the threat: a gradient attack
          nudges ~65 of the 68 features a little, rather than one a lot.
        * **Clipped.** Without a cap a single degenerate column still dominates
          the mean. ``Z_CLIP`` bounds any one feature's contribution, which makes
          this an L1 diagonal Mahalanobis distance in all but name.
        """
        z = np.abs(raw - self._median) / self._mad
        np.clip(z, 0.0, Z_CLIP, out=z)
        return z.mean(axis=1)

    def check_distribution(self, raw: np.ndarray) -> np.ndarray:
        return self.distribution_score(raw) > self._z_threshold

    # --- combined ----------------------------------------------------------

    def check(self, X_scaled: np.ndarray, families: tuple[str, ...] = FAMILIES) -> ValidationResult:
        """Validate scaled rows, reporting which family fired.

        Attribution matters: a combined reject rate cannot tell you whether the
        defense rests on arithmetic (which an adaptive attacker can satisfy) or
        on distribution (which is harder to satisfy silently). Phase 8 needs
        exactly this breakdown.
        """
        raw = self.to_raw(X_scaled)
        checks = {
            "range": self.check_range,
            "integrality": self.check_integrality,
            "dependency": self.check_dependency,
            "distribution": self.check_distribution,
        }
        by_family = {f: checks[f](raw) for f in families}
        reject = np.zeros(len(raw), dtype=bool)
        for mask in by_family.values():
            reject |= mask
        return ValidationResult(reject=reject, by_family=by_family)
