"""The AE generator: turn true-attack flows into adversarial ones.

The whole phase is four white-box evasion attacks run against three models. What
makes the output trustworthy is not the attacks -- ART implements those -- but
four choices made here:

1. **One fixed sample of source rows, shared by all twelve runs.** Every
   (attack, model) pair perturbs the *same* 5,000 test rows. That is what lets
   the twelve results be compared to each other, lets phase 5 pair every
   adversarial row with its exact clean original, and lets phase 4 measure its
   catch-rate against the identical clean baseline.

2. **The true label is passed to the attack.** With ``y=None`` ART substitutes
   the model's own prediction, which makes the result depend on model error and
   quietly changes the meaning of "evasion". We always attack away from ground
   truth.

3. **DeepFool is not eps-parameterised.** Its ``epsilon`` is an overshoot
   multiplier applied once the boundary is crossed, not a perturbation budget,
   so it is excluded from the eps sweep. Sweeping it would produce a curve that
   looks meaningful and is not.

4. **Both the clean and the perturbed prediction are recorded.** Evasion is only
   interesting on rows the model got right to begin with, and keeping both lets
   any downstream phase choose that denominator without regenerating anything.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from art.attacks.evasion import (
    BasicIterativeMethod,
    DeepFool,
    FastGradientMethod,
    ProjectedGradientDescent,
)
from art.estimators.classification import TensorFlowV2Classifier

log = logging.getLogger(__name__)

ATTACK_NAMES = ("fgsm", "bim", "pgd", "deepfool")

# Attacks whose strength is governed by an L-inf budget, and so can be swept.
EPS_ATTACKS = ("fgsm", "bim", "pgd")


def select_attack_rows(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Draw ``n`` attack-class row indices, reproducibly.

    Sampled from rows where ``y == 1`` only: an adversarial example is a
    *malicious* flow disguised as benign, so perturbing a benign row would have
    no attacker motive and no measurable success criterion.

    Returned sorted, so the selection is identical across runs and reads
    sequentially off the source array.
    """
    pool = np.flatnonzero(y == 1)
    if len(pool) < n:
        raise ValueError(
            f"asked for {n} attack rows but the split only has {len(pool)}. "
            "Lower n_samples in config/attacks.yaml or use a larger split."
        )
    rng = np.random.default_rng(seed)
    idx = rng.choice(pool, size=n, replace=False)
    return np.sort(idx).astype(np.int32)


def build_attack(
    name: str,
    classifier: TensorFlowV2Classifier,
    params: dict,
    batch_size: int,
    eps_override: float | None = None,
):
    """Construct one ART attack from its config block.

    ``eps_override`` lets the sweep reuse the configured settings and vary only
    the budget, so a sweep point differs from the canonical run in exactly one
    parameter.
    """
    name = name.lower()
    p = dict(params)
    if eps_override is not None:
        if name not in EPS_ATTACKS:
            raise ValueError(f"{name} has no eps budget to override")
        p["eps"] = float(eps_override)

    if name == "fgsm":
        eps = float(p["eps"])
        return FastGradientMethod(
            estimator=classifier,
            norm=np.inf,
            eps=eps,
            # Unused when minimal=False (the single-step form), but ART still
            # validates it as > 0, and a stale default larger than eps is
            # confusing to read back in the saved metadata.
            eps_step=eps,
            targeted=False,
            num_random_init=0,
            batch_size=batch_size,
            minimal=False,
        )

    if name == "bim":
        return BasicIterativeMethod(
            estimator=classifier,
            eps=float(p["eps"]),
            eps_step=float(p["eps_step"]),
            max_iter=int(p["max_iter"]),
            targeted=False,
            batch_size=batch_size,
            verbose=False,
        )

    if name == "pgd":
        return ProjectedGradientDescent(
            estimator=classifier,
            norm=np.inf,
            eps=float(p["eps"]),
            eps_step=float(p["eps_step"]),
            max_iter=int(p["max_iter"]),
            targeted=False,
            num_random_init=int(p.get("num_random_init", 0)),
            batch_size=batch_size,
            verbose=False,
        )

    if name == "deepfool":
        return DeepFool(
            classifier=classifier,
            max_iter=int(p["max_iter"]),
            # Overshoot multiplier, not a budget.
            epsilon=float(p["epsilon"]),
            # ART defaults this to 10; with 2 classes anything above 2 is waste.
            nb_grads=int(p.get("nb_grads", 2)),
            # ART defaults this to 1, which would mean 5,000 sequential
            # forward/backward passes per model.
            batch_size=batch_size,
            verbose=False,
        )

    raise ValueError(f"unknown attack {name!r}; expected one of {ATTACK_NAMES}")


def one_hot(y: np.ndarray) -> np.ndarray:
    """Match the 2-unit softmax targets phase 2 trained against."""
    out = np.zeros((len(y), 2), dtype=np.float32)
    out[np.arange(len(y)), y.astype(np.int64)] = 1.0
    return out


def run_attack(attack, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Generate adversarial examples and report wall-clock cost.

    ``y`` is the *true* label, one-hot encoded. For a non-targeted attack ART
    reads it as "the class to move away from".
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    t0 = time.time()
    X_adv = attack.generate(x=X, y=one_hot(y))
    elapsed = time.time() - t0
    return X_adv.astype(np.float32), elapsed


def perturbation_norms(
    X_clean: np.ndarray, X_adv: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-sample L-inf distance, L2 distance, and count of features touched.

    L-inf is the budget the attack was configured with, so it doubles as a
    correctness check. L2 is the honest measure of *how much traffic changed*,
    and phase 4 uses it as a magnitude rule. The touched-feature count matters
    to phase 4 too: a dense perturbation trips far more domain rules than a
    sparse one, which is most of why DeepFool and FGSM behave differently there.
    """
    delta = (X_adv - X_clean).astype(np.float64)
    return (
        np.abs(delta).max(axis=1).astype(np.float32),
        np.linalg.norm(delta, axis=1).astype(np.float32),
        (delta != 0).sum(axis=1).astype(np.int32),
    )


def predict_labels(model, X: np.ndarray, batch_size: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Hard labels plus the full probability vector."""
    prob = model.predict(X, batch_size=batch_size, verbose=0).astype(np.float32)
    return prob.argmax(axis=1).astype(np.int8), prob


def summarise(
    y_true: np.ndarray,
    pred_clean: np.ndarray,
    pred_adv: np.ndarray,
    linf: np.ndarray,
    l2: np.ndarray,
    n_changed: np.ndarray,
) -> dict:
    """The numbers that describe one (attack, model) cell.

    Every source row is a true attack, so on this subset accuracy, recall and
    detection rate are the same quantity, and ``1 - accuracy`` is the attack
    success rate. Precision and FPR are undefined here (there are no benign rows
    to be wrong about) and are deliberately not reported.
    """
    n = len(y_true)
    correct_before = pred_clean == 1
    n_correct_before = int(correct_before.sum())

    return {
        "n_samples": n,
        # Was the model right on these rows before we touched them? Anything
        # below ~1.0 means the collapse figure below is flattered by rows the
        # model already misread.
        "accuracy_clean": float(correct_before.mean()),
        "accuracy_adv": float((pred_adv == 1).mean()),
        # The honest denominator: of the rows the model classified correctly,
        # how many did the attack flip?
        "evasion_rate_on_correct": (
            float((pred_adv[correct_before] == 0).mean()) if n_correct_before else 0.0
        ),
        "attack_success_rate": float((pred_adv == 0).mean()),
        "n_flipped": int((correct_before & (pred_adv == 0)).sum()),
        "n_correct_before": n_correct_before,
        "linf_mean": float(linf.mean()),
        "linf_max": float(linf.max()),
        "l2_mean": float(l2.mean()),
        "l2_max": float(l2.max()),
        # Out of 68 features, how many the attack actually moved, on average.
        "features_changed_mean": float(n_changed.mean()),
    }
