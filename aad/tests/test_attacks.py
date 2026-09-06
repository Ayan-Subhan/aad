"""Contract tests for the AE generator.

Deliberately synthetic: a tiny model on random data, so these run in seconds and
need neither the 684 MB of raw CSVs nor a trained baseline. What they check is
not "does the attack work" (ART's problem) but "did we wire it up such that the
numbers mean what we say they mean" -- which is our problem, and the failure
mode that produces plausible wrong results rather than a crash.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorflow import keras  # noqa: E402

from aad.attacks.generate import (  # noqa: E402
    EPS_ATTACKS,
    build_attack,
    one_hot,
    perturbation_norms,
    run_attack,
    select_attack_rows,
    summarise,
)
from aad.attacks.wrappers import as_logits, wrap  # noqa: E402
from aad.models.base import compile_model  # noqa: E402

N_FEATURES = 12
N_ROWS = 64
EPS = 0.1


@pytest.fixture(scope="module")
def classifier():
    """A trained-enough 2-unit softmax model, matching the phase 2 contract."""
    keras.utils.set_random_seed(0)
    rng = np.random.default_rng(0)
    X = rng.random((512, N_FEATURES)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(np.int64)

    inputs = keras.Input(shape=(N_FEATURES,))
    h = keras.layers.Dense(16, activation="relu")(inputs)
    outputs = keras.layers.Dense(2, activation="softmax")(h)
    model = keras.Model(inputs, outputs, name="tiny")
    compile_model(model, "adam", 0.01)
    model.fit(X, one_hot(y), epochs=12, batch_size=64, verbose=0)
    return model, wrap(model, N_FEATURES)


@pytest.fixture(scope="module")
def source_rows():
    rng = np.random.default_rng(1)
    X = rng.random((N_ROWS, N_FEATURES)).astype(np.float32)
    return X, np.ones(N_ROWS, dtype=np.int64)


# --- row selection ---------------------------------------------------------


def test_select_attack_rows_is_reproducible():
    y = np.array([0, 1] * 50)
    a = select_attack_rows(y, 10, seed=42)
    b = select_attack_rows(y, 10, seed=42)
    assert np.array_equal(a, b), "same seed must give the same rows across runs"


def test_select_attack_rows_only_picks_attacks():
    y = np.array([0] * 90 + [1] * 10)
    idx = select_attack_rows(y, 10, seed=42)
    assert (y[idx] == 1).all(), "adversarial examples must start from malicious flows"


def test_select_attack_rows_rejects_impossible_request():
    y = np.array([0] * 90 + [1] * 5)
    with pytest.raises(ValueError, match="only has 5"):
        select_attack_rows(y, 10, seed=42)


# --- attack contracts ------------------------------------------------------


@pytest.mark.parametrize("name", ["fgsm", "bim", "pgd", "deepfool"])
def test_attack_output_stays_in_the_feature_domain(classifier, source_rows, name):
    """clip_values=(0,1) must hold: phase 1 scaled every feature into that range,
    so a row outside it is not a flow the pipeline can represent."""
    _, clf = classifier
    X, y = source_rows
    params = {
        "fgsm": {"eps": EPS},
        "bim": {"eps": EPS, "eps_step": 0.01, "max_iter": 5},
        "pgd": {"eps": EPS, "eps_step": 0.01, "max_iter": 5},
        "deepfool": {"max_iter": 10, "epsilon": 1e-6, "nb_grads": 2},
    }[name]

    X_adv, _ = run_attack(build_attack(name, clf, params, batch_size=32), X, y)

    assert X_adv.dtype == np.float32
    assert X_adv.shape == X.shape
    assert X_adv.min() >= -1e-6 and X_adv.max() <= 1 + 1e-6


@pytest.mark.parametrize("name", EPS_ATTACKS)
def test_eps_budget_is_respected(classifier, source_rows, name):
    """The perturbation must stay inside the L-inf ball it was given. If it does
    not, every eps-vs-accuracy claim in the report is meaningless."""
    _, clf = classifier
    X, y = source_rows
    params = {"eps": EPS, "eps_step": 0.01, "max_iter": 5}
    X_adv, _ = run_attack(build_attack(name, clf, params, batch_size=32), X, y)
    linf, _, _ = perturbation_norms(X, X_adv)
    assert linf.max() <= EPS + 1e-5, f"{name} escaped its budget: {linf.max()}"


def test_larger_eps_perturbs_more(classifier, source_rows):
    """Monotonicity of the budget itself -- the precondition for the sweep."""
    _, clf = classifier
    X, y = source_rows
    norms = []
    for eps in (0.01, 0.1, 0.3):
        X_adv, _ = run_attack(build_attack("fgsm", clf, {"eps": eps}, 32), X, y)
        norms.append(perturbation_norms(X, X_adv)[1].mean())
    assert norms[0] < norms[1] < norms[2]


def test_eps_override_only_changes_eps(classifier):
    """A sweep point must differ from the canonical run in exactly one knob."""
    _, clf = classifier
    base = {"eps": 0.1, "eps_step": 0.01, "max_iter": 7}
    swept = build_attack("pgd", clf, base, 32, eps_override=0.3)
    assert swept.eps == 0.3
    assert swept.eps_step == 0.01 and swept.max_iter == 7
    assert base["eps"] == 0.1, "config dict must not be mutated in place"


def test_deepfool_rejects_an_eps_override(classifier):
    """DeepFool's epsilon is an overshoot multiplier, not a budget. Sweeping it
    would silently produce a meaningless curve, so it must fail loudly."""
    _, clf = classifier
    with pytest.raises(ValueError, match="no eps budget"):
        build_attack("deepfool", clf, {"max_iter": 5, "epsilon": 1e-6}, 32, eps_override=0.1)


def test_unknown_attack_name_is_rejected(classifier):
    _, clf = classifier
    with pytest.raises(ValueError, match="unknown attack"):
        build_attack("carlini", clf, {}, 32)


# --- wrapper contracts -----------------------------------------------------


def test_wrap_rejects_a_sigmoid_model():
    """A single sigmoid output gives DeepFool no class gradient to work with,
    which is why phase 2 builds 2-unit softmax models from the start."""
    inputs = keras.Input(shape=(N_FEATURES,))
    outputs = keras.layers.Dense(1, activation="sigmoid")(inputs)
    with pytest.raises(ValueError, match="expected 2"):
        wrap(keras.Model(inputs, outputs, name="sigmoid_model"), N_FEATURES)


def test_logits_view_preserves_every_prediction(classifier, source_rows):
    """DeepFool runs against the logit view for full attack strength. Softmax is
    monotone, so stripping it must leave argmax -- and therefore every reported
    number -- untouched. If this drifts, the two views are different models."""
    model, _ = classifier
    X, _ = source_rows
    logit_model = as_logits(model)
    assert np.array_equal(
        model.predict(X, verbose=0).argmax(axis=1),
        logit_model.predict(X, verbose=0).argmax(axis=1),
    )
    # And the logits really are unsquashed: probabilities cannot leave [0,1].
    assert logit_model.predict(X, verbose=0).min() < 0.0


def test_as_logits_rejects_a_model_that_has_no_softmax(classifier):
    model, _ = classifier
    with pytest.raises(ValueError, match="expected a softmax"):
        as_logits(as_logits(model))


def test_wrap_rejects_a_pre_reshaped_input():
    """CNN and LSTM must reshape inside the graph so all three models share one
    perturbation domain."""
    inputs = keras.Input(shape=(N_FEATURES, 1))
    x = keras.layers.Flatten()(inputs)
    outputs = keras.layers.Dense(2, activation="softmax")(x)
    with pytest.raises(ValueError, match="must reshape inside the graph"):
        wrap(keras.Model(inputs, outputs, name="seq_model"), N_FEATURES)


# --- reporting -------------------------------------------------------------


def test_summarise_uses_the_correct_denominator():
    """Evasion is measured over rows the model got right; accuracy is over all."""
    y = np.ones(4, dtype=np.int64)
    pred_clean = np.array([1, 1, 0, 1], dtype=np.int8)  # one already missed
    pred_adv = np.array([0, 1, 0, 0], dtype=np.int8)    # two of the three flipped
    z = np.zeros(4, dtype=np.float32)
    m = summarise(y, pred_clean, pred_adv, z, z, z.astype(np.int32))

    assert m["accuracy_clean"] == 0.75
    assert m["accuracy_adv"] == 0.25
    assert m["attack_success_rate"] == 0.75          # 3 of 4 read as benign
    assert m["evasion_rate_on_correct"] == pytest.approx(2 / 3)
    assert m["n_flipped"] == 2


def test_perturbation_norms_counts_touched_features():
    X = np.zeros((2, 4), dtype=np.float32)
    X_adv = np.array([[0.1, 0, 0, 0], [0.1, 0.2, -0.3, 0]], dtype=np.float32)
    linf, l2, n_changed = perturbation_norms(X, X_adv)
    assert np.array_equal(n_changed, [1, 3])
    assert linf == pytest.approx([0.1, 0.3])
    assert l2[1] == pytest.approx(np.sqrt(0.01 + 0.04 + 0.09))
