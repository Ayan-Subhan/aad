"""Keras -> ART bridge for the three baseline NIDS models.

One function does the real work. Everything here exists to make sure the wrapper
agrees with how the models were built and trained in phase 2, because every way
of getting this wrong produces plausible-looking numbers rather than an error:

* **Same loss object.** ``models/base.py:compile_model`` trains under
  ``CategoricalCrossentropy`` over a 2-unit softmax. ART differentiates whatever
  loss it is handed, so handing it a different one (or logits-vs-probabilities
  mismatch via ``from_logits``) silently produces attack gradients that are not
  the training gradients.
* **Same input space.** All three models take a flat ``(n_features,)`` input and
  reshape internally, so ``input_shape`` is the same for MLP, CNN and LSTM and
  the perturbation lives in one 68-dimensional domain for every model.
* **Same value range.** Phase 1 MinMax-scaled and clipped features to [0,1], so
  ``clip_values=(0,1)`` is the true domain boundary, not an arbitrary choice.
  If the two disagree, eps stops meaning "fraction of the feature's range".
"""

from __future__ import annotations

import logging
from pathlib import Path

from tensorflow import keras

from art.estimators.classification import TensorFlowV2Classifier

from ..models.base import N_CLASSES

log = logging.getLogger(__name__)

# Phase 1 scaled every feature into this range and clipped the overflow.
CLIP_VALUES = (0.0, 1.0)


def as_logits(model: keras.Model) -> keras.Model:
    """Return the same model with its final softmax removed.

    DeepFool works by estimating the *distance to each decision boundary* and
    stepping just far enough to cross the nearest one. Softmax squashes those
    distances into [0,1] and saturates, so the distances DeepFool measures are
    compressed and it under-steps -- ART warns about exactly this. On logits the
    geometry is linear and the estimate is faithful.

    Removing the softmax does not change a single prediction: softmax is
    monotone, so ``argmax`` is identical either way. It only changes the surface
    the attack navigates. The new Dense layer shares the trained weights.
    """
    last = model.layers[-1]
    if last.activation is not keras.activations.softmax:
        raise ValueError(f"expected a softmax output layer, found {last.activation}")

    logits = keras.layers.Dense(last.units, activation=None, name="logits")(last.input)
    logit_model = keras.Model(model.input, logits, name=f"{model.name}_logits")
    logit_model.get_layer("logits").set_weights(last.get_weights())
    return logit_model


def wrap(
    model: keras.Model, n_features: int, from_logits: bool = False
) -> TensorFlowV2Classifier:
    """Wrap a compiled Keras model as an ART white-box classifier.

    ``from_logits`` selects the loss variant, and must match whether ``model``
    ends in a softmax or emits raw logits. Getting this pair wrong is silent:
    the attack still runs and still produces output, it just climbs the wrong
    surface.
    """
    out_shape = model.output_shape[-1]
    if out_shape != N_CLASSES:
        raise ValueError(
            f"{model.name} emits {out_shape} units, expected {N_CLASSES}. ART's "
            "attacks need a probability vector; a single sigmoid gives DeepFool "
            "nothing to work with."
        )
    if model.input_shape[1:] != (n_features,):
        raise ValueError(
            f"{model.name} takes {model.input_shape[1:]}, expected ({n_features},). "
            "CNN and LSTM must reshape inside the graph, not at the caller."
        )

    return TensorFlowV2Classifier(
        model=model,
        nb_classes=N_CLASSES,
        input_shape=(n_features,),
        # Matches what phase 2 compiled with (from_logits=False on a softmax).
        loss_object=keras.losses.CategoricalCrossentropy(from_logits=from_logits),
        clip_values=CLIP_VALUES,
    )


def load_wrapped(
    models_dir: Path, name: str, n_features: int, tag: str = ""
) -> tuple[keras.Model, TensorFlowV2Classifier, TensorFlowV2Classifier]:
    """Load ``{name}{tag}.keras`` and wrap it two ways.

    Returns ``(keras_model, softmax_classifier, logit_classifier)``:

    * the raw Keras model, for fast batched ``predict`` calls needing no gradient;
    * the softmax classifier, used by FGSM/BIM/PGD so their gradients are the
      ones phase 2 trained under;
    * the logit classifier, used by DeepFool alone -- see ``as_logits``.

    Both classifiers wrap the same weights and make identical predictions.
    """
    path = models_dir / f"{name}{tag}.keras"
    if not path.exists():
        raise FileNotFoundError(
            f"no trained model at {path}. Run scripts/02_train_baselines.py first."
        )
    model = keras.models.load_model(path)
    log.info("loaded %s (%d params) from %s", name, model.count_params(), path.name)
    return (
        model,
        wrap(model, n_features),
        wrap(as_logits(model), n_features, from_logits=True),
    )
