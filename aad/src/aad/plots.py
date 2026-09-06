"""Figures, shared across phases.

Kept out of the phase scripts because phases 5-7 need the same styling, and
because a plotting bug should never be able to destroy an expensive run: every
function here reads a saved report file, never live state.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on a headless/CI run
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

log = logging.getLogger(__name__)

MODEL_ORDER = ["mlp", "cnn", "lstm"]
ATTACK_ORDER = ["fgsm", "bim", "pgd", "deepfool"]

# Colour-blind-safe, and distinguishable in greyscale print.
ATTACK_COLORS = {
    "fgsm": "#4C72B0",
    "bim": "#DD8452",
    "pgd": "#55A868",
    "deepfool": "#C44E52",
}
CLEAN_COLOR = "#8C8C8C"

# BIM and PGD frequently agree to the decimal -- PGD is BIM with more iterations
# -- so on a line chart one is drawn exactly on top of the other and appears
# missing. Distinct dash patterns and marker shapes keep every series readable
# where they coincide; colour alone is not enough.
ATTACK_STYLES = {
    "fgsm": {"linestyle": "-", "marker": "o", "linewidth": 1.8},
    "bim": {"linestyle": "--", "marker": "s", "linewidth": 2.6},
    "pgd": {"linestyle": ":", "marker": "^", "linewidth": 1.8},
    "deepfool": {"linestyle": "-.", "marker": "D", "linewidth": 1.8},
}


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)


def figure3_evasion(metrics: dict, out_path: Path) -> Path:
    """Figure 3 - clean vs. adversarial accuracy, grouped by model.

    The paper's headline result: three models above 99.7% on clean traffic,
    collapsing under perturbation. One grey bar per model for the clean
    baseline, then one coloured bar per attack.
    """
    results = metrics["results"]
    models = [m for m in MODEL_ORDER if m in results]
    attacks = [a for a in ATTACK_ORDER if any(a in results[m] for m in models)]

    n_bars = len(attacks) + 1
    width = 0.8 / n_bars
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(9, 4.8))

    clean = [100 * results[m][attacks[0]]["accuracy_clean"] for m in models]
    ax.bar(x - 0.4 + width / 2, clean, width, label="clean", color=CLEAN_COLOR)

    for i, attack in enumerate(attacks, start=1):
        vals = [
            100 * results[m][attack]["accuracy_adv"] if attack in results[m] else np.nan
            for m in models
        ]
        ax.bar(
            x - 0.4 + width * (i + 0.5),
            vals,
            width,
            label=attack.upper(),
            color=ATTACK_COLORS.get(attack),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.set_ylabel("Detection rate on attack flows (%)")
    ax.set_ylim(0, 105)
    ax.set_title(
        "Figure 3 - baseline NIDS accuracy under white-box evasion\n"
        f"{metrics['n_samples']:,} true-attack test flows per cell",
        fontsize=11,
    )
    ax.legend(frameon=False, ncol=n_bars, loc="upper center", bbox_to_anchor=(0.5, -0.09))
    _style(ax)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)
    return out_path


def figure4_eps_sweep(sweep: pd.DataFrame, out_path: Path) -> Path:
    """Figure 4 - accuracy vs. eps, one panel per model.

    The result the paper does not have. A single operating point cannot
    distinguish a model that degrades gracefully from one that falls off a
    cliff, and a non-monotone curve is evidence of gradient masking.
    """
    models = [m for m in MODEL_ORDER if m in set(sweep["model"])]
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4.2), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, model in zip(axes, models):
        sub = sweep[sweep["model"] == model]
        for attack in [a for a in ATTACK_ORDER if a in set(sub["attack"])]:
            s = sub[sub["attack"] == attack].sort_values("eps")
            ax.plot(
                s["eps"],
                100 * s["accuracy_adv"],
                markersize=5.5,
                markerfacecolor="none",
                label=attack.upper(),
                color=ATTACK_COLORS.get(attack),
                **ATTACK_STYLES.get(attack, {"linestyle": "-", "marker": "o"}),
            )
        ax.set_xscale("log")
        ax.set_xlabel(r"$\epsilon$  (L$_\infty$ budget, fraction of feature range)")
        ax.set_title(model.upper(), fontsize=11)
        _style(ax)

    axes[0].set_ylabel("Detection rate on attack flows (%)")
    axes[0].set_ylim(-3, 105)
    axes[-1].legend(frameon=False, title="attack")
    fig.suptitle("Figure 4 - robustness vs. perturbation budget", fontsize=12)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)
    return out_path
