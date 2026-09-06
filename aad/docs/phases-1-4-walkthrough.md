# Phases 1–4: a complete technical walkthrough

Everything the project has done so far, in the order it was done, with the real
numbers and every decision that could have gone another way. Where a choice was
made, the reasoning is given; where a bug was found, what it was and how it
surfaced.

Companion document: [`phase03-04-explained.md`](phase03-04-explained.md) explains
the *concepts* from first principles (what an artefact is, softmax vs sigmoid, what
ε means). This document is the *record* of what was built and what it produced.

**Status:** Phases 1–4 complete on the full CSE-CIC-IDS2018 subset. 38 tests pass.
Phases 5–8 not started.

---

## Contents

1. [Environment](#1-environment)
2. [Phase 1 — data pipeline](#2-phase-1--data-pipeline)
3. [Phase 2 — baseline NIDS](#3-phase-2--baseline-nids)
4. [Phase 3 — AE generator](#4-phase-3--ae-generator)
5. [Phase 4 — analyzer and validator](#5-phase-4--analyzer-and-validator)
6. [Bugs found, and how](#6-bugs-found-and-how)
7. [Reproducibility](#7-reproducibility)
8. [Open decisions](#8-open-decisions)
9. [What phases 5–8 inherit](#9-what-phases-58-inherit)

---

## 1. Environment

| Component | Version | Why this one |
| --- | --- | --- |
| Python | 3.10.11 | The paper's exact version. TF 2.13 has no wheels for 3.12+ |
| TensorFlow | 2.13.1 | Matches the paper. CPU-only on native Windows (GPU support ended at 2.10) |
| ART | 1.17.1 | Supplies FGSM / BIM / PGD / DeepFool and the TF2 wrapper |
| numpy | 1.24.3 | Ceiling imposed by `tensorflow-intel` 2.13.1 |
| scikit-learn | 1.3.2 | MinMaxScaler now; the RF discriminator in phase 5 |

CPU-only is fine at this model size. Phase 2's full training run took 63 minutes.

### The dataset, and a fix to the documented download

Two capture files from CSE-CIC-IDS2018:

| File | Bytes | Contributes |
| --- | ---: | --- |
| `Friday-16-02-2018_TrafficForML_CICFlowMeter.csv` | 333,723,605 | DoS-Hulk, DoS-SlowHTTPTest, Benign |
| `Friday-23-02-2018_TrafficForML_CICFlowMeter.csv` | 382,840,456 | BruteForce-Web, BruteForce-XSS, SQL Injection, Benign |

**The README's S3 region was wrong.** It documented `eu-west-3`, which now returns
`301 PermanentRedirect`. The bucket has moved to **`ca-central-1`**. Anyone
following the setup instructions would have hit a hard failure. Fixed in the
README, together with a plain-HTTPS download that needs no AWS CLI and the
expected byte sizes so a truncated transfer is obvious.

---

## 2. Phase 1 — data pipeline

`scripts/01_prepare.py`, config in `config/data.yaml`. Runtime: **46.8 s**.

### 2.1 Three defects in the raw CSVs

Handled in `src/aad/data/loader.py`:

1. **An embedded header row.** `02-16` repeats its header partway through the
   file. Left in, it forces every column in that chunk to `object` dtype and
   silently poisons the numeric conversion. Detected by `Label == "Label"` and
   dropped — **1 row**.
2. **Literal `Infinity` strings.** `Flow Byts/s` and `Flow Pkts/s` hold the text
   `Infinity` wherever `Flow Duration == 0` (a division by zero inside
   CICFlowMeter). Coerced via `pd.to_numeric(errors="coerce")`.
3. **Inconsistent label spacing.** `"Brute Force -Web"` vs `"Brute Force-Web"`.
   Absorbed by slugifying to `bruteforceweb` before mapping.

Read in 200,000-row chunks — the files do not need to be held in memory at once.

### 2.2 Cleaning ledger

| Step | Rows | Detail |
| --- | ---: | --- |
| Ingested | 2,097,150 | both files |
| Embedded header removed | −1 | |
| Non-finite cells → NaN | 7,662 cells | `Flow Byts/s` 1,954, `Flow Pkts/s` 5,708 |
| Rows dropped for NaN | −5,708 | **all benign** — no attack row lost |
| Exact duplicates dropped | −594,987 | see below |
| **Remaining** | **1,496,454** | |

### 2.3 The deduplication decision — the one real judgement call

594,987 rows are exact duplicates, and they are wildly uneven across classes:

| Class | Raw | After dedup | Retained |
| --- | ---: | ---: | ---: |
| Benign | 1,494,781 | 1,350,659 | 90.4% |
| DoS-Hulk (Type1) | 461,912 | 145,199 | 31.4% |
| **DoS-SlowHTTPTest (Type2)** | **139,890** | **55** | **0.04%** |
| BruteForce-Web (Type3) | 362 | 340 | 93.9% |
| BruteForce-XSS (Type4) | 151 | 150 | 99.3% |
| SQL Injection (Type5) | 53 | 51 | 96.2% |

SlowHTTPTest produces near-identical CICFlowMeter vectors: 99.96% of it is literal
repetition of **55 distinct flows**.

This matters enormously. A leakage probe (test rows byte-identical to a training
row) shows:

| Variant | Overlap | Attack share |
| --- | ---: | ---: |
| `data/processed` (deduped) | **0 / 50,000** | 9.74% |
| `data/processed_paperparity` | **11,017 / 50,000** | 28.80% |

Keeping duplicates means 22% of test rows are *literally memorised from training*.
That is the mechanism behind the 99.98% accuracies this literature reports. We use
the deduplicated variant as primary and keep the parity variant to demonstrate the
effect.

### 2.4 Feature selection

78 candidate columns after dropping `Timestamp`. Ten have **zero variance on the
training split** and are removed — they carry no signal and, more importantly for
phase 3, no usable gradient:

```
Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags, CWE Flag Count,
Fwd Byts/b Avg, Fwd Pkts/b Avg, Fwd Blk Rate Avg,
Bwd Byts/b Avg, Bwd Pkts/b Avg, Bwd Blk Rate Avg
```

Constant-column detection runs on **train only**, so the feature set is not chosen
with any knowledge of val or test. **68 features remain.**

### 2.5 Split and scaling

Stratified 70/15/15 on the **6-way** label, not the binary one. Stratifying on
binary would let all 51 SQL-Injection rows land in one partition; stratifying on
the 6-way label guarantees each class appears in all three.

| Split | Rows | Attack | Benign |
| --- | ---: | ---: | ---: |
| train | 1,047,517 | 102,056 | 945,461 |
| val | 224,468 | 21,869 | 202,599 |
| test | 224,469 | **21,870** | 202,599 |

Per-class test counts: Hulk 21,780 · Web 51 · XSS 23 · SlowHTTPTest 9 · SQLi 7.

**MinMaxScaler fitted on train only**, then applied to val and test. Fitting on the
full frame before splitting is the single most common error in this literature and
silently inflates every reported number.

Val and test legitimately contain values outside the train min/max, which would
land outside `[0,1]`. Those are **clipped**, and the cost is recorded: 20 cells out
of 15,263,892 in test (0.0001%). Negligible, but it is a real effect — clipping a
value breaks any arithmetic identity it participated in, which phase 4's tests
pin down explicitly.

Why `[0,1]` at all: ε in FGSM/PGD is expressed in input units. Unscaled, ε=0.1
means "0.1 microseconds" on `Flow Duration` (nothing) and "0.1 packets" on
`Tot Fwd Pkts` (a large relative change). Scaling makes one ε mean the same thing
across all 68 features, and it is what makes ART's `clip_values=(0,1)` correct.

### 2.6 Outputs

```
data/processed/{train,val,test}.npz   X float32 [0,1], y binary, y_multiclass
data/processed/scaler.pkl             MinMaxScaler, train-fitted
data/processed/feature_names.json     the 68 retained names, in order
data/interim/combined.parquet         cleaned, pre-split, unscaled (162 MB)
```

---

## 3. Phase 2 — baseline NIDS

`scripts/02_train_baselines.py`, config in `config/models.yaml`. Runtime: **63 min**.

### 3.1 The architectural constraint that shapes everything downstream

All three models take a **flat `(68,)` input** and reshape *inside the graph*:

```python
# models/cnn.py and models/lstm.py
x = keras.layers.Reshape((n_features, 1), name="to_sequence")(inputs)
```

This is not cosmetic. It means all three models share one input space, so phase 3
perturbs them in the same 68-dimensional domain, one ε means the same thing for
all three, and one validator and one discriminator serve all of them.

**Every model ends in a 2-unit softmax**, never a 1-unit sigmoid. ART's attacks
differentiate a probability *vector*; DeepFool in particular needs a per-class
score to measure distance to each decision boundary. A single sigmoid output has
no second class to differentiate against. `wrappers.py` enforces this and raises
rather than wrapping a 1-unit model.

| Model | Architecture | Params |
| --- | --- | ---: |
| MLP | Dense 128 → 64 → 32, ReLU, dropout 0.2 | 19,234 |
| CNN | Conv1D 64×3 → MaxPool → Conv1D 128×3 → GlobalMaxPool → Dense 64 | 33,346 |
| LSTM | LSTM 64 → Dense 32 | 19,042 |

Trained with Adam (lr 1e-3), batch 1024, categorical cross-entropy over one-hot
labels, early stopping on `val_loss` (patience 4), `ReduceLROnPlateau` (factor 0.5,
patience 2). `class_weight` is off by default, matching the paper.

### 3.2 Results (clean test set, 224,469 rows)

| Model | Accuracy | Precision | Recall | F1 | Loss | FPR | Epochs | Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP | 99.9898% | 99.9817% | 99.9131% | 99.9474% | 0.00054 | 0.0020% | 20 | 54 s |
| CNN | 99.9795% | 99.9817% | 99.8080% | 99.8947% | 0.00087 | 0.0020% | 30 | 1,237 s |
| LSTM | 99.8387% | 99.8055% | 98.5368% | 99.1671% | 0.00930 | 0.0207% | 13 | 2,486 s |

All three clear the 99.5% target.

### 3.3 The LSTM discrepancy — worth a decision

Against Ayan's committed run: MLP reproduces to 4 decimal places, CNN to 3. **The
LSTM does not** — ours reaches 99.8387% (FPR 0.0207%) against theirs 99.6997%
(FPR 0.2685%), early-stopping at 13 epochs instead of running all 30.

Ours is *better*: a 13× lower false-positive rate. The cause is thread-scheduling
non-determinism in the recurrent kernel on CPU, which `keras.utils.set_random_seed`
does not pin. MLP and CNN reproducing bit-identically rules out any pipeline
difference.

Consequence: our LSTM's adversarial collapse in phase 3 is measured from a
different baseline than the paper's, so the paper's LSTM figure (99.80% → 4.89%)
is not directly comparable to ours. **Agree with Ayan which LSTM is canonical
before writing up.**

### 3.4 The web-attack blind spot

Headline accuracy conceals this completely:

| Class | Support | MLP | CNN | LSTM |
| --- | ---: | ---: | ---: | ---: |
| Benign | 202,599 | 99.998 | 99.998 | 99.979 |
| Type1 DoS-Hulk | 21,780 | 99.995 | 100.000 | 98.944 |
| Type2 DoS-SlowHTTPTest | 9 | 100.000 | 100.000 | **0.000** |
| Type3 BruteForce-Web | 51 | 66.667 | 49.020 | **0.000** |
| Type4 BruteForce-XSS | 23 | 95.652 | 39.130 | **0.000** |
| Type5 SQL-Injection | 7 | 100.000 | 71.429 | **0.000** |

The LSTM detects **none** of the four minority attack types. Because they total 90
rows out of 224,469, missing all of them costs 0.04% of accuracy. Ayan's run shows
the same pattern, so this is a property of the setup, not a regression.

Report this table beside the headline one — a single accuracy figure is actively
misleading here.

---

## 4. Phase 3 — AE generator

`scripts/03_generate_ae.py`, `src/aad/attacks/`, config in `config/attacks.yaml`.
Attack time: **12.2 min** for the 12 cells.

Threat model: **white-box, evasion, non-targeted**. The attacker holds the
architecture, weights and training data, and perturbs at inference time to flip
attack → benign.

### 4.1 Wrapping Keras for ART

```python
TensorFlowV2Classifier(
    model=model,
    nb_classes=2,
    input_shape=(68,),
    loss_object=keras.losses.CategoricalCrossentropy(),
    clip_values=(0.0, 1.0),
)
```

**The loss object is the setting that silently ruins results.** An attack ascends
the gradient *of a loss function*. Phase 2 trained under `CategoricalCrossentropy`;
handing ART anything else means the attack climbs a different surface than the one
training descended. It would still run, still produce plausible output, and be
measuring the wrong thing. `wrappers.py` uses the same loss `models/base.py`
compiles with.

`clip_values=(0,1)` is correct *because* phase 1 scaled and clipped to `[0,1]`. If
the two disagreed, ε would stop meaning "fraction of the feature's range".

### 4.2 A second wrapper, for DeepFool only

ART emitted a warning that turned out to matter:

> *It seems that the attacked model is predicting probabilities. DeepFool expects
> logits as model output to achieve its full attack strength.*

DeepFool navigates *distances to the decision boundary*. Softmax squashes those
distances into `[0,1]` and saturates, so DeepFool under-estimates how far it must
step and overshoots in compensation.

`as_logits()` rebuilds the model with the final softmax replaced by a linear layer
sharing the same trained weights, wrapped with `from_logits=True`. Softmax is
monotone, so **no prediction changes** — `argmax` is identical, verified by test.
Only the surface the attack navigates changes.

Measured effect on the MLP, same 100% evasion:

| | Mean L2 | L∞ |
| --- | ---: | ---: |
| DeepFool on softmax | 3.77 | 1.000 |
| DeepFool on logits | **0.85** | 0.746 |

A **4.4× smaller** perturbation for the same result. This also propagates into
phase 4: a smaller perturbation trips fewer domain rules, so leaving it unfixed
would have inflated the validator's catch rate against DeepFool.

FGSM, BIM and PGD keep the softmax view, so their gradients remain the ones phase 2
trained under.

### 4.3 The canonical rows

```python
pool = np.flatnonzero(y == 1)        # attack flows only
rng  = np.random.default_rng(42)
idx  = np.sort(rng.choice(pool, size=5000, replace=False))
```

One fixed, seeded set of **5,000 rows**, drawn only from `y == 1`, and **shared by
all twelve (attack × model) cells**. Fingerprint `idx.sum() = 557721064` is written
into the metrics file so a changed selection is detectable.

Three reasons the sharing matters: the twelve results become directly comparable
(only the attack differs); phase 5 can pair every adversarial row with its exact
clean original; and phase 4 can score catch-rate against an identical clean
baseline.

Only attack rows are used because an adversarial example is *a real attack
disguised as benign*. Perturbing a benign row has no attacker motive and no
measurable success criterion.

**Sample composition limitation.** Of 21,870 attack rows in test, 21,780 are
DoS-Hulk. The web-attack classes have 7–51 rows each, so stratifying across attack
types is impossible at 5,000 scale and the sample is ~99.6% Hulk. State this
rather than paper over it.

### 4.4 The four attacks

| Attack | Mechanism | Parameters |
| --- | --- | --- |
| **FGSM** | one step of size ε along `sign(∇ loss)` | ε=0.1 |
| **BIM** | FGSM iterated in small steps, re-clipped into the ε-ball each time | ε=0.1, step 0.01, 10 iters |
| **PGD** | BIM with more iterations; strongest first-order attack | ε=0.1, step 0.01, 40 iters, `num_random_init=0` |
| **DeepFool** | steps to the *nearest* decision boundary — minimal perturbation, no budget | 50 iters, overshoot 1e-6, `nb_grads=2`, `batch_size=512` |

Three ART defaults that had to be overridden:

- **`DeepFool.batch_size` defaults to 1.** Left alone, 5,000 sequential
  forward/backward passes per model.
- **`DeepFool.nb_grads` defaults to 10** ("top-10 classes"). We have two.
- **`DeepFool.epsilon` is an overshoot multiplier, not a budget.** It scales the
  final step by `(1+ε)` past the boundary. It is therefore excluded from the ε
  sweep, and `build_attack()` raises a `ValueError` if you try to sweep it —
  sweeping it would produce a curve that looks meaningful and is not.

**The true label is passed explicitly** to `generate()`. With `y=None`, ART
substitutes the model's own prediction, which makes results depend on model error
and quietly changes what "evasion" means.

### 4.5 Results

| Model | Clean | FGSM | BIM | PGD | DeepFool |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP | 99.88% | 0.00% | 0.00% | 0.00% | 0.12% |
| CNN | 99.76% | 0.00% | 0.00% | 0.00% | 0.24% |
| LSTM | 98.54% | 0.00% | 0.00% | 0.00% | 1.78% |

Detection rate on the 5,000 attack flows. Because every source row is a true
attack, **accuracy = recall = detection rate**, and **attack success rate = 1 −
accuracy**. Precision and FPR are undefined on an all-attack subset and are
deliberately not reported.

Against the paper (MLP 24.95%, CNN 49.76%, LSTM 4.89%), **our collapse is total**.
The defense therefore has a harder problem than the paper implies.

Perturbation cost, and the DeepFool result:

| Model | Attack | Mean L2 | Features moved (of 68) | Time |
| --- | --- | ---: | ---: | ---: |
| MLP | FGSM | 0.7118 | 57.0 | 0.1 s |
| MLP | PGD | 0.6229 | 46.3 | 3.2 s |
| MLP | **DeepFool** | **0.0434** | 57.0 | 2.2 s |
| CNN | **DeepFool** | **0.0357** | 30.0 | 52.5 s |
| LSTM | **DeepFool** | **0.0975** | 45.3 | 512.0 s |

**DeepFool achieves ~100% evasion at 16× smaller L2 than FGSM.** That is the single
most important number in this phase, and section 5.6 explains why.

### 4.6 The ε sweep

The brief specified ε over `{0.01, 0.05, 0.1, 0.3}`. On these models that grid is
almost entirely uninformative — every model is at 0% by ε=0.05, so three of the four
points sit flat on the floor. Values below 0.05 were added, keeping the original
four. Detection rate (%), 1,000-row subset of the same canonical rows:

| Model | Attack | 0.001 | 0.002 | 0.005 | 0.01 | 0.02 | 0.05 | 0.1 | 0.3 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP | FGSM | 99.9 | 99.9 | **8.0** | 2.6 | 0.0 | 0.0 | 0.0 | 0.0 |
| MLP | BIM | 99.9 | 99.9 | 7.9 | 2.4 | 0.0 | 0.0 | 0.0 | 0.0 |
| MLP | PGD | 99.9 | 99.9 | 7.9 | 2.4 | 0.0 | 0.0 | 0.0 | 0.0 |
| CNN | FGSM | 99.9 | 99.9 | 99.8 | 91.7 | **5.0** | 0.0 | 0.0 | 0.0 |
| CNN | BIM | 99.9 | 99.9 | 99.8 | 91.1 | 4.9 | 0.0 | 0.0 | 0.0 |
| CNN | PGD | 99.9 | 99.9 | 99.8 | 91.1 | 4.9 | 0.0 | 0.0 | 0.0 |
| LSTM | FGSM | 98.2 | 98.2 | 98.2 | 92.9 | **1.8** | 0.0 | 0.0 | 0.0 |
| LSTM | BIM | 98.2 | 98.2 | 98.1 | 84.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| LSTM | PGD | 98.2 | 98.2 | 98.1 | 84.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Verified **monotone non-increasing** in ε for every row — no gradient masking.

**Robustness ordering: CNN > LSTM > MLP.** The MLP breaks between ε=0.002 and
0.005; CNN and LSTM hold to ε=0.01 and break by 0.02 — roughly a **4× larger
budget**. All three are cliffs, not gradual decay.

This ordering contradicts the paper's, which has the LSTM most fragile and the MLP
in the middle. Only "CNN most robust" agrees. And none of it is visible at the
paper's single operating point, where everything is simply zero.

### 4.7 The artefact schema

Each of `artifacts/adversarial/{attack}_{model}.npz` (12 files, 8.0 MB total):

| Key | Purpose |
| --- | --- |
| `X_adv` | the perturbed flows |
| `X_clean` | the originals, same order — phase 5 needs exact pairs; phase 4 needs the baseline |
| `idx` | row indices into `test.npz`, so any phase can re-derive |
| `y_true`, `y_multiclass` | true label (all attack) and attack type |
| `pred_clean`, `pred_adv` | before/after predictions — the honest evasion denominator |
| `prob_adv` | full probability vectors |
| `linf`, `l2`, `n_changed` | per-row perturbation size and sparsity |
| `meta` | JSON: params, seed, ART/TF versions, elapsed time, metrics |

Storing `X_clean` duplicates data deliberately: it makes each file self-contained
so phase 5 cannot pair rows against a differently-ordered split, and it is what
makes phase 4's `attributable` metric possible at all.

Two assertions run after every cell and abort the run on violation — both check
things that would invalidate every downstream phase while still writing files that
look fine:

```python
# 1. did clip_values actually apply?
X_adv.min() >= -1e-6 and X_adv.max() <= 1 + 1e-6
# 2. did the attack honour its budget? (eps-attacks only)
linf_max <= eps + 1e-5
```

The script is **resumable** — a cell whose `.npz` exists is skipped unless
`--force`, and figure generation is wrapped in `try/except` so a plotting bug
cannot destroy 12 minutes of attack computation.

---

## 5. Phase 4 — analyzer and validator

`scripts/04_analyze_validate.py`, `src/aad/defense/`.

### 5.1 The premise

The IDS never sees packets — it sees 68 *statistics* computed from packets. A
gradient attack treats those as free variables, but they were derived from the same
packets and must agree with one another. Move them independently and you describe a
flow that could not exist: 7.4 packets, a minimum above a maximum, a byte rate
contradicting its own duration.

The validator checks **arithmetic, not learned behaviour**, so it needs no
adversarial training data and has nothing to overfit.

### 5.2 Separation of concerns

- `analyzer.py` reads **benign training** flows and writes `rules.json`.
- `validator.py` reads `rules.json` and checks rows — pure numpy.

The analyzer never sees adversarial examples (so the validator cannot be tuned to
the attacks it will face) and never sees val or test (so the FPR is measured on
rows the rules were not fitted to).

**The residual arithmetic lives in `validator.py`, and `analyzer.py` imports it**
to calibrate. Two separate implementations of "how far is this rule violated?"
could drift, and every threshold would then gate something other than what it was
calibrated against.

### 5.3 Raw units

Checks run in original units (packets, bytes, microseconds), not scaled `[0,1]`.
After MinMax scaling "is this a whole number?" becomes "does this sit on a lattice
of spacing `1/(max−min)`?" — checkable but unreadable and fragile.

`data_min` and `data_range` are copied out of `scaler.pkl` **into `rules.json`**, so
the validator needs no scikit-learn at inference and is cheap enough to sit in front
of every flow.

### 5.4 The four rule families

**Range** — per-feature `[min, max]` over benign traffic, padded by 1% of span. The
padding is not cosmetic: benign val/test rows legitimately fall slightly outside
whatever train happened to contain.

**Integrality — 25 columns.** Discovered empirically (`|v − round(v)| ≤ tol` for
*every* benign row), never hardcoded:

```
Dst Port, Protocol, Tot Fwd Pkts, Tot Bwd Pkts, Fwd Pkt Len Max, Fwd Pkt Len Min,
Bwd Pkt Len Max, Bwd Pkt Len Min, Fwd PSH Flags, Pkt Len Min, Pkt Len Max,
FIN/SYN/RST/PSH/ACK/URG/ECE Flag Cnt, Down/Up Ratio, Subflow Fwd Pkts,
Subflow Bwd Pkts, Init Fwd Win Byts, Init Bwd Win Byts, Fwd Act Data Pkts,
Fwd Seg Size Min
```

A further **34 columns look integral but are excluded as unverifiable** at float32
precision — see [§6.3](#63-vacuous-integrality-rules). They are listed in
`rules.json` rather than silently dropped.

**Dependency — 14 invariants**, each with a tolerance calibrated from its own benign
residual distribution:

| Kind | Count | Example | Typical tolerance |
| --- | ---: | --- | ---: |
| Ordering | 8 | `Fwd Pkt Len Min ≤ Mean ≤ Max` | 1e-6 – 1.9e-4 |
| Product | 2 | `TotLen Fwd Pkts ≈ Tot Fwd Pkts × Fwd Pkt Len Mean` | 1.2e-6 |
| Rate | 4 | `Flow Pkts/s ≈ (Fwd+Bwd pkts) / (Duration/1e6)` | 1.2e-6 |

Residuals are **relative**, `|actual − expected| / (|expected| + 1)`, because
`Flow Duration` runs to 1e8 while `Down/Up Ratio` is single digits — one absolute
tolerance cannot serve both.

**Distribution** — robust outlier score:

```python
z = |raw − median| / MAD     # median absolute deviation, not std
z = clip(z, 0, 50)           # no single feature may dominate
score = z.mean(axis=1)
```

Median/MAD because flow features are heavy-tailed enough that a standard deviation
is dominated by the very tail it should detect. Mean-of-clipped rather than max —
see [§6.2](#62-a-silently-dead-rule-family). Threshold **10.1424**, calibrated on
benign *validation* rows at a 1% budget.

### 5.5 Results

**False-positive rate on 202,599 unseen benign test flows: 0.948%** — well under the
3% usability line.

| Family | Benign FPR |
| --- | ---: |
| range | 0.0005% |
| integrality | 0.0000% |
| dependency | 0.0015% |
| distribution | 0.9472% |
| **combined** | **0.9482%** |

Almost all of it is the distribution family spending its deliberate budget. The
three arithmetic families cost essentially nothing — they describe how CICFlowMeter
computes, so genuine traffic satisfies them.

> Note: the 1% budget is spent by the distribution family *alone*, and the four
> families are OR-ed, so the combined FPR is roughly their sum. Tighten
> `--fpr-budget` if the combined figure must sit under a hard limit.

**Catch rate: 100% on all twelve cells.** Per family:

| Cell | range | integrality | dependency | distribution |
| --- | ---: | ---: | ---: | ---: |
| FGSM × MLP/CNN/LSTM | 0.00–0.02% | 100% | 100% | 100% |
| BIM × MLP | 0.00% | 100% | 100% | 100% |
| BIM × CNN | 0.02% | 100% | 100% | **17.6%** |
| BIM × LSTM | 0.02% | 100% | 100% | 99.9% |
| PGD × MLP/LSTM | 0.00–0.02% | 100% | 100% | 99.9–100% |
| PGD × CNN | 0.02% | 100% | 100% | 95.0% |
| **DeepFool × MLP** | 0.02% | 100% | 100% | **1.2%** |
| **DeepFool × CNN** | 0.02% | 100% | 100% | **1.0%** |
| **DeepFool × LSTM** | 0.02% | 100% | 100% | **2.3%** |

Clean attack rows are rejected only **0.091%**, so **`attributable` = 99.88%** on
every cell. The gate genuinely detects *perturbation*, not *maliciousness* — see
[§6.4](#64-the-catch-rate-that-measures-the-wrong-thing) for why that column exists.

### 5.6 The finding that sets up phase 8

Read the DeepFool rows above. Its perturbation is 16× smaller than FGSM's, small
enough to stay **inside the benign distribution** — so the distribution family
catches it only 1.0–2.3%, against 100% for FGSM.

DeepFool is stopped **purely by arithmetic**: integrality and dependency.

Both of those are exactly what an adaptive attacker can satisfy deliberately. An
adversary that

1. uses DeepFool-style minimal perturbation (already evades distribution),
2. rounds the perturbation onto the integer lattice (defeats integrality),
3. re-derives the dependent features from the rounded values (defeats dependency),

passes **every gate**. The range family is already contributing ~0.02%.

This is no longer a hypothesis borrowed from Carlini & Wagner — it is visible in
our own per-family numbers. It is the strongest argument in the project, and it is
the reason the per-family breakdown is reported rather than a single number.

---

## 6. Bugs found, and how

All four surfaced from *running* the code, not from reading it. Three were in the
validator, one in the attack wrapper.

### 6.1 Integrality tolerance ignored float32

Phase 1 stores `X` as float32, so recovering raw units costs about `range × 2⁻²⁴`:

| Column | Span | Round-trip error |
| --- | ---: | ---: |
| `SYN Flag Cnt` | 2 | 2.4e-08 |
| `Tot Fwd Pkts` | 198 | 2.4e-06 |
| `Init Fwd Win Byts` | 65,535 | 7.8e-04 |
| `Flow Duration` | 1.2e08 | **1.43** |

A flat `1e-6` tolerance would call `Flow Duration = 12345678.6` fractional on
*genuine benign traffic* and reject nearly everything. Now scaled per column, with
~16× headroom over float32 noise while staying ~5 orders of magnitude below what an
ε=0.1 perturbation produces. **Found by:** a test asserting benign rows pass.

### 6.2 A silently dead rule family

The distribution score originally used `z.max(axis=1)` — a row scored by its single
worst feature. Many flow columns are almost always zero (the flag counts), so their
MAD sits on its floor and any nonzero value pegs `z` at the ceiling. That happened
for more than 1% of benign rows, so the calibrated 99th-percentile threshold *was*
the ceiling, and `score > threshold` was never true.

The family reported a clean column of zeroes and detected nothing. It did not
crash, and no test caught it until one asserted the threshold was below the cap.

Fixed by averaging clipped z-scores, which also better matches the threat — a
gradient attack nudges ~65 of 68 features a little rather than one a lot. Now
spends 0.947% against its 1% budget. **Found by:** noticing a column of exact
`0.000%` in the first real run.

### 6.3 Vacuous integrality rules

After fixing §6.1, the per-column tolerance scaled with range — and on wide columns
grew past **0.5**, at which point every real number is within tolerance of an
integer and the check can never fail.

On real data this declared **59 of 68 columns "integral"**, including `Flow Byts/s`
(tolerance **850**), `Flow IAT Std` and `Flow Duration` — none of which are counts.
Detection was unaffected (the genuinely integral columns did the work), but
`rules.json` asserted something false, and the write-up would have repeated it.

Now capped at 0.25; columns exceeding it are recorded as
`excluded_unverifiable`. The honest set is **25 columns**, and catch rate is
unchanged at 100%. **Found by:** reading the discovered column list and noticing
rates and standard deviations in it.

### 6.4 The catch rate that measures the wrong thing

Every adversarial row is built from a real attack flow. Real attack flows can
already sit outside the benign envelope the rules were mined from — that is what
makes them attacks. So a raw catch rate partly measures *maliciousness* rather than
*perturbation*.

On the synthetic fixture used during development this was stark: the range family
rejected **100% of clean, unperturbed attack rows**, and the gate would have
reported a triumphant "100% catch rate" while contributing nothing as an
adversarial-example detector.

The report now carries three columns — `catch_%`, `clean_orig_%` (the same rows
unperturbed) and `attributable_%` (rejected only *after* perturbation). On real
data `clean_orig` is only 0.12%, so `attributable` is 99.88% and the gate is
genuinely doing its job — but that could not be known without measuring it.

**This is also the caution to apply to the paper's own 100% discriminator
accuracy.** Always ask what the baseline was.

### 6.5 A plotting bug worth mentioning

BIM and PGD produce near-identical curves (PGD is BIM with more iterations), so on
Figure 4 one was drawn exactly on top of the other and appeared missing entirely.
Fixed with distinct dash patterns and marker shapes — colour alone is not enough
when series coincide.

---

## 7. Reproducibility

**Phase 1 reproduces Ayan's committed `prepare_meta.json` exactly.** Every count,
split size, clipping statistic and constant-column name is identical; the only diff
was `elapsed_seconds` (46.8 vs 130.6). Phase 2 reproduces MLP and CNN
bit-identically; the LSTM does not (§3.3).

Seeding covers python, numpy and TensorFlow via `keras.utils.set_random_seed(42)`,
plus a separate seeded `default_rng` for the phase-3 row sampler. `num_random_init=0`
keeps PGD deterministic.

Everything downstream of phase 1 reads from disk rather than memory, so any phase
can be re-run alone. Phase 3 caches per cell.

### Test suite

**38 tests, ~12 s, no dataset required** — synthetic fixtures only. They check the
wiring rather than the algorithms:

- ε budgets are honoured; `clip_values` is applied; output is float32
- the row sampler is reproducible and draws only attack rows
- DeepFool refuses an ε override; unknown attack names raise
- the wrapper rejects a sigmoid model and a pre-reshaped input
- the logits view preserves every prediction
- benign traffic passes within budget; hand-built violations are caught
- **regression guards** for §6.1, §6.2 and §6.3

### Full rebuild

```powershell
python scripts\01_prepare.py           # ~47 s
python scripts\02_train_baselines.py   # ~63 min
python scripts\03_generate_ae.py       # ~15 min
python scripts\04_analyze_validate.py  # ~40 s
python -m pytest tests\ -q             # ~12 s
```

---

## 8. Open decisions

1. **Which LSTM is canonical** — ours (99.8387%, FPR 0.021%) or Ayan's (99.6997%,
   FPR 0.269%)? Phase 3's collapse is measured from that baseline, and phases 6–7
   will be too.
2. **The ε grid.** Values `{0.001, 0.002, 0.005, 0.02}` were added to the specified
   four because the original grid put three of four points flat at zero. Revert if
   strict adherence to the brief is wanted.
3. **The FPR budget.** 1% for the distribution family gives 0.948% combined. If a
   hard limit applies, tighten `--fpr-budget`.
4. **DoS-Hulk dominance.** The canonical 5,000 rows are ~99.6% Hulk because the
   other classes have 7–51 test rows. Nothing can be done at this scale; it needs
   stating as a limitation.

---

## 9. What phases 5–8 inherit

| Phase | Consumes | Note |
| --- | --- | --- |
| 5 — discriminator | `X_adv` + `X_clean` from the 12 `.npz` | Pairs are exact. **Split by attack family, not randomly** — hold out DeepFool and report both numbers |
| 6 — EIDS | `X_adv` with `y_true` (still attack) | Hold out AEs from the *other* two models to measure cross-model transferability |
| 7 — pipeline | `Validator.check()` returns per-family masks | Gate attribution is already available; ablation needs `families=(...)` |
| 8 — adaptive attacker | §5.6 | The bypass is already characterised: minimal perturbation + integer rounding + re-derived dependents |

A caution for phase 5: the discriminator's task is clean vs. perturbed. Given that
integrality alone separates the two classes perfectly on this data, a Random Forest
will trivially reach ~100% — and that number will mean as little as the paper's.
The leave-one-attack-out split is the honest measurement.
