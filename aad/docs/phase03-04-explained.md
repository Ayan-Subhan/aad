# Phases 3 & 4 explained, from the ground up

This document explains every decision in the AE generator (phase 3) and the
inbound analyzer + validator (phase 4): what each term means, why each choice was
made, and what could go wrong if it were made differently.

Read the glossary first if any of the vocabulary is unfamiliar — the rest of the
document assumes it.

---

## Part 0 — Glossary

### Artefact

In this project, an **artefact is simply a file written to disk** by one phase and
read back by a later one. Nothing more mysterious than that.

The discipline is: *every stage writes its output to a file, and every downstream
stage reads from that file rather than from memory.* Phase 3 does not hand its
adversarial examples to phase 5 as a Python variable; it writes
`artifacts/adversarial/pgd_lstm.npz`, and phase 5 opens that file.

Why bother? Because generating the adversarial examples takes hours. If phase 3
and phase 5 were one long script, a typo in the phase-5 plotting code would
destroy hours of attack computation. With artefacts, you re-run phase 5 in
seconds against the file phase 3 already produced.

The word is also used in the research sense — "the artefacts of this project" =
the concrete outputs you can point at: the 12 `.npz` files, `rules.json`, the
figures, the tables. Same idea.

> The repo has an `artifacts/` directory (American spelling) that holds exactly
> these. `data/` holds inputs; `artifacts/` holds everything the code produced.

### Sigmoid vs. softmax (and why the models use softmax)

Both turn a network's raw output numbers into something you can read as a
probability. They differ in shape:

| | Sigmoid | Softmax |
| --- | --- | --- |
| Output units | **1** number | **2** numbers (one per class) |
| Reads as | "P(attack) = 0.87" | "P(benign) = 0.13, P(attack) = 0.87" |
| Sum | n/a | always exactly 1.0 |

For plain binary classification they are equivalent — `sigmoid = 0.87` and
`softmax = [0.13, 0.87]` say the same thing. **But not for us.**

The attacks need to compute a *gradient*: "if I nudge feature 12 upward, how does
the score for class *benign* change?" DeepFool in particular works by asking, for
every class, how far the input is from that class's decision boundary, and then
stepping toward the nearest one. With a single sigmoid output there is no
per-class score to differentiate — there is one number, and the "benign" class
exists only implicitly as `1 − p`. ART's `class_gradient` has nothing to return.

So `src/aad/models/base.py` fixes `N_CLASSES = 2` and
`OUTPUT_ACTIVATION = "softmax"`, and all three models end with:

```python
keras.layers.Dense(2, activation="softmax", name="softmax")
```

This had to be decided in **phase 2**, when the models were built. Converting a
trained sigmoid model to softmax afterwards is possible but fiddly and
error-prone, which is why the phase-2 config comments call the choice
"deliberate". `wrappers.py` enforces it — it raises rather than wrapping a
1-unit model, so the mistake surfaces as an error instead of as strange numbers.

### Canonical rows

"Canonical" here just means **the one official, agreed-upon version** — as opposed
to a variant, a subset, or a one-off.

Concretely: we draw **5,000 specific rows** from the test set — always the same
5,000, chosen once by a seeded random number generator. Those are the *canonical
rows*. Every one of the 12 (attack × model) combinations perturbs **that same
set**.

```
                        the canonical 5,000 rows
                                  |
        +-------------+-----------+-----------+-------------+
        |             |           |           |             |
     FGSM×MLP     BIM×MLP     PGD×CNN    DeepFool×LSTM   ... (12 total)
```

Three reasons this matters:

1. **Comparability.** If FGSM×MLP used different rows from PGD×MLP, you could not
   tell whether PGD looks stronger because it *is* stronger or because it got
   easier rows. Same rows = the only difference is the attack.
2. **Pairing.** Phase 5's discriminator learns "clean vs. perturbed". It needs
   every adversarial row matched to *its own* clean original. Shared rows make
   that pairing exact.
3. **Fair validator scoring.** Phase 4 measures its false-positive rate on clean
   rows and its catch-rate on adversarial ones. If those were different
   populations the comparison would be meaningless.

The rows are chosen only from flows where `y == 1` (genuine attacks), because an
adversarial example is *a real attack disguised as benign traffic*. Perturbing a
benign row would have no attacker motive and no measurable success criterion.

> **Honest limitation.** The test split has 21,870 attack rows, of which 21,780
> are DoS-Hulk. The web-attack classes have 7–51 rows each. So the canonical 5,000
> will be ~99.6% DoS-Hulk, and stratifying across attack types is simply not
> possible at this scale. State this in the write-up rather than papering over it.

### Epsilon (ε) — the perturbation budget

ε is **how much the attacker is allowed to change each feature**.

Phase 1 rescaled every feature into the range `[0, 1]` (MinMax scaling), so a
feature's full observed span is exactly 1.0. Therefore:

> **ε = 0.1 means "you may move any feature by at most 10% of its observed range."**

This is measured in the **L∞ norm** ("L-infinity"), which is just *the largest
single change across all 68 features*. If an attack moves feature 3 by 0.08,
feature 40 by 0.10, and everything else by less, the L∞ distance is 0.10.

ε is the whole point of adversarial ML: a *small* ε means the change is subtle
and arguably undetectable; a large ε means you have mangled the traffic beyond
recognition. Reporting an evasion rate without its ε is meaningless — you could
always achieve 100% evasion by setting ε = 1.

This is why phase 1's MinMax scaling was "not optional": if features were left in
raw units, ε = 0.1 would mean "0.1 microseconds" on `Flow Duration` (nothing at
all) and "0.1 packets" on `Tot Fwd Pkts` (a huge relative change). One number
cannot serve both. Scaling to [0,1] makes ε mean the same thing everywhere.

### The sweep

A **sweep** is running the same experiment repeatedly while varying one knob, to
get a *curve* instead of a single point.

Here we sweep ε over `{0.01, 0.05, 0.1, 0.3}` and record accuracy at each. The
paper reports one number ("the MLP drops to 24.95%"), which cannot distinguish
between two very different situations:

```
  100% |*                             100% |*----*
       | \                                 |      \
       |  \                                |       \
       |   *----*----*                     |        *----*
    0% +--------------- eps             0% +--------------- eps
       0.01        0.3                     0.01        0.3

  (a) collapses immediately              (b) degrades gracefully
      -- fragile at any budget               -- robust until a threshold
```

Both could report "24.95% at ε=0.1". They are not the same model, and the curve
is the stronger result. A **non-monotone** curve (accuracy going *up* as ε rises)
is itself a finding: it means the gradients are unreliable — "gradient masking" —
and the attack is failing for numerical reasons rather than the model being
robust. Athalye et al. built a whole paper on that.

**Cost note.** The sweep is 3 attacks × 3 models × 4 ε = 36 extra runs. So the 12
canonical artefacts use the full 5,000 rows, and the sweep uses a fixed 1,000-row
subset of *those same rows*. The curve's shape is unaffected; compute drops ~5×.

### White-box, evasion, non-targeted

The **threat model** — the assumptions about what the attacker knows and wants.
Every design choice follows from it.

- **White-box** — the attacker has the architecture, the weights, and the training
  data. This is the *strongest* assumption, which makes it the *safest* one for a
  defender: if you survive white-box, you survive less-informed attackers too.
  (Grey-box = partial knowledge; black-box = query access only.)
- **Evasion** — the attacker modifies input *at inference time*. Contrast with
  **poisoning**, where the attacker corrupts the training data. We do not do
  poisoning.
- **Non-targeted** — the attacker wants "anything but the correct answer". Since
  we have two classes, this collapses to "make the attack look benign", which is
  exactly the attacker's real goal.

### Other terms in one line each

| Term | Meaning |
| --- | --- |
| **AE** | Adversarial Example — a perturbed input crafted to fool a model |
| **Gradient** | Which direction to move an input to change the model's output most |
| **Evasion rate** | Fraction of attacks that got past the model |
| **FPR** | False-Positive Rate — legitimate traffic wrongly rejected |
| **Attack success rate** | Fraction of adversarial rows the model called benign |
| **`.npz`** | NumPy's zip container: several named arrays in one file |
| **ART** | Adversarial Robustness Toolbox, IBM's attack library |
| **Estimator / classifier (ART)** | ART's wrapper around your model, giving it one API |
| **Calibration** | Setting a threshold from data instead of guessing it |
| **Ablation** | Turning components off one at a time to see what each contributes |

---

## Part 1 — Phase 3: the AE generator

### What it does, in one paragraph

Take 5,000 real attack flows that the models correctly detect. For each of the 3
models and 4 attacks, use the model's own gradients to compute a tiny change to
each flow that makes the model call it benign. Save the results. Measure how far
detection fell.

### Step 1 — Wrap the models for ART (`attacks/wrappers.py`)

ART cannot attack a Keras model directly; it needs its own uniform interface. The
wrapper is `TensorFlowV2Classifier`, and four settings must agree with phase 2:

```python
TensorFlowV2Classifier(
    model=model,
    nb_classes=2,                                    # binary: benign / attack
    input_shape=(68,),                               # flat feature vector
    loss_object=keras.losses.CategoricalCrossentropy(),
    clip_values=(0.0, 1.0),                          # phase 1's scaled range
)
```

**`loss_object` — the one that silently ruins results.** An attack works by
computing the gradient *of a loss function* and stepping in the direction that
increases it. Phase 2 trained under `CategoricalCrossentropy`. If we handed ART a
different loss, the attack would climb a different hill than the one the model
descended during training — it would still produce output, still look plausible,
and be measuring the wrong thing. So we pass the same loss the models were
compiled with in `models/base.py:compile_model`.

**`input_shape=(68,)` for all three models.** Look at `models/cnn.py`: the model
takes a flat `(68,)` input and does `Reshape((68, 1))` *inside the graph*. Same in
`models/lstm.py`. This was a deliberate phase-2 decision, and it pays off here —
all three models live in the same 68-dimensional input space, so one ε means the
same thing everywhere, one validator serves all three, and phase 5's
discriminator sees one consistent feature space.

**`clip_values=(0,1)`.** ART clamps every perturbed value into this range. It is
correct *because* phase 1 scaled and clipped to `[0,1]`. If the two disagreed, ε
would stop meaning "fraction of the feature's range" and the artefacts could hold
values the pipeline cannot represent.

The wrapper actively rejects a 1-unit sigmoid model and a pre-reshaped input,
because both are mistakes that otherwise produce numbers rather than errors.

**One extra wrapper, for DeepFool only.** `load_wrapped()` returns *two* ART
classifiers over the same weights: the softmax one above, and a "logits" one with
the final softmax stripped (`as_logits`) and `from_logits=True`. FGSM, BIM and PGD
use the softmax view, so their gradients are the ones phase 2 trained under.
DeepFool uses the logits view — see below for why. Stripping the softmax changes
no prediction (softmax is monotone, so `argmax` is identical); it changes only the
surface the attack navigates.

### Step 2 — Pick the canonical rows (`attacks/generate.py`)

```python
def select_attack_rows(y, n, seed):
    pool = np.flatnonzero(y == 1)      # attack flows only
    rng = np.random.default_rng(seed)  # seeded => same rows every run
    return np.sort(rng.choice(pool, size=n, replace=False)).astype(np.int32)
```

Seeded, so re-running phase 3 next week gives byte-identical selection. Sorted, so
it reads sequentially off disk and is easy to eyeball. See "Canonical rows" above
for why this set is shared across all twelve cells.

### Step 3 — The four attacks

All four ask the same question — *which direction makes this look benign?* — and
differ in how they walk it.

#### FGSM — Fast Gradient Sign Method

One single step, of the full size ε, in the direction of the gradient's sign:

```
x_adv = x + ε · sign(∇ₓ loss(x, y))
```

Only the *sign* is used, so every touched feature moves by exactly ±ε. One
gradient computation — extremely fast, and correspondingly crude.

#### BIM — Basic Iterative Method

FGSM applied repeatedly in small steps, re-computing the gradient each time and
clipping back inside the ε-ball:

```
x₀ = x;  xₜ₊₁ = clip_ε( xₜ + ε_step · sign(∇ loss(xₜ, y)) )
```

With `ε_step=0.01, max_iter=10`, ten small steps instead of one big one. It
follows the curve of the loss surface, so it is strictly stronger than FGSM.

#### PGD — Projected Gradient Descent

BIM with more iterations (40) and, optionally, a random start inside the ε-ball.
Widely regarded as the strongest first-order attack. We set `num_random_init=0`
so the run is deterministic and reproducible.

#### DeepFool — the odd one out

Instead of a fixed budget, DeepFool estimates the *distance to the nearest
decision boundary* and steps just far enough to cross it, iterating until the
label flips. It answers "what is the **smallest** change that works?" rather than
"how much damage can I do within ε?"

**This is why DeepFool is excluded from the ε sweep**, and it is the single
easiest mistake to make in this phase. DeepFool's `epsilon=1e-6` parameter is
**not a budget** — it is an *overshoot multiplier*. Having found the boundary,
ART pushes `(1 + epsilon)` times as far, so the point lands just past it rather
than exactly on it. Sweeping that value would produce a curve that looks
meaningful and is not. `build_attack()` raises a `ValueError` if you try.

Two more DeepFool defaults that must be overridden:

| Parameter | ART default | Ours | Why |
| --- | --- | --- | --- |
| `batch_size` | **1** | 512 | Default = 5,000 sequential passes per model. Hours wasted. |
| `nb_grads` | **10** | 2 | "Gradients for the top-10 classes." We have 2 classes. |

**And DeepFool needs logits, not probabilities.** ART warns about this explicitly:

> *It seems that the attacked model is predicting probabilities. DeepFool expects
> logits as model output to achieve its full attack strength.*

Because DeepFool navigates *distances to the decision boundary*, and softmax
squashes those distances into [0,1] and saturates, it under-estimates how far it
must step and overshoots in compensation. This is not cosmetic. Measured on our
smoke fixture, switching DeepFool to the logits view cut the mean L2 perturbation
against the MLP from **3.77 to 0.85** — a 4.4× smaller change for the same 100%
evasion. That is DeepFool finally doing its actual job, which is to find the
*minimal* perturbation.

It also matters downstream: a smaller perturbation trips fewer of phase 4's rules,
so leaving this unfixed would have inflated the validator's catch rate against
DeepFool.

### Step 4 — Generate, with the true label

```python
X_adv = attack.generate(x=X, y=one_hot(y))
```

Passing `y` explicitly matters. With `y=None`, ART substitutes *the model's own
prediction* as the label to move away from. On rows the model already got wrong,
that would mean attacking away from "benign" — i.e. pushing the row back toward
being detected. Results would depend on model error in a way nobody would notice.
We always attack away from ground truth.

### Step 5 — Measure honestly

Every source row is a true attack, so on this subset three quantities collapse
into one:

> **accuracy = recall = detection rate**, and **attack success rate = 1 − accuracy**

Precision and FPR are *undefined* here — there are no benign rows to be wrong
about — so the report omits them rather than printing a meaningless 1.0.

The subtle metric is `evasion_rate_on_correct`. Suppose the LSTM already
misclassified 60 of the 5,000 rows before any attack. Those 60 are not evasions —
the attack deserves no credit for them. So:

```
accuracy_adv           = (predicted attack) / (all 5,000)
evasion_rate_on_correct = (flipped to benign) / (the ones it got right to begin with)
```

Both are recorded. `accuracy_adv` is what compares to the paper; the other is the
honest attack-strength number.

### Step 6 — Save the artefact

Each `artifacts/adversarial/{attack}_{model}.npz` holds:

| Key | What it is | Who needs it |
| --- | --- | --- |
| `X_adv` | the perturbed flows | phases 5, 6 — the artefact itself |
| `X_clean` | the originals, same order | phase 5 needs exact clean/adv pairs |
| `idx` | row numbers into `test.npz` | lets any phase re-derive everything |
| `y_true` | all 1 (attack) | phase 6 trains on the *true* label |
| `y_multiclass` | which attack type (Hulk, XSS…) | per-type reporting |
| `pred_clean`, `pred_adv` | before/after predictions | the honest denominator |
| `prob_adv` | full probability vectors | confidence analysis |
| `linf`, `l2` | per-row perturbation size | phase 4's magnitude rule |
| `n_changed` | how many of 68 features moved | explains phase 4's catch rates |
| `meta` | JSON: params, seed, versions | reproducibility |

Storing `X_clean` alongside `X_adv` duplicates data, and that is deliberate — it
makes each file self-contained, so phase 5 cannot accidentally pair rows against
a differently-ordered test split.

### Step 7 — Fail loudly

Two assertions run after every cell:

```python
# 1. clip_values actually applied?
if X_adv.min() < -1e-6 or X_adv.max() > 1 + 1e-6: raise AssertionError(...)

# 2. did the attack honour its budget?
if attack in EPS_ATTACKS and linf_max > eps + 1e-5: raise AssertionError(...)
```

Both check things that, if violated, would invalidate every downstream phase
while still producing files that look fine.

### What "good" looks like

The paper reports MLP → 24.95%, CNN → 49.76%, LSTM → 4.89%. If a model does **not**
collapse, the wrapper is wrong (nearly always a loss-object or softmax mismatch),
not the attack. If accuracy rises with ε, suspect gradient masking.

---

## Part 2 — Phase 4: the analyzer and validator

### The idea

The IDS does not see packets. It sees 68 **statistics** computed from packets by
CICFlowMeter — packet counts, durations, means, maxima, rates.

A gradient attack treats those 68 numbers as free variables and moves them
independently. But they are not independent. They were *derived from the same
packets*, so they must agree with each other. Move them freely and you get a row
describing a flow that could not physically exist:

- `Tot Fwd Pkts = 7.4` — you cannot send four-tenths of a packet
- `Fwd Pkt Len Min = 900` while `Fwd Pkt Len Max = 500` — a minimum above a maximum
- `Flow Pkts/s` that contradicts its own `Flow Duration` and packet count

The validator checks **arithmetic, not learned behaviour**. That gives it two
properties the paper's discriminator does not have: it needs no adversarial
training data, and there is nothing for it to overfit.

### The analyzer mines; the validator checks

- `defense/analyzer.py` — reads **benign training** flows and writes `rules.json`.
- `defense/validator.py` — reads `rules.json` and checks rows. Pure numpy.

The analyzer sees **only benign training data**. Never adversarial examples (so
the validator cannot be tuned to the attacks it will face), and never val or test
(so the false-positive rate is measured on rows the rules were not fitted to).

**One subtlety worth understanding.** The *residual arithmetic* lives in
`validator.py`, and the analyzer **imports it** to calibrate thresholds. If mining
and checking each had their own copy of "how far is this rule violated?", they
could drift apart, and every threshold would gate something other than what it
was calibrated on. One implementation, used by both.

### Working in raw units

The rules are checked in **original units** (packets, bytes, microseconds), not
the scaled `[0,1]` space.

Why: after MinMax scaling, "is this a whole number?" becomes "does this sit on a
lattice of spacing `1/(max−min)`?" — technically checkable, practically
unreadable and numerically fragile.

So the validator inverts the scaling first:

```python
raw = X_scaled * data_range + data_min
```

`data_min` and `data_range` are copied out of `scaler.pkl` **into `rules.json`**,
so the validator needs no scikit-learn at inference time — it is pure numpy and
therefore cheap enough to sit in front of every flow.

### The four rule families

#### 1. Range — has this value ever been seen?

Per-feature `[min, max]` over benign traffic, widened by 1% of the span. The
padding is not cosmetic: benign val/test rows legitimately fall slightly outside
whatever train happened to contain, and a zero-tolerance range rule would charge
that to the false-positive budget.

#### 2. Integrality — are counts whole numbers?

**The rule that does most of the work.** A gradient step adds a real-valued delta
to every feature it touches, so any count-like column lands off the integers
almost surely. FGSM at ε=0.1 cheerfully produces 7.4 packets.

Which columns are counts is **discovered, not hardcoded**:

```python
integral = (np.abs(raw - np.rint(raw)) <= 1e-6).all(axis=0)
```

A column is integral if *every* benign row has a whole number in it. Hardcoding a
list invites a silent mismatch when the feature set changes; deriving it means
`rules.json` states exactly what was found, and the write-up can show the
discovered set is the expected one — ports, protocol, packet counts, header
lengths, flag counts, window sizes, subflow counts.

**The tolerance must be per column, and this is subtle enough that it broke the
first implementation.** Phase 1 stores `X` as `float32`, so recovering raw units
costs roughly `range × 2⁻²⁴`:

| Column | Span | Round-trip error |
| --- | ---: | ---: |
| `SYN Flag Cnt` | 2 | 2.4e-08 |
| `Tot Fwd Pkts` | 198 | 2.4e-06 |
| `Init Fwd Win Byts` | 65,535 | 7.8e-04 |
| `Flow Duration` | 1.2e08 | **1.43** |

A flat `1e-6` tolerance would therefore declare `Flow Duration = 12345678.6` a
fractional value on *genuine benign traffic* and reject nearly everything. Scaling
the tolerance with each column's range leaves ~16× headroom over float32 noise
while staying ~5 orders of magnitude below the shift an ε=0.1 perturbation makes.
The separation is not close, but it is not automatic either.

#### 3. Dependency — do the features still agree?

Arithmetic invariants, declared by column name in `validator.py` so a rule
referencing a column phase 1 dropped is skipped rather than crashing:

| Kind | Example | Count |
| --- | --- | ---: |
| Ordering | `Fwd Pkt Len Min ≤ Mean ≤ Max` | 8 families |
| Product | `TotLen Fwd Pkts ≈ Tot Fwd Pkts × Fwd Pkt Len Mean` | 2 |
| Rate | `Flow Pkts/s ≈ (Fwd + Bwd pkts) / (Flow Duration / 1e6)` | 4 |

Residuals are **relative**, not absolute:

```python
np.abs(actual - expected) / (np.abs(expected) + 1.0)
```

`Flow Duration` runs to 1e8 while `Down/Up Ratio` is single digits — one absolute
tolerance cannot serve both. The `+1` floor keeps the ratio finite when both sides
are near zero (common for idle/active timers).

#### 4. Distribution — is the row jointly plausible?

Every value can be individually legal while the combination is bizarre. A robust
z-score catches that:

```python
z = |raw − median| / MAD          # MAD = median absolute deviation
z = clip(z, 0, 50)                # no single feature may dominate
score = z.mean(axis=1)            # the row's joint implausibility
```

**Median/MAD rather than mean/std**, because flow features are heavy-tailed
enough that a standard deviation is dominated by the very tail it is meant to
detect. One extreme flow inflates σ until nothing looks unusual any more.

**Mean rather than max, and clipped** — this also took a failed run to get right.
Scoring a row by its single worst feature sounds natural and silently kills the
rule. Many flow columns are almost always zero (the flag counts), so their MAD
sits on its floor and any nonzero value pegs `z` at the ceiling. That happened for
more than 1% of benign rows, so the calibrated 99th-percentile threshold *was* the
ceiling, and `score > threshold` was never true. The family reported a clean
column of zeroes and detected nothing.

Averaging clipped z-scores fixes it and is a better match to the threat anyway: a
gradient attack nudges ~65 of the 68 features a little, rather than one a lot.
After the fix the family spends 0.91% against its 1% budget — calibration working
as intended.

### Calibration: thresholds from data, not from taste

Every tolerance is computed, never guessed.

**Dependency tolerances** come from the residual distribution on benign traffic —
the 99.9th percentile of what real traffic shows. CICFlowMeter's own arithmetic is
not exact (it rounds; it computes means over slightly different packet sets than
the totals), so strict equality would reject real flows. The percentile absorbs
that measurement noise while staying far below what a gradient step produces.

**The distribution threshold** is calibrated on **benign validation** rows at a 1%
false-positive budget:

```python
threshold = np.percentile(scores, 99)
```

Calibrating on the same rows the median/MAD were mined from would report an
optimistic FPR — exactly the leak this project criticises in other papers. Mine on
train, calibrate on val, report on test.

> Why calibrate at all? Because a hand-picked tolerance gets tuned, in practice,
> until the catch rate looks good. Fixing the false-positive rate *by construction*
> and then reporting whatever catch rate falls out is the honest direction.

### Reporting: two numbers, never blended

```
catch-rate  = adversarial rows stopped        (want high)
FPR         = legitimate rows wrongly stopped (want ~0)
```

These are reported **separately and per rule family**. A validator catching 100%
of adversarial rows while rejecting 3% of legitimate traffic is unusable, and a
single combined "accuracy" would hide that completely. On a network carrying a
million flows an hour, 3% is 30,000 dropped legitimate connections.

#### The catch-rate trap — read this before quoting any number

A raw catch rate is **not** a measure of adversarial detection, and the first run
of phase 4 made that obvious.

Every adversarial row was built from a *real attack flow*. Real attack flows
already sit outside the benign envelope the rules were mined from — that is what
makes them attacks. So the range family rejected 100% of clean, **unperturbed**
attack rows. It was detecting *maliciousness*, not *perturbation*, and it would
have reported a triumphant "100% catch rate" while contributing nothing whatsoever
as an adversarial-example detector.

The report therefore carries three columns per cell:

| Column | Meaning |
| --- | --- |
| `catch_%` | rejected among the adversarial rows |
| `clean_orig_%` | rejected among **the same rows unperturbed** (`X_clean`) |
| `attributable_%` | rejected *only after* perturbation — what the gate actually adds |

`attributable_%` is the honest number. This is exactly why phase 3 stores
`X_clean` inside every `.npz` rather than only `X_adv`: without the clean
originals the comparison is impossible, and the catch rate cannot be interpreted
at all.

The same caution applies to the paper's own 100% discriminator accuracy. Ask what
the baseline was before believing any detection figure.

Per-family attribution matters for phase 8: a combined boolean cannot tell you
whether the defense rests on *arithmetic* (which an adaptive attacker can satisfy)
or on *distribution* (harder to satisfy silently).

### The finding to expect — and to write up

Integrality alone will probably catch nearly everything.

That is worth stating plainly, because it cuts both ways. It is the real reason
detector-based defenses report high numbers on gradient attacks — and it is also
their weakness. An adaptive adversary who **rounds the perturbation onto the
integer lattice and re-derives the dependent features** passes this gate
untouched. Carlini & Wagner broke ten detection defenses exactly this way;
Athalye et al. broke seven of nine.

So: report the high catch rate, then explain precisely why it does not mean the
defense is strong. That is phase 8's premise, and it is the contribution the
proposal is built around — not an embarrassment in the reproduction.

---

## Part 3 — Running it

```powershell
# from the aad/ directory, with .venv activated
..\.venv\Scripts\Activate.ps1

# always smoke-test first: full code path, ~seconds
python scripts\03_generate_ae.py --models mlp --attacks fgsm --n-samples 50 --no-sweep

# the real run: 12 artefacts + the sweep + figures
python scripts\03_generate_ae.py

# phase 4
python scripts\04_analyze_validate.py

# contract tests - synthetic fixtures, no dataset needed
python -m pytest tests\ -v
```

Useful flags:

| Flag | Effect |
| --- | --- |
| `--models mlp` | one model instead of three |
| `--attacks fgsm pgd` | a subset of attacks |
| `--n-samples 50` | tiny run for debugging |
| `--no-sweep` | skip the ε curve |
| `--force` | regenerate cells that already have a `.npz` |
| `--tag paperparity` | write to a separate namespace, e.g. for the non-deduplicated variant |

Phase 3 is **resumable**: a cell whose `.npz` already exists is skipped unless
`--force`. PGD-against-LSTM and DeepFool are expensive enough that a crash in the
plotting code must not cost the whole run — and that is exactly why the figures
are wrapped in `try/except`.

### Outputs

```
artifacts/
├── adversarial/{fgsm,bim,pgd,deepfool}_{mlp,cnn,lstm}.npz    12 files
├── rules.json                                                the mined rule set
└── reports/
    ├── adversarial_metrics.json      every number phase 3 produced
    ├── table4_evasion.{csv,md}       Table IV
    ├── eps_sweep.csv                 the ε curve as data
    ├── table5_validator.{csv,md}     Table V — catch-rate / FPR
    ├── validator_metrics.json
    └── figures/{fig3_evasion,fig4_eps_sweep}.png
```

`artifacts/reports/` is deliberately **not** gitignored — those tables are small,
diffable text and are the point of the project. The `.npz` files and models are
ignored, because they are large and regenerable.

---

## Part 4 — Things that will bite

| Symptom | Almost certainly |
| --- | --- |
| A model does not collapse under attack | Wrapper mismatch — wrong loss object, or a sigmoid model |
| DeepFool takes hours | `batch_size` left at ART's default of 1 |
| Accuracy *rises* with ε | Gradient masking — report it, it is a finding |
| Validator FPR above 3% | Range tolerance too tight, or distribution calibrated on train |
| Catch rate is 100% on everything | Check `attributable_%`, not `catch_%` — clean attack rows are rejected too |
| A rule family reports exactly 0.000% forever | Its threshold saturated. Check it is below the score's ceiling |
| Integrality rejects benign traffic | Tolerance not scaled per column; float32 round-trip on wide columns |
| DeepFool perturbations are huge | It is running on softmax instead of logits |
| `.npz` rows do not line up in phase 5 | Use the stored `idx`, never re-sample |
| Numbers change between runs | An unseeded RNG. Seed numpy, random, and `tf.random` — `set_seeds()` does all three |

---

## Part 5 — Where this sits in the project

```
Phase 1  data         →  data/processed/*.npz          (done)
Phase 2  baselines    →  artifacts/models/*.keras      (done)
Phase 3  AE generator →  artifacts/adversarial/*.npz   ← you are here
Phase 4  validator    →  artifacts/rules.json          ← and here
Phase 5  discriminator   trains on phase 3's output
Phase 6  EIDS            adversarially trains on phase 3's output
Phase 7  pipeline        wires validator → discriminator → EIDS
Phase 8  contribution    the adaptive attacker that phase 4 invites
```

Phase 3 is the bottleneck for everything after it: phases 5 and 6 both consume
its `.npz` files as training data. Get it right, and get it saved to disk.
