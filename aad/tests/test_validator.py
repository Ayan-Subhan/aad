"""Contract tests for the analyzer and validator.

Built on a small synthetic "benign" population with the real CICFlowMeter column
names, so the declared dependency rules resolve exactly as they will on the real
feature set -- without needing the dataset on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aad.defense.analyzer import calibrate_distribution, mine  # noqa: E402
from aad.defense.validator import Z_CLIP, Validator, norm_name  # noqa: E402

# A subset of the real schema, chosen so every rule family has something to bite.
FEATURES = [
    "Dst Port", "Protocol", "Flow Duration",
    "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Max",
    "Flow Pkts/s", "SYN Flag Cnt", "Down/Up Ratio",
]
I = {name: i for i, name in enumerate(FEATURES)}


def make_benign(n: int = 4000, seed: int = 0) -> np.ndarray:
    """Rows that satisfy every invariant, because they are built from packets."""
    rng = np.random.default_rng(seed)
    n_fwd = rng.integers(1, 200, n).astype(np.float64)
    n_bwd = rng.integers(1, 200, n).astype(np.float64)
    lo = rng.integers(0, 60, n).astype(np.float64)
    mean = lo + rng.integers(0, 500, n)
    hi = mean + rng.integers(0, 500, n)
    duration = rng.integers(1_000, 10_000_000, n).astype(np.float64)

    raw = np.zeros((n, len(FEATURES)))
    raw[:, I["Dst Port"]] = rng.integers(1, 65535, n)
    raw[:, I["Protocol"]] = rng.choice([6, 17], n)
    raw[:, I["Flow Duration"]] = duration
    raw[:, I["Tot Fwd Pkts"]] = n_fwd
    raw[:, I["Tot Bwd Pkts"]] = n_bwd
    raw[:, I["TotLen Fwd Pkts"]] = n_fwd * mean
    raw[:, I["TotLen Bwd Pkts"]] = n_bwd * mean
    raw[:, I["Fwd Pkt Len Min"]] = lo
    raw[:, I["Fwd Pkt Len Mean"]] = mean
    raw[:, I["Fwd Pkt Len Max"]] = hi
    raw[:, I["Flow Pkts/s"]] = (n_fwd + n_bwd) / (duration / 1e6)
    raw[:, I["SYN Flag Cnt"]] = rng.integers(0, 3, n)
    raw[:, I["Down/Up Ratio"]] = rng.random(n) * 5
    return raw


@pytest.fixture(scope="module")
def fitted():
    """Mine rules from synthetic benign traffic, exactly as phase 4 does."""
    raw = make_benign(4000, seed=0)
    data_min, data_max = raw.min(axis=0), raw.max(axis=0)
    data_range = np.maximum(data_max - data_min, 1e-9)
    scaled = ((raw - data_min) / data_range).astype(np.float32)

    rules = mine(scaled, FEATURES, data_min, data_range)
    val = make_benign(2000, seed=1)
    val_scaled = np.clip((val - data_min) / data_range, 0, 1).astype(np.float32)
    rules = calibrate_distribution(rules, val_scaled, fpr_budget=0.01)
    return rules, Validator(rules), data_min, data_range


def to_scaled(raw, data_min, data_range):
    return np.clip((raw - data_min) / data_range, 0, 1).astype(np.float32)


# --- mining ----------------------------------------------------------------


def test_integer_columns_are_discovered_not_hardcoded(fitted):
    rules, _, _, _ = fitted
    found = {FEATURES[i] for i in rules["integrality"]["indices"]}
    for expected in ["Dst Port", "Protocol", "Tot Fwd Pkts", "Tot Bwd Pkts", "SYN Flag Cnt"]:
        assert expected in found, f"{expected} is a count and should be integral"
    assert "Down/Up Ratio" not in found, "a genuinely continuous column must not be flagged"


def test_no_integrality_rule_is_vacuous(fitted):
    """Regression: the per-column tolerance scales with the column's range, and
    on wide columns it grew past 0.5 -- at which point every real number is
    within tolerance of an integer and the check can never fail. On real data
    that silently declared 59 of 68 columns "integral", including `Flow Byts/s`
    (tolerance 850) and `Flow IAT Std`, neither of which is a count. Harmless to
    the catch rate, but rules.json asserted something false."""
    rules, _, _, _ = fitted
    for col, tol in zip(rules["integrality"]["columns"], rules["integrality"]["tolerance"]):
        assert tol < 0.5, f"{col} has tolerance {tol}: every real number passes"
        assert tol <= rules["integrality"]["max_tolerance"]


def test_unverifiable_integer_columns_are_recorded_not_dropped(fitted):
    """Excluded columns must still be listed, so the count is not mistaken for
    'every count-like column in the schema'."""
    rules, _, _, _ = fitted
    assert "excluded_unverifiable" in rules["integrality"]
    assert set(rules["integrality"]["excluded_unverifiable"]).isdisjoint(
        rules["integrality"]["columns"]
    )


def test_dependency_rules_resolve_against_the_schema(fitted):
    rules, _, _, _ = fitted
    names = {r["name"] for r in rules["dependency"]["rules"]}
    assert any("Fwd Pkt Len Min" in n for n in names), "ordering rule missing"
    assert any("TotLen Fwd Pkts" in n for n in names), "product rule missing"
    assert any("Flow Pkts/s" in n for n in names), "rate rule missing"


def test_rules_reference_only_present_columns():
    """A rule naming a column phase 1 dropped must be skipped, not crash.

    Phase 1 removed 10 zero-variance columns, so any declared invariant may find
    one of its operands missing.
    """
    names = ["Dst Port", "Protocol", "Down/Up Ratio"]  # no rule can be built
    raw = make_benign(500, seed=2)[:, [I[n] for n in names]]
    data_min, data_range = raw.min(axis=0), np.maximum(raw.max(axis=0) - raw.min(axis=0), 1e-9)
    rules = mine(((raw - data_min) / data_range).astype(np.float32), names, data_min, data_range)
    assert rules["dependency"]["rules"] == []


# --- false positives -------------------------------------------------------


def test_benign_traffic_passes_within_budget(fitted):
    """The whole point. A gate rejecting real traffic is unusable whatever it catches."""
    _, v, data_min, data_range = fitted
    unseen = to_scaled(make_benign(2000, seed=99), data_min, data_range)
    assert v.check(unseen).reject_rate <= 0.03


def test_arithmetic_families_only_fire_on_clipped_rows(fitted):
    """Range, integrality and dependency describe how CICFlowMeter computes, so
    on genuine benign flows they cost essentially nothing -- unlike the
    distribution family, which spends a deliberate 1% budget.

    The one exception is real and worth pinning down: phase 1 clips val/test
    into [0,1] after scaling, and clipping a value breaks the arithmetic
    identity it participated in. So the dependency family may fire on exactly
    those rows and no others. In the real test split that is 20 cells out of
    15.3 million (see prepare_meta.json), i.e. under 0.01% of rows.
    """
    _, v, data_min, data_range = fitted
    fresh = make_benign(2000, seed=7)
    unclipped = (fresh - data_min) / data_range
    clipped = to_scaled(fresh, data_min, data_range)
    altered_by_clip = (unclipped.astype(np.float32) != clipped).any(axis=1)

    rates = v.check(clipped).family_rates()
    assert rates["integrality"] == 0.0
    # Every dependency violation must be attributable to the clip.
    dep = v.check(clipped).by_family["dependency"]
    assert not (dep & ~altered_by_clip).any(), "an unclipped benign row broke an invariant"
    assert rates["dependency"] < 0.01


# --- catching violations ---------------------------------------------------


def test_fractional_packet_count_is_caught(fitted):
    """The single most important rule: FGSM produces 7.4 packets."""
    _, v, data_min, data_range = fitted
    raw = make_benign(200, seed=3)
    raw[:, I["Tot Fwd Pkts"]] += 0.4
    res = v.check(to_scaled(raw, data_min, data_range))
    assert res.by_family["integrality"].all()


def test_broken_min_mean_max_ordering_is_caught(fitted):
    """Violation injected in *scaled* space, so every value stays inside the
    benign envelope. An adversarial row is already in [0,1] and phase 1 clips,
    so an out-of-range violation would be masked by the clip and caught by the
    range family anyway -- this isolates the dependency family."""
    _, v, data_min, data_range = fitted
    scaled = to_scaled(make_benign(200, seed=4), data_min, data_range)
    scaled[:, I["Fwd Pkt Len Min"]] = 1.0   # that column's own maximum
    scaled[:, I["Fwd Pkt Len Mean"]] = 0.0  # that column's own minimum
    res = v.check(scaled)
    assert res.by_family["dependency"].mean() > 0.99


def test_broken_total_equals_count_times_mean_is_caught(fitted):
    _, v, data_min, data_range = fitted
    scaled = to_scaled(make_benign(200, seed=5), data_min, data_range)
    scaled[:, I["TotLen Fwd Pkts"]] = 0.0  # in range, but no longer count x mean
    res = v.check(scaled)
    assert res.by_family["dependency"].mean() > 0.99


def test_out_of_range_value_is_caught(fitted):
    _, v, data_min, data_range = fitted
    raw = make_benign(100, seed=6)
    raw[:, I["Down/Up Ratio"]] = 1e6
    # Bypass the scaled-space clip, which would otherwise mask the excursion.
    scaled = (raw - data_min) / data_range
    assert v.check(scaled).by_family["range"].all()


# --- attribution and plumbing ----------------------------------------------


def test_distribution_family_actually_fires(fitted):
    """Regression: the distribution family once scored a row by its single worst
    feature. Flag-count columns are almost always zero, so their MAD sits on its
    floor, any nonzero value pegs z at the cap, and the calibrated threshold
    saturated at the cap for >1% of benign rows -- leaving the family unable to
    reject anything at all. It failed silently as a column of zeroes."""
    rules, v, data_min, data_range = fitted
    score = rules["distribution"]["threshold"]
    assert 0 < score < Z_CLIP, f"threshold {score} is saturated at the z cap"

    # And it must reject a diffuse shift across many features -- which is what a
    # gradient attack produces (~65 of 68 columns moved).
    scaled = to_scaled(make_benign(200, seed=11), data_min, data_range)
    scaled = np.clip(scaled + 0.35, 0, 1)
    assert v.check(scaled).by_family["distribution"].mean() > 0.5


def test_sparse_extreme_spike_is_caught(fitted):
    """Three extreme features out of 68 still clear the threshold even though
    the score is a *mean* -- the clipped z-scores are large enough and the
    calibrated threshold low enough. Range catches it too; the families overlap
    here rather than dividing the work."""
    _, v, data_min, data_range = fitted
    raw = make_benign(200, seed=13)
    for col in ["Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts"]:
        raw[:, I[col]] = raw[:, I[col]].max() * 50
    res = v.check((raw - data_min) / data_range)  # unclipped, as a real excursion
    assert res.by_family["range"].all()
    assert res.by_family["distribution"].all()
    assert res.reject.all()


def test_distribution_budget_is_calibrated_not_guessed(fitted):
    """The threshold is the 99th percentile of the benign val score, so the
    family should spend close to its 1% budget on unseen benign rows -- not far
    less (dead) and not far more (unusable)."""
    _, v, data_min, data_range = fitted
    unseen = to_scaled(make_benign(3000, seed=12), data_min, data_range)
    rate = v.check(unseen).family_rates()["distribution"]
    assert 0.001 < rate < 0.05, f"distribution FPR {rate} is nowhere near its 1% budget"


def test_check_reports_which_family_fired(fitted):
    """Phase 8 needs to know whether the defense rests on arithmetic or on
    distribution, so a combined boolean is not enough."""
    _, v, data_min, data_range = fitted
    raw = make_benign(50, seed=8)
    raw[:, I["Tot Fwd Pkts"]] += 0.4
    res = v.check(to_scaled(raw, data_min, data_range))
    assert set(res.by_family) == {"range", "integrality", "dependency", "distribution"}
    assert res.by_family["integrality"].all()
    assert res.reject.all()


def test_selected_families_can_be_run_alone(fitted):
    """The phase 7 ablation switches families on one at a time."""
    _, v, data_min, data_range = fitted
    raw = make_benign(50, seed=9)
    res = v.check(to_scaled(raw, data_min, data_range), families=("integrality",))
    assert set(res.by_family) == {"integrality"}


def test_round_trip_to_raw_units_is_lossless(fitted):
    """Every check happens in raw units, so the inverse transform must be exact."""
    _, v, data_min, data_range = fitted
    raw = make_benign(100, seed=10)
    scaled = (raw - data_min) / data_range
    assert np.allclose(v.to_raw(scaled), raw, rtol=1e-9, atol=1e-6)


def test_norm_name_absorbs_spacing_variants():
    assert norm_name("Brute Force -Web") == norm_name("BruteForce-Web") == "bruteforceweb"
    assert norm_name("TotLen Fwd Pkts") == norm_name("Tot Len  Fwd Pkts")
