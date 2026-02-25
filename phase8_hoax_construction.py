"""
---
version: 0.5.0
created: 2026-02-25
updated: 2026-02-25
---

phase8_hoax_construction.py — Hoax construction method analysis.

Tests HOW a hoax would have been built by generating six construction
methods and comparing their statistical fingerprints to B1/B3:

  1. Genuine: English-frequency plaintext → DoI encode
  2. Random: uniform random numbers
  3. Human-random: biased random (log-normal, digit pref, sequential avoid)
  4. Gibberish-encoded: random letters → DoI encode
  4b. Biased-gibberish: English-vowel-frequency letters → DoI encode
  5. Sequential-gibberish: random letters → DoI encode with nearest-forward scan
  6. Page-constrained: sequential within physical pages, periodic page flips

Key prediction: B1's distinct ratio (57.3%) is too high for genuine (~24%)
but too low for uniform random (~92%). Gibberish encoded with DoI should
produce an intermediate ratio because homophone counts vary by letter.

Usage:
    python3 phase8_hoax_construction.py [--generate | --analyze | --human-tests | --all]
    python3 phase8_hoax_construction.py --all --n-sims 500 --no-plots
    python3 phase8_hoax_construction.py --fatigue-test --n-sims 10000
"""

from __future__ import annotations

import argparse
import string
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

from beale import (
    B1, B2, B3, BEALE_DOI, ENGLISH_FREQ,
    benford_test, last_digit_test, distinct_ratio,
    decode_book_cipher, encode_book_cipher, encode_sequential_book_cipher,
    encode_page_constrained_book_cipher, build_letter_index,
    bigram_score, index_of_coincidence,
    gillogly_strings, gillogly_quality,
    generate_fake_cipher, generate_random_numbers, generate_human_random,
)


# ============================================================================
# 1. POPULATION GENERATION
# ============================================================================

# Biased-gibberish letter distribution: English vowel rates, uniform consonants
_VOWELS = set("aeiou")
_CONSONANTS = [c for c in string.ascii_lowercase if c not in _VOWELS]
_BIASED_GIBBERISH_FREQ: dict[str, float] = {}
_vowel_total = sum(ENGLISH_FREQ[v] for v in _VOWELS)
_consonant_share = 1.0 - _vowel_total
_consonant_each = _consonant_share / len(_CONSONANTS)
for _c in string.ascii_lowercase:
    if _c in _VOWELS:
        _BIASED_GIBBERISH_FREQ[_c] = ENGLISH_FREQ[_c]
    else:
        _BIASED_GIBBERISH_FREQ[_c] = _consonant_each


def generate_gibberish_cipher(
    count: int,
    key_words: list[str] | tuple[str, ...],
    rng: np.random.Generator,
    letter_probs: dict[str, float] | None = None,
) -> list[int]:
    """
    Generate cipher by encoding random letters with a book cipher key.

    If letter_probs is None, uses uniform distribution over letters that
    have homophones in the key. If provided, uses those probabilities
    (filtered to available letters).
    """
    index = build_letter_index(key_words)
    available = [c for c in string.ascii_lowercase if index.get(c)]

    if letter_probs is None:
        # Uniform over available letters
        probs = np.ones(len(available)) / len(available)
    else:
        probs = np.array([letter_probs.get(c, 0.001) for c in available])
        probs /= probs.sum()

    plaintext = "".join(rng.choice(available, size=count, p=probs))
    return encode_book_cipher(plaintext, key_words, rng)


def generate_sequential_gibberish_cipher(
    count: int,
    key_words: list[str] | tuple[str, ...],
    rng: np.random.Generator,
    reset_prob: float = 0.0,
) -> list[int]:
    """
    Generate cipher by encoding random letters with sequential (nearest-forward)
    homophone selection. Simulates a hoaxer scanning through the DoI in order
    rather than picking random homophones.

    When reset_prob > 0, simulates the hoaxer periodically losing their place
    and restarting from a random position in the key text.
    """
    index = build_letter_index(key_words)
    available = [c for c in string.ascii_lowercase if index.get(c)]
    probs = np.ones(len(available)) / len(available)
    plaintext = "".join(rng.choice(available, size=count, p=probs))
    return encode_sequential_book_cipher(plaintext, key_words,
                                         reset_prob=reset_prob, rng=rng)


def generate_populations(
    n_sims: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Generate six populations for B1-like and B3-like cipher parameters.

    Returns dict keyed by "{method}_{length}" with lists of cipher number sequences.
    Also returns the actual B1/B3 for convenience.
    """
    rng = np.random.default_rng(seed)

    # Cipher parameters
    configs = {
        "520": {"count": 520, "max_val": 2906},  # B1-like
        "618": {"count": 618, "max_val": 975},    # B3-like
    }

    populations: dict[str, list[list[int]]] = {}

    for length_key, cfg in configs.items():
        count = cfg["count"]
        max_val = cfg["max_val"]

        for method in METHODS:
            pop_key = f"{method}_{length_key}"
            populations[pop_key] = []
            t0 = time.time()

            for i in range(n_sims):
                if (i + 1) % 200 == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    print(f"  {pop_key}: {i+1}/{n_sims} ({rate:.0f}/s)", end="\r")

                if method == "genuine":
                    try:
                        cipher = generate_fake_cipher(count, BEALE_DOI, rng)
                    except ValueError:
                        continue
                elif method == "random":
                    cipher = generate_random_numbers(count, max_val, rng)
                elif method == "human_random":
                    cipher = generate_human_random(count, max_val, rng)
                elif method == "gibberish":
                    cipher = generate_gibberish_cipher(count, BEALE_DOI, rng)
                elif method == "biased_gibberish":
                    cipher = generate_gibberish_cipher(
                        count, BEALE_DOI, rng, _BIASED_GIBBERISH_FREQ
                    )
                elif method == "seq_gibberish":
                    cipher = generate_sequential_gibberish_cipher(
                        count, BEALE_DOI, rng
                    )

                populations[pop_key].append(cipher)

            actual = len(populations[pop_key])
            elapsed = time.time() - t0
            print(f"  {pop_key}: {actual} sims in {elapsed:.1f}s" + " " * 20)

    return populations


# ============================================================================
# 2. STATS BATTERY (extended from phase2)
# ============================================================================

def serial_correlation(numbers: list[int] | tuple[int, ...]) -> float:
    """Lag-1 serial correlation coefficient. Humans produce negative values."""
    if len(numbers) < 2:
        return 0.0
    arr = np.array(numbers, dtype=float)
    return float(np.corrcoef(arr[:-1], arr[1:])[0, 1])


def digit_distribution_chi2(numbers: list[int] | tuple[int, ...]) -> float:
    """Chi-squared test of individual digit frequencies against uniform."""
    all_digits = []
    for n in numbers:
        all_digits.extend(int(d) for d in str(abs(n)))
    observed = Counter(all_digits)
    total = len(all_digits)
    obs_arr = np.array([observed.get(d, 0) for d in range(10)], dtype=float)
    exp_arr = np.full(10, total / 10, dtype=float)
    chi2, _ = sp_stats.chisquare(obs_arr, exp_arr)
    return float(chi2)


def compute_extended_stats(
    cipher: list[int] | tuple[int, ...],
    key_words: list[str] | tuple[str, ...] = BEALE_DOI,
) -> dict:
    """Compute extended stats vector for a cipher (superset of phase2's compute_stats)."""
    dr = distinct_ratio(cipher)
    bf = benford_test(cipher)
    ld10 = last_digit_test(cipher, base=10)
    ld7 = last_digit_test(cipher, base=7)
    ld3 = last_digit_test(cipher, base=3)

    decoded = decode_book_cipher(cipher, key_words)
    clean = "".join(c for c in decoded if c.isalpha())
    bg = bigram_score(clean) if len(clean) >= 2 else -4.0
    ic = index_of_coincidence(clean)

    sc = serial_correlation(cipher)
    dd_chi2 = digit_distribution_chi2(cipher)

    return {
        "distinct_ratio": dr["ratio"],
        "benford_chi2": bf["chi2"],
        "benford_epsilon": bf["epsilon"],
        "ld10_chi2": ld10["chi2"],
        "ld7_chi2": ld7["chi2"],
        "ld3_chi2": ld3["chi2"],
        "ld7_p": ld7["p_value"],
        "ld3_p": ld3["p_value"],
        "bigram_score": bg,
        "ic": ic,
        "serial_corr": sc,
        "digit_dist_chi2": dd_chi2,
    }


# ============================================================================
# 3. DISTRIBUTION COMPARISON + CLASSIFICATION
# ============================================================================

METHODS = ["genuine", "random", "human_random", "gibberish", "biased_gibberish",
           "seq_gibberish"]
METHOD_LABELS = {
    "genuine": "Genuine",
    "random": "Random",
    "human_random": "Human-Rnd",
    "gibberish": "Gibberish",
    "biased_gibberish": "BiasGib",
    "seq_gibberish": "SeqGib",
}

METRICS = [
    "distinct_ratio", "benford_chi2", "benford_epsilon",
    "ld10_chi2", "ld7_chi2", "ld3_chi2", "ld7_p", "ld3_p",
    "bigram_score", "ic", "serial_corr", "digit_dist_chi2",
]

METRIC_LABELS = {
    "distinct_ratio": "Distinct ratio",
    "benford_chi2": "Benford chi2",
    "benford_epsilon": "Benford epsilon",
    "ld10_chi2": "Last-dig b10 chi2",
    "ld7_chi2": "Last-dig b7 chi2",
    "ld3_chi2": "Last-dig b3 chi2",
    "ld7_p": "Last-dig b7 p",
    "ld3_p": "Last-dig b3 p",
    "bigram_score": "Bigram log-prob",
    "ic": "Index of coincid.",
    "serial_corr": "Serial correlation",
    "digit_dist_chi2": "Digit dist chi2",
}


def compute_population_stats(
    populations: dict[str, list[list[int]]],
    n_sims: int,
) -> dict[str, list[dict]]:
    """Compute extended stats for all populations. Returns {pop_key: [stats_dicts]}."""
    all_stats: dict[str, list[dict]] = {}

    for pop_key, ciphers in populations.items():
        all_stats[pop_key] = []
        t0 = time.time()

        for i, cipher in enumerate(ciphers):
            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  Stats {pop_key}: {i+1}/{len(ciphers)} ({rate:.0f}/s)", end="\r")

            all_stats[pop_key].append(compute_extended_stats(cipher))

        elapsed = time.time() - t0
        print(f"  Stats {pop_key}: {len(ciphers)} done in {elapsed:.1f}s" + " " * 20)

    return all_stats


def print_comparison_table(
    pop_stats: dict[str, list[dict]],
    cipher_stats: dict,
    cipher_name: str,
    length_key: str,
) -> dict[str, str]:
    """
    Print comparison table + z-score classification for one cipher.

    Returns dict of {metric: best_method} for the classification matrix.
    """
    print(f"\n{'='*90}")
    print(f"CLASSIFICATION: {cipher_name} (length={length_key})")
    print(f"{'='*90}")

    # Header
    method_cols = " ".join(f"{METHOD_LABELS[m]:>12}" for m in METHODS)
    print(f"\n{'Metric':<20} {'Actual':>8}  {method_cols}  Best")
    print("-" * (20 + 8 + 2 + 13 * len(METHODS) + 6))

    votes: dict[str, float] = {m: 0.0 for m in METHODS}
    best_methods: dict[str, str] = {}

    for metric in METRICS:
        actual = cipher_stats[metric]

        # Compute mean/std for each method
        method_info: dict[str, tuple[float, float, float]] = {}
        for method in METHODS:
            key = f"{method}_{length_key}"
            if key not in pop_stats or not pop_stats[key]:
                method_info[method] = (0.0, 1.0, 99.0)
                continue
            vals = [s[metric] for s in pop_stats[key]]
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            z = abs(actual - mean) / max(std, 1e-10)
            method_info[method] = (mean, std, z)

        # Find best (lowest z-score)
        best = min(METHODS, key=lambda m: method_info[m][2])
        best_z = method_info[best][2]
        best_methods[metric] = best

        # Vote: only count if z < 3 (otherwise outlier to all)
        if best_z < 3.0:
            votes[best] += 1.0
        # Half vote to second-best if also close
        sorted_methods = sorted(METHODS, key=lambda m: method_info[m][2])
        if len(sorted_methods) > 1 and method_info[sorted_methods[1]][2] < 2.0:
            votes[sorted_methods[1]] += 0.3

        # Format row
        label = METRIC_LABELS.get(metric, metric)
        cols = ""
        for method in METHODS:
            mean, std, z = method_info[method]
            marker = "*" if method == best else " "
            cols += f" {mean:>10.4f}{marker} "

        best_label = METHOD_LABELS[best]
        z_str = f"(z={best_z:.1f})"
        print(f"{label:<20} {actual:>8.4f}  {cols} {best_label} {z_str}")

    # Vote tally
    print(f"\n  Z-score votes:")
    for method in METHODS:
        bar = "#" * int(votes[method] * 3)
        print(f"    {METHOD_LABELS[method]:<12} {votes[method]:>5.1f}  {bar}")

    winner = max(METHODS, key=lambda m: votes[m])
    print(f"\n  >> {cipher_name} most resembles: {METHOD_LABELS[winner]} "
          f"(score: {votes[winner]:.1f})")

    return best_methods


# ============================================================================
# 4. HUMAN-RANDOM SPECIFIC TESTS
# ============================================================================

def runs_test(numbers: list[int] | tuple[int, ...]) -> dict:
    """
    Count ascending/descending runs. Humans alternate too frequently,
    producing more runs than expected.
    """
    if len(numbers) < 2:
        return {"n_runs": 0, "expected": 0, "z": 0.0}

    n = len(numbers)
    runs = 1
    for i in range(1, n):
        if (numbers[i] > numbers[i - 1]) != (numbers[i - 1] > numbers[max(0, i - 2)]):
            if i > 1:  # skip first comparison
                runs += 1

    # For truly random data, expected runs ≈ (2n-1)/3
    expected = (2 * n - 1) / 3
    var = (16 * n - 29) / 90
    z = (runs - expected) / max(var ** 0.5, 1e-10)

    return {"n_runs": runs, "expected": round(expected, 1), "z": round(z, 2)}


def gap_test(numbers: list[int] | tuple[int, ...]) -> dict:
    """
    Distribution of gaps between repeated values. Humans space too evenly
    (low variance). Returns CV (coefficient of variation) of gap sizes.
    """
    positions: dict[int, list[int]] = {}
    for i, v in enumerate(numbers):
        positions.setdefault(v, []).append(i)

    gaps = []
    for indices in positions.values():
        for j in range(1, len(indices)):
            gaps.append(indices[j] - indices[j - 1])

    if not gaps:
        return {"mean_gap": 0.0, "std_gap": 0.0, "cv": 0.0, "n_gaps": 0}

    arr = np.array(gaps, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    cv = std / mean if mean > 0 else 0.0

    return {
        "mean_gap": round(mean, 1),
        "std_gap": round(std, 1),
        "cv": round(cv, 3),
        "n_gaps": len(gaps),
    }


def first_digit_test(numbers: list[int] | tuple[int, ...]) -> dict:
    """
    First-digit distribution compared to both Benford and uniform.
    Returns chi2 against each — different hoax methods diverge differently.
    """
    import math

    first_digits = []
    for n in numbers:
        s = str(abs(n))
        if s[0] != "0":
            first_digits.append(int(s[0]))

    observed = Counter(first_digits)
    total = len(first_digits)
    obs_arr = np.array([observed.get(d, 0) for d in range(1, 10)], dtype=float)

    # Benford expected
    benford_exp = np.array([math.log10(1 + 1 / d) * total for d in range(1, 10)])
    chi2_benford, _ = sp_stats.chisquare(obs_arr, benford_exp)

    # Uniform expected
    uniform_exp = np.full(9, total / 9, dtype=float)
    chi2_uniform, _ = sp_stats.chisquare(obs_arr, uniform_exp)

    return {
        "chi2_benford": round(float(chi2_benford), 2),
        "chi2_uniform": round(float(chi2_uniform), 2),
        "ratio": round(float(chi2_benford) / max(float(chi2_uniform), 0.01), 3),
    }


def print_human_random_tests(cipher: tuple[int, ...], name: str) -> None:
    """Run and print all human-random specific tests on a cipher."""
    print(f"\n{'='*60}")
    print(f"HUMAN-RANDOM TESTS: {name} ({len(cipher)} numbers)")
    print(f"{'='*60}")

    # Serial correlation
    sc = serial_correlation(cipher)
    print(f"\n  Serial correlation (lag-1): {sc:.4f}")
    print(f"    Random ≈ 0, Human-random < 0 (avoid close values)")
    if sc < -0.05:
        print(f"    >> Negative — consistent with human avoidance")
    elif abs(sc) < 0.05:
        print(f"    >> Near zero — consistent with random or encoded")
    else:
        print(f"    >> Positive — unusual for any construction method")

    # Runs test
    rt = runs_test(cipher)
    print(f"\n  Runs test: {rt['n_runs']} runs (expected {rt['expected']}, z={rt['z']})")
    print(f"    Humans: z > 0 (too many alternations)")
    if rt["z"] > 2.0:
        print(f"    >> Significantly more runs than random — human signal")
    elif rt["z"] < -2.0:
        print(f"    >> Significantly fewer runs — unusual")
    else:
        print(f"    >> Within normal range")

    # Gap test
    gt = gap_test(cipher)
    print(f"\n  Gap test: mean={gt['mean_gap']}, std={gt['std_gap']}, "
          f"CV={gt['cv']} ({gt['n_gaps']} gaps)")
    print(f"    Humans: low CV (too-even spacing). Random: high CV.")

    # First-digit distribution
    fdt = first_digit_test(cipher)
    print(f"\n  First-digit test:")
    print(f"    chi2 vs Benford:  {fdt['chi2_benford']:.1f}")
    print(f"    chi2 vs Uniform:  {fdt['chi2_uniform']:.1f}")
    print(f"    Benford/Uniform ratio: {fdt['ratio']:.3f}")
    print(f"    Genuine: low Benford chi2. Human-random: high Benford, lower Uniform.")

    # Generate reference values from 1000 random sequences
    rng = np.random.default_rng(42)
    max_val = max(cipher)
    ref_sc = []
    ref_runs_z = []
    for _ in range(1000):
        ref = generate_random_numbers(len(cipher), max_val, rng)
        ref_sc.append(serial_correlation(ref))
        ref_runs_z.append(runs_test(ref)["z"])

    sc_pct = float(np.mean(np.array(ref_sc) <= sc) * 100)
    rz_pct = float(np.mean(np.array(ref_runs_z) <= rt["z"]) * 100)

    print(f"\n  Percentile vs 1000 random (same length/range):")
    print(f"    Serial correlation: {sc_pct:.1f}th percentile")
    print(f"    Runs z-score: {rz_pct:.1f}th percentile")


# ============================================================================
# 5. DISTINCT RATIO DEEP DIVE
# ============================================================================

def print_distinct_ratio_analysis(
    pop_stats: dict[str, list[dict]],
    b1_stats: dict,
    b3_stats: dict,
) -> None:
    """Deep dive into distinct ratio distributions across all methods."""
    print(f"\n{'='*70}")
    print("DISTINCT RATIO DEEP DIVE")
    print(f"{'='*70}")

    b1_dr = b1_stats["distinct_ratio"]
    b3_dr = b3_stats["distinct_ratio"]
    print(f"\n  B1 actual: {b1_dr:.3f} ({b1_dr:.1%})")
    print(f"  B3 actual: {b3_dr:.3f} ({b3_dr:.1%})")

    for length_key, cipher_name, actual_dr in [
        ("520", "B1", b1_dr),
        ("618", "B3", b3_dr),
    ]:
        print(f"\n  --- {cipher_name} (length={length_key}) ---")
        print(f"  {'Method':<18} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} "
              f"{'Pctile':>8} {'|z|':>8}")
        print(f"  {'-'*66}")

        best_z = 999.0
        best_method = ""

        for method in METHODS:
            key = f"{method}_{length_key}"
            if key not in pop_stats or not pop_stats[key]:
                continue
            vals = [s["distinct_ratio"] for s in pop_stats[key]]
            arr = np.array(vals)
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            pctile = float(np.mean(arr <= actual_dr) * 100)
            z = abs(actual_dr - mean) / max(std, 1e-10)

            if z < best_z:
                best_z = z
                best_method = method

            label = METHOD_LABELS[method]
            print(f"  {label:<18} {mean:>8.3f} {std:>8.3f} {min(vals):>8.3f} "
                  f"{max(vals):>8.3f} {pctile:>7.1f}% {z:>8.1f}")

        print(f"\n  >> {cipher_name} distinct ratio best matches: "
              f"{METHOD_LABELS[best_method]} (z={best_z:.2f})")


def plot_distinct_ratio_histograms(
    pop_stats: dict[str, list[dict]],
    b1_stats: dict,
    b3_stats: dict,
    save_dir: Path,
) -> None:
    """Plot distinct ratio distributions for all methods, marking B1/B3."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping plots")
        return

    colors = {
        "genuine": "#2ecc71",
        "random": "#95a5a6",
        "human_random": "#e74c3c",
        "gibberish": "#3498db",
        "biased_gibberish": "#9b59b6",
        "seq_gibberish": "#e67e22",
    }

    for length_key, cipher_name, actual_dr, marker_color in [
        ("520", "B1", b1_stats["distinct_ratio"], "red"),
        ("618", "B3", b3_stats["distinct_ratio"], "blue"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))

        for method in METHODS:
            key = f"{method}_{length_key}"
            if key not in pop_stats or not pop_stats[key]:
                continue
            vals = [s["distinct_ratio"] for s in pop_stats[key]]
            ax.hist(vals, bins=40, alpha=0.4, label=METHOD_LABELS[method],
                    color=colors[method], density=True)

        ax.axvline(actual_dr, color=marker_color, linewidth=2.5, linestyle="--",
                   label=f"{cipher_name} actual ({actual_dr:.1%})")

        ax.set_xlabel("Distinct Ratio")
        ax.set_ylabel("Density")
        ax.set_title(f"Distinct Ratio Distributions — {cipher_name}")
        ax.legend(fontsize=8)
        plt.tight_layout()

        path = save_dir / f"phase8_distinct_ratio_{cipher_name.lower()}.png"
        plt.savefig(str(path), dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close()

    # Combined plot showing all metrics for best discriminators
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    key_metrics = [
        ("distinct_ratio", "Distinct Ratio"),
        ("serial_corr", "Serial Correlation"),
        ("bigram_score", "Bigram Score"),
        ("benford_chi2", "Benford Chi2"),
        ("ic", "Index of Coincidence"),
        ("digit_dist_chi2", "Digit Dist Chi2"),
    ]

    for idx, (metric, title) in enumerate(key_metrics):
        ax = axes[idx // 3][idx % 3]
        for method in METHODS:
            key = f"{method}_520"
            if key not in pop_stats or not pop_stats[key]:
                continue
            vals = [s[metric] for s in pop_stats[key]]
            ax.hist(vals, bins=30, alpha=0.35, label=METHOD_LABELS[method],
                    color=colors[method], density=True)

        b1_val = b1_stats[metric]
        ax.axvline(b1_val, color="red", linewidth=2, linestyle="--", label="B1")
        ax.set_title(title, fontsize=9)
        if idx == 0:
            ax.legend(fontsize=6)

    plt.suptitle("Phase 8: B1 vs Construction Method Distributions", fontsize=12)
    plt.tight_layout()
    path = save_dir / "phase8_multi_metric_b1.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


# ============================================================================
# 6. SUMMARY + CLASSIFICATION MATRIX
# ============================================================================

def print_classification_matrix(
    b1_best: dict[str, str],
    b3_best: dict[str, str],
) -> None:
    """Print final classification matrix."""
    print(f"\n{'='*70}")
    print("CLASSIFICATION MATRIX")
    print(f"{'='*70}")

    # Count votes per method for each cipher
    b1_votes: Counter = Counter(b1_best.values())
    b3_votes: Counter = Counter(b3_best.values())

    header = f"{'':>20}" + "".join(f"{METHOD_LABELS[m]:>14}" for m in METHODS)
    print(f"\n{header}")
    print("-" * (20 + 14 * len(METHODS)))

    for cipher_name, votes in [("B1", b1_votes), ("B3", b3_votes)]:
        row = f"{cipher_name + ' metrics:':<20}"
        best_count = 0
        best_method = ""
        for method in METHODS:
            count = votes.get(method, 0)
            marker = ""
            if count > best_count:
                best_count = count
                best_method = method
            row += f"{count:>14}"
        row_with_best = row
        print(row_with_best)

    # Interpretation
    b1_winner = b1_votes.most_common(1)[0] if b1_votes else ("unknown", 0)
    b3_winner = b3_votes.most_common(1)[0] if b3_votes else ("unknown", 0)

    print(f"\n  B1 best match: {METHOD_LABELS.get(b1_winner[0], b1_winner[0])} "
          f"({b1_winner[1]}/{len(METRICS)} metrics)")
    print(f"  B3 best match: {METHOD_LABELS.get(b3_winner[0], b3_winner[0])} "
          f"({b3_winner[1]}/{len(METRICS)} metrics)")

    # Interpretation guide
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")

    if b1_winner[0] == b3_winner[0]:
        print(f"\n  B1 and B3 match the SAME method: {METHOD_LABELS[b1_winner[0]]}")
        if "gibberish" in b1_winner[0]:
            print("  → Hoaxer likely wrote gibberish and encoded with the DoI.")
            print("    Gillogly strings are artifacts of DoI homophone structure.")
        elif b1_winner[0] == "human_random":
            print("  → Hoaxer likely invented numbers by hand.")
            print("    Gillogly strings at p<10^-12 make this unlikely unless")
            print("    the hoaxer deliberately planted alphabetical runs.")
        elif b1_winner[0] == "random":
            print("  → Pure uniform random. Unlikely given non-random features.")
        elif b1_winner[0] == "genuine":
            print("  → Consistent with genuine encoding. But previous phases")
            print("    showed no key text produces English output.")
    else:
        print(f"\n  B1 and B3 match DIFFERENT methods:")
        print(f"    B1 → {METHOD_LABELS[b1_winner[0]]}")
        print(f"    B3 → {METHOD_LABELS[b3_winner[0]]}")
        print("  → Different construction for each cipher — further evidence of hoax.")
        print("    The hoaxer may have used different techniques or")
        print("    constructed them at different times.")

    print()


# ============================================================================
# 7. RESET PROBABILITY SWEEP (Phase 8c)
# ============================================================================

def run_reset_sweep(
    n_sims: int = 500,
    seed: int = 42,
) -> None:
    """
    Sweep reset_prob to find values matching B1/B3 serial correlation.

    Models a hoaxer who encodes gibberish letters by scanning forward through
    the DoI, but periodically loses their place and restarts from a random
    position. Higher reset_prob = more random, lower serial correlation.
    """
    rng_master = np.random.default_rng(seed)
    index = build_letter_index(BEALE_DOI)
    available = [c for c in string.ascii_lowercase if index.get(c)]
    probs_uniform = np.ones(len(available)) / len(available)

    b1_stats = compute_extended_stats(B1)
    b3_stats = compute_extended_stats(B3)
    b1_sc = b1_stats["serial_corr"]
    b1_dr = b1_stats["distinct_ratio"]
    b3_sc = b3_stats["serial_corr"]
    b3_dr = b3_stats["distinct_ratio"]

    print(f"\n{'='*80}")
    print("PHASE 8c: RESET PROBABILITY SWEEP")
    print(f"{'='*80}")
    print(f"  Model: sequential gibberish + random cursor reset per step")
    print(f"  Targets: B1 sc={b1_sc:.3f} dr={b1_dr:.3f} | "
          f"B3 sc={b3_sc:.3f} dr={b3_dr:.3f}")
    print(f"  Sims per point: {n_sims}")

    reset_probs = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                   0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.0]

    configs = [
        ("B1", 520, b1_sc, b1_dr),
        ("B3", 618, b3_sc, b3_dr),
    ]

    for cipher_name, count, target_sc, target_dr in configs:
        print(f"\n  --- {cipher_name} (n={count}, target sc={target_sc:.3f}, "
              f"dr={target_dr:.3f}) ---")
        print(f"  {'p_reset':>8} {'SC mean':>8} {'SC std':>8} {'DR mean':>8} "
              f"{'DR std':>8} {'SC z':>8} {'DR z':>8}  {'SC match':>9}")
        print(f"  {'-'*72}")

        best_sc_z = 999.0
        best_p_sc = 0.0
        best_combined_z = 999.0
        best_p_combined = 0.0

        for p in reset_probs:
            sc_vals = []
            dr_vals = []

            for i in range(n_sims):
                rng = np.random.default_rng(rng_master.integers(0, 2**31))
                pt = "".join(rng.choice(available, size=count, p=probs_uniform))
                cipher = encode_sequential_book_cipher(
                    pt, BEALE_DOI, reset_prob=p, rng=rng
                )
                arr = np.array(cipher, dtype=float)
                sc_vals.append(float(np.corrcoef(arr[:-1], arr[1:])[0, 1]))
                dr_vals.append(len(set(cipher)) / len(cipher))

            sc_mean = float(np.mean(sc_vals))
            sc_std = float(np.std(sc_vals))
            dr_mean = float(np.mean(dr_vals))
            dr_std = float(np.std(dr_vals))
            sc_z = abs(target_sc - sc_mean) / max(sc_std, 1e-10)
            dr_z = abs(target_dr - dr_mean) / max(dr_std, 1e-10)
            combined_z = (sc_z**2 + dr_z**2) ** 0.5

            marker = ""
            if sc_z < best_sc_z:
                best_sc_z = sc_z
                best_p_sc = p
            if combined_z < best_combined_z:
                best_combined_z = combined_z
                best_p_combined = p
            if sc_z < 2.0:
                marker = "<< match" if sc_z < 1.0 else "< close"

            print(f"  {p:>8.2f} {sc_mean:>8.3f} {sc_std:>8.3f} {dr_mean:>8.3f} "
                  f"{dr_std:>8.3f} {sc_z:>8.1f} {dr_z:>8.1f}  {marker:>9}")

        print(f"\n  Best SC match: p_reset={best_p_sc:.2f} (z={best_sc_z:.2f})")
        print(f"  Best combined (SC+DR): p_reset={best_p_combined:.2f} "
              f"(z={best_combined_z:.2f})")

    print(f"\n  {'='*80}")
    print("  INTERPRETATION")
    print(f"  {'='*80}")
    print("  If best p_reset differs for B1 vs B3, the hoaxer used different")
    print("  levels of care/attention for each cipher.")
    print("  p_reset ≈ 0 → methodical scanning (barely loses place)")
    print("  p_reset ≈ 0.5-0.7 → frequent restarts (sloppy, impatient)")
    print("  p_reset ≈ 1.0 → pure random selection (no sequential structure)")
    print()


# ============================================================================
# 8. PAGE-CONSTRAINED CONSTRUCTION MODEL (Phase 8d)
# ============================================================================

def run_page_model(
    n_sims: int = 500,
    seed: int = 42,
) -> None:
    """
    Final construction model combining all findings.

    B1: Full DoI (all 4 pages), sequential gibberish + reset probability.
        Ward was sloppy/impatient — lost his place frequently.

    B3: DoI[:975] (first 3 pages of 4-page octavo at 325 wpp),
        page-constrained sequential + reset probability.
        Ward was methodical — scanned within each page, rarely lost place.

    The 975-word truncation matches B3's max value exactly and corresponds
    to a page boundary at 325 words/page (standard 1880s octavo).
    """
    rng_master = np.random.default_rng(seed)

    b1_stats = compute_extended_stats(B1)
    b3_stats = compute_extended_stats(B3)
    b1_sc, b1_dr = b1_stats["serial_corr"], b1_stats["distinct_ratio"]
    b3_sc, b3_dr = b3_stats["serial_corr"], b3_stats["distinct_ratio"]

    print(f"\n{'='*80}")
    print("PHASE 8d: PAGE-CONSTRAINED CONSTRUCTION MODEL")
    print(f"{'='*80}")
    print(f"  B3 max value: {max(B3)} (DoI has {len(BEALE_DOI)} words)")
    print(f"  At 325 wpp (standard 1880s octavo): page 3 ends at word 975")
    print(f"  B3 uses first 3 pages only; B1 uses all 4 pages")
    print()

    # --- B1: Full DoI, reset sweep ---
    print(f"  --- B1: Full DoI ({len(BEALE_DOI)} words), sequential + reset ---")
    print(f"  Target: sc={b1_sc:.3f}, dr={b1_dr:.3f}")
    print(f"  {'p_reset':>8} {'SC':>7} {'SC_std':>7} {'DR':>7} {'DR_std':>7}"
          f" {'SC_z':>6} {'DR_z':>6}  notes")
    print(f"  {'-'*72}")

    index_full = build_letter_index(BEALE_DOI)
    avail_full = [c for c in string.ascii_lowercase if index_full.get(c)]
    probs_full = np.ones(len(avail_full)) / len(avail_full)

    b1_best_z = 999.0
    b1_best_p = 0.0

    for p in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        scs, drs = [], []
        for _ in range(n_sims):
            rng = np.random.default_rng(rng_master.integers(0, 2**31))
            pt = "".join(rng.choice(avail_full, size=520, p=probs_full))
            c = encode_sequential_book_cipher(
                pt, BEALE_DOI, reset_prob=p, rng=rng
            )
            arr = np.array(c, dtype=float)
            scs.append(float(np.corrcoef(arr[:-1], arr[1:])[0, 1]))
            drs.append(len(set(c)) / len(c))

        sc_m, sc_s = float(np.mean(scs)), float(np.std(scs))
        dr_m, dr_s = float(np.mean(drs)), float(np.std(drs))
        sc_z = abs(b1_sc - sc_m) / max(sc_s, 1e-10)
        dr_z = abs(b1_dr - dr_m) / max(dr_s, 1e-10)
        combined = (sc_z**2 + dr_z**2) ** 0.5

        if combined < b1_best_z:
            b1_best_z = combined
            b1_best_p = p

        marker = ""
        if sc_z < 1.0 and dr_z < 1.5:
            marker = "<< MATCH"
        elif sc_z < 2.0 and dr_z < 2.0:
            marker = "< close"

        print(f"  {p:>8.2f} {sc_m:>7.3f} {sc_s:>7.3f} {dr_m:>7.3f}"
              f" {dr_s:>7.3f} {sc_z:>6.1f} {dr_z:>6.1f}  {marker}")

    print(f"\n  B1 best: p_reset={b1_best_p:.2f} (combined z={b1_best_z:.2f})")

    # --- B3: DoI[:975], page-constrained, reset sweep ---
    key_975 = BEALE_DOI[:975]
    index_975 = build_letter_index(key_975)
    avail_975 = [c for c in string.ascii_lowercase if index_975.get(c)]
    probs_975 = np.ones(len(avail_975)) / len(avail_975)

    print(f"\n  --- B3: DoI[:975] (3 pages x 325 words), page-constrained ---")
    print(f"  Target: sc={b3_sc:.3f}, dr={b3_dr:.3f}")
    print(f"  {'p_reset':>8} {'SC':>7} {'SC_std':>7} {'DR':>7} {'DR_std':>7}"
          f" {'SC_z':>6} {'DR_z':>6}  notes")
    print(f"  {'-'*72}")

    b3_best_z = 999.0
    b3_best_p = 0.0

    for p in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]:
        scs, drs = [], []
        for _ in range(n_sims):
            rng = np.random.default_rng(rng_master.integers(0, 2**31))
            pt = "".join(rng.choice(avail_975, size=618, p=probs_975))
            c = encode_page_constrained_book_cipher(
                pt, key_975, words_per_page=325, reset_prob=p, rng=rng
            )
            arr = np.array(c, dtype=float)
            scs.append(float(np.corrcoef(arr[:-1], arr[1:])[0, 1]))
            drs.append(len(set(c)) / len(c))

        sc_m, sc_s = float(np.mean(scs)), float(np.std(scs))
        dr_m, dr_s = float(np.mean(drs)), float(np.std(drs))
        sc_z = abs(b3_sc - sc_m) / max(sc_s, 1e-10)
        dr_z = abs(b3_dr - dr_m) / max(dr_s, 1e-10)
        combined = (sc_z**2 + dr_z**2) ** 0.5

        if combined < b3_best_z:
            b3_best_z = combined
            b3_best_p = p

        marker = ""
        if sc_z < 1.0 and dr_z < 1.5:
            marker = "<< MATCH"
        elif sc_z < 2.0 and dr_z < 2.0:
            marker = "< close"

        print(f"  {p:>8.2f} {sc_m:>7.3f} {sc_s:>7.3f} {dr_m:>7.3f}"
              f" {dr_s:>7.3f} {sc_z:>6.1f} {dr_z:>6.1f}  {marker}")

    print(f"\n  B3 best: p_reset={b3_best_p:.2f} (combined z={b3_best_z:.2f})")

    # --- Summary ---
    print(f"\n  {'='*78}")
    print("  CONSTRUCTION MODEL SUMMARY")
    print(f"  {'='*78}")
    print(f"  Hoaxer: James B. Ward, working from a 4-page octavo printing of the DoI")
    print(f"  Method: wrote gibberish letters, encoded by scanning forward through DoI")
    print()
    print(f"  B1 (520 numbers):")
    print(f"    - Used all 4 pages of DoI (words 1-{len(BEALE_DOI)})")
    print(f"    - Lost his place ~{b1_best_p:.0%} of the time (sloppy, impatient)")
    print(f"    - Best match at p_reset={b1_best_p:.2f}")
    print()
    print(f"  B3 (618 numbers):")
    print(f"    - Used only first 3 pages (words 1-975, max value = {max(B3)})")
    print(f"    - Stayed on each page, rarely lost place (~{b3_best_p:.0%} reset rate)")
    print(f"    - Page-constrained selection → lower distinct ratio")
    print(f"    - Best match at p_reset={b3_best_p:.2f}")
    print()
    print(f"  The two ciphers show different construction discipline:")
    print(f"    B1 = hasty, jumping around the full text")
    print(f"    B3 = careful, page-by-page scanning through first 3 pages")
    print()


# ============================================================================
# 9. PAGE BOUNDARY SIGNIFICANCE TEST (Phase 8e)
# ============================================================================

def run_page_boundary_test(
    n_sims: int = 10000,
    seed: int = 42,
) -> None:
    """
    Test whether B3's max=975 aligning with a page boundary is significant.

    Three parts:
      1a. B1 range census — how many numbers exceed DoI length?
      1b. B3 page boundary probability — Monte Carlo + analytical
      1c. Summary
    """
    rng = np.random.default_rng(seed)
    doi_len = len(BEALE_DOI)  # 1311
    wpp = 325  # words per page (standard 1880s octavo)
    page_boundaries = [wpp * i for i in range(1, 5)]  # 325, 650, 975, 1300

    print(f"\n{'='*80}")
    print("PHASE 8e-1: PAGE BOUNDARY SIGNIFICANCE TEST")
    print(f"{'='*80}")
    print(f"  DoI length: {doi_len} words")
    print(f"  Words/page (octavo): {wpp}")
    print(f"  Page boundaries: {page_boundaries}")

    # --- 1a: B1 range census ---
    print(f"\n  --- B1 Range Census ---")
    b1_over = [n for n in B1 if n > doi_len]
    b1_in_range = [n for n in B1 if 1 <= n <= doi_len]
    print(f"  B1 total numbers: {len(B1)}")
    print(f"  Numbers > {doi_len} (out of DoI range): {len(b1_over)} "
          f"({len(b1_over)/len(B1)*100:.1f}%)")
    print(f"  Out-of-range values: {sorted(b1_over)}")

    decoded_b1 = decode_book_cipher(B1, BEALE_DOI)
    n_unknown = decoded_b1.count("?")
    print(f"  B1 decoded '?' chars (out-of-range): {n_unknown}/{len(B1)}")

    b1_max_inrange = max(b1_in_range)
    nearest_boundary = min(page_boundaries, key=lambda b: abs(b - b1_max_inrange))
    print(f"  B1 max value in range: {b1_max_inrange} "
          f"(nearest page boundary: {nearest_boundary}, "
          f"delta={abs(b1_max_inrange - nearest_boundary)})")

    b1_max_all = max(B1)
    print(f"  B1 max value overall: {b1_max_all}")

    # --- 1b: B3 page boundary probability ---
    print(f"\n  --- B3 Page Boundary Probability ---")
    b3_max = max(B3)
    b3_min_useful = min(B3)
    print(f"  B3 max: {b3_max}")
    print(f"  B3 min: {b3_min_useful}")
    print(f"  B3 count: {len(B3)}")

    # Analytical: P(max lands within ±delta of any page boundary)
    delta = 5  # tolerance window
    print(f"\n  Analytical test (tolerance ±{delta} of any boundary):")
    # Range of possible max values: for a 618-number cipher with DoI,
    # max could reasonably be anywhere from ~600 to 1311
    # We test: given max in [max(B3_without_max), doi_len],
    # what's P(within ±delta of a page boundary)?
    b3_sorted = sorted(B3)
    second_max = b3_sorted[-2]
    # Max must be >= second_max and <= doi_len
    feasible_range = doi_len - second_max + 1
    boundary_hits = 0
    for b in page_boundaries:
        lo = max(b - delta, second_max)
        hi = min(b + delta, doi_len)
        if hi >= lo:
            boundary_hits += hi - lo + 1
    p_analytical = boundary_hits / feasible_range if feasible_range > 0 else 1.0
    print(f"  Second-largest B3 value: {second_max}")
    print(f"  Feasible max range: [{second_max}, {doi_len}] ({feasible_range} values)")
    print(f"  Boundary-adjacent values (±{delta}): {boundary_hits}")
    print(f"  P(analytical): {p_analytical:.4f} ({p_analytical*100:.1f}%)")

    # Exact hit (delta=0)
    exact_hits = sum(1 for b in page_boundaries if second_max <= b <= doi_len)
    p_exact = exact_hits / feasible_range if feasible_range > 0 else 1.0
    print(f"  P(exact boundary hit): {p_exact:.4f} ({p_exact*100:.2f}%)")

    # Monte Carlo: generate random cipher sets, check max alignment
    print(f"\n  Monte Carlo ({n_sims} simulations):")
    print(f"  Model: {len(B3)} numbers drawn uniformly from [1, {doi_len}]")
    mc_exact = 0
    mc_near = 0
    mc_maxes = []

    for _ in range(n_sims):
        sim = rng.integers(1, doi_len + 1, size=len(B3))
        sim_max = int(sim.max())
        mc_maxes.append(sim_max)
        if sim_max in page_boundaries:
            mc_exact += 1
        if any(abs(sim_max - b) <= delta for b in page_boundaries):
            mc_near += 1

    p_mc_exact = mc_exact / n_sims
    p_mc_near = mc_near / n_sims
    print(f"  P(max = exact boundary): {p_mc_exact:.4f} ({p_mc_exact*100:.1f}%)")
    print(f"  P(max within ±{delta} of boundary): {p_mc_near:.4f} "
          f"({p_mc_near*100:.1f}%)")

    mc_arr = np.array(mc_maxes)
    print(f"  Simulated max range: [{int(mc_arr.min())}, {int(mc_arr.max())}]")
    print(f"  Simulated max mean: {float(mc_arr.mean()):.1f}")
    print(f"  B3 max=975 percentile: {float(np.mean(mc_arr <= 975) * 100):.1f}%")

    # More realistic MC: sequential encoding (as in our hoax model)
    print(f"\n  Monte Carlo — sequential encoding model ({min(n_sims, 2000)} sims):")
    n_seq_sims = min(n_sims, 2000)
    index_full = build_letter_index(BEALE_DOI)
    avail = [c for c in string.ascii_lowercase if index_full.get(c)]
    probs = np.ones(len(avail)) / len(avail)
    seq_maxes = []
    seq_exact = 0
    seq_near = 0

    for _ in range(n_seq_sims):
        sub_rng = np.random.default_rng(rng.integers(0, 2**31))
        pt = "".join(sub_rng.choice(avail, size=618, p=probs))
        cipher = encode_sequential_book_cipher(
            pt, BEALE_DOI, reset_prob=0.01, rng=sub_rng
        )
        sim_max = max(cipher)
        seq_maxes.append(sim_max)
        if sim_max in page_boundaries:
            seq_exact += 1
        if any(abs(sim_max - b) <= delta for b in page_boundaries):
            seq_near += 1

    p_seq_exact = seq_exact / n_seq_sims
    p_seq_near = seq_near / n_seq_sims
    print(f"  P(max = exact boundary): {p_seq_exact:.4f} ({p_seq_exact*100:.1f}%)")
    print(f"  P(max within ±{delta} of boundary): {p_seq_near:.4f} "
          f"({p_seq_near*100:.1f}%)")

    seq_arr = np.array(seq_maxes)
    print(f"  Sequential max range: [{int(seq_arr.min())}, {int(seq_arr.max())}]")
    print(f"  Sequential max mean: {float(seq_arr.mean()):.1f}")

    # --- 1c: Summary ---
    print(f"\n  --- Summary ---")
    print(f"  B1: {len(B1) - len(b1_over)}/{len(B1)} ({(len(B1)-len(b1_over))/len(B1)*100:.0f}%) "
          f"numbers are within DoI range [1, {doi_len}]")
    print(f"  B3: ALL {len(B3)} numbers within [1, {b3_max}] — "
          f"max is exactly page boundary 3×{wpp}")

    if p_mc_exact < 0.05:
        print(f"  >> B3 page boundary alignment IS statistically significant "
              f"(p={p_mc_exact:.3f})")
    else:
        print(f"  >> B3 page boundary alignment is NOT significant by uniform MC "
              f"(p={p_mc_exact:.3f})")
        print(f"     But the PAGE-CONSTRAINED model explains it directly:")
        print(f"     if the hoaxer only used 3 pages, max CAN'T exceed 975.")

    print(f"\n  Key insight: the significance test is actually the wrong framing.")
    print(f"  The question isn't 'what's P(random max = 975)?'")
    print(f"  It's: 'does max=975 constrain the construction model?'")
    print(f"  Answer: yes — it tells us the hoaxer used exactly 3 of 4 pages.")
    print()


# ============================================================================
# 10. GILLOGLY ARTIFACT TEST (Phase 8e)
# ============================================================================

def generate_alphabet_plaintext(
    length: int,
    alpha_prob: float,
    rng: np.random.Generator,
    available: list[str] | None = None,
) -> str:
    """
    Generate pseudo-gibberish that occasionally falls into alphabetical runs.

    Models a hoaxer writing random letters who sometimes gets lazy and writes
    sequential alphabet characters (a, b, c, d, ...) — a natural human
    fallback when generating arbitrary letters quickly.

    Args:
        length: Number of characters.
        alpha_prob: Per-step probability of switching to/staying in alphabet mode.
        rng: NumPy random generator.
        available: Letters available for random mode (default: a-z).

    Returns:
        Plaintext string mixing random and alphabetical segments.
    """
    if available is None:
        available = list(string.ascii_lowercase)
    avail_set = set(available)
    # Build sorted alphabet of available letters for alphabet mode
    alpha_seq = sorted(available)
    probs = np.ones(len(available)) / len(available)
    alpha_cursor = 0  # position in alpha_seq
    in_alpha_mode = False
    result: list[str] = []

    for _ in range(length):
        if in_alpha_mode:
            # Stay in alpha mode or exit
            if rng.random() < alpha_prob:
                result.append(alpha_seq[alpha_cursor % len(alpha_seq)])
                alpha_cursor += 1
            else:
                in_alpha_mode = False
                result.append(str(rng.choice(available, p=probs)))
        else:
            # Random mode — chance of entering alpha mode
            if rng.random() < alpha_prob * 0.3:  # lower entry rate than stay rate
                in_alpha_mode = True
                alpha_cursor = int(rng.integers(0, len(alpha_seq)))
                result.append(alpha_seq[alpha_cursor % len(alpha_seq)])
                alpha_cursor += 1
            else:
                result.append(str(rng.choice(available, p=probs)))

    return "".join(result)


def run_gillogly_artifact_test(
    n_sims: int = 1000,
    seed: int = 42,
) -> None:
    """
    Test whether Gillogly strings are predicted by the hoax model.

    Four parts:
      2a. DoI natural runs — first letters of consecutive words
      2b. Sequential encoding → no Gillogly strings (confirm negative)
      2c. Sequential numbering → Gillogly strings (confirm positive)
      2d. Hybrid model sweep — find alpha where SC, DR, and Gillogly all match
    """
    rng_master = np.random.default_rng(seed)
    doi_len = len(BEALE_DOI)

    b1_actual_sc = serial_correlation(B1)
    b1_actual_dr = distinct_ratio(B1)["ratio"]
    b1_gillogly = gillogly_strings(B1, BEALE_DOI, min_run=5)
    b1_longest = max((r["length"] for r in b1_gillogly), default=0)

    print(f"\n{'='*80}")
    print("PHASE 8e-2: GILLOGLY ARTIFACT TEST")
    print(f"{'='*80}")
    print(f"  B1 actual: sc={b1_actual_sc:.3f}, dr={b1_actual_dr:.3f}, "
          f"longest Gillogly run={b1_longest}")
    print(f"  B1 Gillogly runs (≥5):")
    for r in b1_gillogly:
        print(f"    pos {r['start']}-{r['end']}: '{r['letters']}' (len={r['length']})")

    # --- 2a: DoI natural runs ---
    print(f"\n  --- 2a: DoI Natural Runs ---")
    doi_first_letters = "".join(
        w[0].lower() for w in BEALE_DOI if w and w[0].isalpha()
    )
    doi_quality = gillogly_quality(doi_first_letters, min_run=5)
    print(f"  DoI first-letter sequence: {len(doi_first_letters)} letters")
    print(f"  Longest ascending run: {doi_quality['longest_run']}")
    print(f"  Runs ≥5:")
    for r in doi_quality["runs"]:
        print(f"    pos {r['start']}-{r['end']}: '{r['letters']}' (len={r['length']})")

    # --- 2b: Sequential encoding → no strings ---
    print(f"\n  --- 2b: Sequential Encoding → Gillogly Runs ---")
    print(f"  Model: encode_sequential_book_cipher (B1 params: 520 numbers, "
          f"full DoI, reset_prob=0.65)")
    print(f"  Generating {n_sims} ciphers...")

    index = build_letter_index(BEALE_DOI)
    avail = [c for c in string.ascii_lowercase if index.get(c)]
    probs = np.ones(len(avail)) / len(avail)

    enc_longest_runs: list[int] = []
    for i in range(n_sims):
        sub_rng = np.random.default_rng(rng_master.integers(0, 2**31))
        pt = "".join(sub_rng.choice(avail, size=520, p=probs))
        cipher = encode_sequential_book_cipher(
            pt, BEALE_DOI, reset_prob=0.65, rng=sub_rng
        )
        decoded = decode_book_cipher(cipher, BEALE_DOI)
        q = gillogly_quality(decoded, min_run=3)
        enc_longest_runs.append(q["longest_run"])

    enc_arr = np.array(enc_longest_runs)
    print(f"  Longest run distribution:")
    print(f"    mean={float(enc_arr.mean()):.1f}, "
          f"median={float(np.median(enc_arr)):.0f}, "
          f"max={int(enc_arr.max())}, "
          f"std={float(enc_arr.std()):.1f}")
    print(f"    P(longest ≥ 11): {float(np.mean(enc_arr >= 11)):.4f}")
    print(f"    P(longest ≥ 17): {float(np.mean(enc_arr >= 17)):.4f}")
    print(f"  >> Encoding random letters produces SHORT runs (gibberish in = "
          f"gibberish out)")

    # --- 2c: Alphabet-mode plaintext → strings ---
    print(f"\n  --- 2c: Alphabet-Mode Plaintext → Gillogly Runs ---")
    print(f"  Model: hoaxer writes gibberish but occasionally falls into")
    print(f"  alphabetical sequences (a,b,c,d,...) — natural human fallback")
    print(f"  alpha_prob=0.5 (high, to show mechanism clearly)")
    print(f"  Generating {n_sims} ciphers...")

    alpha_longest_runs: list[int] = []
    for i in range(n_sims):
        sub_rng = np.random.default_rng(rng_master.integers(0, 2**31))
        pt = generate_alphabet_plaintext(
            length=520, alpha_prob=0.5, rng=sub_rng, available=avail,
        )
        cipher = encode_sequential_book_cipher(
            pt, BEALE_DOI, reset_prob=0.65, rng=sub_rng,
        )
        decoded = decode_book_cipher(cipher, BEALE_DOI)
        q = gillogly_quality(decoded, min_run=3)
        alpha_longest_runs.append(q["longest_run"])

    alpha_arr = np.array(alpha_longest_runs)
    print(f"  Longest run distribution:")
    print(f"    mean={float(alpha_arr.mean()):.1f}, "
          f"median={float(np.median(alpha_arr)):.0f}, "
          f"max={int(alpha_arr.max())}, "
          f"std={float(alpha_arr.std()):.1f}")
    print(f"    P(longest ≥ 11): {float(np.mean(alpha_arr >= 11)):.4f}")
    print(f"    P(longest ≥ 17): {float(np.mean(alpha_arr >= 17)):.4f}")
    if float(np.mean(alpha_arr >= 11)) > 0.01:
        print(f"  >> Alphabet-mode plaintext DOES produce Gillogly-like runs")
    else:
        print(f"  >> Alphabet-mode effect visible but needs tuning")

    # --- 2d: Hybrid model sweep ---
    print(f"\n  --- 2d: Hybrid Model Sweep ---")
    print(f"  Mixing parameter alpha_prob = tendency to fall into alphabet mode")
    print(f"  Hoaxer writes mostly random gibberish, but with probability")
    print(f"  alpha_prob occasionally lapses into alphabetical sequences (a,b,c,...)")
    print(f"  Target: sc≈{b1_actual_sc:.3f}, dr≈{b1_actual_dr:.3f}, "
          f"longest_run≥11 sometimes")
    print(f"  Sims per alpha: {n_sims}")

    alphas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

    print(f"\n  {'a_prob':>6} {'SC':>7} {'SC_z':>6} {'DR':>7} {'DR_z':>6} "
          f"{'LR mean':>7} {'P(≥11)':>7} {'P(≥17)':>7}  notes")
    print(f"  {'-'*74}")

    best_combined = 999.0
    best_alpha = 0.0

    for alpha in alphas:
        sc_vals: list[float] = []
        dr_vals: list[float] = []
        lr_vals: list[int] = []

        for _ in range(n_sims):
            sub_rng = np.random.default_rng(rng_master.integers(0, 2**31))

            # Generate plaintext with alphabet-mode segments
            pt = generate_alphabet_plaintext(
                length=520, alpha_prob=alpha, rng=sub_rng, available=avail,
            )
            # Encode with standard sequential model (B1 params)
            cipher = encode_sequential_book_cipher(
                pt, BEALE_DOI, reset_prob=0.65, rng=sub_rng,
            )

            arr = np.array(cipher, dtype=float)
            sc_vals.append(float(np.corrcoef(arr[:-1], arr[1:])[0, 1]))
            dr_vals.append(len(set(cipher)) / len(cipher))

            decoded = decode_book_cipher(cipher, BEALE_DOI)
            q = gillogly_quality(decoded, min_run=3)
            lr_vals.append(q["longest_run"])

        sc_m = float(np.mean(sc_vals))
        sc_s = float(np.std(sc_vals))
        dr_m = float(np.mean(dr_vals))
        dr_s = float(np.std(dr_vals))
        lr_m = float(np.mean(lr_vals))
        lr_arr = np.array(lr_vals)
        p_11 = float(np.mean(lr_arr >= 11))
        p_17 = float(np.mean(lr_arr >= 17))

        sc_z = abs(b1_actual_sc - sc_m) / max(sc_s, 1e-10)
        dr_z = abs(b1_actual_dr - dr_m) / max(dr_s, 1e-10)
        combined = (sc_z**2 + dr_z**2) ** 0.5

        marker = ""
        if sc_z < 1.5 and dr_z < 1.5 and p_11 > 0.01:
            marker = "<< SWEET SPOT"
        elif sc_z < 2.0 and dr_z < 2.0:
            marker = "< close"

        if combined < best_combined:
            best_combined = combined
            best_alpha = alpha

        print(f"  {alpha:>6.2f} {sc_m:>7.3f} {sc_z:>6.1f} {dr_m:>7.3f} "
              f"{dr_z:>6.1f} {lr_m:>7.1f} {p_11:>7.3f} {p_17:>7.3f}  {marker}")

    print(f"\n  Best SC+DR match: alpha={best_alpha:.2f} "
          f"(combined z={best_combined:.2f})")

    # --- Summary ---
    print(f"\n  {'='*78}")
    print("  GILLOGLY ARTIFACT INTERPRETATION")
    print(f"  {'='*78}")
    print(f"  1. Pure random gibberish → NO Gillogly strings (runs ≈ 5-6)")
    print(f"  2. Alphabet-laced gibberish → Gillogly strings emerge")
    print(f"  3. The mechanism: when a hoaxer writes 'random' letters, they")
    print(f"     occasionally fall into alphabetical sequences (a,b,c,d,...)")
    print(f"     because the alphabet is the strongest letter-sequence in")
    print(f"     human memory. These encode to DoI homophones that decode")
    print(f"     back as alphabetical runs — i.e., Gillogly strings.")
    print(f"\n  B1's 17-char run 'abcdefghiijklmmno' (pos 187-203) decodes from")
    print(f"  cipher numbers {list(B1[187:204])}")
    print(f"  — each pointing to a DoI word starting with the NEXT alphabet")
    print(f"  letter. This is exactly what you get when you encode 'abcde...'")
    print(f"  through the DoI: each letter maps to a homophone for that letter.")
    print(f"\n  Gillogly strings are not counter-evidence to the hoax —")
    print(f"  they are a FINGERPRINT of human-generated 'gibberish' that")
    print(f"  isn't truly random but contaminated with alphabetical patterns.")
    print()


# ============================================================================
# 11. FATIGUE GRADIENT TEST (Phase 8f)
# ============================================================================

def _quarter_serial_correlations(
    cipher: tuple[int, ...] | list[int],
    n_segments: int = 4,
) -> list[float]:
    """Split cipher into n_segments and compute SC for each."""
    n = len(cipher)
    boundaries = [n * i // n_segments for i in range(n_segments + 1)]
    scs = []
    for i in range(n_segments):
        segment = cipher[boundaries[i]:boundaries[i + 1]]
        scs.append(serial_correlation(segment))
    return scs


def run_fatigue_gradient_test(
    n_perms: int = 10000,
    n_model_sims: int = 1000,
    seed: int = 42,
) -> None:
    """
    Phase 8f: Test whether the Q1→Q4 serial correlation gradient is
    statistically significant via permutation test.

    The hoaxer got lazier (more sequential) as each cipher progressed.
    Is this gradient real or could it arise from any random ordering?
    """
    rng = np.random.default_rng(seed)

    print(f"\n{'='*80}")
    print("PHASE 8f: FATIGUE GRADIENT SIGNIFICANCE TEST")
    print(f"{'='*80}")
    print(f"  Permutations: {n_perms:,}")
    print(f"  H0: ordering has no positional structure (gradient = 0)")
    print(f"  H1: serial correlation increases Q1→Q4 (hoaxer fatigue)")
    print()

    # ------------------------------------------------------------------
    # Step 1: Observed quarter-by-quarter SC
    # ------------------------------------------------------------------
    print("  STEP 1: Observed quarter serial correlations")
    print(f"  {'':>4} {'Q1':>7} {'Q2':>7} {'Q3':>7} {'Q4':>7}  {'slope':>7} {'Q4-Q1':>7}")
    print(f"  {'-'*52}")

    ciphers = {"B1": B1, "B3": B3}
    observed: dict[str, dict[str, float]] = {}

    for name, cipher in ciphers.items():
        qscs = _quarter_serial_correlations(cipher, 4)
        slope = float(np.polyfit([1, 2, 3, 4], qscs, 1)[0])
        diff = qscs[3] - qscs[0]
        observed[name] = {"slope": slope, "diff": diff, "qscs": qscs}
        print(f"  {name:>4} {qscs[0]:>7.2f} {qscs[1]:>7.2f} "
              f"{qscs[2]:>7.2f} {qscs[3]:>7.2f}  {slope:>7.3f} {diff:>7.2f}")

    print()

    # ------------------------------------------------------------------
    # Step 2: Permutation test (10K permutations)
    # ------------------------------------------------------------------
    print(f"  STEP 2: Permutation test ({n_perms:,} permutations)")
    print(f"  {'cipher':>6} {'stat':>8} {'observed':>10} {'p-value':>10} {'sig':>5}")
    print(f"  {'-'*48}")

    p_values: dict[str, dict[str, float]] = {}

    for name, cipher in ciphers.items():
        obs_slope = observed[name]["slope"]
        obs_diff = observed[name]["diff"]
        perm_slopes = np.empty(n_perms)
        perm_diffs = np.empty(n_perms)
        arr = np.array(cipher)

        for i in range(n_perms):
            perm = rng.permutation(arr)
            qscs = _quarter_serial_correlations(perm, 4)
            perm_slopes[i] = float(np.polyfit([1, 2, 3, 4], qscs, 1)[0])
            perm_diffs[i] = qscs[3] - qscs[0]

        p_slope = float(np.mean(perm_slopes >= obs_slope))
        p_diff = float(np.mean(perm_diffs >= obs_diff))
        p_values[name] = {"slope": p_slope, "diff": p_diff}

        for stat, obs, pv in [("slope", obs_slope, p_slope),
                               ("Q4-Q1", obs_diff, p_diff)]:
            sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
            pv_str = f"<{1/n_perms:.4f}" if pv == 0.0 else f"{pv:.4f}"
            print(f"  {name:>6} {stat:>8} {obs:>10.4f} {pv_str:>10} {sig:>5}")

    print()

    # ------------------------------------------------------------------
    # Step 3: Robustness across partition sizes
    # ------------------------------------------------------------------
    print("  STEP 3: Robustness across partition sizes")
    for name, cipher in ciphers.items():
        print(f"\n  {name}:")
        print(f"  {'segments':>10} {'obs_slope':>10} {'p_value':>10} {'sig':>5}")
        print(f"  {'-'*40}")
        arr = np.array(cipher)

        for n_seg in [3, 4, 5, 6, 8]:
            obs_qscs = _quarter_serial_correlations(cipher, n_seg)
            xs = list(range(1, n_seg + 1))
            obs_slope = float(np.polyfit(xs, obs_qscs, 1)[0])

            count_ge = 0
            for _ in range(n_perms):
                perm = rng.permutation(arr)
                perm_qscs = _quarter_serial_correlations(perm, n_seg)
                perm_slope = float(np.polyfit(xs, perm_qscs, 1)[0])
                if perm_slope >= obs_slope:
                    count_ge += 1
            p = count_ge / n_perms

            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            pv_str = f"<{1/n_perms:.4f}" if p == 0.0 else f"{p:.4f}"
            print(f"  {n_seg:>10} {obs_slope:>10.4f} {pv_str:>10} {sig:>5}")

    print()

    # ------------------------------------------------------------------
    # Step 4: Combined significance
    # ------------------------------------------------------------------
    print("  STEP 4: Combined significance (B1 × B3 under independence)")
    p_b1 = p_values["B1"]["slope"]
    p_b3 = p_values["B3"]["slope"]
    # Use floor of 1/n_perms for zero counts to avoid 0 × anything = 0
    p_b1_adj = max(p_b1, 1 / n_perms)
    p_b3_adj = max(p_b3, 1 / n_perms)
    p_combined = p_b1_adj * p_b3_adj
    fmt_p = lambda p: f"<{1/n_perms:.4f}" if p == 0.0 else f"{p:.4f}"
    print(f"  B1 slope p = {fmt_p(p_b1)}")
    print(f"  B3 slope p = {fmt_p(p_b3)}")
    print(f"  Combined p ≤ {p_combined:.2e}")
    if p_combined < 0.01:
        print(f"  → Significant at p<0.01: gradient is real in BOTH ciphers")
    elif p_combined < 0.05:
        print(f"  → Significant at p<0.05")
    else:
        print(f"  → Not significant at p<0.05 when combined")
    print()

    # ------------------------------------------------------------------
    # Step 5: Model prediction check
    # ------------------------------------------------------------------
    print(f"  STEP 5: Model prediction ({n_model_sims:,} simulated ciphers)")
    print(f"  Does the phase 8d construction model inherently produce a gradient?")
    print()

    index_full = build_letter_index(BEALE_DOI)
    avail_full = [c for c in string.ascii_lowercase if index_full.get(c)]
    probs_full = np.ones(len(avail_full)) / len(avail_full)

    key_975 = BEALE_DOI[:975]
    index_975 = build_letter_index(key_975)
    avail_975 = [c for c in string.ascii_lowercase if index_975.get(c)]
    probs_975 = np.ones(len(avail_975)) / len(avail_975)

    models = {
        "B1 model": {
            "length": 520,
            "encoder": lambda pt, r: encode_sequential_book_cipher(
                pt, BEALE_DOI, reset_prob=0.65, rng=r),
            "avail": avail_full,
            "probs": probs_full,
        },
        "B3 model": {
            "length": 618,
            "encoder": lambda pt, r: encode_page_constrained_book_cipher(
                pt, key_975, words_per_page=325, reset_prob=0.01, rng=r),
            "avail": avail_975,
            "probs": probs_975,
        },
    }

    print(f"  {'model':>10} {'mean_slope':>11} {'std_slope':>11} {'pct_positive':>13}")
    print(f"  {'-'*50}")

    for mname, cfg in models.items():
        slopes = []
        for _ in range(n_model_sims):
            sim_rng = np.random.default_rng(rng.integers(0, 2**31))
            pt = "".join(sim_rng.choice(cfg["avail"], size=cfg["length"],
                                        p=cfg["probs"]))
            c = cfg["encoder"](pt, sim_rng)
            qscs = _quarter_serial_correlations(c, 4)
            slopes.append(float(np.polyfit([1, 2, 3, 4], qscs, 1)[0]))

        mean_s = float(np.mean(slopes))
        std_s = float(np.std(slopes))
        pct_pos = float(np.mean(np.array(slopes) > 0)) * 100
        print(f"  {mname:>10} {mean_s:>11.4f} {std_s:>11.4f} {pct_pos:>12.1f}%")

    print()
    print("  INTERPRETATION:")
    print("  If model mean_slope ≈ 0 → fatigue gradient is INDEPENDENT evidence")
    print("  If model mean_slope > 0 → gradient is PREDICTED by model (confirmation)")
    print()


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 8: Hoax construction method analysis"
    )
    parser.add_argument("--generate", action="store_true",
                        help="Generate all populations + compute stats")
    parser.add_argument("--analyze", action="store_true",
                        help="Run classification (requires --generate first or --all)")
    parser.add_argument("--human-tests", action="store_true",
                        help="Run human-random specific tests on B1/B3")
    parser.add_argument("--reset-sweep", action="store_true",
                        help="Sweep reset probability for sequential encoding")
    parser.add_argument("--page-model", action="store_true",
                        help="Page-constrained construction model (final)")
    parser.add_argument("--boundary-test", action="store_true",
                        help="B3 page boundary significance test (phase 8e)")
    parser.add_argument("--gillogly-test", action="store_true",
                        help="Gillogly strings as hoax artifact (phase 8e)")
    parser.add_argument("--fatigue-test", action="store_true",
                        help="Fatigue gradient significance test (phase 8f)")
    parser.add_argument("--all", action="store_true",
                        help="Everything: generate, analyze, human-tests, sweeps, plots")
    parser.add_argument("--n-sims", type=int, default=1000,
                        help="Simulations per population (default: 1000)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--save-dir", type=str, default=".",
                        help="Directory for output files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not any([args.generate, args.analyze, args.human_tests,
                args.reset_sweep, args.page_model,
                args.boundary_test, args.gillogly_test,
                args.fatigue_test, args.all]):
        parser.print_help()
        sys.exit(1)

    save_dir = Path(args.save_dir)
    do_generate = args.generate or args.all
    do_analyze = args.analyze or args.all
    do_human = args.human_tests or args.all
    do_plots = not args.no_plots and args.all

    print("=" * 70)
    print("PHASE 8: HOAX CONSTRUCTION METHOD ANALYSIS")
    print(f"Simulations: {args.n_sims} per method per cipher length")
    print("=" * 70)

    # Compute real cipher stats
    print("\nComputing stats for real ciphers...")
    b1_stats = compute_extended_stats(B1)
    b2_stats = compute_extended_stats(B2)
    b3_stats = compute_extended_stats(B3)

    print(f"  B2 distinct ratio: {b2_stats['distinct_ratio']:.3f} (expect ~0.236)")
    print(f"  B1 distinct ratio: {b1_stats['distinct_ratio']:.3f}")
    print(f"  B3 distinct ratio: {b3_stats['distinct_ratio']:.3f}")

    populations = None
    pop_stats = None

    if do_generate:
        print("\n--- Generating populations ---")
        populations = generate_populations(n_sims=args.n_sims, seed=args.seed)

        print("\n--- Computing population stats ---")
        pop_stats = compute_population_stats(populations, args.n_sims)

    if do_analyze:
        if pop_stats is None:
            print("\nERROR: Need --generate or --all to compute stats first.")
            sys.exit(1)

        # B2 calibration
        print("\n--- B2 calibration (should classify as Genuine) ---")
        for metric in ["distinct_ratio", "bigram_score", "ic", "serial_corr"]:
            key = "genuine_618"  # B2 is 763 but closest match
            if key in pop_stats:
                vals = [s[metric] for s in pop_stats[key]]
                pct = float(np.mean(np.array(vals) <= b2_stats[metric]) * 100)
                print(f"  B2 {metric}: {b2_stats[metric]:.4f} "
                      f"(genuine pctile: {pct:.1f}%)")

        # Classification
        b1_best = print_comparison_table(pop_stats, b1_stats, "B1", "520")
        b3_best = print_comparison_table(pop_stats, b3_stats, "B3", "618")

        # Distinct ratio deep dive
        print_distinct_ratio_analysis(pop_stats, b1_stats, b3_stats)

        # Classification matrix
        print_classification_matrix(b1_best, b3_best)

    if do_human:
        print_human_random_tests(B1, "B1")
        print_human_random_tests(B3, "B3")

    if do_plots and pop_stats is not None:
        print("\n--- Generating plots ---")
        plot_distinct_ratio_histograms(pop_stats, b1_stats, b3_stats, save_dir)

    if args.reset_sweep or args.all:
        run_reset_sweep(n_sims=args.n_sims, seed=args.seed)

    if args.page_model or args.all:
        run_page_model(n_sims=args.n_sims, seed=args.seed)

    if args.boundary_test or args.all:
        run_page_boundary_test(n_sims=args.n_sims, seed=args.seed)

    if args.gillogly_test or args.all:
        run_gillogly_artifact_test(n_sims=args.n_sims, seed=args.seed)

    if args.fatigue_test or args.all:
        run_fatigue_gradient_test(
            n_perms=args.n_sims, n_model_sims=args.n_sims, seed=args.seed
        )

    print("\n" + "=" * 70)
    print("PHASE 8 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
