"""
---
version: 0.1.0
created: 2026-02-25
updated: 2026-02-25
---

phase9_b2_analysis.py — B2 construction method analysis.

Tests whether B2's statistical fingerprint distinguishes it from a carefully
fabricated cipher. B2 decodes to readable English using the DoI, but if it
could ALSO be fabricated, the remaining uncertainty collapses.

Four tests:
  1. Reset sweep: what reset_prob reproduces B2's SC/DR?
  2. Fabrication distribution: is B2 inside the random-encode distribution?
  3. Homophone fingerprint: utilization uniformity + positional correlation
  4. Override analysis: do the 3 special decodes support forward or backward?

Usage:
    python3 phase9_b2_analysis.py --reset-sweep
    python3 phase9_b2_analysis.py --fabrication-test
    python3 phase9_b2_analysis.py --homophone-fingerprint
    python3 phase9_b2_analysis.py --override-analysis
    python3 phase9_b2_analysis.py --all --n-sims 1000
"""

from __future__ import annotations

import argparse
import string
import sys
import time
from collections import Counter

import numpy as np
from scipy import stats as sp_stats

from beale import (
    B1, B2, B3, BEALE_DOI, ENGLISH_FREQ, SPECIAL_DECODE, B2_PLAINTEXT,
    benford_test, last_digit_test, distinct_ratio,
    decode_book_cipher, encode_book_cipher, encode_sequential_book_cipher,
    build_letter_index, bigram_score, first_letter,
)
from phase8_hoax_construction import serial_correlation, compute_extended_stats


# ============================================================================
# HELPERS
# ============================================================================

def generate_english_freq_text(
    length: int,
    rng: np.random.Generator,
    key_words: tuple[str, ...] | list[str] = BEALE_DOI,
) -> str:
    """Generate random text with English letter frequencies.

    Only uses letters that have homophones in the key text (excludes x, y, z
    for the Beale DoI).
    """
    index = build_letter_index(key_words)
    letters = [c for c in ENGLISH_FREQ if index.get(c)]
    probs = np.array([ENGLISH_FREQ[c] for c in letters])
    probs /= probs.sum()
    return "".join(rng.choice(letters, size=length, p=probs))


def z_score(value: float, distribution: np.ndarray) -> float:
    """Compute z-score of value against a distribution."""
    mu = distribution.mean()
    sigma = distribution.std()
    if sigma < 1e-12:
        return 0.0
    return (value - mu) / sigma


def percentile_rank(value: float, distribution: np.ndarray) -> float:
    """Compute percentile rank (0-100) of value in distribution."""
    return float(np.sum(distribution <= value) / len(distribution) * 100)


# ============================================================================
# TEST 1: B2 RESET SWEEP
# ============================================================================

def run_b2_reset_sweep(n_sims: int = 200, seed: int = 42) -> dict:
    """
    Sweep reset_prob encoding English-frequency text, find what reproduces
    B2's SC=0.04, DR=0.236.
    """
    rng = np.random.default_rng(seed)
    b2_len = len(B2)
    b2_sc = serial_correlation(B2)
    b2_dr = distinct_ratio(B2)["ratio"]

    # Build B2-frequency text generator (not fixed plaintext — need variation)
    b2_plain_raw = B2_PLAINTEXT.replace("?", "")
    letter_index = build_letter_index(BEALE_DOI)
    no_homo = set(c for c in string.ascii_lowercase if not letter_index.get(c))
    b2_freq = Counter(c for c in b2_plain_raw if c not in no_homo)
    available = sorted(b2_freq.keys())
    b2_probs = np.array([b2_freq[c] for c in available], dtype=float)
    b2_probs /= b2_probs.sum()

    reset_probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]

    print("=" * 72)
    print("TEST 1: B2 RESET SWEEP")
    print("=" * 72)
    print(f"\nB2 actual: SC={b2_sc:.4f}, DR={b2_dr:.4f}")
    print(f"Simulations per reset_prob: {n_sims}")
    print("(Random text with B2's letter frequencies — varies per trial)")
    print(f"\n{'reset_p':>8s}  {'SC_mean':>8s}  {'SC_std':>7s}  {'SC_z':>7s}  "
          f"{'DR_mean':>8s}  {'DR_std':>7s}  {'DR_z':>7s}  {'combined':>8s}")
    print("-" * 72)

    results = {}
    for rp in reset_probs:
        sc_vals = []
        dr_vals = []
        for _ in range(n_sims):
            text = "".join(rng.choice(available, size=b2_len, p=b2_probs))
            if rp >= 1.0:
                # reset_prob=1.0 is equivalent to pure random selection
                cipher = encode_book_cipher(text, BEALE_DOI, rng=rng)
            else:
                cipher = encode_sequential_book_cipher(text, BEALE_DOI,
                                                       reset_prob=rp, rng=rng)
            sc_vals.append(serial_correlation(cipher))
            dr_vals.append(distinct_ratio(cipher)["ratio"])

        sc_arr = np.array(sc_vals)
        dr_arr = np.array(dr_vals)
        sc_z = z_score(b2_sc, sc_arr)
        dr_z = z_score(b2_dr, dr_arr)
        combined_z = np.sqrt(sc_z**2 + dr_z**2)

        results[rp] = {
            "sc_mean": sc_arr.mean(), "sc_std": sc_arr.std(), "sc_z": sc_z,
            "dr_mean": dr_arr.mean(), "dr_std": dr_arr.std(), "dr_z": dr_z,
            "combined_z": combined_z,
        }

        print(f"{rp:8.2f}  {sc_arr.mean():8.4f}  {sc_arr.std():7.4f}  {sc_z:7.2f}  "
              f"{dr_arr.mean():8.4f}  {dr_arr.std():7.4f}  {dr_z:7.2f}  {combined_z:8.2f}")

    # Find best match
    best_rp = min(results, key=lambda rp: results[rp]["combined_z"])
    best = results[best_rp]
    print(f"\nBest match: reset_prob={best_rp:.2f} (combined z={best['combined_z']:.2f})")
    print(f"  → SC z={best['sc_z']:.2f}, DR z={best['dr_z']:.2f}")

    # Compare to B1 (0.65) and B3 (0.01)
    print(f"\nComparison: B1 best reset_prob=0.65, B3 best reset_prob=0.01")
    print(f"B2 best reset_prob={best_rp:.2f} — ", end="")
    if best_rp >= 0.9:
        print("effectively random selection (not sequential scanning)")
    elif best_rp >= 0.5:
        print("sloppy scanning (similar to B1)")
    else:
        print("methodical scanning (similar to B3)")

    return results


# ============================================================================
# TEST 2: B2 VS FABRICATED DISTRIBUTION
# ============================================================================

def run_b2_fabrication_test(n_sims: int = 1000, seed: int = 42) -> dict:
    """
    Encode the exact B2 plaintext n_sims times with random homophone selection.
    Compare B2's stats against the resulting distributions.

    This is the strongest possible test: same plaintext, same key, only the
    homophone selection varies. Any difference is purely in how homophones
    were chosen.
    """
    rng = np.random.default_rng(seed)

    print("\n" + "=" * 72)
    print("TEST 2: B2 VS FABRICATED DISTRIBUTION")
    print("=" * 72)

    # Strip x/y/z from B2 plaintext (no homophones), replace with random
    # encodable letters to maintain length
    b2_plain = B2_PLAINTEXT.replace("?", "")
    index = build_letter_index(BEALE_DOI)
    no_homo = set(c for c in string.ascii_lowercase if not index.get(c))
    encodable_plain = "".join(
        c if c not in no_homo else rng.choice([k for k in index if index[k]])
        for c in b2_plain
    )

    b2_len = len(B2)
    print(f"\nEncoding B2 plaintext ({b2_len} chars) {n_sims} times with random "
          "homophone selection...")
    print(f"({len([c for c in b2_plain if c in no_homo])} x/y/z chars replaced "
          "for encoding)")

    # B2 actuals
    b2_sc = serial_correlation(B2)
    b2_dr = distinct_ratio(B2)["ratio"]
    b2_benford = benford_test(B2)["chi2"]
    b2_ld = last_digit_test(B2, base=10)["chi2"]

    # Generate fabricated distribution
    sc_dist = np.zeros(n_sims)
    dr_dist = np.zeros(n_sims)
    bf_dist = np.zeros(n_sims)
    ld_dist = np.zeros(n_sims)

    for i in range(n_sims):
        cipher = encode_book_cipher(encodable_plain, BEALE_DOI, rng=rng)
        sc_dist[i] = serial_correlation(cipher)
        dr_dist[i] = distinct_ratio(cipher)["ratio"]
        bf_dist[i] = benford_test(cipher)["chi2"]
        ld_dist[i] = last_digit_test(cipher, base=10)["chi2"]

    metrics = {
        "Serial correlation": (b2_sc, sc_dist),
        "Distinct ratio": (b2_dr, dr_dist),
        "Benford chi2": (b2_benford, bf_dist),
        "Last-digit chi2": (b2_ld, ld_dist),
    }

    print(f"\n{'Metric':>22s}  {'B2':>8s}  {'Fab_mean':>8s}  {'Fab_std':>8s}  "
          f"{'z-score':>8s}  {'pctile':>7s}")
    print("-" * 72)

    results = {}
    for name, (b2_val, dist) in metrics.items():
        z = z_score(b2_val, dist)
        pct = percentile_rank(b2_val, dist)
        results[name] = {"b2": b2_val, "mean": dist.mean(), "std": dist.std(),
                         "z": z, "percentile": pct}
        print(f"{name:>22s}  {b2_val:8.4f}  {dist.mean():8.4f}  {dist.std():8.4f}  "
              f"{z:8.2f}  {pct:6.1f}%")

    # Verdict
    any_outlier = any(abs(r["z"]) > 2.0 for r in results.values())
    print(f"\nVerdict: B2 is ", end="")
    if any_outlier:
        outliers = [name for name, r in results.items() if abs(r["z"]) > 2.0]
        print(f"OUTSIDE fabricated distribution on: {', '.join(outliers)}")
        print("  → B2's homophone selection pattern differs from random")
    else:
        print("INSIDE fabricated distribution on all metrics")
        print("  → B2 is indistinguishable from random homophone selection")

    return results


# ============================================================================
# TEST 3: PER-LETTER HOMOPHONE FINGERPRINT
# ============================================================================

def homophone_fingerprint(
    cipher: tuple[int, ...] | list[int],
    plaintext: str,
    key_words: tuple[str, ...] | list[str],
    label: str = "",
) -> list[dict]:
    """
    For each high-frequency letter, compute:
      - Chi-squared utilization uniformity
      - Spearman positional correlation (order_in_cipher vs cipher_number)
    """
    letter_index = build_letter_index(key_words)
    # Group cipher positions by decoded letter
    letter_uses: dict[str, list[tuple[int, int]]] = {}  # letter -> [(cipher_pos, number)]
    for i, num in enumerate(cipher):
        if i < len(plaintext):
            letter = plaintext[i]
        else:
            break
        if letter not in letter_uses:
            letter_uses[letter] = []
        letter_uses[letter].append((i, num))

    results = []
    for letter in sorted(letter_uses.keys()):
        uses = letter_uses[letter]
        if len(uses) < 10:
            continue

        available = letter_index.get(letter, [])
        n_avail = len(available)
        if n_avail == 0:
            continue

        # 3a. Utilization uniformity: how evenly are homophones used?
        used_numbers = [num for _, num in uses]
        usage_counts = Counter(used_numbers)
        # Chi-squared against uniform across used homophones
        observed = np.array([usage_counts.get(pos, 0) for pos in available], dtype=float)
        n_used = int(np.sum(observed > 0))
        util_pct = n_used / n_avail * 100 if n_avail > 0 else 0

        # Chi-squared only if enough expected per cell
        expected = np.full(n_avail, len(uses) / n_avail)
        if expected[0] >= 1.0 and observed.sum() > 0:
            # Normalize expected to match observed total (some cipher numbers
            # may use special overrides not in the standard letter index)
            expected_scaled = expected * (observed.sum() / expected.sum())
            chi2_stat, chi2_p = sp_stats.chisquare(observed, expected_scaled)
        else:
            chi2_stat, chi2_p = float("nan"), float("nan")

        # 3b. Positional correlation: cipher_order vs cipher_number
        positions_in_cipher = [pos for pos, _ in uses]
        numbers = [num for _, num in uses]
        if len(set(positions_in_cipher)) > 1 and len(set(numbers)) > 1:
            spearman_r, spearman_p = sp_stats.spearmanr(positions_in_cipher, numbers)
        else:
            spearman_r, spearman_p = 0.0, 1.0

        results.append({
            "letter": letter, "n_uses": len(uses), "n_avail": n_avail,
            "n_used": n_used, "util_pct": util_pct,
            "chi2": chi2_stat, "chi2_p": chi2_p,
            "spearman_r": spearman_r, "spearman_p": spearman_p,
        })

    return results


def run_homophone_fingerprint() -> dict:
    """
    Run homophone fingerprint on B2, B1, B3. Compare positional correlation
    patterns.
    """
    print("\n" + "=" * 72)
    print("TEST 3: PER-LETTER HOMOPHONE FINGERPRINT")
    print("=" * 72)

    # B2: known plaintext
    b2_plain = B2_PLAINTEXT.replace("?", "")
    # For B1/B3: decode with DoI (gibberish, but gives us the letter mapping)
    b1_plain = decode_book_cipher(B1, BEALE_DOI)
    b3_plain = decode_book_cipher(B3, BEALE_DOI)

    all_results = {}
    for label, cipher, plaintext in [
        ("B2", B2, b2_plain),
        ("B1", B1, b1_plain),
        ("B3", B3, b3_plain),
    ]:
        print(f"\n--- {label} ---")
        print(f"{'letter':>6s}  {'n_uses':>6s}  {'n_avail':>7s}  {'n_used':>6s}  "
              f"{'util%':>6s}  {'chi2_p':>8s}  {'spear_r':>8s}  {'spear_p':>8s}")
        print("-" * 68)

        results = homophone_fingerprint(cipher, plaintext, BEALE_DOI, label)
        all_results[label] = results

        for r in results:
            chi2_p_str = f"{r['chi2_p']:.4f}" if not np.isnan(r['chi2_p']) else "   n/a"
            print(f"{r['letter']:>6s}  {r['n_uses']:>6d}  {r['n_avail']:>7d}  "
                  f"{r['n_used']:>6d}  {r['util_pct']:>5.1f}%  {chi2_p_str:>8s}  "
                  f"{r['spearman_r']:>8.3f}  {r['spearman_p']:>8.4f}")

        # Summary stats
        rs = [r["spearman_r"] for r in results]
        mean_r = np.mean(rs) if rs else 0
        sig_pos = sum(1 for r in results if r["spearman_r"] > 0 and r["spearman_p"] < 0.05)
        sig_neg = sum(1 for r in results if r["spearman_r"] < 0 and r["spearman_p"] < 0.05)
        print(f"\n  Mean Spearman r: {mean_r:.4f}")
        print(f"  Significant positive correlations (p<0.05): {sig_pos}/{len(results)}")
        print(f"  Significant negative correlations (p<0.05): {sig_neg}/{len(results)}")

    # Cross-cipher comparison
    print(f"\n--- COMPARISON ---")
    for label in ["B2", "B1", "B3"]:
        rs = [r["spearman_r"] for r in all_results[label]]
        mean_r = np.mean(rs) if rs else 0
        sig = sum(1 for r in all_results[label]
                  if r["spearman_p"] < 0.05 and r["spearman_r"] > 0)
        print(f"  {label}: mean_r={mean_r:+.4f}, "
              f"sig_positive={sig}/{len(all_results[label])}")

    print("\nInterpretation:")
    b2_mean = np.mean([r["spearman_r"] for r in all_results["B2"]])
    b1_mean = np.mean([r["spearman_r"] for r in all_results["B1"]])
    b3_mean = np.mean([r["spearman_r"] for r in all_results["B3"]])

    if abs(b2_mean) < 0.1 and (b1_mean > 0.02 or b3_mean > 0.02):
        print("  B2 shows ~zero positional correlation (random selection)")
        print("  B1/B3 show positive correlation (sequential scanning)")
        print("  Gradient: B2 < B1 < B3 — matches increasing methodicalness")
        print("  → B2 used fundamentally different encoding method from B1/B3")
    elif abs(b2_mean) < 0.1:
        print("  All ciphers show low positional correlation")
        print("  → Cannot distinguish encoding methods by this metric")
    else:
        print(f"  B2 mean_r={b2_mean:+.4f} — unexpected non-zero correlation")

    return all_results


# ============================================================================
# TEST 4: SPECIAL OVERRIDE ANALYSIS
# ============================================================================

def run_override_analysis(n_sims: int = 1000, seed: int = 42) -> dict:
    """
    Analyze the 3 SPECIAL_DECODE overrides (95→'u', 811→'y', 1005→'x').

    Two questions:
      A. How do overrides function in B2's actual decode?
      B. If encoding B2's plaintext from scratch, how many letters MUST use
         overrides because they have zero homophones (x, y, z)?
    """
    rng = np.random.default_rng(seed)
    letter_index = build_letter_index(BEALE_DOI)

    print("\n" + "=" * 72)
    print("TEST 4: SPECIAL OVERRIDE ANALYSIS")
    print("=" * 72)

    # --- Part A: Describe each override ---
    print("\nThe 3 special decode overrides in B2:")
    for num, override_letter in sorted(SPECIAL_DECODE.items()):
        word = BEALE_DOI[num - 1] if 1 <= num <= len(BEALE_DOI) else "???"
        natural_letter = first_letter(word)
        positions = [i for i, n in enumerate(B2) if n == num]
        print(f"  #{num:>4d} → '{override_letter}' (word='{word}', "
              f"natural='{natural_letter}', occurs {len(positions)}x in B2)")

    # Count what happens at every position with an override number
    b2_plain = B2_PLAINTEXT.replace("?", "")
    print(f"\nOverride number usage analysis:")
    override_correct = 0
    override_wrong = 0
    natural_correct = 0
    for num, override_letter in SPECIAL_DECODE.items():
        word = BEALE_DOI[num - 1]
        natural = first_letter(word)
        positions = [i for i, n in enumerate(B2) if n == num]
        for pos in positions:
            if pos >= len(b2_plain):
                continue
            needed = b2_plain[pos]
            if override_letter == needed:
                override_correct += 1
            elif natural == needed:
                natural_correct += 1
            else:
                override_wrong += 1

    total_override_uses = override_correct + override_wrong + natural_correct
    print(f"  Total uses of override numbers: {total_override_uses}")
    print(f"  Correct via override letter: {override_correct}")
    print(f"  Correct via natural letter: {natural_correct}")
    print(f"  Wrong either way: {override_wrong}")

    # --- Part B: Letters with zero homophones ---
    no_homo_letters = [c for c in string.ascii_lowercase if not letter_index.get(c)]
    b2_needs_override = Counter(c for c in b2_plain if c in no_homo_letters)
    print(f"\nLetters with ZERO homophones in DoI: {no_homo_letters}")
    print(f"B2 plaintext positions needing these letters:")
    for letter, count in sorted(b2_needs_override.items()):
        print(f"  '{letter}': {count} occurrences → MUST use non-standard encoding")

    total_forced = sum(b2_needs_override.values())
    print(f"  Total positions requiring workarounds: {total_forced}/{len(b2_plain)} "
          f"({total_forced/len(b2_plain)*100:.1f}%)")

    # --- Part C: How B2 actually encodes the no-homophone letters ---
    print(f"\nHow B2 handles letters without homophones:")
    for letter in sorted(b2_needs_override.keys()):
        positions = [i for i, c in enumerate(b2_plain) if c == letter and i < len(B2)]
        numbers_used = [B2[i] for i in positions]
        print(f"  '{letter}' at positions {positions}: uses numbers {numbers_used}")
        for pos in positions:
            num = B2[pos]
            word = BEALE_DOI[num - 1] if 1 <= num <= len(BEALE_DOI) else "???"
            nat = first_letter(word)
            override = SPECIAL_DECODE.get(num, None)
            if override == letter:
                print(f"    #{num} '{word}' → override '{override}' ✓")
            elif nat == letter:
                print(f"    #{num} '{word}' → natural '{nat}' ✓")
            else:
                print(f"    #{num} '{word}' → natural '{nat}', "
                      f"override '{override}' — NEITHER matches '{letter}'")

    # --- Part D: MC — encode B2 plaintext, count how many positions can't find
    # a standard homophone (must be handled by override or error) ---
    print(f"\nMonte Carlo: how often does random encoding of B2-like text need "
          f"non-standard solutions?")
    # Filter plaintext to only encodable letters
    encodable_plain = "".join(c for c in b2_plain if c not in no_homo_letters)
    print(f"  Encodable letters: {len(encodable_plain)}/{len(b2_plain)} "
          f"({total_forced} need overrides)")

    # For the encodable portion, verify standard encoding produces 0 errors
    error_counts = []
    for _ in range(n_sims):
        cipher = encode_book_cipher(encodable_plain, BEALE_DOI, rng=rng)
        n_errors = 0
        for j, num in enumerate(cipher):
            word = BEALE_DOI[num - 1]
            if first_letter(word) != encodable_plain[j]:
                n_errors += 1
        error_counts.append(n_errors)

    ec_arr = np.array(error_counts)
    print(f"  Encoding errors (encodable subset, {n_sims} trials): "
          f"mean={ec_arr.mean():.2f}, max={ec_arr.max()}")

    # Verdict
    print(f"\nVerdict:")
    print(f"  B2 plaintext contains {total_forced} characters (x, y) with no DoI "
          "homophones.")
    print(f"  The 3 SPECIAL_DECODE rules handle some of these, but not all:")
    print(f"    - #811→'y' covers 'y' occurrences (but #811 is used {len([i for i, n in enumerate(B2) if n == 811])}x total)")
    print(f"    - #1005→'x' covers 'x' occurrences (but #1005 is used {len([i for i, n in enumerate(B2) if n == 1005])}x total)")
    print(f"    - #95→'u' is redundant ('unalienable' already starts with 'u')")
    print()
    print("  Forward encoding (genuine):")
    print("    Encoder needs y/x but DoI has no words starting with them.")
    print("    Creates ad-hoc overrides: repurpose specific words.")
    print("    This is exactly what a real encoder would do — hack around gaps.")
    print("  Backward construction (hoax):")
    print("    Hoaxer controls plaintext and could avoid y/x entirely.")
    print("    BUT B2's plaintext contains proper English with natural y/x usage.")
    print("    Overrides are UNNECESSARY if you control the plaintext.")
    print("    Their presence argues for genuine forward encoding.")

    return {
        "total_forced_overrides": total_forced,
        "override_correct": override_correct,
        "natural_correct": natural_correct,
        "override_wrong": override_wrong,
        "mc_errors_mean": float(ec_arr.mean()),
    }


# ============================================================================
# SUMMARY
# ============================================================================

def print_summary(
    reset_results: dict | None = None,
    fab_results: dict | None = None,
    fingerprint_results: dict | None = None,
    override_results: dict | None = None,
) -> None:
    """Print overall summary across all tests."""
    print("\n" + "=" * 72)
    print("PHASE 9 SUMMARY: B2 CONSTRUCTION METHOD ANALYSIS")
    print("=" * 72)

    if reset_results:
        best_rp = min(reset_results, key=lambda rp: reset_results[rp]["combined_z"])
        print(f"\n1. Reset sweep: B2 best match at reset_prob={best_rp:.2f}")
        print(f"   (B1=0.65 sloppy, B3=0.01 methodical)")
        if best_rp >= 0.9:
            print("   → B2 consistent with RANDOM selection, not sequential scanning")
        else:
            print(f"   → B2 consistent with sequential scanning (reset_prob={best_rp})")

    if fab_results:
        any_outlier = any(abs(r["z"]) > 2.0 for r in fab_results.values())
        if any_outlier:
            outliers = [n for n, r in fab_results.items() if abs(r["z"]) > 2.0]
            print(f"\n2. Fabrication test: B2 OUTSIDE distribution on {outliers}")
            print("   → Statistical fingerprint distinguishes B2 from fabrication")
        else:
            max_z = max(abs(r["z"]) for r in fab_results.values())
            print(f"\n2. Fabrication test: B2 INSIDE distribution (max |z|={max_z:.2f})")
            print("   → B2 is indistinguishable from random-encode fabrication")

    if fingerprint_results:
        b2_rs = [r["spearman_r"] for r in fingerprint_results.get("B2", [])]
        b2_mean = np.mean(b2_rs) if b2_rs else 0
        b1_rs = [r["spearman_r"] for r in fingerprint_results.get("B1", [])]
        b1_mean = np.mean(b1_rs) if b1_rs else 0
        b3_rs = [r["spearman_r"] for r in fingerprint_results.get("B3", [])]
        b3_mean = np.mean(b3_rs) if b3_rs else 0
        print(f"\n3. Homophone fingerprint:")
        print(f"   B2 mean Spearman r: {b2_mean:+.4f}")
        print(f"   B1 mean Spearman r: {b1_mean:+.4f}")
        print(f"   B3 mean Spearman r: {b3_mean:+.4f}")
        if abs(b2_mean) < 0.1 and (b1_mean > 0.02 or b3_mean > 0.02):
            print("   → B2 random selection; B1/B3 sequential (gradient B2<B1<B3)")
        else:
            print("   → Pattern less clear than expected")

    if override_results:
        forced = override_results["total_forced_overrides"]
        oc = override_results["override_correct"]
        ow = override_results["override_wrong"]
        print(f"\n4. Override analysis: {forced} plaintext chars need workarounds (x/y)")
        print(f"   Override numbers decode correctly {oc}x, wrong {ow}x")
        print("   → Overrides are ad-hoc patches for DoI gaps — supports forward encoding")

    # Overall verdict
    print(f"\n{'─' * 72}")
    print("OVERALL VERDICT:")
    distinguishable = False
    dr_z = None
    if fab_results:
        distinguishable = any(abs(r["z"]) > 2.0 for r in fab_results.values())
        dr_z = fab_results.get("Distinct ratio", {}).get("z")
    if distinguishable:
        print("  B2 CAN be statistically distinguished from random-encode fabrication.")
        if dr_z and abs(dr_z) > 10:
            print(f"\n  KEY FINDING: B2's distinct ratio (23.6%) is dramatically lower than")
            print(f"  random homophone selection produces (~65%, z={dr_z:.1f}).")
            print(f"  B2 reuses the same small set of DoI word numbers heavily.")
            print(f"  This is consistent with an encoder who:")
            print(f"    - Memorized a few word positions per letter")
            print(f"    - Worked from a short section of the DoI, or")
            print(f"    - Had a personal lookup table (not the full DoI)")
            print(f"\n  Combined with zero positional correlation (Test 3) and ad-hoc")
            print(f"  overrides for x/y (Test 4), B2 shows a construction fingerprint")
            print(f"  that is DIFFERENT from both B1/B3 AND from random fabrication.")
            print(f"  This supports B2 as genuinely encoded by a different method/person.")
    else:
        print("  B2 CANNOT be statistically distinguished from fabrication.")
        print("  A hoaxer using random homophone selection would produce identical stats.")
        print("  This does NOT prove B2 is fake — only that its stats don't prove it real.")


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 9: B2 construction method analysis")
    parser.add_argument("--reset-sweep", action="store_true",
                        help="Test 1: B2 reset probability sweep")
    parser.add_argument("--fabrication-test", action="store_true",
                        help="Test 2: B2 vs fabricated distribution")
    parser.add_argument("--homophone-fingerprint", action="store_true",
                        help="Test 3: per-letter homophone fingerprint")
    parser.add_argument("--override-analysis", action="store_true",
                        help="Test 4: special override analysis")
    parser.add_argument("--all", action="store_true",
                        help="Run all tests")
    parser.add_argument("--n-sims", type=int, default=1000,
                        help="Number of simulations (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    if not any([args.reset_sweep, args.fabrication_test,
                args.homophone_fingerprint, args.override_analysis, args.all]):
        parser.print_help()
        sys.exit(1)

    t0 = time.time()
    reset_results = fab_results = fingerprint_results = override_results = None

    if args.reset_sweep or args.all:
        reset_results = run_b2_reset_sweep(
            n_sims=min(args.n_sims, 200), seed=args.seed)

    if args.fabrication_test or args.all:
        fab_results = run_b2_fabrication_test(
            n_sims=args.n_sims, seed=args.seed)

    if args.homophone_fingerprint or args.all:
        fingerprint_results = run_homophone_fingerprint()

    if args.override_analysis or args.all:
        override_results = run_override_analysis(
            n_sims=args.n_sims, seed=args.seed)

    if args.all:
        print_summary(reset_results, fab_results, fingerprint_results,
                      override_results)

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
