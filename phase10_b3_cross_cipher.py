"""
---
version: 0.1.0
created: 2026-02-25
updated: 2026-02-25
---

phase10_b3_cross_cipher.py — B3 length feasibility & cross-cipher session analysis.

Two independent analyses:

  1. B3 LENGTH: Can 618 characters encode 30 names + addresses + next-of-kin?
     Spoiler: no. Monte Carlo with period-appropriate names shows minimum ~1200.

  2. CROSS-CIPHER: Statistical dependencies between B1 and B3 that would indicate
     single-session construction by the same person.

Usage:
    python3 phase10_b3_cross_cipher.py --b3-length
    python3 phase10_b3_cross_cipher.py --cross-cipher
    python3 phase10_b3_cross_cipher.py --all --n-sims 1000
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
    B1, B2, B3, BEALE_DOI, B2_PLAINTEXT,
    build_letter_index, distinct_ratio, first_letter,
)
from phase8_hoax_construction import serial_correlation


# ============================================================================
# TEST 1: B3 LENGTH FEASIBILITY
# ============================================================================

# Period-appropriate names (1810s-1820s Virginia/frontier).
# Top male first names from US census data 1800-1830.
FIRST_NAMES_1820 = [
    "john", "william", "james", "thomas", "robert", "charles", "george",
    "joseph", "samuel", "benjamin", "daniel", "david", "henry", "andrew",
    "edward", "richard", "abraham", "isaac", "jacob", "peter", "stephen",
    "jonathan", "nathaniel", "jeremiah", "bartholomew", "alexander",
    "frederick", "patrick", "michael", "matthew", "timothy", "zachariah",
    "elijah", "josiah", "caleb", "silas", "ezekiel", "cornelius", "reuben",
    "solomon",
]

# Common surnames of the era.
LAST_NAMES_1820 = [
    "smith", "jones", "johnson", "williams", "brown", "davis", "miller",
    "wilson", "moore", "taylor", "anderson", "thomas", "jackson", "white",
    "harris", "martin", "thompson", "robinson", "clark", "lewis", "walker",
    "hall", "allen", "young", "king", "wright", "hill", "scott", "green",
    "adams", "baker", "nelson", "carter", "mitchell", "campbell", "roberts",
    "turner", "phillips", "parker", "evans", "edwards", "collins", "stewart",
    "morris", "murphy", "cook", "rogers", "morgan", "peterson", "cooper",
]

# States/territories relevant to 1820s Virginia frontier party.
STATES_1820 = [
    "virginia", "maryland", "pennsylvania", "kentucky", "tennessee",
    "northcarolina", "southcarolina", "georgia", "ohio", "missouri",
]

# County/town examples (period-appropriate).
TOWNS_1820 = [
    "lynchburg", "richmond", "bedford", "lexington", "charlottesville",
    "staunton", "fincastle", "liberty", "abingdon", "danville",
    "fredericksburg", "norfolk", "winchester", "petersburg", "roanoke",
    "buford", "newlondon", "baltimore", "philadelphia", "louisville",
]


def generate_person_record(rng: np.random.Generator) -> str:
    """Generate a realistic 1820s person record as it would appear in a
    no-spaces-no-punctuation book cipher decode.

    Format variations (all lowercase, concatenated):
      - Minimal: firstnamelastnametownstate
      - With kin: firstnamelastnametownstatekinfirstkinlast
    """
    first = rng.choice(FIRST_NAMES_1820)
    last = rng.choice(LAST_NAMES_1820)
    town = rng.choice(TOWNS_1820)
    state = rng.choice(STATES_1820)

    # Next of kin (as B2 describes: "paper number three" has names)
    kin_first = rng.choice(FIRST_NAMES_1820)
    kin_last = rng.choice(LAST_NAMES_1820)

    # Various plausible encodings
    formats = [
        f"{first}{last}{town}{state}",
        f"{first}{last}{town}{state}{kin_first}{kin_last}",
        f"{first}{last}of{town}{state}",
        f"{first}{last}{town}{state}nextofkin{kin_first}{kin_last}",
    ]
    return rng.choice(formats)


def run_b3_length_test(n_sims: int = 10000, seed: int = 42) -> dict:
    """Test whether 618 characters can encode 30 person records."""
    rng = np.random.default_rng(seed)

    print("=" * 72)
    print("TEST 1: B3 LENGTH FEASIBILITY")
    print("=" * 72)

    b3_len = len(B3)
    n_people = 30
    print(f"\nB3 has {b3_len} cipher numbers → {b3_len} plaintext characters")
    print(f"B2 states paper #3 contains names of {n_people} partners")
    print(f"Available characters per person: {b3_len / n_people:.1f}")

    # Analytical bounds
    print(f"\n--- ANALYTICAL BOUNDS ---")

    # Minimal: just first + last name
    min_name = min(len(f) + len(l) for f in FIRST_NAMES_1820 for l in LAST_NAMES_1820)
    avg_name = np.mean([len(f) + len(l) for f in FIRST_NAMES_1820
                        for l in LAST_NAMES_1820])
    print(f"Name only (first+last): min={min_name}, avg={avg_name:.0f}")
    print(f"  30 names (min): {min_name * 30} chars")
    print(f"  30 names (avg): {avg_name * 30:.0f} chars")

    # With location
    min_loc = min(len(t) + len(s) for t in TOWNS_1820 for s in STATES_1820)
    avg_loc = np.mean([len(t) + len(s) for t in TOWNS_1820 for s in STATES_1820])
    print(f"Location (town+state): min={min_loc}, avg={avg_loc:.0f}")
    print(f"  30 × (name+loc) min: {(min_name + min_loc) * 30} chars")
    print(f"  30 × (name+loc) avg: {(avg_name + avg_loc) * 30:.0f} chars")

    # With next of kin
    print(f"  30 × (name+loc+kin) avg: {(avg_name * 2 + avg_loc) * 30:.0f} chars")

    # Monte Carlo
    print(f"\n--- MONTE CARLO ({n_sims} simulations) ---")
    lengths = np.zeros(n_sims)
    for i in range(n_sims):
        records = [generate_person_record(rng) for _ in range(n_people)]
        total = sum(len(r) for r in records)
        lengths[i] = total

    print(f"Total characters for 30 person records:")
    print(f"  Mean:   {lengths.mean():.0f}")
    print(f"  Median: {np.median(lengths):.0f}")
    print(f"  Min:    {lengths.min():.0f}")
    print(f"  Max:    {lengths.max():.0f}")
    print(f"  Std:    {lengths.std():.0f}")
    print(f"  5th %%:  {np.percentile(lengths, 5):.0f}")

    fits = np.sum(lengths <= b3_len)
    print(f"\n  Simulations that fit in {b3_len} chars: {fits}/{n_sims} "
          f"({fits/n_sims*100:.2f}%)")

    # What B2 actually says about B3's content
    print(f"\n--- B2's DESCRIPTION OF B3 ---")
    print("  B2 says: 'paper number three the names of the parties'")
    print("  B2 also says: relatives/contacts for distributing shares")
    print(f"  With 30 people × (name + address + next-of-kin):")
    print(f"    Minimum realistic: ~{int((avg_name + avg_loc) * 30)} chars")
    print(f"    With kin names:    ~{int((avg_name * 2 + avg_loc) * 30)} chars")
    print(f"    Available:          {b3_len} chars")

    ratio = b3_len / lengths.mean()
    print(f"\n  B3 is {ratio:.0%} of the average required length")

    # Even without addresses — just names
    name_only_lengths = np.zeros(n_sims)
    for i in range(n_sims):
        total = sum(
            len(rng.choice(FIRST_NAMES_1820)) + len(rng.choice(LAST_NAMES_1820))
            for _ in range(n_people)
        )
        name_only_lengths[i] = total

    name_fits = np.sum(name_only_lengths <= b3_len)
    print(f"\n  Even names-only (no addresses, no kin):")
    print(f"    Mean: {name_only_lengths.mean():.0f}, "
          f"fits in {b3_len}: {name_fits}/{n_sims} ({name_fits/n_sims*100:.1f}%)")

    # Verdict
    print(f"\nVERDICT:")
    if fits == 0:
        print(f"  IMPOSSIBLE: {b3_len} chars cannot encode 30 person records")
        print(f"  with names + locations + next-of-kin ({lengths.mean():.0f} needed).")
        if name_fits / n_sims > 0.9:
            print(f"  Names-only fits easily ({name_only_lengths.mean():.0f} avg chars)")
            print(f"  but B2 explicitly describes addresses and relatives.")
        elif name_fits > 0:
            print(f"  Names-only barely fits ({name_fits/n_sims*100:.1f}% of trials)")
            print(f"  but B2 explicitly describes addresses and relatives.")
        else:
            print(f"  Even names-only doesn't fit ({name_only_lengths.mean():.0f} needed).")
        print(f"  B3 is structurally impossible as a genuine name list.")
    else:
        print(f"  {fits/n_sims*100:.1f}% of simulations fit — marginal feasibility.")

    return {
        "b3_len": b3_len,
        "mean_required": float(lengths.mean()),
        "min_required": float(lengths.min()),
        "pct_fit": float(fits / n_sims * 100),
        "names_only_mean": float(name_only_lengths.mean()),
        "names_only_pct_fit": float(name_fits / n_sims * 100),
    }


# ============================================================================
# TEST 2: CROSS-CIPHER SESSION ANALYSIS
# ============================================================================

def run_cross_cipher_analysis(n_sims: int = 1000, seed: int = 42) -> dict:
    """Analyze statistical dependencies between B1 and B3."""
    rng = np.random.default_rng(seed)

    print("\n" + "=" * 72)
    print("TEST 2: CROSS-CIPHER SESSION ANALYSIS")
    print("=" * 72)

    results = {}

    # --- 2a. Number overlap ---
    print("\n--- 2a. NUMBER OVERLAP ---")
    b1_set = set(B1)
    b3_set = set(B3)
    b2_set = set(B2)
    overlap_13 = b1_set & b3_set
    overlap_12 = b1_set & b2_set
    overlap_23 = b2_set & b3_set

    print(f"  B1 distinct values: {len(b1_set)}")
    print(f"  B3 distinct values: {len(b3_set)}")
    print(f"  B1∩B3 overlap: {len(overlap_13)} ({len(overlap_13)/len(b1_set|b3_set)*100:.1f}% of union)")
    print(f"  B1∩B2 overlap: {len(overlap_12)} ({len(overlap_12)/len(b1_set|b2_set)*100:.1f}% of union)")
    print(f"  B2∩B3 overlap: {len(overlap_23)} ({len(overlap_23)/len(b2_set|b3_set)*100:.1f}% of union)")

    # Expected overlap under independence (random draws from same DoI)
    # B3 range is 1-975, B1 range is 1-2906 (but 98% within 1-1311)
    b1_in_range = [n for n in B1 if n <= 975]
    print(f"\n  B1 values ≤975: {len(b1_in_range)}/{len(B1)} ({len(b1_in_range)/len(B1)*100:.0f}%)")
    print(f"  B3 values ≤975: {len(B3)}/{len(B3)} (100%)")

    results["overlap_13"] = len(overlap_13)
    results["overlap_12"] = len(overlap_12)
    results["overlap_23"] = len(overlap_23)

    # --- 2b. Cursor carryover test ---
    print(f"\n--- 2b. CURSOR CARRYOVER ---")
    print("  If B3 was encoded immediately after B1, the DoI cursor position")
    print("  from B1's end might carry into B3's start.")

    # B1's last number and B3's first number
    b1_last = B1[-1]
    b3_first = B3[0]
    print(f"\n  B1 last number: {b1_last}")
    print(f"  B3 first number: {b3_first}")
    print(f"  Gap: {b3_first - b1_last} positions")

    # Is B3's first number near B1's last? Compare to random baseline.
    # Under null: B3's first is drawn uniformly from 1-975
    # Test: is |B3[0] - B1[-1]| unusually small?
    b1_last_5 = list(B1[-5:])
    b3_first_5 = list(B3[:5])
    print(f"  B1 last 5: {b1_last_5}")
    print(f"  B3 first 5: {b3_first_5}")

    # Monte Carlo: random gap distribution
    # If the hoaxer was scanning forward, the next number should be > cursor
    forward_continuation = b3_first > b1_last
    print(f"  Forward continuation (B3[0] > B1[-1]): {forward_continuation}")

    # But B1 ends at 760, and B3 starts at 317 — that's a backward jump.
    # Check if B3[0] is consistent with a page flip (B1 ends on page 3,
    # B3 starts at top of page 1)
    words_per_page = 325
    b1_last_page = (b1_last - 1) // words_per_page + 1
    b3_first_page = (b3_first - 1) // words_per_page + 1
    print(f"  B1 ends on page: {b1_last_page} (word {b1_last})")
    print(f"  B3 starts on page: {b3_first_page} (word {b3_first})")
    if b3_first_page == 1 and b1_last_page > 1:
        print("  → B3 starts at page 1 after B1 ended later — consistent with")
        print("    flipping back to start for a new cipher")

    results["b1_last"] = b1_last
    results["b3_first"] = b3_first
    results["forward_continuation"] = forward_continuation

    # --- 2c. Fatigue continuity ---
    print(f"\n--- 2c. FATIGUE CONTINUITY ---")
    print("  Phase 8f found Q1→Q4 fatigue gradient in both ciphers.")
    print("  If B3 was written after B1, B3's Q1 might start at B1's Q4 level.")

    # Compute quarter-by-quarter SC for both ciphers
    for label, cipher in [("B1", B1), ("B3", B3)]:
        n = len(cipher)
        q_size = n // 4
        quarters = []
        for q in range(4):
            start = q * q_size
            end = start + q_size if q < 3 else n
            sc = serial_correlation(cipher[start:end])
            quarters.append(sc)
        print(f"  {label} SC by quarter: " +
              "  ".join(f"Q{i+1}={sc:.3f}" for i, sc in enumerate(quarters)))

    b1_q4_sc = serial_correlation(B1[3 * (len(B1) // 4):])
    b3_q1_sc = serial_correlation(B3[:len(B3) // 4])
    print(f"\n  B1 Q4 SC: {b1_q4_sc:.3f}")
    print(f"  B3 Q1 SC: {b3_q1_sc:.3f}")

    if b3_q1_sc < b1_q4_sc:
        print("  → B3 Q1 < B1 Q4: fatigue RESETS between ciphers")
        print("    Consistent with a break between B1 and B3 (or B3 first)")
    else:
        print("  → B3 Q1 ≥ B1 Q4: fatigue CONTINUES")
        print("    Consistent with B3 written immediately after B1")

    results["b1_q4_sc"] = b1_q4_sc
    results["b3_q1_sc"] = b3_q1_sc

    # --- 2d. Concatenated sequence analysis ---
    print(f"\n--- 2d. CONCATENATED SEQUENCE ---")
    b1b3 = list(B1) + list(B3)
    b3b1 = list(B3) + list(B1)

    sc_b1 = serial_correlation(B1)
    sc_b3 = serial_correlation(B3)
    sc_b1b3 = serial_correlation(b1b3)
    sc_b3b1 = serial_correlation(b3b1)

    print(f"  SC(B1):     {sc_b1:.4f}")
    print(f"  SC(B3):     {sc_b3:.4f}")
    print(f"  SC(B1+B3):  {sc_b1b3:.4f}")
    print(f"  SC(B3+B1):  {sc_b3b1:.4f}")

    # If independent, SC(B1+B3) ≈ weighted average of SC(B1) and SC(B3)
    # with a junction effect
    n1, n3 = len(B1), len(B3)
    weighted_avg = (n1 * sc_b1 + n3 * sc_b3) / (n1 + n3)
    print(f"  Weighted avg: {weighted_avg:.4f}")
    print(f"  Junction effect B1→B3: {sc_b1b3 - weighted_avg:+.4f}")
    print(f"  Junction effect B3→B1: {sc_b3b1 - weighted_avg:+.4f}")

    # Monte Carlo: what junction effect do independent ciphers produce?
    junction_effects = np.zeros(n_sims)
    letter_index = build_letter_index(BEALE_DOI)
    available = [c for c in string.ascii_lowercase if letter_index.get(c)]

    for i in range(n_sims):
        # Generate two independent ciphers with similar properties
        from beale import encode_sequential_book_cipher
        text1 = "".join(rng.choice(available, size=n1))
        text3 = "".join(rng.choice(available, size=n3))
        c1 = encode_sequential_book_cipher(text1, BEALE_DOI, reset_prob=0.65, rng=rng)
        c3 = encode_sequential_book_cipher(text3, BEALE_DOI, reset_prob=0.01, rng=rng)
        sc_combined = serial_correlation(c1 + c3)
        sc_weighted = (n1 * serial_correlation(c1) + n3 * serial_correlation(c3)) / (n1 + n3)
        junction_effects[i] = sc_combined - sc_weighted

    actual_je = sc_b1b3 - weighted_avg
    je_z = (actual_je - junction_effects.mean()) / junction_effects.std()
    je_pct = float(np.sum(junction_effects <= actual_je) / n_sims * 100)

    print(f"\n  MC junction effect distribution: mean={junction_effects.mean():.4f}, "
          f"std={junction_effects.std():.4f}")
    print(f"  Actual junction effect: {actual_je:.4f}")
    print(f"  z-score: {je_z:.2f}, percentile: {je_pct:.1f}%")

    if abs(je_z) < 2:
        print("  → Junction effect is NORMAL — no evidence of cursor continuity")
    else:
        print("  → Junction effect is UNUSUAL — possible cursor link between ciphers")

    results["junction_effect"] = actual_je
    results["junction_z"] = je_z

    # --- 2e. Number range and page usage ---
    print(f"\n--- 2e. PAGE USAGE PATTERNS ---")
    for label, cipher in [("B1", B1), ("B3", B3), ("B2", B2)]:
        pages = Counter()
        for n in cipher:
            page = min((n - 1) // words_per_page + 1, 5)  # 5 = overflow
            pages[page] += 1
        total = len(cipher)
        print(f"  {label} page distribution:")
        for p in sorted(pages.keys()):
            pct = pages[p] / total * 100
            bar = "█" * int(pct / 2)
            if p == 5:
                print(f"    >p4: {pages[p]:>4d} ({pct:5.1f}%) {bar}")
            else:
                print(f"    p{p}:  {pages[p]:>4d} ({pct:5.1f}%) {bar}")

    # B3 uses ONLY pages 1-3, B1 uses all 4 (phase 8 finding)
    b3_max = max(B3)
    b1_max_in_range = max(n for n in B1 if n <= 1311)
    print(f"\n  B3 max value: {b3_max} (page {(b3_max-1)//325+1})")
    print(f"  B1 max in-range: {b1_max_in_range} (page {(b1_max_in_range-1)//325+1})")

    # --- 2f. Shared homophone preferences ---
    print(f"\n--- 2f. SHARED HOMOPHONE PREFERENCES ---")
    print("  Do B1 and B3 favor the same specific numbers for common letters?")

    # For each letter decoded from both ciphers, compare preferred homophones
    from beale import decode_book_cipher
    b1_decoded = decode_book_cipher(B1, BEALE_DOI)
    b3_decoded = decode_book_cipher(B3, BEALE_DOI)

    shared_prefs = []
    print(f"\n  {'letter':>6s}  {'B1_top':>8s}  {'B3_top':>8s}  {'shared':>6s}  "
          f"{'jaccard':>8s}")
    print(f"  {'-'*42}")

    for letter in sorted(set(b1_decoded) & set(b3_decoded)):
        if not letter.isalpha():
            continue
        b1_nums = [B1[i] for i, c in enumerate(b1_decoded) if c == letter]
        b3_nums = [B3[i] for i, c in enumerate(b3_decoded) if c == letter]
        if len(b1_nums) < 5 or len(b3_nums) < 5:
            continue

        # Top 3 most-used numbers for this letter in each cipher
        b1_top = set(n for n, _ in Counter(b1_nums).most_common(3))
        b3_top = set(n for n, _ in Counter(b3_nums).most_common(3))
        shared = b1_top & b3_top
        jaccard = len(shared) / len(b1_top | b3_top) if b1_top | b3_top else 0

        shared_prefs.append(jaccard)
        b1_top_str = ",".join(str(n) for n in sorted(b1_top))
        b3_top_str = ",".join(str(n) for n in sorted(b3_top))
        print(f"  {letter:>6s}  {b1_top_str:>8s}  {b3_top_str:>8s}  "
              f"{len(shared):>6d}  {jaccard:>8.2f}")

    mean_jaccard = np.mean(shared_prefs) if shared_prefs else 0
    print(f"\n  Mean Jaccard similarity of top-3 homophones: {mean_jaccard:.3f}")

    # MC baseline: what Jaccard do independent encoders produce?
    mc_jaccards = []
    for _ in range(n_sims):
        text1 = "".join(rng.choice(available, size=n1))
        text3 = "".join(rng.choice(available, size=n3))
        from beale import encode_book_cipher
        c1 = encode_sequential_book_cipher(text1, BEALE_DOI, reset_prob=0.65, rng=rng)
        c3 = encode_sequential_book_cipher(text3, BEALE_DOI, reset_prob=0.01, rng=rng)
        d1 = decode_book_cipher(c1, BEALE_DOI)
        d3 = decode_book_cipher(c3, BEALE_DOI)

        jacs = []
        for letter in set(d1) & set(d3):
            if not letter.isalpha():
                continue
            nums1 = [c1[i] for i, c in enumerate(d1) if c == letter]
            nums3 = [c3[i] for i, c in enumerate(d3) if c == letter]
            if len(nums1) < 5 or len(nums3) < 5:
                continue
            top1 = set(n for n, _ in Counter(nums1).most_common(3))
            top3 = set(n for n, _ in Counter(nums3).most_common(3))
            j = len(top1 & top3) / len(top1 | top3) if top1 | top3 else 0
            jacs.append(j)
        mc_jaccards.append(np.mean(jacs) if jacs else 0)

    mc_arr = np.array(mc_jaccards)
    j_z = (mean_jaccard - mc_arr.mean()) / mc_arr.std() if mc_arr.std() > 0 else 0
    print(f"  MC baseline mean Jaccard: {mc_arr.mean():.3f} ± {mc_arr.std():.3f}")
    print(f"  Actual: {mean_jaccard:.3f}, z-score: {j_z:.2f}")

    if mean_jaccard > mc_arr.mean() and j_z > 2:
        print("  → B1 and B3 share more homophone preferences than expected")
        print("    Suggests same person's number-to-letter mental lookup table")
    elif j_z < -2:
        print("  → B1 and B3 share FEWER preferences than expected")
        print("    Suggests deliberate avoidance or different encoders")
    else:
        print("  → Homophone preferences are consistent with independent encoding")

    results["mean_jaccard"] = mean_jaccard
    results["jaccard_z"] = j_z

    return results


# ============================================================================
# SUMMARY
# ============================================================================

def print_summary(
    b3_results: dict | None = None,
    cross_results: dict | None = None,
) -> None:
    """Print overall summary."""
    print("\n" + "=" * 72)
    print("PHASE 10 SUMMARY")
    print("=" * 72)

    if b3_results:
        print(f"\n1. B3 LENGTH FEASIBILITY:")
        print(f"   B3 has {b3_results['b3_len']} chars for 30 people")
        print(f"   Mean required: {b3_results['mean_required']:.0f} chars")
        print(f"   Fits: {b3_results['pct_fit']:.1f}% of simulations")
        if b3_results["pct_fit"] == 0:
            print(f"   → STRUCTURALLY IMPOSSIBLE as described by B2")
        print(f"   Names-only fits: {b3_results['names_only_pct_fit']:.1f}%")

    if cross_results:
        print(f"\n2. CROSS-CIPHER SESSION ANALYSIS:")
        print(f"   a. Number overlap: {cross_results['overlap_13']} shared values")
        je_z = cross_results.get("junction_z", 0)
        print(f"   b. Cursor carryover: {'no' if not cross_results['forward_continuation'] else 'yes'} "
              f"forward continuation")
        print(f"   c. Fatigue: B1-Q4 SC={cross_results['b1_q4_sc']:.3f}, "
              f"B3-Q1 SC={cross_results['b3_q1_sc']:.3f} "
              f"({'reset' if cross_results['b3_q1_sc'] < cross_results['b1_q4_sc'] else 'continues'})")
        print(f"   d. Junction effect z={je_z:.2f} "
              f"({'normal' if abs(je_z) < 2 else 'unusual'})")
        j_z = cross_results.get("jaccard_z", 0)
        print(f"   e. Homophone preference Jaccard z={j_z:.2f}")


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 10: B3 length feasibility & cross-cipher analysis")
    parser.add_argument("--b3-length", action="store_true",
                        help="Test 1: B3 length feasibility")
    parser.add_argument("--cross-cipher", action="store_true",
                        help="Test 2: Cross-cipher session analysis")
    parser.add_argument("--all", action="store_true",
                        help="Run all tests")
    parser.add_argument("--n-sims", type=int, default=1000,
                        help="Number of simulations (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    if not any([args.b3_length, args.cross_cipher, args.all]):
        parser.print_help()
        sys.exit(1)

    t0 = time.time()
    b3_results = cross_results = None

    if args.b3_length or args.all:
        b3_results = run_b3_length_test(n_sims=args.n_sims, seed=args.seed)

    if args.cross_cipher or args.all:
        cross_results = run_cross_cipher_analysis(
            n_sims=args.n_sims, seed=args.seed)

    if args.all:
        print_summary(b3_results, cross_results)

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
