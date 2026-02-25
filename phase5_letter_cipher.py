"""
---
version: 0.1.0
created: 2026-02-25
updated: 2026-02-25
---

phase5_letter_cipher.py — Letter-index cipher hypothesis for B1/B3.

Tests the hypothesis that Beale cipher numbers index individual characters
(not words) in the key text. The DoI has 6,480 alpha chars — B1 max=2906,
B3 max=975, both fit with zero out-of-range.

Sections:
  1. Validation — round-trip encode/decode, word vs letter comparison
  2. DoI test — letter-decode B1/B3 with DoI, score bigrams
  3. Homophone analysis — letter-level vs word-level homophone counts
  4. Statistical signatures — synthetic letter-index ciphers vs B1/B3
  5. Corpus sweep — all cached Gutenberg texts as letter-index keys

Usage:
    python3 phase5_letter_cipher.py --doi-only     # sections 1-3
    python3 phase5_letter_cipher.py --stats         # section 4 (synthetic comparison)
    python3 phase5_letter_cipher.py --sweep         # section 5 (corpus sweep)
    python3 phase5_letter_cipher.py --results       # show sweep results
"""

from __future__ import annotations

import argparse
import json
import string
import time
from collections import Counter
from pathlib import Path

import numpy as np

from beale import (
    B1, B2, B3, BEALE_DOI, SPECIAL_DECODE, BIGRAM_FLOOR,
    decode_book_cipher, decode_letter_cipher, encode_letter_cipher,
    text_to_alpha, score_letter_cipher, bigram_score, index_of_coincidence,
    build_letter_index, load_gutenberg_text, text_to_words,
    benford_test, last_digit_test, distinct_ratio,
    gillogly_quality,
)


# ============================================================================
# 1. VALIDATION
# ============================================================================

def run_validation() -> None:
    """Round-trip encode/decode and word vs letter comparison."""
    print("=" * 70)
    print("SECTION 1: VALIDATION")
    print("=" * 70)

    doi_text = " ".join(BEALE_DOI)
    alpha = text_to_alpha(doi_text)
    print(f"\nDoI alpha char count: {len(alpha)}")
    print(f"DoI word count: {len(BEALE_DOI)}")
    print(f"B1 max number: {max(B1)}, B3 max number: {max(B3)}")
    print(f"B1 in-range (letter): {max(B1) <= len(alpha)}")
    print(f"B3 in-range (letter): {max(B3) <= len(alpha)}")

    # Round-trip test
    rng = np.random.default_rng(42)
    plaintext = "thequickbrownfoxjumpsoverthelazydog"
    encoded = encode_letter_cipher(plaintext, doi_text, rng)
    decoded = decode_letter_cipher(encoded, doi_text)
    assert decoded == plaintext, f"Round-trip FAILED: '{decoded}' != '{plaintext}'"
    print(f"\nRound-trip test: PASS")
    print(f"  plaintext:  {plaintext}")
    print(f"  encoded:    {encoded[:10]}... ({len(encoded)} numbers)")
    print(f"  decoded:    {decoded}")

    # Confirm word-decode != letter-decode
    word_dec = decode_book_cipher(B1[:20], BEALE_DOI)
    letter_dec = decode_letter_cipher(B1[:20], doi_text)
    print(f"\nWord decode B1[:20]:   {word_dec}")
    print(f"Letter decode B1[:20]: {letter_dec}")
    assert word_dec != letter_dec, "Word and letter decodes should differ!"
    print("Word != Letter: PASS")


# ============================================================================
# 2. DOI LETTER-DECODE TEST
# ============================================================================

def run_doi_test() -> None:
    """Score B1 and B3 letter-decoded with the DoI."""
    print("\n" + "=" * 70)
    print("SECTION 2: DOI LETTER-DECODE TEST")
    print("=" * 70)

    doi_text = " ".join(BEALE_DOI)

    for name, cipher in [("B1", B1), ("B3", B3)]:
        sc = score_letter_cipher(cipher, doi_text)
        print(f"\n{name} letter-decode with DoI:")
        print(f"  bigram score:  {sc['bigram_score']:.3f}  (English ~-2.3, random ~-4.0)")
        print(f"  IC:            {sc['ic']:.4f}  (English ~0.067, random ~0.038)")
        print(f"  in-range:      {sc['in_range']:.1%}")
        print(f"  preview:       {sc['preview']}")

    # Also show B2 word-decode baseline for comparison
    b2_decoded = decode_book_cipher(B2, BEALE_DOI, SPECIAL_DECODE)
    b2_bg = bigram_score(b2_decoded)
    print(f"\nB2 word-decode baseline: bigram={b2_bg:.3f}")


# ============================================================================
# 3. HOMOPHONE ANALYSIS
# ============================================================================

def run_homophone_analysis() -> None:
    """Compare homophone counts: letter-level vs word-level."""
    print("\n" + "=" * 70)
    print("SECTION 3: HOMOPHONE ANALYSIS")
    print("=" * 70)

    doi_text = " ".join(BEALE_DOI)
    alpha = text_to_alpha(doi_text)

    # Word-level homophones (first-letter index)
    word_index = build_letter_index(BEALE_DOI)
    # Letter-level homophones (every occurrence of each letter)
    letter_counts = Counter(alpha)

    print(f"\n{'Letter':>6} {'Word-level':>12} {'Letter-level':>14} {'Ratio':>8}")
    print("-" * 44)
    for c in string.ascii_lowercase:
        wc = len(word_index.get(c, []))
        lc = letter_counts.get(c, 0)
        ratio = lc / wc if wc > 0 else float("inf")
        flag = " <<<" if c == "e" else ""
        print(f"    {c}  {wc:>10}  {lc:>12}  {ratio:>7.1f}x{flag}")

    print(f"\n  Total: words={sum(len(v) for v in word_index.values())}, "
          f"letters={len(alpha)}")
    print(f"  Letter-level gives ~{letter_counts['e']}x homophones for 'e' "
          f"vs ~{len(word_index.get('e', []))} word-level")


# ============================================================================
# 4. STATISTICAL SIGNATURES
# ============================================================================

def run_stats(n_synthetic: int = 100) -> None:
    """Generate synthetic letter-index ciphers, compare stats to B1/B3."""
    print("\n" + "=" * 70)
    print("SECTION 4: STATISTICAL SIGNATURES")
    print(f"Generating {n_synthetic} synthetic letter-index ciphers from DoI...")
    print("=" * 70)

    doi_text = " ".join(BEALE_DOI)
    alpha = text_to_alpha(doi_text)
    rng = np.random.default_rng(42)

    # Generate synthetic ciphers
    from beale import ENGLISH_FREQ

    synth_b1: list[list[int]] = []
    synth_b3: list[list[int]] = []
    for _ in range(n_synthetic):
        # Random English-frequency plaintext, encode as letter-index
        available = [c for c in ENGLISH_FREQ if Counter(alpha).get(c, 0) > 0]
        probs = np.array([ENGLISH_FREQ[c] for c in available])
        probs /= probs.sum()
        pt1 = "".join(rng.choice(available, size=len(B1), p=probs))
        pt3 = "".join(rng.choice(available, size=len(B3), p=probs))
        synth_b1.append(encode_letter_cipher(pt1, doi_text, rng))
        synth_b3.append(encode_letter_cipher(pt3, doi_text, rng))

    # Compute stats for real and synthetic
    def stats_row(numbers: list[int] | tuple[int, ...]) -> dict:
        bf = benford_test(numbers)
        ld = last_digit_test(numbers, base=10)
        dr = distinct_ratio(numbers)
        return {
            "benford_chi2": bf["chi2"],
            "benford_p": bf["p_value"],
            "last_digit_chi2": ld["chi2"],
            "last_digit_p": ld["p_value"],
            "distinct_ratio": dr["ratio"],
            "max_val": max(numbers),
        }

    b1_stats = stats_row(B1)
    b3_stats = stats_row(B3)

    synth_b1_stats = [stats_row(s) for s in synth_b1]
    synth_b3_stats = [stats_row(s) for s in synth_b3]

    print(f"\n{'Metric':<25} {'B1 actual':>12} {'Synth B1 mean':>14} {'Synth B1 std':>14}")
    print("-" * 67)
    for key in ["benford_chi2", "last_digit_chi2", "distinct_ratio", "max_val"]:
        b1_val = b1_stats[key]
        synth_vals = [s[key] for s in synth_b1_stats]
        print(f"  {key:<23} {b1_val:>12.2f} {np.mean(synth_vals):>14.2f} "
              f"{np.std(synth_vals):>14.2f}")

    # How many synthetic ciphers have B1's stats or more extreme?
    print(f"\n  B1 Benford chi2 percentile among synthetic: "
          f"{100 * np.mean([s['benford_chi2'] >= b1_stats['benford_chi2'] for s in synth_b1_stats]):.0f}%")
    print(f"  B1 distinct ratio percentile: "
          f"{100 * np.mean([s['distinct_ratio'] >= b1_stats['distinct_ratio'] for s in synth_b1_stats]):.0f}%")

    print(f"\n{'Metric':<25} {'B3 actual':>12} {'Synth B3 mean':>14} {'Synth B3 std':>14}")
    print("-" * 67)
    for key in ["benford_chi2", "last_digit_chi2", "distinct_ratio", "max_val"]:
        b3_val = b3_stats[key]
        synth_vals = [s[key] for s in synth_b3_stats]
        print(f"  {key:<23} {b3_val:>12.2f} {np.mean(synth_vals):>14.2f} "
              f"{np.std(synth_vals):>14.2f}")

    print(f"\n  B3 Benford chi2 percentile among synthetic: "
          f"{100 * np.mean([s['benford_chi2'] >= b3_stats['benford_chi2'] for s in synth_b3_stats]):.0f}%")
    print(f"  B3 distinct ratio percentile: "
          f"{100 * np.mean([s['distinct_ratio'] >= b3_stats['distinct_ratio'] for s in synth_b3_stats]):.0f}%")

    # Key question: do B1/B3 look more like letter-index or word-index?
    # Word-index has max ~1311, letter-index has max ~6480
    # B1 max=2906, B3 max=975
    print("\n--- KEY COMPARISON ---")
    print(f"  DoI words: {len(BEALE_DOI)}, DoI alpha chars: {len(alpha)}")
    print(f"  B1 max={max(B1)}: exceeds word count (1311), fits letter count (6480)")
    print(f"  B3 max={max(B3)}: fits both word count (1311) and letter count (6480)")
    print(f"  B1 MUST use either a different text or letter-index encoding")
    print(f"  B3 is ambiguous — could be either encoding with the DoI")


# ============================================================================
# 5. CORPUS SWEEP — Letter-index keys
# ============================================================================

STATE_FILE = Path("letter_search_state.json")
CACHE_DIR = Path(".gutenberg_cache")


def _load_state(state_file: Path) -> dict:
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {"tested": {}, "results": []}


def _save_state(state: dict, state_file: Path) -> None:
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def run_sweep(state_file: Path = STATE_FILE) -> None:
    """Sweep all cached Gutenberg texts as letter-index keys."""
    print("\n" + "=" * 70)
    print("SECTION 5: CORPUS SWEEP — LETTER-INDEX KEYS")
    print("=" * 70)

    if not CACHE_DIR.exists():
        print("No .gutenberg_cache/ directory found. Run phase4_corpus.py --sweep first.")
        return

    cached_files = sorted(CACHE_DIR.glob("pg*.txt"))
    print(f"Cached texts: {len(cached_files)}")

    state = _load_state(state_file)
    already = len(state["tested"])
    print(f"Already tested: {already}")

    # Baseline
    doi_text = " ".join(BEALE_DOI)
    b2_decoded = decode_book_cipher(B2, BEALE_DOI, SPECIAL_DECODE)
    b2_bg = bigram_score(b2_decoded)
    print(f"B2 word-decode baseline: bigram={b2_bg:.3f}\n")

    tested = 0
    scored = 0
    t0 = time.time()

    for filepath in cached_files:
        fname = filepath.name
        if fname in state["tested"]:
            continue

        try:
            raw_text = load_gutenberg_text(filepath)
        except Exception:
            state["tested"][fname] = {"status": "parse_error"}
            tested += 1
            continue

        alpha = text_to_alpha(raw_text)
        n_alpha = len(alpha)

        # B3 needs 975 chars, B1 needs 2906
        if n_alpha < max(B3):
            state["tested"][fname] = {"status": "too_short", "alpha_len": n_alpha}
            tested += 1
            continue

        # Score B3 (always fits)
        b3_sc = score_letter_cipher(B3, raw_text)

        # Score B1 if text long enough
        if n_alpha >= max(B1):
            b1_sc = score_letter_cipher(B1, raw_text)
        else:
            b1_sc = {"bigram_score": None, "ic": None, "in_range": 0.0, "preview": ""}

        scored += 1
        tested += 1

        # Extract title from filename
        title = fname.replace(".txt", "")

        flag = ""
        if b3_sc["bigram_score"] > -3.1 or (b1_sc["bigram_score"] or -4) > -3.1:
            flag = " ***"

        b1_bg_str = f"{b1_sc['bigram_score']:>7.3f}" if b1_sc["bigram_score"] is not None else "    N/A"

        if scored % 100 == 0 or flag:
            elapsed = time.time() - t0
            rate = tested / elapsed if elapsed > 0 else 0
            print(f"  [{title:>12}] alpha={n_alpha:>6}  "
                  f"B1={b1_bg_str}  B3={b3_sc['bigram_score']:>7.3f}"
                  f"{flag}  ({rate:.0f}/s)")

        result = {
            "file": fname,
            "alpha_len": n_alpha,
            "b1_bigram": b1_sc["bigram_score"],
            "b1_ic": b1_sc["ic"],
            "b1_in_range": b1_sc["in_range"],
            "b3_bigram": b3_sc["bigram_score"],
            "b3_ic": b3_sc["ic"],
            "b3_in_range": b3_sc["in_range"],
            "b3_preview": b3_sc["preview"][:60],
        }
        state["tested"][fname] = {"status": "ok"}
        state["results"].append(result)

        if scored % 200 == 0:
            _save_state(state, state_file)

    _save_state(state, state_file)
    elapsed = time.time() - t0
    print(f"\nSweep complete: {tested} checked, {scored} scored in {elapsed:.1f}s")
    _show_results(state)


def _show_results(state: dict) -> None:
    """Display ranked letter-index sweep results."""
    results = state.get("results", [])
    if not results:
        print("\nNo results yet.")
        return

    print(f"\n--- TOP 20 LETTER-INDEX CANDIDATES ---")
    print(f"  Total scored: {len(results)}")

    for cipher_name, bg_key in [("B3", "b3_bigram"), ("B1", "b1_bigram")]:
        valid = [r for r in results if r.get(bg_key) is not None]
        if not valid:
            print(f"\n  {cipher_name}: no valid results")
            continue

        sorted_r = sorted(valid, key=lambda x: -x[bg_key])
        print(f"\n  {cipher_name} (sorted by bigram score, {len(valid)} texts):")
        print(f"  {'Rank':>4} {'File':<20} {'Alpha':>7} {'Bigram':>8} {'IC':>8} {'InRange':>8}")
        print("  " + "-" * 60)
        for rank, r in enumerate(sorted_r[:20], 1):
            print(f"  {rank:>4} {r['file']:<20} {r['alpha_len']:>7} "
                  f"{r[bg_key]:>8.3f} {r.get(bg_key.replace('bigram', 'ic')) or 0:>8.4f} "
                  f"{r.get(bg_key.replace('bigram', 'in_range')) or 0:>7.1%}")

    # Best overall
    best_b3 = max(results, key=lambda x: x.get("b3_bigram") or -999)
    has_b1 = [r for r in results if r.get("b1_bigram") is not None]
    if has_b1:
        best_b1 = max(has_b1, key=lambda x: x["b1_bigram"])
        print(f"\n  Best B1: {best_b1['file']} (bigram={best_b1['b1_bigram']:.3f})")
    print(f"  Best B3: {best_b3['file']} (bigram={best_b3.get('b3_bigram', 0):.3f})")
    print(f"  B2 word-decode baseline: bigram ~-2.805")


def show_results(state_file: Path = STATE_FILE) -> None:
    """Load and display results from previous runs."""
    state = _load_state(state_file)
    _show_results(state)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Letter-index cipher hypothesis for Beale B1/B3"
    )
    parser.add_argument("--doi-only", action="store_true",
                        help="Sections 1-3: validation, DoI test, homophones")
    parser.add_argument("--stats", action="store_true",
                        help="Section 4: synthetic cipher comparison")
    parser.add_argument("--sweep", action="store_true",
                        help="Section 5: corpus sweep with letter-index keys")
    parser.add_argument("--results", action="store_true",
                        help="Show results from previous sweep")
    parser.add_argument("--state-file", type=str, default=str(STATE_FILE),
                        help="State file for sweep (default: letter_search_state.json)")
    args = parser.parse_args()

    state_file = Path(args.state_file)

    # Default: run everything
    run_all = not (args.doi_only or args.stats or args.sweep or args.results)

    if args.results:
        show_results(state_file)
        return

    if run_all or args.doi_only:
        run_validation()
        run_doi_test()
        run_homophone_analysis()

    if run_all or args.stats:
        run_stats()

    if run_all or args.sweep:
        run_sweep(state_file)


if __name__ == "__main__":
    main()
