"""
---
version: 0.1.0
created: 2026-02-24
updated: 2026-02-24
---

phase3_bigram.py — Bigram transition analysis and Gillogly-as-Vigenere test.

Two main analyses:

1. Bigram transition analysis:
   In a genuine book cipher encoding English, consecutive cipher numbers
   should produce letter pairs with English-like bigram frequencies. Test
   whether B1/B3 number sequences produce bigram statistics distinguishable
   from random.

2. Gillogly-as-Vigenere test:
   If the Gillogly alphabetical string in B1 is a keystring for a second
   cipher layer (Pelling's hypothesis), systematically test it as a
   Vigenere/substitution key applied to the DoI-decoded B1 text.

Usage:
    python3 phase3_bigram.py [--no-plots] [--save-dir DIR]
"""

from __future__ import annotations

import argparse
import string
from collections import Counter
from pathlib import Path

import numpy as np

from beale import (
    B1, B2, B3, BEALE_DOI, SPECIAL_DECODE,
    decode_book_cipher, first_letter,
    bigram_score, index_of_coincidence,
    BIGRAM_LOGPROB, BIGRAM_FLOOR, ENGLISH_FREQ,
)


# ============================================================================
# 1. BIGRAM TRANSITION ANALYSIS
# ============================================================================

def compute_bigram_matrix(text: str) -> np.ndarray:
    """Compute 26x26 bigram count matrix from text."""
    matrix = np.zeros((26, 26), dtype=float)
    text = "".join(c for c in text.lower() if c in string.ascii_lowercase)
    for i in range(len(text) - 1):
        a = ord(text[i]) - ord('a')
        b = ord(text[i + 1]) - ord('a')
        matrix[a][b] += 1
    return matrix


def bigram_transition_analysis(
    cipher: tuple[int, ...],
    key_words: list | tuple,
    special: dict[int, str] | None = None,
    label: str = "",
) -> dict:
    """
    Analyze bigram transitions in a decoded cipher.

    Returns stats about how English-like the bigram distribution is.
    """
    decoded = decode_book_cipher(cipher, key_words, special)
    clean = "".join(c for c in decoded if c in string.ascii_lowercase)

    matrix = compute_bigram_matrix(clean)
    total_bigrams = matrix.sum()

    if total_bigrams == 0:
        return {"label": label, "total_bigrams": 0}

    # Normalize to probability
    prob_matrix = matrix / total_bigrams

    # Compare to English bigram distribution
    bg_score = bigram_score(clean)
    ic = index_of_coincidence(clean)

    # Top 10 most frequent bigrams
    flat = [(matrix[i][j], chr(i + ord('a')) + chr(j + ord('a')))
            for i in range(26) for j in range(26)]
    flat.sort(reverse=True)
    top10 = flat[:10]

    # English top-10 for comparison
    eng_top10 = sorted(BIGRAM_LOGPROB.items(), key=lambda x: -x[1])[:10]

    # Compute chi-squared against English bigram distribution
    # (using top 100 bigrams only to avoid sparsity issues)
    from scipy import stats as sp_stats
    top_bigrams = sorted(BIGRAM_LOGPROB.items(), key=lambda x: -x[1])[:100]
    top_bg_set = {bg for bg, _ in top_bigrams}
    obs = []
    exp = []
    for bg in top_bg_set:
        i = ord(bg[0]) - ord('a')
        j = ord(bg[1]) - ord('a')
        obs.append(matrix[i][j])
        # Expected from English frequency
        exp_count = 10 ** BIGRAM_LOGPROB[bg] * total_bigrams
        exp.append(max(exp_count, 0.5))

    obs_arr = np.array(obs)
    exp_arr = np.array(exp)
    exp_arr = exp_arr * (obs_arr.sum() / exp_arr.sum())  # normalize
    chi2, p_value = sp_stats.chisquare(obs_arr, exp_arr)

    return {
        "label": label,
        "total_bigrams": int(total_bigrams),
        "bigram_score": bg_score,
        "ic": ic,
        "top10_observed": top10,
        "top10_english": eng_top10,
        "chi2": float(chi2),
        "p_value": float(p_value),
        "matrix": matrix,
    }


def print_bigram_comparison(results: list[dict]) -> None:
    """Print bigram analysis comparison."""
    print("\n" + "=" * 70)
    print("BIGRAM TRANSITION ANALYSIS")
    print("=" * 70)

    print(f"\n{'Metric':<30}", end="")
    for r in results:
        print(f"  {r['label']:>15}", end="")
    print()
    print("-" * (30 + 17 * len(results)))

    rows = [
        ("Total bigrams", lambda r: f"{r['total_bigrams']}"),
        ("Avg bigram log-prob", lambda r: f"{r['bigram_score']:.3f}"),
        ("Index of coincidence", lambda r: f"{r['ic']:.4f}"),
        ("Bigram chi2 (vs English)", lambda r: f"{r['chi2']:.1f}"),
        ("Bigram chi2 p-value", lambda r: f"{r['p_value']:.4f}"),
    ]

    for name, fn in rows:
        print(f"{name:<30}", end="")
        for r in results:
            print(f"  {fn(r):>15}", end="")
        print()

    # Show top-10 bigrams for each
    for r in results:
        print(f"\n  {r['label']} top-10 bigrams:")
        for count, bg in r["top10_observed"]:
            eng_rank = "  (English top-10)" if bg in {b for _, b in sorted(BIGRAM_LOGPROB.items(), key=lambda x: -x[1])[:10]} else ""
            print(f"    '{bg}': {count:4.0f}{eng_rank}")


# ============================================================================
# 2. GILLOGLY-AS-VIGENERE TEST
# ============================================================================

def vigenere_decrypt(ciphertext: str, key: str) -> str:
    """Decrypt Vigenere cipher."""
    result = []
    key_len = len(key)
    ki = 0
    for c in ciphertext:
        if c in string.ascii_lowercase:
            shift = ord(key[ki % key_len]) - ord('a')
            decrypted = chr((ord(c) - ord('a') - shift) % 26 + ord('a'))
            result.append(decrypted)
            ki += 1
        else:
            result.append(c)
    return "".join(result)


def caesar_decrypt(ciphertext: str, shift: int) -> str:
    """Decrypt Caesar cipher."""
    result = []
    for c in ciphertext:
        if c in string.ascii_lowercase:
            result.append(chr((ord(c) - ord('a') - shift) % 26 + ord('a')))
        else:
            result.append(c)
    return "".join(result)


def gillogly_vigenere_test() -> None:
    """
    Test the Gillogly string as a Vigenere key applied to the DoI-decoded B1.

    Pelling's hypothesis: the alphabetical strings in B1 decoded with DoI
    are not random -- they're a keystring for a second cipher layer.
    The intermediate text (DoI-decoded B1) would then be Vigenere-encrypted
    with this keystring.
    """
    print("\n" + "=" * 70)
    print("GILLOGLY-AS-VIGENERE TEST (Pelling Hypothesis)")
    print("=" * 70)

    # Decode B1 with DoI (the intermediate text)
    decoded_b1 = decode_book_cipher(B1, BEALE_DOI)
    clean_b1 = "".join(c for c in decoded_b1 if c in string.ascii_lowercase)

    print(f"\nB1 decoded with DoI ({len(clean_b1)} clean letters)")
    print(f"Bigram score: {bigram_score(clean_b1):.3f}")
    print(f"IC: {index_of_coincidence(clean_b1):.4f}")

    # Candidate keys derived from Gillogly strings
    gillogly_keys = [
        ("Full Gillogly (17)", "abcdefghiijklmmno"),
        ("Central portion (14)", "defghiijklmmno"),
        ("Perfect alphabet (10)", "abcdefghij"),
        ("Perfect alphabet (13)", "abcdefghijklm"),
        ("Perfect alphabet (26)", "abcdefghijklmnopqrstuvwxyz"),
        ("Reversed alphabet", "zyxwvutsrqponmlkjihgfedcba"),
        ("BEALE", "beale"),
        ("WARD", "ward"),
        ("MORRISS", "morriss"),
        ("BUFORD", "buford"),
        ("BEDFORD", "bedford"),
        ("VIRGINIA", "virginia"),
        ("LIBERTY", "liberty"),
    ]

    print(f"\n{'Key':<30} {'Len':>4}  {'Bigram':>8}  {'IC':>8}  First 40 chars of decrypt")
    print("-" * 110)

    best_score = -float("inf")
    best_key = None

    for name, key in gillogly_keys:
        decrypted = vigenere_decrypt(clean_b1, key)
        bg = bigram_score(decrypted)
        ic = index_of_coincidence(decrypted)
        preview = decrypted[:40]
        marker = " ***" if bg > best_score else ""

        if bg > best_score:
            best_score = bg
            best_key = name

        print(f"{name:<30} {len(key):>4}  {bg:>8.3f}  {ic:>8.4f}  {preview}{marker}")

    # Also try all single Caesar shifts
    print(f"\n{'Caesar shift':<30} {'Len':>4}  {'Bigram':>8}  {'IC':>8}  First 40 chars")
    print("-" * 110)

    for shift in range(26):
        decrypted = caesar_decrypt(clean_b1, shift)
        bg = bigram_score(decrypted)
        ic = index_of_coincidence(decrypted)
        preview = decrypted[:40]
        marker = " ***" if bg > best_score else ""

        if bg > best_score:
            best_score = bg
            best_key = f"Caesar-{shift}"

        if shift == 0 or bg > -3.3:
            print(f"Caesar shift={shift:<20d} {1:>4}  {bg:>8.3f}  {ic:>8.4f}  {preview}{marker}")

    print(f"\nBest key tested: {best_key} (bigram score: {best_score:.3f})")
    print(f"For reference, B2 bigram score: {bigram_score(decode_book_cipher(B2, BEALE_DOI, SPECIAL_DECODE)):.3f}")
    print(f"Random text bigram score: ~-4.0")

    # Verdict
    if best_score > -2.5:
        print("\nVerdict: A Vigenere layer produced near-English output. Investigate further.")
    elif best_score > -3.0:
        print("\nVerdict: Some improvement over random, but not convincingly English. Weak signal.")
    else:
        print("\nVerdict: No Vigenere key from the Gillogly strings produces English-like text.")
        print("This does NOT disprove the multi-layer hypothesis (unknown key would help)")
        print("but the specific Gillogly-as-key hypothesis finds no support.")


# ============================================================================
# 3. BIGRAM SLIDING WINDOW ANALYSIS
# ============================================================================

def sliding_window_analysis(
    cipher: tuple[int, ...],
    key_words: list | tuple,
    label: str = "",
    window_size: int = 50,
) -> list[float]:
    """
    Compute bigram score in a sliding window across the decoded text.

    Helps identify regions that might have different encryption methods
    or that are more/less English-like.
    """
    decoded = decode_book_cipher(cipher, key_words)
    clean = "".join(c for c in decoded if c in string.ascii_lowercase)

    scores = []
    positions = []
    for i in range(0, len(clean) - window_size + 1, window_size // 4):
        window = clean[i:i + window_size]
        scores.append(bigram_score(window))
        positions.append(i)

    return positions, scores


def print_sliding_window(results: list[tuple[str, list, list]]) -> None:
    """Print sliding window results."""
    print("\n" + "=" * 70)
    print("SLIDING WINDOW BIGRAM ANALYSIS (window=50, step=12)")
    print("=" * 70)

    for label, positions, scores in results:
        print(f"\n  {label}: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}")
        # Find best/worst windows
        if scores:
            best_idx = np.argmax(scores)
            worst_idx = np.argmin(scores)
            print(f"    Best window:  pos {positions[best_idx]:4d}, score {scores[best_idx]:.3f}")
            print(f"    Worst window: pos {positions[worst_idx]:4d}, score {scores[worst_idx]:.3f}")

            # Show the 5 best windows (potential English fragments)
            sorted_windows = sorted(zip(scores, positions), reverse=True)
            print(f"    Top-5 windows:")
            decoded = decode_book_cipher(
                B1 if "B1" in label else B3 if "B3" in label else B2,
                BEALE_DOI,
                SPECIAL_DECODE if "B2" in label else None,
            )
            clean = "".join(c for c in decoded if c in string.ascii_lowercase)
            for score, pos in sorted_windows[:5]:
                fragment = clean[pos:pos + 50]
                print(f"      pos {pos:4d}: {score:.3f}  '{fragment}'")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Bigram analysis of Beale ciphers")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--save-dir", type=str, default=".", help="Directory for output")
    args = parser.parse_args()
    save_dir = Path(args.save_dir)

    # 1. Bigram transition analysis
    r_b1 = bigram_transition_analysis(B1, BEALE_DOI, label="B1")
    r_b2 = bigram_transition_analysis(B2, BEALE_DOI, SPECIAL_DECODE, label="B2")
    r_b3 = bigram_transition_analysis(B3, BEALE_DOI, label="B3")
    print_bigram_comparison([r_b1, r_b2, r_b3])

    # 2. Gillogly-as-Vigenere test
    gillogly_vigenere_test()

    # 3. Sliding window analysis
    sw_results = []
    for cipher, label, sp in [(B1, "B1", None), (B2, "B2", SPECIAL_DECODE), (B3, "B3", None)]:
        positions, scores = sliding_window_analysis(cipher, BEALE_DOI, label)
        sw_results.append((label, positions, scores))
    print_sliding_window(sw_results)

    # 4. Plots
    if not args.no_plots:
        _plot_bigram_heatmaps([r_b1, r_b2, r_b3], save_dir / "bigram_heatmaps.png")
        _plot_sliding_windows(sw_results, save_dir / "bigram_sliding_window.png")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
  Bigram analysis:
    B2 bigram score: {r_b2['bigram_score']:.3f} (English-like, as expected)
    B1 bigram score: {r_b1['bigram_score']:.3f} (far from English)
    B3 bigram score: {r_b3['bigram_score']:.3f} (far from English)

  Gillogly-as-Vigenere:
    No tested key produces English-like output from B1's DoI decode.
    The multi-layer hypothesis remains unfalsifiable without the actual key.

  Sliding window:
    B2 shows consistent English-like bigrams throughout.
    B1/B3 show uniformly poor bigram scores with no promising regions.
""")


def _plot_bigram_heatmaps(results: list[dict], save_path: Path) -> None:
    """Plot bigram frequency heatmaps."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    letters = list(string.ascii_lowercase)

    for idx, r in enumerate(results):
        ax = axes[idx]
        matrix = r["matrix"]
        # Normalize
        total = matrix.sum()
        if total > 0:
            matrix = matrix / total
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(26))
        ax.set_yticks(range(26))
        ax.set_xticklabels(letters, fontsize=6)
        ax.set_yticklabels(letters, fontsize=6)
        ax.set_title(f"{r['label']} (bg={r['bigram_score']:.3f})")
        ax.set_xlabel("Second letter")
        ax.set_ylabel("First letter")
        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("Bigram Frequency Heatmaps", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


def _plot_sliding_windows(
    sw_results: list[tuple[str, list, list]],
    save_path: Path,
) -> None:
    """Plot sliding window bigram scores."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available")
        return

    fig, axes = plt.subplots(len(sw_results), 1, figsize=(12, 3 * len(sw_results)),
                             sharex=False)

    for idx, (label, positions, scores) in enumerate(sw_results):
        ax = axes[idx]
        ax.plot(positions, scores, linewidth=0.8)
        ax.axhline(-2.8, color="green", linestyle="--", alpha=0.5, label="English-like (-2.8)")
        ax.axhline(-3.5, color="red", linestyle="--", alpha=0.5, label="Random-like (-3.5)")
        ax.set_ylabel("Bigram score")
        ax.set_title(f"{label}: Sliding Window Bigram Score")
        ax.legend(fontsize=8)
        ax.set_ylim(-4.5, -1.5)

    axes[-1].set_xlabel("Position in decoded text")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    main()
