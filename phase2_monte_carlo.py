"""
---
version: 0.1.0
created: 2026-02-24
updated: 2026-02-24
---

phase2_monte_carlo.py — Monte Carlo classification of B1/B3.

Generates two populations of synthetic ciphers:
  1. Genuine: random English-frequency text encoded with the DoI as key
  2. Fake: uniformly random numbers in a matching range

Runs the stats battery on each synthetic cipher and computes where
B1 and B3 fall within the distributions.

If B1/B3 fall within the genuine distribution, that's evidence for
authenticity. If they fall within the fake distribution (or outside both),
that's evidence for fabrication.

Usage:
    python3 phase2_monte_carlo.py [--n-sims N] [--no-plots] [--save-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from beale import (
    B1, B2, B3, BEALE_DOI, SPECIAL_DECODE,
    benford_test, last_digit_test, distinct_ratio, letter_frequency_test,
    bigram_score, index_of_coincidence,
    decode_book_cipher, first_letter,
    generate_fake_cipher, generate_random_numbers,
    ENGLISH_FREQ,
)


def compute_stats(cipher: list[int] | tuple[int, ...], key_words=BEALE_DOI) -> dict:
    """Compute a compact stats vector for a cipher."""
    dr = distinct_ratio(cipher)
    bf = benford_test(cipher)
    ld10 = last_digit_test(cipher, base=10)
    ld7 = last_digit_test(cipher, base=7)
    ld3 = last_digit_test(cipher, base=3)

    decoded = decode_book_cipher(cipher, key_words)
    clean = "".join(c for c in decoded if c.isalpha())
    bg = bigram_score(clean) if len(clean) >= 2 else -4.0
    ic = index_of_coincidence(clean)

    return {
        "distinct_ratio": dr["ratio"],
        "benford_chi2": bf["chi2"],
        "benford_epsilon": bf["epsilon"],
        "ld10_chi2": ld10["chi2"],
        "ld10_p": ld10["p_value"],
        "ld7_chi2": ld7["chi2"],
        "ld7_p": ld7["p_value"],
        "ld3_chi2": ld3["chi2"],
        "ld3_p": ld3["p_value"],
        "bigram_score": bg,
        "ic": ic,
    }


def run_monte_carlo(
    n_sims: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Generate genuine and fake cipher populations, compute stats.

    Returns dict with:
        genuine_stats: list of stat dicts for genuine ciphers
        fake_stats: list of stat dicts for random ciphers
        b1_stats, b2_stats, b3_stats: stat dicts for real ciphers
    """
    rng = np.random.default_rng(seed)

    # Real cipher stats
    print("Computing stats for real ciphers...")
    b1_stats = compute_stats(B1)
    b2_stats = compute_stats(B2)
    b3_stats = compute_stats(B3)

    # Generate ciphers matching B1 (520 numbers, range up to ~1311 since DoI is key)
    # and B3 (618 numbers, range up to ~975)
    # We'll generate for both lengths

    genuine_stats_520: list[dict] = []
    genuine_stats_618: list[dict] = []
    fake_stats_520: list[dict] = []
    fake_stats_618: list[dict] = []

    t0 = time.time()

    for i in range(n_sims):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_sims * 2 - (i + 1)) / rate  # *2 for fake pass
            print(f"  Genuine {i+1}/{n_sims} ({rate:.0f}/s, ETA {eta:.0f}s)", end="\r")

        # Genuine: encode random English text with DoI
        try:
            g520 = generate_fake_cipher(520, BEALE_DOI, rng)
            genuine_stats_520.append(compute_stats(g520))
        except ValueError:
            pass  # Skip if a rare letter has no homophone

        try:
            g618 = generate_fake_cipher(618, BEALE_DOI, rng)
            genuine_stats_618.append(compute_stats(g618))
        except ValueError:
            pass

    print(f"\n  Genuine: {len(genuine_stats_520)} sims (520), {len(genuine_stats_618)} sims (618)")

    for i in range(n_sims):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  Fake {i+1}/{n_sims}", end="\r")

        # Fake (B1-like): random numbers 1-2906, 520 of them
        f520 = generate_random_numbers(520, 2906, rng)
        fake_stats_520.append(compute_stats(f520))

        # Fake (B3-like): random numbers 1-975, 618 of them
        f618 = generate_random_numbers(618, 975, rng)
        fake_stats_618.append(compute_stats(f618))

    elapsed = time.time() - t0
    print(f"\n  Fake: {len(fake_stats_520)} sims (520), {len(fake_stats_618)} sims (618)")
    print(f"  Total time: {elapsed:.1f}s")

    return {
        "genuine_520": genuine_stats_520,
        "genuine_618": genuine_stats_618,
        "fake_520": fake_stats_520,
        "fake_618": fake_stats_618,
        "b1_stats": b1_stats,
        "b2_stats": b2_stats,
        "b3_stats": b3_stats,
    }


def compute_percentiles(
    real_val: float,
    genuine_dist: list[float],
    fake_dist: list[float],
) -> dict:
    """Compute where a real value falls in genuine and fake distributions."""
    g_arr = np.array(genuine_dist)
    f_arr = np.array(fake_dist)

    g_pct = float(np.mean(g_arr <= real_val) * 100)
    f_pct = float(np.mean(f_arr <= real_val) * 100)

    return {
        "value": real_val,
        "genuine_pct": g_pct,
        "fake_pct": f_pct,
        "genuine_mean": float(np.mean(g_arr)),
        "genuine_std": float(np.std(g_arr)),
        "fake_mean": float(np.mean(f_arr)),
        "fake_std": float(np.std(f_arr)),
    }


def print_classification(results: dict) -> None:
    """Print classification results for B1 and B3."""
    metrics = [
        "distinct_ratio", "benford_chi2", "benford_epsilon",
        "ld10_chi2", "ld7_chi2", "ld3_chi2",
        "ld7_p", "ld3_p",
        "bigram_score", "ic",
    ]
    metric_labels = {
        "distinct_ratio": "Distinct ratio",
        "benford_chi2": "Benford chi2",
        "benford_epsilon": "Benford epsilon",
        "ld10_chi2": "Last-digit (b10) chi2",
        "ld7_chi2": "Last-digit (b7) chi2",
        "ld3_chi2": "Last-digit (b3) chi2",
        "ld7_p": "Last-digit (b7) p-val",
        "ld3_p": "Last-digit (b3) p-val",
        "bigram_score": "Bigram log-prob",
        "ic": "Index of coincidence",
    }

    for cipher_name, stats_key, gen_key, fake_key in [
        ("B1 (520 numbers)", "b1_stats", "genuine_520", "fake_520"),
        ("B3 (618 numbers)", "b3_stats", "genuine_618", "fake_618"),
    ]:
        print(f"\n{'='*70}")
        print(f"CLASSIFICATION: {cipher_name}")
        print(f"{'='*70}")

        print(f"\n{'Metric':<25} {'Value':>10} {'Gen pct':>10} {'Fake pct':>10}  "
              f"{'Gen mean':>10} {'Fake mean':>10}  Verdict")
        print("-" * 100)

        genuine_votes = 0
        fake_votes = 0
        ambiguous = 0

        for m in metrics:
            real_val = results[stats_key][m]
            gen_vals = [s[m] for s in results[gen_key]]
            fake_vals = [s[m] for s in results[fake_key]]

            pct = compute_percentiles(real_val, gen_vals, fake_vals)

            # Determine verdict: closer to which distribution?
            g_zscore = abs(real_val - pct["genuine_mean"]) / max(pct["genuine_std"], 1e-10)
            f_zscore = abs(real_val - pct["fake_mean"]) / max(pct["fake_std"], 1e-10)

            if g_zscore < f_zscore and g_zscore < 2.0:
                verdict = "GENUINE"
                genuine_votes += 1
            elif f_zscore < g_zscore and f_zscore < 2.0:
                verdict = "FAKE"
                fake_votes += 1
            elif g_zscore < f_zscore:
                verdict = "~genuine"
                genuine_votes += 0.5
                ambiguous += 1
            else:
                verdict = "~fake"
                fake_votes += 0.5
                ambiguous += 1

            label = metric_labels.get(m, m)
            print(f"{label:<25} {real_val:>10.4f} {pct['genuine_pct']:>9.1f}% "
                  f"{pct['fake_pct']:>9.1f}%  {pct['genuine_mean']:>10.4f} "
                  f"{pct['fake_mean']:>10.4f}  {verdict}")

        total = genuine_votes + fake_votes
        print(f"\n  Votes: GENUINE={genuine_votes:.1f}  FAKE={fake_votes:.1f}  "
              f"(ambiguous={ambiguous})")
        if total > 0:
            fake_pct = fake_votes / total * 100
            print(f"  Fake probability estimate: {fake_pct:.0f}%")


def plot_distributions(
    results: dict,
    save_dir: Path,
) -> None:
    """Generate distribution comparison plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping plots")
        return

    metrics_to_plot = [
        ("distinct_ratio", "Distinct Value Ratio"),
        ("benford_chi2", "Benford Chi-Squared"),
        ("ld7_p", "Last-Digit Base-7 p-value"),
        ("bigram_score", "Bigram Log-Probability"),
        ("ic", "Index of Coincidence"),
    ]

    for cipher_name, stats_key, gen_key, fake_key, marker_color in [
        ("B1", "b1_stats", "genuine_520", "fake_520", "red"),
        ("B3", "b3_stats", "genuine_618", "fake_618", "blue"),
    ]:
        fig, axes = plt.subplots(1, len(metrics_to_plot),
                                 figsize=(4 * len(metrics_to_plot), 4),
                                 squeeze=False)

        for col, (metric, title) in enumerate(metrics_to_plot):
            ax = axes[0][col]
            gen_vals = [s[metric] for s in results[gen_key]]
            fake_vals = [s[metric] for s in results[fake_key]]
            real_val = results[stats_key][metric]

            ax.hist(gen_vals, bins=30, alpha=0.5, label="Genuine", color="green", density=True)
            ax.hist(fake_vals, bins=30, alpha=0.5, label="Fake", color="gray", density=True)
            ax.axvline(real_val, color=marker_color, linewidth=2, linestyle="--",
                      label=f"{cipher_name} actual")
            ax.set_title(title, fontsize=9)
            ax.legend(fontsize=7)

        plt.suptitle(f"{cipher_name} vs Synthetic Distributions", fontsize=12)
        plt.tight_layout()
        path = save_dir / f"monte_carlo_{cipher_name.lower()}.png"
        plt.savefig(str(path), dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo classification of Beale ciphers")
    parser.add_argument("--n-sims", type=int, default=1000, help="Number of simulations per class")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--save-dir", type=str, default=".", help="Directory for output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)

    print("=" * 70)
    print("MONTE CARLO CLASSIFICATION OF BEALE CIPHERS")
    print(f"Simulations: {args.n_sims} genuine + {args.n_sims} fake per cipher length")
    print("=" * 70)

    results = run_monte_carlo(n_sims=args.n_sims, seed=args.seed)

    # Also show B2 for calibration
    print("\n--- B2 calibration (should classify as GENUINE) ---")
    b2_stats = results["b2_stats"]
    for m in ["distinct_ratio", "bigram_score", "ic"]:
        gen_vals = [s[m] for s in results["genuine_618"]]
        pct = float(np.mean(np.array(gen_vals) <= b2_stats[m]) * 100)
        print(f"  B2 {m}: {b2_stats[m]:.4f} (genuine percentile: {pct:.1f}%)")

    print_classification(results)

    if not args.no_plots:
        print("\nGenerating plots...")
        plot_distributions(results, save_dir)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
  This Monte Carlo analysis generates two populations of synthetic ciphers:
    - Genuine: random English text encoded with the DoI as key
    - Fake: uniformly random numbers in matching ranges

  For each metric, we check whether B1/B3 fall closer to the genuine or
  fake distribution. The more metrics that align with "fake", the stronger
  the evidence for fabrication.

  Key discriminators:
    - Distinct ratio: genuine ~24%, fake ~57% (for B1 range)
    - Last-digit base-7 p-value: genuine <0.05, fake >0.05
    - Bigram score: genuine ~-2.8, fake ~-3.5
""")


if __name__ == "__main__":
    main()
