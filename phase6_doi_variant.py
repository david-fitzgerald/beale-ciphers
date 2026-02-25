"""
---
version: 0.1.0
created: 2026-02-25
updated: 2026-02-25
---

phase6_doi_variant.py — DoI variant optimization for Beale B1.

The Gillogly strings (p < 10^-12) prove DoI involvement in B1. But the
longest run breaks at position 204 (cipher 301 → 'h', needs 'p'). This
module tests whether a slightly different DoI variant could extend the
alphabetical runs — evidence that the original key was a close but not
exact match for our reconstructed DoI word list.

Key constraint: position 204, cipher=301, word="history"→'h', needs 'p'.
Word 304="present"→'p' is offset +3. No simple ±1 fix works alone.

Sections:
  1. Constraint mapping — decode table around Gillogly runs
  2. Global offset testing — apply ±N to ALL cipher numbers
  3. Per-position greedy — try ±1/±2/±3 at each break point
  4. Word insertion/deletion — insert/delete common words in DoI
  5. Hill-climbing optimizer — random mutations, accept if quality improves
  6. Multi-run coherence — can ONE variant improve all three runs?

Usage:
    python3 phase6_doi_variant.py --constraints    # section 1
    python3 phase6_doi_variant.py --offsets        # section 2
    python3 phase6_doi_variant.py --mutations      # sections 3-4
    python3 phase6_doi_variant.py --optimize       # section 5
    python3 phase6_doi_variant.py --all            # everything
"""

from __future__ import annotations

import argparse
import copy
import string
import time
from pathlib import Path

import numpy as np

from beale import (
    B1, BEALE_DOI,
    decode_book_cipher, first_letter,
    bigram_score, gillogly_quality,
)


# ============================================================================
# HELPERS
# ============================================================================

# The three main Gillogly runs in B1 (positions in decoded output)
MAIN_RUNS = [
    {"name": "run1", "start": 43, "end": 53, "expected_len": 11},
    {"name": "run2", "start": 83, "end": 93, "expected_len": 11},
    {"name": "run3", "start": 187, "end": 203, "expected_len": 17},
]

COMMON_WORDS = ["the", "a", "and", "of", "to", "in", "is", "that", "it", "for"]


def decode_with_wordlist(cipher: tuple[int, ...], words: list[str]) -> str:
    """Decode B1 with a given word list (no special overrides)."""
    return decode_book_cipher(cipher, words)


def measure_runs(decoded: str, min_run: int = 5) -> dict:
    """Wrapper around gillogly_quality for decoded text."""
    return gillogly_quality(decoded, min_run)


# ============================================================================
# 1. CONSTRAINT MAPPING
# ============================================================================

def run_constraints() -> None:
    """Map out constraints around Gillogly run break points."""
    print("=" * 70)
    print("SECTION 1: CONSTRAINT MAPPING")
    print("=" * 70)

    words = list(BEALE_DOI)
    decoded = decode_with_wordlist(B1, words)

    # Show context around each major run
    for run_info in MAIN_RUNS:
        start = max(0, run_info["start"] - 3)
        end = min(len(decoded), run_info["end"] + 5)
        print(f"\n--- {run_info['name']}: positions {run_info['start']}-{run_info['end']} ---")
        print(f"  {'Pos':>4} {'Cipher':>7} {'Word':>15} {'Letter':>7} {'Needed':>7} "
              f"{'±1 word':>15} {'±1 letter':>10}")
        print("  " + "-" * 70)

        for pos in range(start, end + 1):
            if pos >= len(B1):
                break
            cnum = B1[pos]
            word = words[cnum - 1] if 1 <= cnum <= len(words) else "???"
            letter = first_letter(word) if word != "???" else "?"

            # What letter would continue the ascending run?
            if pos > 0 and pos < len(decoded):
                prev = decoded[pos - 1]
                needed = chr(max(ord(prev), ord(letter)))  # at least prev
                if letter < decoded[pos - 1]:
                    needed = decoded[pos - 1]  # must be >= previous
                else:
                    needed = "-"  # already fine
            else:
                needed = "-"

            # What do ±1/±2/±3 offsets give?
            offset_letters = []
            for offset in [-3, -2, -1, 1, 2, 3]:
                adj = cnum + offset
                if 1 <= adj <= len(words):
                    w = words[adj - 1]
                    offset_letters.append(f"{offset:+d}={first_letter(w)}")

            # Mark if this is a run break
            marker = ""
            if pos > run_info["start"] and pos <= run_info["end"] + 1:
                if pos < len(decoded) and pos > 0 and decoded[pos] < decoded[pos - 1]:
                    marker = " <<<BREAK"

            print(f"  {pos:>4} {cnum:>7} {word:>15} {letter:>7} {needed:>7}  "
                  f"{', '.join(offset_letters[:4])}{marker}")

    # Detailed analysis of the main break at position 204
    print("\n--- CRITICAL BREAK: Position 204 ---")
    cnum = B1[203]  # 0-indexed
    print(f"  Cipher number: {cnum}")
    print(f"  Current word: '{words[cnum - 1]}' → '{first_letter(words[cnum - 1])}'")
    print(f"  Previous decoded: '{decoded[202]}' (pos 202)")
    print(f"  Need: >= '{decoded[202]}' (ideally 'p' to continue abcdefghiijklmmno...)")
    for offset in range(-5, 6):
        adj = cnum + offset
        if 1 <= adj <= len(words):
            w = words[adj - 1]
            fl = first_letter(w)
            mark = " <<<" if fl >= "o" else ""
            print(f"    offset {offset:+d}: word {adj} = '{w}' → '{fl}'{mark}")


# ============================================================================
# 2. GLOBAL OFFSET TESTING
# ============================================================================

def run_offsets() -> None:
    """Test global offsets applied to ALL cipher numbers."""
    print("\n" + "=" * 70)
    print("SECTION 2: GLOBAL OFFSET TESTING")
    print("=" * 70)

    words = list(BEALE_DOI)
    baseline_decoded = decode_with_wordlist(B1, words)
    baseline_gq = measure_runs(baseline_decoded)
    print(f"\nBaseline: longest_run={baseline_gq['longest_run']}, "
          f"total={baseline_gq['total_run_length']}, score={baseline_gq['score']:.1f}")

    print(f"\n{'Offset':>7} {'Longest':>8} {'Total':>6} {'#Runs':>6} {'Score':>7}  Best run")
    print("-" * 55)

    for offset in range(-10, 11):
        shifted = tuple(n + offset for n in B1)
        # Clamp to valid range
        decoded = decode_with_wordlist(shifted, words)
        gq = measure_runs(decoded)
        best_run = max(gq["runs"], key=lambda r: r["length"]) if gq["runs"] else {"length": 0, "letters": ""}
        marker = " <<<" if gq["longest_run"] > baseline_gq["longest_run"] else ""
        print(f"  {offset:>+5} {gq['longest_run']:>8} {gq['total_run_length']:>6} "
              f"{gq['run_count']:>6} {gq['score']:>7.1f}  "
              f"'{best_run.get('letters', '')[:20]}'{marker}")


# ============================================================================
# 3. PER-POSITION GREEDY OPTIMIZATION
# ============================================================================

def run_greedy() -> None:
    """At each run break, try small offsets to extend the run."""
    print("\n" + "=" * 70)
    print("SECTION 3: PER-POSITION GREEDY OPTIMIZATION")
    print("=" * 70)

    words = list(BEALE_DOI)
    decoded = decode_with_wordlist(B1, words)

    # Find all break points within or just after the main runs
    for run_info in MAIN_RUNS:
        print(f"\n--- {run_info['name']}: positions {run_info['start']}-{run_info['end']} ---")
        # Look at the position after the run ends — that's the break
        end_pos = run_info["end"] + 1
        if end_pos >= len(B1):
            continue

        cnum = B1[end_pos]
        letter = decoded[end_pos]
        prev_letter = decoded[end_pos - 1]
        print(f"  Break at pos {end_pos}: cipher={cnum}, decoded='{letter}', "
              f"previous='{prev_letter}', need >= '{prev_letter}'")

        # Try offsets on this single position
        best_offset = 0
        best_extension = 0
        for offset in range(-5, 6):
            if offset == 0:
                continue
            adj = cnum + offset
            if not (1 <= adj <= len(words)):
                continue
            new_letter = first_letter(words[adj - 1])
            if new_letter >= prev_letter:
                # Check how far the run would extend
                extension = 1
                for look in range(end_pos + 1, min(end_pos + 20, len(decoded))):
                    if decoded[look] >= (new_letter if look == end_pos + 1 else decoded[look - 1]):
                        extension += 1
                    else:
                        break
                if extension > best_extension:
                    best_extension = extension
                    best_offset = offset
                print(f"    offset {offset:+d}: word {adj}='{words[adj-1]}'→'{new_letter}', "
                      f"extends run by {extension}")

        if best_offset:
            print(f"  Best: offset {best_offset:+d}, extends by {best_extension}")
        else:
            print(f"  No single-offset fix available")


# ============================================================================
# 4. WORD INSERTION/DELETION
# ============================================================================

def run_mutations() -> None:
    """Test inserting/deleting common words at various positions in the DoI."""
    print("\n" + "=" * 70)
    print("SECTION 4: WORD INSERTION/DELETION")
    print("=" * 70)

    words = list(BEALE_DOI)
    baseline_decoded = decode_with_wordlist(B1, words)
    baseline_gq = measure_runs(baseline_decoded)
    print(f"\nBaseline: longest_run={baseline_gq['longest_run']}, "
          f"total={baseline_gq['total_run_length']}, score={baseline_gq['score']:.1f}")

    # Test inserting common words at positions 1-500
    print("\n--- INSERTIONS (common words at positions 1-500) ---")
    print(f"  {'Word':>10} {'Pos':>5} {'Longest':>8} {'Total':>6} {'Score':>7} {'Delta':>7}")
    print("  " + "-" * 50)

    improvements: list[dict] = []
    for insert_word in COMMON_WORDS[:5]:
        for pos in range(1, min(501, len(words))):
            modified = words[:pos] + [insert_word] + words[pos:]
            decoded = decode_with_wordlist(B1, modified)
            gq = measure_runs(decoded)
            delta = gq["score"] - baseline_gq["score"]
            if delta > 0.5:
                improvements.append({
                    "action": "insert",
                    "word": insert_word,
                    "pos": pos,
                    "longest": gq["longest_run"],
                    "total": gq["total_run_length"],
                    "score": gq["score"],
                    "delta": delta,
                })

    # Sort by score and show top 20
    improvements.sort(key=lambda x: -x["score"])
    for imp in improvements[:20]:
        print(f"  {imp['word']:>10} {imp['pos']:>5} {imp['longest']:>8} "
              f"{imp['total']:>6} {imp['score']:>7.1f} {imp['delta']:>+7.1f}")

    if not improvements:
        print("  No single insertion improves the score by >0.5")

    # Test deletions at positions 1-500
    print("\n--- DELETIONS (positions 1-500) ---")
    print(f"  {'Pos':>5} {'Deleted':>15} {'Longest':>8} {'Total':>6} {'Score':>7} {'Delta':>7}")
    print("  " + "-" * 55)

    del_improvements: list[dict] = []
    for pos in range(min(500, len(words))):
        modified = words[:pos] + words[pos + 1:]
        decoded = decode_with_wordlist(B1, modified)
        gq = measure_runs(decoded)
        delta = gq["score"] - baseline_gq["score"]
        if delta > 0.5:
            del_improvements.append({
                "action": "delete",
                "word": words[pos],
                "pos": pos,
                "longest": gq["longest_run"],
                "total": gq["total_run_length"],
                "score": gq["score"],
                "delta": delta,
            })

    del_improvements.sort(key=lambda x: -x["score"])
    for imp in del_improvements[:20]:
        print(f"  {imp['pos']:>5} {imp['word']:>15} {imp['longest']:>8} "
              f"{imp['total']:>6} {imp['score']:>7.1f} {imp['delta']:>+7.1f}")

    if not del_improvements:
        print("  No single deletion improves the score by >0.5")

    return improvements + del_improvements


# ============================================================================
# 5. HILL-CLIMBING OPTIMIZER
# ============================================================================

def run_optimizer(n_iterations: int = 10000, seed: int = 42) -> None:
    """
    Targeted mutations near Gillogly run break points.

    Strategy: mutations only matter if they shift the word at a cipher number
    used in or near a Gillogly run. We collect cipher numbers from positions
    around each run, then target insertions/deletions/swaps just before those
    word positions in the DoI.
    """
    print("\n" + "=" * 70)
    print(f"SECTION 5: HILL-CLIMBING OPTIMIZER ({n_iterations} iterations)")
    print("=" * 70)

    rng = np.random.default_rng(seed)

    # Collect all cipher numbers in/near the three main runs (± 3 positions)
    target_cnums: set[int] = set()
    for run_info in MAIN_RUNS:
        for pos in range(max(0, run_info["start"] - 3),
                         min(len(B1), run_info["end"] + 5)):
            target_cnums.add(B1[pos])
    # These are the DoI word positions that matter — mutations before them shift decode
    target_positions = sorted(target_cnums)
    print(f"\nTarget cipher numbers (near runs): {len(target_positions)}")
    print(f"  Range: {min(target_positions)}-{max(target_positions)}")

    words = list(BEALE_DOI)
    # Pad word list to cover max(B1) if needed (with dummy words that won't help)
    while len(words) < max(B1):
        words.append("the")

    best_words = words[:]
    decoded = decode_with_wordlist(B1, best_words)
    best_gq = measure_runs(decoded)
    best_score = best_gq["score"]

    print(f"Baseline: longest={best_gq['longest_run']}, total={best_gq['total_run_length']}, "
          f"score={best_score:.1f}")

    accepts = 0
    t0 = time.time()

    for i in range(n_iterations):
        candidate = best_words[:]

        # Pick a target position near one of the Gillogly-relevant cipher numbers
        target = int(rng.choice(target_positions))
        # Mutate near this position (±5)
        pos = max(0, target + int(rng.integers(-5, 6)))
        pos = min(pos, len(candidate) - 1)

        mutation = rng.choice(["insert", "delete", "swap", "replace"],
                              p=[0.3, 0.2, 0.2, 0.3])

        if mutation == "insert":
            word = rng.choice(COMMON_WORDS)
            candidate.insert(pos, word)
        elif mutation == "delete":
            candidate.pop(pos)
        elif mutation == "swap" and pos < len(candidate) - 1:
            candidate[pos], candidate[pos + 1] = candidate[pos + 1], candidate[pos]
        elif mutation == "replace":
            candidate[pos] = str(rng.choice(COMMON_WORDS))
        else:
            continue

        decoded = decode_with_wordlist(B1, candidate)
        gq = measure_runs(decoded)
        score = gq["score"]

        if score > best_score:
            best_words = candidate
            best_score = score
            best_gq = gq
            accepts += 1

            if accepts <= 30 or accepts % 10 == 0:
                print(f"  [{i:>6}] ACCEPT: longest={gq['longest_run']}, "
                      f"total={gq['total_run_length']}, score={score:.1f} "
                      f"({mutation} near pos {pos})")

        if (i + 1) % 2000 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1:>6}] Progress: {accepts} accepts, "
                  f"best_score={best_score:.1f}, {(i+1)/elapsed:.0f} iter/s")

    elapsed = time.time() - t0
    print(f"\nOptimizer complete: {n_iterations} iterations, {accepts} accepts, "
          f"{elapsed:.1f}s")
    print(f"Final: longest={best_gq['longest_run']}, total={best_gq['total_run_length']}, "
          f"score={best_score:.1f}")

    # Show what changed
    orig = list(BEALE_DOI)
    while len(orig) < max(B1):
        orig.append("the")
    print(f"\nWord list length: {len(orig)} → {len(best_words)}")
    diffs = 0
    for j in range(min(len(orig), len(best_words))):
        if orig[j] != best_words[j]:
            if diffs < 20:
                print(f"  pos {j}: '{orig[j]}' → '{best_words[j]}'")
            diffs += 1
    if diffs >= 20:
        print(f"  ... ({diffs} total differences)")

    # Show the resulting Gillogly runs
    decoded = decode_with_wordlist(B1, best_words)
    gq = measure_runs(decoded, min_run=5)
    print(f"\nResulting runs (>= 5):")
    for run in gq["runs"]:
        print(f"  pos {run['start']}-{run['end']}: '{run['letters']}' (len {run['length']})")


# ============================================================================
# 6. MULTI-RUN COHERENCE TEST
# ============================================================================

def run_coherence() -> None:
    """Test whether ONE variant can improve all three Gillogly runs simultaneously."""
    print("\n" + "=" * 70)
    print("SECTION 6: MULTI-RUN COHERENCE TEST")
    print("=" * 70)

    words = list(BEALE_DOI)
    baseline_decoded = decode_with_wordlist(B1, words)
    baseline_gq = measure_runs(baseline_decoded)

    print(f"\nBaseline runs:")
    for run in baseline_gq["runs"]:
        print(f"  pos {run['start']}-{run['end']}: '{run['letters']}' (len {run['length']})")

    # For each run, find the best single mutations to extend it
    # Then check if the same mutation helps or hurts other runs
    print("\n--- Per-run best mutations ---")

    run_bests: dict[str, list[dict]] = {}
    for run_info in MAIN_RUNS:
        name = run_info["name"]
        run_bests[name] = []

        # Try insertions near the run break
        break_pos = run_info["end"] + 1
        # The cipher number at the break determines which DoI position matters
        if break_pos >= len(B1):
            continue

        break_cipher = B1[break_pos]
        # The DoI words near this cipher number are what matter
        # Inserting/deleting words before break_cipher shifts the mapping

        best_muts: list[dict] = []

        # Test insertions at positions 1 to break_cipher+5
        search_range = min(break_cipher + 5, len(words))
        for pos in range(max(1, break_cipher - 50), search_range):
            for insert_word in COMMON_WORDS[:3]:
                modified = words[:pos] + [insert_word] + words[pos:]
                decoded = decode_with_wordlist(B1, modified)

                # Measure each run individually
                run_letters = decoded[run_info["start"]:run_info["end"] + 3]
                run_len = 1
                for j in range(1, len(run_letters)):
                    if run_letters[j] >= run_letters[j - 1]:
                        run_len += 1
                    else:
                        break

                if run_len > run_info["expected_len"]:
                    gq = measure_runs(decoded)
                    best_muts.append({
                        "action": f"insert '{insert_word}' at {pos}",
                        "pos": pos,
                        "run_extension": run_len - run_info["expected_len"],
                        "total_score": gq["score"],
                        "all_runs": [(r["start"], r["length"]) for r in gq["runs"]],
                    })

        run_bests[name] = sorted(best_muts, key=lambda x: -x["total_score"])[:5]

    # Report
    for name, bests in run_bests.items():
        print(f"\n  {name}:")
        if not bests:
            print(f"    No single mutation extends this run")
            continue
        for b in bests[:3]:
            print(f"    {b['action']}: extends by {b['run_extension']}, "
                  f"total_score={b['total_score']:.1f}")
            for rs, rl in b["all_runs"]:
                print(f"      run at {rs}: len {rl}")

    # Coherence check: do the best mutations for each run agree?
    print("\n--- COHERENCE ANALYSIS ---")
    all_actions = set()
    for name, bests in run_bests.items():
        if bests:
            for b in bests[:1]:
                all_actions.add(b["action"])

    if not all_actions:
        print("  No mutations improve any run. Consistent with hoax artifacts.")
    elif len(all_actions) == 1:
        print(f"  ALL runs improved by same mutation: {all_actions.pop()}")
        print("  STRONG genuine signal — single variant fixes everything!")
    else:
        print(f"  Different runs need different mutations:")
        for name, bests in run_bests.items():
            if bests:
                print(f"    {name}: {bests[0]['action']}")
            else:
                print(f"    {name}: no fix found")

        # Check if any single mutation improves ALL runs
        print("\n  Testing if any mutation improves all three runs simultaneously...")
        baseline_runs = {
            r["name"]: r["expected_len"] for r in MAIN_RUNS
        }

        found_coherent = False
        for name, bests in run_bests.items():
            for b in bests[:3]:
                # Check this mutation against all runs
                all_better = True
                for run_info in MAIN_RUNS:
                    run_improved = False
                    for rs, rl in b["all_runs"]:
                        if abs(rs - run_info["start"]) < 3 and rl >= run_info["expected_len"]:
                            run_improved = True
                            break
                    if not run_improved:
                        all_better = False
                        break
                if all_better:
                    print(f"  FOUND: '{b['action']}' improves or maintains all runs!")
                    found_coherent = True

        if not found_coherent:
            print("  No single mutation improves all three runs.")
            print("  This is consistent with the Gillogly strings being hoax artifacts")
            print("  (each independently constructed, not derived from a single variant).")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DoI variant optimization for Beale B1 Gillogly strings"
    )
    parser.add_argument("--constraints", action="store_true",
                        help="Section 1: constraint mapping around run breaks")
    parser.add_argument("--offsets", action="store_true",
                        help="Section 2: global offset testing (±10)")
    parser.add_argument("--mutations", action="store_true",
                        help="Sections 3-4: greedy + word insert/delete")
    parser.add_argument("--optimize", action="store_true",
                        help="Section 5: hill-climbing optimizer")
    parser.add_argument("--coherence", action="store_true",
                        help="Section 6: multi-run coherence test")
    parser.add_argument("--all", action="store_true",
                        help="Run everything")
    parser.add_argument("--iterations", type=int, default=10000,
                        help="Optimizer iterations (default: 10000)")
    args = parser.parse_args()

    run_everything = args.all or not (
        args.constraints or args.offsets or args.mutations
        or args.optimize or args.coherence
    )

    if run_everything or args.constraints:
        run_constraints()

    if run_everything or args.offsets:
        run_offsets()

    if run_everything or args.mutations:
        run_greedy()
        run_mutations()

    if run_everything or args.optimize:
        run_optimizer(args.iterations)

    if run_everything or args.coherence:
        run_coherence()


if __name__ == "__main__":
    main()
