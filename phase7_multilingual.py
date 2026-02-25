"""
---
version: 0.1.0
created: 2026-02-25
updated: 2026-02-25
---

phase7_multilingual.py — Multi-language hypothesis for B1/B3.

The cipher author proved competence by constructing B2. If B1/B3 plaintext
is Latin, French, or Spanish rather than English, our English-only bigram
scoring would miss a correct key. An educated early-1800s Virginian would
know Latin (classical education) and likely French (elite second language).

Sections:
  1. Build bigram tables — download/cache reference texts, compute tables
  2. Calibration — score known texts in each language against all tables
  3. DoI test — score B1/B3 DoI decodes against all 4 language tables
  4. Corpus rescan — re-score top hits from phases 4/5 against all languages
  5. Targeted sweep — full corpus sweep with multi-language scoring

Usage:
    python3 phase7_multilingual.py --build-tables     # section 1 (downloads refs)
    python3 phase7_multilingual.py --calibrate        # section 2
    python3 phase7_multilingual.py --doi-test         # section 3
    python3 phase7_multilingual.py --rescan           # section 4
    python3 phase7_multilingual.py --sweep            # section 5
    python3 phase7_multilingual.py --results          # show sweep results
"""

from __future__ import annotations

import argparse
import json
import string
import time
import urllib.request
from pathlib import Path

import numpy as np

from beale import (
    B1, B2, B3, BEALE_DOI, SPECIAL_DECODE, BIGRAM_FLOOR, BIGRAM_LOGPROB,
    decode_book_cipher, bigram_score, index_of_coincidence,
    build_bigram_table, score_with_bigram_table,
    text_to_words, load_gutenberg_text, text_to_alpha,
    decode_letter_cipher, score_letter_cipher,
)


# ============================================================================
# REFERENCE TEXTS — one large corpus per language
# ============================================================================

CACHE_DIR = Path(".gutenberg_cache")
TABLE_FILE = Path("bigram_tables.json")
STATE_FILE = Path("multilingual_search_state.json")

# Reference texts for building bigram tables.
# Multiple per language for robustness. Need 50K+ chars each.
REFERENCE_TEXTS: dict[str, list[dict]] = {
    "latin": [
        {"id": 23306, "title": "Meditationes de prima philosophia (Descartes)"},
        {"id": 1009, "title": "Divina Commedia (Dante) — Italian, Latin-adjacent"},
    ],
    "french": [
        {"id": 4650, "title": "Candide (Voltaire)"},
        {"id": 8712, "title": "La Conquête de Plassans (Zola)"},
        {"id": 13951, "title": "Les Trois Mousquetaires (Dumas)"},
        {"id": 17989, "title": "Le Comte de Monte-Cristo (Dumas)"},
    ],
    "spanish": [
        {"id": 2000, "title": "Don Quijote (Cervantes)"},
        {"id": 15532, "title": "Don Quijote Part 2 (Cervantes)"},
    ],
}

# Known IC values for calibration reference
LANGUAGE_IC: dict[str, float] = {
    "english": 0.0667,
    "french": 0.0778,
    "spanish": 0.0775,
    "latin": 0.0770,
    "random": 0.0385,
}


# ============================================================================
# 1. BUILD BIGRAM TABLES
# ============================================================================

def _download(text_id: int) -> Path | None:
    """Download a Gutenberg text if not cached."""
    cache_file = CACHE_DIR / f"pg{text_id}.txt"
    if cache_file.exists() and cache_file.stat().st_size > 100:
        return cache_file
    urls = [
        f"https://www.gutenberg.org/cache/epub/{text_id}/pg{text_id}.txt",
        f"https://www.gutenberg.org/files/{text_id}/{text_id}-0.txt",
        f"https://www.gutenberg.org/files/{text_id}/{text_id}.txt",
    ]
    for url in urls:
        try:
            urllib.request.urlretrieve(url, str(cache_file))
            if cache_file.stat().st_size > 100:
                return cache_file
            cache_file.unlink(missing_ok=True)
        except Exception:
            cache_file.unlink(missing_ok=True)
    return None


def build_tables() -> dict[str, dict[str, float]]:
    """Download reference texts and build bigram tables for each language."""
    print("=" * 70)
    print("SECTION 1: BUILD LANGUAGE BIGRAM TABLES")
    print("=" * 70)

    CACHE_DIR.mkdir(exist_ok=True)
    tables: dict[str, dict[str, float]] = {}

    # English table from hardcoded data (already have it)
    tables["english"] = dict(BIGRAM_LOGPROB)
    print(f"\n  english: using hardcoded table (676 bigrams)")

    for lang, refs in REFERENCE_TEXTS.items():
        print(f"\n  {lang}:")
        combined_text = ""
        for ref in refs:
            print(f"    [{ref['id']}] {ref['title']}...", end=" ", flush=True)
            filepath = _download(ref["id"])
            if filepath is None:
                print("DOWNLOAD FAILED")
                continue
            try:
                raw = load_gutenberg_text(filepath)
                alpha = text_to_alpha(raw)
                combined_text += raw
                print(f"OK ({len(alpha):,} alpha chars)")
            except Exception as e:
                print(f"PARSE ERROR: {e}")

        if len(text_to_alpha(combined_text)) < 10000:
            print(f"    WARNING: only {len(text_to_alpha(combined_text))} chars — table may be unreliable")

        table = build_bigram_table(combined_text)
        tables[lang] = table
        alpha_len = len(text_to_alpha(combined_text))
        print(f"    Built table from {alpha_len:,} total alpha chars")

        # Show top 10 bigrams
        top = sorted(table.items(), key=lambda x: -x[1])[:10]
        print(f"    Top bigrams: {', '.join(f'{bg}={p:.2f}' for bg, p in top)}")

    # Save tables
    with open(TABLE_FILE, "w") as f:
        json.dump(tables, f, indent=2)
    print(f"\nSaved tables to {TABLE_FILE}")

    return tables


def load_tables() -> dict[str, dict[str, float]]:
    """Load saved bigram tables, or build if missing."""
    if TABLE_FILE.exists():
        with open(TABLE_FILE) as f:
            return json.load(f)
    return build_tables()


# ============================================================================
# 2. CALIBRATION
# ============================================================================

def run_calibration(tables: dict[str, dict[str, float]]) -> None:
    """Score known texts in each language against all tables to establish baselines."""
    print("\n" + "=" * 70)
    print("SECTION 2: CALIBRATION — CROSS-LANGUAGE SCORING BASELINES")
    print("=" * 70)

    # Calibration samples: score each reference text against all language tables
    samples: list[dict] = []

    # English samples
    doi_text = " ".join(BEALE_DOI)
    samples.append({"name": "DoI (English)", "lang": "english", "text": doi_text})

    # B2 known decode
    b2_decoded = decode_book_cipher(B2, BEALE_DOI, SPECIAL_DECODE)
    samples.append({"name": "B2 decoded (English)", "lang": "english", "text": b2_decoded})

    # Load reference texts for other languages
    for lang, refs in REFERENCE_TEXTS.items():
        for ref in refs:
            filepath = CACHE_DIR / f"pg{ref['id']}.txt"
            if filepath.exists():
                try:
                    raw = load_gutenberg_text(filepath)
                    # Take first 5000 alpha chars for speed
                    alpha = text_to_alpha(raw)[:5000]
                    samples.append({
                        "name": f"{ref['title'][:30]} ({lang})",
                        "lang": lang,
                        "text": alpha,
                    })
                except Exception:
                    pass

    # Random baseline
    rng = np.random.default_rng(42)
    random_text = "".join(rng.choice(list(string.ascii_lowercase), size=1000))
    samples.append({"name": "Random 26-uniform", "lang": "random", "text": random_text})

    # Score each sample against each language table
    langs = list(tables.keys())
    print(f"\n{'Sample':<35} {'IC':>7} ", end="")
    for lang in langs:
        print(f" {lang:>10}", end="")
    print("  best_match")
    print("-" * (50 + 11 * len(langs)))

    for s in samples:
        ic = index_of_coincidence(s["text"])
        scores: dict[str, float] = {}
        for lang in langs:
            scores[lang] = score_with_bigram_table(s["text"], tables[lang])

        best = max(scores, key=lambda k: scores[k])
        marker = " <<<" if best == s.get("lang") else " !!!" if s.get("lang") != "random" else ""

        print(f"  {s['name']:<33} {ic:>6.4f} ", end="")
        for lang in langs:
            print(f" {scores[lang]:>10.3f}", end="")
        print(f"  {best}{marker}")

    print("\n  <<< = correct match, !!! = wrong match")
    print("  Each text should score highest against its own language table.")


# ============================================================================
# 3. DOI TEST — Score B1/B3 decodes against all languages
# ============================================================================

def run_doi_test(tables: dict[str, dict[str, float]]) -> None:
    """Score B1/B3 decoded with DoI against all language tables."""
    print("\n" + "=" * 70)
    print("SECTION 3: DOI DECODE — MULTI-LANGUAGE SCORING")
    print("=" * 70)

    doi_words = list(BEALE_DOI)
    doi_text = " ".join(BEALE_DOI)
    langs = list(tables.keys())

    # Word-level decode
    print("\n--- Word-level decode (standard book cipher) ---")
    print(f"\n{'Cipher':<8} {'IC':>7} ", end="")
    for lang in langs:
        print(f" {lang:>10}", end="")
    print("  best")
    print("-" * (20 + 11 * len(langs)))

    for name, cipher, special in [("B1", B1, None), ("B2", B2, SPECIAL_DECODE), ("B3", B3, None)]:
        decoded = decode_book_cipher(cipher, doi_words, special)
        ic = index_of_coincidence(decoded)
        scores: dict[str, float] = {}
        for lang in langs:
            scores[lang] = score_with_bigram_table(decoded, tables[lang])
        best = max(scores, key=lambda k: scores[k])
        print(f"  {name:<6} {ic:>6.4f} ", end="")
        for lang in langs:
            print(f" {scores[lang]:>10.3f}", end="")
        print(f"  {best}")

    # Letter-level decode
    print("\n--- Letter-level decode (character index cipher) ---")
    print(f"\n{'Cipher':<8} {'IC':>7} ", end="")
    for lang in langs:
        print(f" {lang:>10}", end="")
    print("  best")
    print("-" * (20 + 11 * len(langs)))

    for name, cipher in [("B1", B1), ("B3", B3)]:
        decoded = decode_letter_cipher(cipher, doi_text)
        ic = index_of_coincidence(decoded)
        scores: dict[str, float] = {}
        for lang in langs:
            scores[lang] = score_with_bigram_table(decoded, tables[lang])
        best = max(scores, key=lambda k: scores[k])
        print(f"  {name:<6} {ic:>6.4f} ", end="")
        for lang in langs:
            print(f" {scores[lang]:>10.3f}", end="")
        print(f"  {best}")

    print("\n  B2 baseline should score highest on 'english'.")
    print("  If B1/B3 score highest on a non-English language, investigate further.")


# ============================================================================
# 4. CORPUS RESCAN — Re-score top hits from phases 4/5
# ============================================================================

def run_rescan(tables: dict[str, dict[str, float]]) -> None:
    """Re-score top corpus hits from phases 4 and 5 against all languages."""
    print("\n" + "=" * 70)
    print("SECTION 4: CORPUS RESCAN — TOP HITS FROM PHASES 4/5")
    print("=" * 70)

    langs = list(tables.keys())
    doi_words = list(BEALE_DOI)

    # Load phase 4 state (word-level)
    p4_state = Path("corpus_search_state.json")
    # Load phase 5 state (letter-level)
    p5_state = Path("letter_search_state.json")

    for state_file, phase_name, decode_mode in [
        (p4_state, "Phase 4 (word-level)", "word"),
        (p5_state, "Phase 5 (letter-level)", "letter"),
    ]:
        if not state_file.exists():
            print(f"\n  {phase_name}: state file not found, skipping")
            continue

        with open(state_file) as f:
            state = json.load(f)

        results = state.get("results", [])
        if not results:
            print(f"\n  {phase_name}: no results")
            continue

        print(f"\n--- {phase_name}: re-scoring top 30 by English bigram ---")

        # Get top 30 by B3 bigram (most have valid B3)
        valid = [r for r in results if r.get("b3_bigram") is not None]
        top = sorted(valid, key=lambda x: -x["b3_bigram"])[:30]

        print(f"\n  {'File/ID':<20} ", end="")
        for lang in langs:
            print(f" {lang:>8}", end="")
        print(f" {'best':>8}  {'eng_rank':>8}")
        print("  " + "-" * (24 + 9 * len(langs) + 20))

        for r in top:
            # Find the cached file
            if "file" in r:
                filepath = CACHE_DIR / r["file"]
                label = r["file"][:18]
            elif "id" in r:
                filepath = CACHE_DIR / f"pg{r['id']}.txt"
                label = f"pg{r['id']}"
            else:
                continue

            if not filepath.exists():
                continue

            try:
                raw = load_gutenberg_text(filepath)
                words = text_to_words(raw)
            except Exception:
                continue

            # Decode B3 with this key
            if decode_mode == "word":
                decoded = decode_book_cipher(B3, words)
            else:
                decoded = decode_letter_cipher(B3, raw)

            clean = "".join(c for c in decoded if c in string.ascii_lowercase)
            scores: dict[str, float] = {}
            for lang in langs:
                scores[lang] = score_with_bigram_table(clean, tables[lang])

            best = max(scores, key=lambda k: scores[k])
            # Rank of English among languages
            ranked = sorted(scores, key=lambda k: -scores[k])
            eng_rank = ranked.index("english") + 1 if "english" in ranked else "?"

            marker = " <<<" if best != "english" else ""
            print(f"  {label:<20} ", end="")
            for lang in langs:
                print(f" {scores[lang]:>8.3f}", end="")
            print(f" {best:>8}  {eng_rank:>8}{marker}")


# ============================================================================
# 5. TARGETED SWEEP — Multi-language corpus scoring
# ============================================================================

def run_sweep(tables: dict[str, dict[str, float]]) -> None:
    """Sweep all cached Gutenberg texts, scoring B3 against all language tables."""
    print("\n" + "=" * 70)
    print("SECTION 5: MULTI-LANGUAGE CORPUS SWEEP")
    print("=" * 70)

    if not CACHE_DIR.exists():
        print("No .gutenberg_cache/ found. Run phase4 --sweep first.")
        return

    cached_files = sorted(CACHE_DIR.glob("pg*.txt"))
    print(f"Cached texts: {len(cached_files)}")

    state = _load_state(STATE_FILE)
    already = len(state["tested"])
    print(f"Already tested: {already}")

    langs = list(tables.keys())
    b3_max = max(B3)
    b1_max = max(B1)

    tested = 0
    scored = 0
    hits = 0  # non-English best scores
    t0 = time.time()

    for filepath in cached_files:
        fname = filepath.name
        if fname in state["tested"]:
            continue

        try:
            raw = load_gutenberg_text(filepath)
            words = text_to_words(raw)
        except Exception:
            state["tested"][fname] = {"status": "parse_error"}
            tested += 1
            continue

        if len(words) < b3_max:
            state["tested"][fname] = {"status": "too_short", "words": len(words)}
            tested += 1
            continue

        # Decode B3 with word-level
        decoded = decode_book_cipher(B3, words)
        clean = "".join(c for c in decoded if c in string.ascii_lowercase)

        # Score against all languages
        scores: dict[str, float] = {}
        for lang in langs:
            scores[lang] = score_with_bigram_table(clean, tables[lang])

        best_lang = max(scores, key=lambda k: scores[k])
        best_score = scores[best_lang]

        # Also score B1 if long enough
        b1_scores: dict[str, float] | None = None
        b1_best_lang = None
        if len(words) >= b1_max:
            b1_decoded = decode_book_cipher(B1, words)
            b1_clean = "".join(c for c in b1_decoded if c in string.ascii_lowercase)
            b1_scores = {}
            for lang in langs:
                b1_scores[lang] = score_with_bigram_table(b1_clean, tables[lang])
            b1_best_lang = max(b1_scores, key=lambda k: b1_scores[k])

        scored += 1
        tested += 1

        # Flag any text where a non-English language scores best AND above noise
        is_hit = (best_lang != "english" and best_score > -3.2) or (
            b1_best_lang is not None and b1_best_lang != "english"
            and b1_scores[b1_best_lang] > -3.2
        )
        if is_hit:
            hits += 1
            print(f"  [{fname:<14}] B3 best={best_lang}({best_score:.3f}) "
                  f"B1 best={b1_best_lang or 'N/A'}"
                  f"({b1_scores[b1_best_lang]:.3f if b1_scores and b1_best_lang else 0:.3f}) <<<")

        result = {
            "file": fname,
            "words": len(words),
            "b3_scores": scores,
            "b3_best_lang": best_lang,
            "b1_scores": b1_scores,
            "b1_best_lang": b1_best_lang,
        }
        state["tested"][fname] = {"status": "ok"}
        state["results"].append(result)

        if scored % 500 == 0:
            elapsed = time.time() - t0
            rate = tested / elapsed if elapsed > 0 else 0
            print(f"  --- Progress: {scored} scored, {hits} non-English hits, "
                  f"{rate:.0f}/s ---")
            _save_state(state, STATE_FILE)

    _save_state(state, STATE_FILE)
    elapsed = time.time() - t0
    print(f"\nSweep complete: {scored} scored, {hits} non-English hits, {elapsed:.1f}s")
    _show_results(state, tables)


# ============================================================================
# STATE AND RESULTS
# ============================================================================

def _load_state(state_file: Path) -> dict:
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {"tested": {}, "results": []}


def _save_state(state: dict, state_file: Path) -> None:
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def _show_results(state: dict, tables: dict[str, dict[str, float]] | None = None) -> None:
    """Display ranked multi-language results."""
    results = state.get("results", [])
    if not results:
        print("\nNo results yet.")
        return

    langs = list(tables.keys()) if tables else ["english", "latin", "french", "spanish"]

    print(f"\n--- TOP 20 BY EACH LANGUAGE (B3) ---")
    print(f"  Total scored: {len(results)}")

    for lang in langs:
        valid = [r for r in results if r.get("b3_scores", {}).get(lang) is not None]
        if not valid:
            continue
        top = sorted(valid, key=lambda x: -x["b3_scores"][lang])[:10]
        print(f"\n  Best B3 matches for {lang.upper()}:")
        print(f"  {'Rank':>4} {'File':<16} {'Score':>8} {'Best lang':>10} {'Eng score':>10}")
        print("  " + "-" * 54)
        for rank, r in enumerate(top[:10], 1):
            eng = r["b3_scores"].get("english", 0)
            print(f"  {rank:>4} {r['file']:<16} {r['b3_scores'][lang]:>8.3f} "
                  f"{r['b3_best_lang']:>10} {eng:>10.3f}")

    # Non-English hits
    non_eng = [r for r in results if r.get("b3_best_lang") and r["b3_best_lang"] != "english"]
    if non_eng:
        print(f"\n--- NON-ENGLISH BEST MATCHES ({len(non_eng)} total) ---")
        by_score = sorted(non_eng, key=lambda x: -max(x["b3_scores"].values()))
        for r in by_score[:20]:
            best = r["b3_best_lang"]
            print(f"  {r['file']:<16} best={best}({r['b3_scores'][best]:.3f}) "
                  f"eng={r['b3_scores'].get('english', 0):.3f}")
    else:
        print(f"\n  No texts where non-English language scored higher than English.")


def show_results() -> None:
    """Load and display results."""
    state = _load_state(STATE_FILE)
    tables = load_tables() if TABLE_FILE.exists() else None
    _show_results(state, tables)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-language hypothesis for Beale B1/B3"
    )
    parser.add_argument("--build-tables", action="store_true",
                        help="Section 1: download refs and build bigram tables")
    parser.add_argument("--calibrate", action="store_true",
                        help="Section 2: cross-language scoring calibration")
    parser.add_argument("--doi-test", action="store_true",
                        help="Section 3: score DoI decodes against all languages")
    parser.add_argument("--rescan", action="store_true",
                        help="Section 4: re-score top phase 4/5 hits")
    parser.add_argument("--sweep", action="store_true",
                        help="Section 5: full corpus sweep with multi-language scoring")
    parser.add_argument("--results", action="store_true",
                        help="Show sweep results")
    args = parser.parse_args()

    run_all = not (args.build_tables or args.calibrate or args.doi_test
                   or args.rescan or args.sweep or args.results)

    if args.results:
        show_results()
        return

    # Always need tables
    if args.build_tables or run_all:
        tables = build_tables()
    else:
        tables = load_tables()

    if run_all or args.calibrate:
        run_calibration(tables)

    if run_all or args.doi_test:
        run_doi_test(tables)

    if run_all or args.rescan:
        run_rescan(tables)

    if args.sweep:
        run_sweep(tables)


if __name__ == "__main__":
    main()
