"""
---
version: 0.1.0
created: 2026-02-24
updated: 2026-02-24
---

phase4_corpus.py — Corpus-based key text search for B1 and B3.

Downloads and tests texts from Project Gutenberg as candidate key texts
for B1 and B3. Scores each text by the bigram quality of the decoded output.

Design:
  - Incremental: tracks which texts have been tested in a JSON state file
  - Resumable: can be interrupted and restarted
  - Tests any text available pre-1885 (pamphlet publication date)
  - If Beale = Ward hoax, any text he could access is fair game

Usage:
    python3 phase4_corpus.py [--max-texts N] [--state-file FILE] [--no-download]
    python3 phase4_corpus.py --results            # show results from previous runs
    python3 phase4_corpus.py --sweep [--sweep-start N] [--sweep-end N]  # brute-force Gutenberg
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from beale import (
    B1, B2, B3, BEALE_DOI, SPECIAL_DECODE,
    decode_book_cipher, first_letter,
    text_to_words, load_gutenberg_text,
    bigram_score, index_of_coincidence,
    BIGRAM_LOGPROB,
)


# ============================================================================
# CANDIDATE TEXT CATALOG
# ============================================================================

# Gutenberg IDs of candidate key texts.
# If Ward published the pamphlet in 1885, any text available by then is fair game.
# Organized by category, covering founding documents through Victorian literature.
CANDIDATE_TEXTS: list[dict] = [
    # ---- Founding & political documents ----
    {"id": 5, "title": "US Bill of Rights", "year": 1789},
    {"id": 18, "title": "US Constitution", "year": 1787},
    {"id": 1404, "title": "Federalist Papers", "year": 1788},
    {"id": 2009, "title": "Inaugural Addresses of Presidents", "year": 1789},
    {"id": 3755, "title": "Common Sense (Paine)", "year": 1776},
    {"id": 31270, "title": "The Age of Reason (Paine)", "year": 1794},
    {"id": 3743, "title": "Rights of Man (Paine)", "year": 1791},
    {"id": 7370, "title": "Second Treatise of Government (Locke)", "year": 1689},
    {"id": 10000, "title": "Magna Carta", "year": 1215},
    {"id": 16895, "title": "Virginia Statute for Religious Freedom (Jefferson)", "year": 1786},

    # ---- Bible & religious ----
    {"id": 10, "title": "KJV Bible", "year": 1611},
    {"id": 17, "title": "Song of Solomon (KJV)", "year": 1611},
    {"id": 8001, "title": "Pilgrim's Progress (Bunyan)", "year": 1678},
    {"id": 22, "title": "Paradise Lost (Milton)", "year": 1667},
    {"id": 8106, "title": "Paradise Regained (Milton)", "year": 1671},

    # ---- Shakespeare ----
    {"id": 1524, "title": "Hamlet", "year": 1603},
    {"id": 1533, "title": "Macbeth", "year": 1606},
    {"id": 1513, "title": "Romeo and Juliet", "year": 1597},
    {"id": 1519, "title": "The Tempest", "year": 1611},
    {"id": 1526, "title": "Julius Caesar", "year": 1599},
    {"id": 1532, "title": "King Lear", "year": 1606},
    {"id": 1531, "title": "Merchant of Venice", "year": 1600},
    {"id": 1514, "title": "A Midsummer Night's Dream", "year": 1596},
    {"id": 1041, "title": "Shakespeare's Sonnets", "year": 1609},
    {"id": 1521, "title": "Henry V", "year": 1599},
    {"id": 1522, "title": "Othello", "year": 1604},

    # ---- 18th century / Enlightenment ----
    {"id": 20203, "title": "Autobiography of Benjamin Franklin", "year": 1791},
    {"id": 4705, "title": "Letters from an American Farmer (Crevecoeur)", "year": 1782},
    {"id": 3207, "title": "Wealth of Nations (Adam Smith)", "year": 1776},
    {"id": 1232, "title": "The Prince (Machiavelli)", "year": 1532},
    {"id": 2680, "title": "Meditations (Marcus Aurelius)", "year": 180},
    {"id": 1228, "title": "Aesop's Fables", "year": -600},
    {"id": 4965, "title": "Leviathan (Hobbes)", "year": 1651},
    {"id": 25717, "title": "Emile (Rousseau)", "year": 1762},

    # ---- Early American lit (1790-1830, Beale-contemporary) ----
    {"id": 9345, "title": "Last of the Mohicans (Cooper)", "year": 1826},
    {"id": 2186, "title": "The Pioneers (Cooper)", "year": 1823},
    {"id": 1945, "title": "The Spy (Cooper)", "year": 1821},
    {"id": 6249, "title": "Rip Van Winkle (Irving)", "year": 1819},
    {"id": 2048, "title": "The Sketch Book (Irving)", "year": 1819},

    # ---- Early-mid 19th century British ----
    {"id": 84, "title": "Frankenstein (Shelley)", "year": 1818},
    {"id": 1342, "title": "Pride and Prejudice (Austen)", "year": 1813},
    {"id": 158, "title": "Emma (Austen)", "year": 1815},
    {"id": 161, "title": "Sense and Sensibility (Austen)", "year": 1811},
    {"id": 768, "title": "Wuthering Heights (Bronte)", "year": 1847},
    {"id": 1260, "title": "Jane Eyre (Bronte)", "year": 1847},
    {"id": 46, "title": "A Christmas Carol (Dickens)", "year": 1843},
    {"id": 98, "title": "A Tale of Two Cities (Dickens)", "year": 1859},
    {"id": 1400, "title": "Great Expectations (Dickens)", "year": 1861},
    {"id": 730, "title": "Oliver Twist (Dickens)", "year": 1838},
    {"id": 766, "title": "David Copperfield (Dickens)", "year": 1850},
    {"id": 580, "title": "The Pickwick Papers (Dickens)", "year": 1837},
    {"id": 19337, "title": "Bleak House (Dickens)", "year": 1853},

    # ---- Mid 19th century American (pre-1885) ----
    {"id": 2701, "title": "Moby-Dick (Melville)", "year": 1851},
    {"id": 15, "title": "Bartleby the Scrivener (Melville)", "year": 1853},
    {"id": 205, "title": "Walden (Thoreau)", "year": 1854},
    {"id": 71, "title": "Leaves of Grass (Whitman)", "year": 1855},
    {"id": 1322, "title": "The Scarlet Letter (Hawthorne)", "year": 1850},
    {"id": 7688, "title": "The House of the Seven Gables (Hawthorne)", "year": 1851},
    {"id": 76, "title": "Adventures of Huckleberry Finn (Twain)", "year": 1884},
    {"id": 74, "title": "Adventures of Tom Sawyer (Twain)", "year": 1876},
    {"id": 245, "title": "The Prince and the Pauper (Twain)", "year": 1881},
    {"id": 2852, "title": "The Gilded Age (Twain & Warner)", "year": 1873},

    # ---- Poe (Virginia-connected, treasure themes!) ----
    {"id": 2147, "title": "The Gold Bug (Poe)", "year": 1843},
    {"id": 2148, "title": "The Fall of the House of Usher (Poe)", "year": 1839},
    {"id": 2149, "title": "The Murders in the Rue Morgue (Poe)", "year": 1841},
    {"id": 932, "title": "Tales of Edgar Allan Poe", "year": 1845},

    # ---- Slavery, abolition, Civil War era ----
    {"id": 23, "title": "Narrative of Frederick Douglass", "year": 1845},
    {"id": 203, "title": "Uncle Tom's Cabin (Stowe)", "year": 1852},

    # ---- European classics available in English by 1885 ----
    {"id": 996, "title": "Don Quixote (Cervantes)", "year": 1605},
    {"id": 1661, "title": "Adventures of Sherlock Holmes (Doyle)", "year": 1892},
    {"id": 345, "title": "Dracula (Stoker)", "year": 1897},
    {"id": 135, "title": "Les Miserables (Hugo)", "year": 1862},
    {"id": 2600, "title": "War and Peace (Tolstoy)", "year": 1869},
    {"id": 1184, "title": "The Count of Monte Cristo (Dumas)", "year": 1844},
    {"id": 1259, "title": "Twenty Thousand Leagues (Verne)", "year": 1870},
    {"id": 2413, "title": "Around the World in 80 Days (Verne)", "year": 1873},

    # ---- Science / natural philosophy ----
    {"id": 2009, "title": "Origin of Species (Darwin)", "year": 1859},
    {"id": 4217, "title": "A Portrait of the Artist as a Young Man (control, post-1885)", "year": 1916},

    # ---- Virginia & Bedford County connections ----
    {"id": 7849, "title": "Notes on the State of Virginia (Jefferson)", "year": 1785},

    # ---- Post-1885 controls ----
    {"id": 4300, "title": "Ulysses (Joyce, control)", "year": 1922},
]

# Deduplicate by ID
_seen_ids = set()
_deduped = []
for t in CANDIDATE_TEXTS:
    if t["id"] not in _seen_ids:
        _seen_ids.add(t["id"])
        _deduped.append(t)
CANDIDATE_TEXTS = _deduped


# ============================================================================
# DOWNLOAD AND SCORING
# ============================================================================

def download_gutenberg(text_id: int, cache_dir: Path) -> Path | None:
    """Download a Gutenberg text, caching locally. Tries multiple URL patterns."""
    cache_file = cache_dir / f"pg{text_id}.txt"
    if cache_file.exists():
        return cache_file

    # Try plain text URLs in order of reliability
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
            continue

    return None


def score_key_text(
    words: list[str],
    cipher: tuple[int, ...],
    cipher_name: str,
) -> dict:
    """Score a candidate key text against a cipher."""
    max_num = max(cipher)
    if len(words) < max_num:
        in_range_pct = sum(1 for n in cipher if n <= len(words)) / len(cipher)
    else:
        in_range_pct = 1.0

    decoded = decode_book_cipher(cipher, words)
    clean = "".join(c for c in decoded if c in "abcdefghijklmnopqrstuvwxyz")

    bg = bigram_score(clean) if len(clean) >= 2 else -4.0
    ic = index_of_coincidence(clean)

    # Count unknown characters
    unknown = sum(1 for c in decoded if c == '?')

    return {
        "cipher": cipher_name,
        "word_count": len(words),
        "max_cipher_num": max_num,
        "in_range_pct": in_range_pct,
        "bigram_score": bg,
        "ic": ic,
        "unknown_pct": unknown / len(cipher) if cipher else 0,
        "decoded_preview": decoded[:80],
    }


def score_text_segments(
    full_words: list[str],
    cipher: tuple[int, ...],
    cipher_name: str,
    segment_size: int = 2000,
    step: int = 500,
) -> list[dict]:
    """
    Score overlapping segments of a long text as candidate key texts.

    Long texts (like the Bible) might contain the key text as a substring.
    """
    results = []
    max_num = max(cipher)

    for start in range(0, max(1, len(full_words) - segment_size + 1), step):
        segment = full_words[start:start + segment_size]
        if len(segment) < max_num * 0.5:
            continue

        decoded = decode_book_cipher(cipher, segment)
        clean = "".join(c for c in decoded if c in "abcdefghijklmnopqrstuvwxyz")
        bg = bigram_score(clean) if len(clean) >= 2 else -4.0

        if bg > -3.0:  # Only keep promising segments
            results.append({
                "start_word": start,
                "segment_words": len(segment),
                "bigram_score": bg,
                "decoded_preview": decoded[:60],
            })

    return results


# ============================================================================
# ENGLISH DETECTION & SWEEP SUPPORT
# ============================================================================

# Top English words — if a text has many of these in its first 500 words, it's English prose.
_ENGLISH_MARKERS = frozenset([
    "the", "and", "of", "to", "a", "in", "that", "is", "was", "for",
    "it", "with", "as", "his", "on", "be", "at", "by", "this", "had",
    "not", "but", "from", "or", "have", "an", "they", "which", "her",
    "were", "there", "been", "their", "has", "would", "each", "she", "he",
])

# B3 max is 975, B1 max is 2906. Minimum useful word count.
_MIN_WORDS_B3 = 975
_MIN_WORDS_B1 = 2906


def is_english_prose(words: list[str], threshold: float = 0.25) -> bool:
    """Quick check: are enough of the first 500 words common English?"""
    sample = words[:500]
    if len(sample) < 50:
        return False
    hits = sum(1 for w in sample if w in _ENGLISH_MARKERS)
    return hits / len(sample) >= threshold


def extract_gutenberg_title(raw_text: str) -> str:
    """Pull title from Gutenberg boilerplate header."""
    lines = raw_text.split("\n")[:60]
    for line in lines:
        stripped = line.strip()
        low = stripped.lower()
        # "Title: X" format (older files)
        if low.startswith("title:"):
            return stripped[6:].strip()[:80]
    # Newer format: title is first non-empty line after "*** START OF"
    past_start = False
    for line in lines:
        stripped = line.strip()
        if "*** START OF" in stripped.upper():
            past_start = True
            continue
        if past_start and stripped and not stripped.startswith("*"):
            return stripped[:80]
    return "Unknown"


def sweep_gutenberg(
    start_id: int,
    end_id: int,
    cache_dir: Path,
    state: dict,
    state_file: Path,
    rate_limit: float = 0.5,
) -> None:
    """Iterate through Gutenberg IDs, test everything that's English prose."""
    tested_count = 0
    skipped = 0
    hits = 0
    t0 = time.time()

    for text_id in range(start_id, end_id + 1):
        str_id = str(text_id)
        if str_id in state["tested"]:
            continue

        # Rate limit downloads (not cached files)
        cache_file = cache_dir / f"pg{text_id}.txt"
        if not cache_file.exists():
            time.sleep(rate_limit)

        filepath = download_gutenberg(text_id, cache_dir)
        if filepath is None:
            state["tested"][str_id] = {"status": "download_failed"}
            if tested_count % 50 == 0:
                save_state(state, state_file)
            tested_count += 1
            continue

        try:
            full_raw = filepath.read_text(encoding="utf-8", errors="replace")
            raw_text = load_gutenberg_text(filepath)
            words = text_to_words(raw_text)
        except Exception:
            state["tested"][str_id] = {"status": "parse_error"}
            tested_count += 1
            continue

        # Filter: must be English prose with enough words for at least B3
        if not is_english_prose(words):
            state["tested"][str_id] = {"status": "not_english", "words": len(words)}
            skipped += 1
            tested_count += 1
            continue

        if len(words) < _MIN_WORDS_B3:
            state["tested"][str_id] = {"status": "too_short", "words": len(words)}
            skipped += 1
            tested_count += 1
            continue

        title = extract_gutenberg_title(full_raw)

        # Score B3 always (range 1-975, most texts qualify)
        b3_score = score_key_text(words, B3, "B3")

        # Score B1 only if text is long enough (range 1-2906)
        if len(words) >= _MIN_WORDS_B1:
            b1_score = score_key_text(words, B1, "B1")
        else:
            b1_score = {"bigram_score": None, "ic": None, "in_range_pct": 0, "decoded_preview": ""}

        hits += 1
        elapsed = time.time() - t0
        rate = tested_count / elapsed if elapsed > 0 else 0

        flag = ""
        if b3_score["bigram_score"] > -3.1 or (b1_score["bigram_score"] or -4) > -3.1:
            flag = " ***"

        print(f"  [{text_id:>5}] {title[:50]:<50} "
              f"({len(words):>7} words) "
              f"B1={b1_score['bigram_score'] or 0:>7.3f}  "
              f"B3={b3_score['bigram_score']:>7.3f}"
              f"{flag}")

        result = {
            "id": text_id,
            "title": title,
            "year": None,  # Unknown in sweep mode
            "word_count": len(words),
            "b1_bigram": b1_score["bigram_score"],
            "b1_ic": b1_score["ic"],
            "b1_in_range": b1_score["in_range_pct"],
            "b1_preview": b1_score.get("decoded_preview", ""),
            "b3_bigram": b3_score["bigram_score"],
            "b3_ic": b3_score["ic"],
            "b3_in_range": b3_score["in_range_pct"],
            "b3_preview": b3_score["decoded_preview"],
        }
        state["tested"][str_id] = {"status": "ok", "title": title}
        state["results"].append(result)

        # Save every 10 scored texts
        if hits % 10 == 0:
            save_state(state, state_file)

        tested_count += 1

        # Progress every 100
        if tested_count % 100 == 0:
            print(f"  --- Progress: {tested_count} checked, {hits} scored, "
                  f"{skipped} skipped, {rate:.1f}/s, "
                  f"ID {text_id}/{end_id} ---")

    save_state(state, state_file)
    print(f"\nSweep complete: {tested_count} checked, {hits} English texts scored, "
          f"{skipped} skipped")


# ============================================================================
# STATE MANAGEMENT
# ============================================================================

def load_state(state_file: Path) -> dict:
    """Load or initialize state file."""
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {"tested": {}, "results": []}


def save_state(state: dict, state_file: Path) -> None:
    """Save state to file."""
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Corpus search for Beale cipher key texts")
    parser.add_argument("--max-texts", type=int, default=0,
                        help="Max texts to test (0 = all)")
    parser.add_argument("--state-file", type=str, default="corpus_search_state.json",
                        help="State file for incremental runs")
    parser.add_argument("--cache-dir", type=str, default=".gutenberg_cache",
                        help="Directory for downloaded texts")
    parser.add_argument("--no-download", action="store_true",
                        help="Only test already-cached texts")
    parser.add_argument("--results", action="store_true",
                        help="Show results from previous runs")
    parser.add_argument("--segments", action="store_true",
                        help="Also test text segments (slower)")
    parser.add_argument("--sweep", action="store_true",
                        help="Brute-force sweep through Gutenberg IDs")
    parser.add_argument("--sweep-start", type=int, default=1,
                        help="Start ID for sweep (default: 1)")
    parser.add_argument("--sweep-end", type=int, default=10000,
                        help="End ID for sweep (default: 10000)")
    args = parser.parse_args()

    state_file = Path(args.state_file)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(exist_ok=True)

    state = load_state(state_file)

    if args.results:
        _show_results(state)
        return

    if args.sweep:
        print("=" * 70)
        print("GUTENBERG SWEEP: BRUTE-FORCE KEY TEXT SEARCH")
        print(f"Range: {args.sweep_start} - {args.sweep_end}")
        print(f"State file: {state_file}")
        already = sum(1 for k, v in state["tested"].items()
                      if args.sweep_start <= int(k) <= args.sweep_end)
        print(f"Already tested in range: {already}")
        print("=" * 70)

        b2_baseline = score_key_text(list(BEALE_DOI), B2, "B2")
        print(f"Baseline (B2 with DoI): bigram={b2_baseline['bigram_score']:.3f}\n")

        sweep_gutenberg(
            args.sweep_start, args.sweep_end,
            cache_dir, state, state_file,
        )
        _show_results(state)
        return

    print("=" * 70)
    print("CORPUS KEY TEXT SEARCH FOR BEALE CIPHERS")
    print(f"Candidate texts: {len(CANDIDATE_TEXTS)}")
    print(f"State file: {state_file}")
    print(f"Cache dir: {cache_dir}")
    print("=" * 70)

    # Baseline: DoI key for B2
    b2_baseline = score_key_text(list(BEALE_DOI), B2, "B2")
    print(f"\nBaseline (B2 with DoI): bigram={b2_baseline['bigram_score']:.3f}")

    tested_count = 0
    max_texts = args.max_texts if args.max_texts > 0 else len(CANDIDATE_TEXTS)

    for text_info in CANDIDATE_TEXTS:
        text_id = text_info["id"]
        title = text_info["title"]
        str_id = str(text_id)

        if str_id in state["tested"]:
            continue

        if tested_count >= max_texts:
            break

        # Download
        if args.no_download:
            filepath = cache_dir / f"pg{text_id}.txt"
            if not filepath.exists():
                print(f"  [{text_id}] {title}: not cached, skipping")
                continue
        else:
            print(f"  [{text_id}] {title}: downloading...", end=" ", flush=True)
            filepath = download_gutenberg(text_id, cache_dir)
            if filepath is None:
                print("FAILED")
                state["tested"][str_id] = {"status": "download_failed", "title": title}
                save_state(state, state_file)
                continue
            print("OK", end=" ", flush=True)

        # Load and process
        try:
            raw_text = load_gutenberg_text(filepath)
            words = text_to_words(raw_text)
            print(f"({len(words)} words)", end=" ", flush=True)
        except Exception as e:
            print(f"PARSE ERROR: {e}")
            state["tested"][str_id] = {"status": "parse_error", "title": title, "error": str(e)}
            save_state(state, state_file)
            continue

        if len(words) < 100:
            print("too short")
            state["tested"][str_id] = {"status": "too_short", "title": title, "words": len(words)}
            save_state(state, state_file)
            continue

        # Score against B1 and B3
        b1_score = score_key_text(words, B1, "B1")
        b3_score = score_key_text(words, B3, "B3")

        print(f"B1={b1_score['bigram_score']:.3f}  B3={b3_score['bigram_score']:.3f}")

        result = {
            "id": text_id,
            "title": title,
            "year": text_info.get("year"),
            "word_count": len(words),
            "b1_bigram": b1_score["bigram_score"],
            "b1_ic": b1_score["ic"],
            "b1_in_range": b1_score["in_range_pct"],
            "b1_preview": b1_score["decoded_preview"],
            "b3_bigram": b3_score["bigram_score"],
            "b3_ic": b3_score["ic"],
            "b3_in_range": b3_score["in_range_pct"],
            "b3_preview": b3_score["decoded_preview"],
        }

        state["tested"][str_id] = {"status": "ok", "title": title}
        state["results"].append(result)
        save_state(state, state_file)

        # Segment analysis for long texts
        if args.segments and len(words) > 3000:
            print(f"    Scanning segments...", end=" ", flush=True)
            b1_segs = score_text_segments(words, B1, "B1")
            b3_segs = score_text_segments(words, B3, "B3")
            if b1_segs:
                best_b1 = max(b1_segs, key=lambda x: x["bigram_score"])
                print(f"B1 best segment: start={best_b1['start_word']}, "
                      f"bg={best_b1['bigram_score']:.3f}")
            if b3_segs:
                best_b3 = max(b3_segs, key=lambda x: x["bigram_score"])
                print(f"B3 best segment: start={best_b3['start_word']}, "
                      f"bg={best_b3['bigram_score']:.3f}")
            if not b1_segs and not b3_segs:
                print("no promising segments")

        tested_count += 1

    _show_results(state)


def _show_results(state: dict) -> None:
    """Display ranked results."""
    results = state.get("results", [])
    if not results:
        print("\nNo results yet. Run without --results to test texts.")
        return

    print("\n" + "=" * 70)
    print("RESULTS: KEY TEXT CANDIDATES RANKED BY BIGRAM SCORE")
    print("=" * 70)

    # B2 baseline for reference
    b2_baseline = score_key_text(list(BEALE_DOI), B2, "B2")

    for cipher_name, bg_key, ic_key, range_key, preview_key in [
        ("B1", "b1_bigram", "b1_ic", "b1_in_range", "b1_preview"),
        ("B3", "b3_bigram", "b3_ic", "b3_in_range", "b3_preview"),
    ]:
        print(f"\n--- {cipher_name} candidates (sorted by bigram score) ---")
        print(f"  B2 baseline (DoI): bigram={b2_baseline['bigram_score']:.3f}")
        print(f"  Random baseline: bigram ~-4.000\n")

        sorted_results = sorted(results, key=lambda x: -(x.get(bg_key) or -999))
        print(f"  {'Rank':>4} {'ID':>6} {'Year':>6} {'Words':>7} {'Bigram':>8} {'IC':>8} "
              f"{'InRange':>8}  Title")
        print("  " + "-" * 90)

        for rank, r in enumerate(sorted_results[:20], 1):
            year_str = str(r.get("year") or "?")
            in_range = r.get(range_key) or 0
            bg_val = r.get(bg_key)
            ic_val = r.get(ic_key)
            bg_str = f"{bg_val:>8.3f}" if bg_val is not None else "     N/A"
            ic_str = f"{ic_val:>8.4f}" if ic_val is not None else "     N/A"
            print(f"  {rank:>4} {r['id']:>6} {year_str:>6} {r['word_count']:>7} "
                  f"{bg_str} {ic_str} {in_range:>7.1%}  {r['title']}")

    # Show any promising results
    print(f"\n--- VERDICT ({len(results)} texts tested) ---")
    has_b1 = [r for r in results if r.get("b1_bigram") is not None]
    best_b1 = max(has_b1, key=lambda x: x["b1_bigram"]) if has_b1 else None
    best_b3 = max(results, key=lambda x: x.get("b3_bigram") or -999)
    if best_b1:
        print(f"  Best B1 candidate: {best_b1['title']} (bigram={best_b1['b1_bigram']:.3f})")
    else:
        print(f"  Best B1 candidate: none tested")
    print(f"  Best B3 candidate: {best_b3['title']} (bigram={best_b3['b3_bigram']:.3f})")
    print(f"  B2 baseline:       DoI (bigram={b2_baseline['bigram_score']:.3f})")
    print()

    b2_bg = b2_baseline["bigram_score"]
    if best_b1 and best_b1["b1_bigram"] > b2_bg * 0.9:
        print(f"  B1: {best_b1['title']} approaches B2 quality. INVESTIGATE FURTHER.")
    elif best_b1 and best_b1["b1_bigram"] > -3.0:
        print(f"  B1: Some improvement over random, but no strong candidate found.")
    else:
        print(f"  B1: No candidate key text produces English-like output.")

    if best_b3["b3_bigram"] > b2_bg * 0.9:
        print(f"  B3: {best_b3['title']} approaches B2 quality. INVESTIGATE FURTHER.")
    elif best_b3["b3_bigram"] > -3.0:
        print(f"  B3: Some improvement over random, but no strong candidate found.")
    else:
        print(f"  B3: No candidate key text produces English-like output.")


if __name__ == "__main__":
    main()
