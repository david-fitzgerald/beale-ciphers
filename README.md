# Beale Ciphers: Computational Cryptanalysis

Statistical and computational analysis of the three Beale ciphers (~1820, published 1885). Twelve phases of automated testing across 9,500+ texts, multiple encoding hypotheses, multi-language scoring, and hoax construction method identification via Monte Carlo simulation.

## Key Findings

**Verdict: 91-96% probability of hoax, with specific construction method identified and all anomalies explained.**

The three ciphers have fundamentally different statistical fingerprints. Cipher #2 (solved, keyed to the Declaration of Independence) behaves like a genuine book cipher. Ciphers #1 and #3 fail every test that #2 passes:

| Test | B2 (solved) | B1 (location) | B3 (names) |
|------|:-----------:|:-------------:|:----------:|
| Bigram score | -2.805 (English) | -3.375 (noise) | -3.331 (noise) |
| Benford's law | Moderate fit | Fails (p<0.001) | Fails (p<0.001) |
| Last-digit base-7 | Non-uniform (genuine) | Uniform (hoax) | Uniform (hoax) |
| Last-digit base-3 | Non-uniform (genuine) | Uniform (hoax) | Uniform (hoax) |
| Distinct ratio | 24% (expected) | 57% (anomalous) | 43% (anomalous) |
| Serial correlation | 0.04 (random) | 0.25 (sequential) | 0.62 (sequential) |
| Word-level key search (8,594 texts) | DoI = correct key | No key found | No key found |
| Letter-index key search (9,428 texts) | N/A | No key found | No key found |
| Multi-language (Latin/French/Spanish) | N/A | Noise in all 4 | Noise in all 4 |

### How They Were Built

Phase 8 identifies the specific construction method through Monte Carlo simulation. The hoaxer (likely James B. Ward, who published the pamphlet in 1885) wrote gibberish letters — occasionally lapsing into alphabetical sequences — and encoded them by scanning forward through a physical printing of the Declaration of Independence.

The key evidence: **B3's maximum cipher value is exactly 975.** The Beale DoI has 1,311 words. At 325 words per page — standard for an 1880s octavo book — word 975 falls exactly at the end of page 3 of a 4-page printing. Ward never turned to page 4.

| | B2 (genuine) | B3 (hoax) | B1 (hoax) |
|---|---|---|---|
| **DoI pages used** | All (via index) | First 3 only (1-975) | All 4 (1-1311) |
| **Method** | Pre-built homophone index | Page-constrained sequential scan | Sloppy sequential scan |
| **Reset probability** | N/A (random selection) | ~1% (methodical) | ~65% (impatient) |
| **Serial correlation** | 0.04 (random) | 0.62 (strongly sequential) | 0.25 (weakly sequential) |
| **Model fit** | N/A | SC z=0.3, DR z=0.1 | SC z=0.3, DR z=0.9 |
| **Estimated time** | 5-6 hours | ~75-90 min | ~45-60 min |

Both B1 and B3 match the model within 1 standard deviation on serial correlation AND distinct ratio simultaneously.

### Fatigue Gradient

Both ciphers show increasing serial correlation from start to finish — the hoaxer got lazier as he went:

| Quarter | B3 serial corr | B1 serial corr |
|---------|:-----------:|:-----------:|
| Q1 | 0.08 | -0.07 |
| Q2 | 0.57 | 0.26 |
| Q3 | 0.46 | 0.31 |
| Q4 | 0.69 | 0.36 |

**Phase 8f formalizes this with a permutation test.** Under H0 (no positional structure), we shuffle each cipher's numbers 10,000 times and recompute the Q1→Q4 SC slope. B1 slope p < 0.001, B3 slope p < 0.0001, combined p ≤ 4×10⁻⁸. The gradient is robust across 3-8 segment partitions (all p < 0.005). Crucially, the phase 8d construction model does NOT predict this (simulated mean slope ≈ 0, 50% positive) — the fatigue gradient is **independent evidence** of sequential human construction, not an artifact of the encoding mechanism.

This is consistent with a single evening session: B2 built carefully during the day with a prepared index, B3 started in the evening with discipline that faded, B1 banged out last by lamplight.

### The Gillogly Paradox (Resolved)

B1 contains a 17-character alphabetical string (`abcdefghiijklmmno`) at positions 187-203 when decoded with the DoI. The probability of this occurring by chance is less than 10^-12. This was the strongest counter-evidence against the hoax hypothesis — until phase 8e.

**The explanation:** the hoaxer's "gibberish" wasn't truly random. When humans generate arbitrary letters, they occasionally lapse into alphabetical sequences (a, b, c, d, ...) — the alphabet is the strongest letter-sequence in memory. These alphabetical plaintext segments, encoded through the DoI, produce cipher numbers that decode back as ascending runs — i.e., Gillogly strings.

The cipher numbers at B1[187-203] are `[147, 436, 195, 320, 37, 122, 113, 6, ...]` — NOT sequential positions. Each number points to a DoI word starting with the next alphabet letter: 147→*alter*, 436→*bodies*, 195→*changed*, 320→*direct*, 37→*equal*, etc. This is exactly what encoding "abcdefghijklmno" through the DoI produces.

Monte Carlo confirmation (1000 sims):
- Pure random gibberish → longest runs ≈ 5-6. P(≥17) = 0.
- Alphabet-laced gibberish (alpha_prob=0.50) → P(≥17) = 0.1%, max observed = 19
- Alphabet-laced gibberish (alpha_prob=0.70) → P(≥17) = 11%
- **SC and DR remain within 1σ of B1 at all alpha levels** — the encoding mechanism is unchanged

Additional DoI variant evidence:
- No close DoI variant (single word insert/delete) extends the run past 17
- Hill-climbing optimizer reached 24 chars but needed 711 word changes (brute-force, not a plausible variant)
- The three main runs (len 11, 11, 17) cannot be improved simultaneously by any single mutation

### Letter-Index Hypothesis (Ruled Out)

The DoI has 6,480 alphabetic characters. B1 max=2906 and B3 max=975 both fit as character indices with zero out-of-range numbers. However:

- Bigram scores are noise (-3.25 each)
- B1/B3 distinct ratios (0.43-0.57) look nothing like synthetic letter-index ciphers (0.96)
- 9,428 texts tested as letter-index keys: best scores -3.07/-3.09, all noise

## Phases

### Elimination (phases 1-7)

| Phase | Script | What it does |
|-------|--------|-------------|
| 1 | `phase1_reproduce.py` | Reproduces published analyses: B2 decode, Gillogly strings, Wase base-dependence, homophone utilization |
| 2 | `phase2_monte_carlo.py` | Monte Carlo classification: genuine vs fake cipher populations, where B1/B3 fall |
| 3 | `phase3_bigram.py` | Bigram transition analysis, Gillogly-as-Vigenere test, Pelling multi-layer hypothesis |
| 4 | `phase4_corpus.py` | Word-level corpus search: 8,594 Gutenberg texts as candidate B1/B3 keys |
| 5 | `phase5_letter_cipher.py` | Letter-index hypothesis: 9,428 texts tested as character-level keys |
| 6 | `phase6_doi_variant.py` | DoI variant optimization: can a close variant extend the Gillogly strings? |
| 7 | `phase7_multilingual.py` | Multi-language hypothesis: Latin/French/Spanish bigram scoring, reversed text/key |

### Hoax reconstruction (phase 8)

| Phase | CLI flag | What it does |
|-------|----------|-------------|
| 8a | `--generate --analyze` | Construction method survey: 6 methods × 1000 sims. B1/B3 classify as gibberish-encoded. |
| 8b | `--generate` | Sequential encoding: hoaxer scanned forward through DoI for homophones. Explains positive serial correlation. |
| 8c | `--reset-sweep` | Reset probability calibration: B1 ≈ 0.65 (sloppy), B3 ≈ 0.01 (methodical). Different discipline per cipher. |
| 8d | `--page-model` | Page-constrained physical model: B3 uses first 3 pages (max=975), B1 uses all 4. Both match SC+DR within 1σ. |
| 8e | `--boundary-test` | Page boundary significance: B3 max=975 at page boundary (p=0.001). B1 max in-range=1300, also a boundary. |
| 8e | `--gillogly-test` | Gillogly artifact analysis: alphabet-contaminated gibberish explains 17-char run. SC/DR preserved at all alpha levels. |
| 8f | `--fatigue-test` | Fatigue gradient permutation test: Q1→Q4 SC slope significant (combined p≤4×10⁻⁸). Independent of model. |

## Reproduce

### Prerequisites

```
Python 3.10+
pip install numpy scipy matplotlib
```

### Quick verification (< 1 min)

```bash
# Verify B2 decode, run stats battery on all three ciphers
python3 beale.py

# Reproduce published analyses with plots
python3 phase1_reproduce.py
```

### Full analysis (< 5 min each)

```bash
# Monte Carlo classification (1000 simulations)
python3 phase2_monte_carlo.py

# Bigram transitions and Vigenere test
python3 phase3_bigram.py

# DoI variant optimization (all sections)
python3 phase6_doi_variant.py --all
```

### Hoax reconstruction (phase 8, ~5 min)

```bash
# Full analysis: all phases 8a-8f
python3 phase8_hoax_construction.py --all --n-sims 1000

# Just the page-constrained model (8d)
python3 phase8_hoax_construction.py --page-model --n-sims 1000

# Page boundary + Gillogly artifact tests (8e)
python3 phase8_hoax_construction.py --boundary-test --gillogly-test --n-sims 1000

# Fatigue gradient permutation test (8f, 10K perms recommended)
python3 phase8_hoax_construction.py --fatigue-test --n-sims 10000
```

### Corpus searches (require Gutenberg downloads)

```bash
# Download and test curated candidate texts (36+ texts)
python3 phase4_corpus.py

# Brute-force sweep through Gutenberg IDs 1-10000
# Downloads ~9,500 texts (~4GB), takes several hours
python3 phase4_corpus.py --sweep --sweep-start 1 --sweep-end 10000

# Letter-index sweep of all cached texts (~35 min)
python3 phase5_letter_cipher.py --sweep

# View results from any previous run
python3 phase4_corpus.py --results
python3 phase5_letter_cipher.py --results
```

### Multi-language analysis (phase 7)

```bash
# Build language bigram tables (downloads Latin/French/Spanish refs)
python3 phase7_multilingual.py --build-tables

# Calibration + DoI test (instant once tables built)
python3 phase7_multilingual.py --calibrate --doi-test

# Re-score top corpus hits from phases 4/5 against all languages
python3 phase7_multilingual.py --rescan

# Full multi-language corpus sweep
python3 phase7_multilingual.py --sweep
```

### Phase 5 and 6 quick modes

```bash
# Letter-index: just DoI test + homophone analysis (instant)
python3 phase5_letter_cipher.py --doi-only

# Letter-index: statistical signature comparison (< 1 min)
python3 phase5_letter_cipher.py --stats

# DoI variant: just constraint mapping and offsets (instant)
python3 phase6_doi_variant.py --constraints --offsets

# DoI variant: hill-climbing optimizer (< 2 sec)
python3 phase6_doi_variant.py --optimize --iterations 10000
```

## File Structure

```
beale.py                    # Shared module: ciphers, codecs, stats, scoring
beale_doi_wordlist.py       # Beale-variant DoI as 1311-word tuple
phase1_reproduce.py         # Reproduce published analyses
phase2_monte_carlo.py       # Monte Carlo classification
phase3_bigram.py            # Bigram analysis
phase4_corpus.py            # Word-level corpus search
phase5_letter_cipher.py     # Letter-index hypothesis
phase6_doi_variant.py       # DoI variant optimization
phase7_multilingual.py      # Multi-language hypothesis
phase8_hoax_construction.py # Hoax construction method identification
```

## Key References

- Gillogly, J.F. (1980). "Breaking the Beale Cipher: Not Yet." *Cryptologia* 4(3).
- Nickell, J. (1982). "Discovered: The Secret of Beale's Treasure." *Virginia Magazine of History and Biography* 90(3).
- Wase, P. (2013). "The Beale Ciphers: Number-Theoretic Analysis." *Cryptologia* 37(3).
- Pelling, N. *Cipher Mysteries* blog. Ongoing analysis of multi-layer hypothesis.

## License

MIT
