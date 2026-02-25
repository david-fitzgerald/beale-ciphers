# Beale Ciphers: Statistical Proof of a 140-Year-Old Hoax

In 1885, a pamphlet appeared in Virginia describing three coded messages hiding the location of a buried treasure worth millions. One cipher was solvable using the Declaration of Independence as a key. The other two have never been cracked — because they were never real ciphers.

This repo contains a complete computational cryptanalysis: 11 phases of automated testing across 9,500+ texts, Monte Carlo simulation of six construction methods, and a formal Bayesian model. **The result: >99% probability that ciphers B1 and B3 are fabricated, with the specific construction method identified and the hoaxer's physical workflow reconstructed.**

## The Three Ciphers

| | B2 (solved) | B1 ("vault location") | B3 ("names of depositors") |
|---|---|---|---|
| **Bigram score** | -2.805 (English) | -3.375 (noise) | -3.331 (noise) |
| **Benford's law** | Moderate fit | Fails (p<0.001) | Fails (p<0.001) |
| **Last-digit uniformity** | Non-uniform (genuine) | Uniform (hoax signal) | Uniform (hoax signal) |
| **Distinct ratio** | 24% (expected) | 57% (anomalous) | 43% (anomalous) |
| **Serial correlation** | 0.04 (random selection) | 0.25 (sequential scan) | 0.62 (strongly sequential) |
| **Key search (9,500+ texts)** | DoI = correct key | No key found | No key found |

B2 behaves like a genuine book cipher. B1 and B3 fail every test that B2 passes.

## How They Were Built

The hoaxer worked from a physical 4-page octavo printing of the Declaration of Independence (~325 words/page). He wrote gibberish letters — occasionally lapsing into alphabetical sequences — and encoded them by scanning forward through the pages, picking the next word that started with each needed letter.

The smoking gun: **B3's maximum cipher value is exactly 975.** The Beale DoI has 1,311 words. At 325 words per page, word 975 is the last word on page 3 of a 4-page printing. The hoaxer never turned to page 4. B1's maximum in-range value is 1,300 = exactly 4 × 325. Both ciphers' ranges land on exact page boundaries — and 325 is the **only** words-per-page value in the plausible range (250-400) where both numbers hit simultaneously (P ≈ 0.0001).

| | B2 (genuine) | B3 (fabricated first) | B1 (fabricated second) |
|---|---|---|---|
| **DoI pages used** | All (via memorized index) | First 3 only (1-975) | All 4 (1-1300) |
| **Method** | Homophone lookup from memory | Page-constrained sequential scan | Sloppy sequential scan |
| **Reset probability** | N/A (random selection) | ~1% (methodical) | ~65% (lost place constantly) |
| **Serial correlation** | 0.04 | 0.62 | 0.25 |
| **Model fit (SC, DR)** | N/A | z = 0.3, 0.1 | z = 0.3, 0.9 |

Both B1 and B3 match the construction model within 1σ on serial correlation AND distinct ratio simultaneously.

## The Fatigue Gradient

Both ciphers show monotonically increasing serial correlation from start to finish — the hoaxer got lazier as he went:

| Quarter | B3 serial corr | B1 serial corr |
|---------|:-----------:|:-----------:|
| Q1 | 0.08 | -0.07 |
| Q2 | 0.57 | 0.26 |
| Q3 | 0.46 | 0.31 |
| Q4 | 0.69 | 0.36 |

Permutation test (10,000 shuffles): B1 slope p < 0.001, B3 slope p < 0.0001, combined p ≤ 4×10⁻⁸. The construction model does NOT predict this — it's independent evidence of sequential human construction.

## The Gillogly Paradox (Resolved)

B1 contains a 17-character alphabetical string (`abcdefghiijklmmno`) at positions 187-203 when decoded with the DoI. Probability of this by chance: <10⁻¹². This was the strongest argument against the hoax hypothesis — until we found the mechanism.

When humans generate "random" letters, they occasionally lapse into alphabetical sequences — the alphabet is the strongest letter-sequence in memory. These alphabetical segments, encoded through the DoI, produce cipher numbers that decode back as ascending runs.

The cipher numbers at B1[187-203] are `[147, 436, 195, 320, 37, 122, 113, 6, ...]` — NOT sequential positions in the DoI. Each points to a word starting with the next alphabet letter: 147→*alter*, 436→*bodies*, 195→*changed*, 320→*direct*, 37→*equal*. This is exactly what encoding "abcdefghijklmno" through the DoI produces.

Monte Carlo (1,000 sims): pure random gibberish → longest runs ≈ 5-6. Alphabet-laced gibberish (alpha_prob=0.70) → P(≥17) = 11%. SC and DR remain within 1σ of B1 at all alpha levels.

## B2 Is Genuinely Different

B2 is statistically distinguishable from fabrication (phase 9). Its distinct ratio (23.6%) is dramatically lower than random homophone selection produces (~65%, z = -41.5). The encoder reused a small set of memorized DoI positions — only 180 distinct numbers for 763 positions. Per-letter Spearman correlation shows zero positional ordering (random lookup), unlike B1/B3's sequential scanning pattern.

Override patches for letters with no DoI homophones (x, y) support genuine forward encoding — a hoaxer controlling the plaintext would simply avoid those letters.

## B3 Is Structurally Impossible

B3 has 618 characters to encode 30 people's names, residences, and next-of-kin (as described in B2's decoded text). Monte Carlo with period-appropriate 1820s names: mean required length = 1,194 characters. 0 of 10,000 simulations fit. B3 is 52% of the minimum length needed.

## Formal Bayesian Model (Phase 11)

Five independent evidence streams with conservative likelihoods:

| Stream | P(data\|hoax) | P(data\|genuine) | LR |
|--------|:---:|:---:|---:|
| Corpus failure (0/9,500 texts) | 0.95 | 0.10 | 9.5 |
| Construction model fit | 0.30 | 0.02 | 15 |
| Page boundaries (dual hit) | 0.40 | 0.007 | 57 |
| Fatigue gradient | 0.15 | 0.005 | 30 |
| B3 length impossibility | 0.80 | 0.01 | 80 |

Combined Bayes Factor ≈ 2 × 10⁷. Even starting from a strong prior toward genuine (P(hoax) = 0.01), the posterior exceeds 99.99%. Leave-one-out analysis: dropping any single stream still yields BF > 10⁵.

Phase 11 also addresses multiple testing (11/15 tests survive Bonferroni), cross-validation (parameters recover in open grid), test statistic independence (4 independent evidence groups), and rules out multi-text key schemes quantitatively.

**Known residuals:** junction effect residual z ≈ 4; B2's memorization mechanism is descriptive, not mechanistic; specific Gillogly error patterns not modeled at character level.

## Reproduce

```
Python 3.10+
pip install numpy scipy matplotlib
```

```bash
# Quick verification (< 1 min): B2 decode, stats battery
python3 beale.py

# Full hoax reconstruction (phase 8, ~5 min)
python3 phase8_hoax_construction.py --all --n-sims 1000

# B2 construction analysis (phase 9, ~2 min)
python3 phase9_b2_analysis.py --all --n-sims 1000

# B3 length + cross-cipher analysis (phase 10, ~1 min)
python3 phase10_b3_cross_cipher.py --all --n-sims 1000

# Methodological rigor: all 7 sections (phase 11, ~2 min)
python3 phase11_methodology.py --all --n-sims 1000

# Fatigue gradient with high-resolution permutation test
python3 phase8_hoax_construction.py --fatigue-test --n-sims 10000
```

### Corpus searches (require Gutenberg downloads)

```bash
# Word-level: 8,594 texts (~4GB download, several hours)
python3 phase4_corpus.py --sweep --sweep-start 1 --sweep-end 10000

# Letter-index: 9,428 texts (~35 min on cached texts)
python3 phase5_letter_cipher.py --sweep

# Multi-language scoring
python3 phase7_multilingual.py --build-tables && python3 phase7_multilingual.py --sweep
```

## Phases

| Phase | Script | What |
|-------|--------|------|
| 1 | `phase1_reproduce.py` | Reproduce published analyses: B2 decode, Gillogly strings, Wase base-dependence |
| 2 | `phase2_monte_carlo.py` | Monte Carlo classification: genuine vs fake cipher populations |
| 3 | `phase3_bigram.py` | Bigram transitions, Gillogly-as-Vigenere, Pelling multi-layer hypothesis |
| 4 | `phase4_corpus.py` | Word-level corpus search: 8,594 Gutenberg texts as candidate keys |
| 5 | `phase5_letter_cipher.py` | Letter-index hypothesis: 9,428 texts as character-level keys |
| 6 | `phase6_doi_variant.py` | DoI variant optimization: can a close variant extend Gillogly strings? |
| 7 | `phase7_multilingual.py` | Multi-language hypothesis: Latin/French/Spanish bigram scoring |
| 8 | `phase8_hoax_construction.py` | Hoax construction method: 6 methods, page-constrained model, Gillogly mechanism, fatigue gradient |
| 9 | `phase9_b2_analysis.py` | B2 construction analysis: reset sweep, fabrication test, homophone fingerprint |
| 10 | `phase10_b3_cross_cipher.py` | B3 length feasibility, cross-cipher session analysis, construction ordering |
| 11 | `phase11_methodology.py` | Methodological rigor: Bayesian model, multiple comparison, cross-validation |

## File Structure

```
beale.py                     # Shared module: ciphers, codecs, stats, scoring
beale_doi_wordlist.py        # Beale-variant DoI as 1311-word tuple
phase1_reproduce.py          # Reproduce published analyses
phase2_monte_carlo.py        # Monte Carlo classification
phase3_bigram.py             # Bigram analysis
phase4_corpus.py             # Word-level corpus search
phase5_letter_cipher.py      # Letter-index hypothesis
phase6_doi_variant.py        # DoI variant optimization
phase7_multilingual.py       # Multi-language hypothesis
phase8_hoax_construction.py  # Hoax construction method identification
phase9_b2_analysis.py        # B2 construction method analysis
phase10_b3_cross_cipher.py   # B3 length + cross-cipher analysis
phase11_methodology.py       # Methodological rigor response
```

## Key References

- Gillogly, J.F. (1980). "Breaking the Beale Cipher: Not Yet." *Cryptologia* 4(3).
- Nickell, J. (1982). "Discovered: The Secret of Beale's Treasure." *Virginia Magazine of History and Biography* 90(3).
- Wase, P. (2013). "The Beale Ciphers: Number-Theoretic Analysis." *Cryptologia* 37(3).
- Pelling, N. *Cipher Mysteries* blog. Ongoing analysis of multi-layer hypothesis.

## License

MIT
