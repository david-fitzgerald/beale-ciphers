Endpoint — project complete. No further work; preserved for reference.

# Beale Ciphers — Claude Instructions

Cryptanalytic classification of the three Beale ciphers (~1820, published 1885). P(hoax) >99.99%. Published to HN February 2026.

**Phase:** complete | **Date:** 2026-02

## Serves
- **Primary:** `research/lost-treasure/` — cryptanalytic proof-of-concept (P(hoax) >99.99%)

## Scope

Cryptanalytic research into the three Beale ciphers (~1820, published 1885). Pure code-breaking and statistical classification problem -- not a physical treasure hunt.

## Docker

```bash
~/Dropbox/pg/.docker/run.sh projects/complete/beale-ciphers
```

## Key Files

- `cryptanalysis-research.md` — Comprehensive review of all statistical analysis (Gillogly 1980, Nickell 1982, Benford's law, last-digit analysis, modern ML attempts). Includes cipher properties, anomalies, and proposed definitive tests.
- `beale.py` — Shared module for all cryptanalysis scripts. Seven sections: data constants, word-level codec (book cipher), letter-level codec (letter-index cipher), stats battery (Benford, last-digit, distinct ratio, letter frequency, Gillogly string detection, `gillogly_quality`), scoring (bigram log-prob, IC, composite English-likeness), corpus utils (Gutenberg loading, fake cipher generation), output utils (comparison tables, plots). Run directly for self-test.
- `beale_doi_wordlist.py` — The Beale-variant Declaration of Independence as a 1311-word Python tuple. Eight modifications from the standard DoI text, plus three special decode overrides (word 95 -> 'u', word 811 -> 'y', word 1005 -> 'x').
- `phase1_reproduce.py` — Reproduces published analyses: B2 decode validation, Gillogly strings, full stats battery on all three ciphers, Wase base-dependence analysis, homophone utilization. Generates three comparison plots.
- `phase2_monte_carlo.py` — Monte Carlo classification. Generates genuine (English-frequency encoded with DoI) and fake (uniform random) cipher populations, scores where B1/B3 fall in the distributions. Default 1000 sims; use `--n-sims` to adjust.
- `phase3_bigram.py` — Bigram transition analysis and Gillogly-as-Vigenere test. Tests Pelling's multi-layer hypothesis with 13 candidate keys. Includes sliding window analysis to find promising regions.
- `phase4_corpus.py` — Incremental Gutenberg corpus search. Tests 36+ texts as candidate B1/B3 keys, ranked by bigram quality. State saved to JSON for resumability. Use `--max-texts N` to limit, `--results` to view.
- `phase5_letter_cipher.py` — Letter-index cipher hypothesis. Tests whether cipher numbers index individual characters (not words). Sections: validation, DoI letter-decode, homophone analysis, synthetic cipher statistics, corpus sweep (9,428 texts). Use `--doi-only`, `--stats`, `--sweep`, `--results`.
- `phase6_doi_variant.py` — DoI variant optimization for Gillogly strings. Constraint mapping, global offset testing, per-position greedy, word insertion/deletion, hill-climbing optimizer, multi-run coherence test. Use `--constraints`, `--offsets`, `--mutations`, `--optimize`, `--coherence`, `--all`.
- `phase7_multilingual.py` — Multi-language hypothesis. Builds Latin/French/Spanish bigram tables from Gutenberg reference texts. Scores DoI decodes against all languages. Includes reversed text/key tests. Corpus rescan with multi-language scoring. Use `--build-tables`, `--calibrate`, `--doi-test`, `--rescan`, `--sweep`.
- `phase8_hoax_construction.py` — Hoax construction method analysis. Phases 8a-8f. Tests six construction methods against B1/B3 plus targeted models. Sequential-gibberish (8b) encodes random letters using nearest-forward DoI scan. Reset sweep (8c) finds optimal "lose your place" probability. Page-constrained model (8d) simulates working from a physical 4-page octavo printing: B3 uses only first 3 pages (words 1-975), B1 uses all 4. Phase 8e adds page boundary significance testing and Gillogly artifact analysis: B3 max=975 is statistically significant (p=0.001 vs uniform); Gillogly strings explained as alphabet-contaminated gibberish — hoaxer's "random" letters occasionally fell into alphabetical sequences which encode/decode as ascending runs. Phase 8f: fatigue gradient permutation test — Q1→Q4 serial correlation gradient is statistically significant (B1 p<0.001, B3 p<0.0001, combined p≤4e-8); model does NOT predict this, so it's independent evidence. Use `--generate`, `--analyze`, `--human-tests`, `--reset-sweep`, `--page-model`, `--boundary-test`, `--gillogly-test`, `--fatigue-test`, `--all`.
- `phase9_b2_analysis.py` — B2 construction method analysis. Four tests: (1) reset sweep shows B2 matches reset_prob≈0.95 (random selection, not sequential), (2) fabrication test shows B2 OUTSIDE random-encode distribution (DR z=-41.5, Benford z=-3.3), (3) per-letter homophone fingerprint shows B2 zero positional correlation vs B1/B3 positive, (4) override analysis shows 15 positions need x/y workarounds — ad-hoc patches supporting forward encoding. Key finding: B2's extreme homophone reuse (DR=23.6% vs random ~65%) is diagnostic. Use `--reset-sweep`, `--fabrication-test`, `--homophone-fingerprint`, `--override-analysis`, `--all`.
- `phase10_b3_cross_cipher.py` — B3 length feasibility and cross-cipher session analysis. (1) B3 has 618 chars for 30 people — 0% of MC simulations fit names+addresses+kin (1194 needed). (2) Cross-cipher: fatigue resets between B1→B3 (B1-Q4=0.36, B3-Q1=0.08), no cursor carryover (B3 starts page 1 after B1 ends page 3), junction effect z=-7.1, homophone preferences independent (Jaccard z=-0.4). Use `--b3-length`, `--cross-cipher`, `--all`.
- `doi_printing_research.md` — Research into identifying the specific 4-page octavo DoI printing. Key constraint: must use "inalienable" (not "unalienable"). Best candidate: "The American's Guide" (Hogan & Thompson, 1840, LOC digitized). Next steps: check LOC page images, Shaw-Shoemaker database, Viemeister's book.
- `phase11_methodology.py` — Methodological rigor response to critique.md. Seven sections: (11a) Formal Bayesian model — 5 evidence streams, prior sweep, leave-one-out; BF≈2×10⁷ even with conservative likelihoods. (11b) Multiple comparison correction — 15 p-values with BH FDR and Bonferroni; all key findings survive. (11c) Cross-validation — cross-cipher z>5 (different processes confirmed), half-cipher pass, full parameter grid. (11d) Correlation matrix — SC/DR/Benford correlated, LD10/LD7/Bigram independent; 4 evidence groups not 6. (11e) Page boundary sweep — only wpp=325 gives dual hit at 975 and 1300; P(chance)≈0.0001. (11f) Gillogly likelihood ratios — LR(alpha vs genuine)>100 for alpha≥0.3; SC/DR within 2σ at all levels. (11g) Multi-text key test — multi-text raises DR further from B1/B3, ruling out composite keys. Use `--correlation-matrix`, `--multiple-comparison`, `--cross-validation`, `--page-sweep`, `--gillogly-lr`, `--multi-text`, `--bayesian`, `--all`.

## Generated Artifacts

- `*.png` — Statistical comparison plots (benford, last-digit, homophone, bigram heatmaps, sliding windows, Monte Carlo distributions)
- `corpus_search_state.json` — Incremental state for phase4 word-level corpus search
- `letter_search_state.json` — Incremental state for phase5 letter-index corpus sweep
- `multilingual_search_state.json` — State for phase7 multi-language sweep
- `bigram_tables.json` — Cached Latin/French/Spanish bigram log-probability tables
- `.gutenberg_cache/` — Downloaded Gutenberg texts

## Status — COMPLETE

All 11 phases done. P(hoax) >99.99% (BF ≈ 2×10⁷). Hoax construction method fully reconstructed. HN post published Feb 2026.

Phases 1-7 eliminated all genuine-plaintext hypotheses. Phases 8a-8f reconstructed the exact hoax method. Phase 9 confirmed B2 is statistically distinguishable from fabrication. Phase 10 proved B3 structurally impossible as described and analyzed cross-cipher dependencies. Phase 11 addresses 9 methodological critiques with formal Bayesian model, multiple comparison corrections, cross-validation, and additional quantitative tests.

### Phases 1-7: Elimination

1. **B2 validated** (phase 1): 763-character decode matches known plaintext (19/19 on first 19 chars)
2. **Gillogly confirmed** (phase 1): 17-char alphabetical string at positions 187-203 in B1 (p < 10^-12)
3. **Wase reproduced** (phase 1): B1/B3 last digits uniform in non-base-10 (hoax signal); B2 non-uniform in all bases (genuine signal)
4. **Monte Carlo classification** (phase 2): B1 classifies ~71% fake, B3 ~35% fake
5. **Vigenere test** (phase 3): No Gillogly-derived key produces English from B1's DoI decode
6. **Word-level corpus search** (phase 4): 8,594 Gutenberg texts tested as B1/B3 keys — zero hits
7. **Letter-index hypothesis** (phase 5): 9,428 texts tested as letter-index keys — all noise level
8. **DoI variant optimization** (phase 6): No single mutation improves all three Gillogly runs simultaneously. Consistent with independently constructed hoax artifacts.
9. **Multi-language hypothesis** (phase 7): B1/B3 DoI decodes score noise-zone in English, Latin, French, Spanish. Non-English plaintext with DoI key ruled out.

### Phase 8: Hoax Reconstruction

10. **Construction method survey** (8a): Six methods tested with 1000 sims each — genuine, uniform random, human-random, gibberish-encoded, biased-gibberish, sequential-gibberish. B1/B3 both classify closest to gibberish-encoded (random letters → DoI homophones). Neither genuine nor random match; human-random also ruled out.
11. **Sequential encoding mechanism** (8b): Hoaxer scanned forward through DoI to find homophones rather than picking randomly. Produces positive serial correlation (matching B1's 0.25 and B3's 0.46) that random homophone selection cannot.
12. **Reset probability calibration** (8c): Swept "lose your place" probability. B1 best match: reset_prob=0.65 (sloppy — lost place 2/3 of the time). B3 best match: reset_prob=0.01 (methodical — rarely lost place). Different discipline levels for each cipher.
13. **Page-constrained physical model** (8d): B3's max value is exactly 975 = 3×325 words/page (standard 1880s octavo). Model: the hoaxer worked from a physical 4-page DoI printing. **B3**: first 3 pages only, page-constrained homophone selection. Match: SC z=0.3, DR z=0.1. **B1**: all 4 pages, unconstrained. Match: SC z=0.3, DR z=0.9. Both within 1σ on serial correlation AND distinct ratio simultaneously.
14. **Page boundary significance** (8e): B3 max=975 is statistically significant (p=0.001 under uniform MC, 0th percentile of simulated maxes). B1's max in-range value = 1300 = exactly 4×325, also a page boundary (delta=0). Both ciphers' number ranges align with octavo page boundaries. 98% of B1 numbers fall within DoI range [1, 1311].
15. **Gillogly strings explained** (8e): Pure random gibberish encoded with DoI → longest ascending runs ≈ 5-6 (no Gillogly strings). But alphabet-laced gibberish — hoaxer occasionally writes a,b,c,d,... instead of random letters — produces long runs when decoded. At alpha_prob=0.50: P(≥17)=0.1%. At alpha_prob=0.70: P(≥17)=11%. SC and DR remain within 1σ of B1 at all alpha levels (encoding mechanism unchanged). B1's 17-char run 'abcdefghiijklmmno' at positions 187-203 decodes from cipher numbers [147, 436, 195, 320, 37, ...] — NOT sequential positions, but each pointing to a DoI word starting with the next alphabet letter. This is exactly what encoding "abcdefghijklmno" through the DoI produces.
16. **Fatigue gradient** (8f): Q1→Q4 serial correlation rises monotonically in both ciphers (B1: -0.07→0.36, B3: 0.08→0.69). Permutation test: B1 slope p<0.001, B3 slope p<0.0001, combined p≤4×10⁻⁸. Robust across 3-8 segment partitions. Construction model does NOT predict this (mean slope ≈ 0, 50% positive) — fatigue gradient is independent evidence of sequential human construction.

### Phase 9: B2 Construction Analysis

17. **B2 reset sweep** (9a): SC best match at reset_prob≈0.95 — effectively random homophone selection, not sequential scanning. Confirms B2 used a fundamentally different encoding method from B1 (0.65) and B3 (0.01).
18. **B2 vs fabrication** (9b): Encoding B2's plaintext 1000× with random homophone selection. B2 is OUTSIDE the fabricated distribution on: distinct ratio (z=-41.5), Benford chi2 (z=-3.3), last-digit chi2 (z=+9.1). Only serial correlation is inside (z=+1.4). B2's DR=23.6% vs random ~65% is the key signal.
19. **Homophone fingerprint** (9c): Per-letter Spearman positional correlation. B2 mean r=-0.02 (zero — random selection). B1 mean r=+0.05 (weak positive — sloppy sequential). B3 mean r=+0.12 (stronger — methodical sequential). Gradient B2 < B1 < B3 confirms three distinct encoding behaviors.
20. **Override analysis** (9d): B2 plaintext has 15 positions needing x/y (zero DoI homophones). 3 SPECIAL_DECODE rules cover 4 of these; other 11 are transcription errors. Override presence supports forward encoding — a hoaxer controlling plaintext would simply avoid x/y. The #95→'u' override is redundant (word already starts with 'u').
21. **B2 homophone reuse** (9b finding): B2 uses only 180 distinct numbers for 763 positions. The encoder heavily reused a small set of memorized DoI positions per letter (e.g., 'a' uses 6 of 165 available homophones = 3.6%). This is consistent with genuine encoding from memory or a personal lookup table, not systematic DoI traversal.

### Phase 10: B3 Length & Cross-Cipher

22. **B3 length feasibility** (10a): B3 has 618 characters for 30 people's names + addresses + next-of-kin. MC with period-appropriate 1820s names: mean required = 1194 chars, 0% of 10,000 simulations fit. Names-only (no addresses) fits at 380 chars avg, but B2 explicitly describes addresses and relatives. B3 is **structurally impossible** as described — it's 52% of the minimum length needed.
23. **Cross-cipher session** (10b): Five tests for B1↔B3 dependencies. (a) Number overlap: 162 shared values (40.6% of union) — normal given shared DoI key. (b) No cursor carryover: B1 ends at word 760 (page 3), B3 starts at 317 (page 1) — flipped back to start. (c) Fatigue resets: all three ciphers start fresh (B1 Q1=-0.07, B2 Q1=-0.05, B3 Q1=+0.08) — separate sessions. (d) Junction effect z=-7.1, mostly structural (~80% from different page ranges and SC levels, residual z≈4 from boundary content). (e) Homophone preferences: Jaccard z=-0.4 (normal) — no shared mental lookup table.
24. **Construction ordering** (10b deep analysis): B2→B3→B1 is the best-fit ordering. Evidence: (a) Discipline degrades — B3 methodical (rp=0.01) → B1 sloppy (rp=0.65), natural motivation decay. (b) Gillogly strings — B3 has no extreme runs (max=7), B1 has 17-char and two 11-char runs; alphabet contamination is a second-attempt laziness artifact. (c) Page expansion — B3 uses pages 1-3 only, B1 expands to all 4. (d) Fatigue gaps — B2→B3 drop is tiny (-0.065), B3→B1 drop is massive (-0.756); the hoaxer made B3 soon after studying B2, waited longer for B1. (e) Motivation — B2 explicitly references B3 (names), so B3 needed to exist first; B1 (location) is pure theater. Permutation test: B1→B3 concatenation produces significant fatigue slope (p<0.0001); B3→B1 does not (p=0.72).

## Key Finding

Weight of evidence: **>99% hoax** for B1/B3 (Bayesian model, phase 11a; log₁₀ BF ≈ 7.3 even with conservative likelihoods). All major statistical anomalies can be modeled by the page-constrained sequential-gibberish construction method. B1/B3 fail every word-level, letter-level, multi-language, and reversed-key test. No key text in 9,500+ Gutenberg texts produces language-like output. The construction method: **the hoaxer worked from a 4-page octavo printing of the DoI (~325 words/page), wrote alphabet-contaminated gibberish, and encoded it by scanning forward through the physical pages.** Ward is the most parsimonious candidate for the hoaxer, but the method identification does not identify a person.

**B2 is statistically distinguishable from fabrication** (phase 9). Its extreme homophone reuse (DR=23.6%, z=-41.5 vs random encoding) and zero positional correlation indicate encoding from memory or a personal lookup table — not systematic DoI traversal. Override patches for x/y (no DoI homophones) support genuine forward encoding. B2's construction fingerprint is DIFFERENT from both B1/B3 and from random fabrication, supporting a different author or method.

**Construction order: B2 → B3 → B1** (phase 10). The hoaxer studied B2 (the genuine cipher) to learn the encoding method, then fabricated B3 first and B1 second. Evidence: (a) discipline degrades — B3 methodical (rp=0.01) → B1 sloppy (rp=0.65), natural motivation decay; (b) Gillogly strings — B3 has none (max run=7), B1 has 17-char and two 11-char runs from increasing alphabet contamination; (c) page expansion — B3 uses pages 1-3 only, B1 expands to all 4; (d) B2 explicitly references B3 (names list), so B3 had to exist first; B1 (vault location) is pure theater.

- **B3** (written first): Used only the first 3 pages (words 1-975; max cipher value = 975 exactly). Page-constrained homophone selection. Very methodical (reset_prob ≈ 0.01) — fresh effort, careful execution. Model matches actual SC and DR within 0.3 sigma. No extreme Gillogly strings (max run = 7).
- **B1** (written second): Used all 4 pages (max in-range value = 1300 = page 4 boundary). Much sloppier (reset_prob ≈ 0.65 — lost their place 2/3 of the time). Matches within 1 sigma. Heavy alphabet contamination produced Gillogly strings (17-char, two 11-char runs) — a second-attempt laziness artifact.

This model accounts for all major statistical anomalies: (1) positive serial correlation from sequential scanning, (2) depressed distinct ratio from page-constrained homophone pools, (3) Gillogly strings from alphabet-contaminated gibberish encoded through DoI, (4) B3's 975 ceiling as a physical page boundary (p=0.001), (5) B1's 1300 in-range max as another page boundary, (6) B3→B1 discipline degradation as motivation decay, (7) Q1→Q4 fatigue gradient in both ciphers (independent evidence, p≤4×10⁻⁸ combined), (8) B3 length impossibility as structural evidence (618 chars for content requiring ~1194).

**Known residuals** (not fully explained): junction effect residual z≈4 after accounting for structural differences; B2's memorization mechanism (why only ~180 distinct numbers for 763 positions) is descriptive, not mechanistic; specific Gillogly error patterns (why 'ii' and 'mm' in the 17-char run) not modeled at the character level.

## Remaining Gaps

| Gap | What | Impact | Status |
|-----|------|--------|--------|
| **B2 reverse-engineering** | Phase 9 complete. B2 statistically distinguishable from fabrication (DR z=-41.5). Homophone reuse pattern inconsistent with the B1/B3 hoax method. | Resolved — B2 supports genuine encoding | Done (phase 9) |
| **Cross-cipher session analysis** | Phase 10 complete. Fatigue resets between ciphers; no cursor carryover; junction effect unusual (z=-7.1); homophone preferences independent. Suggests break between B1/B3 or B3 written first. | Modest — no shared-session signal | Done (phase 10) |
| **Historical DoI printing ID** | Find the specific 4-page octavo printing. 325 wpp is a testable prediction. Must use "inalienable" (not "unalienable"). Best lead: "The American's Guide" (1840, LOC digitized). See `doi_printing_research.md`. | High if found — physical corroboration | Open (not pursuing; revisit if interest returns) |
| **Publication** | Analysis is publication-grade for Cryptologia or similar. README restructured for public audience. HN post published Feb 2026. | External validation | Done |

## Dependencies

Python 3.10+, numpy, scipy, matplotlib (for plots).
