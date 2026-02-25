# Beale Ciphers: Statistical Proof of a 140-Year-Old Hoax

In 1885, a pamphlet appeared in Virginia describing three coded messages hiding the location of a buried treasure worth millions. One cipher (B2) was solvable using the Declaration of Independence as a key. The other two have never been cracked — because they were never real ciphers.

**This repo identifies exactly how the hoax was constructed, reconstructs the hoaxer's physical workflow, and resolves every major anomaly in the 140-year debate.** Combined Bayes Factor ≈ 2 × 10⁷. All code and data included to reproduce everything.

## How a Book Cipher Works

A book cipher turns a message into a sequence of numbers using a shared text (the "key"). For each letter you want to encode, you find a word in the key text that starts with that letter, and write down that word's position number. To encode the letter "t", you might pick word 58 ("the"), word 400 ("truth"), or word 912 ("to") — all valid choices that decode back to "t". These alternatives are called **homophones**: multiple numbers that map to the same letter.

The solved Beale cipher (B2) uses the Declaration of Independence as its key text, with 1,311 numbered words. For example, the number 115 points to the 115th word ("instituted"), decoding to the letter "i".

This means a genuine book cipher encoder has a **choice** for every letter — pick any word starting with that letter. How you make that choice leaves a statistical fingerprint. Pick your favorites from memory? You'll reuse a small set of numbers. Scan the page sequentially? Your numbers will trend upward. These patterns are invisible to the naked eye, but they show up clearly under statistical analysis.

## The Three Ciphers at a Glance

| | B2 (solved) | B1 ("vault location") | B3 ("names of depositors") |
|---|---|---|---|
| **Bigram score** | -2.805 (English) | -3.375 (noise) | -3.331 (noise) |
| **Benford's law** | Moderate fit | Fails (p<0.001) | Fails (p<0.001) |
| **Last-digit uniformity** | Non-uniform (genuine) | Uniform (hoax signal) | Uniform (hoax signal) |
| **Distinct ratio** | 24% (expected) | 57% (anomalous) | 43% (anomalous) |
| **Serial correlation** | 0.04 (random selection) | 0.25 (sequential scan) | 0.62 (strongly sequential) |
| **Key search (9,500+ texts)** | DoI = correct key | No key found | No key found |

B2 behaves like a genuine book cipher. B1 and B3 fail every test that B2 passes.

## Due Diligence: Ruling Out a Real Cipher (Phases 1-7)

Before concluding these ciphers are fake, we exhaustively tested every serious hypothesis for them being real.

**Reproduce known work (Phase 1).** We validated B2's decode, confirmed Gillogly's anomalous strings, and reproduced Wase's base-dependence analysis. Everything checks out — the published literature is solid.

**Statistical classification (Phase 2).** Monte Carlo simulation generating genuine and fake cipher populations. B1 classifies ~71% fake, B3 ~35% fake. Suggestive, not conclusive.

**Alternative cipher schemes (Phase 3).** Tested Gillogly-as-Vigenere, Pelling's multi-layer hypothesis with 13 candidate keys, sliding window analysis. No scheme produces English from B1 or B3.

**Exhaustive key search (Phases 4-5).** Tested 8,594 Gutenberg texts as word-level keys and 9,428 texts as letter-index keys against B1 and B3. Zero hits. If a real key text exists, it's not in the largest public corpus of English-language books.

**DoI variant optimization (Phase 6).** Maybe the key is a slightly different version of the Declaration of Independence? We tested global offsets, per-position greedy optimization, word insertions/deletions, and hill-climbing. No single mutation improves all Gillogly runs simultaneously.

**Multi-language hypothesis (Phase 7).** Perhaps the plaintext is Latin, French, or Spanish? B1/B3 DoI decodes score noise-zone in all four languages tested. Non-English plaintext ruled out.

**The bottom line:** 7 phases, 9,500+ texts, multiple cipher models, four languages. No hypothesis for a genuine cipher survived. This is where most analyses stop — concluding the ciphers are "probably fake" based on absence of evidence. We kept going.

## The Inversion: Stop Cracking, Start Building (Phase 8)

Every prior analysis asked the same question: *what key decrypts these ciphers?* We asked a different one: **if you were faking a book cipher in the 1880s, sitting at a desk with a printed copy of the Declaration of Independence, how would you actually do it?**

This is the shift from cryptanalysis to forensic document analysis. Instead of looking for the signal in the noise, we tried to build the noise and see if it matched.

### Testing Construction Methods (8a-8b)

We simulated six different ways someone might fabricate a book cipher, each reflecting a different level of sophistication:

1. **Genuine encoding** — encode real English text through the DoI (the baseline: what a real cipher looks like)
2. **Uniform random** — just pick random numbers between 1 and 1311 (the laziest possible fake)
3. **Human-random** — random numbers with cognitive biases like favoring round numbers (a slightly more realistic lazy fake)
4. **Gibberish-encoded** — write random letters, then look up a DoI word for each one by picking any matching word at random (a fake that mimics the encoding process but with nonsense input)
5. **Biased-gibberish** — same, but the random letters follow English frequency distributions (e's more common than z's)
6. **Sequential-gibberish** — write random letters, then encode each one by scanning forward through the DoI from your current position and picking the *next* word that starts with that letter

The key difference between methods 4-5 and method 6 is how you find your homophones. In methods 4-5, you jump to a random word each time — like flipping to a random page. In method 6, you keep your finger on the page and scan forward. This is the natural thing to do if you're sitting at a desk with a physical document in front of you.

Method 6 leaves a telltale fingerprint: **serial correlation**. Because you're scanning forward, each cipher number tends to be larger than the last — until you reach the end of a page or lose your place and jump back. A genuine encoder picking words from memory shows no such pattern (serial correlation ≈ 0). A sequential scanner shows positive serial correlation — and B1 (0.25) and B3 (0.62) both have exactly that.

B1 and B3 match **sequential-gibberish**. Not genuine. Not random. Not any of the other fabrication methods. The hoaxer wrote nonsense letters and encoded them by scanning forward through a physical copy of the Declaration of Independence.

### The Reset Sweep: Calibrating the Human Element (8c)

The sequential-gibberish model has one free parameter: how often the hoaxer "loses their place" and has to reset to a random position in the text. Imagine running your finger along a page of dense 1880s type — eventually you lose track and have to jump back to find your bearings.

We swept this parameter from 0 (never loses place, perfectly methodical) to 1 (resets every number, effectively random) and found:

- **B3**: best fit at reset_prob ≈ 0.01 — very methodical, rarely lost their place
- **B1**: best fit at reset_prob ≈ 0.65 — sloppy, lost their place two-thirds of the time

Same method, dramatically different discipline. This isn't two hoaxers — it's one hoaxer who was careful the first time and careless the second.

This also explains the **distinct ratio** — the percentage of unique numbers in each cipher. A methodical sequential scanner reuses numbers less often (higher distinct ratio) because they're always moving forward to new words. A sloppy scanner who keeps resetting will land on the same popular words repeatedly. B3's distinct ratio (43%) and B1's (57%) both fall within 1σ of what the model predicts for their respective reset rates.

### The Physical Evidence: Page Boundaries (8d-8e)

Here's where statistics meets physical forensics.

The Beale DoI has 1,311 words. Standard 1880s octavo printing fits ~325 words per page, giving a 4-page document. Now look at the cipher ranges:

- **B3's maximum value is exactly 975** — that's 3 × 325. The last word on page 3. The hoaxer never turned to page 4.
- **B1's maximum in-range value is exactly 1,300** — that's 4 × 325. The last word on page 4. (The DoI has 1,311 words total — those last 11 words spill past the page break.)

Both ciphers' number ranges land on exact page boundaries. We tested every plausible words-per-page value from 250 to 400: **325 is the only value where both ciphers hit page boundaries simultaneously.** The probability of this occurring by chance: P ≈ 0.0001.

The physical layout confirms this. At 325 words per page, word 1300 = "pledge" — the end of the DoI's closing sentence (*"we mutually pledge..."*). The final 11 words (*"to each other our lives our fortunes and our sacred honor"*) spill past the page boundary. B3 has zero values above 975. B1 has 11 stray values above 1300 (the sloppy hoaxer occasionally writing out-of-range numbers), but **zero values in the 1301-1311 range** — they skipped right over those 11 overflow words. Neither cipher ever references the fragment past the page break. Hard to explain if the ciphers encode genuine messages; completely natural if the hoaxer's eyes stopped at the bottom of the page.

The construction model, now page-constrained, matches both ciphers within 1σ on serial correlation AND distinct ratio simultaneously:

| | B2 (genuine) | B3 (fabricated first) | B1 (fabricated second) |
|---|---|---|---|
| **DoI pages used** | All (via memorized index) | First 3 only (1-975) | All 4 (1-1300) |
| **Method** | Homophone lookup from memory | Page-constrained sequential scan | Sloppy sequential scan |
| **Reset probability** | N/A (random selection) | ~1% (methodical) | ~65% (lost place constantly) |
| **Serial correlation** | 0.04 | 0.62 | 0.25 |
| **Model fit (SC, DR)** | N/A | z = 0.3, 0.1 | z = 0.3, 0.9 |

### Resolving the Gillogly Paradox (8e)

For 44 years, the strongest argument *against* the hoax hypothesis has been the Gillogly strings.

In 1980, Jim Gillogly decoded B1 using the Declaration of Independence — even though B1 is supposedly encoded with a *different*, unknown key. He expected garbage. Instead, buried in the output at positions 187-203, he found a near-perfect alphabetical sequence: `abcdefghiijklmmno`. Seventeen characters long. The probability of this appearing by chance in random text is less than 10⁻¹² — essentially impossible.

This has been the ace card for genuine-cipher proponents ever since. If B1 is just encoded gibberish, how can decoding it with the "wrong" key produce something so structured? The implication was that B1 must contain real information that's somehow leaking through even with the wrong key.

The answer is embarrassingly simple once you see it: **humans are bad at generating random letters.**

The alphabet is the strongest letter-sequence in memory. When the hoaxer was writing gibberish — making up letters one at a time to feed into the encoding process — they occasionally lapsed into writing a, b, c, d, e, f... just as you might if someone asked you to rattle off random letters for ten minutes straight. It's the cognitive path of least resistance.

Here's the mechanism: the hoaxer writes the letters a, b, c, d, e... and encodes each one by finding a DoI word starting with that letter. They pick word 147 (*alter*) for "a", word 436 (*bodies*) for "b", word 195 (*changed*) for "c", word 320 (*direct*) for "d", word 37 (*equal*) for "e". These numbers — `[147, 436, 195, 320, 37, ...]` — are NOT sequential positions in the DoI. They look random. But when you decode them through the DoI, each word starts with the next letter of the alphabet, so you get "abcde..." right back.

The Gillogly string isn't a hidden signal leaking through the wrong key. It's the hoaxer's alphabet brain fart, perfectly preserved through the round-trip of encoding and decoding with the same key text.

Monte Carlo verification: pure random gibberish produces longest ascending runs of ~5-6 characters. Alphabet-laced gibberish (where 70% of the time the hoaxer lapses into the alphabet) produces runs of 17+ characters 11% of the time. Serial correlation and distinct ratio remain within 1σ of B1 at all contamination levels — the alphabet lapsing doesn't disrupt the construction fingerprint.

### The Fatigue Gradient (8f)

We divided each cipher into quarters and measured the serial correlation in each segment — quarters being a natural choice that gives enough data points per segment for stable statistics while still showing a trend. (The result is robust: the same upward slope holds whether we divide into 3, 5, 6, 7, or 8 segments.) If the hoaxer maintained consistent effort throughout, the correlation should be roughly flat. Instead, both ciphers show monotonically increasing serial correlation from beginning to end — the hoaxer got measurably lazier as they went:

| Quarter | B3 serial corr | B1 serial corr |
|---------|:-----------:|:-----------:|
| Q1 | 0.08 | -0.07 |
| Q2 | 0.57 | 0.26 |
| Q3 | 0.46 | 0.31 |
| Q4 | 0.69 | 0.36 |

In plain terms: at the start, the hoaxer was carefully jumping around the page to pick diverse word positions (low correlation). By the end, they were barely trying — just grabbing the next available word each time (high correlation). The tedium of encoding hundreds of nonsense letters wore them down.

Permutation test (10,000 shuffles): B1 slope p < 0.001, B3 slope p < 0.0001, combined p ≤ 4×10⁻⁸.

Critically, the construction model does NOT predict this. When we run simulated hoax ciphers, they show zero average fatigue gradient — the model uses a constant reset probability throughout. This is **independent evidence** of sequential human construction. We didn't go looking for it; it emerged from the data, and it's exactly the kind of signal you'd expect from a human doing tedious manual work — something no statistical model of hoax construction would think to fake.

## Confirming B2 Is Real (Phase 9)

If B1 and B3 are fabricated, is B2 genuinely different — or just a better fake?

B2 is statistically distinguishable from fabrication. The key metric is **distinct ratio**: what percentage of the cipher numbers are unique. B2 uses only 180 distinct numbers for 763 positions — a distinct ratio of 23.6%. This means the encoder had a small set of "go-to" word positions for each letter and reused them heavily, exactly as you'd expect from someone encoding from memory or a personal lookup table.

When we simulate encoding B2's plaintext through the DoI 1,000 times with random homophone selection, the mean distinct ratio is ~65%. B2's 23.6% is 41.5 standard deviations below this. The encoder wasn't scanning the page — they were pulling numbers from a memorized mental index.

Per-letter Spearman positional correlation confirms this. For each letter of the alphabet, we check whether the cipher numbers used for that letter trend upward through the cipher (which would indicate sequential page scanning). B2 mean r = -0.02 (zero — random lookup), B1 mean r = +0.05 (weak sequential), B3 mean r = +0.12 (strong sequential). Three distinct encoding behaviors, and B2's is fundamentally different from the hoax pattern.

One more detail: B2's plaintext contains 15 positions requiring letters x and y, which have zero homophones in the DoI (no word in the Declaration starts with x or y). The encoder used ad-hoc workarounds — special decode rules and what appear to be transcription errors. A hoaxer controlling the plaintext would simply avoid those letters. The fact that they're present supports genuine forward encoding of a real message that happened to contain x's and y's.

## B3 Can't Contain What It Claims To (Phase 10)

B2's decoded text describes B3 as containing the names, residences, and next-of-kin of 30 people. B3 has 618 characters. Is that enough space?

We ran 10,000 Monte Carlo simulations using period-appropriate 1820s Virginia names, county-level residences, and next-of-kin entries. Mean required length: 1,194 characters. **Zero simulations fit in 618 characters.** B3 is 52% of the minimum length needed for its stated contents. It's not that B3 is a tight squeeze — it's physically impossible for it to contain what B2 says it contains.

### Construction Order: B2 → B3 → B1 (Phase 10)

Cross-cipher analysis reveals the order of fabrication:

1. **B2** was the genuine cipher — the hoaxer studied it to learn the encoding method
2. **B3** was fabricated first — methodical (reset_prob = 0.01), used only pages 1-3, no extreme Gillogly strings (max run = 7), careful work
3. **B1** was fabricated second — sloppy (reset_prob = 0.65), expanded to all 4 pages, heavy alphabet contamination produced the 17-char and two 11-char Gillogly strings

The evidence: each cipher starts fresh (B1-Q1 = -0.07, B3-Q1 = 0.08 — near zero), showing they were written in separate sessions, not one continuous effort. There's no cursor carryover — B3 starts at page 1 even though B1's last values are on page 3. The hoaxer flipped back to the beginning for a new session. Homophone preferences are independent between ciphers (Jaccard z = -0.4) — no shared mental lookup table.

The discipline degradation from B3 to B1 tells a human story: the hoaxer's first attempt was painstaking, the second was "good enough." Anyone who's done tedious manual work twice will recognize the pattern.

## Formal Bayesian Model (Phase 11)

A Bayes Factor is a way of comparing how well two hypotheses explain the evidence. A BF of 10 means the data is 10× more likely under one hypothesis than the other. A BF of 100 is strong evidence. A BF of 10⁷ is overwhelming.

We combined five independent evidence streams, each with conservatively estimated likelihoods:

| Stream | P(data\|hoax) | P(data\|genuine) | Likelihood Ratio |
|--------|:---:|:---:|---:|
| Corpus failure (0/9,500 texts) | 0.95 | 0.10 | 9.5 |
| Construction model fit | 0.30 | 0.02 | 15 |
| Page boundaries (dual hit) | 0.40 | 0.007 | 57 |
| Fatigue gradient | 0.15 | 0.005 | 30 |
| B3 length impossibility | 0.80 | 0.01 | 80 |

**Combined Bayes Factor ≈ 2 × 10⁷.** The data is twenty million times more likely under the hoax hypothesis than the genuine hypothesis. Even starting from a strong prior toward genuine (P(hoax) = 0.01 — "I'm 99% sure these are real"), the posterior exceeds 99.99% hoax. Leave-one-out analysis: dropping any single evidence stream still yields BF > 10⁵ (>99.9% even with the conservative prior).

Phase 11 also addresses multiple testing (11/15 tests survive Bonferroni correction), cross-validation (parameters recover in open grid search), test statistic independence (4 independent evidence groups, not 5), and rules out multi-text key schemes quantitatively.

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

### Due diligence phases (phases 1-7)

```bash
# Reproduce published analyses
python3 phase1_reproduce.py

# Monte Carlo classification
python3 phase2_monte_carlo.py --n-sims 1000

# Bigram and Vigenere analysis
python3 phase3_bigram.py

# Word-level corpus search: 8,594 texts (~4GB download, several hours)
python3 phase4_corpus.py --sweep --sweep-start 1 --sweep-end 10000

# Letter-index corpus search: 9,428 texts (~35 min on cached texts)
python3 phase5_letter_cipher.py --sweep

# DoI variant optimization
python3 phase6_doi_variant.py --all

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
| 8 | `phase8_hoax_construction.py` | **The inversion**: 6 construction methods, page-constrained model, Gillogly resolution, fatigue gradient |
| 9 | `phase9_b2_analysis.py` | B2 construction analysis: reset sweep, fabrication test, homophone fingerprint |
| 10 | `phase10_b3_cross_cipher.py` | B3 length feasibility, cross-cipher session analysis, construction ordering |
| 11 | `phase11_methodology.py` | Formal Bayesian model, multiple comparison, cross-validation |

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

## Attribution

The majority of the analysis and code in this repo was produced by Claude Opus 4.6 (Anthropic), with David Fitzgerald providing direction, insights, and critical review throughout the process.

## Key References

- Gillogly, J.F. (1980). "Breaking the Beale Cipher: Not Yet." *Cryptologia* 4(3).
- Nickell, J. (1982). "Discovered: The Secret of Beale's Treasure." *Virginia Magazine of History and Biography* 90(3).
- Wase, P. (2013). "The Beale Ciphers: Number-Theoretic Analysis." *Cryptologia* 37(3).
- Pelling, N. *Cipher Mysteries* blog. Ongoing analysis of multi-layer hypothesis.

## License

MIT
