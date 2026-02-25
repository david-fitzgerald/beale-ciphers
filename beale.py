"""
---
version: 0.4.0
created: 2026-02-24
updated: 2026-02-25
---

beale.py — Shared module for Beale cipher cryptanalysis.

Seven sections:
  1. Data constants (cipher texts, DoI word list, special decode rules)
  2. Codec — word-level (encode/decode book cipher)
  3. Codec — letter-level (encode/decode letter-index cipher)
  4. Stats battery (Benford, last-digit, distinct ratio, letter freq, Gillogly)
  5. Scoring (English-likeness via bigram/trigram log-probability)
  6. Corpus utils (load/process texts for key-text search)
  7. Output utils (formatting, comparison tables, plots)
"""

from __future__ import annotations

import math
import string
import warnings
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from beale_doi_wordlist import BEALE_DOI

# ============================================================================
# 1. DATA CONSTANTS
# ============================================================================

# Cipher texts: tuples of ints, exactly as published in the 1885 pamphlet.
# Source: FSU datasets / Cipher Foundation transcription, cross-verified.

B1: tuple[int, ...] = (
    71, 194, 38, 1701, 89, 76, 11, 83, 1629, 48, 94, 63, 132, 16, 111, 95,
    84, 341, 975, 14, 40, 64, 27, 81, 139, 213, 63, 90, 1120, 8, 15, 3, 126,
    2018, 40, 74, 758, 485, 604, 230, 436, 664, 582, 150, 251, 284, 308, 231,
    124, 211, 486, 225, 401, 370, 11, 101, 305, 139, 189, 17, 33, 88, 208,
    193, 145, 1, 94, 73, 416, 918, 263, 28, 500, 538, 356, 117, 136, 219, 27,
    176, 130, 10, 460, 25, 485, 18, 436, 65, 84, 200, 283, 118, 320, 138, 36,
    416, 280, 15, 71, 224, 961, 44, 16, 401, 39, 88, 61, 304, 12, 21, 24,
    283, 134, 92, 63, 246, 486, 682, 7, 219, 184, 360, 780, 18, 64, 463, 474,
    131, 160, 79, 73, 440, 95, 18, 64, 581, 34, 69, 128, 367, 460, 17, 81,
    12, 103, 820, 62, 116, 97, 103, 862, 70, 60, 1317, 471, 540, 208, 121,
    890, 346, 36, 150, 59, 568, 614, 13, 120, 63, 219, 812, 2160, 1780, 99,
    35, 18, 21, 136, 872, 15, 28, 170, 88, 4, 30, 44, 112, 18, 147, 436, 195,
    320, 37, 122, 113, 6, 140, 8, 120, 305, 42, 58, 461, 44, 106, 301, 13,
    408, 680, 93, 86, 116, 530, 82, 568, 9, 102, 38, 416, 89, 71, 216, 728,
    965, 818, 2, 38, 121, 195, 14, 326, 148, 234, 18, 55, 131, 234, 361, 824,
    5, 81, 623, 48, 961, 19, 26, 33, 10, 1101, 365, 92, 88, 181, 275, 346,
    201, 206, 86, 36, 219, 324, 829, 840, 64, 326, 19, 48, 122, 85, 216, 284,
    919, 861, 326, 985, 233, 64, 68, 232, 431, 960, 50, 29, 81, 216, 321,
    603, 14, 612, 81, 360, 36, 51, 62, 194, 78, 60, 200, 314, 676, 112, 4,
    28, 18, 61, 136, 247, 819, 921, 1060, 464, 895, 10, 6, 66, 119, 38, 41,
    49, 602, 423, 962, 302, 294, 875, 78, 14, 23, 111, 109, 62, 31, 501, 823,
    216, 280, 34, 24, 150, 1000, 162, 286, 19, 21, 17, 340, 19, 242, 31, 86,
    234, 140, 607, 115, 33, 191, 67, 104, 86, 52, 88, 16, 80, 121, 67, 95,
    122, 216, 548, 96, 11, 201, 77, 364, 218, 65, 667, 890, 236, 154, 211,
    10, 98, 34, 119, 56, 216, 119, 71, 218, 1164, 1496, 1817, 51, 39, 210,
    36, 3, 19, 540, 232, 22, 141, 617, 84, 290, 80, 46, 207, 411, 150, 29,
    38, 46, 172, 85, 194, 39, 261, 543, 897, 624, 18, 212, 416, 127, 931, 19,
    4, 63, 96, 12, 101, 418, 16, 140, 230, 460, 538, 19, 27, 88, 612, 1431,
    90, 716, 275, 74, 83, 11, 426, 89, 72, 84, 1300, 1706, 814, 221, 132, 40,
    102, 34, 868, 975, 1101, 84, 16, 79, 23, 16, 81, 122, 324, 403, 912, 227,
    936, 447, 55, 86, 34, 43, 212, 107, 96, 314, 264, 1065, 323, 428, 601,
    203, 124, 95, 216, 814, 2906, 654, 820, 2, 301, 112, 176, 213, 71, 87,
    96, 202, 35, 10, 2, 41, 17, 84, 221, 736, 820, 214, 11, 60, 760,
)

B2: tuple[int, ...] = (
    115, 73, 24, 807, 37, 52, 49, 17, 31, 62, 647, 22, 7, 15, 140, 47, 29,
    107, 79, 84, 56, 239, 10, 26, 811, 5, 196, 308, 85, 52, 160, 136, 59,
    211, 36, 9, 46, 316, 554, 122, 106, 95, 53, 58, 2, 42, 7, 35, 122, 53,
    31, 82, 77, 250, 196, 56, 96, 118, 71, 140, 287, 28, 353, 37, 1005, 65,
    147, 807, 24, 3, 8, 12, 47, 43, 59, 807, 45, 316, 101, 41, 78, 154, 1005,
    122, 138, 191, 16, 77, 49, 102, 57, 72, 34, 73, 85, 35, 371, 59, 196, 81,
    92, 191, 106, 273, 60, 394, 620, 270, 220, 106, 388, 287, 63, 3, 6, 191,
    122, 43, 234, 400, 106, 290, 314, 47, 48, 81, 96, 26, 115, 92, 158, 191,
    110, 77, 85, 197, 46, 10, 113, 140, 353, 48, 120, 106, 2, 607, 61, 420,
    811, 29, 125, 14, 20, 37, 105, 28, 248, 16, 159, 7, 35, 19, 301, 125,
    110, 486, 287, 98, 117, 511, 62, 51, 220, 37, 113, 140, 807, 138, 540, 8,
    44, 287, 388, 117, 18, 79, 344, 34, 20, 59, 511, 548, 107, 603, 220, 7,
    66, 154, 41, 20, 50, 6, 575, 122, 154, 248, 110, 61, 52, 33, 30, 5, 38,
    8, 14, 84, 57, 540, 217, 115, 71, 29, 84, 63, 43, 131, 29, 138, 47, 73,
    239, 540, 52, 53, 79, 118, 51, 44, 63, 196, 12, 239, 112, 3, 49, 79, 353,
    105, 56, 371, 557, 211, 505, 125, 360, 133, 143, 101, 15, 284, 540, 252,
    14, 205, 140, 344, 26, 811, 138, 115, 48, 73, 34, 205, 316, 607, 63, 220,
    7, 52, 150, 44, 52, 16, 40, 37, 158, 807, 37, 121, 12, 95, 10, 15, 35,
    12, 131, 62, 115, 102, 807, 49, 53, 135, 138, 30, 31, 62, 67, 41, 85, 63,
    10, 106, 807, 138, 8, 113, 20, 32, 33, 37, 353, 287, 140, 47, 85, 50, 37,
    49, 47, 64, 6, 7, 71, 33, 4, 43, 47, 63, 1, 27, 600, 208, 230, 15, 191,
    246, 85, 94, 511, 2, 270, 20, 39, 7, 33, 44, 22, 40, 7, 10, 3, 811, 106,
    44, 486, 230, 353, 211, 200, 31, 10, 38, 140, 297, 61, 603, 320, 302,
    666, 287, 2, 44, 33, 32, 511, 548, 10, 6, 250, 557, 246, 53, 37, 52, 83,
    47, 320, 38, 33, 807, 7, 44, 30, 31, 250, 10, 15, 35, 106, 160, 113, 31,
    102, 406, 230, 540, 320, 29, 66, 33, 101, 807, 138, 301, 316, 353, 320,
    220, 37, 52, 28, 540, 320, 33, 8, 48, 107, 50, 811, 7, 2, 113, 73, 16,
    125, 11, 110, 67, 102, 807, 33, 59, 81, 158, 38, 43, 581, 138, 19, 85,
    400, 38, 43, 77, 14, 27, 8, 47, 138, 63, 140, 44, 35, 22, 177, 106, 250,
    314, 217, 2, 10, 7, 1005, 4, 20, 25, 44, 48, 7, 26, 46, 110, 230, 807,
    191, 34, 112, 147, 44, 110, 121, 125, 96, 41, 51, 50, 140, 56, 47, 152,
    540, 63, 807, 28, 42, 250, 138, 582, 98, 643, 32, 107, 140, 112, 26, 85,
    138, 540, 53, 20, 125, 371, 38, 36, 10, 52, 118, 136, 102, 420, 150, 112,
    71, 14, 20, 7, 24, 18, 12, 807, 37, 67, 110, 62, 33, 21, 95, 220, 511,
    102, 811, 30, 83, 84, 305, 620, 15, 2, 10, 8, 220, 106, 353, 105, 106, 60,
    275, 72, 8, 50, 205, 185, 112, 125, 540, 65, 106, 807, 138, 96, 110, 16,
    73, 33, 807, 150, 409, 400, 50, 154, 285, 96, 106, 316, 270, 205, 101,
    811, 400, 8, 44, 37, 52, 40, 241, 34, 205, 38, 16, 46, 47, 85, 24, 44,
    15, 64, 73, 138, 807, 85, 78, 110, 33, 420, 505, 53, 37, 38, 22, 31, 10,
    110, 106, 101, 140, 15, 38, 3, 5, 44, 7, 98, 287, 135, 150, 96, 33, 84,
    125, 807, 191, 96, 511, 118, 40, 370, 643, 466, 106, 41, 107, 603, 220,
    275, 30, 150, 105, 49, 53, 287, 250, 208, 134, 7, 53, 12, 47, 85, 63,
    138, 110, 21, 112, 140, 485, 486, 505, 14, 73, 84, 575, 1005, 150, 200,
    16, 42, 5, 4, 25, 42, 8, 16, 811, 125, 160, 32, 205, 603, 807, 81, 96,
    405, 41, 600, 136, 14, 20, 28, 26, 353, 302, 246, 8, 131, 160, 140, 84,
    440, 42, 16, 811, 40, 67, 101, 102, 194, 138, 205, 51, 63, 241, 540, 122,
    8, 10, 63, 140, 47, 48, 140, 288,
)

B3: tuple[int, ...] = (
    317, 8, 92, 73, 112, 89, 67, 318, 28, 96, 107, 41, 631, 78, 146, 397,
    118, 98, 114, 246, 348, 116, 74, 88, 12, 65, 32, 14, 81, 19, 76, 121,
    216, 85, 33, 66, 15, 108, 68, 77, 43, 24, 122, 96, 117, 36, 211, 301, 15,
    44, 11, 46, 89, 18, 136, 68, 317, 28, 90, 82, 304, 71, 43, 221, 198, 176,
    310, 319, 81, 99, 264, 380, 56, 37, 319, 2, 44, 53, 28, 44, 75, 98, 102,
    37, 85, 107, 117, 64, 88, 136, 48, 154, 99, 175, 89, 315, 326, 78, 96,
    214, 218, 311, 43, 89, 51, 90, 75, 128, 96, 33, 28, 103, 84, 65, 26, 41,
    246, 84, 270, 98, 116, 32, 59, 74, 66, 69, 240, 15, 8, 121, 20, 77, 89,
    31, 11, 106, 81, 191, 224, 328, 18, 75, 52, 82, 117, 201, 39, 23, 217,
    27, 21, 84, 35, 54, 109, 128, 49, 77, 88, 1, 81, 217, 64, 55, 83, 116,
    251, 269, 311, 96, 54, 32, 120, 18, 132, 102, 219, 211, 84, 150, 219,
    275, 312, 64, 10, 106, 87, 75, 47, 21, 29, 37, 81, 44, 18, 126, 115, 132,
    160, 181, 203, 76, 81, 299, 314, 337, 351, 96, 11, 28, 97, 318, 238, 106,
    24, 93, 3, 19, 17, 26, 60, 73, 88, 14, 126, 138, 234, 286, 297, 321, 365,
    264, 19, 22, 84, 56, 107, 98, 123, 111, 214, 136, 7, 33, 45, 40, 13, 28,
    46, 42, 107, 196, 227, 344, 198, 203, 247, 116, 19, 8, 212, 230, 31, 6,
    328, 65, 48, 52, 59, 41, 122, 33, 117, 11, 18, 25, 71, 36, 45, 83, 76,
    89, 92, 31, 65, 70, 83, 96, 27, 33, 44, 50, 61, 24, 112, 136, 149, 176,
    180, 194, 143, 171, 205, 296, 87, 12, 44, 51, 89, 98, 34, 41, 208, 173,
    66, 9, 35, 16, 95, 8, 113, 175, 90, 56, 203, 19, 177, 183, 206, 157, 200,
    218, 260, 291, 305, 618, 951, 320, 18, 124, 78, 65, 19, 32, 124, 48, 53,
    57, 84, 96, 207, 244, 66, 82, 119, 71, 11, 86, 77, 213, 54, 82, 316, 245,
    303, 86, 97, 106, 212, 18, 37, 15, 81, 89, 16, 7, 81, 39, 96, 14, 43,
    216, 118, 29, 55, 109, 136, 172, 213, 64, 8, 227, 304, 611, 221, 364, 819,
    375, 128, 296, 1, 18, 53, 76, 10, 15, 23, 19, 71, 84, 120, 134, 66, 73,
    89, 96, 230, 48, 77, 26, 101, 127, 936, 218, 439, 178, 171, 61, 226, 313,
    215, 102, 18, 167, 262, 114, 218, 66, 59, 48, 27, 19, 13, 82, 48, 162,
    119, 34, 127, 139, 34, 128, 129, 74, 63, 120, 11, 54, 61, 73, 92, 180,
    66, 75, 101, 124, 265, 89, 96, 126, 274, 896, 917, 434, 461, 235, 890,
    312, 413, 328, 381, 96, 105, 217, 66, 118, 22, 77, 64, 42, 12, 7, 55, 24,
    83, 67, 97, 109, 121, 135, 181, 203, 219, 228, 256, 21, 34, 77, 319, 374,
    382, 675, 684, 717, 864, 203, 4, 18, 92, 16, 63, 82, 22, 46, 55, 69, 74,
    112, 134, 186, 175, 119, 213, 416, 312, 343, 264, 119, 186, 218, 343,
    417, 845, 951, 124, 209, 49, 617, 856, 924, 936, 72, 19, 28, 11, 35, 42,
    40, 66, 85, 94, 112, 65, 82, 115, 119, 236, 244, 186, 172, 112, 85, 6,
    56, 38, 44, 85, 72, 32, 47, 73, 96, 124, 217, 314, 319, 221, 644, 817,
    821, 934, 922, 416, 975, 10, 22, 18, 46, 137, 181, 101, 39, 86, 103, 116,
    138, 164, 212, 218, 296, 815, 380, 412, 460, 495, 675, 820, 952,
)

# All three ciphers, keyed by number.
CIPHERS: dict[int, tuple[int, ...]] = {1: B1, 2: B2, 3: B3}

# Special decode overrides for B2. These numbers decode to letters that
# differ from the naive first-letter-of-word rule.
# See cryptanalysis-research.md Section 3 for rationale.
SPECIAL_DECODE: dict[int, str] = {
    95: "u",    # word 95 = "inalienable", decodes as 'u' (unalienable variant)
    811: "y",   # word 811 = "fundamentally", decodes as 'y'
    1005: "x",  # word 1005 = "have", decodes as 'x'
}

# The known correct B2 plaintext (with documented transcription errors).
B2_PLAINTEXT = (
    "ihavedepositedinthecountyofbedfordaboutfourmilesfrombufordsinanexcavation"
    "orvaultsixfeetbelowthesurfaceofthegroundthefollowingarticulesbelongingjo"
    "intlytothepar?iesnamedinpapernumberthreethefi?stdepositconsistedoftenhu"
    "ndredandfourteenpoundsofgoldandthirtyeighthundredandtwelvepoundsofsilver"
    "depositednov?eighteennineteenthesecondwasmadedeceighteentwentyoneandcons"
    "istedofnineteenhundredandsevenpoyndsofgoldandfourteenhundredandtwelvepou"
    "ndsofsilveralsojewe?sobtainedinst?ouisxnexchangetosavetranlportationand"
    "valuedat?hirteenthousanddollarstheaboveissecurelypackediironpotswithi?on"
    "coversthevaultisroughlylinedwithstoneanythevesselsrestonsoli?rocka?darecover"
    "edwithotherspaperone?escribesthcexactlocalityofthevaultsolhatnodifficu?t"
    "ywillbchadinfinditit"
)

# English letter frequencies (approximate, for scoring).
ENGLISH_FREQ: dict[str, float] = {
    "e": 0.1270, "t": 0.0906, "a": 0.0817, "o": 0.0751, "i": 0.0697,
    "n": 0.0675, "s": 0.0633, "h": 0.0609, "r": 0.0599, "d": 0.0425,
    "l": 0.0403, "c": 0.0278, "u": 0.0276, "m": 0.0241, "w": 0.0236,
    "f": 0.0223, "g": 0.0202, "y": 0.0197, "p": 0.0193, "b": 0.0129,
    "v": 0.0098, "k": 0.0077, "j": 0.0015, "x": 0.0015, "q": 0.0010,
    "z": 0.0007,
}

# English bigram log-probabilities (top 100 bigrams).
# Built from large English corpus. Values are log10(probability).
# Missing bigrams get a floor value.
_BIGRAM_RAW: dict[str, float] = {
    "th": -1.43, "he": -1.53, "in": -1.67, "er": -1.74, "an": -1.76,
    "re": -1.82, "on": -1.87, "nd": -1.88, "at": -1.91, "en": -1.91,
    "es": -1.92, "ed": -1.93, "te": -1.95, "ti": -1.96, "or": -1.97,
    "st": -1.99, "ar": -2.02, "is": -2.03, "al": -2.06, "nt": -2.06,
    "to": -2.07, "it": -2.09, "ng": -2.10, "se": -2.12, "ha": -2.15,
    "le": -2.16, "ou": -2.17, "de": -2.18, "io": -2.22, "ne": -2.22,
    "me": -2.24, "ea": -2.25, "ra": -2.27, "ri": -2.27, "ro": -2.28,
    "ve": -2.29, "co": -2.30, "ce": -2.31, "of": -2.31, "li": -2.33,
    "as": -2.33, "ta": -2.34, "si": -2.35, "el": -2.36, "ll": -2.36,
    "no": -2.37, "la": -2.38, "ma": -2.39, "ns": -2.39, "ss": -2.41,
    "pe": -2.41, "di": -2.42, "om": -2.43, "ec": -2.43, "be": -2.44,
    "lo": -2.45, "so": -2.45, "na": -2.45, "em": -2.47, "hi": -2.47,
    "ch": -2.48, "il": -2.48, "ho": -2.50, "us": -2.50, "ur": -2.51,
    "ic": -2.51, "ni": -2.52, "ie": -2.52, "ac": -2.53, "ut": -2.53,
    "ol": -2.54, "wi": -2.55, "ad": -2.55, "rs": -2.56, "ai": -2.56,
    "ge": -2.57, "et": -2.57, "ov": -2.58, "un": -2.58, "wa": -2.59,
    "sh": -2.59, "ot": -2.60, "nc": -2.60, "tr": -2.60, "ct": -2.61,
    "ul": -2.62, "op": -2.63, "po": -2.63, "wo": -2.64, "fo": -2.64,
    "wh": -2.65, "ts": -2.65, "pr": -2.66, "bl": -2.67, "tu": -2.68,
    "ow": -2.68, "rt": -2.69, "mo": -2.69, "pl": -2.70, "ca": -2.70,
}
BIGRAM_FLOOR: float = -4.0  # log10(prob) for unseen bigrams

BIGRAM_LOGPROB: dict[str, float] = {}
for _a in string.ascii_lowercase:
    for _b in string.ascii_lowercase:
        BIGRAM_LOGPROB[_a + _b] = _BIGRAM_RAW.get(_a + _b, BIGRAM_FLOOR)


# ============================================================================
# 2. CODEC — Word-level book cipher encode/decode
# ============================================================================

def first_letter(word: str) -> str:
    """Return the first alphabetic character of a word, lowercased."""
    for c in word:
        if c.isalpha():
            return c.lower()
    return "?"


def decode_book_cipher(
    cipher: Sequence[int],
    key_words: Sequence[str],
    special: dict[int, str] | None = None,
) -> str:
    """
    Decode a book cipher: each number indexes into key_words (1-based),
    take the first letter of that word.

    Args:
        cipher: Sequence of positive integers.
        key_words: The key text as a sequence of words (0-indexed internally).
        special: Optional dict of {cipher_number: override_letter}.

    Returns:
        Decoded string (lowercase letters, '?' for out-of-range).
    """
    if special is None:
        special = {}
    result: list[str] = []
    for num in cipher:
        if num in special:
            result.append(special[num])
        elif 1 <= num <= len(key_words):
            result.append(first_letter(key_words[num - 1]))
        else:
            result.append("?")
    return "".join(result)


def decode_b2() -> str:
    """Convenience: decode B2 with the Beale DoI and special rules."""
    return decode_book_cipher(B2, BEALE_DOI, SPECIAL_DECODE)


def build_letter_index(key_words: Sequence[str]) -> dict[str, list[int]]:
    """
    Build a reverse index: letter -> sorted list of 1-based word positions
    whose first letter matches.
    """
    index: dict[str, list[int]] = {c: [] for c in string.ascii_lowercase}
    for i, word in enumerate(key_words):
        letter = first_letter(word)
        if letter in index:
            index[letter].append(i + 1)
    return index


def encode_book_cipher(
    plaintext: str,
    key_words: Sequence[str],
    rng: np.random.Generator | None = None,
) -> list[int]:
    """
    Encode plaintext as a book cipher using random homophone selection.

    Args:
        plaintext: Lowercase alphabetic string.
        key_words: The key text as a sequence of words.
        rng: NumPy random generator (for reproducibility).

    Returns:
        List of cipher numbers (1-based word positions).

    Raises:
        ValueError: If a letter has no available homophone in the key text.
    """
    if rng is None:
        rng = np.random.default_rng()
    index = build_letter_index(key_words)
    result: list[int] = []
    for ch in plaintext:
        ch = ch.lower()
        if ch not in index or not index[ch]:
            raise ValueError(f"No homophone available for letter '{ch}'")
        result.append(int(rng.choice(index[ch])))
    return result


def encode_sequential_book_cipher(
    plaintext: str,
    key_words: Sequence[str],
    reset_prob: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """
    Encode plaintext by scanning sequentially through the key text.

    For each plaintext letter, selects the nearest-forward homophone position
    (smallest position > cursor). Wraps to the beginning when past the end.

    When reset_prob > 0, simulates losing your place: before each lookup,
    with probability reset_prob the cursor jumps to a random position in
    the key text. Models a hoaxer who periodically loses track of where
    they are in the document and restarts from wherever their eye lands.

    Args:
        plaintext: Lowercase alphabetic string.
        key_words: The key text as a sequence of words.
        reset_prob: Probability of random cursor reset per step (0 = perfect).
        rng: NumPy random generator (required when reset_prob > 0).

    Returns:
        List of cipher numbers (1-based word positions).

    Raises:
        ValueError: If a letter has no available homophone in the key text.
    """
    from bisect import bisect_right

    index = build_letter_index(key_words)
    max_pos = len(key_words)
    cursor = 0
    result: list[int] = []

    for ch in plaintext:
        ch = ch.lower()
        positions = index.get(ch)
        if not positions:
            raise ValueError(f"No homophone available for letter '{ch}'")

        # Simulate losing your place
        if reset_prob > 0.0 and rng is not None and rng.random() < reset_prob:
            cursor = int(rng.integers(0, max_pos))

        # Find smallest position > cursor
        idx = bisect_right(positions, cursor)
        if idx < len(positions):
            pos = positions[idx]
        else:
            # Wrap: take the first position
            pos = positions[0]

        result.append(pos)
        cursor = pos

    return result


# ============================================================================
# 3. CODEC — Letter-index cipher encode/decode
# ============================================================================

def text_to_alpha(text: str) -> str:
    """Extract only alphabetic characters from text, lowercased."""
    return "".join(c.lower() for c in text if c.isalpha())


def decode_letter_cipher(cipher: Sequence[int], key_text: str) -> str:
    """
    Decode a letter-index cipher: each number indexes into the Nth alpha
    character (1-based) of the key text.

    Args:
        cipher: Sequence of positive integers.
        key_text: Raw key text (alpha chars extracted automatically).

    Returns:
        Decoded string (lowercase letters, '?' for out-of-range).
    """
    alpha = text_to_alpha(key_text)
    n = len(alpha)
    result: list[str] = []
    for num in cipher:
        if 1 <= num <= n:
            result.append(alpha[num - 1])
        else:
            result.append("?")
    return "".join(result)


def encode_letter_cipher(
    plaintext: str,
    key_text: str,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """
    Encode plaintext as a letter-index cipher with random homophone selection.

    For each plaintext letter, randomly selects among all positions in the
    key text where that letter appears.

    Args:
        plaintext: Lowercase alphabetic string.
        key_text: Raw key text (alpha chars extracted automatically).
        rng: NumPy random generator (for reproducibility).

    Returns:
        List of cipher numbers (1-based character positions).

    Raises:
        ValueError: If a letter has no occurrence in the key text.
    """
    if rng is None:
        rng = np.random.default_rng()
    alpha = text_to_alpha(key_text)
    # Build reverse index: letter -> list of 1-based positions
    index: dict[str, list[int]] = {c: [] for c in string.ascii_lowercase}
    for i, ch in enumerate(alpha):
        if ch in index:
            index[ch].append(i + 1)
    result: list[int] = []
    for ch in plaintext.lower():
        if ch not in index or not index[ch]:
            raise ValueError(f"No occurrence of letter '{ch}' in key text")
        result.append(int(rng.choice(index[ch])))
    return result


def score_letter_cipher(cipher: Sequence[int], key_text: str) -> dict:
    """
    Score a letter-index decode for English-likeness.

    Returns dict with:
        bigram_score: average bigram log-probability
        ic: index of coincidence
        in_range: fraction of cipher numbers within key text alpha length
        preview: first 80 chars of decoded text
        decoded: full decoded string
    """
    alpha = text_to_alpha(key_text)
    n = len(alpha)
    in_range = sum(1 for num in cipher if 1 <= num <= n) / len(cipher) if cipher else 0.0
    decoded = decode_letter_cipher(cipher, key_text)
    clean = "".join(c for c in decoded if c in string.ascii_lowercase)
    bg = bigram_score(clean) if len(clean) >= 2 else BIGRAM_FLOOR
    ic = index_of_coincidence(clean)
    return {
        "bigram_score": bg,
        "ic": ic,
        "in_range": in_range,
        "preview": decoded[:80],
        "decoded": decoded,
    }


# ============================================================================
# 4. STATS BATTERY — Statistical tests for cipher classification (incl. gillogly_quality)
# ============================================================================

def benford_test(numbers: Sequence[int]) -> dict:
    """
    Test first-digit distribution against Benford's law.

    Returns dict with:
        observed: Counter of first digits 1-9
        expected: Benford expected counts
        chi2: chi-squared statistic
        p_value: p-value from chi-squared test (8 dof)
        epsilon: estimated epsilon-Benford parameter
    """
    from scipy import stats as sp_stats

    first_digits = []
    for n in numbers:
        s = str(abs(n))
        if s[0] != "0":
            first_digits.append(int(s[0]))

    observed = Counter(first_digits)
    total = sum(observed.values())

    # Benford expected proportions
    benford_p = {d: math.log10(1 + 1 / d) for d in range(1, 10)}
    expected = {d: benford_p[d] * total for d in range(1, 10)}

    # Chi-squared
    obs_arr = np.array([observed.get(d, 0) for d in range(1, 10)], dtype=float)
    exp_arr = np.array([expected[d] for d in range(1, 10)], dtype=float)
    chi2, p_value = sp_stats.chisquare(obs_arr, exp_arr)

    # Epsilon estimation (simplified: max absolute deviation from Benford)
    obs_prop = obs_arr / total
    ben_prop = exp_arr / total
    epsilon = float(np.max(np.abs(obs_prop - ben_prop)))

    return {
        "observed": dict(observed),
        "expected": {d: round(expected[d], 1) for d in range(1, 10)},
        "chi2": float(chi2),
        "p_value": float(p_value),
        "epsilon": epsilon,
        "n": total,
    }


def last_digit_test(numbers: Sequence[int], base: int = 10) -> dict:
    """
    Test last-digit distribution for uniformity in the given base.

    Returns dict with:
        observed: Counter of last digits
        chi2: chi-squared statistic
        p_value: p-value (dof = base - 1)
        base: the base tested
    """
    from scipy import stats as sp_stats

    last_digits = [n % base for n in numbers if n > 0]
    observed = Counter(last_digits)
    total = len(last_digits)
    expected_per = total / base

    obs_arr = np.array([observed.get(d, 0) for d in range(base)], dtype=float)
    exp_arr = np.full(base, expected_per, dtype=float)
    chi2, p_value = sp_stats.chisquare(obs_arr, exp_arr)

    return {
        "observed": dict(observed),
        "chi2": float(chi2),
        "p_value": float(p_value),
        "base": base,
        "n": total,
    }


def distinct_ratio(numbers: Sequence[int]) -> dict:
    """
    Compute distinct-value-to-total ratio.

    B2 baseline: ~24%. High ratios (>40%) suggest anomalous cipher.
    """
    total = len(numbers)
    distinct = len(set(numbers))
    return {
        "total": total,
        "distinct": distinct,
        "ratio": distinct / total if total > 0 else 0.0,
    }


def letter_frequency_test(
    cipher: Sequence[int],
    key_words: Sequence[str],
    special: dict[int, str] | None = None,
) -> dict:
    """
    Decode cipher with key, compute letter frequencies, compare to English.

    Returns dict with:
        decoded_freq: observed letter frequency distribution
        chi2: chi-squared against English frequencies
        p_value: p-value (25 dof)
        kl_divergence: KL divergence from English
    """
    from scipy import stats as sp_stats

    decoded = decode_book_cipher(cipher, key_words, special)
    decoded_clean = [c for c in decoded if c in string.ascii_lowercase]
    total = len(decoded_clean)
    if total == 0:
        return {"decoded_freq": {}, "chi2": float("inf"), "p_value": 0.0, "kl_divergence": float("inf")}

    counts = Counter(decoded_clean)
    obs_freq = {c: counts.get(c, 0) / total for c in string.ascii_lowercase}

    # Chi-squared against English
    obs_arr = np.array([counts.get(c, 0) for c in string.ascii_lowercase], dtype=float)
    exp_arr = np.array([ENGLISH_FREQ.get(c, 0.0005) * total for c in string.ascii_lowercase], dtype=float)
    # Ensure no zero expected values, then renormalize to match observed total
    exp_arr = np.maximum(exp_arr, 0.5)
    exp_arr = exp_arr * (obs_arr.sum() / exp_arr.sum())
    chi2, p_value = sp_stats.chisquare(obs_arr, exp_arr)

    # KL divergence: D_KL(observed || english)
    kl = 0.0
    for c in string.ascii_lowercase:
        p = obs_freq.get(c, 1e-6)
        q = ENGLISH_FREQ.get(c, 1e-6)
        if p > 0:
            kl += p * math.log2(p / q)

    return {
        "decoded_freq": obs_freq,
        "chi2": float(chi2),
        "p_value": float(p_value),
        "kl_divergence": kl,
        "n": total,
    }


def gillogly_strings(
    cipher: Sequence[int],
    key_words: Sequence[str],
    min_run: int = 5,
) -> list[dict]:
    """
    Detect alphabetical runs in the decoded output (Gillogly's test).

    Applies the DoI key to the cipher and looks for sequences where
    consecutive decoded letters are in alphabetical order.

    Args:
        cipher: Cipher numbers.
        key_words: Key text words.
        min_run: Minimum length of alphabetical run to report.

    Returns:
        List of dicts with {start, end, length, letters, cipher_nums}.
    """
    decoded = decode_book_cipher(cipher, key_words)
    runs: list[dict] = []
    run_start = 0

    for i in range(1, len(decoded)):
        if decoded[i] >= decoded[i - 1]:
            continue
        # End of ascending run
        run_len = i - run_start
        if run_len >= min_run:
            runs.append({
                "start": run_start,
                "end": i - 1,
                "length": run_len,
                "letters": decoded[run_start:i],
                "cipher_nums": list(cipher[run_start:i]),
            })
        run_start = i

    # Check final run
    run_len = len(decoded) - run_start
    if run_len >= min_run:
        runs.append({
            "start": run_start,
            "end": len(decoded) - 1,
            "length": run_len,
            "letters": decoded[run_start:],
            "cipher_nums": list(cipher[run_start:]),
        })

    return runs


def gillogly_quality(decoded: str, min_run: int = 5) -> dict:
    """
    Measure alphabetical-run quality of a decoded string.

    Used to score DoI variant optimization — higher = more Gillogly-like structure.

    Args:
        decoded: Lowercase decoded string.
        min_run: Minimum run length to count.

    Returns:
        dict with longest_run, total_run_length, run_count, score, runs list.
    """
    runs: list[dict] = []
    run_start = 0

    for i in range(1, len(decoded)):
        if decoded[i] >= decoded[i - 1]:
            continue
        run_len = i - run_start
        if run_len >= min_run:
            runs.append({
                "start": run_start,
                "end": i - 1,
                "length": run_len,
                "letters": decoded[run_start:i],
            })
        run_start = i

    # Final run
    run_len = len(decoded) - run_start
    if run_len >= min_run:
        runs.append({
            "start": run_start,
            "end": len(decoded) - 1,
            "length": run_len,
            "letters": decoded[run_start:],
        })

    longest = max((r["length"] for r in runs), default=0)
    total = sum(r["length"] for r in runs)
    return {
        "longest_run": longest,
        "total_run_length": total,
        "run_count": len(runs),
        "score": longest + 0.3 * total,
        "runs": runs,
    }


def run_battery(
    cipher: Sequence[int],
    key_words: Sequence[str] | None = None,
    special: dict[int, str] | None = None,
    label: str = "",
) -> dict:
    """
    Run the full stats battery on a cipher.

    Args:
        cipher: Cipher number sequence.
        key_words: Optional key text (for decode-dependent tests). Defaults to BEALE_DOI.
        special: Optional special decode overrides.
        label: Human-readable label for this cipher.

    Returns:
        Dict with all test results.
    """
    if key_words is None:
        key_words = BEALE_DOI

    results: dict = {"label": label or "unlabeled"}
    results["count"] = len(cipher)
    results["range"] = (min(cipher), max(cipher))
    results["distinct"] = distinct_ratio(cipher)
    results["benford"] = benford_test(cipher)
    results["last_digit_base10"] = last_digit_test(cipher, base=10)
    results["last_digit_base7"] = last_digit_test(cipher, base=7)
    results["last_digit_base3"] = last_digit_test(cipher, base=3)
    results["letter_freq"] = letter_frequency_test(cipher, key_words, special)
    results["gillogly"] = gillogly_strings(cipher, key_words)

    return results


# ============================================================================
# 5. SCORING — English-likeness scoring
# ============================================================================

def bigram_score(text: str) -> float:
    """
    Compute average bigram log-probability for a decoded text.

    Higher (less negative) = more English-like.
    Random text: roughly -4.0. Good English: roughly -2.0 to -2.5.
    """
    text = "".join(c for c in text.lower() if c in string.ascii_lowercase)
    if len(text) < 2:
        return BIGRAM_FLOOR
    total = 0.0
    count = 0
    for i in range(len(text) - 1):
        bg = text[i] + text[i + 1]
        total += BIGRAM_LOGPROB.get(bg, BIGRAM_FLOOR)
        count += 1
    return total / count if count > 0 else BIGRAM_FLOOR


def build_bigram_table(text: str, floor: float = BIGRAM_FLOOR) -> dict[str, float]:
    """
    Build a bigram log-probability table from a corpus text.

    Args:
        text: Raw text (alpha chars extracted, lowercased).
        floor: Log-prob floor for unseen bigrams.

    Returns:
        Dict mapping each of 676 bigrams to log10(probability).
    """
    clean = "".join(c for c in text.lower() if c in string.ascii_lowercase)
    counts: dict[str, int] = {}
    for i in range(len(clean) - 1):
        bg = clean[i] + clean[i + 1]
        counts[bg] = counts.get(bg, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return {a + b: floor for a in string.ascii_lowercase for b in string.ascii_lowercase}
    table: dict[str, float] = {}
    for a in string.ascii_lowercase:
        for b in string.ascii_lowercase:
            bg = a + b
            c = counts.get(bg, 0)
            table[bg] = math.log10(c / total) if c > 0 else floor
    return table


def score_with_bigram_table(text: str, table: dict[str, float], floor: float = BIGRAM_FLOOR) -> float:
    """
    Score text against an arbitrary bigram log-probability table.

    Same interface as bigram_score() but uses a custom table.
    """
    clean = "".join(c for c in text.lower() if c in string.ascii_lowercase)
    if len(clean) < 2:
        return floor
    total = 0.0
    count = 0
    for i in range(len(clean) - 1):
        bg = clean[i] + clean[i + 1]
        total += table.get(bg, floor)
        count += 1
    return total / count if count > 0 else floor


def index_of_coincidence(text: str) -> float:
    """
    Compute the index of coincidence for a text.

    English: ~0.0667. Random (uniform 26): ~0.0385.
    """
    text = "".join(c for c in text.lower() if c in string.ascii_lowercase)
    n = len(text)
    if n < 2:
        return 0.0
    counts = Counter(text)
    ic = sum(c * (c - 1) for c in counts.values()) / (n * (n - 1))
    return ic


def score_decode(
    cipher: Sequence[int],
    key_words: Sequence[str],
    special: dict[int, str] | None = None,
) -> dict:
    """
    Score a candidate decode for English-likeness.

    Returns dict with:
        bigram_score: average bigram log-probability
        ic: index of coincidence
        letter_freq_chi2: chi2 vs English letter frequencies
        in_range_pct: percentage of cipher numbers within key text range
        overall: composite score (higher = more English-like)
    """
    decoded = decode_book_cipher(cipher, key_words, special)
    clean = "".join(c for c in decoded if c in string.ascii_lowercase)

    bg = bigram_score(clean)
    ic = index_of_coincidence(clean)
    freq = letter_frequency_test(cipher, key_words, special)
    in_range = sum(1 for n in cipher if 1 <= n <= len(key_words)) / len(cipher)

    # Composite: weighted sum (arbitrary but defensible weights)
    # bigram_score: range roughly -4 to -2, scale to 0-1
    bg_norm = max(0.0, min(1.0, (bg - BIGRAM_FLOOR) / ((-2.0) - BIGRAM_FLOOR)))
    # IC: English ~0.067, random ~0.038. Scale to 0-1
    ic_norm = max(0.0, min(1.0, (ic - 0.038) / (0.067 - 0.038)))
    # Chi2: lower is better. Invert and normalize roughly.
    chi2_norm = max(0.0, 1.0 - freq["chi2"] / 1000.0)

    overall = 0.4 * bg_norm + 0.3 * ic_norm + 0.2 * chi2_norm + 0.1 * in_range

    return {
        "decoded_text": decoded,
        "bigram_score": bg,
        "ic": ic,
        "letter_freq_chi2": freq["chi2"],
        "in_range_pct": in_range,
        "overall": overall,
    }


# ============================================================================
# 6. CORPUS UTILS — Key text loading and processing
# ============================================================================

def text_to_words(text: str) -> list[str]:
    """
    Split text into words suitable for use as a book cipher key.

    Strips punctuation, lowercases, preserves hyphenated words as single tokens.
    Matches the Beale convention: words are whitespace-delimited tokens.
    """
    import re
    # Replace em-dashes and similar with spaces
    text = re.sub(r"[—–]", " ", text)
    # Extract word-like tokens (letters, hyphens, apostrophes)
    tokens = re.findall(r"[a-zA-Z][a-zA-Z'\-]*", text)
    return [t.lower() for t in tokens]


def load_gutenberg_text(filepath: str | Path) -> str:
    """
    Load a Project Gutenberg text file, stripping the header/footer boilerplate.
    """
    path = Path(filepath)
    text = path.read_text(encoding="utf-8", errors="replace")

    # Strip Gutenberg header
    markers = [
        "*** START OF THIS PROJECT GUTENBERG",
        "*** START OF THE PROJECT GUTENBERG",
        "***START OF THIS PROJECT GUTENBERG",
    ]
    for marker in markers:
        idx = text.find(marker)
        if idx != -1:
            # Find end of the marker line
            nl = text.find("\n", idx)
            if nl != -1:
                text = text[nl + 1:]
            break

    # Strip Gutenberg footer
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG",
        "*** END OF THE PROJECT GUTENBERG",
        "***END OF THIS PROJECT GUTENBERG",
    ]
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break

    return text.strip()


def generate_fake_cipher(
    plaintext_length: int,
    key_words: Sequence[str],
    rng: np.random.Generator | None = None,
) -> list[int]:
    """
    Generate a fake (but statistically genuine-looking) book cipher.

    Generates random English-frequency text, then encodes it with the key.
    Only uses letters that have homophones in the key text.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Only use letters that have homophones available in the key
    index = build_letter_index(key_words)
    available = [c for c in ENGLISH_FREQ if index.get(c)]
    probs = np.array([ENGLISH_FREQ[c] for c in available])
    probs /= probs.sum()
    plaintext = "".join(rng.choice(available, size=plaintext_length, p=probs))

    return encode_book_cipher(plaintext, key_words, rng)


def generate_random_numbers(
    count: int,
    max_val: int,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """
    Generate uniformly random cipher numbers (null hypothesis: pure noise).
    """
    if rng is None:
        rng = np.random.default_rng()
    return [int(x) for x in rng.integers(1, max_val + 1, size=count)]


def generate_human_random(
    count: int,
    max_val: int,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """
    Generate numbers with known human-random biases.

    Simulates: small-number bias (log-normal), digit preference (7 favored,
    0 penalized), sequential avoidance (reject if within 5% of previous),
    repeat suppression (halved probability of exact repeats).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Base distribution: log-normal skewed toward smaller values
    # Tuned so median ≈ max_val * 0.35, bulk within 1-max_val
    sigma = 0.8
    mu = math.log(max_val * 0.35)
    threshold = max_val * 0.05  # sequential avoidance distance

    result: list[int] = []
    prev = -1
    attempts = 0
    max_attempts = count * 20

    while len(result) < count and attempts < max_attempts:
        attempts += 1
        # Draw from log-normal, clamp to range
        raw = rng.lognormal(mu, sigma)
        val = int(round(raw))
        if val < 1 or val > max_val:
            continue

        # Digit preference: boost 7-ending, penalize 0-ending
        last_dig = val % 10
        if last_dig == 0 and rng.random() < 0.3:
            continue  # reject 30% of 0-endings
        if last_dig != 7 and rng.random() < 0.05:
            # Slight chance to shift to nearest 7-ending
            val = val - last_dig + 7
            if val < 1 or val > max_val:
                continue

        # Sequential avoidance: reject if too close to previous
        if prev > 0 and abs(val - prev) < threshold:
            if rng.random() < 0.6:  # reject 60% of close pairs
                continue

        # Repeat avoidance: halve probability of exact repeats
        if val == prev and rng.random() < 0.5:
            continue

        result.append(val)
        prev = val

    # Fallback: fill any remaining with uniform (shouldn't happen)
    while len(result) < count:
        result.append(int(rng.integers(1, max_val + 1)))

    return result


# ============================================================================
# 7. OUTPUT UTILS — Formatting, comparison tables, plots
# ============================================================================

def format_battery_comparison(results: list[dict]) -> str:
    """
    Format a side-by-side comparison table of battery results.

    Args:
        results: List of dicts from run_battery().

    Returns:
        Formatted string table.
    """
    lines: list[str] = []
    labels = [r["label"] for r in results]
    header = f"{'Test':<35} " + " ".join(f"{l:>15}" for l in labels)
    lines.append(header)
    lines.append("-" * len(header))

    rows = [
        ("Count", lambda r: f"{r['count']}"),
        ("Range", lambda r: f"{r['range'][0]}-{r['range'][1]}"),
        ("Distinct values", lambda r: f"{r['distinct']['distinct']}"),
        ("Distinct ratio", lambda r: f"{r['distinct']['ratio']:.1%}"),
        ("Benford chi2", lambda r: f"{r['benford']['chi2']:.1f}"),
        ("Benford p-value", lambda r: f"{r['benford']['p_value']:.4f}"),
        ("Benford epsilon", lambda r: f"{r['benford']['epsilon']:.3f}"),
        ("Last digit (b10) chi2", lambda r: f"{r['last_digit_base10']['chi2']:.1f}"),
        ("Last digit (b10) p", lambda r: f"{r['last_digit_base10']['p_value']:.4f}"),
        ("Last digit (b7) chi2", lambda r: f"{r['last_digit_base7']['chi2']:.1f}"),
        ("Last digit (b7) p", lambda r: f"{r['last_digit_base7']['p_value']:.4f}"),
        ("Last digit (b3) chi2", lambda r: f"{r['last_digit_base3']['chi2']:.1f}"),
        ("Last digit (b3) p", lambda r: f"{r['last_digit_base3']['p_value']:.4f}"),
        ("Letter freq chi2", lambda r: f"{r['letter_freq']['chi2']:.1f}"),
        ("Letter freq p", lambda r: f"{r['letter_freq']['p_value']:.4f}"),
        ("Letter freq KL div", lambda r: f"{r['letter_freq']['kl_divergence']:.3f}"),
        ("Gillogly runs (>=5)", lambda r: f"{len(r['gillogly'])}"),
    ]

    for name, fn in rows:
        vals = []
        for r in results:
            try:
                vals.append(fn(r))
            except (KeyError, TypeError):
                vals.append("N/A")
        lines.append(f"{name:<35} " + " ".join(f"{v:>15}" for v in vals))

    return "\n".join(lines)


def format_decode_preview(
    decoded: str,
    width: int = 70,
    word_breaks: bool = True,
) -> str:
    """
    Format a decoded string for display with line wrapping.
    """
    lines: list[str] = []
    for i in range(0, len(decoded), width):
        chunk = decoded[i : i + width]
        lines.append(f"  {i:4d}: {chunk}")
    return "\n".join(lines)


def plot_benford_comparison(
    results: list[dict],
    save_path: str | Path | None = None,
) -> None:
    """
    Plot first-digit distributions vs Benford's law for multiple ciphers.

    Requires matplotlib (optional dependency).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not available; skipping plot")
        return

    digits = list(range(1, 10))
    benford_expected = [math.log10(1 + 1 / d) for d in digits]

    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), squeeze=False)

    for idx, r in enumerate(results):
        ax = axes[0][idx]
        obs = r["benford"]["observed"]
        total = r["benford"]["n"]
        obs_prop = [obs.get(d, 0) / total for d in digits]

        ax.bar(digits, obs_prop, alpha=0.7, label="Observed", width=0.4, align="edge")
        ax.bar([d + 0.4 for d in digits], benford_expected, alpha=0.5,
               label="Benford", width=0.4, align="edge", color="orange")
        ax.set_title(f"{r['label']}\n(chi2={r['benford']['chi2']:.1f}, p={r['benford']['p_value']:.3f})")
        ax.set_xlabel("First digit")
        ax.set_ylabel("Proportion")
        ax.legend(fontsize=8)
        ax.set_xticks(digits)

    plt.tight_layout()
    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_last_digit_comparison(
    results: list[dict],
    bases: list[int] | None = None,
    save_path: str | Path | None = None,
) -> None:
    """
    Plot last-digit distributions for multiple ciphers across bases.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not available; skipping plot")
        return

    if bases is None:
        bases = [10, 7, 3]

    fig, axes = plt.subplots(len(bases), len(results),
                             figsize=(4 * len(results), 3 * len(bases)),
                             squeeze=False)

    for row, base in enumerate(bases):
        for col, r in enumerate(results):
            ax = axes[row][col]
            key = f"last_digit_base{base}"
            if key not in r:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                continue

            obs = r[key]["observed"]
            total = r[key]["n"]
            digits = list(range(base))
            obs_counts = [obs.get(d, 0) for d in digits]
            expected = total / base

            ax.bar(digits, obs_counts, alpha=0.7, label="Observed")
            ax.axhline(expected, color="red", linestyle="--", label="Expected (uniform)")
            ax.set_title(f"{r['label']} base-{base}\n(p={r[key]['p_value']:.3f})", fontsize=9)
            ax.set_xlabel(f"Last digit (base {base})")
            ax.set_ylabel("Count")
            ax.legend(fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()


# ============================================================================
# SELF-TEST — Run when executed directly
# ============================================================================

def _self_test() -> None:
    """Verify B2 decode and run battery on all three ciphers."""
    print("=== beale.py self-test ===\n")

    # 1. Verify B2 decode
    decoded = decode_b2()
    first_19 = decoded[:19]
    expected_19 = "ihavedepositedinthe"
    match = first_19 == expected_19
    print(f"B2 decode first 19: '{first_19}'")
    print(f"Expected:           '{expected_19}'")
    print(f"Match: {match}")
    assert match, f"B2 decode FAILED: got '{first_19}'"

    # 2. Spot-check key positions
    # Verify specific word-to-letter mappings (letter matters, not specific word)
    checks = [
        (115, "i"),   # "instituted" or similar i-word
        (73, "h"),    # "hold"
        (24, "a"),    # an a-word
        (807, "v"),   # "valuable"
        (811, "y"),   # "fundamentally" -> special decode 'y'
        (1005, "x"),  # "have" -> special decode 'x'
    ]
    for num, expected_letter in checks:
        word = BEALE_DOI[num - 1]
        letter = first_letter(word)
        if num in SPECIAL_DECODE:
            letter = SPECIAL_DECODE[num]
        assert letter == expected_letter, f"Word {num} ('{word}'): expected '{expected_letter}', got '{letter}'"
    print("Key position checks: PASS\n")

    # 3. Word count
    print(f"DoI word count: {len(BEALE_DOI)}")
    print(f"B1: {len(B1)} numbers, range {min(B1)}-{max(B1)}, {len(set(B1))} distinct")
    print(f"B2: {len(B2)} numbers, range {min(B2)}-{max(B2)}, {len(set(B2))} distinct")
    print(f"B3: {len(B3)} numbers, range {min(B3)}-{max(B3)}, {len(set(B3))} distinct")
    print()

    # 4. Run battery on all three
    print("Running stats battery on all three ciphers...")
    r1 = run_battery(B1, label="B1 (Location)")
    r2 = run_battery(B2, special=SPECIAL_DECODE, label="B2 (Contents)")
    r3 = run_battery(B3, label="B3 (Names)")

    print()
    print(format_battery_comparison([r1, r2, r3]))

    # 5. English-likeness scoring
    print("\n\nEnglish-likeness scores:")
    for cipher, label, sp in [(B1, "B1", None), (B2, "B2", SPECIAL_DECODE), (B3, "B3", None)]:
        sc = score_decode(cipher, BEALE_DOI, sp)
        print(f"  {label}: bigram={sc['bigram_score']:.3f}  IC={sc['ic']:.4f}  "
              f"in_range={sc['in_range_pct']:.1%}  overall={sc['overall']:.3f}")

    # 6. Gillogly strings in B1
    gs = r1["gillogly"]
    if gs:
        print(f"\nGillogly strings in B1 (runs >= 5):")
        for g in gs:
            print(f"  pos {g['start']}-{g['end']}: '{g['letters']}' (len {g['length']})")

    print("\n=== Self-test complete ===")


if __name__ == "__main__":
    _self_test()
