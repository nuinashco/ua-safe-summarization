"""
Parity tests: MRougeScorer vs Google's rouge_score reference library.

Tokeniser note
--------------
rouge_score.tokenize lowercases and replaces every non-[a-z0-9] character
with a space, keeping only purely alphanumeric tokens.  Cyrillic is stripped
entirely, so Ukrainian text cannot be used for cross-library parity.  We
replicate this as ``_compat_tokenizer`` for all parity assertions.

rougeLsum note
--------------
For single-sentence predictions the two implementations reduce to the same
LCS computation and are directly comparable.  For multi-sentence predictions
the aggregation strategies differ (token-counter deduplication vs.
prediction-index union), so we only assert invariants there.
"""
from __future__ import annotations

import re

import pytest
from rouge_score import rouge_scorer as _google_rs

from safesum.metrics import MRougeScorer
from safesum.utils import make_uk_sentence_splitter, make_uk_tokenizer, whitespace_tokenizer

ATOL = 1e-4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compat_tokenizer(text: str):
    """Mirrors rouge_score's tokenizer: lowercase, keep [a-z0-9]+ tokens."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).split()


def _google(rtypes: list, ref: str, prd: str) -> dict:
    return {
        k: v.fmeasure
        for k, v in _google_rs.RougeScorer(rtypes, use_stemmer=False)
        .score(ref, prd)
        .items()
    }


def _ours(rtypes: list, ref: str, prd: str, tok=_compat_tokenizer) -> dict:
    return {
        k: v.fmeasure
        for k, v in MRougeScorer(rtypes, tok).score(ref, prd).items()
    }


def assert_parity(rtypes: list, ref: str, prd: str, atol: float = ATOL) -> None:
    g, o = _google(rtypes, ref, prd), _ours(rtypes, ref, prd)
    for rtype in rtypes:
        diff = abs(g[rtype] - o[rtype])
        assert diff <= atol, (
            f"{rtype}: google={g[rtype]:.6f}  ours={o[rtype]:.6f}  diff={diff:.2e}\n"
            f"  ref={ref!r}\n  prd={prd!r}"
        )


# ---------------------------------------------------------------------------
# Zero-score and identity invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "a b c",
    "the quick brown fox",
    "hello world this is a test sentence",
])
def test_identity_all_types(text):
    rtypes = ["rouge1", "rouge2", "rougeL"]
    s = _ours(rtypes, text, text)
    for rtype, score in s.items():
        assert abs(score - 1.0) < ATOL, f"{rtype} identity ≠ 1.0 for {text!r}"
    assert_parity(rtypes, text, text)


@pytest.mark.parametrize("rtype", ["rouge1", "rouge2", "rougeL", "rougeLsum"])
def test_empty_prediction_is_zero(rtype):
    assert _ours([rtype], "a b c", "")[rtype] == 0.0


@pytest.mark.parametrize("rtype", ["rouge1", "rouge2", "rougeL", "rougeLsum"])
def test_empty_reference_is_zero(rtype):
    assert _ours([rtype], "", "a b c")[rtype] == 0.0


def test_no_overlap_is_zero():
    rtypes = ["rouge1", "rouge2", "rougeL"]
    s = _ours(rtypes, "cat dog bird", "one two three")
    for rtype, score in s.items():
        assert score == 0.0
    assert_parity(rtypes, "cat dog bird", "one two three")


# ---------------------------------------------------------------------------
# rouge1 parity
# ---------------------------------------------------------------------------

def test_rouge1_partial_overlap():
    # ref [a,b,c,d], prd [a,b,e,f]: overlap=2 → P=R=F=0.5
    assert_parity(["rouge1"], "a b c d", "a b e f")
    assert abs(_ours(["rouge1"], "a b c d", "a b e f")["rouge1"] - 0.5) < ATOL


def test_rouge1_repeated_tokens_clipped():
    # ref [a,a,a,b], prd [a,b]: min-count clipping → overlap=2
    # P=2/2=1.0, R=2/4=0.5, F=2/3
    assert_parity(["rouge1"], "a a a b", "a b")
    assert abs(_ours(["rouge1"], "a a a b", "a b")["rouge1"] - 2 / 3) < ATOL


def test_rouge1_prediction_longer_than_reference():
    # ref [a,b], prd [a,b,c,d,e,f]: overlap=2, P=2/6, R=2/2=1.0, F=0.5
    assert_parity(["rouge1"], "a b", "a b c d e f")
    assert abs(_ours(["rouge1"], "a b", "a b c d e f")["rouge1"] - 0.5) < ATOL


def test_rouge1_longer_english_sentence():
    ref = "the quick brown fox jumps over the lazy dog"
    prd = "a quick brown fox leaps over the lazy cat"
    assert_parity(["rouge1"], ref, prd)


# ---------------------------------------------------------------------------
# rouge2 parity
# ---------------------------------------------------------------------------

def test_rouge2_single_shared_bigram():
    # ref [(a,b),(b,c)], prd [(a,b),(b,d)]: overlap=(a,b) → P=R=F=0.5
    assert_parity(["rouge2"], "a b c", "a b d")
    assert abs(_ours(["rouge2"], "a b c", "a b d")["rouge2"] - 0.5) < ATOL


def test_rouge2_order_matters():
    # same unigrams, reversed bigrams → rouge2=0, rouge1=1
    assert_parity(["rouge1", "rouge2"], "a b c", "c b a")
    s = _ours(["rouge1", "rouge2"], "a b c", "c b a")
    assert abs(s["rouge1"] - 1.0) < ATOL
    assert s["rouge2"] == 0.0


def test_rouge2_longer_sentence():
    ref = "the quick brown fox jumps over the lazy dog"
    prd = "a quick brown fox jumps over the lazy cat"
    assert_parity(["rouge2"], ref, prd)


# ---------------------------------------------------------------------------
# rougeL parity
# ---------------------------------------------------------------------------

def test_rougel_hypothesis_is_subsequence():
    # prd is a strict subsequence of ref → rougeL precision = 1.0
    assert_parity(["rouge1", "rougeL"], "a b c d e", "a c e")
    s = _ours(["rougeL"], "a b c d e", "a c e")
    # LCS=3, P=3/3=1.0, R=3/5=0.6, F=0.75
    assert abs(s["rougeL"] - 0.75) < ATOL


def test_rougel_reversed_scores_below_rouge1():
    # reversed order: rouge1 perfect, rougeL low (LCS=1)
    assert_parity(["rouge1", "rougeL"], "a b c d", "d c b a")
    s = _ours(["rouge1", "rougeL"], "a b c d", "d c b a")
    assert abs(s["rouge1"] - 1.0) < ATOL
    assert abs(s["rougeL"] - 0.25) < ATOL  # LCS=1/4


def test_rougel_longer_english():
    ref = "the quick brown fox jumps over the lazy dog"
    prd = "a quick brown fox jumps over the lazy cat"
    assert_parity(["rouge1", "rouge2", "rougeL"], ref, prd)


# ---------------------------------------------------------------------------
# rougeLsum — single-sentence parity with Google
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ref,prd", [
    ("a b c d", "a b"),
    ("a b c d", "a b c d"),
    ("the fox jumps over the dog", "fox dog jumps"),
    ("one two three four five", "two four"),
])
def test_rougeLsum_single_sentence_matches_rougeL_and_google(ref, prd):
    s = _ours(["rougeL", "rougeLsum"], ref, prd)
    assert abs(s["rougeL"] - s["rougeLsum"]) < ATOL, (
        f"rougeL={s['rougeL']:.4f} != rougeLsum={s['rougeLsum']:.4f}"
    )
    assert_parity(["rougeLsum"], ref, prd)


# ---------------------------------------------------------------------------
# rougeLsum — multi-sentence invariants (no cross-library parity claim)
# ---------------------------------------------------------------------------

def test_rougeLsum_reversed_sentence_order_exceeds_rougeL():
    # Prediction has same facts as reference but in reversed sentence order.
    # rougeLsum matches each reference sentence independently → higher score.
    ref = "prices rose\ndemand fell"
    prd = "demand fell and prices rose"
    s = _ours(["rougeL", "rougeLsum"], ref, prd)
    assert s["rougeLsum"] > s["rougeL"], (
        f"Expected rougeLsum > rougeL, got rougeL={s['rougeL']:.3f} "
        f"rougeLsum={s['rougeLsum']:.3f}"
    )


def test_rougeLsum_full_coverage():
    # Identical content in reversed sentence order → close to 1.0
    ref = "a b\nc d"
    prd = "c d\na b"
    s = _ours(["rougeLsum"], ref, prd)
    assert abs(s["rougeLsum"] - 1.0) < ATOL


@pytest.mark.parametrize("ref,prd", [
    ("cat sat\ncat ran", "cat"),
    ("a b c\na b d", "a b"),
    ("x y z\nx y w", "x y"),
])
def test_rougeLsum_precision_never_exceeds_one(ref, prd):
    s = MRougeScorer(["rougeLsum"], _compat_tokenizer).score(ref, prd)
    assert s["rougeLsum"].precision <= 1.0 + ATOL, (
        f"Precision={s['rougeLsum'].precision:.4f} > 1.0 for ref={ref!r} prd={prd!r}"
    )


# ---------------------------------------------------------------------------
# Ukrainian — correctness invariants (rouge_score strips Cyrillic entirely)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def uk():
    return make_uk_tokenizer()


def test_uk_identity(uk):
    text = "Президент підписав новий закон про освіту"
    s = MRougeScorer(["rouge1", "rouge2", "rougeL"], uk).score(text, text)
    for rtype, score in s.items():
        assert abs(score.fmeasure - 1.0) < ATOL, f"{rtype} identity ≠ 1.0"


def test_uk_zero_overlap(uk):
    s = MRougeScorer(["rouge1", "rouge2", "rougeL"], uk).score(
        "Погода буде сонячною та теплою",
        "Команда виграла чемпіонат Європи",
    )
    for rtype, score in s.items():
        assert score.fmeasure == 0.0, f"{rtype} expected 0 but got {score.fmeasure}"


def test_both_tokenizers_strip_punctuation(uk):
    # Both whitespace_tokenizer and make_uk_tokenizer now drop punctuation tokens,
    # mirroring the BasicTokenizer step in the XL-Sum multilingual rouge scorer.
    # "освіту," and "освіту." both reduce to "освіту" so they match correctly.
    ref = "Президент підписав закон про освіту, який набере чинності."
    prd = "Президент підписав закон про освіту."
    s_ws = MRougeScorer(["rouge1"], whitespace_tokenizer).score(ref, prd)
    s_uk = MRougeScorer(["rouge1"], uk).score(ref, prd)
    assert abs(s_ws["rouge1"].fmeasure - s_uk["rouge1"].fmeasure) < ATOL


def test_uk_rougeLsum_beats_rougeL_on_reversed_sentences(uk):
    ref = "Ціни зросли.\nПопит знизився."
    prd = "Попит знизився і ціни зросли."
    s = MRougeScorer(["rougeL", "rougeLsum"], uk).score(ref, prd)
    assert s["rougeLsum"].fmeasure > s["rougeL"].fmeasure


# ---------------------------------------------------------------------------
# Normalization — apostrophe variants
# ---------------------------------------------------------------------------

def test_uk_apostrophe_variants_match(uk):
    # ʼ (U+02BC) and ' (U+0027) are both used in Ukrainian; should score 1.0
    ref = "м\u02BCяч"   # ʼ variant
    prd = "м\u0027яч"   # ASCII apostrophe
    s = MRougeScorer(["rouge1"], uk).score(ref, prd)
    assert abs(s["rouge1"].fmeasure - 1.0) < ATOL, (
        f"apostrophe normalisation failed: {s['rouge1'].fmeasure:.4f}"
    )


def test_uk_nfkc_normalization(uk):
    # full-width digits should reduce to ASCII digits after NFKC
    ref = "2024"
    prd = "\uff12\uff10\uff12\uff14"  # ２０２４ full-width
    s = MRougeScorer(["rouge1"], uk).score(ref, prd)
    assert abs(s["rouge1"].fmeasure - 1.0) < ATOL, (
        f"NFKC normalisation failed: {s['rouge1'].fmeasure:.4f}"
    )


# ---------------------------------------------------------------------------
# make_uk_sentence_splitter — linguistic sentence boundaries
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def uk_split():
    return make_uk_sentence_splitter()


def test_uk_sentence_splitter_splits_on_period(uk_split):
    sents = uk_split("Ціни зросли. Попит знизився.")
    assert len(sents) == 2


def test_uk_sentence_splitter_required_for_free_form_rougeLsum(uk, uk_split):
    ref = "Ціни зросли. Попит знизився."
    prd = "Попит знизився. Ціни зросли."  # same facts, reversed, no \n

    s_default = MRougeScorer(["rougeL", "rougeLsum"], uk).score(ref, prd)
    s_split = MRougeScorer(["rougeL", "rougeLsum"], uk, sentence_splitter=uk_split).score(ref, prd)

    assert abs(s_default["rougeLsum"].fmeasure - s_default["rougeL"].fmeasure) < ATOL, (
        "without splitter: rougeLsum should degenerate to rougeL on free-form text"
    )
    assert s_split["rougeLsum"].fmeasure > s_split["rougeL"].fmeasure, (
        f"with splitter: rougeLsum={s_split['rougeLsum'].fmeasure:.3f} "
        f"should exceed rougeL={s_split['rougeL'].fmeasure:.3f}"
    )


def test_uk_score_decreases_with_overlap(uk):
    # Realistic Ukrainian pairs at decreasing overlap — scorer should rank them correctly.
    scorer = MRougeScorer(["rouge1"], uk)
    pairs = [
        ("near-identical",
         "Президент підписав новий закон про освіту, який набере чинності з вересня.",
         "Президент підписав закон про освіту, що набере чинності у вересні."),
        ("partial",
         "Уряд виділив кошти на модернізацію шкіл у сільській місцевості.",
         "Кошти спрямують на ремонт сільських шкіл."),
        ("weak",
         "Національний банк знизив облікову ставку на півтора відсотка.",
         "Центральний банк переглянув грошово-кредитну політику."),
        ("none",
         "Погода у вихідні буде сонячною та теплою.",
         "Футбольна команда виграла чемпіонат Європи."),
    ]
    scores = [(label, scorer.score(ref, prd)["rouge1"].fmeasure) for label, ref, prd in pairs]
    for (la, fa), (lb, fb) in zip(scores, scores[1:]):
        assert fa > fb, f"Expected {la}({fa:.3f}) > {lb}({fb:.3f})"
