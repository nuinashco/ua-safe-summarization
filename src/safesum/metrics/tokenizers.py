from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# Matches any character that is NOT a Unicode letter or digit.
# Used to strip punctuation from tokens before ROUGE scoring.
_PUNCT_RE = re.compile(r"[^\w]", re.UNICODE)

# Apostrophe variants found in Ukrainian corpora (ʼ, `, ' curly).
_APOS_RE = re.compile(r"[ʼ`’]")


def _normalize_uk(text: str) -> str:
    """NFKC + apostrophe normalisation + whitespace collapse."""
    t = unicodedata.normalize("NFKC", str(text))
    t = _APOS_RE.sub("'", t)
    return re.sub(r"\s+", " ", t).strip()


def _strip_punct(token: str) -> str:
    return _PUNCT_RE.sub("", token)


def _require_tokenize_uk():
    try:
        import tokenize_uk  # type: ignore
        return tokenize_uk
    except ImportError as e:
        raise ImportError(
            "tokenize-uk is required for Ukrainian processing. "
            "Install it with: uv add tokenize-uk"
        ) from e


def whitespace_tokenizer(text: str) -> List[str]:
    """
    Whitespace tokenizer with punctuation stripping.

    Lowercases, splits on whitespace, then removes any non-word characters
    from each token (e.g. trailing commas, periods, quotes).  Empty tokens
    after stripping are dropped.  Mirrors the BasicTokenizer pre-processing
    step used in the XL-Sum multilingual rouge scorer.
    """
    return [t for token in text.lower().split() if (t := _strip_punct(token))]


def make_uk_tokenizer():
    """
    Ukrainian word tokenizer using ``tokenize_uk``.

    Applies NFKC normalisation, then splits text into surface-form word tokens
    using Ukrainian orthographic rules (correctly handles apostrophe-containing
    words like ``м'яч``, clitics, etc.).  Standalone punctuation tokens
    emitted by ``tokenize_uk`` are dropped.

    Requires: ``tokenize-uk``
    """
    tok = _require_tokenize_uk()

    def tokenize(text: str) -> List[str]:
        return [
            t.lower()
            for t in tok.tokenize_words(_normalize_uk(text))
            if re.search(r"\w", t, re.UNICODE)
        ]

    return tokenize


def make_uk_sentence_splitter():
    """
    Sentence splitter using ``tokenize_uk.tokenize_sents``.

    Returns a callable ``(text: str) -> List[str]`` that applies NFKC
    normalisation then splits on linguistic sentence boundaries.  Pass to
    ``MRougeScorer`` as ``sentence_splitter`` so that rougeLsum works
    correctly on free-form model output (not just newline-delimited text).

    Requires: ``tokenize-uk``
    """
    tok = _require_tokenize_uk()

    def split(text: str) -> List[str]:
        return [s for s in tok.tokenize_sents(_normalize_uk(text)) if s.strip()]

    return split


def make_hf_tokenizer(hf_tokenizer: "PreTrainedTokenizerBase"):
    """
    Wrap a HuggingFace tokenizer for use with MRougeScorer.

    Strips SentencePiece / BPE space-prefix characters (▁, Ġ) and special
    tokens.  For Ukrainian, prefer ``make_uk_tokenizer()``.
    """
    special = set(hf_tokenizer.all_special_tokens)

    def tokenize(text: str) -> List[str]:
        return [
            t.lstrip("▁Ġ").lower()
            for t in hf_tokenizer.tokenize(text)
            if t not in special and t.lstrip("▁Ġ")
        ]

    return tokenize
