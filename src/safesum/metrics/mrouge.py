"""Multilingual ROUGE scorer with pluggable tokenization."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from safesum.metrics._text import lcs_length, lcs_ref_indices, ngrams, split_sentences


Tokenizer = Callable[[str], List[str]]


@dataclass(frozen=True)
class Score:
    precision: float
    recall: float
    fmeasure: float


def _f1(p: float, r: float) -> float:
    return 0.0 if p + r == 0.0 else 2 * p * r / (p + r)


def _prf(overlap: int, pred_total: int, ref_total: int) -> Score:
    if overlap == 0 or pred_total == 0 or ref_total == 0:
        return Score(0.0, 0.0, 0.0)
    p = overlap / pred_total
    r = overlap / ref_total
    return Score(p, r, _f1(p, r))


class MRougeScorer:
    """
    ROUGE scorer with language-agnostic tokenization.

    Supported variants:
        - ``rouge1`` .. ``rougeN`` — n-gram overlap F1
        - ``rougeL``               — sentence-level LCS F1
        - ``rougeLsum``            — summary-level union-LCS F1 (Lin 2004)

    Tokenization is delegated to the ``tokenizer`` callable. See
    ``safesum.utils.tokenizers`` for whitespace, Ukrainian, and
    HuggingFace-backed implementations.
    """

    def __init__(
        self,
        rouge_types: Iterable[str],
        tokenizer: Tokenizer,
        sentence_splitter: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        self.rouge_types: Tuple[str, ...] = tuple(rouge_types)
        self._tokenize = tokenizer
        self._split_sentences = sentence_splitter or split_sentences
        for rtype in self.rouge_types:
            self._n_for(rtype)  # validates

    def score(self, reference: str, prediction: str) -> Dict[str, Score]:
        """Score a single (reference, prediction) pair."""
        ref_tok = self._tokenize(reference or "")
        prd_tok = self._tokenize(prediction or "")
        return {
            rtype: self._score_variant(rtype, reference, prediction, ref_tok, prd_tok)
            for rtype in self.rouge_types
        }

    def score_corpus(
        self,
        references: Sequence[str],
        predictions: Sequence[str],
    ) -> Dict[str, Score]:
        """Macro-averaged scores across a parallel corpus."""
        if len(references) != len(predictions):
            raise ValueError("references and predictions must have the same length")
        if not references:
            return {k: Score(0.0, 0.0, 0.0) for k in self.rouge_types}

        sums = {k: [0.0, 0.0, 0.0] for k in self.rouge_types}
        for ref, prd in zip(references, predictions):
            for rtype, s in self.score(ref, prd).items():
                acc = sums[rtype]
                acc[0] += s.precision
                acc[1] += s.recall
                acc[2] += s.fmeasure
        n = len(references)
        return {k: Score(v[0] / n, v[1] / n, v[2] / n) for k, v in sums.items()}

    # ------------------------------------------------------------------
    # Variant dispatch
    # ------------------------------------------------------------------

    def _score_variant(
        self,
        rtype: str,
        reference: str,
        prediction: str,
        ref_tok: List[str],
        prd_tok: List[str],
    ) -> Score:
        if rtype == "rougeL":
            return self._score_l(ref_tok, prd_tok)
        if rtype == "rougeLsum":
            return self._score_lsum(reference, prediction)
        return self._score_n(ref_tok, prd_tok, self._n_for(rtype))

    @staticmethod
    def _n_for(rtype: str) -> int:
        """Return n for rougeN variants, or 0 for rougeL / rougeLsum."""
        if rtype in ("rougeL", "rougeLsum"):
            return 0
        if rtype.startswith("rouge") and rtype[5:].isdigit():
            n = int(rtype[5:])
            if n > 0:
                return n
        raise ValueError(f"Unsupported ROUGE type: {rtype!r}")

    @staticmethod
    def _score_n(ref_tok: List[str], prd_tok: List[str], n: int) -> Score:
        ref_ng = ngrams(ref_tok, n)
        prd_ng = ngrams(prd_tok, n)
        overlap = sum(min(c, prd_ng.get(g, 0)) for g, c in ref_ng.items())
        return _prf(overlap, sum(prd_ng.values()), sum(ref_ng.values()))

    @staticmethod
    def _score_l(ref_tok: List[str], prd_tok: List[str]) -> Score:
        return _prf(lcs_length(ref_tok, prd_tok), len(prd_tok), len(ref_tok))

    def _score_lsum(self, reference: str, prediction: str) -> Score:
        """
        Summary-level LCS (Lin 2004), matching the XL-Sum / ROUGE-1.5.5 algorithm.

        For each reference sentence:
          1. Compute LCS against every candidate sentence; take the union of
             matched *reference* token indices.
          2. Walk the resulting reference tokens and credit a hit only when
             the token is still available in both the reference and candidate
             token-count budgets (prevents double-counting repeated tokens).
        """
        ref_sents = [self._tokenize(s) for s in self._split_sentences(reference or "")]
        prd_sents = [self._tokenize(s) for s in self._split_sentences(prediction or "")]
        m = sum(len(s) for s in ref_sents)
        n = sum(len(s) for s in prd_sents)
        if not m or not n:
            return Score(0.0, 0.0, 0.0)

        token_cnts_r = Counter(t for s in ref_sents for t in s)
        token_cnts_c = Counter(t for s in prd_sents for t in s)

        hits = 0
        for ref_sent in ref_sents:
            # Union of ref indices matched by any candidate sentence
            union_idx = sorted(set().union(*(lcs_ref_indices(ref_sent, prd) for prd in prd_sents)))
            for t in (ref_sent[i] for i in union_idx):
                if token_cnts_c[t] > 0 and token_cnts_r[t] > 0:
                    hits += 1
                    token_cnts_c[t] -= 1
                    token_cnts_r[t] -= 1

        if not hits:
            return Score(0.0, 0.0, 0.0)
        p = hits / n
        r = hits / m
        return Score(p, r, _f1(p, r))


def rouge_report(
    predictions: Sequence[str],
    references: Sequence[str],
    tokenizer: Tokenizer,
    rouge_types: Tuple[str, ...] = ("rouge1", "rouge2", "rougeL", "rougeLsum"),
    as_percent: bool = True,
) -> Dict[str, float]:
    """Convenience wrapper: macro-averaged F1 per ROUGE type."""
    scorer = MRougeScorer(rouge_types, tokenizer)
    corpus = scorer.score_corpus(references, predictions)
    scale = 100.0 if as_percent else 1.0
    return {k: s.fmeasure * scale for k, s in corpus.items()}
