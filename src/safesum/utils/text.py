"""Generic text and sequence utilities shared by metrics."""
from __future__ import annotations

from collections import Counter
from typing import List, Sequence, Tuple


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences on newlines.

    Matches the convention used by the XL-Sum multilingual rouge scorer.
    If no newlines are present the whole text is returned as a single sentence.
    """
    sents = [s.strip() for s in text.split("\n") if s.strip()]
    return sents if sents else [text]


def ngrams(tokens: Sequence[str], n: int) -> Counter[Tuple[str, ...]]:
    """Count all n-grams in ``tokens`` as tuples."""
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """
    Length of the Longest Common Subsequence of ``a`` and ``b``.

    Runs in O(|a| * |b|) time and O(|b|) memory using the rolling-row
    optimisation of the standard DP.
    """
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ai in a:
        curr = [0] * (len(b) + 1)
        for j, bj in enumerate(b, 1):
            curr[j] = prev[j - 1] + 1 if ai == bj else max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def lcs_ref_indices(ref: Sequence[str], prd: Sequence[str]) -> frozenset:
    """
    Return the set of ``ref``-token indices that participate in an LCS of
    ``ref`` and ``prd``.

    Used by ROUGE-Lsum to build the union of matched reference positions
    across all candidate sentences (XL-Sum / ROUGE-1.5.5 algorithm).
    Requires the full O(|ref|*|prd|) DP table for backtracking.
    """
    m, n = len(ref), len(prd)
    if not m or not n:
        return frozenset()
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i, ri in enumerate(ref, 1):
        for j, pj in enumerate(prd, 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if ri == pj else max(dp[i - 1][j], dp[i][j - 1])
    matched: set = set()
    i, j = m, n
    while i > 0 and j > 0:
        if ref[i - 1] == prd[j - 1] and dp[i][j] == dp[i - 1][j - 1] + 1:
            matched.add(i - 1)  # ref index
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return frozenset(matched)
