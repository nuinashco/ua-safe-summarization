from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from transformers import pipeline

from safesum.metrics import MRougeScorer, make_uk_tokenizer, make_uk_sentence_splitter


def _extract_text(completions: List[Any]) -> List[str]:
    out = []
    for c in completions:
        if isinstance(c, list):
            out.append(c[0]["content"] if c else "")
        else:
            out.append(str(c))
    return out


@dataclass
class ToxicityReward:
    """Returns p(non-toxic) in [0, 1] — higher is better."""

    reward_model: str = "textdetox/xlmr-large-toxicity-classifier-v2"

    def __post_init__(self) -> None:
        self.pipe = pipeline(
            task="text-classification",
            model=self.reward_model,
            device_map="auto",
        )

    def __call__(self, completions: List[Any], **_) -> List[float]:
        texts = _extract_text(completions)
        preds = self.pipe(texts, truncation=True, return_all_scores=True)
        return [
            float(next((s["score"] for s in row if s["label"] == "LABEL_0"), 0.0))
            for row in preds
        ]


@dataclass
class RougeReward:
    """Returns rougeLsum F1 against the reference summary in [0, 1]."""

    rouge_type: str = "rougeLsum"

    def __post_init__(self) -> None:
        self.scorer = MRougeScorer([self.rouge_type], make_uk_tokenizer(), make_uk_sentence_splitter())

    def __call__(self, completions: List[Any], summary: List[str] | None = None, **_) -> List[float]:
        texts = _extract_text(completions)
        if not summary:
            return [0.0] * len(texts)
        return [
            self.scorer.score(ref, pred)[self.rouge_type].fmeasure
            for ref, pred in zip(summary, texts)
        ]


@dataclass
class LengthReward:
    """Soft reward in [-1, 1]: +1 inside [min_tokens, max_tokens], linearly penalised outside."""

    tokenizer: str
    min_tokens: int = 50
    max_tokens: int = 400

    def __post_init__(self) -> None:
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        self.span = max(self.max_tokens - self.min_tokens, 1)

    def __call__(self, completions: List[Any], **_) -> List[float]:
        scores = []
        for text in _extract_text(completions):
            n = len(self.tokenizer.encode(text))
            if self.min_tokens <= n <= self.max_tokens:
                scores.append(1.0)
            else:
                dist = max(self.min_tokens - n, n - self.max_tokens)
                scores.append(max(-1.0, 1.0 - dist / self.span))
        return scores


REWARD_REGISTRY = {
    "toxicity": lambda **kw: ToxicityReward(**kw),
    "rouge": lambda **kw: RougeReward(**kw),
    "length": lambda **kw: LengthReward(**kw),
}
