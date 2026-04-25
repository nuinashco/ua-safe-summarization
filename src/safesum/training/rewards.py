from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
        self.__name__ = "toxicity"
        self._tokenizer = AutoTokenizer.from_pretrained(self.reward_model)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.reward_model, torch_dtype=torch.float16, device_map="auto"
        ).eval()

    def __call__(self, completions: List[Any], **_) -> List[float]:
        texts = _extract_text(completions)
        device = next(self._model.parameters()).device
        inputs = self._tokenizer(
            texts, truncation=True, padding=True, return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            logits = self._model(**inputs).logits
        # LABEL_0 = non-toxic; softmax col 0 = p(non-toxic)
        return torch.softmax(logits, dim=-1)[:, 0].cpu().tolist()


@dataclass
class RougeReward:
    """Returns rougeLsum F1 against the reference summary in [0, 1]."""

    rouge_type: str = "rougeLsum"

    def __post_init__(self) -> None:
        self.__name__ = "rouge"
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
    """Length-window reward in [-1, 1].

    +1 inside [min_tokens, max_tokens]; strong linear penalty below min_tokens,
    softer linear penalty above max_tokens.
    """

    tokenizer: str
    min_tokens: int = 12
    max_tokens: int = 100

    def __post_init__(self) -> None:
        from transformers import AutoTokenizer
        self.__name__ = "length"
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)

    def __call__(self, completions: List[Any], **_) -> List[float]:
        scores = []
        for text in _extract_text(completions):
            n = len(self.tokenizer.encode(text, add_special_tokens=False))
            if self.min_tokens <= n <= self.max_tokens:
                score = 1.0
            elif n < self.min_tokens:
                score = -1.0 + 2.0 * n / max(self.min_tokens, 1)
            else:
                over = n - self.max_tokens
                score = 1.0 - 2.0 * over / max(self.max_tokens, 1)
                score = max(-1.0, score)
            scores.append(float(score))
        return scores


REWARD_REGISTRY = {
    "toxicity": lambda **kw: ToxicityReward(**kw),
    "rouge": lambda **kw: RougeReward(**kw),
    "length": lambda **kw: LengthReward(**kw),
}
