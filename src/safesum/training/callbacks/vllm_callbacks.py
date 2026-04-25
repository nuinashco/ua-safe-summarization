"""vLLM-backed evaluation callbacks (ROUGE, toxicity, …)."""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

import wandb
from omegaconf import DictConfig

from safesum.metrics import MRougeScorer, make_uk_sentence_splitter, make_uk_tokenizer

try:
    from vllm import SamplingParams
except ImportError:
    pass

log = logging.getLogger(__name__)


class VLLMEvalCallback(ABC):
    """Protocol for any evaluation task that needs vLLM generation.

    Implement :meth:`setup` to load data once before training.
    Implement :meth:`prompts` to return the prompts for this callback.
    Implement :meth:`score` to compute metrics from pre-generated completions.

    The manager generates once across all registered callbacks (deduplicating
    shared prompts) and routes completions to each :meth:`score`.
    """

    @abstractmethod
    def setup(self, cfg: DictConfig, tokenizer) -> None:
        """Load data and build prompts. Called once before training starts."""

    @property
    @abstractmethod
    def prompts(self) -> list[str]:
        """Return the prompts this callback needs generated."""

    @property
    def max_new_tokens(self) -> int:
        return getattr(self, "_max_new_tokens", 128)

    @abstractmethod
    def score(self, completions: list[str], step: int) -> dict:
        """Compute and log metrics from pre-generated completions."""


def generate_and_score(llm, eval_callbacks: list[VLLMEvalCallback], step: int, metrics: dict | None) -> None:
    """Collect prompts from all callbacks, generate once (deduped), score each."""
    prompt_to_idx: dict[str, int] = {}
    for cb in eval_callbacks:
        for p in cb.prompts:
            if p not in prompt_to_idx:
                prompt_to_idx[p] = len(prompt_to_idx)

    all_prompts = list(prompt_to_idx.keys())
    max_new_tokens = max((cb.max_new_tokens for cb in eval_callbacks), default=128)
    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens)

    log.info(
        "Generating %d unique prompt(s) for %d eval callback(s) at step %d",
        len(all_prompts), len(eval_callbacks), step,
    )
    outputs = llm.generate(all_prompts, sampling_params, use_tqdm=True)
    completions = [o.outputs[0].text.strip() for o in outputs]

    for cb in eval_callbacks:
        try:
            cb_completions = [completions[prompt_to_idx[p]] for p in cb.prompts]
            result = cb.score(cb_completions, step)
            if result and metrics is not None:
                metrics.update({k.replace("/", "_"): v for k, v in result.items()})
        except Exception:
            log.exception("%s failed at step %d; continuing", type(cb).__name__, step)


class RougeEvalCallback(VLLMEvalCallback):
    """Macro-averaged ROUGE-1/2/L/Lsum on the Ukrainian validation split."""

    def __init__(self, val_dataset=None) -> None:
        self._val_dataset = val_dataset
        self._val_prompts: list[str] | None = None
        self._val_refs: list[str] | None = None
        self._max_new_tokens: int = 128

    def setup(self, cfg: DictConfig, tokenizer) -> None:
        ds_cfg = cfg.dataset
        self._max_new_tokens = cfg.get("vllm", {}).get("max_new_tokens", 128)

        if self._val_dataset is not None:
            ds = self._val_dataset
        else:
            from datasets import load_dataset
            log.info("Loading val split '%s' for ROUGE eval", ds_cfg.split)
            ds = load_dataset(
                ds_cfg.path,
                split=ds_cfg.split,
                token=os.environ.get("HF_TOKEN"),
            )

        num_samples = cfg.dataset.get("num_samples")
        if num_samples and num_samples < len(ds):
            ds = ds.select(range(num_samples))
        log.info("ROUGE eval dataset: %d samples", len(ds))

        self._val_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in ds[ds_cfg.prompt_column]
        ]
        self._val_refs = list(ds[ds_cfg.summary_column])

    @property
    def prompts(self) -> list[str]:
        return self._val_prompts

    def score(self, completions: list[str], step: int) -> dict:
        scorer = MRougeScorer(
            rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
            tokenizer=make_uk_tokenizer(),
            sentence_splitter=make_uk_sentence_splitter(),
        )
        corpus = scorer.score_corpus(self._val_refs, completions)
        report = {f"eval/{k}": round(v.fmeasure * 100, 4) for k, v in corpus.items()}
        log.info("Step %d ROUGE: %s", step, report)
        if wandb.run is not None:
            wandb.log(report)
        return report


class ToxicityEvalCallback(VLLMEvalCallback):
    """Mean p(non-toxic) of greedy rollouts on the configured eval dataset.

    Note: TRL's GRPOTrainer also logs the toxicity reward when ``eval_dataset``
    is set, but it triggers an HF generate pass and a separate compute path.
    This callback piggy-backs on the same vLLM batch as ROUGE for free, and
    adds flagged-ratio + completion-length sanity metrics.
    """

    _DEFAULT_REWARD_MODEL = "textdetox/xlmr-large-toxicity-classifier-v2"

    def __init__(
        self,
        reward_model: str = _DEFAULT_REWARD_MODEL,
        val_dataset=None,
    ) -> None:
        from safesum.training.rewards import ToxicityReward
        self._val_dataset = val_dataset
        self._reward_fn = ToxicityReward(reward_model=reward_model)
        self._prompts: list[str] | None = None
        self._max_new_tokens: int = 128

    def setup(self, cfg: DictConfig, tokenizer) -> None:
        ds_cfg = cfg.dataset
        self._max_new_tokens = cfg.get("vllm", {}).get("max_new_tokens", 128)

        if self._val_dataset is not None:
            ds = self._val_dataset
        else:
            from datasets import load_dataset
            log.info("Loading val split '%s' for toxicity eval", ds_cfg.split)
            ds = load_dataset(
                ds_cfg.path,
                split=ds_cfg.split,
                token=os.environ.get("HF_TOKEN"),
            )

        n = cfg.get("num_samples")
        if n and n < len(ds):
            ds = ds.select(range(n))
        log.info("Toxicity eval dataset: %d samples", len(ds))

        self._prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in ds[ds_cfg.prompt_column]
        ]

    @property
    def prompts(self) -> list[str]:
        return self._prompts

    def score(self, completions: list[str], step: int) -> dict:
        scores = self._reward_fn([[{"role": "assistant", "content": t}] for t in completions])
        word_lens = [len(t.split()) for t in completions]

        n = max(len(scores), 1)
        report = {
            "eval/tox_p_non_toxic_mean": round(sum(scores) / n, 4),
            "eval/tox_flagged_ratio": round(sum(1 for s in scores if s < 0.5) / n, 4),
            "eval/completion_word_len_mean": round(sum(word_lens) / n, 2),
        }
        log.info("Step %d toxicity: %s", step, report)
        if wandb.run is not None:
            wandb.log(report)
        return report


EVAL_CALLBACK_REGISTRY: dict[str, type[VLLMEvalCallback]] = {
    "rouge": RougeEvalCallback,
    "toxicity": ToxicityEvalCallback,
}
