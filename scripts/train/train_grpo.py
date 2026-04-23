#!/usr/bin/env -S uv run python
"""Unsloth GRPO training entrypoint driven by a Hydra config."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, List

import hydra
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import pipeline
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from trl import GRPOConfig, GRPOTrainer


from safesum.metrics import MRougeScorer, make_uk_tokenizer, make_uk_sentence_splitter
from safesum.training import configure_wandb, save_run_id
from safesum.training.model_utils import load_base_model

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reward helpers
# ---------------------------------------------------------------------------

def _extract_text(completions: List[Any]) -> List[str]:
    """Normalise TRL completion format (message list or plain string) to strings."""
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
    """Returns ROUGE-L F1 against the reference summary in [0, 1]."""

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


def _make_length_reward(min_words: int, max_words: int) -> Callable:
    """Soft reward in [-1, 1]: +1 inside range, linearly penalised outside."""
    span = max(max_words - min_words, 1)

    def length_reward(completions: List[Any], **_) -> List[float]:
        scores = []
        for text in _extract_text(completions):
            n = len(text.split())
            if min_words <= n <= max_words:
                scores.append(1.0)
            else:
                dist = max(min_words - n, n - max_words)
                scores.append(max(-1.0, 1.0 - dist / span))
        return scores

    return length_reward


# ---------------------------------------------------------------------------
# Main pipeline helpers
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../../configs", config_name="train_grpo")
def main(cfg: DictConfig) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    configure_wandb(cfg)

    model, tokenizer = _load_model(cfg)
    tokenizer = get_chat_template(tokenizer, chat_template=cfg.dataset.chat_template)
    train_ds = _load_dataset(cfg)
    reward_fns = _build_rewards(cfg)
    trainer = _build_trainer(cfg, model, tokenizer, train_ds, reward_fns)

    trainer.train()

    log.info("Saving model to %s", cfg.training.output_dir)
    trainer.save_model(cfg.training.output_dir)
    tokenizer.save_pretrained(cfg.training.output_dir)

    save_run_id(cfg)


def _load_model(cfg: DictConfig):
    model, tokenizer = load_base_model(cfg)
    lora = OmegaConf.to_container(cfg.lora, resolve=True)
    model = FastModel.get_peft_model(model, **lora)
    return model, tokenizer


def _load_dataset(cfg: DictConfig):
    log.info("Loading dataset %s  split=%s", cfg.dataset.path, cfg.dataset.split)
    ds = load_dataset(cfg.dataset.path, split=cfg.dataset.split, token=os.environ.get("HF_TOKEN"))
    log.info("Loaded %d examples  columns=%s", len(ds), ds.column_names)

    prompt_col = cfg.dataset.prompt_column
    summary_col = cfg.dataset.summary_column

    def _to_grpo_row(batch: dict) -> dict:
        prompts = [[{"role": "user", "content": p}] for p in batch[prompt_col]]
        return {"prompt": prompts, "summary": batch[summary_col]}

    return ds.map(
        _to_grpo_row,
        batched=True,
        num_proc=cfg.dataset.num_proc,
        remove_columns=ds.column_names,
        desc="Formatting for GRPO",
    )


def _build_rewards(cfg: DictConfig) -> list:
    return [
        ToxicityReward(reward_model=cfg.reward.toxicity_model),
        RougeReward(rouge_type=cfg.reward.rouge_type),
        _make_length_reward(cfg.reward.length_min, cfg.reward.length_max),
    ]


def _build_trainer(cfg: DictConfig, model, tokenizer, train_ds, reward_fns: list) -> GRPOTrainer:
    training_kwargs = OmegaConf.to_container(cfg.training, resolve=True)
    return GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fns,
        args=GRPOConfig(**training_kwargs),
        train_dataset=train_ds,
    )


if __name__ == "__main__":
    main()
