#!/usr/bin/env -S uv run python
"""Unsloth GRPO training entrypoint driven by a Hydra config."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch._dynamo
import hydra
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from trl import GRPOConfig, GRPOTrainer

from safesum.training import (
    configure_wandb,
    save_run_id,
    REWARD_REGISTRY,
    build_eval_callbacks,
    VLLMEngine,
    VLLMManagerCallback,
    TRLVLLMManagerCallback,
)
from safesum.training.model_utils import load_base_model

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../../configs", config_name="train_grpo")
def main(cfg: DictConfig) -> None:
    torch._dynamo.config.cache_size_limit = 2048
    torch._dynamo.config.recompile_limit = 2048
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    configure_wandb(cfg)

    model, tokenizer = load_base_model(cfg)
    tokenizer = get_chat_template(tokenizer, chat_template=cfg.dataset.chat_template)

    train_ds = _load_train_dataset(cfg)
    sft_raw_ds = _load_sft_eval_dataset(cfg)
    reward_fns, reward_weights, rewards_by_type = _build_rewards(cfg)

    grpo_eval_ds = _load_grpo_eval_dataset(cfg)

    trainer = _build_trainer(cfg, model, tokenizer, train_ds, reward_fns, reward_weights, eval_ds=grpo_eval_ds)
    _attach_eval_callback(trainer, cfg, tokenizer, sft_raw_ds)

    trainer.train()

    log.info("Saving model to %s", cfg.training.output_dir)
    trainer.save_model(cfg.training.output_dir)
    tokenizer.save_pretrained(cfg.training.output_dir)

    save_run_id(cfg)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _load_train_dataset(cfg: DictConfig):
    """Load the toxicity-rewrite training set, formatted for GRPO (no summary refs)."""
    log.info("Loading GRPO train dataset %s  split=%s", cfg.dataset.path, cfg.dataset.split)
    ds = load_dataset(cfg.dataset.path, split=cfg.dataset.split, token=os.environ.get("HF_TOKEN"))
    log.info("Loaded %d examples  columns=%s", len(ds), ds.column_names)
    return _format_for_grpo(ds, cfg.dataset.prompt_column, None, cfg)


def _load_grpo_eval_dataset(cfg: DictConfig):
    """Load the GRPO dataset validation split for TRL's reward/KL eval loop."""
    split = cfg.dataset.get("eval_split", "validation")
    log.info("Loading GRPO eval dataset %s  split=%s", cfg.dataset.path, split)
    ds = load_dataset(cfg.dataset.path, split=split, token=os.environ.get("HF_TOKEN"))
    n = cfg.dataset.get("eval_num_samples")
    if n and n < len(ds):
        ds = ds.select(range(n))
    log.info("GRPO eval dataset: %d samples", len(ds))
    return _format_for_grpo(ds, cfg.dataset.prompt_column, None, cfg)


def _load_sft_eval_dataset(cfg: DictConfig):
    """Load SFT validation split for eval callbacks (ROUGE, toxicity).

    Returns raw dataset only — GRPOTrainer's eval_dataset is a slice of the
    GRPO training set so TRL reward/KL metrics stay on the training distribution.
    """
    sft_cfg = cfg.sft_dataset
    log.info("Loading SFT eval dataset %s  split=%s", sft_cfg.path, sft_cfg.split)
    raw_ds = load_dataset(sft_cfg.path, split=sft_cfg.split, token=os.environ.get("HF_TOKEN"))

    n = sft_cfg.get("num_samples")
    if n and n < len(raw_ds):
        raw_ds = raw_ds.select(range(n))
    log.info("SFT eval dataset: %d samples", len(raw_ds))
    return raw_ds


def _format_for_grpo(ds, prompt_col: str, summary_col: str | None, cfg: DictConfig):
    """Convert a dataset to GRPO format: prompts (+ optional summary refs)."""
    has_summary = summary_col is not None

    def _to_grpo_row(batch: dict) -> dict:
        prompts = [[{"role": "user", "content": p}] for p in batch[prompt_col]]
        out: dict = {"prompt": prompts}
        if has_summary:
            out["summary"] = batch[summary_col]
        return out

    return ds.map(
        _to_grpo_row,
        batched=True,
        num_proc=cfg.dataset.num_proc,
        remove_columns=ds.column_names,
        desc="Formatting for GRPO",
    )


# ---------------------------------------------------------------------------
# Reward helpers
# ---------------------------------------------------------------------------

def _build_rewards(cfg: DictConfig) -> tuple[list, list[float], dict]:
    """Parse rewards config ({type, weight, params}) → (reward_fns, weights, by_type)."""
    fns, weights, by_type = [], [], {}
    for reward_cfg in cfg.rewards:
        reward_cfg = OmegaConf.to_container(reward_cfg, resolve=True)
        rtype = reward_cfg["type"]
        weight = float(reward_cfg.get("weight", 1.0))
        params = reward_cfg.get("params", {})
        fn = REWARD_REGISTRY[rtype](**params)
        fns.append(fn)
        weights.append(weight)
        by_type[rtype] = fn
    return fns, weights, by_type


# ---------------------------------------------------------------------------
# Callback helpers
# ---------------------------------------------------------------------------

def _attach_eval_callback(trainer, cfg: DictConfig, tokenizer, sft_raw_ds) -> None:
    """Attach vLLM eval callbacks after the trainer is built.

    With use_vllm=True + colocate mode, trainer.llm already exists and is
    weight-synced before each rollout batch, so we reuse it (TRLVLLMManagerCallback).
    Otherwise we spin up a separate VLLMEngine (VLLMManagerCallback).
    """
    cb_cfg = cfg.get("eval_callbacks", {})
    if not cb_cfg:
        return

    eval_callbacks = build_eval_callbacks(cb_cfg, val_dataset=sft_raw_ds)
    if not eval_callbacks:
        return

    if getattr(trainer, "llm", None) is not None:
        callback = TRLVLLMManagerCallback(trainer.llm, eval_callbacks, cb_cfg, tokenizer)
        log.info("Attached TRLVLLMManagerCallback (reusing colocated vLLM)")
    else:
        engine = VLLMEngine(
            model_name=cfg.model.name,
            gpu_memory_utilization=0.5,
            max_model_len=cfg.model.max_seq_length,
        )
        callback = VLLMManagerCallback(engine, eval_callbacks, cb_cfg, tokenizer)
        log.info("Attached VLLMManagerCallback (separate VLLMEngine)")

    trainer.add_callback(callback)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def _build_trainer(
    cfg: DictConfig,
    model,
    tokenizer,
    train_ds,
    reward_fns: list,
    reward_weights: list[float],
    eval_ds=None,
) -> GRPOTrainer:
    training_kwargs = OmegaConf.to_container(cfg.training, resolve=True)
    training_kwargs["reward_weights"] = reward_weights

    if eval_ds is None:
        training_kwargs.pop("eval_strategy", None)
        training_kwargs.pop("eval_steps", None)
        training_kwargs.pop("per_device_eval_batch_size", None)

    return GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fns,
        args=GRPOConfig(**training_kwargs),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )


if __name__ == "__main__":
    main()
