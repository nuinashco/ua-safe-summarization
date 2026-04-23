#!/usr/bin/env -S uv run python
"""Unsloth GRPO training entrypoint driven by a Hydra config."""

from __future__ import annotations

import logging
import os

import hydra
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from trl import GRPOConfig, GRPOTrainer

from safesum.training import configure_wandb, save_run_id, REWARD_REGISTRY
from safesum.training.model_utils import load_base_model

log = logging.getLogger(__name__)


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
    fns = []
    for reward_cfg in cfg.rewards:
        reward_cfg = OmegaConf.to_container(reward_cfg, resolve=True)
        key = reward_cfg.pop("key")
        fns.append(REWARD_REGISTRY[key](**reward_cfg))
    return fns


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
