#!/usr/bin/env -S uv run python
"""Unsloth SFT training entrypoint driven by a Hydra config."""

from __future__ import annotations

import logging
import os

import hydra
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTConfig, SFTTrainer

from safesum.training import configure_wandb, save_run_id, RougeEvalCallback, VLLMEngine, VLLMManagerCallback
from safesum.training.model_utils import load_base_model

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="train_sft")
def main(cfg: DictConfig) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    configure_wandb(cfg)

    model, tokenizer = load_base_model(cfg)
    tokenizer = get_chat_template(tokenizer, chat_template=cfg.dataset.chat_template)
    train_ds, eval_ds = _load_dataset(cfg, tokenizer)
    trainer = _build_trainer(cfg, model, tokenizer, train_ds, eval_ds)

    trainer.train()

    log.info("Saving model to %s", cfg.training.output_dir)
    trainer.save_model(cfg.training.output_dir)
    tokenizer.save_pretrained(cfg.training.output_dir)

    save_run_id(cfg)


def _load_dataset(cfg: DictConfig, tokenizer):
    log.info("Loading dataset %s  split=%s", cfg.dataset.path, cfg.dataset.split)
    ds = load_dataset(cfg.dataset.path, split=cfg.dataset.split, token=os.environ.get("HF_TOKEN"))
    log.info("Loaded %d examples  columns=%s", len(ds), ds.column_names)
    ds = _format_for_chat(ds, tokenizer, cfg)

    val_split = cfg.dataset.get("val_split")
    if val_split:
        log.info("Loading eval split: %s", val_split)
        eval_ds = load_dataset(cfg.dataset.path, split=val_split, token=os.environ.get("HF_TOKEN"))
        eval_ds = _format_for_chat(eval_ds, tokenizer, cfg)
        log.info("Train/val: %d / %d", len(ds), len(eval_ds))
        return ds, eval_ds

    val_size = cfg.dataset.get("val_size")
    if val_size and val_size > 0:
        split = ds.train_test_split(test_size=val_size, seed=cfg.dataset.get("seed", 42))
        train_ds, eval_ds = split["train"], split["test"]
        log.info("Train/val split: %d / %d", len(train_ds), len(eval_ds))
        return train_ds, eval_ds

    return ds, None


def _format_for_chat(ds, tokenizer, cfg: DictConfig):
    prompt_col = cfg.dataset.prompt_column
    summary_col = cfg.dataset.summary_column
    text_field = cfg.training.dataset_text_field

    def _format(batch: dict) -> dict:
        texts = []
        for prompt, summary in zip(batch[prompt_col], batch[summary_col]):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": summary},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text.removeprefix("<bos>"))
        return {text_field: texts}

    return ds.map(
        _format,
        batched=True,
        num_proc=cfg.dataset.num_proc,
        remove_columns=ds.column_names,
        desc="Rendering chat template",
    )


def _build_trainer(cfg: DictConfig, model, tokenizer, train_ds, eval_ds):
    training_kwargs = OmegaConf.to_container(cfg.training, resolve=True)
    if eval_ds is None:
        training_kwargs.pop("eval_strategy", None)
        training_kwargs.pop("eval_steps", None)
        training_kwargs.pop("per_device_eval_batch_size", None)

    callbacks = []
    eval_cbs = []

    rouge_cfg = cfg.validation.get("rouge_callback")
    if rouge_cfg and rouge_cfg.get("enabled"):
        eval_cbs.append(RougeEvalCallback())

    if eval_cbs:
        engine_cfg = cfg.validation.get("vllm_engine", {})
        engine = VLLMEngine(
            gpu_memory_utilization=engine_cfg.get("gpu_memory_utilization", 0.5),
            max_model_len=cfg.model.get("max_seq_length", 2048),
        )
        callbacks.append(VLLMManagerCallback(engine, eval_cbs, cfg, tokenizer))
        log.info(
            "VLLMManagerCallback registered with %d eval callback(s): %s",
            len(eval_cbs),
            [type(cb).__name__ for cb in eval_cbs],
        )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(**training_kwargs),
        callbacks=callbacks or None,
    )

    if cfg.masking.train_on_responses_only:
        trainer = train_on_responses_only(
            trainer,
            instruction_part=cfg.masking.instruction_part,
            response_part=cfg.masking.response_part,
        )

    return trainer


if __name__ == "__main__":
    main()
