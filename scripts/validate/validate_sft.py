#!/usr/bin/env -S uv run python
"""Post-training ROUGE validation with vLLM — run after train_sft.py finishes.

Resumes the wandb run written to {output_dir}/wandb_run_id.txt so metrics
appear on the same training run. Override with wandb.run_id=<id> if needed.
Metrics are logged as {split}/rouge1 etc., so val and test runs stay separate.

Usage:
    python scripts/validate/validate_sft.py
    python scripts/validate/validate_sft.py validation.num_samples=500
    python scripts/validate/validate_sft.py validation.split=test
    python scripts/validate/validate_sft.py training.output_dir=outputs/my-run wandb.run_id=abc123
"""

from __future__ import annotations

import logging
import os

import hydra
import wandb
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from safesum.metrics import MRougeScorer, make_uk_sentence_splitter, make_uk_tokenizer
from safesum.training import resume_wandb

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="train_sft")
def main(cfg: DictConfig) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    val_cfg = cfg.get("validation", {})
    if not val_cfg.get("enabled", True):
        log.info("Validation disabled in config; exiting.")
        return

    split = val_cfg.split
    if not split:
        log.error("No split configured; set validation.split or dataset.val_split.")
        return

    resume_wandb(cfg)

    log.info("Loading split '%s'", split)
    ds = load_dataset(cfg.dataset.path, split=split, token=os.environ.get("HF_TOKEN"))
    num_samples = val_cfg.get("num_samples")
    if num_samples and num_samples < len(ds):
        ds = ds.select(range(num_samples))
    log.info("Split '%s' samples: %d", split, len(ds))

    prompt_col = cfg.dataset.prompt_column
    summary_col = cfg.dataset.summary_column
    max_new_tokens = val_cfg.get("max_new_tokens", 256)
    output_dir = cfg.training.output_dir

    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in ds[prompt_col]
    ]
    references: list[str] = list(ds[summary_col])

    log.info("Running vLLM inference from %s", output_dir)
    llm = LLM(model=output_dir)
    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens)
    outputs = llm.generate(prompts, sampling_params)
    predictions = [o.outputs[0].text.strip() for o in outputs]

    scorer = MRougeScorer(
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
        tokenizer=make_uk_tokenizer(),
        sentence_splitter=make_uk_sentence_splitter(),
    )
    corpus = scorer.score_corpus(references, predictions)
    report = {f"{split}/{k}": round(v.fmeasure * 100, 4) for k, v in corpus.items()}
    log.info("ROUGE: %s", report)

    if wandb.run is not None:
        wandb.log(report)
        wandb.finish()


if __name__ == "__main__":
    main()
