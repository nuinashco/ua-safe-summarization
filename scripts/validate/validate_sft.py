#!/usr/bin/env -S uv run python
"""Final ROUGE evaluation on an arbitrary split using a saved checkpoint.

Resumes the wandb run written to {output_dir}/wandb_run_id.txt so test-set
metrics appear on the same training run.  Override with wandb.run_id=<id>.

Usage:
    # test set evaluation after training (default: final_eval.split=test)
    python scripts/validate/validate_sft.py

    # different split or checkpoint
    python scripts/validate/validate_sft.py \
        final_eval.split=validation \
        training.output_dir=outputs/my-run \
        wandb.run_id=abc123
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

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

    fe_cfg = cfg.final_eval
    split = fe_cfg.split
    if not split:
        log.error("No split configured; set final_eval.split in the config.")
        return

    resume_wandb(cfg)

    log.info("Loading split '%s'", split)
    ds = load_dataset(cfg.dataset.path, split=split, token=os.environ.get("HF_TOKEN"))
    num_samples = fe_cfg.get("num_samples")
    if num_samples and num_samples < len(ds):
        ds = ds.select(range(num_samples))
    log.info("Split '%s': %d samples", split, len(ds))

    output_dir = cfg.training.output_dir
    max_new_tokens = fe_cfg.get("max_new_tokens", 256)
    gpu_mem_util = fe_cfg.vllm.get("gpu_memory_utilization", 0.9)

    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in ds[cfg.dataset.prompt_column]
    ]
    references: list[str] = list(ds[cfg.dataset.summary_column])

    log.info("Running vLLM inference from %s (gpu_memory_utilization=%.2f)", output_dir, gpu_mem_util)
    llm = LLM(model=output_dir, gpu_memory_utilization=gpu_mem_util)
    outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=max_new_tokens))
    predictions = [o.outputs[0].text.strip() for o in outputs]

    scorer = MRougeScorer(
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
        tokenizer=make_uk_tokenizer(),
        sentence_splitter=make_uk_sentence_splitter(),
    )
    corpus = scorer.score_corpus(references, predictions)
    report = {f"{split}/{k}": round(v.fmeasure * 100, 4) for k, v in corpus.items()}
    log.info("ROUGE: %s", report)

    exp_dir = Path(output_dir).parent
    results_dir = exp_dir / "results" / split
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    log.info("Metrics saved to %s", metrics_path)

    ids = ds["id"] if "id" in ds.column_names else list(range(len(ds)))
    samples_path = results_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as f:
        for sample_id, pred in zip(ids, predictions):
            f.write(json.dumps({"id": sample_id, "prediction": pred}, ensure_ascii=False) + "\n")
    log.info("Samples saved to %s", samples_path)

    if wandb.run is not None:
        wandb.log(report)
        wandb.finish()


if __name__ == "__main__":
    main()
