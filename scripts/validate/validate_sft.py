#!/usr/bin/env -S uv run python
"""ROUGE evaluation on an SFT dataset split using a saved checkpoint.

Results are written to:
  {results_dir}/metrics.json   – merged with any existing metrics
  {results_dir}/samples.json   – predictions keyed by dataset name; existing entries are reused

Usage:
    python scripts/validate/validate_sft.py \\
        --model outputs/gemma3-1b-sft/checkpoints \\
        --split test

    python scripts/validate/validate_sft.py \\
        --model outputs/gemma3-1b-sft/checkpoints \\
        --dataset nuinashco/xlsum-ua-processed \\
        --split validation \\
        --num-samples 500
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from transformers import AutoTokenizer

from safesum.metrics import MRougeScorer, make_uk_sentence_splitter, make_uk_tokenizer
from safesum.validation import (
    build_prompts,
    dataset_key,
    get_ids,
    load_dataset_split,
    load_json,
    run_vllm_inference,
    save_json,
    update_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ROUGE evaluation for SFT checkpoint")
    p.add_argument("--model", required=True, help="Path to model checkpoint")
    p.add_argument("--dataset", default="nuinashco/xlsum-ua-processed")
    p.add_argument("--split", default="test")
    p.add_argument("--prompt-column", default="prompt")
    p.add_argument("--summary-column", default="summary")
    p.add_argument("--results-dir", default=None, help="Defaults to {model}/../results")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--num-samples", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else Path(args.model).parent / "results"
    metrics_path = results_dir / "metrics.json"
    samples_path = results_dir / "samples.json"

    ds = load_dataset_split(args.dataset, args.split, args.num_samples)
    references: list[str] = list(ds[args.summary_column])
    key = dataset_key(args.dataset, args.split)

    samples = load_json(samples_path)
    if key in samples:
        log.info("Reusing existing predictions for '%s'", key)
        predictions = [s["prediction"] for s in samples[key]]
    else:
        log.info("Generating predictions for '%s'", key)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        prompts = build_prompts(ds, tokenizer, args.prompt_column)
        predictions = run_vllm_inference(
            args.model, prompts, args.max_new_tokens, args.gpu_memory_utilization
        )
        ids = get_ids(ds)
        samples[key] = [{"id": sid, "prediction": pred} for sid, pred in zip(ids, predictions)]
        save_json(samples_path, samples)
        log.info("Samples saved to %s", samples_path)

    scorer = MRougeScorer(
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
        tokenizer=make_uk_tokenizer(),
        sentence_splitter=make_uk_sentence_splitter(),
    )
    corpus = scorer.score_corpus(references, predictions)
    report = {f"{key}/{k}": round(v.fmeasure * 100, 4) for k, v in corpus.items()}
    log.info("ROUGE: %s", report)

    update_json(metrics_path, report)
    log.info("Metrics saved to %s", metrics_path)


if __name__ == "__main__":
    main()
