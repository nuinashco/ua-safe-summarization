from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams

log = logging.getLogger(__name__)


def dataset_name(dataset_path: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", dataset_path.lower().split("/")[-1])


def load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _deep_merge(base: dict, updates: dict) -> None:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def update_json(path: Path, updates: dict) -> None:
    data = load_json(path)
    _deep_merge(data, updates)
    save_json(path, data)


def build_prompts(ds, tokenizer, prompt_column: str) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in ds[prompt_column]
    ]


def run_vllm_inference(
    model_path: str,
    prompts: list[str],
    max_new_tokens: int = 256,
    gpu_memory_utilization: float = 0.9,
) -> list[str]:
    log.info(
        "Running vLLM inference from %s (gpu_memory_utilization=%.2f, max_new_tokens=%d)",
        model_path,
        gpu_memory_utilization,
        max_new_tokens,
    )
    llm = LLM(model=model_path, gpu_memory_utilization=gpu_memory_utilization)
    outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=max_new_tokens))
    return [o.outputs[0].text.strip() for o in outputs]


def load_dataset_split(path: str, split: str, num_samples: int | None = None):
    ds = load_dataset(path, split=split, token=os.environ.get("HF_TOKEN"))
    if num_samples and num_samples < len(ds):
        ds = ds.select(range(num_samples))
    log.info("Loaded '%s' split '%s': %d samples", path, split, len(ds))
    return ds


def get_ids(ds) -> list:
    return list(ds["id"]) if "id" in ds.column_names else list(range(len(ds)))
