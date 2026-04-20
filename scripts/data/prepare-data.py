#!/usr/bin/env -S uv run python
import logging
from pathlib import Path

import hydra
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from safesum.utils.data import add_prompt_column, truncate_dataset_column

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="prepare_data")
def main(cfg: DictConfig) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    dataset = _load_dataset(cfg)
    dataset = _truncate_columns(dataset, cfg)
    dataset = _add_prompt_column(dataset, cfg)
    _save(dataset, cfg)


def _load_dataset(cfg: DictConfig) -> Dataset:
    from datasets import load_dataset

    kwargs: dict = {"path": cfg.dataset.path, "split": cfg.dataset.split}
    if cfg.dataset.get("name"):
        kwargs["name"] = cfg.dataset.name

    log.info("Loading %s  split=%s", cfg.dataset.path, cfg.dataset.split)
    dataset = load_dataset(**kwargs)
    log.info("Loaded %d examples", len(dataset))
    return dataset


def _truncate_columns(dataset: Dataset, cfg: DictConfig) -> Dataset:
    for trunc in cfg.get("truncation", []):
        log.info("Truncating '%s' → %d tokens  tokenizer=%s", trunc.column, trunc.max_tokens, trunc.tokenizer)
        tokenizer = AutoTokenizer.from_pretrained(trunc.tokenizer)
        dataset = truncate_dataset_column(dataset, trunc.column, tokenizer, trunc.max_tokens)
    return dataset


def _add_prompt_column(dataset: Dataset, cfg: DictConfig) -> Dataset:
    if not cfg.get("prompt"):
        return dataset
    log.info("Adding prompt column '%s'", cfg.prompt.output_column)
    return add_prompt_column(dataset, cfg.prompt.template, cfg.prompt.output_column, cfg.output.get("num_proc", 1))


def _save(dataset: Dataset, cfg: DictConfig) -> None:
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = cfg.output.format
    filename = cfg.output.get("filename") or f"{cfg.dataset.split}.{fmt}"
    out_path = out_dir / filename

    log.info("Saving to %s", out_path)
    match fmt:
        case "parquet":
            dataset.to_parquet(str(out_path))
        case "json":
            dataset.to_json(str(out_path))
        case "jsonl":
            dataset.to_json(str(out_path), lines=True)
        case "csv":
            dataset.to_csv(str(out_path))
        case _:
            raise ValueError(f"Unsupported format: {fmt!r}. Use parquet | json | jsonl | csv")

    log.info("Done → %s  (%d rows)", out_path, len(dataset))


if __name__ == "__main__":
    main()
