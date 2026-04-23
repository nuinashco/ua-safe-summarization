from __future__ import annotations

from typing import TYPE_CHECKING

from transformers import AutoTokenizer

if TYPE_CHECKING:
    from datasets import Dataset


def truncate_to_tokens(text: str, tokenizer: AutoTokenizer, max_tokens: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def _apply_truncation(example: dict, column: str, tokenizer: AutoTokenizer, max_tokens: int) -> dict:
    return {column: truncate_to_tokens(example[column], tokenizer, max_tokens)}


def truncate_dataset_column(
    dataset: Dataset,
    column: str,
    tokenizer: AutoTokenizer,
    max_tokens: int,
) -> Dataset:
    """Truncate one dataset column to ``max_tokens`` tokens in-place via map."""
    return dataset.map(
        _apply_truncation,
        fn_kwargs={"column": column, "tokenizer": tokenizer, "max_tokens": max_tokens},
        num_proc=1,  # tokenizers are not fork-safe
        desc=f"Truncating {column}",
    )


def _build_prompt(example: dict, template: str, output_column: str) -> dict:
    return {output_column: template.format_map(example)}


def add_prompt_column(
    dataset: Dataset,
    template: str,
    output_column: str,
    num_proc: int = 1,
) -> Dataset:
    """Add a new column by rendering ``template`` against each example row."""
    return dataset.map(
        _build_prompt,
        fn_kwargs={"template": template, "output_column": output_column},
        num_proc=num_proc,
        desc="Building prompts",
    )
