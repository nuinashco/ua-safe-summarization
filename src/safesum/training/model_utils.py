"""Unsloth model loading shared across training scripts."""
from __future__ import annotations

import logging
import os

from omegaconf import DictConfig
from unsloth import FastModel

log = logging.getLogger(__name__)


def load_base_model(cfg: DictConfig):
    """Load Unsloth model + tokenizer from cfg.model.*"""
    log.info("Loading %s  full_finetuning=%s", cfg.model.name, cfg.model.full_finetuning)
    return FastModel.from_pretrained(
        model_name=cfg.model.name,
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=cfg.model.load_in_4bit,
        load_in_8bit=cfg.model.load_in_8bit,
        full_finetuning=cfg.model.full_finetuning,
        token=os.environ.get("HF_TOKEN"),
    )
