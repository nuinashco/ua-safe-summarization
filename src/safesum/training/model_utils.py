"""Unsloth model loading shared across training scripts."""
from __future__ import annotations

import logging
import os

from omegaconf import DictConfig
from unsloth import FastModel

log = logging.getLogger(__name__)


def load_base_model(cfg: DictConfig):
    """Load Unsloth model + tokenizer from cfg.model.*

    Two notable flags:
      * ``cfg.model.fast_inference`` — attaches a colocated vLLM engine
        (``model.vllm_engine``) that GRPOTrainer's ``vllm_mode=colocate``
        consumes. Cannot be combined with ``full_finetuning=True`` —
        Unsloth requires LoRA when fast_inference is on.
      * ``cfg.model.lora`` — when present, wraps the model with PEFT LoRA
        via ``FastModel.get_peft_model`` after the base load.
    """
    log.info("Loading %s  full_finetuning=%s", cfg.model.name, cfg.model.full_finetuning)
    kwargs = dict(
        model_name=cfg.model.name,
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=cfg.model.load_in_4bit,
        load_in_8bit=cfg.model.load_in_8bit,
        full_finetuning=cfg.model.full_finetuning,
        token=os.environ.get("HF_TOKEN"),
    )
    lora_cfg = cfg.model.get("lora")
    if cfg.model.get("fast_inference"):
        kwargs["fast_inference"] = True
        kwargs["gpu_memory_utilization"] = cfg.model.get("gpu_memory_utilization", 0.5)
        if lora_cfg is not None:
            kwargs["max_lora_rank"] = lora_cfg.r
        log.info(
            "Unsloth fast_inference=True (vLLM gpu_memory_utilization=%s, max_lora_rank=%s)",
            kwargs["gpu_memory_utilization"],
            kwargs.get("max_lora_rank"),
        )

    model, tokenizer = FastModel.from_pretrained(**kwargs)

    if lora_cfg is not None:
        alpha = lora_cfg.get("alpha", lora_cfg.r)
        log.info("Wrapping with LoRA  r=%d  alpha=%d", lora_cfg.r, alpha)
        model = FastModel.get_peft_model(
            model,
            r=lora_cfg.r,
            lora_alpha=alpha,
            lora_dropout=lora_cfg.get("dropout", 0),
            bias=lora_cfg.get("bias", "none"),
            finetune_vision_layers=lora_cfg.get("finetune_vision_layers", False),
            finetune_language_layers=lora_cfg.get("finetune_language_layers", True),
            finetune_attention_modules=lora_cfg.get("finetune_attention_modules", True),
            finetune_mlp_modules=lora_cfg.get("finetune_mlp_modules", True),
            random_state=lora_cfg.get("random_state", 3407),
        )

    return model, tokenizer
