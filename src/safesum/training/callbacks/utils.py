"""Utility functions for building eval callbacks from config."""
from __future__ import annotations

import logging

from omegaconf import DictConfig, OmegaConf

from .vllm_callbacks import EVAL_CALLBACK_REGISTRY, VLLMEvalCallback

log = logging.getLogger(__name__)


def build_eval_callbacks(cb_cfg: DictConfig, val_dataset=None) -> list[VLLMEvalCallback]:
    """Instantiate eval callbacks from config.

    Each entry in ``cb_cfg.callbacks`` must have a ``type`` key matching
    :data:`EVAL_CALLBACK_REGISTRY`.  Optional ``params`` are forwarded to the
    constructor.  ``val_dataset``, when provided, is injected so callbacks skip
    their own dataset loading.
    """
    callbacks = []
    for entry in cb_cfg.get("callbacks", []):
        entry = OmegaConf.to_container(entry, resolve=True)
        ctype = entry["type"]
        if ctype not in EVAL_CALLBACK_REGISTRY:
            log.warning("Unknown eval callback type %r; skipping", ctype)
            continue
        kwargs = entry.get("params", {})
        if val_dataset is not None:
            kwargs["val_dataset"] = val_dataset
        callbacks.append(EVAL_CALLBACK_REGISTRY[ctype](**kwargs))
    return callbacks
