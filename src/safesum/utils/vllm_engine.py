"""Shared vLLM engine for in-process generation during SFT training."""
from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator

if TYPE_CHECKING:
    from vllm import LLM

log = logging.getLogger(__name__)

try:
    from vllm import LLM as _LLM
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False


class VLLMEngine:
    """Provides a fresh ``vllm.LLM`` instance scoped to each checkpoint eval.

    vLLM 0.17+ runs its engine core in a separate process (visible as
    ``EngineCore_DP0`` in logs).  In-memory weight sync via ``apply_model``
    requires serialising a closure + GPU tensors over ZMQ IPC, which vLLM's
    default serialiser rejects.  Loading directly from the checkpoint on disk
    avoids IPC entirely and is simpler: the checkpoint is already written
    before ``on_save`` fires.

    Usage::

        with engine.for_checkpoint("/path/to/checkpoint-500") as llm:
            outputs = llm.generate(prompts, sampling_params)
            # llm is destroyed and GPU memory freed on context exit
    """

    def __init__(
        self,
        gpu_memory_utilization: float = 0.5,
        max_model_len: int = 2048,
    ) -> None:
        self._gpu_mem_util = gpu_memory_utilization
        self._max_model_len = max_model_len

    @property
    def available(self) -> bool:
        """True if vLLM is installed in the current environment."""
        return _VLLM_AVAILABLE

    @contextmanager
    def for_checkpoint(self, ckpt_path: str) -> Generator[LLM, None, None]:
        """Create a vLLM LLM from *ckpt_path*, yield it, then free GPU memory.

        Args:
            ckpt_path: Path to a HuggingFace checkpoint directory (must
                       contain ``config.json`` and model weights).
        """
        if not _VLLM_AVAILABLE:
            raise RuntimeError("vLLM is not installed")

        import torch

        log.info(
            "Initialising vLLM from checkpoint %s (gpu_memory_utilization=%.2f)",
            ckpt_path,
            self._gpu_mem_util,
        )
        llm = _LLM(
            model=ckpt_path,
            gpu_memory_utilization=self._gpu_mem_util,
            max_model_len=self._max_model_len,
        )
        try:
            yield llm
        finally:
            del llm
            gc.collect()
            torch.cuda.empty_cache()
            log.info("vLLM instance freed")
