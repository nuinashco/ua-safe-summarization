"""Shared vLLM engine for in-process generation during SFT training."""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm import LLM

log = logging.getLogger(__name__)

try:
    from vllm import LLM as _LLM
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False


class VLLMEngine:
    """Manages a single ``vllm.LLM`` instance shared across eval callbacks.

    Between checkpoint saves the engine sleeps at level 1 (weights offloaded
    to CPU, GPU free for the optimizer).  On each save the manager callback
    calls :meth:`wake_up`, :meth:`sync_weights`, runs all registered evals,
    then calls :meth:`sleep` — one wake/sync/sleep cycle regardless of how
    many eval callbacks are registered.

    Weight sync follows TRL GRPOTrainer colocate mode:
    ``LLM.apply_model(lambda m: m.load_weights(pairs))`` is the stable public
    API (available in vLLM ≥ 0.14) that runs the callable on the model inside
    each worker without exposing internal executor attributes.
    """

    def __init__(
        self,
        model_name: str,
        gpu_memory_utilization: float = 0.5,
        max_model_len: int = 2048,
    ) -> None:
        self._model_name = model_name
        self._gpu_mem_util = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._llm: LLM | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True if vLLM is installed in the current environment."""
        return _VLLM_AVAILABLE

    @property
    def is_initialised(self) -> bool:
        return self._llm is not None

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            raise RuntimeError("VLLMEngine has not been initialised; call init() first")
        return self._llm

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self, model) -> None:
        """Spin up the vLLM engine and immediately sleep it (weights → CPU).

        Args:
            model: The training model; its ``config._name_or_path`` is used as
                   the vLLM model identifier so the architecture is resolved
                   correctly even when ``model_name`` is a shorthand alias.
        """
        if not _VLLM_AVAILABLE:
            raise RuntimeError(
                "vLLM is not installed. Install it to use VLLMEngine."
            )
        model_name = (
            getattr(getattr(model, "config", None), "_name_or_path", None)
            or self._model_name
        )
        # vLLM 0.17+ runs EngineCore in a subprocess by default (SyncMPClient),
        # which serialises apply_model callables via msgpack and rejects lambdas
        # and GPU-tensor closures. InprocClient runs EngineCore in the same
        # process, so collective_rpc passes the callable directly — no pickle.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        log.info(
            "Initialising vLLM engine (model=%s, gpu_memory_utilization=%.2f, "
            "enable_sleep_mode=True, multiprocessing=off)",
            model_name,
            self._gpu_mem_util,
        )
        self._llm = _LLM(
            model=model_name,
            gpu_memory_utilization=self._gpu_mem_util,
            enable_sleep_mode=True,
            max_model_len=self._max_model_len,
        )
        self._llm.sleep(level=1)
        log.info("vLLM engine ready and offloaded to CPU (sleep level 1)")

    def wake_up(self) -> None:
        """Restore engine weights to GPU."""
        self.llm.wake_up()

    def sleep(self) -> None:
        """Offload engine weights to CPU (level 1)."""
        self.llm.sleep(level=1)

    def destroy(self) -> None:
        """Shut down the engine and destroy the NCCL process group it created."""
        if self._llm is not None:
            log.info("Destroying vLLM engine")
            del self._llm
            self._llm = None

        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
                log.info("Destroyed torch distributed process group")
        except Exception:
            log.debug("destroy_process_group skipped or failed", exc_info=True)

    def sync_weights(self, model) -> None:
        """Overwrite live vLLM weights with the current training-model weights.

        Collects ``(name, tensor)`` pairs from the training model and pushes
        them into the vLLM model via ``LLM.apply_model``.  The ``_orig_mod.``
        prefix is stripped to handle ``torch.compile``-wrapped models — same
        fix TRL applies in ``_fix_param_name_to_vllm``.

        Also resets the prefix cache so stale KV entries from the previous
        checkpoint do not affect generation.
        """
        weight_pairs = [
            (name.removeprefix("_orig_mod."), param.data)
            for name, param in model.named_parameters()
        ]
        self.llm.apply_model(lambda m: m.load_weights(iter(weight_pairs)))
        self.llm.reset_prefix_cache()
