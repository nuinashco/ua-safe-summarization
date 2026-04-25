"""vLLM manager TrainerCallbacks that orchestrate generation + eval callbacks."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from omegaconf import DictConfig
from transformers import TrainerCallback

from safesum.utils.vllm_engine import VLLMEngine

from .vllm_callbacks import VLLMEvalCallback, generate_and_score

log = logging.getLogger(__name__)


class BaseVLLMManagerCallback(TrainerCallback, ABC):
    """Base manager: sets up eval callbacks, runs one generation pass per eval step.

    Subclasses implement :meth:`_acquire_llm` / :meth:`_release_llm` to handle
    their specific engine lifecycle (external engine vs. TRL-owned colocated engine).
    Optional hooks :meth:`_init_engine`, :meth:`_is_ready`, and
    :meth:`_teardown_engine` cover init/destroy if needed.
    """

    def __init__(
        self,
        eval_callbacks: list[VLLMEvalCallback],
        cfg: DictConfig,
        tokenizer,
    ) -> None:
        self._eval_callbacks = eval_callbacks
        self._cfg = cfg
        self._tokenizer = tokenizer

    # ------------------------------------------------------------------
    # TrainerCallback hooks
    # ------------------------------------------------------------------

    def on_train_begin(self, args, state, control, model=None, **kwargs) -> None:
        if not state.is_world_process_zero:
            return
        for cb in self._eval_callbacks:
            cb.setup(self._cfg, self._tokenizer)
        self._init_engine(model)

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs) -> None:
        if not state.is_world_process_zero:
            return
        if not self._is_ready():
            return
        step = state.global_step
        llm = self._acquire_llm(model)
        try:
            generate_and_score(llm, self._eval_callbacks, step, metrics)
        finally:
            self._release_llm(llm)

    def on_train_end(self, args, state, control, **kwargs) -> None:
        if not state.is_world_process_zero:
            return
        self._teardown_engine()

    # ------------------------------------------------------------------
    # Override points
    # ------------------------------------------------------------------

    def _init_engine(self, model) -> None:
        """Called in on_train_begin after eval callbacks are set up."""

    def _is_ready(self) -> bool:
        """Return False to skip on_evaluate (e.g. engine not initialised)."""
        return True

    @abstractmethod
    def _acquire_llm(self, model):
        """Return an awake, weight-synced vLLM LLM instance."""

    def _release_llm(self, llm) -> None:
        """Called in the finally block after generation + scoring."""

    def _teardown_engine(self) -> None:
        """Called in on_train_end."""


class VLLMManagerCallback(BaseVLLMManagerCallback):
    """Manages an external :class:`VLLMEngine` across multiple eval callbacks.

    Owns the engine lifecycle: initialises it once in ``on_train_begin``, then
    on each ``on_evaluate`` performs wake → sync-weights → generate-once →
    score-all → sleep.

    Usage::

        engine = VLLMEngine(
            model_name=cfg.model.name,
            gpu_memory_utilization=cfg.eval_callback.vllm.gpu_memory_utilization,
            max_model_len=cfg.model.max_seq_length,
        )
        manager = VLLMManagerCallback(
            engine=engine,
            eval_callbacks=[RougeEvalCallback(), ToxicityEvalCallback(...)],
            cfg=cfg,
            tokenizer=tokenizer,
        )
        trainer = SFTTrainer(..., callbacks=[manager])
    """

    def __init__(
        self,
        engine: VLLMEngine,
        eval_callbacks: list[VLLMEvalCallback],
        cfg: DictConfig,
        tokenizer,
    ) -> None:
        super().__init__(eval_callbacks, cfg, tokenizer)
        self._engine = engine

    def _init_engine(self, model) -> None:
        if not self._engine.available:
            log.error("vLLM not installed; VLLMManagerCallback is disabled")
            return
        self._engine.init(model)

    def _is_ready(self) -> bool:
        return self._engine.is_initialised

    def _acquire_llm(self, model):
        self._engine.wake_up()
        self._engine.sync_weights(model)
        return self._engine.llm

    def _release_llm(self, llm) -> None:
        self._engine.sleep()
        log.info("vLLM offloaded back to CPU")

    def _teardown_engine(self) -> None:
        if self._engine.is_initialised:
            self._engine.destroy()


class TRLVLLMManagerCallback(BaseVLLMManagerCallback):
    """Eval manager that reuses the ``vllm.LLM`` owned by GRPOTrainer.

    When ``use_vllm=True, vllm_mode='colocate'``, GRPOTrainer already owns a
    ``trainer.llm`` that is weight-synced before each rollout batch. This
    manager reuses that instance at eval time — no init, destroy, or
    sync_weights needed.

    Wake/sleep calls are skipped when ``vllm.sleep_mode: false`` — the engine
    is already awake and must not be put to sleep mid-eval.
    """

    def __init__(
        self,
        llm,
        eval_callbacks: list[VLLMEvalCallback],
        cfg: DictConfig,
        tokenizer,
    ) -> None:
        super().__init__(eval_callbacks, cfg, tokenizer)
        self._llm = llm
        self._sleep_mode: bool = cfg.get("vllm", {}).get("sleep_mode", False)

    def _acquire_llm(self, model):
        if self._sleep_mode:
            self._llm.wake_up()
        return self._llm

    def _release_llm(self, llm) -> None:
        if self._sleep_mode:
            self._llm.sleep(level=1)
