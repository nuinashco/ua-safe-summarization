"""Training callbacks for SFT."""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

import wandb
from omegaconf import DictConfig
from transformers import TrainerCallback

from safesum.metrics import MRougeScorer, make_uk_sentence_splitter, make_uk_tokenizer
from safesum.utils.vllm_engine import VLLMEngine

log = logging.getLogger(__name__)

try:
    from vllm import SamplingParams
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class VLLMEvalCallback(ABC):
    """Protocol for any evaluation task that needs vLLM generation.

    Implement :meth:`setup` to load data once before training, and
    :meth:`evaluate` to run the actual evaluation while the engine is awake
    and weight-synced.  Register instances with :class:`VLLMManagerCallback`.

    Example::

        class BLEUEvalCallback(VLLMEvalCallback):
            def setup(self, cfg, tokenizer):
                ...  # build self._val_prompts / self._val_refs

            def evaluate(self, llm, step):
                outputs = llm.generate(self._val_prompts, ...)
                ...  # compute BLEU, log to wandb
    """

    @abstractmethod
    def setup(self, cfg: DictConfig, tokenizer) -> None:
        """Load data and build prompts.  Called once before training starts."""

    @abstractmethod
    def evaluate(self, llm, step: int) -> None:
        """Run evaluation using the awake, weight-synced vLLM instance."""


# ---------------------------------------------------------------------------
# ROUGE eval
# ---------------------------------------------------------------------------

class RougeEvalCallback(VLLMEvalCallback):
    """Macro-averaged ROUGE-1/2/L/Lsum on the Ukrainian validation split."""

    def __init__(self, val_dataset=None) -> None:
        self._val_dataset = val_dataset  # raw dataset passed from train_sft; avoids a second hub load
        self._val_prompts: list[str] | None = None
        self._val_refs: list[str] | None = None
        self._max_new_tokens: int = 128
        self._split: str = "validation"

    def setup(self, cfg: DictConfig, tokenizer) -> None:
        cb_cfg = cfg.eval_callback
        self._split = cb_cfg.split
        self._max_new_tokens = cb_cfg.get("max_new_tokens", 128)

        if self._val_dataset is not None:
            ds = self._val_dataset
        else:
            from datasets import load_dataset
            log.info("Loading val split '%s' for ROUGE eval", self._split)
            ds = load_dataset(
                cfg.dataset.path,
                split=self._split,
                token=os.environ.get("HF_TOKEN"),
            )

        num_samples = cb_cfg.get("num_samples")
        if num_samples and num_samples < len(ds):
            ds = ds.select(range(num_samples))
        log.info("Val split loaded: %d samples", len(ds))

        prompt_col = cfg.dataset.prompt_column
        summary_col = cfg.dataset.summary_column

        self._val_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in ds[prompt_col]
        ]
        self._val_refs = list(ds[summary_col])

    def evaluate(self, llm, step: int) -> None:
        sampling_params = SamplingParams(temperature=0, max_tokens=self._max_new_tokens)
        outputs = llm.generate(self._val_prompts, sampling_params, use_tqdm=False)
        predictions = [o.outputs[0].text.strip() for o in outputs]

        scorer = MRougeScorer(
            rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
            tokenizer=make_uk_tokenizer(),
            sentence_splitter=make_uk_sentence_splitter(),
        )
        corpus = scorer.score_corpus(self._val_refs, predictions)
        report = {f"eval/{k}": round(v.fmeasure * 100, 4) for k, v in corpus.items()}
        log.info("Step %d ROUGE: %s", step, report)

        if wandb.run is not None:
            wandb.log(report)


# ---------------------------------------------------------------------------
# Trainer callback that orchestrates the engine + all eval callbacks
# ---------------------------------------------------------------------------

class VLLMManagerCallback(TrainerCallback):
    """Orchestrates a shared :class:`VLLMEngine` across multiple eval callbacks.

    Owns the engine lifecycle: initialises it once in ``on_train_begin``, then
    on each ``on_evaluate`` performs a single wake → sync-weights → run-all-evals
    → sleep cycle.  Each registered :class:`VLLMEvalCallback` gets the awake,
    weight-synced ``llm`` object and is responsible only for generation and
    metric logging.

    Usage::

        engine = VLLMEngine(
            model_name=cfg.model.name,
            gpu_memory_utilization=cfg.eval_callback.vllm.gpu_memory_utilization,
            max_model_len=cfg.model.max_seq_length,
        )
        manager = VLLMManagerCallback(
            engine=engine,
            eval_callbacks=[RougeEvalCallback()],
            cfg=cfg,
            tokenizer=tokenizer,
        )
        trainer = SFTTrainer(..., callbacks=[manager])

    To add a new eval, implement :class:`VLLMEvalCallback` and append it to
    ``eval_callbacks`` — the engine is shared automatically.
    """

    def __init__(
        self,
        engine: VLLMEngine,
        eval_callbacks: list[VLLMEvalCallback],
        cfg: DictConfig,
        tokenizer,
    ) -> None:
        self._engine = engine
        self._eval_callbacks = eval_callbacks
        self._cfg = cfg
        self._tokenizer = tokenizer

    def on_train_begin(self, args, state, control, model=None, **kwargs) -> None:
        if not state.is_world_process_zero:
            return
        if not self._engine.available:
            log.error("vLLM not installed; VLLMManagerCallback is disabled")
            return

        for cb in self._eval_callbacks:
            cb.setup(self._cfg, self._tokenizer)

        self._engine.init(model)

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs) -> None:
        if not state.is_world_process_zero:
            return
        if not self._engine.is_initialised:
            return

        step = state.global_step
        log.info(
            "vLLM wake+sync for %d eval callback(s) at step %d",
            len(self._eval_callbacks),
            step,
        )
        self._engine.wake_up()
        try:
            self._engine.sync_weights(model)
            for cb in self._eval_callbacks:
                try:
                    cb.evaluate(self._engine.llm, step)
                except Exception:
                    log.exception(
                        "%s failed at step %d; continuing", type(cb).__name__, step
                    )
        finally:
            self._engine.sleep()
            log.info("vLLM offloaded back to CPU after step %d", step)

    def on_train_end(self, args, state, control, **kwargs) -> None:
        if not state.is_world_process_zero:
            return
        if self._engine.is_initialised:
            self._engine.destroy()
