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

    def setup(self, cfg: DictConfig, tokenizer) -> None:
        ds_cfg = cfg.dataset
        self._max_new_tokens = cfg.get("max_new_tokens", 128)

        if self._val_dataset is not None:
            ds = self._val_dataset
        else:
            from datasets import load_dataset
            log.info("Loading val split '%s' for ROUGE eval", ds_cfg.split)
            ds = load_dataset(
                ds_cfg.path,
                split=ds_cfg.split,
                token=os.environ.get("HF_TOKEN"),
            )

        num_samples = cfg.get("num_samples")
        if num_samples and num_samples < len(ds):
            ds = ds.select(range(num_samples))
        log.info("ROUGE eval dataset: %d samples", len(ds))

        self._val_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in ds[ds_cfg.prompt_column]
        ]
        self._val_refs = list(ds[ds_cfg.summary_column])

    def evaluate(self, llm, step: int) -> dict:
        sampling_params = SamplingParams(temperature=0, max_tokens=self._max_new_tokens)
        outputs = llm.generate(self._val_prompts, sampling_params, use_tqdm=True)
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

        return report


# ---------------------------------------------------------------------------
# Toxicity eval
# ---------------------------------------------------------------------------

class ToxicityEvalCallback(VLLMEvalCallback):
    """Mean p(non-toxic) of greedy rollouts on the configured eval dataset.

    Reuses the live :class:`ToxicityReward` instance (shared classifier weights
    on GPU) for scoring, so this callback adds no extra model footprint.

    Note: TRL's GRPOTrainer also logs the toxicity reward when `eval_dataset`
    is set, but it triggers an HF generate pass and a separate compute path.
    This callback piggy-backs on the same vLLM batch as ROUGE for free, and
    adds flagged-ratio + completion-length sanity metrics.
    """

    def __init__(self, val_dataset=None, toxicity_reward_fn=None) -> None:
        self._val_dataset = val_dataset
        self._reward_fn = toxicity_reward_fn
        self._prompts: list[str] | None = None
        self._max_new_tokens: int = 128

    def setup(self, cfg: DictConfig, tokenizer) -> None:
        ds_cfg = cfg.dataset
        self._max_new_tokens = cfg.get("max_new_tokens", 128)

        if self._val_dataset is not None:
            ds = self._val_dataset
        else:
            from datasets import load_dataset
            log.info("Loading val split '%s' for toxicity eval", ds_cfg.split)
            ds = load_dataset(
                ds_cfg.path,
                split=ds_cfg.split,
                token=os.environ.get("HF_TOKEN"),
            )

        n = cfg.get("num_samples")
        if n and n < len(ds):
            ds = ds.select(range(n))
        log.info("Toxicity eval dataset: %d samples", len(ds))

        self._prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in ds[ds_cfg.prompt_column]
        ]

    def evaluate(self, llm, step: int) -> dict:
        sampling_params = SamplingParams(temperature=0, max_tokens=self._max_new_tokens)
        outputs = llm.generate(self._prompts, sampling_params, use_tqdm=True)
        texts = [o.outputs[0].text.strip() for o in outputs]

        scores = self._reward_fn([[{"role": "assistant", "content": t}] for t in texts])
        word_lens = [len(t.split()) for t in texts]

        n = max(len(scores), 1)
        report = {
            "eval/tox_p_non_toxic_mean": round(sum(scores) / n, 4),
            "eval/tox_flagged_ratio": round(sum(1 for s in scores if s < 0.5) / n, 4),
            "eval/completion_word_len_mean": round(sum(word_lens) / n, 2),
        }
        log.info("Step %d toxicity: %s", step, report)
        if wandb.run is not None:
            wandb.log(report)
        return report


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
                    result = cb.evaluate(self._engine.llm, step)
                    if result and metrics is not None:
                        metrics.update({k.replace("/", "_"): v for k, v in result.items()})
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


# ---------------------------------------------------------------------------
# Thin wrapper that reuses TRL's colocated vLLM (use_vllm=True, colocate mode)
# ---------------------------------------------------------------------------

class TRLVLLMManagerCallback(TrainerCallback):
    """Eval callback that reuses the ``vllm.LLM`` owned by GRPOTrainer.

    When ``use_vllm=True, vllm_mode='colocate'``, GRPOTrainer already owns a
    ``trainer.llm`` instance that is weight-synced before each rollout batch.
    This callback reuses that instance at eval time.

    Wake/sleep calls are skipped when ``vllm_enable_sleep_mode: false`` — the
    engine is already awake and must not be put to sleep mid-eval.

    Differences vs :class:`VLLMManagerCallback`:
    - Does **not** call ``init`` or ``destroy`` (lifecycle owned by TRL).
    - Does **not** call ``sync_weights`` (TRL already synced before last rollout).
    - Wraps the raw ``vllm.LLM`` object directly, not a :class:`VLLMEngine`.
    """

    def __init__(
        self,
        llm,
        eval_callbacks: list[VLLMEvalCallback],
        cfg: DictConfig,
        tokenizer,
    ) -> None:
        self._llm = llm
        self._eval_callbacks = eval_callbacks
        self._cfg = cfg
        self._tokenizer = tokenizer
        self._sleep_mode = cfg.get("vllm_sleep_mode", False)

    def on_train_begin(self, args, state, control, model=None, **kwargs) -> None:
        if not state.is_world_process_zero:
            return
        for cb in self._eval_callbacks:
            cb.setup(self._cfg, self._tokenizer)

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs) -> None:
        if not state.is_world_process_zero:
            return
        step = state.global_step
        log.info("vLLM eval (TRL colocate) at step %d", step)
        if self._sleep_mode:
            self._llm.wake_up()
        try:
            for cb in self._eval_callbacks:
                try:
                    result = cb.evaluate(self._llm, step)
                    if result and metrics is not None:
                        metrics.update({k.replace("/", "_"): v for k, v in result.items()})
                except Exception:
                    log.exception("%s failed at step %d; continuing", type(cb).__name__, step)
        finally:
            if self._sleep_mode:
                self._llm.sleep(level=1)
