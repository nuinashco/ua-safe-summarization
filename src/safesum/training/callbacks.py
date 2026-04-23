"""Training callbacks for SFT."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import wandb
from omegaconf import DictConfig
from transformers import TrainerCallback

log = logging.getLogger(__name__)

_VALIDATE_SCRIPT = Path(__file__).parent.parent.parent.parent / "scripts" / "validate" / "validate_sft.py"


class RougeEvalCallback(TrainerCallback):
    """Spawns validate_sft.py via a vLLM subprocess at each checkpoint save."""

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg
        self._proc: subprocess.Popen | None = None

    def on_save(self, args, state, control, **kwargs) -> None:
        rouge_cfg = self._cfg.validation.get("rouge_callback")
        if not rouge_cfg or not rouge_cfg.get("enabled"):
            return

        if self._proc and self._proc.poll() is None:
            log.warning(
                "Previous ROUGE eval still running (pid=%d); skipping step %d",
                self._proc.pid,
                state.global_step,
            )
            return

        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not ckpt_dir.exists():
            log.warning("Checkpoint dir %s not found; skipping ROUGE eval", ckpt_dir)
            return

        cmd = [
            "uv", "run", "python", str(_VALIDATE_SCRIPT),
            f"training.output_dir={ckpt_dir}",
            f"validation.wandb_step={state.global_step}",
        ]
        if wandb.run:
            cmd.append(f"wandb.run_id={wandb.run.id}")

        num_samples = rouge_cfg.get("num_samples")
        if num_samples:
            cmd.append(f"validation.num_samples={num_samples}")

        gpu_mem_util = rouge_cfg.get("gpu_memory_utilization")
        if gpu_mem_util is not None:
            cmd.append(f"validation.gpu_memory_utilization={gpu_mem_util}")

        log_file = Path(args.output_dir) / f"rouge_eval_step{state.global_step}.log"
        log.info("Spawning ROUGE eval for checkpoint-%d → %s", state.global_step, log_file)
        self._proc = subprocess.Popen(
            cmd,
            stdout=log_file.open("w"),
            stderr=subprocess.STDOUT,
        )
        log.info("ROUGE eval subprocess pid=%d", self._proc.pid)
