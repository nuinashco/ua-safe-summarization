"""W&B helpers shared across training and evaluation scripts."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import wandb
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def uses_wandb(cfg: DictConfig) -> bool:
    report_to = cfg.training.get("report_to")
    return report_to == "wandb" or (isinstance(report_to, (list, tuple)) and "wandb" in report_to)


def configure_wandb(cfg: DictConfig) -> None:
    """Set W&B env vars before starting a new training run."""
    if not uses_wandb(cfg):
        return
    os.environ.setdefault("WANDB_LOG_MODEL", "false")
    if cfg.wandb.get("project"):
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb.project)
    if cfg.wandb.get("entity"):
        os.environ.setdefault("WANDB_ENTITY", cfg.wandb.entity)
    if cfg.wandb.get("name"):
        os.environ.setdefault("WANDB_NAME", cfg.wandb.name)


def save_run_id(cfg: DictConfig) -> None:
    """Write the active run ID to {output_dir}/wandb_run_id.txt."""
    if wandb.run is None:
        return
    run_id_path = Path(cfg.training.output_dir) / "wandb_run_id.txt"
    run_id_path.write_text(wandb.run.id)
    log.info("Saved wandb run ID to %s — pass to validate_sft.py to resume this run", run_id_path)


def resume_wandb(cfg: DictConfig) -> None:
    """Resume an existing W&B run for post-training evaluation scripts."""
    if not uses_wandb(cfg):
        return

    run_id = cfg.wandb.get("run_id")
    if not run_id:
        run_id_path = Path(cfg.training.output_dir) / "wandb_run_id.txt"
        if run_id_path.exists():
            run_id = run_id_path.read_text().strip()
            log.info("Resuming wandb run %s (from %s)", run_id, run_id_path)

    if run_id:
        wandb.init(
            id=run_id,
            resume="must",
            project=cfg.wandb.get("project"),
            entity=cfg.wandb.get("entity"),
        )
    else:
        log.warning("No wandb run ID found; metrics will not be logged to wandb.")
