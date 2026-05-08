from __future__ import annotations

import logging
from pathlib import Path

import torch
from ultralytics import YOLO

from rail_inspection.config import TrainConfig

LOGGER = logging.getLogger(__name__)


def _candidate_project_roots(project: str) -> list[Path]:
    rel_project = Path(project)
    return [
        rel_project,
        Path.cwd() / rel_project,
        Path.home() / rel_project,
        Path.home() / "runs" / "detect" / rel_project,
        Path.home() / "runs" / "detect",
    ]


def _latest_checkpoint(project: str, run_name: str, filename: str = "last.pt") -> Path | None:
    candidates: list[Path] = []
    for root in _candidate_project_roots(project):
        preferred = root / run_name / "weights" / filename
        if preferred.exists():
            return preferred
        if root.exists():
            candidates.extend(list(root.glob(f"**/weights/{filename}")))

    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _checkpoint_target_epochs(checkpoint: Path) -> int | None:
    try:
        ckpt = torch.load(str(checkpoint), map_location="cpu")
        train_args = ckpt.get("train_args") or {}
        epochs = train_args.get("epochs")
        return int(epochs) if epochs is not None else None
    except Exception:  # noqa: BLE001
        return None


def run_training(data_yaml: Path, cfg: TrainConfig, resume: bool = False) -> Path:
    run_dir = Path(cfg.project) / cfg.name

    if resume:
        checkpoint = _latest_checkpoint(cfg.project, cfg.name, filename="last.pt")
        if checkpoint and checkpoint.name == "last.pt":
            saved_epochs = _checkpoint_target_epochs(checkpoint)
            if saved_epochs is not None and saved_epochs != cfg.epochs:
                LOGGER.warning(
                    "Checkpoint was created with epochs=%s, requested epochs=%s. "
                    "Starting continuation from last.pt with new epochs target.",
                    saved_epochs,
                    cfg.epochs,
                )
                model = YOLO(str(checkpoint))
                model.train(
                    data=str(data_yaml),
                    epochs=cfg.epochs,
                    imgsz=cfg.imgsz,
                    batch=cfg.batch,
                    device=cfg.device,
                    project=cfg.project,
                    name=cfg.name,
                    val=True,
                    verbose=True,
                )
            else:
                LOGGER.info("Resuming from checkpoint: %s", checkpoint)
                model = YOLO(str(checkpoint))
                model.train(resume=True, epochs=cfg.epochs)
                run_dir = checkpoint.parent.parent
        elif checkpoint and checkpoint.name == "best.pt":
            LOGGER.warning(
                "Found only best checkpoint (%s). Starting a new training run from model %s.",
                checkpoint,
                checkpoint,
            )
            model = YOLO(str(checkpoint))
            model.train(
                data=str(data_yaml),
                epochs=cfg.epochs,
                imgsz=cfg.imgsz,
                batch=cfg.batch,
                device=cfg.device,
                project=cfg.project,
                name=cfg.name,
                val=True,
                verbose=True,
            )
        else:
            best_ckpt = _latest_checkpoint(cfg.project, cfg.name, filename="best.pt")
            if best_ckpt:
                LOGGER.warning(
                    "No last.pt found. Starting new training from best checkpoint: %s",
                    best_ckpt,
                )
                model = YOLO(str(best_ckpt))
                model.train(
                    data=str(data_yaml),
                    epochs=cfg.epochs,
                    imgsz=cfg.imgsz,
                    batch=cfg.batch,
                    device=cfg.device,
                    project=cfg.project,
                    name=cfg.name,
                    val=True,
                    verbose=True,
                )
            else:
                LOGGER.warning("No checkpoint found for resume. Starting fresh training.")
                model = YOLO(cfg.model)
                model.train(
                    data=str(data_yaml),
                    epochs=cfg.epochs,
                    imgsz=cfg.imgsz,
                    batch=cfg.batch,
                    device=cfg.device,
                    project=cfg.project,
                    name=cfg.name,
                    val=True,
                    verbose=True,
                )
    else:
        model = YOLO(cfg.model)
        model.train(
            data=str(data_yaml),
            epochs=cfg.epochs,
            imgsz=cfg.imgsz,
            batch=cfg.batch,
            device=cfg.device,
            project=cfg.project,
            name=cfg.name,
            val=True,
            verbose=True,
        )

    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        fallback_best = _latest_checkpoint(cfg.project, cfg.name, filename="best.pt")
        if fallback_best:
            best_weights = fallback_best
        else:
            raise RuntimeError(f"Training finished but best weights were not found at {best_weights}")

    if not best_weights.exists():
        raise RuntimeError(f"Training finished but best weights were not found at {best_weights}")

    LOGGER.info("Validating best model on val split...")
    trained_model = YOLO(str(best_weights))
    trained_model.val(data=str(data_yaml), split="val", imgsz=cfg.imgsz, device=cfg.device)

    LOGGER.info("Evaluating best model on test split...")
    trained_model.val(data=str(data_yaml), split="test", imgsz=cfg.imgsz, device=cfg.device)
    return best_weights

