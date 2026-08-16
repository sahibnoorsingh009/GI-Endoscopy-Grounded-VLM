"""Deterministic planning and checkpoint helpers for bridge-only training."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from vlm_demo.qwen_bridge_model import (
    BRIDGE_WEIGHTS_NAME,
    SO400MBridgeVisionAdapter,
    save_visual_bridge,
)


OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
TRAINER_STATE_NAME = "trainer_state.json"
LATEST_CHECKPOINT_NAME = "latest_checkpoint.txt"


@dataclass(frozen=True)
class TrainingPlan:
    dataset_size: int
    batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    batches_per_epoch: int
    optimizer_steps_per_epoch: int
    total_optimizer_steps: int


def build_training_plan(
    *,
    dataset_size: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
    max_steps: int | None = None,
) -> TrainingPlan:
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    if batch_size <= 0 or gradient_accumulation_steps <= 0 or epochs <= 0:
        raise ValueError("batch_size, accumulation, and epochs must be positive")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when provided")

    batches_per_epoch = math.ceil(dataset_size / batch_size)
    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch / gradient_accumulation_steps
    )
    natural_total = optimizer_steps_per_epoch * epochs
    total = min(natural_total, max_steps) if max_steps is not None else natural_total
    return TrainingPlan(
        dataset_size=dataset_size,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        epochs=epochs,
        batches_per_epoch=batches_per_epoch,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        total_optimizer_steps=total,
    )


def epoch_batches(
    *,
    dataset_size: int,
    batch_size: int,
    seed: int,
    epoch: int,
) -> list[list[int]]:
    """Return the reproducible shuffled batches for one epoch."""

    if dataset_size <= 0 or batch_size <= 0 or epoch < 0:
        raise ValueError("Invalid epoch batching arguments")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch)
    order = torch.randperm(dataset_size, generator=generator).tolist()
    return [order[start : start + batch_size] for start in range(0, dataset_size, batch_size)]


def checkpoint_name(global_step: int) -> str:
    if global_step <= 0:
        raise ValueError("global_step must be positive")
    return f"checkpoint-{global_step}"


def resolve_resume_checkpoint(
    output_dir: str | Path,
    requested: str | Path | None,
) -> Path | None:
    if requested is None:
        return None
    output_dir = Path(output_dir).expanduser().resolve()
    if str(requested) == "latest":
        pointer = output_dir / LATEST_CHECKPOINT_NAME
        if not pointer.is_file():
            raise FileNotFoundError(f"Latest-checkpoint pointer not found: {pointer}")
        requested = pointer.read_text(encoding="utf-8").strip()
        if not requested:
            raise ValueError(f"Latest-checkpoint pointer is empty: {pointer}")
        path = output_dir / requested
    else:
        path = Path(requested).expanduser()
        if not path.is_absolute():
            path = output_dir / path
    path = path.resolve()
    required = (BRIDGE_WEIGHTS_NAME, OPTIMIZER_NAME, SCHEDULER_NAME, TRAINER_STATE_NAME)
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete training checkpoint {path}: {missing}")
    return path


def save_training_checkpoint(
    *,
    output_dir: str | Path,
    adapter: SO400MBridgeVisionAdapter,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    state: dict[str, Any],
    metadata: dict[str, Any],
    final: bool = False,
) -> Path:
    """Atomically save bridge weights and the state needed for exact continuation."""

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    global_step = int(state["global_step"])
    destination = output_dir / ("final" if final else checkpoint_name(global_step))
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")

    temporary = Path(tempfile.mkdtemp(prefix=".bridge-checkpoint-", dir=output_dir))
    try:
        save_visual_bridge(adapter, temporary, metadata=metadata)
        torch.save(optimizer.state_dict(), temporary / OPTIMIZER_NAME)
        torch.save(scheduler.state_dict(), temporary / SCHEDULER_NAME)
        (temporary / TRAINER_STATE_NAME).write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    pointer_tmp = output_dir / f".{LATEST_CHECKPOINT_NAME}.tmp"
    pointer_tmp.write_text(destination.name + "\n", encoding="utf-8")
    os.replace(pointer_tmp, output_dir / LATEST_CHECKPOINT_NAME)
    return destination


def load_training_checkpoint(
    *,
    checkpoint: str | Path,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    expected_plan: TrainingPlan,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint).expanduser().resolve()
    state = json.loads((checkpoint / TRAINER_STATE_NAME).read_text(encoding="utf-8"))
    saved_plan = state.get("training_plan")
    if saved_plan != asdict(expected_plan):
        raise ValueError(
            "Resume arguments changed the training plan. Resume with the original "
            f"batch/accumulation/epoch/max-step settings.\nSaved: {saved_plan}\n"
            f"Current: {asdict(expected_plan)}"
        )
    optimizer.load_state_dict(
        torch.load(checkpoint / OPTIMIZER_NAME, map_location="cpu", weights_only=True)
    )
    scheduler.load_state_dict(
        torch.load(checkpoint / SCHEDULER_NAME, map_location="cpu", weights_only=True)
    )
    return state
