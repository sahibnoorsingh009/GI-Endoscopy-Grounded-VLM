"""Dataset, cache access, batching, and losses for visual-token distillation."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import Dataset

from vlm_demo.qwen_bridge_training import TrainingPlan
from vlm_demo.qwen_teacher_cache import (
    CACHE_MANIFEST_NAME,
    TrainingImage,
    inspect_teacher_shard,
    load_training_images,
)


class TeacherTokenStore:
    """Image-ID lookup with a small per-worker LRU shard cache."""

    def __init__(self, cache_dir: str | Path, *, max_cached_shards: int = 1) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        manifest_path = self.cache_dir / CACHE_MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete") is not True:
            raise ValueError(f"Teacher cache is not complete: {manifest_path}")
        if max_cached_shards <= 0:
            raise ValueError("max_cached_shards must be positive")
        self.max_cached_shards = max_cached_shards
        self.locations: dict[str, tuple[Path, int]] = {}
        self.shard_image_ids: list[list[str]] = []
        self._cache: OrderedDict[Path, torch.Tensor] = OrderedDict()

        shard_records = manifest.get("shards")
        if not isinstance(shard_records, list) or not shard_records:
            raise ValueError("Teacher cache manifest contains no shards")
        for record in shard_records:
            path = self.cache_dir / str(record["file"])
            inspected = inspect_teacher_shard(path)
            image_ids = [str(value) for value in inspected["image_ids"]]
            self.shard_image_ids.append(image_ids)
            for row, image_id in enumerate(image_ids):
                if image_id in self.locations:
                    raise ValueError(f"Duplicate teacher image ID: {image_id}")
                self.locations[image_id] = (path, row)
        if len(self.locations) != int(manifest["cached_samples"]):
            raise ValueError("Teacher manifest sample count does not match shard metadata")

    @property
    def image_ids(self) -> set[str]:
        return set(self.locations)

    def _load_shard(self, path: Path) -> torch.Tensor:
        if path in self._cache:
            tensor = self._cache.pop(path)
            self._cache[path] = tensor
            return tensor
        tensor = load_file(path, device="cpu")["tokens"]
        if tensor.ndim != 3 or tensor.shape[1:] != (64, 4096):
            raise ValueError(f"Unexpected cached teacher shape in {path}: {tuple(tensor.shape)}")
        self._cache[path] = tensor
        while len(self._cache) > self.max_cached_shards:
            self._cache.popitem(last=False)
        return tensor

    def get(self, image_id: str) -> torch.Tensor:
        try:
            path, row = self.locations[image_id]
        except KeyError as error:
            raise KeyError(f"No teacher tokens for image {image_id}") from error
        return self._load_shard(path)[row]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state


class QwenBridgeDistillationDataset(Dataset):
    def __init__(
        self,
        *,
        train_index: str | Path,
        data_root: str | Path,
        teacher_cache: str | Path,
        so400m_processor: Any,
        limit: int | None = None,
        max_cached_shards: int = 1,
    ) -> None:
        self.images = load_training_images(train_index, data_root, limit=limit)
        self.teacher_store = TeacherTokenStore(
            teacher_cache,
            max_cached_shards=max_cached_shards,
        )
        image_ids = {item.image_id for item in self.images}
        if image_ids != self.teacher_store.image_ids:
            missing = sorted(image_ids.difference(self.teacher_store.image_ids))
            extra = sorted(self.teacher_store.image_ids.difference(image_ids))
            raise ValueError(
                "Teacher cache and requested training images differ: "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )
        self.so400m_processor = so400m_processor
        index_by_id = {item.image_id: index for index, item in enumerate(self.images)}
        self.shard_index_groups = [
            [index_by_id[image_id] for image_id in shard_ids]
            for shard_ids in self.teacher_store.shard_image_ids
        ]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item: TrainingImage = self.images[index]
        with Image.open(item.path) as handle:
            image = handle.convert("RGB")
        pixel_values = self.so400m_processor(
            images=image,
            return_tensors="pt",
        )["pixel_values"].squeeze(0)
        return {
            "pixel_values": pixel_values,
            "teacher_tokens": self.teacher_store.get(item.image_id),
        }


class QwenDistillationCollator:
    def __call__(
        self,
        examples: Sequence[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        return {
            "pixel_values": torch.stack(
                [example["pixel_values"] for example in examples]
            ),
            "teacher_tokens": torch.stack(
                [example["teacher_tokens"] for example in examples]
            ),
        }


def shard_aware_epoch_batches(
    groups: Sequence[Sequence[int]],
    *,
    batch_size: int,
    seed: int,
    epoch: int,
) -> list[list[int]]:
    if batch_size <= 0 or epoch < 0 or not groups:
        raise ValueError("Invalid shard-aware batching arguments")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch)
    group_order = torch.randperm(len(groups), generator=generator).tolist()
    batches: list[list[int]] = []
    for group_index in group_order:
        group = list(groups[group_index])
        if not group:
            raise ValueError("Teacher shard groups cannot be empty")
        local_order = torch.randperm(len(group), generator=generator).tolist()
        shuffled = [group[index] for index in local_order]
        batches.extend(
            shuffled[start : start + batch_size]
            for start in range(0, len(shuffled), batch_size)
        )
    return batches


def build_distillation_plan(
    *,
    groups: Sequence[Sequence[int]],
    dataset_size: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
    max_steps: int | None = None,
) -> TrainingPlan:
    if gradient_accumulation_steps <= 0 or epochs <= 0:
        raise ValueError("Accumulation and epochs must be positive")
    micro_batches = len(
        shard_aware_epoch_batches(
            groups,
            batch_size=batch_size,
            seed=0,
            epoch=0,
        )
    )
    optimizer_steps_per_epoch = (
        micro_batches + gradient_accumulation_steps - 1
    ) // gradient_accumulation_steps
    natural_total = optimizer_steps_per_epoch * epochs
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        total_steps = min(natural_total, max_steps)
    else:
        total_steps = natural_total
    return TrainingPlan(
        dataset_size=dataset_size,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        epochs=epochs,
        batches_per_epoch=micro_batches,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        total_optimizer_steps=total_steps,
    )


def visual_distillation_losses(
    student_tokens: torch.Tensor,
    teacher_tokens: torch.Tensor,
    *,
    normalized_mse_weight: float = 1.0,
    cosine_weight: float = 1.0,
    norm_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    if student_tokens.shape != teacher_tokens.shape:
        raise ValueError(
            f"Student/teacher shapes differ: {tuple(student_tokens.shape)} != "
            f"{tuple(teacher_tokens.shape)}"
        )
    if student_tokens.ndim != 3 or student_tokens.shape[1:] != (64, 4096):
        raise ValueError("Distillation tokens must have shape [batch, 64, 4096]")
    if min(normalized_mse_weight, cosine_weight, norm_weight) < 0:
        raise ValueError("Distillation loss weights cannot be negative")

    student = student_tokens.float()
    teacher = teacher_tokens.float()
    normalized_student = F.layer_norm(student, (student.shape[-1],))
    normalized_teacher = F.layer_norm(teacher, (teacher.shape[-1],))
    normalized_mse = F.mse_loss(normalized_student, normalized_teacher)
    cosine = 1.0 - F.cosine_similarity(student, teacher, dim=-1).mean()
    norm = F.smooth_l1_loss(
        torch.log(student.norm(dim=-1).clamp_min(1e-6)),
        torch.log(teacher.norm(dim=-1).clamp_min(1e-6)),
    )
    total = (
        normalized_mse_weight * normalized_mse
        + cosine_weight * cosine
        + norm_weight * norm
    )
    return {
        "loss": total,
        "normalized_mse": normalized_mse,
        "cosine": cosine,
        "norm": norm,
    }
