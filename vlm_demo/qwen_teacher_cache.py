"""Resumable sharded storage for native-Qwen visual teacher tokens."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


CACHE_CONFIG_NAME = "cache_config.json"
CACHE_MANIFEST_NAME = "cache_manifest.json"
SHARD_GLOB = "teacher_tokens_*.safetensors"
EXPECTED_TRAIN_IMAGES = 7433


@dataclass(frozen=True)
class TrainingImage:
    image_id: str
    path: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_images(
    index_csv: str | Path,
    data_root: str | Path,
    *,
    limit: int | None = None,
) -> list[TrainingImage]:
    index_csv = Path(index_csv).expanduser().resolve()
    data_root = Path(data_root).expanduser().resolve()
    if not index_csv.is_file():
        raise FileNotFoundError(index_csv)
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    with index_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Training index is empty")
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("Teacher cache may read only the official training split")
    image_ids = [str(row["image_id"]) for row in rows]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Training image IDs must be unique")
    if len(rows) != EXPECTED_TRAIN_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_TRAIN_IMAGES} training images, found {len(rows)}"
        )

    images = [
        TrainingImage(
            image_id=image_id,
            path=(data_root / "images" / f"{image_id}.jpg").resolve(),
        )
        for image_id in sorted(image_ids)
    ]
    selected = images[:limit] if limit is not None else images
    missing = [str(item.path) for item in selected if not item.path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Teacher cache is missing {len(missing)} images; first: {missing[0]}"
        )
    return selected


def shard_name(index: int) -> str:
    if index < 0:
        raise ValueError("Shard index cannot be negative")
    return f"teacher_tokens_{index:05d}.safetensors"


def write_teacher_shard(
    path: str | Path,
    tokens: torch.Tensor,
    image_ids: list[str],
    *,
    metadata: dict[str, str] | None = None,
) -> Path:
    path = Path(path).expanduser().resolve()
    if tokens.ndim != 3 or tokens.shape[1:] != (64, 4096):
        raise ValueError(
            f"Teacher tokens must have shape [images, 64, 4096], got {tuple(tokens.shape)}"
        )
    if tokens.shape[0] != len(image_ids) or not image_ids:
        raise ValueError("Teacher token count and non-empty image ID list must match")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("A teacher shard cannot contain duplicate image IDs")
    if not torch.isfinite(tokens).all():
        raise ValueError("Teacher shard contains non-finite values")

    path.parent.mkdir(parents=True, exist_ok=True)
    shard_metadata = dict(metadata or {})
    shard_metadata["image_ids"] = json.dumps(image_ids, separators=(",", ":"))
    shard_metadata["token_shape"] = "64x4096"
    temporary = path.with_name(f".{path.name}.tmp.safetensors")
    if temporary.exists():
        raise FileExistsError(f"Temporary shard already exists: {temporary}")
    save_file(
        {"tokens": tokens.detach().cpu().to(torch.bfloat16).contiguous()},
        temporary,
        metadata=shard_metadata,
    )
    os.replace(temporary, path)
    return path


def inspect_teacher_shard(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if "tokens" not in handle.keys():
            raise ValueError(f"Teacher shard has no tokens tensor: {path}")
        shape = tuple(handle.get_slice("tokens").get_shape())
    try:
        image_ids = json.loads(metadata["image_ids"])
    except (KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"Teacher shard has invalid image_ids metadata: {path}") from error
    if not isinstance(image_ids, list) or shape != (len(image_ids), 64, 4096):
        raise ValueError(
            f"Teacher shard metadata/shape mismatch in {path}: {shape}"
        )
    return {
        "file": path.name,
        "count": len(image_ids),
        "first_image_id": image_ids[0],
        "last_image_id": image_ids[-1],
        "image_ids": image_ids,
        "shape": list(shape),
    }


def inspect_cached_prefix(
    output_dir: str | Path,
    expected_image_ids: list[str],
) -> tuple[int, list[dict[str, Any]]]:
    output_dir = Path(output_dir).expanduser().resolve()
    shards = sorted(output_dir.glob(SHARD_GLOB))
    records: list[dict[str, Any]] = []
    cached_ids: list[str] = []
    for index, path in enumerate(shards):
        if path.name != shard_name(index):
            raise ValueError(
                f"Teacher shards must be contiguous; expected {shard_name(index)}, got {path.name}"
            )
        record = inspect_teacher_shard(path)
        cached_ids.extend(record.pop("image_ids"))
        records.append(record)
    if cached_ids != expected_image_ids[: len(cached_ids)]:
        raise ValueError("Cached teacher image IDs are not the expected training prefix")
    if len(cached_ids) > len(expected_image_ids):
        raise ValueError("Teacher cache contains more images than requested")
    return len(cached_ids), records


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
