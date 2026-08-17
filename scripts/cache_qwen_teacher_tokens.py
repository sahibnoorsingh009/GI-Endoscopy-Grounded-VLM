#!/usr/bin/env python3
"""Cache fixed 64x4096 native-Qwen visual tokens for training images."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from vlm_demo.qwen_teacher import (  # noqa: E402
    build_qwen_visual_teacher,
    extract_qwen_teacher_tokens,
)
from vlm_demo.qwen_teacher_cache import (  # noqa: E402
    CACHE_CONFIG_NAME,
    CACHE_MANIFEST_NAME,
    atomic_write_json,
    inspect_cached_prefix,
    load_training_images,
    sha256_file,
    shard_name,
    write_teacher_shard,
)


DEFAULT_ADAPTER = Path(
    "/workspace/qwen-runs/qwen3-vl-8b-lora-sqrt-balanced-seed42/checkpoint-400"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-index",
        type=Path,
        default=Path("/workspace/qwen-data/hyperkvasir/index/hyperkvasir_train.csv"),
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("/workspace/qwen-data/hyperkvasir")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qwen-adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-every", type=int, default=128)
    return parser.parse_args()


def read_images(paths: list[Path]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as handle:
            images.append(handle.convert("RGB"))
    return images


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen teacher caching")
    if args.batch_size <= 0 or args.shard_size <= 0 or args.report_every <= 0:
        raise ValueError("Batch, shard, and reporting sizes must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_index = args.train_index.expanduser().resolve()
    adapter_path = args.qwen_adapter.expanduser().resolve()
    images = load_training_images(
        train_index,
        args.data_root,
        limit=args.limit,
    )
    image_ids = [item.image_id for item in images]
    config = {
        "architecture": "Qwen3VLNativeVisualTeacher",
        "model_id": args.model_id,
        "qwen_adapter": str(adapter_path),
        "train_index": str(train_index),
        "train_index_sha256": sha256_file(train_index),
        "requested_samples": len(images),
        "target_tokens": 64,
        "hidden_size": 4096,
        "dtype": "bfloat16",
        "target_grid": [8, 8],
        "split": "train",
    }
    config_path = output_dir / CACHE_CONFIG_NAME
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if not args.resume:
            raise FileExistsError(
                f"Teacher cache already exists: {output_dir}; pass --resume"
            )
        if existing != config:
            raise ValueError(
                f"Resume configuration changed.\nSaved: {existing}\nCurrent: {config}"
            )
    else:
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory is non-empty but has no cache config: {output_dir}"
            )
        atomic_write_json(config_path, config)

    processed, shard_records = inspect_cached_prefix(output_dir, image_ids)
    if processed == len(images):
        print(f"Teacher cache already complete: {processed} images")
        return

    estimated_gib = len(images) * 64 * 4096 * 2 / 1024**3
    print("Requested training images:", len(images))
    print("Already cached:", processed)
    print("Estimated token GiB:", round(estimated_gib, 3))
    print("Output directory:", output_dir)

    bundle = build_qwen_visual_teacher(
        qwen_adapter_path=adapter_path,
        qwen_model_id=args.model_id,
        device="cuda",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    buffer_tokens: list[torch.Tensor] = []
    buffer_ids: list[str] = []
    shard_index = len(shard_records)

    for start in range(processed, len(images), args.batch_size):
        batch_items = images[start : start + args.batch_size]
        pil_images = read_images([item.path for item in batch_items])
        fixed, _, _ = extract_qwen_teacher_tokens(bundle, pil_images)
        if fixed.shape != (len(batch_items), 64, 4096):
            raise RuntimeError(f"Unexpected teacher batch shape: {tuple(fixed.shape)}")
        buffer_tokens.append(fixed.detach().cpu())
        buffer_ids.extend(item.image_id for item in batch_items)
        del fixed, pil_images

        buffered = sum(tensor.shape[0] for tensor in buffer_tokens)
        if buffered >= args.shard_size or start + len(batch_items) == len(images):
            combined = torch.cat(buffer_tokens, dim=0)
            while combined.shape[0] >= args.shard_size:
                shard_tokens = combined[: args.shard_size]
                shard_ids = buffer_ids[: args.shard_size]
                path = output_dir / shard_name(shard_index)
                write_teacher_shard(
                    path,
                    shard_tokens,
                    shard_ids,
                    metadata={"model_id": args.model_id},
                )
                shard_records.append(
                    {
                        "file": path.name,
                        "count": len(shard_ids),
                        "first_image_id": shard_ids[0],
                        "last_image_id": shard_ids[-1],
                        "shape": list(shard_tokens.shape),
                    }
                )
                shard_index += 1
                combined = combined[args.shard_size :]
                buffer_ids = buffer_ids[args.shard_size :]
            buffer_tokens = [combined] if combined.shape[0] else []

            cached_count = sum(record["count"] for record in shard_records)
            atomic_write_json(
                output_dir / CACHE_MANIFEST_NAME,
                {
                    "complete": False,
                    "cached_samples": cached_count,
                    "requested_samples": len(images),
                    "shards": shard_records,
                },
            )
            if cached_count and (
                cached_count % args.report_every == 0
                or cached_count == len(images)
            ):
                elapsed = time.perf_counter() - started
                print(
                    f"[{cached_count}/{len(images)}] "
                    f"images_per_second={cached_count / elapsed:.2f}",
                    flush=True,
                )

    if buffer_tokens:
        final_tokens = torch.cat(buffer_tokens, dim=0)
        path = output_dir / shard_name(shard_index)
        write_teacher_shard(
            path,
            final_tokens,
            buffer_ids,
            metadata={"model_id": args.model_id},
        )
        shard_records.append(
            {
                "file": path.name,
                "count": len(buffer_ids),
                "first_image_id": buffer_ids[0],
                "last_image_id": buffer_ids[-1],
                "shape": list(final_tokens.shape),
            }
        )

    cached_count = sum(record["count"] for record in shard_records)
    if cached_count != len(images):
        raise RuntimeError(f"Incomplete teacher cache: {cached_count} != {len(images)}")
    elapsed = time.perf_counter() - started
    atomic_write_json(
        output_dir / CACHE_MANIFEST_NAME,
        {
            "complete": True,
            "cached_samples": cached_count,
            "requested_samples": len(images),
            "token_shape": [64, 4096],
            "dtype": "bfloat16",
            "shards": shard_records,
        },
    )
    print("Teacher cache complete")
    print("Cached samples:", cached_count)
    print("Shards:", len(shard_records))
    print("Elapsed minutes:", round(elapsed / 60, 2))
    print("Peak GPU memory GiB:", round(torch.cuda.max_memory_allocated() / 1024**3, 3))
    print("Cache manifest:", output_dir / CACHE_MANIFEST_NAME)


if __name__ == "__main__":
    main()
