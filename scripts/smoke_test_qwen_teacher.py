#!/usr/bin/env python3
"""Verify the native-Qwen visual teacher contract on one GI image."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from vlm_demo.qwen_teacher import (  # noqa: E402
    build_qwen_visual_teacher,
    extract_qwen_teacher_tokens,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qwen-adapter",
        type=Path,
        default=Path(
            "/workspace/qwen-runs/"
            "qwen3-vl-8b-lora-sqrt-balanced-seed42/checkpoint-400"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/workspace/qwen-data/hyperkvasir"),
    )
    parser.add_argument("--image", type=Path)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    return parser.parse_args()


def resolve_image(path: Path | None, data_root: Path) -> Path:
    if path is not None:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return resolved
    train_index = data_root / "index" / "hyperkvasir_train.csv"
    if not train_index.is_file():
        raise FileNotFoundError(f"Training index not found: {train_index}")
    with train_index.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row.get("split") != "train" for row in rows):
        raise ValueError("Teacher smoke test requires the official training index")
    image_id = sorted(str(row["image_id"]) for row in rows)[0]
    candidate = (data_root / "images" / f"{image_id}.jpg").resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    image_path = resolve_image(args.image, args.data_root)
    torch.cuda.reset_peak_memory_stats()
    bundle = build_qwen_visual_teacher(
        qwen_adapter_path=args.qwen_adapter,
        qwen_model_id=args.model_id,
        device="cuda",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    with Image.open(image_path) as image_handle:
        image = image_handle.convert("RGB")
    fixed, grid_thw, deepstack = extract_qwen_teacher_tokens(bundle, [image])

    if fixed.shape != (1, 64, 4096):
        raise RuntimeError(f"Unexpected fixed teacher shape: {tuple(fixed.shape)}")
    if not torch.isfinite(fixed).all():
        raise RuntimeError("Teacher tokens contain non-finite values")
    if any(parameter.requires_grad for parameter in bundle.model.parameters()):
        raise RuntimeError("The Qwen teacher is not fully frozen")

    print("Image:", image_path)
    print("Qwen adapter:", bundle.adapter_path)
    print("Native image grid:", grid_thw.tolist())
    print("Fixed teacher tokens:", tuple(fixed.shape), fixed.dtype)
    print("DeepStack outputs:", len(deepstack))
    print("Mean teacher-token norm:", round(float(fixed.float().norm(dim=-1).mean()), 4))
    print("Trainable teacher parameters: 0")
    print("Peak GPU memory GiB:", round(torch.cuda.max_memory_allocated() / 1024**3, 3))
    print("Native Qwen visual teacher contract: OK")


if __name__ == "__main__":
    main()
