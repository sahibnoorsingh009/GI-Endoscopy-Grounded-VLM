#!/usr/bin/env python3
"""One-sample forward/backward test of SO400M bridge tokens inside Qwen."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from vlm_demo.qwen_bridge_data import (  # noqa: E402
    QwenBridgeCollator,
    prepare_bridge_example,
)
from vlm_demo.qwen_bridge_model import build_qwen_bridge_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qwen-adapter",
        type=Path,
        default=Path(
            os.environ.get(
                "QWEN_ADAPTER_PATH",
                "/workspace/qwen-runs/"
                "qwen3-vl-8b-lora-sqrt-balanced-seed42/checkpoint-400",
            )
        ),
    )
    parser.add_argument(
        "--so400m-checkpoint",
        type=Path,
        default=(Path(os.environ["SO400M_CKPT"]) if os.getenv("SO400M_CKPT") else None),
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=(Path(os.environ["IMAGE_PATH"]) if os.getenv("IMAGE_PATH") else None),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/workspace/qwen-data/hyperkvasir"),
    )
    return parser.parse_args()


def resolve_image(path: Path | None, data_root: Path) -> Path:
    if path is not None:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return resolved
    preferred = sorted((data_root / "images").glob("HK_000002001.*"))
    candidates = preferred or sorted((data_root / "images").glob("*"))
    candidates = [candidate for candidate in candidates if candidate.is_file()]
    if not candidates:
        raise FileNotFoundError("No smoke-test image found")
    return candidates[0].resolve()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    image_path = resolve_image(args.image, args.data_root)
    bundle = build_qwen_bridge_bundle(
        qwen_adapter_path=args.qwen_adapter,
        so400m_checkpoint_path=args.so400m_checkpoint,
        device="cuda",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    tokenizer = bundle.qwen_processor.tokenizer
    record = {
        "image": str(image_path),
        "conversations": [
            {
                "from": "human",
                "value": (
                    "<image>\nClassify this GI endoscopy image using the "
                    "HyperKvasir taxonomy. Return only the category name."
                ),
            },
            {"from": "gpt", "value": "polyps"},
        ],
    }
    example = prepare_bridge_example(
        record,
        data_root=Path("/"),
        tokenizer=tokenizer,
        so400m_processor=bundle.so400m_processor,
        image_token_id=bundle.model.config.image_token_id,
        num_queries=64,
        max_length=512,
    )
    collator = QwenBridgeCollator(tokenizer.pad_token_id)
    batch = {
        key: value.to("cuda")
        for key, value in collator([example]).items()
    }

    bundle.model.train()
    bundle.model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    outputs = bundle.model(**batch)
    loss = outputs.loss
    if loss is None or not torch.isfinite(loss):
        raise RuntimeError(f"Invalid bridge loss: {loss}")
    loss.backward()

    allowed = {
        id(parameter)
        for parameter in bundle.vision_adapter.trainable_parameters()
    }
    unexpected_gradients = [
        name
        for name, parameter in bundle.model.named_parameters()
        if parameter.grad is not None and id(parameter) not in allowed
    ]
    bridge_gradients = [
        parameter.grad
        for parameter in bundle.vision_adapter.trainable_parameters()
        if parameter.grad is not None
    ]
    if unexpected_gradients:
        raise RuntimeError(
            f"Frozen Qwen/SO400M parameters received gradients: {unexpected_gradients[:5]}"
        )
    if not bridge_gradients:
        raise RuntimeError("The visual bridge received no gradients")
    if not all(torch.isfinite(gradient).all() for gradient in bridge_gradients):
        raise RuntimeError("The visual bridge produced non-finite gradients")
    if not any(gradient.abs().sum() > 0 for gradient in bridge_gradients):
        raise RuntimeError("All visual bridge gradients are zero")

    image_tokens = int(
        (batch["input_ids"] == bundle.model.config.image_token_id).sum().item()
    )
    print("Image:", image_path)
    print("SO400M checkpoint:", bundle.so400m_checkpoint)
    print("Input token shape:", tuple(batch["input_ids"].shape))
    print("Image placeholder tokens:", image_tokens)
    print("SO400M pixels:", tuple(batch["pixel_values"].shape))
    print("Qwen image grid:", batch["image_grid_thw"].tolist())
    print("Trainable bridge parameters:", bundle.vision_adapter.trainable_parameter_count)
    print("Loss:", round(float(loss.item()), 6))
    print("Gradient tensors:", len(bridge_gradients))
    print(
        "Peak GPU memory GiB:",
        round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    )
    print("Qwen + SO400M bridge forward/backward: OK")


if __name__ == "__main__":
    main()
