#!/usr/bin/env python3
"""Run the real SO400M encoder through the untrained visual bridge."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from vlm_demo.so400m_encoder import build_frozen_so400m_encoder  # noqa: E402
from vlm_demo.visual_bridge import (  # noqa: E402
    SO400MVisualBridge,
    VisualBridgeConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("SO400M_CKPT"),
        help="Local compact SO400M checkpoint; otherwise download from HF.",
    )
    parser.add_argument(
        "--image",
        default=os.environ.get("IMAGE_PATH"),
        help="A local GI endoscopy image.",
    )
    parser.add_argument("--queries", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    return parser.parse_args()


def discover_image(explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image does not exist: {path}")
        return path

    root = Path("/workspace/qwen-data/hyperkvasir/images")
    preferred = sorted(root.glob("HK_000002001.*"))
    candidates = preferred or sorted(root.glob("*"))
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        raise FileNotFoundError(
            "No image found; pass --image or set IMAGE_PATH"
        )
    return candidates[0]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This real-checkpoint smoke test requires CUDA")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    image_path = discover_image(args.image)

    encoder, processor, checkpoint_path = build_frozen_so400m_encoder(
        args.checkpoint,
        device=device,
        dtype=dtype,
    )
    bridge = SO400MVisualBridge(
        VisualBridgeConfig(
            num_queries=args.queries,
            depth=args.depth,
        )
    ).to(device=device, dtype=dtype)
    bridge.train()

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt")[
        "pixel_values"
    ].to(device=device, dtype=dtype)

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        features = encoder(pixel_values)

    projected_tokens = bridge(features.dense_tokens)
    target = torch.full_like(projected_tokens.float(), 0.01)
    loss = torch.nn.functional.mse_loss(
        projected_tokens.float(),
        target,
    )
    loss.backward()

    encoder_gradients = [
        parameter.grad
        for parameter in encoder.parameters()
        if parameter.grad is not None
    ]
    bridge_gradients = [
        parameter.grad
        for parameter in bridge.parameters()
        if parameter.grad is not None
    ]

    if encoder_gradients:
        raise RuntimeError("Frozen encoder unexpectedly received gradients")
    if not bridge_gradients:
        raise RuntimeError("Bridge did not receive gradients")
    if not all(torch.isfinite(gradient).all() for gradient in bridge_gradients):
        raise RuntimeError("Bridge produced non-finite gradients")
    if not any(gradient.abs().sum() > 0 for gradient in bridge_gradients):
        raise RuntimeError("All bridge gradients are zero")

    print("Checkpoint:", checkpoint_path)
    print("Image:", image_path)
    print("Input pixels:", tuple(pixel_values.shape), pixel_values.dtype)
    print("SO400M dense:", tuple(features.dense_tokens.shape))
    print("SO400M pooled:", tuple(features.pooled_token.shape))
    print("Qwen-width tokens:", tuple(projected_tokens.shape))
    print("Encoder trainable parameters: 0")
    print("Bridge trainable parameters:", bridge.trainable_parameter_count)
    print("Bridge loss:", round(loss.item(), 6))
    print("Bridge gradient tensors:", len(bridge_gradients))
    print(
        "Peak GPU memory GiB:",
        round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    )
    print("SO400M visual bridge smoke test: OK")


if __name__ == "__main__":
    main()
