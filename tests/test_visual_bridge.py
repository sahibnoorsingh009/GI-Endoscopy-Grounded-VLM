from __future__ import annotations

import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from vlm_demo.so400m_encoder import extract_vision_state  # noqa: E402
from vlm_demo.visual_bridge import (  # noqa: E402
    SO400MVisualBridge,
    VisualBridgeConfig,
)


def test_state_extraction() -> None:
    tensor = torch.zeros(2, 2)
    state = extract_vision_state(
        {"vision_model": {"vision_model.embeddings.weight": tensor}}
    )
    assert list(state) == ["vision_model.embeddings.weight"]

    bare = extract_vision_state({"encoder.layers.0.weight": tensor})
    assert list(bare) == ["vision_model.encoder.layers.0.weight"]


def test_bridge_contract() -> None:
    torch.manual_seed(42)
    config = VisualBridgeConfig(
        input_dim=32,
        output_dim=64,
        num_queries=8,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
    )
    bridge = SO400MVisualBridge(config)
    dense_tokens = torch.randn(3, 17, 32)

    output = bridge(dense_tokens)
    assert output.shape == (3, 8, 64)
    assert torch.isfinite(output).all()
    assert bridge.trainable_parameter_count > 0

    target = torch.randn_like(output)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in bridge.parameters()
        if parameter.requires_grad
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)


def test_mask_contract() -> None:
    config = VisualBridgeConfig(
        input_dim=16,
        output_dim=24,
        num_queries=4,
        depth=1,
        num_heads=4,
    )
    bridge = SO400MVisualBridge(config)
    dense_tokens = torch.randn(2, 7, 16)
    mask = torch.zeros(2, 7, dtype=torch.bool)
    mask[:, -1] = True
    output = bridge(dense_tokens, key_padding_mask=mask)
    assert output.shape == (2, 4, 24)


def main() -> None:
    test_state_extraction()
    test_bridge_contract()
    test_mask_contract()
    print("SO400M visual bridge tests: OK")


if __name__ == "__main__":
    main()
