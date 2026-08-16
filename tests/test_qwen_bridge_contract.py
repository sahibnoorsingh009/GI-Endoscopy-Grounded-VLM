from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from vlm_demo.qwen_bridge_data import (  # noqa: E402
    IGNORE_INDEX,
    build_assistant_labels,
    expand_image_tokens,
    image_grid_thw,
)
from vlm_demo.qwen_bridge_model import (  # noqa: E402
    SO400MBridgeVisionAdapter,
)


class FakeTokenizer:
    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|im_start|>": 10, "<|im_end|>": 11}[token]

    def encode(self, value: str, add_special_tokens: bool = False) -> list[int]:
        assert value == "assistant\n"
        assert not add_special_tokens
        return [20, 21]


class FakeEncoder(nn.Module):
    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        batch = pixel_values.shape[0]
        return SimpleNamespace(
            dense_tokens=torch.ones(batch, 7, 8),
            pooled_token=torch.ones(batch, 8),
        )


class FakeBridge(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query_tokens = nn.Parameter(torch.ones(4, 8))
        self.config = SimpleNamespace(num_queries=4, output_dim=16)

    def forward(self, dense_tokens: torch.Tensor) -> torch.Tensor:
        pooled = dense_tokens.mean(dim=1, keepdim=True)
        value = pooled.mean(dim=-1, keepdim=True) * self.query_tokens.mean()
        return value.expand(dense_tokens.shape[0], 4, 16)


def test_token_expansion_and_labels() -> None:
    original = torch.tensor([1, 99, 2], dtype=torch.long)
    expanded = expand_image_tokens(
        original,
        image_token_id=99,
        num_queries=4,
    )
    assert expanded.tolist() == [1, 99, 99, 99, 99, 2]

    conversation = torch.tensor(
        [10, 30, 21, 7, 11, 10, 20, 21, 42, 43, 11],
        dtype=torch.long,
    )
    labels = build_assistant_labels(conversation, FakeTokenizer())
    supervised = torch.nonzero(labels != IGNORE_INDEX).flatten().tolist()
    assert supervised == [8, 9, 10]
    assert labels[8:11].tolist() == [42, 43, 11]


def test_grid_contract() -> None:
    assert image_grid_thw(64).tolist() == [1, 16, 16]
    try:
        image_grid_thw(63)
    except ValueError:
        pass
    else:
        raise AssertionError("Non-square query count should fail")


def test_qwen_visual_contract() -> None:
    adapter = SO400MBridgeVisionAdapter(
        FakeEncoder(),
        FakeBridge(),
        deepstack_layers=3,
    )
    pixels = torch.zeros(2, 3, 384, 384)
    grid = torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.long)
    projected, deepstack = adapter(pixels, grid_thw=grid)

    assert projected.shape == (8, 16)
    assert len(deepstack) == 3
    assert all(tensor.shape == (8, 16) for tensor in deepstack)
    assert all(torch.count_nonzero(tensor) == 0 for tensor in deepstack)

    projected.sum().backward()
    assert adapter.bridge.query_tokens.grad is not None
    assert adapter.encoder.training


def main() -> None:
    test_token_expansion_and_labels()
    test_grid_contract()
    test_qwen_visual_contract()
    print("Qwen SO400M bridge contract tests: OK")


if __name__ == "__main__":
    main()
