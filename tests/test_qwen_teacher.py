from __future__ import annotations

import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from vlm_demo.qwen_teacher import (  # noqa: E402
    resample_qwen_image_tokens,
    split_qwen_image_tokens,
)


def test_split_and_resample() -> None:
    grid = torch.tensor([[1, 16, 16], [1, 24, 16]], dtype=torch.long)
    first = torch.arange(64 * 4, dtype=torch.float32).reshape(64, 4)
    second = torch.arange(96 * 4, dtype=torch.float32).reshape(96, 4)
    flat = torch.cat((first, second), dim=0)
    split = split_qwen_image_tokens(flat, grid)
    assert [tokens.shape for tokens in split] == [(64, 4), (96, 4)]

    fixed = resample_qwen_image_tokens(flat, grid, target_side=8)
    assert fixed.shape == (2, 64, 4)
    assert torch.equal(fixed[0], first)
    assert torch.isfinite(fixed).all()


def test_invalid_token_count() -> None:
    try:
        split_qwen_image_tokens(
            torch.zeros(63, 4),
            torch.tensor([[1, 16, 16]]),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched flattened token count should fail")


def main() -> None:
    test_split_and_resample()
    test_invalid_token_count()
    print("Qwen visual teacher contracts: OK")


if __name__ == "__main__":
    main()
