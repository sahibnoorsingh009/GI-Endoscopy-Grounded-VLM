"""Frozen native-Qwen visual teacher utilities for SO400M bridge distillation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from vlm_demo.qwen_bridge_model import DEFAULT_QWEN_MODEL_ID


@dataclass
class QwenVisualTeacherBundle:
    model: nn.Module
    visual: nn.Module
    processor: Any
    adapter_path: Path


def build_qwen_visual_teacher(
    *,
    qwen_adapter_path: str | Path,
    qwen_model_id: str = DEFAULT_QWEN_MODEL_ID,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "flash_attention_2",
) -> QwenVisualTeacherBundle:
    """Load the selected native-Qwen visual tower with its frozen LoRA."""

    from peft import PeftModel

    adapter_path = Path(qwen_adapter_path).expanduser().resolve()
    if not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"Qwen adapter_config.json not found under {adapter_path}"
        )

    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        qwen_model_id,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    model = PeftModel.from_pretrained(
        qwen,
        adapter_path,
        is_trainable=False,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.to(device)
    model.eval()

    core_model = model.get_base_model()
    visual = core_model.model.visual
    visual.eval()
    processor = AutoProcessor.from_pretrained(qwen_model_id)
    return QwenVisualTeacherBundle(
        model=model,
        visual=visual,
        processor=processor,
        adapter_path=adapter_path,
    )


def preprocess_qwen_teacher_images(
    processor: Any,
    images: Sequence[Image.Image],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not images:
        raise ValueError("At least one teacher image is required")
    processed = processor.image_processor(
        images=list(images),
        return_tensors="pt",
    )
    if "pixel_values" not in processed or "image_grid_thw" not in processed:
        raise KeyError(
            "Qwen image processor must return pixel_values and image_grid_thw"
        )
    return processed["pixel_values"], processed["image_grid_thw"]


def split_qwen_image_tokens(
    flat_tokens: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    spatial_merge_size: int = 2,
) -> list[torch.Tensor]:
    """Split Qwen's flattened visual output into one tensor per image."""

    if flat_tokens.ndim != 2:
        raise ValueError(
            f"flat_tokens must have shape [tokens, hidden], got {tuple(flat_tokens.shape)}"
        )
    if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
        raise ValueError(f"grid_thw must have shape [images, 3], got {tuple(grid_thw.shape)}")
    if spatial_merge_size <= 0:
        raise ValueError("spatial_merge_size must be positive")

    grid = grid_thw.to(dtype=torch.long, device="cpu")
    divisor = spatial_merge_size**2
    products = grid.prod(dim=-1)
    if torch.any(products % divisor):
        raise ValueError("Every Qwen grid must be divisible by spatial_merge_size squared")
    sizes = (products // divisor).tolist()
    if sum(sizes) != flat_tokens.shape[0]:
        raise ValueError(
            f"Qwen token/grid mismatch: {flat_tokens.shape[0]} != {sum(sizes)}"
        )
    return list(torch.split(flat_tokens, sizes, dim=0))


def resample_qwen_image_tokens(
    flat_tokens: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    target_side: int = 8,
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    """Convert dynamic native-Qwen grids into fixed raster-ordered tokens."""

    if target_side <= 0:
        raise ValueError("target_side must be positive")
    split_tokens = split_qwen_image_tokens(
        flat_tokens,
        grid_thw,
        spatial_merge_size=spatial_merge_size,
    )
    outputs: list[torch.Tensor] = []
    for tokens, raw_grid in zip(split_tokens, grid_thw.tolist(), strict=True):
        temporal, height, width = (int(value) for value in raw_grid)
        if height % spatial_merge_size or width % spatial_merge_size:
            raise ValueError(
                f"Qwen spatial grid is not merge-aligned: {raw_grid}"
            )
        merged_height = height // spatial_merge_size
        merged_width = width // spatial_merge_size
        spatial = tokens.reshape(
            temporal,
            merged_height,
            merged_width,
            tokens.shape[-1],
        ).mean(dim=0)
        spatial = spatial.permute(2, 0, 1).unsqueeze(0)
        pooled = F.adaptive_avg_pool2d(spatial.float(), (target_side, target_side))
        fixed = pooled.squeeze(0).permute(1, 2, 0).reshape(
            target_side * target_side,
            tokens.shape[-1],
        )
        outputs.append(fixed.to(dtype=tokens.dtype))
    return torch.stack(outputs)


@torch.inference_mode()
def extract_qwen_teacher_tokens(
    bundle: QwenVisualTeacherBundle,
    images: Sequence[Image.Image],
    *,
    target_side: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    pixel_values, grid_thw = preprocess_qwen_teacher_images(
        bundle.processor,
        images,
    )
    device = next(bundle.visual.parameters()).device
    dtype = bundle.visual.dtype
    pixel_values = pixel_values.to(device=device, dtype=dtype)
    grid_thw = grid_thw.to(device=device)
    visual_output = bundle.visual(pixel_values, grid_thw=grid_thw)
    if not isinstance(visual_output, tuple) or len(visual_output) != 2:
        raise TypeError("Qwen visual tower must return (tokens, deepstack_tokens)")
    flat_tokens, deepstack = visual_output
    fixed = resample_qwen_image_tokens(
        flat_tokens,
        grid_thw,
        target_side=target_side,
        spatial_merge_size=int(bundle.visual.spatial_merge_size),
    )
    return fixed, grid_thw, deepstack
