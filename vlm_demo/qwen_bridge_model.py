"""Qwen3-VL wrapper that replaces its visual tower with the GI SO400M bridge."""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from vlm_demo.so400m_encoder import (
    FrozenSO400MEncoder,
    build_frozen_so400m_encoder,
)
from vlm_demo.visual_bridge import SO400MVisualBridge, VisualBridgeConfig


DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
BRIDGE_WEIGHTS_NAME = "visual_bridge.safetensors"
BRIDGE_CONFIG_NAME = "visual_bridge_config.json"


class SO400MBridgeVisionAdapter(nn.Module):
    """Expose SO400M bridge outputs through Qwen's visual-tower contract."""

    spatial_merge_size = 2

    def __init__(
        self,
        encoder: FrozenSO400MEncoder,
        bridge: SO400MVisualBridge,
        *,
        deepstack_layers: int = 3,
    ) -> None:
        super().__init__()
        if deepstack_layers < 0:
            raise ValueError("deepstack_layers must be non-negative")
        self.encoder = encoder
        self.bridge = bridge
        self.deepstack_scales = nn.Parameter(torch.zeros(deepstack_layers))

    @property
    def dtype(self) -> torch.dtype:
        return self.bridge.query_tokens.dtype

    @property
    def device(self) -> torch.device:
        return self.bridge.query_tokens.device

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def set_bridge_trainable(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.bridge.parameters():
            parameter.requires_grad = True
        self.deepstack_scales.requires_grad = True

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        ]

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if grid_thw is None:
            raise ValueError("grid_thw is required for bridged Qwen images")
        if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
            raise ValueError(
                f"Expected grid_thw shape [images, 3], got {tuple(grid_thw.shape)}"
            )
        if pixel_values.ndim != 4 or pixel_values.shape[1:] != (3, 384, 384):
            raise ValueError(
                "Bridged pixel_values must have shape [images, 3, 384, 384], "
                f"got {tuple(pixel_values.shape)}"
            )
        if pixel_values.shape[0] != grid_thw.shape[0]:
            raise ValueError(
                "The number of SO400M images and Qwen grids must match: "
                f"{pixel_values.shape[0]} != {grid_thw.shape[0]}"
            )

        expected_tokens = (
            grid_thw.to(dtype=torch.long).prod(dim=-1)
            // self.spatial_merge_size**2
        )
        configured_tokens = self.bridge.config.num_queries
        if not torch.all(expected_tokens == configured_tokens):
            raise ValueError(
                "Every Qwen image grid must describe exactly "
                f"{configured_tokens} merged tokens; got {expected_tokens.tolist()}"
            )

        features = self.encoder(pixel_values)
        projected = self.bridge(features.dense_tokens)
        if projected.shape[:2] != (pixel_values.shape[0], configured_tokens):
            raise RuntimeError(
                f"Unexpected bridge output shape: {tuple(projected.shape)}"
            )

        flattened = projected.reshape(-1, projected.shape[-1])
        deepstack = [
            flattened * scale.to(dtype=flattened.dtype)
            for scale in self.deepstack_scales
        ]
        return flattened, deepstack


@dataclass
class QwenBridgeBundle:
    model: nn.Module
    qwen_processor: Any
    so400m_processor: Any
    vision_adapter: SO400MBridgeVisionAdapter
    so400m_checkpoint: Path


def _bridge_state(adapter: SO400MBridgeVisionAdapter) -> dict[str, torch.Tensor]:
    state = {
        f"bridge.{key}": value.detach().cpu().contiguous()
        for key, value in adapter.bridge.state_dict().items()
    }
    state["deepstack_scales"] = (
        adapter.deepstack_scales.detach().cpu().contiguous()
    )
    return state


def save_visual_bridge(
    adapter: SO400MBridgeVisionAdapter,
    output_dir: str | Path,
    *,
    metadata: dict[str, object] | None = None,
) -> Path:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / BRIDGE_WEIGHTS_NAME
    save_file(_bridge_state(adapter), weights_path)

    payload: dict[str, object] = {
        "architecture": "SO400MBridgeVisionAdapter",
        "bridge": adapter.bridge.config.to_dict(),
        "deepstack_layers": int(adapter.deepstack_scales.numel()),
        "spatial_merge_size": adapter.spatial_merge_size,
    }
    if metadata:
        payload["metadata"] = metadata
    (output_dir / BRIDGE_CONFIG_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return weights_path


def load_visual_bridge(
    adapter: SO400MBridgeVisionAdapter,
    checkpoint: str | Path,
) -> Path:
    checkpoint = Path(checkpoint).expanduser().resolve()
    weights_path = (
        checkpoint / BRIDGE_WEIGHTS_NAME if checkpoint.is_dir() else checkpoint
    )
    if not weights_path.is_file():
        raise FileNotFoundError(f"Visual bridge weights not found: {weights_path}")

    state = load_file(weights_path, device="cpu")
    bridge_state = {
        key.removeprefix("bridge."): value
        for key, value in state.items()
        if key.startswith("bridge.")
    }
    adapter.bridge.load_state_dict(bridge_state, strict=True)

    if "deepstack_scales" not in state:
        raise ValueError("Bridge checkpoint is missing deepstack_scales")
    if state["deepstack_scales"].shape != adapter.deepstack_scales.shape:
        raise ValueError(
            "DeepStack scale shape changed: "
            f"{tuple(state['deepstack_scales'].shape)} != "
            f"{tuple(adapter.deepstack_scales.shape)}"
        )
    adapter.deepstack_scales.data.copy_(
        state["deepstack_scales"].to(adapter.deepstack_scales)
    )
    return weights_path


def build_qwen_bridge_bundle(
    *,
    qwen_adapter_path: str | Path,
    so400m_checkpoint_path: str | Path | None = None,
    bridge_checkpoint_path: str | Path | None = None,
    qwen_model_id: str = DEFAULT_QWEN_MODEL_ID,
    bridge_config: VisualBridgeConfig | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "flash_attention_2",
    use_cache: bool = False,
) -> QwenBridgeBundle:
    """Build Qwen + frozen LoRA + frozen SO400M + trainable bridge."""

    from peft import PeftModel

    qwen_adapter_path = Path(qwen_adapter_path).expanduser().resolve()
    if not (qwen_adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"Qwen adapter_config.json not found under {qwen_adapter_path}"
        )

    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        qwen_model_id,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    # Load PEFT before replacing the native visual module. Otherwise PEFT's
    # generic q_proj/k_proj matching would also modify the SO400M encoder.
    model = PeftModel.from_pretrained(
        qwen,
        qwen_adapter_path,
        is_trainable=False,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False

    encoder, so400m_processor, resolved_so400m = build_frozen_so400m_encoder(
        so400m_checkpoint_path,
        device="cpu",
        dtype=dtype,
    )
    bridge = SO400MVisualBridge(bridge_config or VisualBridgeConfig())
    bridge.to(dtype=dtype)
    vision_adapter = SO400MBridgeVisionAdapter(encoder, bridge)
    if bridge_checkpoint_path is not None:
        load_visual_bridge(vision_adapter, bridge_checkpoint_path)

    # Drop Qwen's original visual tower and install the trained GI encoder.
    qwen.model.visual = vision_adapter
    gc.collect()

    vision_adapter.set_bridge_trainable()
    qwen.config.use_cache = use_cache
    model.config.use_cache = use_cache

    # The frozen LoRA adapter was trained with dropout, but bridge alignment
    # should see the deterministic inference-time Qwen backbone.
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0

    expected_trainable = {id(p) for p in vision_adapter.trainable_parameters()}
    actual_trainable = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    if actual_trainable != expected_trainable:
        raise RuntimeError(
            "Only the visual bridge and DeepStack gates may be trainable"
        )

    qwen_processor = AutoProcessor.from_pretrained(qwen_model_id)
    if device is not None:
        model.to(device)

    return QwenBridgeBundle(
        model=model,
        qwen_processor=qwen_processor,
        so400m_processor=so400m_processor,
        vision_adapter=vision_adapter,
        so400m_checkpoint=resolved_so400m,
    )
