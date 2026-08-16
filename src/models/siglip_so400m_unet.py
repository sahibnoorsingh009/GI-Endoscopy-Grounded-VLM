from __future__ import annotations

import gc
import inspect
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoProcessor, SiglipVisionModel

from .siglip2_unet import ConvNormAct


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a large PyTorch checkpoint without eagerly reading every storage."""
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Classification checkpoint not found: {checkpoint_path}")

    parameters = inspect.signature(torch.load).parameters
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in parameters:
        kwargs["weights_only"] = True
    if "mmap" in parameters:
        kwargs["mmap"] = True

    try:
        checkpoint = torch.load(checkpoint_path, **kwargs)
    except RuntimeError as exc:
        # mmap is supported only for checkpoints saved with the modern zip format.
        if not kwargs.pop("mmap", False):
            raise
        checkpoint = torch.load(checkpoint_path, **kwargs)

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Expected the classification checkpoint to contain a mapping, "
            f"but found {type(checkpoint).__name__}."
        )
    return checkpoint


def _select_state_dict(
    checkpoint: dict[str, Any],
    requested_key: str = "auto",
) -> tuple[dict[str, torch.Tensor], str]:
    """Select EMA weights by default, while also accepting vision-only exports."""
    if requested_key != "auto":
        if requested_key not in checkpoint:
            raise KeyError(
                f"Checkpoint key '{requested_key}' was requested but is unavailable. "
                f"Available keys: {sorted(checkpoint)}"
            )
        state = checkpoint[requested_key]
        if not isinstance(state, dict):
            raise TypeError(f"Checkpoint entry '{requested_key}' is not a state dictionary.")
        return state, requested_key

    for key in ("ema_model", "vision_model", "model", "state_dict"):
        state = checkpoint.get(key)
        if isinstance(state, dict):
            return state, key

    # A raw state dictionary has tensor values at the top level.
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint, "<root>"

    raise KeyError(
        "Could not locate model weights. Expected one of: "
        "ema_model, vision_model, model, state_dict, or a raw state dictionary."
    )


def _map_vision_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map project checkpoint keys to ``SiglipVisionModel`` keys."""
    mapped: dict[str, torch.Tensor] = {}

    for original_key, value in state_dict.items():
        key = original_key
        while key.startswith("module."):
            key = key[len("module.") :]

        if key.startswith("backbone.vision_model."):
            key = "vision_model." + key[len("backbone.vision_model.") :]
        elif key.startswith("encoder.vision_model."):
            key = "vision_model." + key[len("encoder.vision_model.") :]
        elif key.startswith("vision_model."):
            pass
        else:
            continue

        mapped[key] = value

    if not mapped:
        sample = list(state_dict)[:10]
        raise KeyError(
            "The checkpoint contains no recognized SigLIP vision weights. "
            f"Example keys: {sample}"
        )
    return mapped


class SiglipSO400MUNet(nn.Module):
    """U-Net-like segmentation model initialized from the trained SO400M classifier.

    The original classification checkpoint contains a full SigLIP model, an EMA copy,
    and optimizer state. Only ``backbone.vision_model`` is loaded here. The text tower
    and classification head are deliberately excluded from the segmentation model.
    """

    def __init__(
        self,
        checkpoint: str = "google/siglip2-so400m-patch14-384",
        classification_checkpoint: str | None = None,
        classification_state: str = "auto",
        strict_classification_init: bool = True,
        load_base_pretrained: bool = True,
        feature_layers: list[int] | None = None,
        decoder_channels: int = 256,
        out_channels: int = 1,
        train_mode: str = "frozen",
        partial_last_n: int = 6,
    ) -> None:
        super().__init__()
        self.checkpoint = checkpoint
        self.classification_checkpoint = classification_checkpoint
        self.processor = AutoProcessor.from_pretrained(checkpoint)

        full_config = AutoConfig.from_pretrained(checkpoint)
        vision_config = getattr(full_config, "vision_config", full_config)
        if load_base_pretrained:
            self.encoder = SiglipVisionModel.from_pretrained(
                checkpoint,
                config=vision_config,
            )
        else:
            # Transfer runs immediately load either the classification encoder or
            # a preceding segmentation checkpoint, so downloading another full
            # pretrained model file would be redundant.
            self.encoder = SiglipVisionModel(vision_config)

        self.feature_layers = feature_layers or [6, 12, 20, 27]
        hidden = int(self.encoder.config.hidden_size)

        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden, decoder_channels),
                    nn.LayerNorm(decoder_channels),
                )
                for _ in self.feature_layers
            ]
        )
        self.fuse = nn.Sequential(
            ConvNormAct(decoder_channels * len(self.feature_layers), decoder_channels),
            ConvNormAct(decoder_channels, decoder_channels),
        )
        self.up1 = ConvNormAct(decoder_channels, 128)
        self.up2 = ConvNormAct(128, 64)
        self.up3 = ConvNormAct(64, 32)
        self.up4 = ConvNormAct(32, 16)
        self.head = nn.Conv2d(16, out_channels, 1)

        self.classification_init_report: dict[str, Any] | None = None
        if classification_checkpoint:
            self.load_classification_encoder(
                classification_checkpoint,
                state_key=classification_state,
                strict=strict_classification_init,
            )

        self.set_train_mode(train_mode, partial_last_n)

    def load_classification_encoder(
        self,
        checkpoint_path: str | Path,
        state_key: str = "auto",
        strict: bool = True,
    ) -> None:
        checkpoint = _load_checkpoint(checkpoint_path)
        state_dict, selected_key = _select_state_dict(checkpoint, state_key)
        vision_state = _map_vision_state_dict(state_dict)

        incompatible = self.encoder.load_state_dict(vision_state, strict=False)
        ignored_missing = {
            key for key in incompatible.missing_keys if key.endswith("position_ids")
        }
        missing = [key for key in incompatible.missing_keys if key not in ignored_missing]
        unexpected = list(incompatible.unexpected_keys)

        if strict and (missing or unexpected):
            raise RuntimeError(
                "SO400M classification weights did not match the vision encoder. "
                f"Missing keys: {missing[:20]}; unexpected keys: {unexpected[:20]}"
            )

        self.classification_init_report = {
            "path": str(checkpoint_path),
            "state_key": selected_key,
            "loaded_tensors": len(vision_state),
            "epoch": checkpoint.get("epoch"),
            "best_metric": checkpoint.get("best_metric"),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        }
        print("Loaded SO400M classification encoder:", self.classification_init_report)

        del vision_state, state_dict, checkpoint
        gc.collect()

    def _transformer_layers(self) -> nn.ModuleList:
        vision_model = getattr(self.encoder, "vision_model", None)
        nested_encoder = getattr(vision_model, "encoder", None)
        layers = getattr(nested_encoder, "layers", None)
        if layers is None:
            available = [name for name, _ in self.encoder.named_children()]
            raise AttributeError(
                "Could not locate SO400M transformer layers. "
                f"Top-level modules are: {available}"
            )
        return layers

    def set_train_mode(self, mode: str, partial_last_n: int = 6) -> None:
        valid_modes = {"frozen", "partial", "full"}
        if mode not in valid_modes:
            raise ValueError(f"Unknown train_mode: {mode}. Expected one of {sorted(valid_modes)}")

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        if mode == "full":
            for parameter in self.encoder.parameters():
                parameter.requires_grad = True
            return

        if mode == "frozen":
            return

        if partial_last_n < 1:
            raise ValueError("partial_last_n must be at least 1")

        layers = self._transformer_layers()
        for layer in layers[-min(partial_last_n, len(layers)) :]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

        vision_model = getattr(self.encoder, "vision_model", None)
        post_layernorm = getattr(vision_model, "post_layernorm", None)
        if post_layernorm is not None:
            for parameter in post_layernorm.parameters():
                parameter.requires_grad = True

    @staticmethod
    def _grid_shape(token_count: int) -> tuple[int, int]:
        side = int(round(token_count**0.5))
        if side * side != token_count:
            raise ValueError(
                f"SO400M returned {token_count} tokens, which cannot be reshaped "
                "to a square grid. Use the fixed 384x384 processor input."
            )
        return side, side

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_size: tuple[int, int] | None = None,
        **_: Any,
    ) -> torch.Tensor:
        outputs = self.encoder(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        selected = []

        for layer_idx, projection in zip(self.feature_layers, self.projections):
            if layer_idx >= len(hidden_states):
                raise IndexError(
                    f"feature layer {layer_idx} unavailable; model returned "
                    f"{len(hidden_states)} hidden-state tensors"
                )
            tokens = projection(hidden_states[layer_idx])
            batch, token_count, channels = tokens.shape
            grid_h, grid_w = self._grid_shape(token_count)
            selected.append(
                tokens.transpose(1, 2).reshape(batch, channels, grid_h, grid_w)
            )

        target_grid = selected[-1].shape[-2:]
        selected = [
            F.interpolate(x, size=target_grid, mode="bilinear", align_corners=False)
            if x.shape[-2:] != target_grid
            else x
            for x in selected
        ]

        x = self.fuse(torch.cat(selected, dim=1))
        for block in (self.up1, self.up2, self.up3, self.up4):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = block(x)
        if output_size is not None:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return self.head(x)
