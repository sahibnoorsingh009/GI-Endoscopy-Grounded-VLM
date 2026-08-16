from __future__ import annotations
from math import prod
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoProcessor, Siglip2VisionModel

class ConvNormAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.GELU(),
        )

class Siglip2UNet(nn.Module):
    def __init__(
        self,
        checkpoint: str,
        feature_layers: list[int],
        decoder_channels: int = 256,
        out_channels: int = 1,
        train_mode: str = "full",
        partial_last_n: int = 4,
        load_base_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.checkpoint = checkpoint
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        if load_base_pretrained:
            self.encoder = Siglip2VisionModel.from_pretrained(checkpoint)
        else:
            full_config = AutoConfig.from_pretrained(checkpoint)
            vision_config = getattr(full_config, "vision_config", full_config)
            self.encoder = Siglip2VisionModel(vision_config)
        self.feature_layers = feature_layers
        hidden = int(self.encoder.config.hidden_size)

        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, decoder_channels),
                nn.LayerNorm(decoder_channels),
            )
            for _ in feature_layers
        ])
        self.fuse = nn.Sequential(
            ConvNormAct(decoder_channels * len(feature_layers), decoder_channels),
            ConvNormAct(decoder_channels, decoder_channels),
        )
        self.up1 = ConvNormAct(decoder_channels, 128)
        self.up2 = ConvNormAct(128, 64)
        self.up3 = ConvNormAct(64, 32)
        self.up4 = ConvNormAct(32, 16)
        self.head = nn.Conv2d(16, out_channels, 1)
        self.set_train_mode(train_mode, partial_last_n)

    def set_train_mode(self, mode: str, partial_last_n: int) -> None:
        """
        Configure encoder fine-tuning.

        Supports both older and newer Hugging Face Transformers layouts:
        - older: encoder.vision_model.encoder.layers
        - newer: encoder.encoder.layers
        """
        valid_modes = {"frozen", "partial", "full"}
        if mode not in valid_modes:
            raise ValueError(
                f"Unknown train_mode: {mode}. Expected one of {sorted(valid_modes)}"
            )

        # Freeze everything first.
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        if mode == "frozen":
            return

        if mode == "full":
            for parameter in self.encoder.parameters():
                parameter.requires_grad = True
            return

        # Find transformer blocks across Transformers API versions.
        layer_candidates = []

        direct_encoder = getattr(self.encoder, "encoder", None)
        if direct_encoder is not None:
            layer_candidates.append(getattr(direct_encoder, "layers", None))

        vision_model = getattr(self.encoder, "vision_model", None)
        if vision_model is not None:
            nested_encoder = getattr(vision_model, "encoder", None)
            if nested_encoder is not None:
                layer_candidates.append(getattr(nested_encoder, "layers", None))

        model = getattr(self.encoder, "model", None)
        if model is not None:
            nested_encoder = getattr(model, "encoder", None)
            if nested_encoder is not None:
                layer_candidates.append(getattr(nested_encoder, "layers", None))

        layers = next(
            (candidate for candidate in layer_candidates if candidate is not None),
            None,
        )

        if layers is None:
            available = [name for name, _ in self.encoder.named_children()]
            raise AttributeError(
                "Could not locate SigLIP2 transformer layers. "
                f"Top-level modules are: {available}"
            )

        if partial_last_n < 1:
            raise ValueError("partial_last_n must be at least 1")

        number_to_unfreeze = min(partial_last_n, len(layers))
        for layer in layers[-number_to_unfreeze:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

        # Also unfreeze the final normalization layer when available.
        normalization_candidates = [
            getattr(self.encoder, "post_layernorm", None),
            getattr(vision_model, "post_layernorm", None)
            if vision_model is not None
            else None,
        ]

        for module in normalization_candidates:
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad = True

    @staticmethod
    def _grid_shape(spatial_shapes: torch.Tensor | None, token_count: int) -> tuple[int, int]:
        if spatial_shapes is not None:
            h, w = [int(x) for x in spatial_shapes[0].tolist()]
            if h * w == token_count:
                return h, w
        side = int(round(token_count ** 0.5))
        if side * side != token_count:
            raise ValueError(
                f"Cannot infer token grid from {token_count} tokens. "
                "Use a square fixed-resolution batch or provide spatial_shapes."
            )
        return side, side

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        output_size: tuple[int, int] | None = None,
        **_: dict,
    ) -> torch.Tensor:
        outputs = self.encoder(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
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
            tokens = hidden_states[layer_idx]
            tokens = projection(tokens)
            batch, token_count, channels = tokens.shape
            grid_h, grid_w = self._grid_shape(spatial_shapes, token_count)
            fmap = tokens.transpose(1, 2).reshape(batch, channels, grid_h, grid_w)
            selected.append(fmap)

        target_grid = selected[-1].shape[-2:]
        selected = [
            F.interpolate(x, size=target_grid, mode="bilinear", align_corners=False)
            if x.shape[-2:] != target_grid else x
            for x in selected
        ]
        x = self.fuse(torch.cat(selected, dim=1))
        for block in (self.up1, self.up2, self.up3, self.up4):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = block(x)
        if output_size is not None:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return self.head(x)
