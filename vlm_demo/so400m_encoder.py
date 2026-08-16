"""Frozen GI-trained SigLIP2 SO400M vision encoder utilities."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from torch import nn
from transformers import AutoConfig, AutoImageProcessor, SiglipVisionModel


DEFAULT_SO400M_MODEL_ID = "google/siglip2-so400m-patch14-384"
DEFAULT_SO400M_REPO_ID = (
    "Sahibnoor1/gi-siglip2-dino-hyperkvasir-checkpoints"
)
DEFAULT_SO400M_FILENAME = (
    "checkpoints/siglip2_so400m_384_supervised_v1/seed42/"
    "so400m_classifier_seed42_vision_ema.pt"
)


@dataclass(frozen=True)
class SO400MCheckpointSpec:
    """Locations required to reconstruct the frozen SO400M encoder."""

    model_id: str = DEFAULT_SO400M_MODEL_ID
    repo_id: str = DEFAULT_SO400M_REPO_ID
    filename: str = DEFAULT_SO400M_FILENAME
    environment_variable: str = "SO400M_CKPT"

    def resolve(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        local_files_only: bool = False,
    ) -> Path:
        candidate = checkpoint_path or os.environ.get(self.environment_variable)
        if candidate:
            path = Path(candidate).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(
                    f"SO400M checkpoint does not exist: {path}"
                )
            return path

        return Path(
            hf_hub_download(
                repo_id=self.repo_id,
                filename=self.filename,
                local_files_only=local_files_only,
            )
        )


@dataclass
class SO400MFeatures:
    """Dense and pooled outputs of the frozen encoder."""

    dense_tokens: torch.Tensor
    pooled_token: torch.Tensor


def _tensor_mapping(value: Any) -> dict[str, torch.Tensor] | None:
    if not isinstance(value, Mapping):
        return None
    tensors = {
        str(key): tensor
        for key, tensor in value.items()
        if torch.is_tensor(tensor)
    }
    return tensors or None


def extract_vision_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Extract the strict ``SiglipVisionModel`` state from known layouts."""

    direct = _tensor_mapping(checkpoint)
    if direct is not None:
        state = direct
    elif isinstance(checkpoint, Mapping):
        state = None
        for key in (
            "vision_model",
            "state_dict",
            "model_state_dict",
            "model",
            "ema_model",
        ):
            candidate = _tensor_mapping(checkpoint.get(key))
            if candidate is not None:
                state = candidate
                break
        if state is None:
            raise ValueError(
                "Checkpoint does not contain a recognized tensor state dictionary"
            )
    else:
        raise TypeError(
            f"Checkpoint must be a mapping, received {type(checkpoint).__name__}"
        )

    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}

    # A bare SigLIP transformer state uses embeddings.*, encoder.*, and head.*.
    # SiglipVisionModel adds one outer vision_model wrapper.
    if state and not any(key.startswith("vision_model.") for key in state):
        bare_prefixes = ("embeddings.", "encoder.", "post_layernorm.", "head.")
        if all(key.startswith(bare_prefixes) for key in state):
            state = {f"vision_model.{key}": value for key, value in state.items()}

    return state


def load_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        checkpoint = torch.load(path, map_location="cpu")
    return extract_vision_state(checkpoint)


class FrozenSO400MEncoder(nn.Module):
    """A permanently frozen wrapper around the GI-trained SO400M encoder."""

    expected_tokens = 729
    hidden_size = 1152

    def __init__(self, vision_model: SiglipVisionModel) -> None:
        super().__init__()
        self.vision_model = vision_model
        for parameter in self.vision_model.parameters():
            parameter.requires_grad = False
        self.vision_model.eval()

    def train(self, mode: bool = True) -> "FrozenSO400MEncoder":
        # The encoder is intentionally frozen in bridge phase 1.
        super().train(False)
        self.vision_model.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> SO400MFeatures:
        outputs = self.vision_model(pixel_values=pixel_values)
        dense_tokens = outputs.last_hidden_state
        pooled_token = outputs.pooler_output

        if dense_tokens.ndim != 3:
            raise RuntimeError(
                f"Expected rank-3 dense tokens, got {tuple(dense_tokens.shape)}"
            )
        if dense_tokens.shape[1:] != (self.expected_tokens, self.hidden_size):
            raise RuntimeError(
                "Unexpected SO400M feature contract: "
                f"expected (*, {self.expected_tokens}, {self.hidden_size}), "
                f"got {tuple(dense_tokens.shape)}"
            )
        if pooled_token.shape != (dense_tokens.shape[0], self.hidden_size):
            raise RuntimeError(
                f"Unexpected pooled-token shape: {tuple(pooled_token.shape)}"
            )

        return SO400MFeatures(
            dense_tokens=dense_tokens,
            pooled_token=pooled_token,
        )


def build_frozen_so400m_encoder(
    checkpoint_path: str | Path | None = None,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    spec: SO400MCheckpointSpec | None = None,
    local_files_only: bool = False,
) -> tuple[FrozenSO400MEncoder, Any, Path]:
    """Load the exact GI checkpoint and its matching image processor."""

    spec = spec or SO400MCheckpointSpec()
    resolved_path = spec.resolve(
        checkpoint_path,
        local_files_only=local_files_only,
    )

    full_config = AutoConfig.from_pretrained(
        spec.model_id,
        local_files_only=local_files_only,
    )
    vision_config = full_config.vision_config
    vision_model = SiglipVisionModel(vision_config)

    state = load_checkpoint_state(resolved_path)
    vision_model.load_state_dict(state, strict=True)
    del state

    wrapper = FrozenSO400MEncoder(vision_model)
    wrapper.to(device=device, dtype=dtype)
    wrapper.eval()

    processor = AutoImageProcessor.from_pretrained(
        spec.model_id,
        local_files_only=local_files_only,
    )
    return wrapper, processor, resolved_path
