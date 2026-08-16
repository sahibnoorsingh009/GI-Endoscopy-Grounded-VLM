"""Trainable token resampler and projector for SO400M-to-Qwen alignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class VisualBridgeConfig:
    """Architecture of the first-stage SO400M-to-Qwen bridge."""

    input_dim: int = 1152
    output_dim: int = 4096
    num_queries: int = 64
    depth: int = 2
    num_heads: int = 16
    mlp_ratio: float = 2.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if self.num_queries <= 0 or self.depth <= 0:
            raise ValueError("num_queries and depth must be positive")
        if self.num_heads <= 0 or self.input_dim % self.num_heads:
            raise ValueError("num_heads must divide input_dim")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        intermediate_size = int(round(hidden_size * mlp_ratio))
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.layers(hidden_states)


class QueryResamplerBlock(nn.Module):
    """Cross-attend to patches, then mix the learned query tokens."""

    def __init__(self, config: VisualBridgeConfig) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(config.input_dim)
        self.context_norm = nn.LayerNorm(config.input_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.input_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        self.self_norm = nn.LayerNorm(config.input_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.input_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        self.mlp_norm = nn.LayerNorm(config.input_dim)
        self.mlp = FeedForward(
            hidden_size=config.input_dim,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
        )

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized_context = self.context_norm(context)
        cross_output = self.cross_attention(
            query=self.query_norm(queries),
            key=normalized_context,
            value=normalized_context,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        queries = queries + cross_output

        normalized_queries = self.self_norm(queries)
        self_output = self.self_attention(
            query=normalized_queries,
            key=normalized_queries,
            value=normalized_queries,
            need_weights=False,
        )[0]
        queries = queries + self_output
        queries = queries + self.mlp(self.mlp_norm(queries))
        return queries


class SO400MVisualBridge(nn.Module):
    """Compress 729 SO400M tokens and project them into Qwen text width."""

    def __init__(self, config: VisualBridgeConfig | None = None) -> None:
        super().__init__()
        self.config = config or VisualBridgeConfig()
        self.input_norm = nn.LayerNorm(self.config.input_dim)
        self.query_tokens = nn.Parameter(
            torch.empty(self.config.num_queries, self.config.input_dim)
        )
        self.blocks = nn.ModuleList(
            QueryResamplerBlock(self.config)
            for _ in range(self.config.depth)
        )
        self.output_norm = nn.LayerNorm(self.config.input_dim)
        self.projector = nn.Linear(
            self.config.input_dim,
            self.config.output_dim,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.query_tokens,
            mean=0.0,
            std=self.config.input_dim**-0.5,
        )
        nn.init.normal_(self.projector.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.projector.bias)

    def forward(
        self,
        dense_tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if dense_tokens.ndim != 3:
            raise ValueError(
                "dense_tokens must have shape [batch, tokens, hidden], "
                f"got {tuple(dense_tokens.shape)}"
            )
        if dense_tokens.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"Expected input width {self.config.input_dim}, "
                f"got {dense_tokens.shape[-1]}"
            )
        if key_padding_mask is not None:
            expected = dense_tokens.shape[:2]
            if key_padding_mask.shape != expected:
                raise ValueError(
                    f"Expected key_padding_mask shape {expected}, "
                    f"got {tuple(key_padding_mask.shape)}"
                )
            if key_padding_mask.dtype != torch.bool:
                raise TypeError("key_padding_mask must be boolean")

        if dense_tokens.device != self.query_tokens.device:
            raise ValueError(
                "dense_tokens and bridge must be on the same device: "
                f"got {dense_tokens.device} and {self.query_tokens.device}"
            )

        bridge_dtype = self.query_tokens.dtype
        context = self.input_norm(dense_tokens.to(dtype=bridge_dtype))
        queries = self.query_tokens.unsqueeze(0).expand(
            dense_tokens.shape[0], -1, -1
        )

        for block in self.blocks:
            queries = block(
                queries,
                context,
                key_padding_mask=key_padding_mask,
            )

        return self.projector(self.output_norm(queries))

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
