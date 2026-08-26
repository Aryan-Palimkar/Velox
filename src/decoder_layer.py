from __future__ import annotations

import torch
from torch import nn

from .mlp import MLP
from .rmsnorm import RMSNorm
from .self_attention import SelfAttention


class DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        rms_norm_eps: float,
        rope: nn.Module,
    ) -> None:
        super().__init__()

        self.self_attn = SelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rope=rope,
            bias=True,
        )
        self.mlp = MLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache,
        layer_idx: int,
        metadata,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            metadata=metadata,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states
