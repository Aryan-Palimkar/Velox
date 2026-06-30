import torch
from torch import nn
from .self_attention import SelfAttention
from .mlp import MLP
from .rmsnorm import RMSNorm
from .rope import RotaryEmbedding
from engine.utils import BatchMetadata

class DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        rms_norm_eps: float,
        rope,
    ) -> None:
        super().__init__()

        self.self_attn = SelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rope=rope,
            bias=True
        )

        self.mlp = MLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv_cache,
        layer_idx: int,
        prefill: bool,
        metadata: 'BatchMetadata',
    ) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)

        x = self.self_attn(
            hidden_states=x,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            prefill=prefill,
            metadata=metadata,
            positions=positions
        )
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)

        x = self.mlp(x)
        x = residual + x

        return x
