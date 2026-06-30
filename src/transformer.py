from typing import Optional
import torch
from torch import nn
from .config import ModelConfig
from .decoder_layer import DecoderLayer
from .kv_cache import KVCache
from .rmsnorm import RMSNorm
from .rope import RotaryEmbedding
from engine.utils import BatchMetadata

def check_nan(tensor: torch.Tensor, name: str, layer_idx=None) -> bool:
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        prefix = f"Layer {layer_idx} | " if layer_idx is not None else ""
        print(f"NaN/Inf detected at -> {prefix}{name}")
        return True
    return False

def assert_sane(tensor: torch.Tensor, name: str):
    if torch.isnan(tensor).any():
        print(f"\n NaN detected inside {name}")
        import sys; sys.exit(1)
    if torch.isinf(tensor).any():
        print(f"\nInf detected inside {name}")
        import sys; sys.exit(1)


class QwenDecoderModel(nn.Module):
    def __init__(self, config: 'ModelConfig') -> None:
        super().__init__()
        self.config = config

        head_dim = config.hidden_size // config.num_attention_heads
        rope = RotaryEmbedding(head_dim, config.rope_theta, config.rope_scaling)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    intermediate_size=config.intermediate_size,
                    rms_norm_eps=config.rms_norm_eps,
                    rope=rope,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids, kv_cache, prefill, metadata, positions):
        x = self.embed_tokens(input_ids)

        for layer_idx, layer in enumerate(self.layers):
            residual = x
            x = layer.input_layernorm(x)

            attn_out = layer.self_attn(
                hidden_states=x,
                kv_cache=kv_cache,
                layer_idx=layer_idx,
                prefill=prefill,
                metadata=metadata,
                positions=positions
            )

            x = residual + attn_out

            residual = x
            x = layer.post_attention_layernorm(x)

            mlp_out = layer.mlp(x)

            x = residual + mlp_out

        x = self.norm(x)

        return x


class QwenForCausalLM(nn.Module):
    def __init__(self, config: 'ModelConfig') -> None:
        super().__init__()
        self.model = QwenDecoderModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache,
        prefill: bool,
        metadata: 'BatchMetadata',
        positions: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            kv_cache=kv_cache,
            prefill=prefill,
            metadata=metadata,
            positions=positions
        )
        return self.lm_head(hidden_states)
