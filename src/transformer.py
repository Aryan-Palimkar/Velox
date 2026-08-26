from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .config import ModelConfig
from .decoder_layer import DecoderLayer
from .rmsnorm import RMSNorm
from .rope import RotaryEmbedding


class QwenDecoderModel(nn.Module):
    def __init__(self, config: ModelConfig, max_position: Optional[int] = None) -> None:
        super().__init__()
        self.config = config

        head_dim = config.hidden_size // config.num_attention_heads
        rope = RotaryEmbedding(
            head_dim=head_dim,
            base_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            max_position=max_position or config.max_position_embeddings,
        )

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

    def forward(self, input_ids: torch.Tensor, kv_cache, metadata) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                layer_idx=layer_idx,
                metadata=metadata,
            )
        return self.norm(hidden_states)


class QwenForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig, max_position: Optional[int] = None) -> None:
        super().__init__()
        self.config = config
        self.model = QwenDecoderModel(config, max_position=max_position)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, kv_cache, metadata) -> torch.Tensor:
        return self.model(input_ids=input_ids, kv_cache=kv_cache, metadata=metadata)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
