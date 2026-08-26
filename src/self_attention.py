from __future__ import annotations

import torch
from torch import nn

from .attention_prefill import paged_prefill_attention
from .decode_attention import paged_decode_attention


class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope: nn.Module,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.rope = rope

        total_qkv_dim = (num_heads + 2 * num_kv_heads) * self.head_dim
        self.qkv_proj = nn.Linear(hidden_size, total_qkv_dim, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.sm_scale = 1.0 / (self.head_dim**0.5)
        self._q_size = num_heads * self.head_dim
        self._kv_size = num_kv_heads * self.head_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache,
        layer_idx: int,
        metadata,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]

        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self._q_size, self._kv_size, self._kv_size], dim=-1)
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(num_tokens, self.num_kv_heads, self.head_dim)

        self.rope(q, k, metadata.positions)

        kv_cache.write(layer_idx, k, v, metadata.slot_mapping)

        k_scale = kv_cache.k_scale[layer_idx]
        v_scale = kv_cache.v_scale[layer_idx]

        if metadata.is_prefill:
            attn_out = paged_prefill_attention(
                query=q,
                k_cache=kv_cache.k_cache[layer_idx],
                v_cache=kv_cache.v_cache[layer_idx],
                block_tables=metadata.block_tables,
                cu_seq_lens_q=metadata.cu_seq_lens_q,
                kv_lens=metadata.seq_lens,
                max_q_len=metadata.max_q_len,
                sm_scale=self.sm_scale,
                page_size=kv_cache.block_size,
                k_scale=k_scale,
                v_scale=v_scale,
            )
        else:
            attn_out = paged_decode_attention(
                query=q,
                k_cache=kv_cache.k_cache[layer_idx],
                v_cache=kv_cache.v_cache[layer_idx],
                block_tables=metadata.block_tables,
                seq_lens=metadata.seq_lens,
                page_size=kv_cache.block_size,
                sm_scale=self.sm_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                workspace=metadata.decode_workspace,
                num_splits=metadata.num_splits,
            )

        return self.o_proj(attn_out.view(num_tokens, self.hidden_size))
