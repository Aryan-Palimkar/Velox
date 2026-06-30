import torch
import torch.nn as nn
import triton
from engine.utils import BatchMetadata
from .attention_prefill import _attn_prefill_optimized
from .decode_attention import decode_attention

def assert_sane(tensor: torch.Tensor, name: str):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"\nNaN/Inf detected inside {name}")
        import sys; sys.exit(1)

class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.rope = rope

        total_qkv_dim = (num_heads + 2 * num_kv_heads) * self.head_dim
        self.qkv_proj = nn.Linear(hidden_size, total_qkv_dim, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.sm_scale = 1.0 / (self.head_dim ** 0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache,
        layer_idx: int,
        prefill: bool,
        metadata: 'BatchMetadata',
        positions: torch.Tensor
    ) -> torch.Tensor:
        total_tokens = hidden_states.shape[0]

        qkv = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        k_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, k_size, k_size], dim=-1)
        q = q.contiguous().view(total_tokens, self.num_heads, self.head_dim)
        k = k.contiguous().view(total_tokens, self.num_kv_heads, self.head_dim)
        v = v.contiguous().view(total_tokens, self.num_kv_heads, self.head_dim)


        q, k = self.rope.apply_rotary_flat(q, k, positions)

        if prefill:
            cu_lens = metadata.cu_seq_lens_q_cpu
            history_lens = metadata.seq_lens_cpu
            cache_slots = metadata.cache_slots_cpu

            for i, slot in enumerate(metadata.cache_slots):
                start, end = cu_lens[i], cu_lens[i+1]
                seq_len = metadata.prompt_lens[i]
                history_len = history_lens[i]

                kv_cache.k_cache[layer_idx][slot, history_len:history_len + seq_len] = k[start:end]
                kv_cache.v_cache[layer_idx][slot, history_len:history_len + seq_len] = v[start:end]

                if layer_idx == 0:
                    kv_cache.seq_lens[slot] += seq_len

            out = torch.empty_like(q)
            BLOCK_M, BLOCK_N = 64, 64

            grid = (
                len(metadata.cache_slots),
                self.num_heads,
                triton.cdiv(metadata.max_q_len, BLOCK_M)
            )

            k_cache_4d = kv_cache.k_cache[layer_idx]
            v_cache_4d = kv_cache.v_cache[layer_idx]

            _attn_prefill_optimized[grid](
                q, k_cache_4d, v_cache_4d, out,
                metadata.cu_seq_lens_q, metadata.cu_seq_lens_kv, metadata.cache_slots,

                q.stride(0), q.stride(1), q.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                k_cache_4d.stride(0), k_cache_4d.stride(1), k_cache_4d.stride(2), k_cache_4d.stride(3),
                v_cache_4d.stride(0), v_cache_4d.stride(1), v_cache_4d.stride(2), v_cache_4d.stride(3),

                self.sm_scale,
                q_len_key=metadata.max_q_len, kv_len_key=metadata.max_kv_len,
                HEAD_DIM=self.head_dim,
                GROUP_SIZE=self.num_heads // self.num_kv_heads,
                IS_CAUSAL=True,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )

        else:
            kv_cache.write_decode(metadata.cache_slots, metadata.positions, layer_idx, k, v)


            out = decode_attention(
                q,
                kv_cache.k_cache[layer_idx],
                kv_cache.v_cache[layer_idx],
                metadata.cache_slots,
                metadata.seq_lens
            )

        out = out.view(total_tokens, self.hidden_size)
        return self.o_proj(out)
