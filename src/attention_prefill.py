from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attn_prefill_kernel(
    Q_ptr, K_cache_ptr, V_cache_ptr, O_ptr,
    K_scale_ptr, V_scale_ptr,
    cu_seq_lens_q_ptr, kv_lens_ptr, block_tables_ptr,

    stride_qt, stride_qh, stride_qd,
    stride_ot, stride_oh, stride_od,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_sc_slot, stride_sc_head,
    stride_bt_b, stride_bt_n,

    sm_scale,

    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    KV_QUANTIZED: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_q_idx = tl.program_id(1)
    m_block_idx = tl.program_id(2)

    head_k_idx = head_q_idx // GROUP_SIZE

    q_start = tl.load(cu_seq_lens_q_ptr + batch_idx)
    q_end = tl.load(cu_seq_lens_q_ptr + batch_idx + 1)
    q_len = q_end - q_start

    start_m = m_block_idx * BLOCK_M
    if start_m >= q_len:
        return

    kv_len = tl.load(kv_lens_ptr + batch_idx)
    ctx_len = kv_len - q_len

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = (
        Q_ptr
        + (q_start + offs_m[:, None]) * stride_qt
        + head_q_idx * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q_mask = offs_m[:, None] < q_len
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    max_valid_n = tl.minimum(kv_len, ctx_len + start_m + BLOCK_M)

    bt_base = block_tables_ptr + batch_idx * stride_bt_b

    for start_n in range(0, max_valid_n, BLOCK_N):
        n_offsets = start_n + offs_n
        n_mask = n_offsets < kv_len

        logical_block = n_offsets // PAGE_SIZE
        offset_in_block = n_offsets % PAGE_SIZE
        # int64 or the block offset overflows once the pool gets large
        physical_block = tl.load(
            bt_base + logical_block * stride_bt_n, mask=n_mask, other=0
        ).to(tl.int64)

        k_ptrs = (
            K_cache_ptr
            + physical_block[None, :] * stride_kb
            + offset_in_block[None, :] * stride_kt
            + head_k_idx * stride_kh
            + offs_d[:, None] * stride_kd
        )
        v_ptrs = (
            V_cache_ptr
            + physical_block[:, None] * stride_vb
            + offset_in_block[:, None] * stride_vt
            + head_k_idx * stride_vh
            + offs_d[None, :] * stride_vd
        )

        k = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0).to(q.dtype)
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0).to(q.dtype)

        slot = physical_block * PAGE_SIZE + offset_in_block

        qk = tl.dot(q, k).to(tl.float32) * sm_scale

        if KV_QUANTIZED:
            k_scale = tl.load(
                K_scale_ptr + slot * stride_sc_slot + head_k_idx * stride_sc_head,
                mask=n_mask, other=1.0,
            )
            # k scale varies along the score axis, so it applies after the dot
            qk = qk * k_scale[None, :]

        is_fully_past_block = (start_n + BLOCK_N) <= ctx_len
        if is_fully_past_block:
            qk = tl.where(n_mask[None, :], qk, float("-inf"))
        else:
            pos_m = ctx_len + offs_m
            causal_mask = pos_m[:, None] >= n_offsets[None, :]
            qk = tl.where(causal_mask & n_mask[None, :], qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        m_ij_safe = tl.where(m_ij == float("-inf"), 0.0, m_ij)
        p = tl.exp(qk - m_ij_safe[:, None])
        alpha = tl.exp(m_i - m_ij_safe)

        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        if KV_QUANTIZED:
            v_scale = tl.load(
                V_scale_ptr + slot * stride_sc_slot + head_k_idx * stride_sc_head,
                mask=n_mask, other=1.0,
            )
            # v scale varies along the contraction axis, so it folds into p instead
            p = p * v_scale[None, :]

        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_ij_safe

    acc = acc / tl.where(l_i > 0.0, l_i, 1.0)[:, None]

    o_ptrs = (
        O_ptr
        + (q_start + offs_m[:, None]) * stride_ot
        + head_q_idx * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_mask)


def paged_prefill_attention(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    cu_seq_lens_q: torch.Tensor,
    kv_lens: torch.Tensor,
    max_q_len: int,
    sm_scale: float,
    page_size: int,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor:
    total_tokens, num_q_heads, head_dim = query.shape
    num_kv_heads = k_cache.shape[2]
    batch_size = block_tables.shape[0]

    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads ({num_q_heads}) must be a multiple of num_kv_heads ({num_kv_heads})"
        )

    if out is None:
        out = torch.empty_like(query)

    quantized = k_scale is not None
    scale_slot_stride = k_scale.stride(0) if quantized else 0
    scale_head_stride = k_scale.stride(1) if quantized else 0

    grid = (batch_size, num_q_heads, triton.cdiv(max(max_q_len, 1), block_m))

    _paged_attn_prefill_kernel[grid](
        query, k_cache, v_cache, out,
        k_scale if quantized else query,
        v_scale if quantized else query,
        cu_seq_lens_q, kv_lens, block_tables,

        query.stride(0), query.stride(1), query.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        scale_slot_stride, scale_head_stride,
        block_tables.stride(0), block_tables.stride(1),

        sm_scale,

        HEAD_DIM=head_dim,
        GROUP_SIZE=num_q_heads // num_kv_heads,
        PAGE_SIZE=page_size,
        KV_QUANTIZED=quantized,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=2,
    )
    return out
