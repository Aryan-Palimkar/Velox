from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

_MIN_TOKENS_FOR_SPLIT = 512
_MAX_SPLITS = 16


@triton.jit
def _paged_decode_attn_kernel(
    Q_ptr, K_cache_ptr, V_cache_ptr,
    K_scale_ptr, V_scale_ptr,
    block_tables_ptr, seq_lens_ptr,
    O_ptr, partial_o_ptr, partial_lse_ptr,

    stride_qb, stride_qh, stride_qd,
    stride_ob, stride_oh, stride_od,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_sc_slot, stride_sc_head,
    stride_bt_b, stride_bt_n,
    stride_pb, stride_ph, stride_ps, stride_pr, stride_pd,
    stride_lb, stride_lh, stride_ls, stride_lr,

    scale,

    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KV_QUANTIZED: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    WRITE_PARTIAL: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_k_idx = tl.program_id(1)
    split_idx = tl.program_id(2)

    seq_len = tl.load(seq_lens_ptr + batch_idx)

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)
    h_mask = offs_h < GROUP_SIZE

    tokens_per_split = (seq_len + NUM_SPLITS - 1) // NUM_SPLITS
    split_start = split_idx * tokens_per_split
    split_end = tl.minimum(split_start + tokens_per_split, seq_len)

    head_q = head_k_idx * GROUP_SIZE + offs_h
    q_ptrs = (
        Q_ptr
        + batch_idx * stride_qb
        + head_q[:, None] * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0)

    m_i = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, HEAD_DIM], dtype=tl.float32)

    bt_base = block_tables_ptr + batch_idx * stride_bt_b

    for start_n in range(split_start, split_end, BLOCK_N):
        n_offsets = start_n + offs_n
        n_mask = n_offsets < split_end

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

        qk = tl.dot(q, k).to(tl.float32) * scale

        if KV_QUANTIZED:
            k_scale = tl.load(
                K_scale_ptr + slot * stride_sc_slot + head_k_idx * stride_sc_head,
                mask=n_mask, other=1.0,
            )
            # k scale varies along the score axis, so it applies after the dot
            qk = qk * k_scale[None, :]

        qk = tl.where(n_mask[None, :], qk, float("-inf"))

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

    if WRITE_PARTIAL:
        out = acc / tl.where(l_i > 0.0, l_i, 1.0)[:, None]
        lse = tl.where(l_i > 0.0, m_i + tl.log(tl.where(l_i > 0.0, l_i, 1.0)), float("-inf"))

        p_ptrs = (
            partial_o_ptr
            + batch_idx * stride_pb
            + head_k_idx * stride_ph
            + split_idx * stride_ps
            + offs_h[:, None] * stride_pr
            + offs_d[None, :] * stride_pd
        )
        tl.store(p_ptrs, out, mask=h_mask[:, None])

        l_ptrs = (
            partial_lse_ptr
            + batch_idx * stride_lb
            + head_k_idx * stride_lh
            + split_idx * stride_ls
            + offs_h * stride_lr
        )
        tl.store(l_ptrs, lse, mask=h_mask)
    else:
        out = acc / tl.where(l_i > 0.0, l_i, 1.0)[:, None]
        o_ptrs = (
            O_ptr
            + batch_idx * stride_ob
            + head_q[:, None] * stride_oh
            + offs_d[None, :] * stride_od
        )
        tl.store(o_ptrs, out.to(O_ptr.dtype.element_ty), mask=h_mask[:, None])


@triton.jit
def _decode_split_reduce_kernel(
    partial_o_ptr, partial_lse_ptr, O_ptr,

    stride_pb, stride_ph, stride_ps, stride_pr, stride_pd,
    stride_lb, stride_lh, stride_ls, stride_lr,
    stride_ob, stride_oh, stride_od,

    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_k_idx = tl.program_id(1)

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, HEAD_DIM)
    h_mask = offs_h < GROUP_SIZE

    lse_base = partial_lse_ptr + batch_idx * stride_lb + head_k_idx * stride_lh
    o_base = partial_o_ptr + batch_idx * stride_pb + head_k_idx * stride_ph

    max_lse = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    for split in range(NUM_SPLITS):
        lse = tl.load(lse_base + split * stride_ls + offs_h * stride_lr, mask=h_mask, other=-float("inf"))
        max_lse = tl.maximum(max_lse, lse)
    max_lse_safe = tl.where(max_lse == float("-inf"), 0.0, max_lse)

    acc = tl.zeros([BLOCK_H, HEAD_DIM], dtype=tl.float32)
    denom = tl.zeros([BLOCK_H], dtype=tl.float32)
    for split in range(NUM_SPLITS):
        lse = tl.load(lse_base + split * stride_ls + offs_h * stride_lr, mask=h_mask, other=-float("inf"))
        weight = tl.exp(lse - max_lse_safe)
        weight = tl.where(lse == float("-inf"), 0.0, weight)
        part = tl.load(
            o_base + split * stride_ps + offs_h[:, None] * stride_pr + offs_d[None, :] * stride_pd,
            mask=h_mask[:, None], other=0.0,
        )
        acc += part * weight[:, None]
        denom += weight

    out = acc / tl.where(denom > 0.0, denom, 1.0)[:, None]

    head_q = head_k_idx * GROUP_SIZE + offs_h
    o_ptrs = O_ptr + batch_idx * stride_ob + head_q[:, None] * stride_oh + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out.to(O_ptr.dtype.element_ty), mask=h_mask[:, None])


class DecodeWorkspace:
    def __init__(
        self,
        max_batch_size: int,
        num_kv_heads: int,
        block_h: int,
        head_dim: int,
        max_splits: int,
        device: torch.device | str = "cuda",
    ) -> None:
        self.max_splits = max_splits
        self.partial_out = torch.empty(
            (max_batch_size, num_kv_heads, max_splits, block_h, head_dim),
            dtype=torch.float32,
            device=device,
        )
        self.partial_lse = torch.empty(
            (max_batch_size, num_kv_heads, max_splits, block_h),
            dtype=torch.float32,
            device=device,
        )


def choose_num_splits(
    batch_size: int,
    num_kv_heads: int,
    max_seq_len: int,
    max_splits: int,
    device: torch.device | str = "cuda",
) -> int:
    if max_splits <= 1 or max_seq_len < _MIN_TOKENS_FOR_SPLIT:
        return 1
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    base = batch_size * num_kv_heads
    if base >= num_sms:
        return 1
    wanted = min(max_splits, (num_sms + base - 1) // base)
    wanted = min(wanted, max(1, max_seq_len // 128))
    return max(1, wanted)


def paged_decode_attention(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
    sm_scale: Optional[float] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    workspace: Optional[DecodeWorkspace] = None,
    num_splits: int = 1,
    block_n: int = 64,
) -> torch.Tensor:
    batch_size, num_q_heads, head_dim = query.shape
    num_kv_heads = k_cache.shape[2]
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads ({num_q_heads}) must be a multiple of num_kv_heads ({num_kv_heads})"
        )

    group_size = num_q_heads // num_kv_heads
    block_h = max(16, triton.next_power_of_2(group_size))

    if out is None:
        out = torch.empty_like(query)
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    quantized = k_scale is not None
    scale_slot_stride = k_scale.stride(0) if quantized else 0
    scale_head_stride = k_scale.stride(1) if quantized else 0

    use_split = num_splits > 1
    if use_split:
        if workspace is None:
            raise ValueError("split-K decode requires a preallocated DecodeWorkspace")
        partial_o = workspace.partial_out[:batch_size, :, :num_splits]
        partial_lse = workspace.partial_lse[:batch_size, :, :num_splits]
    else:
        partial_o = out
        partial_lse = seq_lens

    p_strides = partial_o.stride() if use_split else (0, 0, 0, 0, 0)
    l_strides = partial_lse.stride() if use_split else (0, 0, 0, 0)

    _paged_decode_attn_kernel[(batch_size, num_kv_heads, num_splits)](
        query, k_cache, v_cache,
        k_scale if quantized else query,
        v_scale if quantized else query,
        block_tables, seq_lens,
        out, partial_o, partial_lse,

        query.stride(0), query.stride(1), query.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        scale_slot_stride, scale_head_stride,
        block_tables.stride(0), block_tables.stride(1),
        p_strides[0], p_strides[1], p_strides[2], p_strides[3], p_strides[4],
        l_strides[0], l_strides[1], l_strides[2], l_strides[3],

        sm_scale,

        HEAD_DIM=head_dim,
        BLOCK_H=block_h,
        GROUP_SIZE=group_size,
        PAGE_SIZE=page_size,
        BLOCK_N=block_n,
        KV_QUANTIZED=quantized,
        NUM_SPLITS=num_splits,
        WRITE_PARTIAL=use_split,
        num_warps=4,
        num_stages=2,
    )

    if use_split:
        _decode_split_reduce_kernel[(batch_size, num_kv_heads)](
            partial_o, partial_lse, out,
            partial_o.stride(0), partial_o.stride(1), partial_o.stride(2),
            partial_o.stride(3), partial_o.stride(4),
            partial_lse.stride(0), partial_lse.stride(1), partial_lse.stride(2),
            partial_lse.stride(3),
            out.stride(0), out.stride(1), out.stride(2),
            HEAD_DIM=head_dim,
            BLOCK_H=block_h,
            GROUP_SIZE=group_size,
            NUM_SPLITS=num_splits,
            num_warps=4,
        )

    return out
