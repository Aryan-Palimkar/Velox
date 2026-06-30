import triton
import triton.language as tl
import torch

@triton.jit
def _attn_prefill_optimized(
    Q_ptr, K_cache_ptr, V_cache_ptr, O_ptr,
    cu_seq_lens_q_ptr, cu_seq_lens_kv_ptr,
    cache_slots_ptr,

    stride_qt, stride_qh, stride_qd,
    stride_ot, stride_oh, stride_od,

    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,

    sm_scale,
    q_len_key, kv_len_key,

    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
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

    kv_start = tl.load(cu_seq_lens_kv_ptr + batch_idx)
    kv_end = tl.load(cu_seq_lens_kv_ptr + batch_idx + 1)
    kv_len = kv_end - kv_start
    ctx_len = kv_len - q_len

    slot = tl.load(cache_slots_ptr + batch_idx)

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_offset = q_start * stride_qt + head_q_idx * stride_qh
    q_ptrs = Q_ptr + q_offset + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    q_mask = offs_m[:, None] < q_len
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)


    k_base = K_cache_ptr + slot * stride_kb + head_k_idx * stride_kh
    v_base = V_cache_ptr + slot * stride_vb + head_k_idx * stride_vh

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if IS_CAUSAL:
        max_valid_n = tl.minimum(kv_len, ctx_len + start_m + BLOCK_M)
    else:
        max_valid_n = kv_len

    for start_n in range(0, max_valid_n, BLOCK_N):
        n_offsets = start_n + offs_n
        n_mask = n_offsets < kv_len

        k_ptrs = k_base + n_offsets[None, :] * stride_ks + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0)

        v_ptrs = v_base + n_offsets[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        qk = tl.dot(q, k)
        qk *= sm_scale

        is_fully_past_block = (start_n + BLOCK_N) <= ctx_len

        if IS_CAUSAL and not is_fully_past_block:
            pos_m = ctx_len + offs_m
            pos_n = n_offsets
            causal_mask = pos_m[:, None] >= pos_n[None, :]
            qk = tl.where(causal_mask & n_mask[None, :], qk, float("-inf"))
        else:
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp(qk - m_ij[:, None])

        alpha = tl.math.exp(m_i - m_ij)
        l_ij = l_i * alpha + tl.sum(p, 1)

        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_ij
        l_i = l_ij

    acc = acc / l_i[:, None]

    o_offset = q_start * stride_ot + head_q_idx * stride_oh
    o_ptrs = O_ptr + o_offset + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(q.dtype), mask=q_mask)
