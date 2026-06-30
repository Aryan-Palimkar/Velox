import torch
import triton
import triton.language as tl

@triton.jit
def _decode_attn_kernel(
    Q_ptr,
    K_cache_ptr,
    V_cache_ptr,
    cache_slots_ptr,
    seq_lens_ptr,
    O_ptr,

    stride_qb, stride_qh, stride_qd,
    stride_ob, stride_oh, stride_od,

    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,

    scale,
    B, H_Q, H_K, D,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_q_idx = tl.program_id(1)

    group_size = H_Q // H_K
    head_k_idx = head_q_idx // group_size

    slot = tl.load(cache_slots_ptr + batch_idx)
    cur_seq_len = tl.load(seq_lens_ptr + batch_idx)

    if cur_seq_len <= 0:
        return

    q_offset = batch_idx * stride_qb + head_q_idx * stride_qh
    d_offsets = tl.arange(0, BLOCK_D)
    q_mask = d_offsets < D

    q = tl.load(Q_ptr + q_offset + d_offsets * stride_qd, mask=q_mask, other=0.0).to(tl.float32)
    q = q * scale

    m_i = tl.zeros([], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([], dtype=tl.float32)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    k_base_ptr = K_cache_ptr + slot * stride_kb + head_k_idx * stride_kh
    v_base_ptr = V_cache_ptr + slot * stride_vb + head_k_idx * stride_vh

    for start_n in range(0, cur_seq_len, BLOCK_N):
        n_offsets = start_n + tl.arange(0, BLOCK_N)
        kv_mask = (n_offsets < cur_seq_len)[:, None] & (d_offsets[None, :] < D)

        k_ptr = k_base_ptr + n_offsets[:, None] * stride_ks + d_offsets[None, :] * stride_kd

        k = tl.load(k_ptr, mask=kv_mask, other=0.0).to(tl.float32)
        s = tl.sum(q[None, :] * k, axis=1)
        s_mask = n_offsets < cur_seq_len
        s = tl.where(s_mask, s, float("-inf"))

        m_ij = tl.max(s, axis=0)
        m_next = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_next)
        p = tl.exp(s - m_next)
        l_next = l_i * alpha + tl.sum(p, axis=0)

        v_ptr = v_base_ptr + n_offsets[:, None] * stride_vs + d_offsets[None, :] * stride_vd
        v = tl.load(v_ptr, mask=kv_mask, other=0.0).to(tl.float32)

        acc = acc * alpha
        acc += tl.sum(p[:, None] * v, axis=0)

        m_i = m_next
        l_i = l_next

    acc = acc / tl.maximum(l_i, 1e-6)

    o_offset = batch_idx * stride_ob + head_q_idx * stride_oh
    o_ptr = O_ptr + o_offset + d_offsets * stride_od

    tl.store(o_ptr, acc.to(Q_ptr.dtype.element_ty), mask=q_mask)


def decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_slots: torch.Tensor,
    seq_lens: torch.Tensor
):
    Active_B, H_Q, D = q.shape
    _, _, H_K, _ = k_cache.shape

    scale = 1.0 / (D ** 0.5)
    out = torch.zeros_like(q)

    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_N = 64 if BLOCK_D <= 128 else 32

    grid = (Active_B, H_Q)

    _decode_attn_kernel[grid](
        q, k_cache, v_cache, cache_slots, seq_lens, out,

        q.stride(0), q.stride(1), q.stride(2),
        out.stride(0), out.stride(1), out.stride(2),

        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),

        scale,
        Active_B, H_Q, H_K, D,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2
    )
    return out
