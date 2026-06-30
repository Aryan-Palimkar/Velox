import torch
import triton
import triton.language as tl

@triton.jit
def _rope_decode_kernel(
    q_ptr, k_ptr,
    cos_ptr, sin_ptr,
    positions_ptr,

    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kd,
    stride_cb, stride_cd,

    B, H_Q, H_K,
    D: tl.constexpr,
    HALF_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    is_q = head_idx < H_Q
    is_k = head_idx < H_K

    if not is_q and not is_k:
        return

    pos = tl.load(positions_ptr + batch_idx)

    d_offsets = tl.arange(0, HALF_D)

    first_half_offsets = d_offsets
    second_half_offsets = d_offsets + HALF_D

    cos_offset = pos * stride_cb + d_offsets * stride_cd
    sin_offset = pos * stride_cb + d_offsets * stride_cd

    cos = tl.load(cos_ptr + cos_offset)
    sin = tl.load(sin_ptr + sin_offset)

    if is_q:
        q_base = q_ptr + batch_idx * stride_qb + head_idx * stride_qh

        q1 = tl.load(q_base + first_half_offsets * stride_qd)
        q2 = tl.load(q_base + second_half_offsets * stride_qd)

        q_rot1 = q1 * cos - q2 * sin
        q_rot2 = q1 * sin + q2 * cos

        tl.store(q_base + first_half_offsets * stride_qd, q_rot1)
        tl.store(q_base + second_half_offsets * stride_qd, q_rot2)

    if is_k:
        k_base = k_ptr + batch_idx * stride_kb + head_idx * stride_kh

        k1 = tl.load(k_base + first_half_offsets * stride_kd)
        k2 = tl.load(k_base + second_half_offsets * stride_kd)

        k_rot1 = k1 * cos - k2 * sin
        k_rot2 = k1 * sin + k2 * cos

        tl.store(k_base + first_half_offsets * stride_kd, k_rot1)
        tl.store(k_base + second_half_offsets * stride_kd, k_rot2)


def apply_rope_decode_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
):
    B, H_Q, D = q.shape
    _, H_K, _ = k.shape

    grid = (B, max(H_Q, H_K))

    _rope_decode_kernel[grid](
        q, k, cos, sin, positions,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        cos.stride(0), cos.stride(1),
        B, H_Q, H_K,
        D=D,
        HALF_D=D // 2,
        num_warps=2
    )

    return q, k
