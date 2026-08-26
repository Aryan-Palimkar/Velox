from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_inplace_kernel(
    q_ptr, k_ptr,
    cos_ptr, sin_ptr,
    positions_ptr,

    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_ct, stride_cd,

    H_Q: tl.constexpr,
    H_K: tl.constexpr,
    HALF_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    pos = tl.load(positions_ptr + token_idx).to(tl.int32)

    offs = tl.arange(0, HALF_D)
    cos = tl.load(cos_ptr + pos * stride_ct + offs * stride_cd).to(tl.float32)
    sin = tl.load(sin_ptr + pos * stride_ct + offs * stride_cd).to(tl.float32)

    if head_idx < H_Q:
        base = q_ptr + token_idx * stride_qt + head_idx * stride_qh
        lo_ptr = base + offs * stride_qd
        hi_ptr = base + (offs + HALF_D) * stride_qd
        lo = tl.load(lo_ptr).to(tl.float32)
        hi = tl.load(hi_ptr).to(tl.float32)
        tl.store(lo_ptr, (lo * cos - hi * sin).to(q_ptr.dtype.element_ty))
        tl.store(hi_ptr, (lo * sin + hi * cos).to(q_ptr.dtype.element_ty))

    if head_idx < H_K:
        base = k_ptr + token_idx * stride_kt + head_idx * stride_kh
        lo_ptr = base + offs * stride_kd
        hi_ptr = base + (offs + HALF_D) * stride_kd
        lo = tl.load(lo_ptr).to(tl.float32)
        hi = tl.load(hi_ptr).to(tl.float32)
        tl.store(lo_ptr, (lo * cos - hi * sin).to(k_ptr.dtype.element_ty))
        tl.store(hi_ptr, (lo * sin + hi * cos).to(k_ptr.dtype.element_ty))


def apply_rope_inplace(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> None:
    num_tokens, num_q_heads, head_dim = query.shape
    num_kv_heads = key.shape[1]
    if num_tokens == 0:
        return
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    if cos.shape[1] != head_dim // 2:
        raise ValueError(
            f"cos table has width {cos.shape[1]} but head_dim // 2 is {head_dim // 2}"
        )

    _rope_inplace_kernel[(num_tokens, max(num_q_heads, num_kv_heads))](
        query, key, cos, sin, positions,
        query.stride(0), query.stride(1), query.stride(2),
        key.stride(0), key.stride(1), key.stride(2),
        cos.stride(0), cos.stride(1),
        H_Q=num_q_heads,
        H_K=num_kv_heads,
        HALF_D=head_dim // 2,
        num_warps=2,
    )
