from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import nn

FP8_E4M3_MAX = 448.0
FP8_E5M2_MAX = 57344.0
INT8_MAX = 127.0

_KV_DTYPES = {
    "auto": None,
    "fp8": torch.float8_e4m3fn,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
    "int8": torch.int8,
}

_WEIGHT_DTYPES = {
    "none": None,
    "int8": torch.int8,
    "fp8": torch.float8_e4m3fn,
    "fp8_e4m3": torch.float8_e4m3fn,
}


def quant_dtype_max(dtype: torch.dtype) -> float:
    if dtype == torch.float8_e4m3fn:
        return FP8_E4M3_MAX
    if dtype == torch.float8_e5m2:
        return FP8_E5M2_MAX
    if dtype == torch.int8:
        return INT8_MAX
    raise ValueError(f"unsupported quantized dtype {dtype}")


def fp8_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) >= (8, 9)


def resolve_kv_cache_dtype(kv_cache_dtype: str, model_dtype: torch.dtype) -> Tuple[torch.dtype, bool]:
    key = (kv_cache_dtype or "auto").lower()
    if key not in _KV_DTYPES:
        raise ValueError(
            f"unknown kv_cache_dtype {kv_cache_dtype!r}; expected one of {sorted(_KV_DTYPES)}"
        )
    target = _KV_DTYPES[key]
    if target is None:
        return model_dtype, False
    if target in (torch.float8_e4m3fn, torch.float8_e5m2) and not fp8_supported():
        raise RuntimeError(
            f"kv_cache_dtype={kv_cache_dtype!r} requires compute capability 8.9 or newer"
        )
    return target, True


def resolve_weight_dtype(quantization: Optional[str]) -> Optional[torch.dtype]:
    key = (quantization or "none").lower()
    if key not in _WEIGHT_DTYPES:
        raise ValueError(
            f"unknown quantization {quantization!r}; expected one of {sorted(_WEIGHT_DTYPES)}"
        )
    target = _WEIGHT_DTYPES[key]
    if target in (torch.float8_e4m3fn,) and not fp8_supported():
        raise RuntimeError(f"quantization={quantization!r} requires compute capability 8.9 or newer")
    return target


@triton.jit
def _quantize_store_kv_kernel(
    k_ptr, v_ptr,
    k_cache_ptr, v_cache_ptr,
    k_scale_ptr, v_scale_ptr,
    slot_mapping_ptr,

    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_ct, stride_ch, stride_cd,
    stride_st, stride_sh,

    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    QMAX: tl.constexpr,
    IS_INT8: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    slot = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)

    offs_d = tl.arange(0, BLOCK_D)
    mask = offs_d < HEAD_DIM

    k = tl.load(
        k_ptr + token_idx * stride_kt + head_idx * stride_kh + offs_d * stride_kd,
        mask=mask, other=0.0,
    ).to(tl.float32)
    v = tl.load(
        v_ptr + token_idx * stride_vt + head_idx * stride_vh + offs_d * stride_vd,
        mask=mask, other=0.0,
    ).to(tl.float32)

    k_amax = tl.max(tl.abs(k), axis=0)
    v_amax = tl.max(tl.abs(v), axis=0)
    k_scale = tl.where(k_amax > 0.0, k_amax / QMAX, 1.0)
    v_scale = tl.where(v_amax > 0.0, v_amax / QMAX, 1.0)

    k_q = k / k_scale
    v_q = v / v_scale

    if IS_INT8:
        k_q = tl.where(k_q >= 0, tl.floor(k_q + 0.5), tl.ceil(k_q - 0.5))
        v_q = tl.where(v_q >= 0, tl.floor(v_q + 0.5), tl.ceil(v_q - 0.5))
        k_q = tl.minimum(tl.maximum(k_q, -QMAX), QMAX)
        v_q = tl.minimum(tl.maximum(v_q, -QMAX), QMAX)

    k_out = k_cache_ptr + slot * stride_ct + head_idx * stride_ch + offs_d * stride_cd
    v_out = v_cache_ptr + slot * stride_ct + head_idx * stride_ch + offs_d * stride_cd
    tl.store(k_out, k_q.to(k_cache_ptr.dtype.element_ty), mask=mask)
    tl.store(v_out, v_q.to(v_cache_ptr.dtype.element_ty), mask=mask)

    tl.store(k_scale_ptr + slot * stride_st + head_idx * stride_sh, k_scale)
    tl.store(v_scale_ptr + slot * stride_st + head_idx * stride_sh, v_scale)


def quantize_and_store_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache_flat: torch.Tensor,
    v_cache_flat: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    num_tokens, num_heads, head_dim = key.shape
    if num_tokens == 0:
        return

    store_dtype = k_cache_flat.dtype
    qmax = quant_dtype_max(store_dtype)

    _quantize_store_kv_kernel[(num_tokens, num_heads)](
        key, value,
        k_cache_flat, v_cache_flat,
        k_scale, v_scale,
        slot_mapping,

        key.stride(0), key.stride(1), key.stride(2),
        value.stride(0), value.stride(1), value.stride(2),
        k_cache_flat.stride(0), k_cache_flat.stride(1), k_cache_flat.stride(2),
        k_scale.stride(0), k_scale.stride(1),

        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        QMAX=qmax,
        IS_INT8=store_dtype == torch.int8,
        num_warps=4,
    )


@triton.jit
def _wq_gemm_kernel(
    a_ptr, b_ptr, scale_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        b_q = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
            other=0.0,
        )
        # Exact for int8 and fp8-e4m3; the per-column scale folds in after the loop.
        acc += tl.dot(a, b_q.to(a.dtype))

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc = acc * scale[None, :]

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc.to(c_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _gemm_config(m: int) -> dict:
    if m <= 16:
        return dict(BLOCK_M=16, BLOCK_N=128, BLOCK_K=128, GROUP_M=1, num_warps=8, num_stages=4)
    if m <= 128:
        return dict(BLOCK_M=32, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_warps=4, num_stages=4)
    return dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_warps=4, num_stages=4)


def quantized_matmul(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    m, k = x.shape
    k_w, n = qweight.shape
    if k != k_w:
        raise ValueError(f"shape mismatch: x has K={k} but qweight has K={k_w}")

    out = torch.empty((m, n), dtype=out_dtype, device=x.device)
    if m == 0:
        return out

    config = _gemm_config(m)
    num_warps = config.pop("num_warps")
    num_stages = config.pop("num_stages")

    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    _wq_gemm_kernel[grid](
        x, qweight, scale, bias if bias is not None else x, out,
        m, n, k,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=bias is not None,
        num_warps=num_warps,
        num_stages=num_stages,
        **config,
    )
    return out


def quantize_weight(weight: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    qmax = quant_dtype_max(dtype)
    weight_f32 = weight.detach().to(torch.float32)
    amax = weight_f32.abs().amax(dim=1)
    scale = torch.where(amax > 0, amax / qmax, torch.ones_like(amax))

    normalized = weight_f32 / scale.unsqueeze(1)
    if dtype == torch.int8:
        normalized = normalized.round().clamp_(-qmax, qmax)
    else:
        normalized = normalized.clamp_(-qmax, qmax)

    qweight = normalized.to(dtype).t().contiguous()
    return qweight, scale.to(torch.float32)


class QuantizedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        quant_dtype: torch.dtype,
        bias: bool = False,
        device: torch.device | str = "cuda",
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_dtype = quant_dtype
        self.compute_dtype = compute_dtype

        self.register_buffer(
            "qweight",
            torch.zeros((in_features, out_features), dtype=quant_dtype, device=device),
        )
        self.register_buffer(
            "scale", torch.ones(out_features, dtype=torch.float32, device=device)
        )
        if bias:
            self.register_buffer(
                "bias", torch.zeros(out_features, dtype=compute_dtype, device=device)
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, quant_dtype: torch.dtype) -> "QuantizedLinear":
        module = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            quant_dtype=quant_dtype,
            bias=linear.bias is not None,
            device=linear.weight.device,
            compute_dtype=linear.weight.dtype,
        )
        module.load_weight(linear.weight, linear.bias)
        return module

    @torch.no_grad()
    def load_weight(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
        qweight, scale = quantize_weight(weight, self.quant_dtype)
        self.qweight.copy_(qweight.to(self.qweight.device))
        self.scale.copy_(scale.to(self.scale.device))
        if bias is not None:
            if self.bias is None:
                raise ValueError("layer was built without a bias but a bias tensor was supplied")
            self.bias.copy_(bias.to(device=self.bias.device, dtype=self.bias.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        out = quantized_matmul(x_2d, self.qweight, self.scale, self.bias, x.dtype)
        return out.reshape(*original_shape[:-1], self.out_features)

    def dequantized_weight(self) -> torch.Tensor:
        return (self.qweight.to(torch.float32) * self.scale.unsqueeze(0)).t().contiguous()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"quant_dtype={self.quant_dtype}, bias={self.bias is not None}"
        )
