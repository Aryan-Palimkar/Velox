import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _rmsnorm_kernel(
    x_ptr, weight_ptr, out_ptr,
    stride_x_row, stride_out_row,
    N, eps,
    BLOCK_N: tl.constexpr
):
    row_idx = tl.program_id(0)

    x_ptrs = x_ptr + row_idx * stride_x_row + tl.arange(0, BLOCK_N)
    w_ptrs = weight_ptr + tl.arange(0, BLOCK_N)

    mask = tl.arange(0, BLOCK_N) < N

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rsqrt = tl.math.rsqrt(var + eps)

    out = x * rsqrt * w
    out_ptrs = out_ptr + row_idx * stride_out_row + tl.arange(0, BLOCK_N)
    tl.store(out_ptrs, out, mask=mask)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.hidden_size = hidden_size
        self.BLOCK_N = triton.next_power_of_2(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(-1, self.hidden_size)
        out_flat = torch.empty_like(x_flat)

        M = x_flat.shape[0]

        _rmsnorm_kernel[(M,)](
            x_flat, self.weight, out_flat,
            x_flat.stride(0), out_flat.stride(0),
            self.hidden_size, self.eps,
            BLOCK_N=self.BLOCK_N
        )

        return out_flat.view_as(x)
