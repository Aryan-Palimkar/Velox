from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from .rope_inplace import apply_rope_inplace


def _rope_scaling_factor(rope_scaling: Optional[Dict[str, Any]]) -> float:
    if not rope_scaling:
        return 1.0
    rope_type = rope_scaling.get("type") or rope_scaling.get("rope_type")
    factor = float(rope_scaling.get("factor", 1.0))
    if rope_type == "linear" and factor > 0:
        return factor
    return 1.0


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        base_theta: float = 10000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position: int = 32768,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")

        self.head_dim = head_dim
        self.base_theta = base_theta
        self.rope_scaling = rope_scaling
        self.max_position = max_position
        self.scaling_factor = _rope_scaling_factor(rope_scaling)

        cos, sin = self._build_table(max_position, device, dtype)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def _build_table(
        self, max_position: int, device: torch.device | str | None, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            self.base_theta
            ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim)
        )
        positions = torch.arange(max_position, device=device, dtype=torch.float32)
        if self.scaling_factor != 1.0:
            positions = positions / self.scaling_factor
        freqs = torch.outer(positions, inv_freq)
        return freqs.cos().to(dtype), freqs.sin().to(dtype)

    # Keep the tables in fp32; model.to(bfloat16) would otherwise take them too.
    def _apply(self, *args, **kwargs):
        module = super()._apply(*args, **kwargs)
        module.cos_cached = module.cos_cached.float()
        module.sin_cached = module.sin_cached.float()
        return module

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        apply_rope_inplace(query, key, positions, self.cos_cached, self.sin_cached)
        return query, key

    apply_rotary_flat = forward

    def reference(
        self, query: torch.Tensor, key: torch.Tensor, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached.index_select(0, positions.to(torch.long)).unsqueeze(1)
        sin = self.sin_cached.index_select(0, positions.to(torch.long)).unsqueeze(1)

        def rotate(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x.float().chunk(2, dim=-1)
            c = cos.float()
            s = sin.float()
            return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).to(x.dtype)

        return rotate(query), rotate(key)

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, base_theta={self.base_theta}, "
            f"max_position={self.max_position}, scaling_factor={self.scaling_factor}"
        )
