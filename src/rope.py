from typing import Optional
import torch
from .rope_inplace import apply_rope_decode_inplace


class RotaryEmbedding:
    def __init__(self, head_dim: int, base_theta: float, rope_scaling: Optional[dict] = None) -> None:
        self.head_dim = head_dim
        self.base_theta = base_theta
        self.rope_scaling = rope_scaling
        inv_freq = 1.0 / (base_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self._inv_freq = inv_freq

    def _scaled_positions(self, positions: torch.Tensor) -> torch.Tensor:
        if not self.rope_scaling:
            return positions
        rope_type = self.rope_scaling.get("type")
        factor = float(self.rope_scaling.get("factor", 1.0))
        if rope_type == "linear" and factor > 0:
            return positions / factor
        return positions

    def get_cos_sin(self, positions: torch.Tensor, device: torch.device, dtype: torch.dtype):
        positions = self._scaled_positions(positions).to(device=device)
        inv_freq = self._inv_freq.to(device=device)
        freqs = torch.outer(positions.float(), inv_freq)
        cos = torch.cos(freqs).to(dtype=dtype)
        sin = torch.sin(freqs).to(dtype=dtype)
        return cos, sin

    def apply_rotary(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        device = q.device
        dtype = q.dtype
        cos, sin = self.get_cos_sin(positions, device, dtype)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        def rotate(x: torch.Tensor) -> torch.Tensor:
            x_even = x[..., ::2]
            x_odd = x[..., 1::2]
            x_rot_even = x_even * cos - x_odd * sin
            x_rot_odd = x_even * sin + x_odd * cos
            x_rot = torch.stack((x_rot_even, x_rot_odd), dim=-1)
            return x_rot.flatten(-2)

        return rotate(q), rotate(k)

    def apply_rotary_batch(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        device = q.device
        dtype = q.dtype

        if positions.dim() == 4:
            positions = positions.squeeze(1).squeeze(-1)

        cos_list = []
        sin_list = []
        for i in range(positions.shape[0]):
            cos_i, sin_i = self.get_cos_sin(positions[i], device, dtype)
            cos_list.append(cos_i)
            sin_list.append(sin_i)

        cos = torch.stack(cos_list, dim=0)
        sin = torch.stack(sin_list, dim=0)

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        def rotate(x: torch.Tensor) -> torch.Tensor:
            x_even = x[..., ::2]
            x_odd = x[..., 1::2]
            x_rot_even = x_even * cos - x_odd * sin
            x_rot_odd = x_even * sin + x_odd * cos
            x_rot = torch.stack((x_rot_even, x_rot_odd), dim=-1)
            return x_rot.flatten(-2)

        return rotate(q), rotate(k)

    def apply_rotary_flat(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        device = q.device
        dtype = q.dtype

        max_pos = positions.max().item() if positions.numel() > 0 else 0
        all_positions = torch.arange(max_pos + 1, device=device)
        cos, sin = self.get_cos_sin(all_positions, device, dtype)

        total_tokens, num_heads, head_dim = q.shape
        _, num_kv_heads, _ = k.shape

        q_reshaped = q
        k_reshaped = k

        apply_rope_decode_inplace(
            q_reshaped,
            k_reshaped,
            positions,
            cos,
            sin
        )

        return q_reshaped, k_reshaped
