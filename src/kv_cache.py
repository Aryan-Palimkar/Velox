from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from .utils import bytes_to_human


class KVCache:
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        verbose: bool = True,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.k_cache: List[Optional[torch.Tensor]] = [None for _ in range(num_layers)]
        self.v_cache: List[Optional[torch.Tensor]] = [None for _ in range(num_layers)]
        self.seq_lens: List[int] = [0 for _ in range(num_layers)]

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        k = k.to(device=self.device, dtype=self.dtype)
        v = v.to(device=self.device, dtype=self.dtype)
        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = k
            self.v_cache[layer_idx] = v
        else:
            self.k_cache[layer_idx] = torch.cat([self.k_cache[layer_idx], k], dim=2)
            self.v_cache[layer_idx] = torch.cat([self.v_cache[layer_idx], v], dim=2)
        self.seq_lens[layer_idx] = self.k_cache[layer_idx].shape[2]
        if self.verbose:
            self._print_layer_stats(layer_idx)

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.k_cache[layer_idx] is None or self.v_cache[layer_idx] is None:
            raise RuntimeError(f"KV cache missing for layer {layer_idx}")
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def _print_layer_stats(self, layer_idx: int) -> None:
        k = self.k_cache[layer_idx]
        v = self.v_cache[layer_idx]
        if k is None or v is None:
            return
        bytes_used = k.numel() * k.element_size() + v.numel() * v.element_size()
        print(
            f"[kv-cache] layer={layer_idx} k={tuple(k.shape)} v={tuple(v.shape)} "
            f"seq_len={self.seq_lens[layer_idx]} bytes={bytes_to_human(bytes_used)}"
        )

    def total_bytes(self) -> int:
        total = 0
        for k, v in zip(self.k_cache, self.v_cache):
            if k is not None:
                total += k.numel() * k.element_size()
            if v is not None:
                total += v.numel() * v.element_size()
        return total
