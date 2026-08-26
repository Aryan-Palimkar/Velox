from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from .block_allocator import BlockAllocator
from .utils import bytes_to_human

DEFAULT_BLOCK_SIZE = 16


class PagedKVCache:
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int = DEFAULT_BLOCK_SIZE,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cuda",
        kv_cache_dtype: str = "auto",
        verbose: bool = True,
    ) -> None:
        if block_size <= 0 or (block_size & (block_size - 1)) != 0:
            raise ValueError(f"block_size must be a power of two, got {block_size}")
        if num_blocks < 2:
            raise ValueError("num_blocks must be at least 2 (block 0 is reserved)")

        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = torch.device(device)
        self.model_dtype = dtype

        from .quantization import resolve_kv_cache_dtype

        self.kv_cache_dtype = kv_cache_dtype
        self.store_dtype, self.is_quantized = resolve_kv_cache_dtype(kv_cache_dtype, dtype)

        self.allocator = BlockAllocator(num_blocks)

        self.k_cache: List[torch.Tensor] = []
        self.v_cache: List[torch.Tensor] = []
        self.k_scale: List[Optional[torch.Tensor]] = []
        self.v_scale: List[Optional[torch.Tensor]] = []

        shape = (num_blocks, block_size, num_kv_heads, head_dim)
        scale_shape = (num_blocks * block_size, num_kv_heads)
        for _ in range(num_layers):
            self.k_cache.append(torch.zeros(shape, dtype=self.store_dtype, device=self.device))
            self.v_cache.append(torch.zeros(shape, dtype=self.store_dtype, device=self.device))
            if self.is_quantized:
                self.k_scale.append(torch.zeros(scale_shape, dtype=torch.float32, device=self.device))
                self.v_scale.append(torch.zeros(scale_shape, dtype=torch.float32, device=self.device))
            else:
                self.k_scale.append(None)
                self.v_scale.append(None)

        self._k_flat = [t.view(-1, num_kv_heads, head_dim) for t in self.k_cache]
        self._v_flat = [t.view(-1, num_kv_heads, head_dim) for t in self.v_cache]

        if verbose:
            print(
                f"[PagedKVCache] {num_blocks} blocks x {block_size} tokens "
                f"({num_blocks * block_size} tokens) x {num_kv_heads} kv-heads x {head_dim} "
                f"dim, dtype={self.store_dtype}, total={bytes_to_human(self.total_bytes())}"
            )

    @property
    def num_slots(self) -> int:
        return self.num_blocks * self.block_size

    @property
    def num_free_blocks(self) -> int:
        return self.allocator.num_free_blocks

    def bytes_per_block(self) -> int:
        per_layer = (
            2 * self.block_size * self.num_kv_heads * self.head_dim * self.k_cache[0].element_size()
        )
        if self.is_quantized:
            per_layer += 2 * self.block_size * self.num_kv_heads * 4
        return per_layer * self.num_layers

    def total_bytes(self) -> int:
        total = 0
        for k, v in zip(self.k_cache, self.v_cache):
            total += k.numel() * k.element_size() + v.numel() * v.element_size()
        for k_s, v_s in zip(self.k_scale, self.v_scale):
            if k_s is not None and v_s is not None:
                total += k_s.numel() * k_s.element_size() + v_s.numel() * v_s.element_size()
        return total

    def blocks_for_tokens(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def write(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.is_quantized:
            from .quantization import quantize_and_store_kv

            quantize_and_store_kv(
                key,
                value,
                self._k_flat[layer_idx],
                self._v_flat[layer_idx],
                self.k_scale[layer_idx],
                self.v_scale[layer_idx],
                slot_mapping,
            )
            return

        self._k_flat[layer_idx].index_copy_(0, slot_mapping, key.to(self.store_dtype))
        self._v_flat[layer_idx].index_copy_(0, slot_mapping, value.to(self.store_dtype))

    def gather_sequence(
        self, layer_idx: int, block_table: List[int], seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        needed = self.blocks_for_tokens(seq_len)
        if needed > len(block_table):
            raise ValueError(
                f"block table holds {len(block_table)} blocks but {needed} "
                f"are needed for seq_len={seq_len}"
            )
        slots = torch.tensor(
            [
                block_table[i // self.block_size] * self.block_size + (i % self.block_size)
                for i in range(seq_len)
            ],
            dtype=torch.long,
            device=self.device,
        )
        keys = self._k_flat[layer_idx].index_select(0, slots)
        values = self._v_flat[layer_idx].index_select(0, slots)
        if self.is_quantized:
            k_scale = self.k_scale[layer_idx].index_select(0, slots).unsqueeze(-1)
            v_scale = self.v_scale[layer_idx].index_select(0, slots).unsqueeze(-1)
            keys = keys.to(torch.float32) * k_scale
            values = values.to(torch.float32) * v_scale
            return keys.to(self.model_dtype), values.to(self.model_dtype)
        return keys, values

    def __repr__(self) -> str:
        return (
            f"PagedKVCache(layers={self.num_layers}, blocks={self.num_blocks}, "
            f"block_size={self.block_size}, free={self.num_free_blocks})"
        )


def compute_num_blocks(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    dtype: torch.dtype,
    gpu_memory_utilization: float = 0.90,
    device: torch.device | str = "cuda",
    kv_cache_dtype: str = "auto",
    reserve_bytes: int = 512 * 1024 * 1024,
) -> int:
    from .quantization import resolve_kv_cache_dtype

    store_dtype, is_quantized = resolve_kv_cache_dtype(kv_cache_dtype, dtype)
    element_size = torch.tensor([], dtype=store_dtype).element_size()

    bytes_per_block = 2 * block_size * num_kv_heads * head_dim * element_size * num_layers
    if is_quantized:
        bytes_per_block += 2 * block_size * num_kv_heads * 4 * num_layers

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("compute_num_blocks requires a CUDA device")

    free_bytes, _ = torch.cuda.mem_get_info(device)
    free_bytes += torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    budget = int(free_bytes * gpu_memory_utilization) - reserve_bytes
    if budget <= 0:
        raise RuntimeError(
            "not enough free GPU memory for a KV cache "
            f"(free={bytes_to_human(free_bytes)}, reserve={bytes_to_human(reserve_bytes)})"
        )
    num_blocks = budget // bytes_per_block
    if num_blocks < 2:
        raise RuntimeError("free GPU memory is too small to hold even one KV cache block")
    return int(num_blocks)
