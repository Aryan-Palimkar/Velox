import torch
from collections import deque
from typing import List, Tuple

class BatchedKVCache:
    def __init__(
        self,
        num_layers: int,
        max_batch_size: int = 16,
        max_seq_len: int = 2048,
        num_kv_head: int = 2,
        head_dim: int = 128,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_kv_head = num_kv_head
        self.head_dim = head_dim

        self.free_slots = deque(range(max_batch_size))
        self.occupied_slots = set()
        self.seq_lens = [0] * max_batch_size

        self.k_cache = []
        self.v_cache = []

        for _ in range(num_layers):
            k_cache = torch.empty(
                max_batch_size, max_seq_len, num_kv_head, head_dim, dtype=dtype, device="cuda"
            )
            v_cache = torch.empty(
                max_batch_size, max_seq_len, num_kv_head, head_dim, dtype=dtype, device="cuda"
            )
            self.k_cache.append(k_cache)
            self.v_cache.append(v_cache)

        print(f"[BatchedKVCache] Allocated {max_batch_size}x{max_seq_len}x{num_kv_head}x{head_dim} KV cache per layer")

    def allocate_slot(self) -> int:
        if not self.free_slots:
            raise RuntimeError("No free slots in KV cache")
        slot = self.free_slots.popleft()
        self.occupied_slots.add(slot)
        self.seq_lens[slot] = 0
        return slot

    def free_slot(self, slot: int) -> None:
        if slot not in self.occupied_slots:
            return
        self.occupied_slots.remove(slot)
        self.free_slots.append(slot)
        self.seq_lens[slot] = 0

    def get(self, slots: List[int], layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.k_cache[layer][slots], self.v_cache[layer][slots]

    def write_prefill_flat(self, slots, layer, k, v, seq_lens):
        offset = 0
        for slot, length in zip(slots, seq_lens):
            if length > self.max_seq_len:
                raise ValueError(f"Prompt length {length} exceeds max_seq_len {self.max_seq_len}")
            self.k_cache[layer][slot, :length] = k[offset:offset+length]
            self.v_cache[layer][slot, :length] = v[offset:offset+length]
            offset += length
        if layer == 0:
            for slot, length in zip(slots, seq_lens):
                self.seq_lens[slot] = length

    def write_decode(self, slots_idx: torch.Tensor, pos_idx: torch.Tensor, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self.k_cache[layer][slots_idx, pos_idx] = k
        self.v_cache[layer][slots_idx, pos_idx] = v
