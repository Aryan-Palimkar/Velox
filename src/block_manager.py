from __future__ import annotations

from typing import List, Optional

from .block_allocator import BlockAllocator, OutOfBlocksError
from .paged_kv_cache import PagedKVCache
from .radix_cache import RadixCache


class BlockSpaceManager:
    def __init__(
        self,
        kv_cache: PagedKVCache,
        enable_prefix_caching: bool = True,
        max_model_len: Optional[int] = None,
    ) -> None:
        self.kv_cache = kv_cache
        self.block_size = kv_cache.block_size
        self.allocator: BlockAllocator = kv_cache.allocator
        self.prefix_cache = RadixCache(self.allocator, self.block_size, enabled=enable_prefix_caching)

        capacity = kv_cache.num_slots
        self.max_model_len = min(max_model_len, capacity) if max_model_len else capacity
        self.max_blocks_per_seq = (self.max_model_len + self.block_size - 1) // self.block_size

    @property
    def num_free_blocks(self) -> int:
        return self.allocator.num_free_blocks

    @property
    def num_total_blocks(self) -> int:
        return self.allocator.num_blocks - 1

    def num_available_blocks(self) -> int:
        return self.allocator.num_free_blocks + self.prefix_cache.num_evictable_blocks()

    def usage(self) -> float:
        total = self.num_total_blocks
        return 0.0 if total == 0 else self.allocator.num_used_blocks / total

    def _acquire(self, count: int) -> List[int]:
        if count <= 0:
            return []
        shortfall = count - self.allocator.num_free_blocks
        if shortfall > 0:
            self.prefix_cache.evict(shortfall)
        return self.allocator.allocate_many(count)

    def can_allocate(self, num_blocks: int) -> bool:
        return num_blocks <= self.num_available_blocks()

    def match_prefix(self, request) -> int:
        if not self.prefix_cache.enabled or request.block_table:
            return 0

        tokens = request.all_token_ids
        # Leave one token uncomputed or there is nothing to sample from.
        max_blocks = max(0, (len(tokens) - 1) // self.block_size)
        if max_blocks == 0:
            return 0

        matched = self.prefix_cache.match_prefix(tokens, max_blocks=max_blocks)
        if not matched:
            return 0

        self.allocator.incref_many(matched)
        request.block_table.extend(matched)
        num_tokens = len(matched) * self.block_size
        request.num_computed_tokens = num_tokens
        request.num_cached_tokens = num_tokens
        request.num_published_blocks = len(matched)
        return num_tokens

    def num_blocks_to_append(self, request, num_new_tokens: int) -> int:
        required = request.num_computed_tokens + num_new_tokens
        needed_blocks = (required + self.block_size - 1) // self.block_size
        return max(0, needed_blocks - len(request.block_table))

    def can_append(self, request, num_new_tokens: int) -> bool:
        required = request.num_computed_tokens + num_new_tokens
        if required > self.max_model_len:
            return False
        return self.can_allocate(self.num_blocks_to_append(request, num_new_tokens))

    def append_slots(self, request, num_new_tokens: int) -> bool:
        required = request.num_computed_tokens + num_new_tokens
        if required > self.max_model_len:
            return False

        extra = self.num_blocks_to_append(request, num_new_tokens)
        if extra == 0:
            return True
        if len(request.block_table) + extra > self.max_blocks_per_seq:
            return False
        try:
            request.block_table.extend(self._acquire(extra))
        except OutOfBlocksError:
            return False
        return True

    def commit(self, request) -> None:
        if not self.prefix_cache.enabled:
            return

        num_full_blocks = request.num_computed_tokens // self.block_size
        num_full_blocks = min(num_full_blocks, len(request.block_table))
        if num_full_blocks <= request.num_published_blocks:
            return

        tokens = request.all_token_ids[: num_full_blocks * self.block_size]
        self.prefix_cache.insert(tokens, request.block_table[:num_full_blocks])
        request.num_published_blocks = num_full_blocks

    def free(self, request) -> None:
        if not request.block_table:
            return
        self.allocator.free_many(request.block_table)
        request.block_table = []
        request.num_published_blocks = 0

    def reset_for_recompute(self, request) -> None:
        self.free(request)
        request.num_computed_tokens = 0
        request.num_cached_tokens = 0

    def slot_for(self, request, token_index: int) -> int:
        block = request.block_table[token_index // self.block_size]
        return block * self.block_size + (token_index % self.block_size)

    def stats(self) -> dict:
        return {
            "total_blocks": self.num_total_blocks,
            "free_blocks": self.allocator.num_free_blocks,
            "used_blocks": self.allocator.num_used_blocks,
            "evictable_blocks": self.prefix_cache.num_evictable_blocks(),
            "usage": self.usage(),
            "prefix_hit_rate": self.prefix_cache.hit_rate,
        }
