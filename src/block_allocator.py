from __future__ import annotations

from typing import Iterable, List, Sequence

# Reserved. Padded cuda graph rows point here so they always hit a valid address.
NULL_BLOCK = 0


class OutOfBlocksError(RuntimeError):
    pass


class BlockAllocator:
    def __init__(self, num_blocks: int) -> None:
        if num_blocks < 2:
            raise ValueError("num_blocks must be at least 2 (block 0 is reserved)")
        self.num_blocks = num_blocks
        self._ref_counts: List[int] = [0] * num_blocks
        self._ref_counts[NULL_BLOCK] = 1
        self._free: List[int] = list(range(num_blocks - 1, NULL_BLOCK, -1))

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - 1 - len(self._free)

    def can_allocate(self, count: int) -> bool:
        return count <= len(self._free)

    def allocate(self) -> int:
        if not self._free:
            raise OutOfBlocksError("KV cache block pool exhausted")
        block = self._free.pop()
        self._ref_counts[block] = 1
        return block

    def allocate_many(self, count: int) -> List[int]:
        if count < 0:
            raise ValueError("count must be non-negative")
        if count > len(self._free):
            raise OutOfBlocksError(
                f"requested {count} blocks but only {len(self._free)} are free"
            )
        return [self.allocate() for _ in range(count)]

    def incref(self, block: int) -> int:
        self._validate(block)
        if self._ref_counts[block] <= 0:
            raise RuntimeError(f"cannot incref block {block}: it is free")
        self._ref_counts[block] += 1
        return self._ref_counts[block]

    def incref_many(self, blocks: Iterable[int]) -> None:
        for block in blocks:
            self.incref(block)

    def free(self, block: int) -> int:
        self._validate(block)
        if block == NULL_BLOCK:
            return self._ref_counts[block]
        count = self._ref_counts[block]
        if count <= 0:
            raise RuntimeError(f"double free of block {block}")
        count -= 1
        self._ref_counts[block] = count
        if count == 0:
            self._free.append(block)
        return count

    def free_many(self, blocks: Iterable[int]) -> None:
        for block in blocks:
            self.free(block)

    def ref_count(self, block: int) -> int:
        self._validate(block)
        return self._ref_counts[block]

    def _validate(self, block: int) -> None:
        if not 0 <= block < self.num_blocks:
            raise IndexError(f"block {block} out of range [0, {self.num_blocks})")

    def assert_consistent(self) -> None:
        free_set = set(self._free)
        if len(free_set) != len(self._free):
            raise AssertionError("free list contains duplicates")
        for block in range(self.num_blocks):
            count = self._ref_counts[block]
            if count < 0:
                raise AssertionError(f"block {block} has negative refcount {count}")
            if (count == 0) != (block in free_set):
                raise AssertionError(
                    f"block {block} refcount={count} but free={block in free_set}"
                )

    def __repr__(self) -> str:
        return (
            f"BlockAllocator(num_blocks={self.num_blocks}, "
            f"free={self.num_free_blocks}, used={self.num_used_blocks})"
        )
