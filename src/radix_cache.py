from __future__ import annotations

import heapq
from itertools import count
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from .block_allocator import BlockAllocator


def block_hash(parent_hash: Optional[int], token_ids: Sequence[int]) -> int:
    return hash((parent_hash, tuple(token_ids)))


class RadixNode:
    __slots__ = (
        "node_hash",
        "token_ids",
        "physical_block",
        "parent",
        "children",
        "last_access",
    )

    def __init__(
        self,
        node_hash: Optional[int],
        token_ids: Tuple[int, ...],
        physical_block: int,
        parent: Optional["RadixNode"],
    ) -> None:
        self.node_hash = node_hash
        self.token_ids = token_ids
        self.physical_block = physical_block
        self.parent = parent
        self.children: Dict[int, "RadixNode"] = {}
        self.last_access = 0

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def is_root(self) -> bool:
        return self.parent is None

    def __repr__(self) -> str:
        return (
            f"RadixNode(block={self.physical_block}, tokens={len(self.token_ids)}, "
            f"children={len(self.children)})"
        )


class RadixCache:
    def __init__(self, allocator: BlockAllocator, block_size: int, enabled: bool = True) -> None:
        self.allocator = allocator
        self.block_size = block_size
        self.enabled = enabled

        self.root = RadixNode(node_hash=None, token_ids=(), physical_block=-1, parent=None)
        self._nodes: Dict[int, RadixNode] = {}
        self._clock = count(1)
        self._evict_heap: List[Tuple[int, int, RadixNode]] = []
        self._node_ids = count(1)
        self._heap_ids: Dict[int, int] = {}

        self.num_hits = 0
        self.num_queries = 0
        self.num_cached_tokens = 0
        self.num_queried_tokens = 0

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def hit_rate(self) -> float:
        if self.num_queried_tokens == 0:
            return 0.0
        return self.num_cached_tokens / self.num_queried_tokens

    def num_evictable_blocks(self) -> int:
        return sum(
            1
            for node in self._nodes.values()
            if node.is_leaf and self.allocator.ref_count(node.physical_block) == 1
        )

    def _touch(self, node: RadixNode) -> None:
        node.last_access = next(self._clock)
        if node.is_leaf:
            self._push_leaf(node)

    def _push_leaf(self, node: RadixNode) -> None:
        node_id = self._heap_ids.get(id(node))
        if node_id is None:
            node_id = next(self._node_ids)
            self._heap_ids[id(node)] = node_id
        heapq.heappush(self._evict_heap, (node.last_access, node_id, node))

    def match_prefix(
        self, token_ids: Sequence[int], max_blocks: Optional[int] = None
    ) -> List[int]:
        if not self.enabled:
            return []

        num_full_blocks = len(token_ids) // self.block_size
        if max_blocks is not None:
            num_full_blocks = min(num_full_blocks, max_blocks)
        if num_full_blocks <= 0:
            return []

        self.num_queries += 1
        self.num_queried_tokens += num_full_blocks * self.block_size

        matched: List[int] = []
        node = self.root
        parent_hash: Optional[int] = None

        for i in range(num_full_blocks):
            chunk = tuple(token_ids[i * self.block_size : (i + 1) * self.block_size])
            key = block_hash(parent_hash, chunk)
            child = node.children.get(key)
            if child is None or child.token_ids != chunk:
                break
            matched.append(child.physical_block)
            self._touch(child)
            node = child
            parent_hash = key

        if matched:
            self.num_hits += 1
            self.num_cached_tokens += len(matched) * self.block_size
        return matched

    def insert(self, token_ids: Sequence[int], physical_blocks: Sequence[int]) -> int:
        if not self.enabled:
            return 0

        num_full_blocks = min(len(token_ids) // self.block_size, len(physical_blocks))
        if num_full_blocks <= 0:
            return 0

        node = self.root
        parent_hash: Optional[int] = None
        created = 0

        for i in range(num_full_blocks):
            chunk = tuple(token_ids[i * self.block_size : (i + 1) * self.block_size])
            key = block_hash(parent_hash, chunk)
            child = node.children.get(key)

            if child is not None and child.token_ids == chunk:
                self._touch(child)
                node = child
                parent_hash = key
                continue

            if child is not None:
                break

            physical = physical_blocks[i]
            new_node = RadixNode(node_hash=key, token_ids=chunk, physical_block=physical, parent=node)
            self.allocator.incref(physical)
            node.children[key] = new_node
            self._nodes[id(new_node)] = new_node
            created += 1
            self._touch(new_node)
            node = new_node
            parent_hash = key

        return created

    def evict(self, num_blocks: int) -> int:
        if num_blocks <= 0 or not self._nodes:
            return 0

        freed = 0
        deferred: List[Tuple[int, int, RadixNode]] = []

        while freed < num_blocks and self._evict_heap:
            entry = heapq.heappop(self._evict_heap)
            last_access, _node_id, node = entry

            if id(node) not in self._nodes or node.last_access != last_access or not node.is_leaf:
                continue

            if self.allocator.ref_count(node.physical_block) != 1:
                # Still held by a live request. Keep the entry or the node becomes
                # permanently unevictable.
                deferred.append(entry)
                continue

            parent = node.parent
            self._remove_node(node)
            freed += 1

            if parent is not None and not parent.is_root and parent.is_leaf:
                self._push_leaf(parent)

        for entry in deferred:
            heapq.heappush(self._evict_heap, entry)
        return freed

    def _remove_node(self, node: RadixNode) -> None:
        parent = node.parent
        if parent is not None and node.node_hash is not None:
            parent.children.pop(node.node_hash, None)
        self._nodes.pop(id(node), None)
        self._heap_ids.pop(id(node), None)
        self.allocator.free(node.physical_block)
        node.parent = None

    def clear(self) -> None:
        for node in list(self._nodes.values()):
            self.allocator.free(node.physical_block)
        self._nodes.clear()
        self._heap_ids.clear()
        self._evict_heap.clear()
        self.root.children.clear()
        self.num_hits = 0
        self.num_queries = 0
        self.num_cached_tokens = 0
        self.num_queried_tokens = 0

    def iter_nodes(self) -> Iterator[RadixNode]:
        return iter(self._nodes.values())

    def assert_consistent(self) -> None:
        seen: Dict[int, RadixNode] = {}
        stack = [(self.root, None)]
        while stack:
            node, parent_hash = stack.pop()
            for key, child in node.children.items():
                if child.parent is not node:
                    raise AssertionError("child.parent does not point back at its parent")
                if child.node_hash != key:
                    raise AssertionError("child stored under a key that is not its hash")
                expected = block_hash(parent_hash, child.token_ids)
                if expected != key:
                    raise AssertionError("node hash is not consistent with its token chain")
                if id(child) not in self._nodes:
                    raise AssertionError("reachable node missing from the node registry")
                seen[id(child)] = child
                stack.append((child, key))

        if set(seen) != set(self._nodes):
            raise AssertionError("node registry and tree disagree on membership")
        for node in self._nodes.values():
            if self.allocator.ref_count(node.physical_block) < 1:
                raise AssertionError(
                    f"cached block {node.physical_block} has no outstanding reference"
                )

    def stats(self) -> Dict[str, float]:
        return {
            "nodes": float(len(self._nodes)),
            "queries": float(self.num_queries),
            "hits": float(self.num_hits),
            "cached_tokens": float(self.num_cached_tokens),
            "queried_tokens": float(self.num_queried_tokens),
            "hit_rate": self.hit_rate,
        }
