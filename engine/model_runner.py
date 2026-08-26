from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.block_manager import BlockSpaceManager
from src.decode_attention import DecodeWorkspace
from src.paged_kv_cache import PagedKVCache

from .request import Request
from .utils import BatchMetadata

logger = logging.getLogger(__name__)


def _decode_buckets(max_num_seqs: int) -> List[int]:
    buckets: List[int] = []
    size = 1
    while size < max_num_seqs:
        buckets.append(size)
        size *= 2
    buckets.append(max_num_seqs)
    return buckets


def _next_power_of_two(value: int) -> int:
    result = 1
    while result < value:
        result *= 2
    return result


class BatchBuffers:
    def __init__(
        self,
        max_num_tokens: int,
        max_num_seqs: int,
        max_blocks_per_seq: int,
        device: torch.device,
    ) -> None:
        def gpu(shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=device)

        def pinned(shape, dtype):
            return torch.zeros(
                shape, dtype=dtype, device="cpu", pin_memory=device.type == "cuda"
            )

        self.input_ids = gpu((max_num_tokens,), torch.long)
        self.positions = gpu((max_num_tokens,), torch.int32)
        self.slot_mapping = gpu((max_num_tokens,), torch.long)
        self.seq_lens = gpu((max_num_seqs,), torch.int32)
        self.cu_seq_lens_q = gpu((max_num_seqs + 1,), torch.int32)
        self.block_tables = gpu((max_num_seqs, max_blocks_per_seq), torch.int32)
        self.gather_index = gpu((max_num_seqs,), torch.long)

        self.h_input_ids = pinned((max_num_tokens,), torch.long)
        self.h_positions = pinned((max_num_tokens,), torch.int32)
        self.h_slot_mapping = pinned((max_num_tokens,), torch.long)
        self.h_seq_lens = pinned((max_num_seqs,), torch.int32)
        self.h_cu_seq_lens_q = pinned((max_num_seqs + 1,), torch.int32)
        self.h_gather_index = pinned((max_num_seqs,), torch.long)
        self.h_block_row = pinned((max_blocks_per_seq,), torch.int32)

    @staticmethod
    def stage(host: torch.Tensor, device: torch.Tensor, values: np.ndarray) -> None:
        count = values.shape[0]
        host[:count] = torch.from_numpy(np.ascontiguousarray(values)).to(host.dtype)
        device[:count].copy_(host[:count], non_blocking=True)


class ModelRunner:
    def __init__(
        self,
        model: torch.nn.Module,
        kv_cache: PagedKVCache,
        block_manager: BlockSpaceManager,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        device: torch.device | str = "cuda",
        enable_cuda_graphs: bool = True,
    ) -> None:
        self.model = model
        self.kv_cache = kv_cache
        self.block_manager = block_manager
        self.device = torch.device(device)
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_tokens = max(max_num_batched_tokens, max_num_seqs)
        self.max_blocks_per_seq = block_manager.max_blocks_per_seq
        self.block_size = kv_cache.block_size
        self.enable_cuda_graphs = enable_cuda_graphs

        num_q_heads = model.config.num_attention_heads
        num_kv_heads = model.config.num_key_value_heads
        head_dim = model.config.hidden_size // num_q_heads
        self.num_kv_heads = num_kv_heads
        self.block_h = max(16, _next_power_of_two(num_q_heads // num_kv_heads))

        self.prefill_buffers = BatchBuffers(
            self.max_num_tokens, max_num_seqs, self.max_blocks_per_seq, self.device
        )
        self.decode_buffers = BatchBuffers(
            max_num_seqs, max_num_seqs, self.max_blocks_per_seq, self.device
        )

        self.block_tables_by_seq = torch.zeros(
            (max_num_seqs + 1, self.max_blocks_per_seq), dtype=torch.int32, device=self.device
        )
        self.pad_row = max_num_seqs
        self._synced_blocks = [0] * (max_num_seqs + 1)

        self.max_splits = 8
        self.decode_workspace = DecodeWorkspace(
            max_batch_size=max_num_seqs,
            num_kv_heads=num_kv_heads,
            block_h=self.block_h,
            head_dim=head_dim,
            max_splits=self.max_splits,
            device=self.device,
        )

        self.graphs: Dict[int, Tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
        self.graph_splits: Dict[int, int] = {}
        self.graph_buckets: List[int] = []
        self._graph_pool = None

    def on_sequence_admitted(self, request: Request) -> None:
        self._synced_blocks[request.seq_index] = 0

    def on_sequence_released(self, request: Request) -> None:
        if request.seq_index is not None:
            self._synced_blocks[request.seq_index] = 0

    def _sync_block_table(self, request: Request, buffers: BatchBuffers) -> None:
        slot = request.seq_index
        synced = self._synced_blocks[slot]
        total = len(request.block_table)
        if total <= synced:
            return
        count = total - synced
        buffers.h_block_row[:count] = torch.tensor(
            request.block_table[synced:total], dtype=torch.int32
        )
        self.block_tables_by_seq[slot, synced:total].copy_(
            buffers.h_block_row[:count], non_blocking=True
        )
        self._synced_blocks[slot] = total

    def prepare_prefill(self, batch: Sequence[Request]) -> Tuple[torch.Tensor, BatchMetadata]:
        buffers = self.prefill_buffers

        input_ids: List[int] = []
        position_chunks: List[np.ndarray] = []
        slot_chunks: List[np.ndarray] = []
        cumulative = [0]
        kv_lens: List[int] = []
        query_lens: List[int] = []

        for request in batch:
            self._sync_block_table(request, buffers)
            start = request.num_computed_tokens
            length = request.current_chunk_size
            end = start + length

            input_ids.extend(request.all_token_ids[start:end])
            position_chunks.append(np.arange(start, end, dtype=np.int32))
            slot_chunks.append(self._slot_array(request, start, length))
            cumulative.append(cumulative[-1] + length)
            kv_lens.append(end)
            query_lens.append(length)

        num_tokens = cumulative[-1]
        num_seqs = len(batch)

        buffers.stage(buffers.h_input_ids, buffers.input_ids, np.asarray(input_ids, dtype=np.int64))
        buffers.stage(buffers.h_positions, buffers.positions, np.concatenate(position_chunks))
        buffers.stage(
            buffers.h_slot_mapping, buffers.slot_mapping, np.concatenate(slot_chunks)
        )
        buffers.stage(buffers.h_seq_lens, buffers.seq_lens, np.asarray(kv_lens, dtype=np.int32))
        buffers.stage(
            buffers.h_cu_seq_lens_q,
            buffers.cu_seq_lens_q,
            np.asarray(cumulative, dtype=np.int32),
        )
        width = self._gather_block_tables(batch, num_seqs, buffers)

        metadata = BatchMetadata(
            is_prefill=True,
            num_tokens=num_tokens,
            num_seqs=num_seqs,
            positions=buffers.positions[:num_tokens],
            slot_mapping=buffers.slot_mapping[:num_tokens],
            block_tables=buffers.block_tables[:num_seqs, :width],
            seq_lens=buffers.seq_lens[:num_seqs],
            cu_seq_lens_q=buffers.cu_seq_lens_q[: num_seqs + 1],
            max_q_len=max(query_lens),
            max_seq_len=max(kv_lens),
            seq_lens_cpu=kv_lens,
            query_lens_cpu=query_lens,
        )
        return buffers.input_ids[:num_tokens], metadata

    def prepare_decode(
        self, batch: Sequence[Request], padded_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, BatchMetadata]:
        buffers = self.decode_buffers
        num_seqs = len(batch)
        total = padded_size or num_seqs

        input_ids = np.zeros(total, dtype=np.int64)
        positions = np.zeros(total, dtype=np.int32)
        slots = np.zeros(total, dtype=np.int64)
        kv_lens = np.ones(total, dtype=np.int32)

        for row, request in enumerate(batch):
            self._sync_block_table(request, buffers)
            computed = request.num_computed_tokens
            input_ids[row] = request.all_token_ids[computed]
            positions[row] = computed
            slots[row] = self.block_manager.slot_for(request, computed)
            kv_lens[row] = computed + 1

        buffers.stage(buffers.h_input_ids, buffers.input_ids, input_ids)
        buffers.stage(buffers.h_positions, buffers.positions, positions)
        buffers.stage(buffers.h_slot_mapping, buffers.slot_mapping, slots)
        buffers.stage(buffers.h_seq_lens, buffers.seq_lens, kv_lens)
        width = self._gather_block_tables(batch, total, buffers)

        metadata = BatchMetadata(
            is_prefill=False,
            num_tokens=total,
            num_seqs=total,
            positions=buffers.positions[:total],
            slot_mapping=buffers.slot_mapping[:total],
            block_tables=buffers.block_tables[:total, :width],
            seq_lens=buffers.seq_lens[:total],
            max_q_len=1,
            max_seq_len=int(kv_lens.max()),
            seq_lens_cpu=kv_lens.tolist(),
            query_lens_cpu=[1] * total,
            decode_workspace=self.decode_workspace,
        )
        return buffers.input_ids[:total], metadata

    def _slot_array(self, request: Request, start: int, count: int) -> np.ndarray:
        block_size = self.block_size
        table = np.asarray(request.block_table, dtype=np.int64)
        indices = np.arange(start, start + count, dtype=np.int64)
        return table[indices // block_size] * block_size + (indices % block_size)

    def _gather_block_tables(
        self, batch: Sequence[Request], total: int, buffers: BatchBuffers
    ) -> int:
        rows = np.full(total, self.pad_row, dtype=np.int64)
        for row, request in enumerate(batch):
            rows[row] = request.seq_index
        buffers.stage(buffers.h_gather_index, buffers.gather_index, rows)

        torch.index_select(
            self.block_tables_by_seq,
            0,
            buffers.gather_index[:total],
            out=buffers.block_tables[:total],
        )
        width = max((len(request.block_table) for request in batch), default=1)
        return max(width, 1)

    def forward(self, input_ids: torch.Tensor, metadata: BatchMetadata) -> torch.Tensor:
        return self.model(input_ids=input_ids, kv_cache=self.kv_cache, metadata=metadata)

    def execute_prefill(self, batch: Sequence[Request]) -> torch.Tensor:
        input_ids, metadata = self.prepare_prefill(batch)
        hidden = self.forward(input_ids, metadata)
        return hidden.index_select(0, metadata.logits_indices())

    def execute_decode(self, batch: Sequence[Request]) -> torch.Tensor:
        bucket = self._pick_bucket(len(batch))
        if bucket is None:
            input_ids, metadata = self.prepare_decode(batch)
            metadata.num_splits = self._runtime_splits(len(batch), metadata.max_seq_len)
            return self.forward(input_ids, metadata)[: len(batch)]

        self.prepare_decode(batch, padded_size=bucket)
        graph, output = self.graphs[bucket]
        graph.replay()
        return output[: len(batch)]

    def _pick_bucket(self, batch_size: int) -> Optional[int]:
        if not self.graphs:
            return None
        for bucket in self.graph_buckets:
            if bucket >= batch_size:
                return bucket
        return None

    def _runtime_splits(self, batch_size: int, max_seq_len: int) -> int:
        from src.decode_attention import choose_num_splits

        return choose_num_splits(
            batch_size, self.num_kv_heads, max_seq_len, self.max_splits, self.device
        )

    @torch.no_grad()
    def capture_decode_graphs(self, warmup_iters: int = 2) -> None:
        if not self.enable_cuda_graphs or self.device.type != "cuda":
            return

        buffers = self.decode_buffers
        buckets = _decode_buckets(self.max_num_seqs)
        self._graph_pool = torch.cuda.graph_pool_handle()

        buffers.input_ids.zero_()
        buffers.positions.zero_()
        buffers.slot_mapping.zero_()
        buffers.block_tables.zero_()
        buffers.seq_lens.fill_(1)

        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream())

        for bucket in buckets:
            num_splits = self._capture_splits(bucket)
            metadata = self._graph_metadata(bucket, num_splits)
            input_ids = buffers.input_ids[:bucket]

            with torch.cuda.stream(stream):
                for _ in range(warmup_iters):
                    self.forward(input_ids, metadata)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._graph_pool):
                output = self.forward(input_ids, metadata)

            self.graphs[bucket] = (graph, output)
            self.graph_splits[bucket] = num_splits

        self.graph_buckets = sorted(self.graphs)
        torch.cuda.synchronize()
        logger.info("captured decode CUDA graphs for batch sizes %s", self.graph_buckets)

    def _capture_splits(self, bucket: int) -> int:
        num_sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        base = bucket * self.num_kv_heads
        if base >= num_sms:
            return 1
        return max(1, min(self.max_splits, (num_sms + base - 1) // base))

    def _graph_metadata(self, bucket: int, num_splits: int) -> BatchMetadata:
        buffers = self.decode_buffers
        return BatchMetadata(
            is_prefill=False,
            num_tokens=bucket,
            num_seqs=bucket,
            positions=buffers.positions[:bucket],
            slot_mapping=buffers.slot_mapping[:bucket],
            block_tables=buffers.block_tables[:bucket],
            seq_lens=buffers.seq_lens[:bucket],
            max_q_len=1,
            max_seq_len=1,
            num_splits=num_splits,
            seq_lens_cpu=[1] * bucket,
            query_lens_cpu=[1] * bucket,
            decode_workspace=self.decode_workspace,
        )

    @property
    def uses_cuda_graphs(self) -> bool:
        return bool(self.graphs)
