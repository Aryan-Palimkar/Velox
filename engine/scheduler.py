from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from src.block_manager import BlockSpaceManager
from src.paged_kv_cache import PagedKVCache

from .request import FinishReason, Request, RequestStatus

logger = logging.getLogger(__name__)


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 2048
    max_prefill_chunk_size: int = 512
    max_model_len: int = 4096
    enable_prefix_caching: bool = True
    watermark: float = 0.01


@dataclass
class SchedulerOutput:
    prefill_batch: List[Request] = field(default_factory=list)
    decode_batch: List[Request] = field(default_factory=list)
    preempted: List[Request] = field(default_factory=list)
    num_batched_tokens: int = 0

    def is_empty(self) -> bool:
        return not self.prefill_batch and not self.decode_batch


class Scheduler:
    def __init__(
        self,
        kv_cache: PagedKVCache,
        config: Optional[SchedulerConfig] = None,
        block_manager: Optional[BlockSpaceManager] = None,
        **overrides,
    ) -> None:
        self.config = config or SchedulerConfig(**overrides)
        self.kv_cache = kv_cache
        self.block_manager = block_manager or BlockSpaceManager(
            kv_cache,
            enable_prefix_caching=self.config.enable_prefix_caching,
            max_model_len=self.config.max_model_len,
        )
        self.config.max_model_len = self.block_manager.max_model_len

        self.waiting: Deque[Request] = deque()
        self.running: List[Request] = []
        self.requests: Dict[str, Request] = {}

        self._free_seq_indices: Deque[int] = deque(range(self.config.max_num_seqs))
        self._watermark_blocks = max(1, int(self.block_manager.num_total_blocks * self.config.watermark))

        self.num_preemptions = 0
        self.num_finished = 0

        self.on_sequence_admitted = None
        self.on_sequence_released = None

    def add_request(self, request: Request) -> None:
        limit = self.config.max_model_len
        if request.num_prompt_tokens >= limit:
            raise ValueError(
                f"prompt of {request.num_prompt_tokens} tokens does not fit in "
                f"max_model_len={limit} (at least one token must remain for generation)"
            )
        room = limit - request.num_prompt_tokens
        if request.sampling_params.max_tokens > room:
            raise ValueError(
                f"prompt of {request.num_prompt_tokens} tokens plus "
                f"max_tokens={request.sampling_params.max_tokens} exceeds "
                f"max_model_len={limit}"
            )
        self.requests[request.request_id] = request
        self.waiting.append(request)

    def get_request(self, request_id: str) -> Optional[Request]:
        return self.requests.get(request_id)

    def abort_request(self, request_id: str) -> bool:
        request = self.requests.get(request_id)
        if request is None or request.finished:
            return False
        request.finish(FinishReason.ABORT)
        self._release(request)
        try:
            self.waiting.remove(request)
        except ValueError:
            pass
        if request in self.running:
            self.running.remove(request)
        self.requests.pop(request_id, None)
        return True

    @property
    def num_waiting(self) -> int:
        return len(self.waiting)

    @property
    def num_running(self) -> int:
        return len(self.running)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def _acquire_seq_index(self, request: Request) -> bool:
        if request.seq_index is not None:
            return True
        if not self._free_seq_indices:
            return False
        request.seq_index = self._free_seq_indices.popleft()
        if self.on_sequence_admitted is not None:
            self.on_sequence_admitted(request)
        return True

    def _release_seq_index(self, request: Request) -> None:
        if request.seq_index is None:
            return
        if self.on_sequence_released is not None:
            self.on_sequence_released(request)
        self._free_seq_indices.append(request.seq_index)
        request.seq_index = None

    def _release(self, request: Request) -> None:
        self.block_manager.commit(request)
        self.block_manager.free(request)
        self._release_seq_index(request)

    def _preempt(self, request: Request) -> None:
        self.block_manager.commit(request)
        self.block_manager.reset_for_recompute(request)
        self._release_seq_index(request)
        request.status = RequestStatus.PREEMPTED
        request.num_preemptions += 1
        self.num_preemptions += 1
        if request in self.running:
            self.running.remove(request)
        self.waiting.appendleft(request)
        logger.debug("preempted %s (total preemptions: %d)", request.request_id, self.num_preemptions)

    def _make_room(self, request: Request, num_tokens: int, output: SchedulerOutput) -> bool:
        while not self.block_manager.can_append(request, num_tokens):
            if not self.running:
                return False
            # Always the newest, even when that is the caller. Skipping past it to an
            # older request is what would let a late arrival starve an early one.
            victim = self.running[-1]
            self._preempt(victim)
            output.preempted.append(victim)
            if victim is request:
                return False
        return True

    def schedule(self) -> SchedulerOutput:
        output = SchedulerOutput()
        self._reap_finished()

        token_budget = self.config.max_num_batched_tokens
        self._schedule_decodes(output, token_budget)
        token_budget -= len(output.decode_batch)

        token_budget = self._schedule_running_prefills(output, token_budget)
        self._admit_new_requests(output, token_budget)

        output.num_batched_tokens = len(output.decode_batch) + sum(
            req.current_chunk_size for req in output.prefill_batch
        )
        return output

    def _reap_finished(self) -> None:
        if not self.running:
            return
        still_running = []
        for request in self.running:
            if request.finished or request.status is RequestStatus.ABORTED:
                self._release(request)
                self.requests.pop(request.request_id, None)
                self.num_finished += 1
            else:
                still_running.append(request)
        self.running = still_running

    def _schedule_decodes(self, output: SchedulerOutput, token_budget: int) -> None:
        for request in list(self.running):
            if request.status is not RequestStatus.RUNNING_DECODE:
                continue
            if request not in self.running:
                continue
            if token_budget - len(output.decode_batch) <= 0:
                break

            if not self._make_room(request, 1, output) or not self.block_manager.append_slots(
                request, 1
            ):
                if request in self.running:
                    self._preempt(request)
                    output.preempted.append(request)
                continue

            request.current_chunk_size = 1
            output.decode_batch.append(request)

    def _max_chunk(self, request: Request, headroom_blocks: int) -> int:
        manager = self.block_manager
        max_blocks = min(
            len(request.block_table) + max(0, headroom_blocks), manager.max_blocks_per_seq
        )
        limit = min(
            max_blocks * manager.block_size - request.num_computed_tokens,
            manager.max_model_len - request.num_computed_tokens,
        )
        return max(0, limit)

    def _schedule_running_prefills(self, output: SchedulerOutput, token_budget: int) -> int:
        is_oldest_prefill = True
        for request in list(self.running):
            if request.status is not RequestStatus.RUNNING_PREFILL:
                continue
            if request not in self.running:
                continue
            if token_budget <= 0:
                break

            oldest = is_oldest_prefill
            is_oldest_prefill = False

            wanted = min(
                request.num_uncomputed_tokens,
                token_budget,
                self.config.max_prefill_chunk_size,
            )
            chunk = min(wanted, self._max_chunk(request, self.block_manager.num_available_blocks()))

            while chunk <= 0 and oldest and self.running and self.running[-1] is not request:
                victim = self.running[-1]
                self._preempt(victim)
                output.preempted.append(victim)
                chunk = min(
                    wanted, self._max_chunk(request, self.block_manager.num_available_blocks())
                )

            if chunk <= 0 or not self.block_manager.append_slots(request, chunk):
                continue

            request.current_chunk_size = chunk
            output.prefill_batch.append(request)
            token_budget -= chunk
        return token_budget

    def _admit_new_requests(self, output: SchedulerOutput, token_budget: int) -> None:
        while self.waiting and token_budget > 0:
            if len(self.running) >= self.config.max_num_seqs:
                break

            request = self.waiting[0]
            if not self._acquire_seq_index(request):
                break

            self.block_manager.match_prefix(request)
            wanted = min(
                request.num_uncomputed_tokens,
                token_budget,
                self.config.max_prefill_chunk_size,
            )
            headroom = self.block_manager.num_available_blocks() - self._watermark_blocks
            chunk = min(wanted, self._max_chunk(request, headroom))

            if chunk <= 0 or not self.block_manager.append_slots(request, chunk):
                self._rollback_admission(request)
                break

            self.waiting.popleft()
            request.status = RequestStatus.RUNNING_PREFILL
            request.current_chunk_size = chunk
            self.running.append(request)
            output.prefill_batch.append(request)
            token_budget -= chunk

    def _rollback_admission(self, request: Request) -> None:
        self.block_manager.free(request)
        request.num_computed_tokens = 0
        request.num_cached_tokens = 0
        request.num_published_blocks = 0
        self._release_seq_index(request)

    def commit(self, request: Request) -> None:
        self.block_manager.commit(request)

    def finish(self, request: Request, reason: FinishReason) -> None:
        request.finish(reason)

    def stats(self) -> Dict[str, float]:
        block_stats = self.block_manager.stats()
        return {
            "num_running": float(len(self.running)),
            "num_waiting": float(len(self.waiting)),
            "num_preemptions": float(self.num_preemptions),
            "num_finished": float(self.num_finished),
            **{k: float(v) for k, v in block_stats.items()},
        }
