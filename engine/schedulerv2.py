from typing import Dict, List, Optional, Tuple
from collections import deque
from .request import Request, RequestStatus
from src.kv_cache_batched import BatchedKVCache

class Scheduler:
    def __init__(
        self,
        kv_cache: BatchedKVCache,
        max_batch_size: int = 32,
        max_num_batched_tokens: int = 2048,
        max_prefill_chunk_size: int = 256,
    ):
        self.kv_cache = kv_cache
        self.max_batch_size = min(max_batch_size, kv_cache.max_batch_size)
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_prefill_chunk_size = max_prefill_chunk_size

        self.waiting_queue: deque[Request] = deque()
        self.running_queue: deque[Request] = deque()
        self.requests: Dict[str, Request] = {}

    def add_request(self, request: Request):
        self.waiting_queue.append(request)
        self.requests[request.request_id] = request

    def schedule(self) -> Tuple[List[Request], List[Request], bool]:
        active_running = deque()
        for req in self.running_queue:
            if req.finished or req.status == RequestStatus.ABORTED:
                if req.cache_slot is not None:
                    self.kv_cache.free_slot(req.cache_slot)
                    req.cache_slot = None
            else:
                active_running.append(req)
        self.running_queue = active_running

        prefill_batch: List[Request] = []
        decode_batch: List[Request] = []

        for req in self.running_queue:
            if req.status == RequestStatus.RUNNING_DECODE:
                decode_batch.append(req)

        remaining_slots = self.max_batch_size - len(self.running_queue)
        remaining_tokens = self.max_num_batched_tokens - len(decode_batch)

        while self.waiting_queue and remaining_tokens > 0:
            req = self.waiting_queue[0]

            uncomputed_tokens = req.num_prompt_tokens - req.num_computed_tokens

            chunk_size = min(uncomputed_tokens, remaining_tokens, self.max_prefill_chunk_size)

            if chunk_size <= 0:
                break

            if req.status == RequestStatus.WAITING_PREFILL:
                if remaining_slots <= 0:
                    break
                req = self.waiting_queue.popleft()
                req.cache_slot = self.kv_cache.allocate_slot()
                req.status = RequestStatus.RUNNING_PREFILL
                self.running_queue.append(req)
                remaining_slots -= 1
            else:
                req = self.waiting_queue.popleft()
                if req not in self.running_queue:
                    self.running_queue.append(req)

            req.current_chunk_size = chunk_size
            prefill_batch.append(req)
            remaining_tokens -= chunk_size

        has_waiting = len(self.waiting_queue) > 0
        return prefill_batch, decode_batch, has_waiting

    def get_request(self, request_id: str) -> Optional[Request]:
        return self.requests.get(request_id)

    def abort_request(self, request_id: str) -> bool:
        req = self.requests.get(request_id)
        if req is None or req.status == RequestStatus.ABORTED:
            return False

        req.status = RequestStatus.ABORTED
        req.finished = True
        req.stop_reason = "aborted"

        if req.cache_slot is not None:
            self.kv_cache.free_slot(req.cache_slot)
            req.cache_slot = None

        self.waiting_queue = deque([r for r in self.waiting_queue if r.request_id != request_id])
        self.running_queue = deque([r for r in self.running_queue if r.request_id != request_id])

        return True
