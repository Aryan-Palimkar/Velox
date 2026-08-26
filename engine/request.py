from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, List, Optional

from engine.utils import SamplingParams

logger = logging.getLogger(__name__)


class RequestStatus(Enum):
    WAITING = auto()
    RUNNING_PREFILL = auto()
    RUNNING_DECODE = auto()
    PREEMPTED = auto()
    FINISHED = auto()
    ABORTED = auto()


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    ABORT = "abort"


@dataclass(frozen=True)
class RequestEvent:
    kind: str
    token_id: Optional[int] = None
    finish_reason: Optional[FinishReason] = None


Listener = Callable[["Request", RequestEvent], None]


class Request:
    __slots__ = (
        "request_id",
        "arrival_time",
        "status",
        "prompt_token_ids",
        "num_prompt_tokens",
        "sampling_params",
        "output_token_ids",
        "all_token_ids",
        "num_computed_tokens",
        "num_cached_tokens",
        "num_published_blocks",
        "block_table",
        "seq_index",
        "current_chunk_size",
        "finished",
        "finish_reason",
        "num_preemptions",
        "first_token_time",
        "finish_time",
        "_listeners",
    )

    def __init__(
        self,
        request_id: str,
        prompt_token_ids: List[int],
        sampling_params: SamplingParams,
        arrival_time: Optional[float] = None,
    ) -> None:
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must contain at least one token")

        self.request_id = request_id
        self.arrival_time = arrival_time if arrival_time is not None else time.time()
        self.status = RequestStatus.WAITING

        self.prompt_token_ids = prompt_token_ids
        self.num_prompt_tokens = len(prompt_token_ids)
        self.sampling_params = sampling_params

        self.output_token_ids: List[int] = []
        self.all_token_ids: List[int] = list(prompt_token_ids)

        self.num_computed_tokens = 0
        self.num_cached_tokens = 0
        self.num_published_blocks = 0

        self.block_table: List[int] = []
        self.seq_index: Optional[int] = None
        self.current_chunk_size = 0

        self.finished = False
        self.finish_reason: Optional[FinishReason] = None
        self.num_preemptions = 0

        self.first_token_time: Optional[float] = None
        self.finish_time: Optional[float] = None

        self._listeners: List[Listener] = []

    @property
    def num_tokens(self) -> int:
        return len(self.all_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def num_uncomputed_tokens(self) -> int:
        return max(0, len(self.all_token_ids) - self.num_computed_tokens)

    @property
    def is_decoding(self) -> bool:
        return self.num_uncomputed_tokens <= 1

    @property
    def max_tokens(self) -> int:
        return self.sampling_params.max_tokens

    def tokens_for_chunk(self, chunk_size: int) -> List[int]:
        start = self.num_computed_tokens
        return self.all_token_ids[start : start + chunk_size]

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)
        self.all_token_ids.append(token_id)
        if self.first_token_time is None:
            self.first_token_time = time.time()
        self._emit(RequestEvent(kind="token", token_id=token_id))

    def refresh_status(self) -> None:
        if self.finished:
            return
        self.status = (
            RequestStatus.RUNNING_DECODE if self.is_decoding else RequestStatus.RUNNING_PREFILL
        )

    def should_stop(self, token_id: int, eos_token_ids: frozenset[int]) -> Optional[FinishReason]:
        params = self.sampling_params
        if not params.ignore_eos and (
            token_id in eos_token_ids or token_id in params.stop_token_ids
        ):
            return FinishReason.STOP
        if len(self.output_token_ids) >= params.max_tokens:
            return FinishReason.LENGTH
        return None

    def finish(self, reason: FinishReason) -> None:
        if self.finished:
            return
        self.finished = True
        self.finish_reason = reason
        self.finish_time = time.time()
        self.status = RequestStatus.ABORTED if reason is FinishReason.ABORT else RequestStatus.FINISHED
        self._emit(RequestEvent(kind="finish", finish_reason=reason))

    def add_listener(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def _emit(self, event: RequestEvent) -> None:
        for listener in self._listeners:
            try:
                listener(self, event)
            except Exception:
                logger.exception("request %s: listener raised", self.request_id)

    @property
    def time_to_first_token(self) -> Optional[float]:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    def __repr__(self) -> str:
        return (
            f"Request(id={self.request_id}, status={self.status.name}, "
            f"prompt={self.num_prompt_tokens}, computed={self.num_computed_tokens}, "
            f"output={len(self.output_token_ids)}, blocks={len(self.block_table)})"
        )
