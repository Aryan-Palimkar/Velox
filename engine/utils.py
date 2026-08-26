from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch


@dataclass
class SamplingParams:
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    ignore_eos: bool = False
    max_tokens: int = 512
    seed: Optional[int] = None
    stop_token_ids: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.n != 1:
            raise ValueError("only n=1 is supported")
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative (0 disables the filter)")
        if self.repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be positive")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0

    @property
    def has_penalties(self) -> bool:
        return (
            self.repetition_penalty != 1.0
            or self.presence_penalty != 0.0
            or self.frequency_penalty != 0.0
        )

    @staticmethod
    def from_optional(
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        ignore_eos: bool = False,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        stop_token_ids: Optional[Sequence[int]] = None,
    ) -> "SamplingParams":
        return SamplingParams(
            n=1 if n is None else n,
            temperature=1.0 if temperature is None else temperature,
            top_p=1.0 if top_p is None else top_p,
            top_k=0 if top_k is None else top_k,
            repetition_penalty=1.0 if repetition_penalty is None else repetition_penalty,
            presence_penalty=0.0 if presence_penalty is None else presence_penalty,
            frequency_penalty=0.0 if frequency_penalty is None else frequency_penalty,
            ignore_eos=ignore_eos,
            max_tokens=512 if max_tokens is None else max_tokens,
            seed=seed,
            stop_token_ids=list(stop_token_ids or []),
        )


@dataclass
class BatchMetadata:
    is_prefill: bool
    num_tokens: int
    num_seqs: int

    positions: torch.Tensor
    slot_mapping: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor

    cu_seq_lens_q: Optional[torch.Tensor] = None

    max_q_len: int = 1
    max_seq_len: int = 1
    num_splits: int = 1
    decode_workspace: Optional[object] = None

    seq_lens_cpu: List[int] = field(default_factory=list)
    query_lens_cpu: List[int] = field(default_factory=list)

    def logits_indices(self) -> torch.Tensor:
        if self.is_prefill:
            assert self.cu_seq_lens_q is not None
            return self.cu_seq_lens_q[1:].to(torch.long) - 1
        return torch.arange(self.num_seqs, device=self.positions.device)
