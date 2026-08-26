from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

from .utils import SamplingParams

_SHORTLIST_SIZE = 1024


@dataclass
class SamplingTensors:
    temperature: torch.Tensor
    top_p: torch.Tensor
    top_k: torch.Tensor
    repetition_penalty: torch.Tensor
    presence_penalty: torch.Tensor
    frequency_penalty: torch.Tensor
    all_greedy: bool
    any_greedy: bool
    no_penalties: bool
    max_top_k: int

    @staticmethod
    def build(
        params: Sequence[SamplingParams], device: torch.device, vocab_size: int
    ) -> "SamplingTensors":
        rows = [
            [
                p.temperature,
                p.top_p,
                float(p.top_k if 0 < p.top_k < vocab_size else 0),
                p.repetition_penalty,
                p.presence_penalty,
                p.frequency_penalty,
            ]
            for p in params
        ]
        packed = torch.tensor(rows, dtype=torch.float32).pin_memory().to(device, non_blocking=True)
        top_k_values = [int(row[2]) for row in rows]
        return SamplingTensors(
            temperature=packed[:, 0],
            top_p=packed[:, 1],
            top_k=packed[:, 2],
            repetition_penalty=packed[:, 3],
            presence_penalty=packed[:, 4],
            frequency_penalty=packed[:, 5],
            all_greedy=all(p.is_greedy for p in params),
            any_greedy=any(p.is_greedy for p in params),
            no_penalties=not any(p.has_penalties for p in params),
            max_top_k=max(top_k_values) if top_k_values else 0,
        )


class TokenHistory:
    def __init__(self, max_num_seqs: int, max_model_len: int, device: torch.device | str) -> None:
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.device = torch.device(device)
        self.tokens = torch.zeros((max_num_seqs, max_model_len), dtype=torch.long, device=device)
        self.lengths = torch.zeros(max_num_seqs, dtype=torch.long, device=device)

    def reset_sequence(self, seq_index: int, prompt_token_ids: Sequence[int]) -> None:
        length = min(len(prompt_token_ids), self.max_model_len)
        row = torch.tensor(prompt_token_ids[:length], dtype=torch.long)
        self.tokens[seq_index, :length].copy_(row)
        self.lengths[seq_index] = length

    def append(self, seq_indices: torch.Tensor, token_ids: torch.Tensor) -> None:
        lengths = self.lengths.index_select(0, seq_indices)
        positions = lengths.clamp(max=self.max_model_len - 1)
        self.tokens[seq_indices, positions] = token_ids
        self.lengths[seq_indices] = (lengths + 1).clamp(max=self.max_model_len)

    def gather(self, seq_indices: torch.Tensor, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        width = max(1, min(width, self.max_model_len))
        lengths = self.lengths.index_select(0, seq_indices)
        tokens = self.tokens.index_select(0, seq_indices)[:, :width]
        mask = torch.arange(width, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
        return tokens, mask


class Sampler:
    def __init__(self, device: torch.device | str = "cuda") -> None:
        self.device = torch.device(device)
        self._generators: Dict[str, torch.Generator] = {}

    def sample(
        self,
        logits: torch.Tensor,
        sampling_params: Sequence[SamplingParams],
        request_ids: Optional[Sequence[str]] = None,
        history: Optional[TokenHistory] = None,
        seq_indices: Optional[torch.Tensor] = None,
        history_width: int = 0,
    ) -> torch.Tensor:
        batch, vocab_size = logits.shape
        if len(sampling_params) != batch:
            raise ValueError(
                f"got {len(sampling_params)} sampling params for a batch of {batch}"
            )

        tensors = SamplingTensors.build(sampling_params, self.device, vocab_size)
        logits = logits.to(torch.float32)

        if not tensors.no_penalties and history is not None and seq_indices is not None:
            logits = self._apply_penalties(logits, tensors, history, seq_indices, history_width)

        if tensors.all_greedy:
            return logits.argmax(dim=-1)

        greedy_tokens = logits.argmax(dim=-1) if tensors.any_greedy else None

        temperature = tensors.temperature.clamp_min(1e-5).unsqueeze(1)
        scaled = logits / temperature

        tokens = self._sample_from_shortlist(scaled, tensors, sampling_params, request_ids)

        if greedy_tokens is not None:
            is_greedy = tensors.temperature == 0.0
            tokens = torch.where(is_greedy, greedy_tokens, tokens)
        return tokens

    def _apply_penalties(
        self,
        logits: torch.Tensor,
        tensors: SamplingTensors,
        history: TokenHistory,
        seq_indices: torch.Tensor,
        width: int,
    ) -> torch.Tensor:
        tokens, mask = history.gather(seq_indices, width)
        active = mask.to(logits.dtype)

        frequency = tensors.frequency_penalty.unsqueeze(1)
        if torch.any(tensors.frequency_penalty != 0.0):
            logits.scatter_add_(1, tokens, -frequency * active)

        gathered = logits.gather(1, tokens)
        repetition = tensors.repetition_penalty.unsqueeze(1)
        penalized = torch.where(gathered > 0, gathered / repetition, gathered * repetition)
        penalized = penalized - tensors.presence_penalty.unsqueeze(1)
        logits.scatter_(1, tokens, torch.where(mask, penalized, gathered))
        return logits

    def _sample_from_shortlist(
        self,
        logits: torch.Tensor,
        tensors: SamplingTensors,
        sampling_params: Sequence[SamplingParams],
        request_ids: Optional[Sequence[str]],
    ) -> torch.Tensor:
        vocab_size = logits.shape[1]
        shortlist = min(vocab_size, max(_SHORTLIST_SIZE, tensors.max_top_k))

        log_norm = torch.logsumexp(logits, dim=-1, keepdim=True)
        values, indices = torch.topk(logits, shortlist, dim=-1)
        probs = torch.exp(values - log_norm)

        any_nucleus = any(p.top_p < 1.0 for p in sampling_params)
        needs_full_sort = (
            any_nucleus
            and shortlist < vocab_size
            and bool(((probs.sum(dim=-1) < tensors.top_p) & (tensors.top_p < 1.0)).any())
        )
        if needs_full_sort:
            values, indices = torch.sort(logits, dim=-1, descending=True)
            probs = torch.exp(values - log_norm)
            shortlist = vocab_size

        keep = torch.ones_like(probs, dtype=torch.bool)

        if tensors.max_top_k > 0:
            ranks = torch.arange(shortlist, device=logits.device).unsqueeze(0)
            top_k = tensors.top_k.to(torch.long).unsqueeze(1)
            keep &= (top_k == 0) | (ranks < top_k)

        exclusive_cumulative = probs.cumsum(dim=-1) - probs
        keep &= exclusive_cumulative < tensors.top_p.unsqueeze(1)

        probs = probs * keep
        total = probs.sum(dim=-1, keepdim=True)
        probs = probs / torch.where(total > 0, total, torch.ones_like(total))

        noise = self._exponential_noise(probs.shape, sampling_params, request_ids, probs.device)
        local = (probs / noise).argmax(dim=-1)
        return indices.gather(1, local.unsqueeze(1)).squeeze(1)

    def _exponential_noise(
        self,
        shape: torch.Size,
        sampling_params: Sequence[SamplingParams],
        request_ids: Optional[Sequence[str]],
        device: torch.device,
    ) -> torch.Tensor:
        noise = torch.empty(shape, dtype=torch.float32, device=device)
        seeded_rows = [
            (i, p.seed) for i, p in enumerate(sampling_params) if p.seed is not None
        ]

        if len(seeded_rows) < shape[0]:
            noise.exponential_()

        for row, seed in seeded_rows:
            key = request_ids[row] if request_ids is not None else f"row-{row}-{seed}"
            generator = self._generators.get(key)
            if generator is None:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)
                self._generators[key] = generator
            noise[row].exponential_(generator=generator)

        return noise.clamp_min(1e-10)

    def release(self, request_id: str) -> None:
        self._generators.pop(request_id, None)
