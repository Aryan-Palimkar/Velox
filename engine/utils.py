from dataclasses import dataclass
import torch

@dataclass
class SamplingParams:
    n: int = 1
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.05
    ignore_eos: bool = False
    max_tokens: int = 512

    @staticmethod
    def from_optional(
        n: int | None = 1,
        temperature: float | None = 1.0,
        top_p: float | None = 1.0,
        top_k: int = 1,
        repetition_penalty: float | None = None,
        ignore_eos: bool = False,
        max_tokens: int | None = 32
    ) -> "SamplingParams":
        return SamplingParams(
            n=1 if n is None else n,
            temperature=0.7 if temperature is None else temperature,
            top_p=0.8 if top_p is None else top_p,
            top_k=20 if top_k is None else top_k,
            repetition_penalty=1.05 if repetition_penalty is None else repetition_penalty,
            ignore_eos=ignore_eos,
            max_tokens=512 if max_tokens is None else max_tokens
        )


@dataclass
class BatchMetadata:
    cu_seq_lens_q: torch.Tensor
    cu_seq_lens_kv: torch.Tensor
    max_q_len: int
    max_kv_len: int
    positions: torch.Tensor
    cache_slots: torch.Tensor | list[int]
    seq_lens: torch.Tensor | list[int]
    prompt_lens: list[int]

    cu_seq_lens_q_cpu: list[int]
    seq_lens_cpu: list[int]
    cache_slots_cpu: list[int]
