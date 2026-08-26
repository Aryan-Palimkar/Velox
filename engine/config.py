from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

_DTYPES = {
    "auto": None,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def resolve_dtype(name: str) -> torch.dtype:
    key = (name or "auto").lower()
    if key not in _DTYPES:
        raise ValueError(f"unknown dtype {name!r}; expected one of {sorted(_DTYPES)}")
    dtype = _DTYPES[key]
    if dtype is not None:
        return dtype
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


@dataclass
class EngineConfig:
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dtype: str = "auto"
    device: str = "cuda"
    download_dir: Optional[str] = None

    max_model_len: int = 4096
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 2048
    max_prefill_chunk_size: int = 512

    block_size: int = 16
    num_gpu_blocks: Optional[int] = None
    gpu_memory_utilization: float = 0.90
    kv_cache_reserve_mb: int = 512
    kv_cache_dtype: str = "auto"

    enable_prefix_caching: bool = True
    enable_cuda_graphs: bool = True
    quantization: Optional[str] = None

    seed: Optional[int] = None

    def torch_dtype(self) -> torch.dtype:
        return resolve_dtype(self.dtype)

    def validate(self) -> None:
        if self.max_num_seqs < 1:
            raise ValueError("max_num_seqs must be at least 1")
        if self.max_num_batched_tokens < self.max_prefill_chunk_size:
            raise ValueError(
                "max_num_batched_tokens must be at least max_prefill_chunk_size"
            )
        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                "max_num_batched_tokens must leave room for one token per running sequence"
            )
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.block_size <= 0 or (self.block_size & (self.block_size - 1)) != 0:
            raise ValueError("block_size must be a power of two")
