from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
from torch import nn

from src.quantization import QuantizedLinear, resolve_weight_dtype

logger = logging.getLogger(__name__)

_LAYER_TARGETS = (
    "self_attn.qkv_proj",
    "self_attn.o_proj",
    "mlp.gate_up_proj",
    "mlp.down_proj",
)


def _resolve(root: nn.Module, path: str) -> Tuple[nn.Module, str]:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _swap(root: nn.Module, path: str, quant_dtype: torch.dtype) -> int:
    parent, name = _resolve(root, path)
    linear = getattr(parent, name)
    if not isinstance(linear, nn.Linear):
        raise TypeError(f"{path} is a {type(linear).__name__}, not nn.Linear")

    quantized = QuantizedLinear.from_linear(linear, quant_dtype)
    setattr(parent, name, quantized)
    saved = linear.weight.numel() * (linear.weight.element_size() - quantized.qweight.element_size())
    del linear
    return saved


@torch.no_grad()
def quantize_model(
    model: nn.Module,
    quantization: str,
    include_lm_head: bool = True,
    targets: Optional[List[str]] = None,
) -> int:
    quant_dtype = resolve_weight_dtype(quantization)
    if quant_dtype is None:
        return 0

    if targets is None:
        num_layers = len(model.model.layers)
        targets = [
            f"model.layers.{index}.{name}"
            for index in range(num_layers)
            for name in _LAYER_TARGETS
        ]
        if include_lm_head:
            targets.append("lm_head")

    saved = 0
    for path in targets:
        saved += _swap(model, path, quant_dtype)

    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    logger.info(
        "quantized %d linear layers to %s, saving %.2f GB",
        len(targets),
        quantization,
        saved / 1e9,
    )
    return saved
