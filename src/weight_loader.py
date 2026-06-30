from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM

from .config import ModelConfig
from .tokenizer import Tokenizer
from .transformer import QwenForCausalLM


def load_hf_state_dict(model_name: str, dtype: torch.dtype, cache_dir: Optional[str] = None) -> Dict[str, torch.Tensor]:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=dtype,
        trust_remote_code=False,
    )
    state_dict = model.state_dict()
    del model
    return state_dict


def load_hf_assets(
    model_name: str, dtype: torch.dtype, cache_dir: Optional[str] = None
) -> Tuple[ModelConfig, Tokenizer, Dict[str, torch.Tensor]]:
    config = ModelConfig.from_hf(model_name, cache_dir=cache_dir)
    tokenizer = Tokenizer(model_name, cache_dir=cache_dir)
    state_dict = load_hf_state_dict(model_name, dtype=dtype, cache_dir=cache_dir)
    return config, tokenizer, state_dict


def load_weights_into_model(model: QwenForCausalLM, state_dict: Dict[str, torch.Tensor]) -> None:
    model_params = dict(model.named_parameters())
    loaded = []
    missing = []
    unexpected = []

    with torch.no_grad():
        for name, param in model_params.items():
            if name not in state_dict:
                missing.append(name)
                continue
            src = state_dict[name]
            if src.shape != param.shape:
                raise ValueError(f"shape mismatch for {name}: {tuple(src.shape)} vs {tuple(param.shape)}")
            param.copy_(src.to(dtype=param.dtype))
            loaded.append((name, tuple(param.shape)))

    for name in state_dict.keys():
        if name not in model_params:
            unexpected.append(name)

    for name, shape in loaded:
        print(f"[weights] loaded {name} shape={shape}")
    if missing:
        print("[weights] missing tensors:")
        for name in missing:
            print(f"  - {name}")
    if unexpected:
        print("[weights] unexpected tensors:")
        for name in unexpected:
            print(f"  - {name}")
