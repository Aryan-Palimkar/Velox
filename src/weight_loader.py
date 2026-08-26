from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file

from .config import ModelConfig

_SINGLE_FILE = "model.safetensors"
_INDEX_FILE = "model.safetensors.index.json"

_FUSED_SOURCES = (
    "self_attn.q_proj.",
    "self_attn.k_proj.",
    "self_attn.v_proj.",
    "mlp.gate_proj.",
    "mlp.up_proj.",
)


def _resolve_local_dir(model_id: str, cache_dir: Optional[str]) -> Optional[str]:
    if os.path.isdir(model_id):
        return model_id
    return None


def _list_shard_files(model_id: str, cache_dir: Optional[str]) -> List[str]:
    local_dir = _resolve_local_dir(model_id, cache_dir)

    if local_dir is not None:
        index_path = os.path.join(local_dir, _INDEX_FILE)
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as handle:
                index = json.load(handle)
            names = sorted(set(index["weight_map"].values()))
            return [os.path.join(local_dir, name) for name in names]
        single = os.path.join(local_dir, _SINGLE_FILE)
        if os.path.exists(single):
            return [single]
        raise FileNotFoundError(f"no safetensors weights found in {local_dir}")

    try:
        index_path = hf_hub_download(repo_id=model_id, filename=_INDEX_FILE, cache_dir=cache_dir)
    except Exception:
        return [hf_hub_download(repo_id=model_id, filename=_SINGLE_FILE, cache_dir=cache_dir)]

    with open(index_path, "r", encoding="utf-8") as handle:
        index = json.load(handle)
    names = sorted(set(index["weight_map"].values()))
    root = snapshot_download(repo_id=model_id, allow_patterns=names, cache_dir=cache_dir)
    return [os.path.join(root, name) for name in names]


def load_raw_state_dict(model_id: str, cache_dir: Optional[str] = None) -> Dict[str, torch.Tensor]:
    state_dict: Dict[str, torch.Tensor] = {}
    for shard in _list_shard_files(model_id, cache_dir):
        state_dict.update(load_file(shard, device="cpu"))
    if not state_dict:
        raise RuntimeError(f"checkpoint for {model_id} contained no tensors")
    return state_dict


def _is_fused_source(name: str) -> bool:
    return any(marker in name for marker in _FUSED_SOURCES)


def fuse_qwen_state_dict(
    raw: Dict[str, torch.Tensor],
    num_layers: int,
    tie_word_embeddings: bool,
) -> Dict[str, torch.Tensor]:
    fused: Dict[str, torch.Tensor] = {
        name: tensor for name, tensor in raw.items() if not _is_fused_source(name)
    }

    for layer in range(num_layers):
        prefix = f"model.layers.{layer}."
        try:
            q_w = raw[f"{prefix}self_attn.q_proj.weight"]
            k_w = raw[f"{prefix}self_attn.k_proj.weight"]
            v_w = raw[f"{prefix}self_attn.v_proj.weight"]
            gate_w = raw[f"{prefix}mlp.gate_proj.weight"]
            up_w = raw[f"{prefix}mlp.up_proj.weight"]
        except KeyError as exc:
            raise KeyError(f"checkpoint is missing {exc.args[0]} (layer {layer})") from exc

        fused[f"{prefix}self_attn.qkv_proj.weight"] = torch.cat([q_w, k_w, v_w], dim=0)
        fused[f"{prefix}mlp.gate_up_proj.weight"] = torch.cat([gate_w, up_w], dim=0)

        q_b = raw.get(f"{prefix}self_attn.q_proj.bias")
        k_b = raw.get(f"{prefix}self_attn.k_proj.bias")
        v_b = raw.get(f"{prefix}self_attn.v_proj.bias")
        if q_b is not None and k_b is not None and v_b is not None:
            fused[f"{prefix}self_attn.qkv_proj.bias"] = torch.cat([q_b, k_b, v_b], dim=0)

    if tie_word_embeddings and "lm_head.weight" not in fused:
        embed = fused.get("model.embed_tokens.weight")
        if embed is None:
            raise KeyError("tie_word_embeddings is set but model.embed_tokens.weight is absent")
        fused["lm_head.weight"] = embed

    return fused


@torch.no_grad()
def load_state_dict_into_model(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = True,
) -> None:
    params = dict(model.state_dict(keep_vars=True))

    missing: List[str] = []
    for name, param in params.items():
        source = state_dict.get(name)
        if source is None:
            missing.append(name)
            continue
        if tuple(source.shape) != tuple(param.shape):
            raise ValueError(
                f"shape mismatch for {name}: checkpoint {tuple(source.shape)} vs model {tuple(param.shape)}"
            )
        param.copy_(source.to(device=param.device, dtype=param.dtype))

    unexpected = [name for name in state_dict if name not in params]

    if strict and missing:
        raise RuntimeError(f"missing weights for: {', '.join(sorted(missing))}")
    if missing:
        print(f"[weights] missing {len(missing)} tensors: {', '.join(sorted(missing)[:8])}")
    if unexpected:
        print(f"[weights] ignoring {len(unexpected)} unused checkpoint tensors")


def load_qwen_weights(
    model: torch.nn.Module,
    config: ModelConfig,
    model_id: Optional[str] = None,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> None:
    model_id = model_id or config.model_name
    if verbose:
        print(f"[weights] loading {model_id}")
    raw = load_raw_state_dict(model_id, cache_dir=cache_dir)
    fused = fuse_qwen_state_dict(raw, config.num_hidden_layers, config.tie_word_embeddings)
    load_state_dict_into_model(model, fused, strict=True)
    del raw, fused
    if verbose:
        total = sum(p.numel() for p in model.parameters())
        print(f"[weights] loaded {total / 1e9:.3f}B parameters")
