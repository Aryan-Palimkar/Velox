from dataclasses import dataclass
from typing import Any, Dict, Optional
from transformers import AutoConfig


@dataclass
class ModelConfig:
    model_name: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float
    rope_scaling: Optional[Dict[str, Any]]
    tie_word_embeddings: bool

    @staticmethod
    def from_hf(model_name: str, cache_dir: Optional[str] = None) -> "ModelConfig":
        hf_config = AutoConfig.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=False,
        )
        cfg = hf_config.to_dict()
        return ModelConfig(
            model_name=model_name,
            hidden_size=int(cfg["hidden_size"]),
            num_hidden_layers=int(cfg["num_hidden_layers"]),
            num_attention_heads=int(cfg["num_attention_heads"]),
            num_key_value_heads=int(cfg.get("num_key_value_heads", cfg["num_attention_heads"])),
            intermediate_size=int(cfg["intermediate_size"]),
            rms_norm_eps=float(cfg.get("rms_norm_eps", cfg.get("layer_norm_eps", 1e-6))),
            vocab_size=int(cfg["vocab_size"]),
            max_position_embeddings=int(cfg.get("max_position_embeddings", cfg.get("seq_length", 4096))),
            rope_theta=float(cfg.get("rope_theta", 10000.0)),
            rope_scaling=cfg.get("rope_scaling"),
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
        )
