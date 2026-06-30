from typing import Optional
from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self, model_name: str, cache_dir: Optional[str] = None) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_fast=True,
            trust_remote_code=False,
        )

    def encode(self, text: str, return_tensors: str = "pt"):
        return self._tokenizer(text, return_tensors=return_tensors).input_ids

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    @property
    def bos_token_id(self) -> int:
        return self._tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.eos_token_id
