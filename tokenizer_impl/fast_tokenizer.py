"""Tokenizer implementation using the standalone ``tokenizers`` (Rust) library only.

Matches Hugging Face **Fast** tokenizer outputs for the same serialized tokenizer assets
(``tokenizer.json``, including ``onnx/tokenizer.json`` Hub fallbacks) without importing
``transformers``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Tokenizer
from .loader import load_rust_tokenizer


class FastTokenizer(Tokenizer):
    """Encode/decode via ``tokenizers.Tokenizer`` (Rust backend)."""

    def __init__(self, rust_tokenizer: Any, model_id: str | None = None) -> None:
        self._tok = rust_tokenizer
        self.model_id = model_id

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> FastTokenizer:
        try:
            rs = load_rust_tokenizer(model_id, **kwargs)
        except ImportError as e:
            raise ImportError(
                "FastTokenizer.from_pretrained requires `pip install tokenizers huggingface_hub`"
            ) from e
        return cls(rs, model_id=model_id)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        **kwargs: Any,
    ) -> list[int] | np.ndarray:
        # Match HF PreTrainedTokenizer.encode defaults (no truncation / padding unless kwargs).
        self._tok.no_truncation()
        self._tok.no_padding()
        enc = self._tok.encode(text, add_special_tokens=add_special_tokens)
        return np.asarray(enc.ids, dtype=np.int32)

    def decode(
        self,
        token_ids: list[int] | np.ndarray,
        skip_special_tokens: bool = True,
        **kwargs: Any,
    ) -> str:
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.astype(np.int32).tolist()
        return self._tok.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_encode(
        self,
        texts: list[str],
        padding: bool | str = False,
        truncation: bool | str = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not texts:
            return {"input_ids": [], "attention_mask": []}

        if truncation or max_length is not None:
            ml = max_length if max_length is not None else 512
            self._tok.enable_truncation(ml)
        else:
            self._tok.no_truncation()

        if padding is True or padding == "longest" or padding == "max_length":
            length = None
            if padding == "max_length" and max_length is not None:
                length = max_length
            self._tok.enable_padding(direction="right", length=length)
        else:
            self._tok.no_padding()

        encodings = self._tok.encode_batch(texts, add_special_tokens=add_special_tokens)

        input_ids = [e.ids for e in encodings]
        attention_mask = [e.attention_mask for e in encodings]

        out: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if return_tensors == "np":
            out["input_ids"] = np.asarray(input_ids, dtype=object)
            out["attention_mask"] = np.asarray(attention_mask, dtype=object)
        return out

    def __call__(self, texts: str | list[str], **kwargs: Any) -> dict[str, Any]:
        if isinstance(texts, str):
            self._tok.no_truncation()
            self._tok.no_padding()
            enc = self._tok.encode(
                texts,
                add_special_tokens=kwargs.get("add_special_tokens", True),
            )
            return {
                "input_ids": enc.ids,
                "attention_mask": enc.attention_mask,
            }
        return self.batch_encode(texts, **kwargs)
