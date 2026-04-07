"""AutoTokenizer entrypoint (Rust ``tokenizers`` backend; no ``transformers``)."""

from __future__ import annotations

from typing import Any

from .fast_tokenizer import FastTokenizer


class AutoTokenizer:
    """Load a tokenizer the same way as Hugging Face ``AutoTokenizer`` (same assets)."""

    @staticmethod
    def from_pretrained(model_id: str, **kwargs: Any) -> FastTokenizer:
        return FastTokenizer.from_pretrained(model_id, **kwargs)


def from_pretrained(model_id: str, **kwargs: Any) -> FastTokenizer:
    """Alias for :meth:`AutoTokenizer.from_pretrained`."""
    return AutoTokenizer.from_pretrained(model_id, **kwargs)
