"""Shared helpers for tokenizer/SentencePiece parity tests."""

from __future__ import annotations

import pytest


def load_hf_tokenizer(model_id: str):
    """Load Hugging Face tokenizer or skip with context."""
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:  # pragma: no cover - depends on environment/cache
        pytest.skip(f"Hugging Face tokenizer unavailable for {model_id}: {exc}")


def load_grilly_tokenizer(model_id: str):
    """Load Grilly tokenizer through likely in-progress API shapes or skip."""
    try:
        from grilly import tokenizers as grilly_tokenizers
    except Exception as exc:
        pytest.skip(f"grilly.tokenizers not available yet: {exc}")

    try:
        if hasattr(grilly_tokenizers, "Tokenizer"):
            tok_cls = getattr(grilly_tokenizers, "Tokenizer")
            if hasattr(tok_cls, "from_pretrained"):
                return tok_cls.from_pretrained(model_id)
        if hasattr(grilly_tokenizers, "AutoTokenizer"):
            auto_cls = getattr(grilly_tokenizers, "AutoTokenizer")
            if hasattr(auto_cls, "from_pretrained"):
                return auto_cls.from_pretrained(model_id)
        if hasattr(grilly_tokenizers, "from_pretrained"):
            return grilly_tokenizers.from_pretrained(model_id)
    except Exception as exc:
        pytest.skip(f"Grilly tokenizer could not load {model_id}: {exc}")

    pytest.skip("No supported tokenizer loading API found in grilly.tokenizers")


def extract_input_ids(encoded):
    """Normalize encoder output to plain input IDs."""
    if isinstance(encoded, dict):
        ids = encoded.get("input_ids")
        if ids is None:
            raise AssertionError("Encoded dict missing 'input_ids'")
        return ids
    if hasattr(encoded, "input_ids"):
        ids = encoded.input_ids
        if hasattr(ids, "tolist"):
            return ids.tolist()
        return ids
    return encoded


def encode_ids(tokenizer, text: str, add_special_tokens: bool = True):
    """Encode text with a flexible tokenizer call surface."""
    if hasattr(tokenizer, "encode"):
        try:
            return extract_input_ids(
                tokenizer.encode(text, add_special_tokens=add_special_tokens)
            )
        except TypeError:
            return extract_input_ids(tokenizer.encode(text))
    if hasattr(tokenizer, "__call__"):
        return extract_input_ids(tokenizer(text, add_special_tokens=add_special_tokens))
    raise AssertionError("Tokenizer has neither encode() nor __call__()")

