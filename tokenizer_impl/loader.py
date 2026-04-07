"""Load ``tokenizers`` (Rust) tokenizer files from the Hub or local paths — no ``transformers``."""

from __future__ import annotations

import os
from typing import Any


def load_rust_tokenizer(model_id: str, **kwargs: Any):
    """Return ``tokenizers.Tokenizer`` from a Hub id or a local directory / ``tokenizer.json`` path."""
    from tokenizers import Tokenizer as RsTokenizer

    try:
        from huggingface_hub.errors import EntryNotFoundError
    except ImportError:
        from huggingface_hub.utils import EntryNotFoundError  # type: ignore[no-redef]

    if os.path.isfile(model_id) and model_id.endswith(".json"):
        return RsTokenizer.from_file(model_id)
    if os.path.isdir(model_id):
        for rel in ("tokenizer.json", os.path.join("onnx", "tokenizer.json")):
            p = os.path.join(model_id, rel)
            if os.path.isfile(p):
                return RsTokenizer.from_file(p)
        raise FileNotFoundError(
            f"No tokenizer.json or onnx/tokenizer.json under {model_id!r}",
        )

    token = kwargs.get("token") or kwargs.get("use_auth_token")
    revision = kwargs.get("revision")

    from huggingface_hub import hf_hub_download

    # Root tokenizer.json first; some repos (e.g. google/mt5-small) ship only onnx/tokenizer.json.
    for rel in ("tokenizer.json", "onnx/tokenizer.json"):
        try:
            path = hf_hub_download(
                repo_id=model_id,
                filename=rel,
                token=token,
                revision=revision,
            )
            return RsTokenizer.from_file(path)
        except EntryNotFoundError:
            continue

    try:
        tok = RsTokenizer.from_pretrained(model_id, **kwargs)
        if tok is not None:
            return tok
    except (AttributeError, TypeError, OSError, ValueError, EntryNotFoundError):
        pass

    raise FileNotFoundError(
        f"Could not load Rust tokenizer for {model_id!r} "
        "(tried tokenizer.json, onnx/tokenizer.json, Tokenizer.from_pretrained)",
    )
