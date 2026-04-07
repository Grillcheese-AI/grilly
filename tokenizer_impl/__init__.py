"""
Grilly tokenizers (P0 roadmap).

Uses the standalone Rust ``tokenizers`` library (same ``tokenizer.json`` assets as HF Fast
tokenizers) — **no** ``transformers`` / ``AutoTokenizer`` delegation in this package.
Native GPU merge/BPE kernels will layer behind the same :class:`Tokenizer` interface.

Public API:
    from grilly.tokenizers import AutoTokenizer, Tokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    ids = tok.encode("Hello")
"""

from __future__ import annotations

from .auto import AutoTokenizer, from_pretrained
from .base import Tokenizer as TokenizerBase
from .fast_tokenizer import FastTokenizer
from .gpu import GPU_TOKENIZER_AVAILABLE, numpy_to_input_ids_buffers, wrap_ids_as_vulkan_tensors

Tokenizer = FastTokenizer

__all__ = [
    "AutoTokenizer",
    "FastTokenizer",
    "Tokenizer",
    "TokenizerBase",
    "from_pretrained",
    "GPU_TOKENIZER_AVAILABLE",
    "numpy_to_input_ids_buffers",
    "wrap_ids_as_vulkan_tensors",
]
