"""Batch ONNX embedding module for GPU-accelerated text encoding.

Wraps GrillyOnnxModel for efficient batch inference:
- Tokenizes input text using a simple whitespace tokenizer (or a custom callable)
- Pads sequences to uniform length within each mini-batch
- Runs the ONNX model via GrillyOnnxModel.forward()
- Mean-pools the last hidden state to produce sentence embeddings

Usage:
    embedder = BatchEmbedder("path/to/model.onnx")
    embeddings = embedder.encode(["hello world", "foo bar"], batch_size=512)

    # Custom HuggingFace tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    embedder = BatchEmbedder("model.onnx", tokenizer=tok)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .onnx_loader import GrillyOnnxModel, OnnxModelLoader


# ---------------------------------------------------------------------------
# Simple whitespace tokenizer
# ---------------------------------------------------------------------------


class WhitespaceTokenizer:
    """Minimal word-level tokenizer that assigns integer IDs based on a vocabulary.

    On first use the vocabulary is built from the seen words.  Unknown words
    receive the ``[UNK]`` id (1).  The special ``[PAD]`` token always has id 0.

    This tokenizer is intentionally simple.  For production workloads use a
    HuggingFace tokenizer (pass it as the ``tokenizer`` argument to
    ``BatchEmbedder``).
    """

    PAD_ID = 0
    UNK_ID = 1

    def __init__(self, max_length: int = 128):
        self.max_length = max_length
        self._vocab: dict[str, int] = {"[PAD]": self.PAD_ID, "[UNK]": self.UNK_ID}
        self._next_id = 2

    def _word_id(self, word: str) -> int:
        if word not in self._vocab:
            self._vocab[word] = self._next_id
            self._next_id += 1
        return self._vocab[word]

    def encode(self, text: str) -> list[int]:
        """Tokenize a single string and return a list of token IDs (no padding)."""
        tokens = text.lower().split()
        ids = [self._word_id(w) for w in tokens][: self.max_length]
        return ids if ids else [self.UNK_ID]

    @property
    def pad_token_id(self) -> int:
        return self.PAD_ID


# ---------------------------------------------------------------------------
# HuggingFace tokenizer adapter
# ---------------------------------------------------------------------------


class _HFTokenizerAdapter:
    """Thin adapter so HuggingFace tokenizers expose the same interface."""

    def __init__(self, hf_tokenizer: Any, max_length: int = 128):
        self._tok = hf_tokenizer
        self.max_length = max_length

    def encode(self, text: str) -> list[int]:
        encoded = self._tok.encode(text, add_special_tokens=True, max_length=self.max_length,
                                   truncation=True)
        # encode() can return a list or a tensor-like; normalise to list[int]
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        return list(encoded)

    @property
    def pad_token_id(self) -> int:
        return self._tok.pad_token_id if self._tok.pad_token_id is not None else 0


# ---------------------------------------------------------------------------
# Padding helper
# ---------------------------------------------------------------------------


def _pad_batch(sequences: list[list[int]], pad_id: int) -> np.ndarray:
    """Pad a list of token-id sequences to uniform length.

    Returns an int64 array of shape ``(len(sequences), max_len)``.
    """
    max_len = max(len(s) for s in sequences)
    padded = np.full((len(sequences), max_len), fill_value=pad_id, dtype=np.int64)
    for i, seq in enumerate(sequences):
        padded[i, : len(seq)] = seq
    return padded


# ---------------------------------------------------------------------------
# BatchEmbedder
# ---------------------------------------------------------------------------


class BatchEmbedder:
    """GPU-accelerated batch sentence encoder using an ONNX model.

    Parameters
    ----------
    model_path:
        Path to a ``.onnx`` file compatible with ``OnnxModelLoader``.
    tokenizer:
        An optional custom tokenizer.  Must expose ``encode(text) -> list[int]``
        and a ``pad_token_id`` property.  If *None*, a ``WhitespaceTokenizer``
        is used.  You may also pass a HuggingFace tokenizer directly.
    max_length:
        Maximum token sequence length (used by the built-in tokenizer).
    model:
        Pre-loaded ``GrillyOnnxModel`` instance.  If provided, ``model_path``
        is ignored and no file I/O occurs.  Useful for testing.
    """

    def __init__(
        self,
        model_path: str | None = None,
        *,
        tokenizer: Any = None,
        max_length: int = 128,
        model: GrillyOnnxModel | None = None,
    ):
        if model is not None:
            self._model = model
        elif model_path is not None:
            self._model: GrillyOnnxModel = OnnxModelLoader.load(model_path)
        else:
            raise ValueError("Either model_path or model must be provided.")

        if tokenizer is None:
            self._tokenizer: Any = WhitespaceTokenizer(max_length=max_length)
        elif hasattr(tokenizer, "encode") and hasattr(tokenizer, "pad_token_id"):
            # Assume the tokenizer already matches the required interface
            self._tokenizer = tokenizer
        else:
            # Wrap HuggingFace-style tokenizer
            self._tokenizer = _HFTokenizerAdapter(tokenizer, max_length=max_length)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        texts: list[str],
        batch_size: int = 512,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode a list of strings into sentence embeddings.

        Parameters
        ----------
        texts:
            Input sentences.
        batch_size:
            Number of sentences to process per forward pass.
        show_progress:
            If *True*, print a simple progress indicator to stdout.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(len(texts), embedding_dim)``.
        """
        if not texts:
            raise ValueError("texts must be a non-empty list of strings.")

        all_embeddings: list[np.ndarray] = []
        n = len(texts)
        pad_id = self._tokenizer.pad_token_id

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_texts = texts[start:end]

            if show_progress:
                print(f"Encoding {start}–{end} / {n} …")

            # Tokenize
            token_ids = [self._tokenizer.encode(t) for t in batch_texts]

            # Pad to uniform length within this mini-batch
            input_ids = _pad_batch(token_ids, pad_id)  # (B, L)

            # Forward pass — the ONNX model returns the last hidden state or
            # a pooled output. We always mean-pool over the token dimension.
            output = self._model.forward(input_ids)

            embedding = self._mean_pool(output)  # (B, D)
            all_embeddings.append(embedding)

        return np.concatenate(all_embeddings, axis=0).astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        """Encode a single string and return a 1-D embedding vector.

        Parameters
        ----------
        text:
            The input sentence.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(embedding_dim,)``.
        """
        result = self.encode([text], batch_size=1)
        return result[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_pool(output: np.ndarray) -> np.ndarray:
        """Mean-pool a model output over the token (sequence) dimension.

        Handles three common output shapes:
        - ``(B, L, D)`` — standard last-hidden-state tensor → mean over axis 1.
        - ``(B, D)``   — already pooled (e.g. pooler_output) → returned as-is.
        - ``(L, D)``   — single-sample sequence → mean over axis 0, then add
                         a batch dimension.
        """
        if output is None:
            raise ValueError("Model returned None — check that the ONNX graph is valid.")

        arr = np.asarray(output, dtype=np.float32)

        if arr.ndim == 3:
            # (B, L, D) — mean over token axis
            return arr.mean(axis=1)
        if arr.ndim == 2:
            # Could be (B, D) or (L, D) for a single sample
            # Treat as already-pooled batch output
            return arr
        if arr.ndim == 1:
            # Single embedding vector → wrap in batch dim
            return arr[np.newaxis, :]

        raise ValueError(
            f"Unexpected model output shape {arr.shape}. "
            "Expected (B, L, D), (B, D), or (D,)."
        )

    def __repr__(self) -> str:
        tok_name = type(self._tokenizer).__name__
        return f"BatchEmbedder(model={self._model!r}, tokenizer={tok_name})"


__all__ = [
    "BatchEmbedder",
    "WhitespaceTokenizer",
]
