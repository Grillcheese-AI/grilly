"""Abstract tokenizer interface (P0 GPU tokenizer roadmap)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Tokenizer(ABC):
    """Minimal encode/decode surface aligned with common HF usage."""

    @abstractmethod
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        **kwargs: Any,
    ) -> list[int] | np.ndarray:
        """Return token ids for a single string."""

    @abstractmethod
    def decode(
        self,
        token_ids: list[int] | np.ndarray,
        skip_special_tokens: bool = True,
        **kwargs: Any,
    ) -> str:
        """Decode ids back to text."""

    @abstractmethod
    def batch_encode(
        self,
        texts: list[str],
        padding: bool | str = False,
        truncation: bool | str = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Batch encode; return dict with at least ``input_ids`` (list of lists or ndarray)."""
