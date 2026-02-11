"""
WorkingMemory - Internal scratchpad for cognitive operations.

Provides limited-capacity working memory with activation decay and attention.
"""

import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from grilly.experimental.cognitive.capsule import CapsuleEncoder
from grilly.experimental.vsa.ops import HolographicOps


class WorkingMemorySlot(Enum):
    """Types of working memory slots."""

    FOCUS = "focus"  # Current attention target
    CONTEXT = "context"  # Conversation/situational context
    CANDIDATE = "candidate"  # Potential response being evaluated
    RETRIEVED = "retrieved"  # Just retrieved from long-term memory
    GOAL = "goal"  # Current communicative goal


@dataclass
class WorkingMemoryItem:
    """Item in working memory with activation and metadata."""

    vector: np.ndarray
    content: str  # Human-readable
    slot: WorkingMemorySlot
    capsule_vector: np.ndarray | None = None
    activation: float = 1.0
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"
    confidence: float = 1.0


class WorkingMemory:
    """
    Working memory as internal attention.

    Key properties:
    - Limited capacity (7±2 items)
    - Activation decay over time
    - Competition between items
    - Binding of items into chunks

    This is where "thinking" happens - composing and testing
    before committing to output.
    """

    DEFAULT_DIM = 4096
    DEFAULT_CAPACITY = 7
    DEFAULT_DECAY_RATE = 0.1
    DEFAULT_CAPSULE_DIM = 32
    DEFAULT_SEMANTIC_DIMS = 28

    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        capacity: int = DEFAULT_CAPACITY,
        decay_rate: float = DEFAULT_DECAY_RATE,
        capsule_dim: int = DEFAULT_CAPSULE_DIM,
        semantic_dims: int = DEFAULT_SEMANTIC_DIMS,
    ):
        """Initialize the instance."""

        self.dim = dim
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.capsule_dim = capsule_dim
        self.semantic_dims = semantic_dims
        self.capsule_encoder: CapsuleEncoder | None = None

        if capsule_dim > 0:
            self.capsule_encoder = CapsuleEncoder(
                input_dim=dim, capsule_dim=capsule_dim, semantic_dims=semantic_dims
            )

        # The slots
        self.items: list[WorkingMemoryItem] = []

        # Focus of attention (index into items)
        self.focus_idx: int | None = None

        # Binding buffer for chunking
        self.binding_buffer: list[np.ndarray] = []

    def add(
        self,
        vector: np.ndarray,
        content: str,
        slot: WorkingMemorySlot,
        confidence: float = 1.0,
        source: str = "unknown",
        capsule_vector: np.ndarray | None = None,
        cognitive_features: np.ndarray | None = None,
    ) -> int:
        """Add item to working memory."""
        if capsule_vector is None and self.capsule_encoder is not None:
            capsule_vector = self.capsule_encoder.encode_vector(vector, cognitive_features)

        item = WorkingMemoryItem(
            vector=vector,
            capsule_vector=capsule_vector,
            content=content,
            slot=slot,
            activation=1.0,
            confidence=confidence,
            source=source,
        )

        # Check capacity
        if len(self.items) >= self.capacity:
            # Remove least activated item
            min_idx = min(range(len(self.items)), key=lambda i: self.items[i].activation)
            self.items.pop(min_idx)
            if self.focus_idx is not None:
                if min_idx == self.focus_idx:
                    self.focus_idx = None
                elif min_idx < self.focus_idx:
                    self.focus_idx -= 1

        self.items.append(item)
        return len(self.items) - 1

    def attend(self, idx: int):
        """Direct attention to an item (boosts activation)."""
        if 0 <= idx < len(self.items):
            self.focus_idx = idx
            self.items[idx].activation = 1.0

    def decay(self):
        """Apply activation decay to all items."""
        for item in self.items:
            item.activation *= 1 - self.decay_rate

        # Boost focused item
        if self.focus_idx is not None:
            self.items[self.focus_idx].activation = min(
                1.0, self.items[self.focus_idx].activation + 0.2
            )

    def get_by_slot(self, slot: WorkingMemorySlot) -> list[WorkingMemoryItem]:
        """Get all items of a particular type."""
        return [item for item in self.items if item.slot == slot]

    def get_context_vector(self) -> np.ndarray:
        """
        Get weighted sum of all items as context.

        This is what feeds into comprehension/production.
        """
        if not self.items:
            return np.zeros(self.dim, dtype=np.float32)

        weighted = []
        for item in self.items:
            weighted.append(item.vector * item.activation)

        return HolographicOps.bundle(weighted, normalize=True)

    def get_context_capsule(self) -> np.ndarray | None:
        """
        Get weighted capsule context vector if available.
        """
        if self.capsule_encoder is None:
            return None

        capsule_vectors = [
            item.capsule_vector * item.activation
            for item in self.items
            if item.capsule_vector is not None
        ]

        if not capsule_vectors:
            return np.zeros(self.capsule_dim, dtype=np.float32)

        context = np.sum(capsule_vectors, axis=0)
        norm = np.linalg.norm(context)
        if norm > 0:
            context = context / norm

        return context.astype(np.float32)

    def bind_focused(self) -> np.ndarray:
        """
        Bind all items in binding buffer with focus.

        This creates chunks - bound representations of multiple items.
        """
        if self.focus_idx is None or not self.binding_buffer:
            return np.zeros(self.dim, dtype=np.float32)

        result = self.items[self.focus_idx].vector.copy()
        for vec in self.binding_buffer:
            result = HolographicOps.convolve(result, vec)

        self.binding_buffer.clear()
        return result

    def clear_candidates(self):
        """Clear all candidate responses."""
        self.items = [item for item in self.items if item.slot != WorkingMemorySlot.CANDIDATE]
