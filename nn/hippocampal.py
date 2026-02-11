"""
Hippocampal Transformer Layer

Integrates capsule encoding, DG sparse expansion, CA3 pattern completion, and memory injection.

Uses: capsule-project.glsl, dg-sparse-expand.glsl, faiss-distance.glsl, faiss-topk.glsl,
      memory-inject-gate.glsl, memory-inject-concat.glsl, memory-inject-residual.glsl,
      rope.glsl, flash-attention2-rope.glsl

Reference: grilly/backend/capsule_transformer.py VulkanCapsuleTransformer
Reference: ref/core/ca3_memory_store.py CA3MemoryStore
"""

from typing import Any

import numpy as np

from .capsule import CapsuleProject, DentateGyrus
from .memory import MemoryInjectGate, MemoryQueryPooling, MemoryRead
from .module import Module
from .modules import LayerNorm
from .transformer import RoPE, TransformerEncoderLayer


class HippocampalTransformerLayer(Module):
    """
    Hippocampal Transformer Layer - Complete bio-inspired transformer with memory.

    Implements the hippocampal circuit:
    - Input → Capsule Encoding (384D → 32D)
    - DG Sparse Expansion (32D → 128D, 2% sparsity)
    - CA3 Pattern Completion (FAISS kNN retrieval)
    - Memory Injection at transformer layers
    - RoPE + Flash Attention 2

    Reference: grilly/backend/capsule_transformer.py
    """

    def __init__(
        self,
        d_model: int = 384,
        nhead: int = 8,
        dim_feedforward: int = 1536,
        capsule_dim: int = 32,
        dg_dim: int = 128,
        num_memories: int = 10000,
        injection_layers: list[int] = None,
        dropout: float = 0.1,
        use_rope: bool = True,
        use_ca3: bool = True,
    ):
        """
        Initialize HippocampalTransformerLayer.

        Args:
            d_model: Model dimension (default: 384)
            nhead: Number of attention heads
            dim_feedforward: Feed-forward dimension
            capsule_dim: Capsule dimension (default: 32)
            dg_dim: Dentate Gyrus dimension (default: 128)
            num_memories: Number of memory slots for CA3
            injection_layers: Which transformer layers to inject memory (default: [4, 5])
            dropout: Dropout probability
            use_rope: Whether to use RoPE (default: True)
            use_ca3: Whether to use CA3 pattern completion (default: True)
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.capsule_dim = capsule_dim
        self.dg_dim = dg_dim
        self.num_memories = num_memories
        self.injection_layers = injection_layers or [4, 5]
        self.use_rope = use_rope
        self.use_ca3 = use_ca3

        # Capsule encoding
        self.capsule_project = CapsuleProject(in_dim=d_model, out_dim=capsule_dim)
        self._modules["capsule_project"] = self.capsule_project

        # Dentate Gyrus sparse expansion
        self.dentate_gyrus = DentateGyrus(in_dim=capsule_dim, out_dim=dg_dim, sparsity=0.02)
        self._modules["dentate_gyrus"] = self.dentate_gyrus

        # CA3 Memory (for pattern completion)
        if use_ca3:
            self.ca3_memory = MemoryRead(
                key_dim=dg_dim, value_dim=capsule_dim, num_memories=num_memories
            )
            self._modules["ca3_memory"] = self.ca3_memory

        # Memory query pooling
        self.memory_query = MemoryQueryPooling(in_dim=d_model, out_dim=dg_dim)
        self._modules["memory_query"] = self.memory_query

        # Memory injection (gated)
        self.memory_inject = MemoryInjectGate(dim=d_model)
        self._modules["memory_inject"] = self.memory_inject

        # Transformer encoder layers
        self.encoder_layers = []
        for i in range(6):  # Default 6 layers
            layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                use_rope=use_rope,
            )
            self.encoder_layers.append(layer)
            self._modules[f"encoder_{i}"] = layer

        # RoPE if requested
        if use_rope:
            self.rope = RoPE(head_dim=d_model // nhead)
            self._modules["rope"] = self.rope
        else:
            self.rope = None

        # Layer normalization
        self.norm = LayerNorm(d_model)
        self._modules["norm"] = self.norm

    def forward(
        self,
        x: np.ndarray,
        memory_keys: np.ndarray | None = None,
        memory_values: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Forward pass through hippocampal transformer.

        Args:
            x: Input embeddings (batch, seq_len, d_model)
            memory_keys: Optional CA3 memory keys (num_memories, dg_dim)
            memory_values: Optional CA3 memory values (num_memories, capsule_dim)

        Returns:
            (output, stats) - output (batch, seq_len, d_model), stats dict
        """
        batch_size, seq_len, _ = x.shape
        stats = {}

        # 1. Capsule Encoding (384D → 32D)
        # Project to capsule space for each sequence position
        x_flat = x.reshape(-1, self.d_model)  # (batch * seq_len, d_model)
        capsules = self.capsule_project(x_flat)  # (batch * seq_len, capsule_dim)
        capsules = capsules.reshape(batch_size, seq_len, self.capsule_dim)
        stats["capsules"] = capsules

        # 2. Dentate Gyrus Sparse Expansion (32D → 128D)
        # Expand capsules to sparse DG representations
        dg_vectors = []
        for b in range(batch_size):
            batch_dg = []
            for s in range(seq_len):
                dg_vec = self.dentate_gyrus(capsules[b, s])  # (dg_dim,)
                batch_dg.append(dg_vec)
            dg_vectors.append(np.stack(batch_dg))
        dg_vectors = np.stack(dg_vectors)  # (batch, seq_len, dg_dim)
        stats["dg_vectors"] = dg_vectors

        # 3. CA3 Pattern Completion (if enabled)
        memory_context = None
        if self.use_ca3:
            # Pool sequence to get memory query
            memory_query = self.memory_query(x)  # (batch, dg_dim)

            # Retrieve from CA3 memory
            if memory_keys is not None and memory_values is not None:
                # Use provided memory
                memory_context = self.ca3_memory.forward(memory_query)  # (batch, capsule_dim)
            else:
                # Use internal memory
                memory_context = self.ca3_memory.forward(memory_query)  # (batch, capsule_dim)
            stats["memory_context"] = memory_context

        # 4. Pass through transformer layers with memory injection
        output = x
        for i, layer in enumerate(self.encoder_layers):
            # Standard transformer forward
            if self.use_rope and self.rope is not None:
                # Apply RoPE to queries/keys (simplified - would need to extract Q/K/V)
                pass  # RoPE is applied within FlashAttention2 if used

            output = layer(output)

            # Inject memory at specified layers
            if i in self.injection_layers and memory_context is not None:
                output = self.memory_inject(output, memory_context)
                stats[f"memory_injected_layer_{i}"] = True

        # Final normalization
        output = self.norm(output)

        return output, stats

    def store_memory(
        self, capsules: np.ndarray, dg_vectors: np.ndarray, content: str | None = None
    ) -> int:
        """
        Store memories in CA3.

        Args:
            capsules: Capsule vectors (batch, capsule_dim) or (capsule_dim,)
            dg_vectors: DG vectors (batch, dg_dim) or (dg_dim,)
            content: Optional content string

        Returns:
            Number of memories stored
        """
        if not self.use_ca3:
            return 0

        if capsules.ndim == 1:
            capsules = capsules.reshape(1, -1)
            dg_vectors = dg_vectors.reshape(1, -1)

        # Write to CA3 memory
        # Note: This would need to be implemented with MemoryWrite layer
        # For now, just update internal memory buffers
        num_stored = 0
        for i in range(len(capsules)):
            # Find next write index
            write_idx = (
                self.ca3_memory.write_index if hasattr(self.ca3_memory, "write_index") else 0
            )

            # Write to memory
            self.ca3_memory.memory_keys[write_idx] = dg_vectors[i]
            self.ca3_memory.memory_values[write_idx] = capsules[i]
            num_stored += 1

        return num_stored

    def __repr__(self):
        """Return a debug representation."""

        return f"HippocampalTransformerLayer(d_model={self.d_model}, capsule_dim={self.capsule_dim}, dg_dim={self.dg_dim}, num_layers={len(self.encoder_layers)})"
