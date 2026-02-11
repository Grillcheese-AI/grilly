"""
Transformer Layers

Uses: rope.glsl, flash-attention2-rope.glsl, attention-prosody-modulation.glsl

Reference: grilly/backend/capsule_transformer.py
"""

import logging

import numpy as np

from .module import Module
from .modules import Dropout, LayerNorm, Linear, MultiheadAttention

logger = logging.getLogger(__name__)


class RoPE(Module):
    """
    Rotary Position Embeddings (RoPE).

    Uses: rope.glsl

    Reference: grilly/shaders/rope.glsl
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 512,
        base: float = 10000.0,
        scaling: float = 1.0,
        use_precomputed: bool = True,
    ):
        """
        Initialize RoPE layer.

        Args:
            head_dim: Dimension of each attention head (must be even)
            max_seq_len: Maximum sequence length
            base: Base for frequency computation (default: 10000.0)
            scaling: Scaling factor for extended context (default: 1.0)
            use_precomputed: Whether to use precomputed cos/sin tables
        """
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling = scaling
        self.use_precomputed = use_precomputed

        # Precompute cos/sin tables if requested
        if use_precomputed:
            self.cos_table, self.sin_table = self._precompute_tables()
            self._buffers["cos_table"] = self.cos_table
            self._buffers["sin_table"] = self.sin_table
        else:
            self.cos_table = None
            self.sin_table = None

    def _precompute_tables(self) -> tuple[np.ndarray, np.ndarray]:
        """Precompute cos/sin tables for all positions and dimension pairs"""
        cos_table = np.zeros((self.max_seq_len, self.head_dim // 2), dtype=np.float32)
        sin_table = np.zeros((self.max_seq_len, self.head_dim // 2), dtype=np.float32)

        for pos in range(self.max_seq_len):
            for dim_pair in range(self.head_dim // 2):
                theta = self._compute_theta(pos, dim_pair)
                cos_table[pos, dim_pair] = np.cos(theta)
                sin_table[pos, dim_pair] = np.sin(theta)

        return cos_table, sin_table

    def _compute_theta(self, pos: int, dim_pair: int) -> float:
        """Compute theta for a given position and dimension pair"""
        position = pos / self.scaling
        freq_exp = -2.0 * dim_pair / self.head_dim
        freq = self.base**freq_exp
        return position * freq

    def forward(self, x: np.ndarray, position_ids: np.ndarray = None) -> np.ndarray:
        """
        Forward pass - apply RoPE to Q or K.

        Args:
            x: Input tensor (batch, seq_len, num_heads, head_dim)
            position_ids: Optional position indices (batch, seq_len). If None, uses [0, 1, 2, ...]

        Returns:
            Rotated tensor (same shape)
        """
        backend = self._get_backend()

        # Try GPU shader if available
        if hasattr(backend, "attention") and hasattr(backend.attention, "apply_rope"):
            try:
                return backend.attention.apply_rope(x, position_ids, self.base, self.scaling)
            except Exception:
                pass  # Fall back to CPU

        # CPU fallback
        batch_size, seq_len, num_heads, head_dim = x.shape
        assert head_dim == self.head_dim

        if position_ids is None:
            position_ids = np.arange(seq_len, dtype=np.int32)
            position_ids = np.tile(position_ids, (batch_size, 1))

        # Use ModernBERT's rotate_half approach
        result = x.copy()

        # Compute inv_freq (same as ModernBERT)
        inv_freq = 1.0 / (self.base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

        # Apply RoPE to each position
        for b in range(batch_size):
            for s in range(seq_len):
                pos = position_ids[b, s]
                freqs = pos * inv_freq
                # Expand to full head_dim: [freqs, freqs]
                freqs_expanded = np.repeat(freqs, 2)[:head_dim]
                cos = np.cos(freqs_expanded)
                sin = np.sin(freqs_expanded)

                for h in range(num_heads):
                    q_h = result[b, s, h, :]
                    # rotate_half: [-q[head_dim//2:], q[:head_dim//2]]
                    q_half = head_dim // 2
                    q_rotated = np.concatenate([-q_h[q_half:], q_h[:q_half]])
                    # Apply: (q * cos) + (rotate_half(q) * sin)
                    result[b, s, h, :] = q_h * cos + q_rotated * sin

        return result

    def __repr__(self):
        """Return a debug representation."""

        return f"RoPE(head_dim={self.head_dim}, max_seq_len={self.max_seq_len})"


class ProsodyModulatedAttention(Module):
    """
    Prosody-modulated attention.

    Uses: attention-prosody-modulation.glsl

    Reference: ref/brain/gpu_brain.py attention-prosody-modulation
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        prosody_dim: int = 2,  # valence, arousal
        dropout: float = 0.0,
    ):
        """
        Initialize ProsodyModulatedAttention layer.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            prosody_dim: Dimension of prosody/emotion features (default: 2 for valence/arousal)
            dropout: Dropout probability
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.prosody_dim = prosody_dim
        self.dropout = dropout

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Standard attention
        self.attention = MultiheadAttention(embed_dim, num_heads, dropout)
        self._modules["attention"] = self.attention

        # Prosody modulation weights
        limit = np.sqrt(6.0 / (prosody_dim + embed_dim))
        self.prosody_weight = np.random.uniform(-limit, limit, (embed_dim, prosody_dim)).astype(
            np.float32
        )
        self._parameters["prosody_weight"] = self.prosody_weight

    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        prosody: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward pass with prosody modulation.

        Args:
            query: Query tensor (batch, seq_len, embed_dim)
            key: Key tensor (batch, seq_len, embed_dim)
            value: Value tensor (batch, seq_len, embed_dim)
            prosody: Prosody features (batch, prosody_dim) - e.g., [valence, arousal]
            mask: Optional attention mask

        Returns:
            (output, attention_weights)
        """
        backend = self._get_backend()

        # Expand prosody to (batch, seq_len, prosody_dim) if needed
        if prosody.ndim == 2 and prosody.shape[1] == self.prosody_dim:
            # (batch, prosody_dim) -> (batch, seq_len, prosody_dim)
            batch_size, seq_len = query.shape[:2]
            prosody = np.tile(prosody[:, None, :], (1, seq_len, 1))

        # Reshape prosody_weight to (num_heads, prosody_dim)
        # Original: (embed_dim, prosody_dim) -> split per head: (num_heads, head_dim, prosody_dim)
        # For simplicity, use average per head or reshape
        prosody_weight_per_head = self.prosody_weight.reshape(
            self.num_heads, self.head_dim, self.prosody_dim
        )
        # Average across head_dim to get (num_heads, prosody_dim)
        prosody_weight_reshaped = prosody_weight_per_head.mean(axis=1)

        # Try GPU shader if available
        if hasattr(backend, "attention") and hasattr(backend.attention, "apply_prosody_modulation"):
            try:
                # Get attention scores first
                q_reshaped = query.reshape(
                    query.shape[0], query.shape[1], self.num_heads, self.head_dim
                )
                k_reshaped = key.reshape(key.shape[0], key.shape[1], self.num_heads, self.head_dim)

                # Compute attention scores
                scores = backend.attention.attention_scores(
                    q_reshaped, k_reshaped, self.num_heads, self.head_dim
                )

                # Apply prosody modulation
                backend.attention.apply_prosody_modulation(
                    scores, prosody, prosody_weight_reshaped, prosody_strength=0.3
                )

                # Apply softmax and compute output
                # For now, use standard attention with modulated scores
                # This is a simplified integration - full implementation would integrate with attention_output
                output, attn_weights = self.attention(query, key, value, mask)
                return output, attn_weights
            except Exception as e:
                logger.debug(f"GPU prosody modulation failed: {e}, using CPU fallback")
                pass  # Fall back to CPU

        # CPU fallback with standard attention
        # Compute prosody modulation
        prosody_mod = np.dot(prosody.mean(axis=1), self.prosody_weight.T)  # (batch, embed_dim)
        prosody_mod = prosody_mod[:, None, :]  # (batch, 1, embed_dim)

        # Modulate queries (simplified - full implementation would modulate attention scores)
        query_modulated = query + prosody_mod

        # Standard attention
        output, attn_weights = self.attention(query_modulated, key, value, mask)

        return output, attn_weights

    def __repr__(self):
        """Return a debug representation."""

        return f"ProsodyModulatedAttention(embed_dim={self.embed_dim}, num_heads={self.num_heads})"


class TransformerEncoderLayer(Module):
    """
    Transformer Encoder Layer.

    Combines attention, feed-forward, and normalization.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_rope: bool = False,
        use_prosody: bool = False,
    ):
        """
        Initialize TransformerEncoderLayer.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Feed-forward dimension (default: 4 * d_model)
            dropout: Dropout probability
            activation: Activation function ('gelu', 'relu', 'silu')
            use_rope: Whether to use RoPE (default: False)
            use_prosody: Whether to use prosody-modulated attention (default: False)
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward or (4 * d_model)
        self.dropout = dropout
        self.use_rope = use_rope
        self.use_prosody = use_prosody

        # Self-attention
        if use_prosody:
            self.self_attn = ProsodyModulatedAttention(d_model, nhead, dropout=dropout)
        else:
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self._modules["self_attn"] = self.self_attn

        # RoPE if requested
        if use_rope:
            self.rope = RoPE(d_model // nhead)
            self._modules["rope"] = self.rope
        else:
            self.rope = None

        # Feed-forward
        self.linear1 = Linear(d_model, self.dim_feedforward)
        self.linear2 = Linear(self.dim_feedforward, d_model)
        self._modules["linear1"] = self.linear1
        self._modules["linear2"] = self.linear2

        # Normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self._modules["norm1"] = self.norm1
        self._modules["norm2"] = self.norm2

        # Dropout
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self._modules["dropout1"] = self.dropout1
        self._modules["dropout2"] = self.dropout2

        # Activation
        if activation == "gelu":
            from .modules import GELU

            self.activation = GELU()
        elif activation == "relu":
            from .modules import ReLU

            self.activation = ReLU()
        elif activation == "silu":
            from .modules import SiLU

            self.activation = SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        self._modules["activation"] = self.activation

    def forward(
        self, src: np.ndarray, src_mask: np.ndarray | None = None, prosody: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Forward pass.

        Args:
            src: Source tensor (batch, seq_len, d_model)
            src_mask: Optional source mask
            prosody: Optional prosody features (batch, prosody_dim)

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Self-attention with residual
        src2 = self.norm1(src)

        if self.use_prosody and prosody is not None:
            src2, _ = self.self_attn(src2, src2, src2, prosody, src_mask)
        else:
            src2, _ = self.self_attn(src2, src2, src2, src_mask)

        src = src + self.dropout1(src2)

        # Feed-forward with residual
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src2))))
        src = src + src2

        return src

    def __repr__(self):
        """Return a debug representation."""

        return f"TransformerEncoderLayer(d_model={self.d_model}, nhead={self.nhead})"


class TransformerDecoderLayer(Module):
    """
    Transformer Decoder Layer.

    Similar to encoder but with cross-attention.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_rope: bool = False,
    ):
        """
        Initialize TransformerDecoderLayer.

        Args are the same as TransformerEncoderLayer.
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward or (4 * d_model)
        self.dropout = dropout
        self.use_rope = use_rope

        # Self-attention
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self._modules["self_attn"] = self.self_attn

        # Cross-attention
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self._modules["cross_attn"] = self.cross_attn

        # RoPE if requested
        if use_rope:
            self.rope = RoPE(d_model // nhead)
            self._modules["rope"] = self.rope
        else:
            self.rope = None

        # Feed-forward
        self.linear1 = Linear(d_model, self.dim_feedforward)
        self.linear2 = Linear(self.dim_feedforward, d_model)
        self._modules["linear1"] = self.linear1
        self._modules["linear2"] = self.linear2

        # Normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self._modules["norm1"] = self.norm1
        self._modules["norm2"] = self.norm2
        self._modules["norm3"] = self.norm3

        # Dropout
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self._modules["dropout1"] = self.dropout1
        self._modules["dropout2"] = self.dropout2
        self._modules["dropout3"] = self.dropout3

        # Activation
        if activation == "gelu":
            from .modules import GELU

            self.activation = GELU()
        elif activation == "relu":
            from .modules import ReLU

            self.activation = ReLU()
        elif activation == "silu":
            from .modules import SiLU

            self.activation = SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        self._modules["activation"] = self.activation

    def forward(
        self,
        tgt: np.ndarray,
        memory: np.ndarray,
        tgt_mask: np.ndarray | None = None,
        memory_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Forward pass.

        Args:
            tgt: Target tensor (batch, seq_len, d_model)
            memory: Memory tensor from encoder (batch, seq_len, d_model)
            tgt_mask: Optional target mask
            memory_mask: Optional memory mask

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Self-attention with residual
        tgt2 = self.norm1(tgt)
        tgt2, _ = self.self_attn(tgt2, tgt2, tgt2, tgt_mask)
        tgt = tgt + self.dropout1(tgt2)

        # Cross-attention with residual
        tgt2 = self.norm2(tgt)
        tgt2, _ = self.cross_attn(tgt2, memory, memory, memory_mask)
        tgt = tgt + self.dropout2(tgt2)

        # Feed-forward with residual
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt2))))
        tgt = tgt + tgt2

        return tgt

    def __repr__(self):
        """Return a debug representation."""

        return f"TransformerDecoderLayer(d_model={self.d_model}, nhead={self.nhead})"
