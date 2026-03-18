"""
Attention modules — MultiheadAttention, FlashAttention2.
Uses: attention-scores.glsl, attention-output.glsl, flash-attention2.glsl, etc.
"""

import numpy as np

from .module import Module
from ._helpers import _get_param_array


class MultiheadAttention(Module):
    """
    Multi-head attention
    Uses: attention-scores.glsl, attention-output.glsl, attention-concat-heads.glsl
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        """Initialize the instance."""

        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        # Import Linear locally to avoid circular imports at module level
        from .linear import Linear

        # Create projection layers
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

        self._modules["q_proj"] = self.q_proj
        self._modules["k_proj"] = self.k_proj
        self._modules["v_proj"] = self.v_proj
        self._modules["out_proj"] = self.out_proj

        # Initialize cached values for backward pass
        self._cached_query = None
        self._cached_key = None
        self._cached_value = None
        self._cached_mask = None
        self._cached_q = None
        self._cached_k = None
        self._cached_v = None
        self._cached_scores = None
        self._cached_scores_pre_softmax = None
        self._cached_attn_output = None

    def forward(
        self, query: np.ndarray, key: np.ndarray, value: np.ndarray, mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for multi-head attention.

        Args:
            query: Query tensor (batch, seq_len_q, embed_dim)
            key: Key tensor (batch, seq_len_k, embed_dim)
            value: Value tensor (batch, seq_len_k, embed_dim)
            mask: Optional attention mask

        Returns:
            (output, attention_weights)
        """
        # Cache inputs for backward pass
        self._cached_query = query
        self._cached_key = key
        self._cached_value = value
        self._cached_mask = mask

        # Project to Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Cache Q, K, V for backward
        self._cached_q = q
        self._cached_k = k
        self._cached_v = v

        # Reshape for multi-head attention
        batch_size, seq_len_q, _ = q.shape
        _, seq_len_k, _ = k.shape

        # Reshape: (batch, seq_len, embed_dim) -> (batch, seq_len, num_heads, head_dim)
        q_4d = q.reshape(batch_size, seq_len_q, self.num_heads, self.head_dim)
        k_4d = k.reshape(batch_size, seq_len_k, self.num_heads, self.head_dim)
        v_4d = v.reshape(batch_size, seq_len_k, self.num_heads, self.head_dim)

        # Compute attention using backend
        backend = self._get_backend()

        # Reshape for attention computation: (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        q_reshaped = q_4d.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_len_q, head_dim)
        k_reshaped = k_4d.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_len_k, head_dim)
        v_reshaped = v_4d.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_len_k, head_dim)

        # Compute attention scores
        scores = backend.attention.attention_scores(
            q_reshaped, k_reshaped, num_heads=self.num_heads, head_dim=self.head_dim
        )

        # Backend may return scores in different shape - normalize to (batch, num_heads, seq_len_q, seq_len_k)
        if scores.shape == (batch_size, seq_len_q, self.num_heads, seq_len_k):
            # Backend returned (batch, seq_len_q, num_heads, seq_len_k) - transpose to (batch, num_heads, seq_len_q, seq_len_k)
            scores = scores.transpose(0, 2, 1, 3)
        elif scores.shape != (batch_size, self.num_heads, seq_len_q, seq_len_k):
            # Unexpected shape - try to infer
            if (
                scores.ndim == 4
                and scores.size == batch_size * self.num_heads * seq_len_q * seq_len_k
            ):
                scores = scores.reshape(batch_size, self.num_heads, seq_len_q, seq_len_k)
            else:
                # Fallback: compute manually
                scores = np.einsum("bhqd,bhkd->bhqk", q_reshaped, k_reshaped) / np.sqrt(
                    self.head_dim
                )

        # Cache pre-softmax scores for backward
        self._cached_scores_pre_softmax = scores.copy()

        # Apply mask if provided
        if mask is not None:
            scores = backend.attention.attention_mask(scores, mask)

        # Apply softmax (CPU for now - backend softmax expects 3D)
        # scores is (batch, num_heads, seq_len_q, seq_len_k)
        scores_max = scores.max(axis=-1, keepdims=True)
        scores_exp = np.exp(scores - scores_max)
        scores_softmax = scores_exp / scores_exp.sum(axis=-1, keepdims=True)

        # Cache softmax scores for backward
        self._cached_scores = scores_softmax.copy()

        # Compute attention output
        # scores_softmax: (batch, num_heads, seq_len_q, seq_len_k)
        # v_reshaped: (batch, num_heads, seq_len_k, head_dim)
        # Output: (batch, num_heads, seq_len_q, head_dim)
        attn_output = np.einsum("bhqk,bhkd->bhqd", scores_softmax, v_reshaped)

        # Cache attention output for backward (in shape: batch, num_heads, seq_len_q, head_dim)
        self._cached_attn_output = attn_output.copy()

        # Reshape back: (batch, num_heads, seq_len_q, head_dim) -> (batch, seq_len_q, embed_dim)
        attn_output_reshaped = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len_q, self.embed_dim
        )
        attn_weights = scores_softmax

        # Project output
        output = self.out_proj(attn_output_reshaped)

        return output, attn_weights

    def backward(
        self,
        grad_output: np.ndarray,
        query: np.ndarray = None,
        key: np.ndarray = None,
        value: np.ndarray = None,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass for multi-head attention.

        Attention: output = softmax(Q @ K^T / sqrt(d)) @ V
        Then: final_output = out_proj(attention_output)

        Args:
            grad_output: Gradient w.r.t. output (batch, seq_len_q, embed_dim)
            query: Query tensor (optional, uses cached if available)
            key: Key tensor (optional, uses cached if available)
            value: Value tensor (optional, uses cached if available)
            mask: Optional attention mask (optional, uses cached if available)

        Returns:
            (grad_query, grad_key, grad_value)
        """
        # Use cached values if available
        if query is None:
            query = self._cached_query
        if key is None:
            key = self._cached_key
        if value is None:
            value = self._cached_value
        if mask is None:
            mask = self._cached_mask

        if query is None or key is None or value is None:
            raise ValueError(
                "Query, key, and value are required for MultiheadAttention backward pass"
            )

        # Get cached intermediate values
        scores = self._cached_scores  # (batch, num_heads, seq_len_q, seq_len_k)
        q = self._cached_q  # (batch, seq_len_q, embed_dim)
        k = self._cached_k  # (batch, seq_len_k, embed_dim)
        v = self._cached_v  # (batch, seq_len_k, embed_dim)

        batch_size, seq_len_q, _ = query.shape
        _, seq_len_k, _ = key.shape

        # Step 1: Backward through output projection
        # _cached_attn_output is (batch, num_heads, seq_len_q, head_dim)
        # Need to reshape to (batch, seq_len_q, embed_dim) for out_proj backward
        attn_output_for_proj = self._cached_attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len_q, self.embed_dim
        )

        # Flatten batch and seq_len for Linear backward: (batch * seq_len, embed_dim)
        grad_output_flat = grad_output.reshape(-1, self.embed_dim)
        attn_output_flat = attn_output_for_proj.reshape(-1, self.embed_dim)

        if hasattr(self.out_proj, "backward"):
            grad_attn_output_flat = self.out_proj.backward(grad_output_flat, attn_output_flat)
            grad_attn_output = grad_attn_output_flat.reshape(batch_size, seq_len_q, self.embed_dim)
        else:
            # Simplified: assume linear projection
            weight = _get_param_array(self.out_proj.weight)
            grad_attn_output = grad_output @ weight.T

        # Reshape grad_attn_output: (batch, seq_len_q, embed_dim) -> (batch, num_heads, seq_len_q, head_dim)
        grad_attn_output = grad_attn_output.reshape(
            batch_size, seq_len_q, self.num_heads, self.head_dim
        )
        grad_attn_output = grad_attn_output.transpose(
            0, 2, 1, 3
        )  # (batch, num_heads, seq_len_q, head_dim)

        # Reshape V for backward: (batch, seq_len_k, embed_dim) -> (batch, num_heads, seq_len_k, head_dim)
        v_reshaped = v.reshape(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        # Step 2: Backward through attention output: grad_V and grad_scores
        # grad_V = scores^T @ grad_attn_output
        # grad_scores = grad_attn_output @ V^T
        # v_reshaped: (batch, num_heads, seq_len_k, head_dim) -> transpose to (batch, num_heads, head_dim, seq_len_k)
        v_reshaped_T = v_reshaped.transpose(0, 1, 3, 2)  # (batch, num_heads, head_dim, seq_len_k)
        grad_scores = np.matmul(
            grad_attn_output, v_reshaped_T
        )  # (batch, num_heads, seq_len_q, seq_len_k)

        # scores: (batch, num_heads, seq_len_q, seq_len_k) -> transpose to (batch, num_heads, seq_len_k, seq_len_q)
        scores_T = scores.transpose(0, 1, 3, 2)  # (batch, num_heads, seq_len_k, seq_len_q)
        grad_v = np.matmul(scores_T, grad_attn_output)  # (batch, num_heads, seq_len_k, head_dim)

        # Step 3: Backward through softmax
        # Softmax backward: grad_pre_softmax = scores * (grad_scores - sum(grad_scores * scores, dim=-1, keepdims=True))
        grad_scores_weighted = grad_scores * scores
        grad_scores_sum = np.sum(grad_scores_weighted, axis=-1, keepdims=True)
        grad_pre_softmax = scores * (grad_scores - grad_scores_sum)

        # Step 4: Backward through attention scores: grad_Q and grad_K
        # scores = Q @ K^T / sqrt(head_dim)
        # grad_Q = grad_pre_softmax @ K / sqrt(head_dim)
        # grad_K = grad_pre_softmax^T @ Q / sqrt(head_dim)
        scale = 1.0 / np.sqrt(self.head_dim)

        # Reshape Q and K for backward
        q_reshaped = q.reshape(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        k_reshaped = k.reshape(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        # grad_pre_softmax: (batch, num_heads, seq_len_q, seq_len_k)
        # k_reshaped: (batch, num_heads, seq_len_k, head_dim)
        grad_q_reshaped = (
            np.matmul(grad_pre_softmax, k_reshaped) * scale
        )  # (batch, num_heads, seq_len_q, head_dim)

        # grad_pre_softmax^T: (batch, num_heads, seq_len_k, seq_len_q)
        grad_pre_softmax_T = grad_pre_softmax.transpose(
            0, 1, 3, 2
        )  # (batch, num_heads, seq_len_k, seq_len_q)
        grad_k_reshaped = (
            np.matmul(grad_pre_softmax_T, q_reshaped) * scale
        )  # (batch, num_heads, seq_len_k, head_dim)

        # Reshape back: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, embed_dim)
        grad_q = grad_q_reshaped.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len_q, self.embed_dim
        )
        grad_k = grad_k_reshaped.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len_k, self.embed_dim
        )
        grad_v = grad_v.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_k, self.embed_dim)

        # Step 5: Backward through Q, K, V projections
        # Flatten batch and seq_len dimensions for Linear backward
        grad_q_flat = grad_q.reshape(-1, self.embed_dim)
        grad_k_flat = grad_k.reshape(-1, self.embed_dim)
        grad_v_flat = grad_v.reshape(-1, self.embed_dim)
        query_flat = query.reshape(-1, self.embed_dim)
        key_flat = key.reshape(-1, self.embed_dim)
        value_flat = value.reshape(-1, self.embed_dim)

        if hasattr(self.q_proj, "backward"):
            grad_query_flat = self.q_proj.backward(grad_q_flat, query_flat)
            grad_query = grad_query_flat.reshape(batch_size, seq_len_q, self.embed_dim)
        else:
            grad_query = grad_q

        if hasattr(self.k_proj, "backward"):
            grad_key_flat = self.k_proj.backward(grad_k_flat, key_flat)
            grad_key = grad_key_flat.reshape(batch_size, seq_len_k, self.embed_dim)
        else:
            grad_key = grad_k

        if hasattr(self.v_proj, "backward"):
            grad_value_flat = self.v_proj.backward(grad_v_flat, value_flat)
            grad_value = grad_value_flat.reshape(batch_size, seq_len_k, self.embed_dim)
        else:
            grad_value = grad_v

        return grad_query, grad_key, grad_value  # Placeholder

    def __repr__(self):
        """Return a debug representation."""

        return f"MultiheadAttention(embed_dim={self.embed_dim}, num_heads={self.num_heads})"


class FlashAttention2(Module):
    """
    Flash Attention 2 (memory-efficient attention)
    Uses: flash-attention2.glsl, flash-attention2-rope.glsl
    """

    def __init__(self, embed_dim: int, num_heads: int, use_rope: bool = False):
        """Initialize the instance."""

        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_rope = use_rope

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        # Initialize cached values for backward pass
        self._cached_q = None
        self._cached_k = None
        self._cached_v = None
        self._cached_mask = None
        self._cached_output = None

    def forward(
        self, q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Forward pass using Flash Attention 2.

        Args:
            q: Query (batch, seq_len, num_heads, head_dim) or (batch, seq_len, embed_dim)
            k: Key (batch, seq_len, num_heads, head_dim) or (batch, seq_len, embed_dim)
            v: Value (batch, seq_len, num_heads, head_dim) or (batch, seq_len, embed_dim)
            mask: Optional attention mask

        Returns:
            Output tensor (batch, seq_len, num_heads, head_dim) or (batch, seq_len, embed_dim)
        """
        # Cache inputs for backward pass
        self._cached_q = q.copy()
        self._cached_k = k.copy()
        self._cached_v = v.copy()
        self._cached_mask = mask

        # Handle different input shapes
        if q.ndim == 3:
            # (batch, seq_len, embed_dim) -> reshape to (batch, seq_len, num_heads, head_dim)
            batch_size, seq_len, _ = q.shape
            q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        backend = self._get_backend()
        if hasattr(backend, "flash_attention2"):
            try:
                output = backend.flash_attention2(q, k, v, mask=mask, use_rope=self.use_rope)
                self._cached_output = output.copy()
                return output
            except Exception:
                pass  # Fall back to CPU

        # CPU fallback: use standard attention computation
        # This is similar to FlashAttention but without tiling
        batch_size, seq_len, _, _ = q.shape

        # Reshape for attention: (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        q_reshaped = q.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_len, head_dim)
        k_reshaped = k.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_len, head_dim)
        v_reshaped = v.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_len, head_dim)

        # Compute attention scores: Q @ K^T / sqrt(head_dim)
        scale = 1.0 / np.sqrt(self.head_dim)
        scores = np.einsum("bhqd,bhkd->bhqk", q_reshaped, k_reshaped) * scale

        # Apply mask if provided
        if mask is not None:
            if mask.ndim == 2:
                # (batch, seq_len) -> expand to (batch, num_heads, seq_len, seq_len)
                mask_expanded = mask[:, None, :, None]  # (batch, 1, seq_len, 1)
                mask_expanded = np.broadcast_to(mask_expanded, scores.shape)
                scores = np.where(mask_expanded > 0, scores, -1e9)
            else:
                scores = scores + mask

        # Apply softmax
        scores_max = scores.max(axis=-1, keepdims=True)
        scores_exp = np.exp(scores - scores_max)
        scores_softmax = scores_exp / scores_exp.sum(axis=-1, keepdims=True)

        # Compute attention output: scores @ V
        output = np.einsum("bhqk,bhkd->bhqd", scores_softmax, v_reshaped)

        # Reshape back: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim)
        output = output.transpose(0, 2, 1, 3)

        self._cached_output = output.copy()
        return output

    def backward(
        self,
        grad_output: np.ndarray,
        q: np.ndarray = None,
        k: np.ndarray = None,
        v: np.ndarray = None,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass for Flash Attention 2.

        FlashAttention2 uses the same mathematical operations as standard attention,
        so the backward pass is similar to MultiheadAttention backward.

        Args:
            grad_output: Gradient w.r.t. output (batch, seq_len, num_heads, head_dim) or (batch, seq_len, embed_dim)
            q: Query tensor (optional, uses cached if available)
            k: Key tensor (optional, uses cached if available)
            v: Value tensor (optional, uses cached if available)
            mask: Optional attention mask (optional, uses cached if available)

        Returns:
            (grad_q, grad_k, grad_v)
        """
        # Use cached values if available
        if q is None:
            q = self._cached_q
        if k is None:
            k = self._cached_k
        if v is None:
            v = self._cached_v
        if mask is None:
            mask = self._cached_mask

        if q is None or k is None or v is None:
            raise ValueError("Query, key, and value are required for FlashAttention2 backward pass")

        # Handle different input shapes
        if q.ndim == 3:
            # (batch, seq_len, embed_dim) -> reshape to (batch, seq_len, num_heads, head_dim)
            batch_size, seq_len, _ = q.shape
            q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        else:
            batch_size, seq_len, _, _ = q.shape

        # Handle grad_output shape
        if grad_output.ndim == 3:
            # (batch, seq_len, embed_dim) -> reshape to (batch, seq_len, num_heads, head_dim)
            grad_output = grad_output.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Reshape for attention computation: (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        q_reshaped = q.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_len, head_dim)
        k_reshaped = k.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_len, head_dim)
        v_reshaped = v.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_len, head_dim)
        grad_output_reshaped = grad_output.transpose(
            0, 2, 1, 3
        )  # (batch, num_heads, seq_len, head_dim)

        # Recompute attention scores for backward (same as forward)
        scale = 1.0 / np.sqrt(self.head_dim)
        scores = np.einsum("bhqd,bhkd->bhqk", q_reshaped, k_reshaped) * scale

        # Apply mask if provided
        if mask is not None:
            if mask.ndim == 2:
                mask_expanded = mask[:, None, :, None]
                mask_expanded = np.broadcast_to(mask_expanded, scores.shape)
                scores = np.where(mask_expanded > 0, scores, -1e9)
            else:
                scores = scores + mask

        # Apply softmax
        scores_max = scores.max(axis=-1, keepdims=True)
        scores_exp = np.exp(scores - scores_max)
        scores_softmax = scores_exp / scores_exp.sum(axis=-1, keepdims=True)

        # Step 1: Backward through attention output: grad_V and grad_scores
        # grad_V = scores^T @ grad_output
        # grad_scores = grad_output @ V^T
        v_reshaped_T = v_reshaped.transpose(0, 1, 3, 2)  # (batch, num_heads, head_dim, seq_len)
        grad_scores = np.matmul(
            grad_output_reshaped, v_reshaped_T
        )  # (batch, num_heads, seq_len, seq_len)

        scores_T = scores_softmax.transpose(0, 1, 3, 2)  # (batch, num_heads, seq_len, seq_len)
        grad_v = np.matmul(scores_T, grad_output_reshaped)  # (batch, num_heads, seq_len, head_dim)

        # Step 2: Backward through softmax
        grad_scores_weighted = grad_scores * scores_softmax
        grad_scores_sum = np.sum(grad_scores_weighted, axis=-1, keepdims=True)
        grad_pre_softmax = scores_softmax * (grad_scores - grad_scores_sum)

        # Step 3: Backward through attention scores: grad_Q and grad_K
        # scores = Q @ K^T / sqrt(head_dim)
        # grad_Q = grad_pre_softmax @ K / sqrt(head_dim)
        # grad_K = grad_pre_softmax^T @ Q / sqrt(head_dim)
        grad_q_reshaped = (
            np.matmul(grad_pre_softmax, k_reshaped) * scale
        )  # (batch, num_heads, seq_len, head_dim)
        grad_pre_softmax_T = grad_pre_softmax.transpose(
            0, 1, 3, 2
        )  # (batch, num_heads, seq_len, seq_len)
        grad_k_reshaped = (
            np.matmul(grad_pre_softmax_T, q_reshaped) * scale
        )  # (batch, num_heads, seq_len, head_dim)

        # Reshape back: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim)
        grad_q = grad_q_reshaped.transpose(0, 2, 1, 3)  # (batch, seq_len, num_heads, head_dim)
        grad_k = grad_k_reshaped.transpose(0, 2, 1, 3)  # (batch, seq_len, num_heads, head_dim)
        grad_v = grad_v.transpose(0, 2, 1, 3)  # (batch, seq_len, num_heads, head_dim)

        # If original inputs were 3D, reshape back
        if self._cached_q.ndim == 3:
            grad_q = grad_q.reshape(batch_size, seq_len, self.embed_dim)
            grad_k = grad_k.reshape(batch_size, seq_len, self.embed_dim)
            grad_v = grad_v.reshape(batch_size, seq_len, self.embed_dim)

        return grad_q, grad_k, grad_v

    def __repr__(self):
        """Return a debug representation."""

        return f"FlashAttention2(embed_dim={self.embed_dim}, num_heads={self.num_heads}, use_rope={self.use_rope})"


class FNetMixing(Module):
    """FNet mixing layer — replaces attention with parameter-free FFT.

    Applies 2D FFT along (seq, hidden) dimensions and takes the real part.
    Achieves 92-97% of BERT accuracy at 80% faster training.
    For block codes, per-block FFT is equivalent to binding — making
    block-code binding a structured FNet mixer.

    References:
        - FNet: Mixing Tokens with Fourier Transforms (Lee-Thorp et al., 2022)
        - HMM-VSA paper (grillcheese)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply FFT mixing: real(FFT2D(x)) along (seq, hidden) dims.

        Args:
            x: (batch, seq_len, hidden_dim) or (seq_len, hidden_dim)

        Returns:
            Mixed tensor of same shape
        """
        return np.fft.fft2(x, axes=(-2, -1)).real.astype(np.float32)

    def __repr__(self):
        return "FNetMixing()"


class HYLAAttention(Module):
    """Hypernetwork Linear Attention (HYLA) — softmax-free attention.

    From the HYLA framework: attention scores generate weights for a local
    value network with nonlinearity (RMSNorm + ReLU). Eliminates the global
    softmax bottleneck — normalization is per-query, requiring no cross-key
    synchronization.

    Architecture:
        scores = Q @ K^T / sqrt(d)           # standard
        v_weights = scores @ V               # linear combination (no softmax)
        output = ReLU(RMSNorm(v_weights))    # local nonlinearity
        output = out_proj(output)

    This is O(L*d) per head when using kernel approximation, vs O(L^2*d) for softmax.
    For standard computation it's the same FLOPs but avoids the softmax sync barrier,
    enabling better GPU utilization on Vulkan.

    References:
        - Vulkan HDC Transformer paper (grillcheese)
        - HYLA: Hypernetwork Linear Attention (2024)

    Uses: attention-scores.glsl, fnn-linear.glsl, snn-rmsnorm.glsl, activation-relu.glsl
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, eps: float = 1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.eps = eps
        self.scale = self.head_dim ** -0.5

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        from .linear import Linear
        from .normalization_modules import RMSNorm

        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)
        self.value_norm = RMSNorm(self.head_dim, eps=eps)

        self._modules["q_proj"] = self.q_proj
        self._modules["k_proj"] = self.k_proj
        self._modules["v_proj"] = self.v_proj
        self._modules["out_proj"] = self.out_proj
        self._modules["value_norm"] = self.value_norm

    def forward(
        self, query: np.ndarray, key: np.ndarray, value: np.ndarray, mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """HYLA forward pass — linear attention with local nonlinearity.

        Args:
            query: (batch, seq_q, embed_dim)
            key: (batch, seq_k, embed_dim)
            value: (batch, seq_k, embed_dim)
            mask: optional attention mask

        Returns:
            (output, attention_scores) — scores are pre-nonlinearity for analysis
        """
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        batch_size, seq_q, _ = q.shape
        _, seq_k, _ = k.shape

        # Reshape to (batch, heads, seq, head_dim)
        q = q.reshape(batch_size, seq_q, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_k, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_k, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Attention scores (no softmax)
        scores = np.einsum("bhqd,bhkd->bhqk", q, k) * self.scale

        # Apply mask if provided
        if mask is not None:
            scores = np.where(mask, scores, -1e9)

        # HYLA: linear combination of values (no softmax normalization)
        # Instead, normalize per-query by the sum of absolute scores
        score_abs_sum = np.abs(scores).sum(axis=-1, keepdims=True) + self.eps
        normalized_scores = scores / score_abs_sum

        # Value aggregation
        v_agg = np.einsum("bhqk,bhkd->bhqd", normalized_scores, v)

        # Local nonlinearity: RMSNorm + ReLU (per head, per position)
        # RMSNorm operates on last dim (head_dim)
        rms = np.sqrt(np.mean(v_agg ** 2, axis=-1, keepdims=True) + self.eps)
        v_normed = v_agg / rms

        # Apply learned RMSNorm weight if available
        weight = _get_param_array(self.value_norm.weight)
        if weight is not None:
            v_normed = v_normed * weight

        # ReLU nonlinearity
        v_activated = np.maximum(v_normed, 0.0)

        # Reshape back: (batch, heads, seq_q, head_dim) -> (batch, seq_q, embed_dim)
        output = v_activated.transpose(0, 2, 1, 3).reshape(batch_size, seq_q, self.embed_dim)
        output = self.out_proj(output)

        return output, scores

    def __repr__(self):
        return f"HYLAAttention(embed_dim={self.embed_dim}, num_heads={self.num_heads})"
