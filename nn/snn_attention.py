"""
SNN Attention modules.

TemporalWiseAttention: Squeeze-excite on T dimension.
MultiDimensionalAttention: Time + Channel + Spatial attention (MA-SNN).
SpikingSelfAttention: Spikformer-style QKV with spiking neurons.
"""

import numpy as np

from .module import Module
from .parameter import Parameter


class TemporalWiseAttention(Module):
    """Temporal-Wise Attention (TWA) via squeeze-and-excite on T dimension.

    Learns to weight different timesteps by importance.
    Placed after conv layers, before spiking neurons: Conv -> TWA -> LIF.

    Architecture:
        GlobalAvgPool(spatial) -> FC(C, C//r) -> ReLU -> FC(C//r, T) -> Sigmoid

    Args:
        T: Number of timesteps
        channels: Number of input channels
        reduction: Reduction ratio for FC bottleneck (default: 4)
    """

    def __init__(self, T, channels, reduction=4):
        super().__init__()
        self.T = T
        self.channels = channels
        self.reduction = reduction

        mid = max(channels // reduction, 1)
        self.fc1_weight = Parameter(
            np.random.randn(mid, channels).astype(np.float32) * np.sqrt(2.0 / channels),
            requires_grad=True,
        )
        self.fc1_bias = Parameter(np.zeros(mid, dtype=np.float32), requires_grad=True)
        self.fc2_weight = Parameter(
            np.random.randn(T, mid).astype(np.float32) * np.sqrt(2.0 / mid),
            requires_grad=True,
        )
        self.fc2_bias = Parameter(np.zeros(T, dtype=np.float32), requires_grad=True)

        self.register_parameter("fc1_weight", self.fc1_weight)
        self.register_parameter("fc1_bias", self.fc1_bias)
        self.register_parameter("fc2_weight", self.fc2_weight)
        self.register_parameter("fc2_bias", self.fc2_bias)

    def forward(self, x_seq):
        """Apply temporal attention to sequence.

        Args:
            x_seq: Input (T, N, C, H, W)

        Returns:
            Attention-weighted output (T, N, C, H, W)
        """
        T, N, C, H, W = x_seq.shape

        # Global average pool over spatial dims: (T, N, C)
        x_avg = x_seq.mean(axis=(3, 4))  # (T, N, C)

        # Average over T: (N, C)
        x_avg_t = x_avg.mean(axis=0)  # (N, C)

        # FC1 + ReLU: (N, C) -> (N, mid)
        fc1_w = np.asarray(self.fc1_weight)
        fc1_b = np.asarray(self.fc1_bias)
        h = x_avg_t @ fc1_w.T + fc1_b
        h = np.maximum(h, 0)  # ReLU

        # FC2 + Sigmoid: (N, mid) -> (N, T)
        fc2_w = np.asarray(self.fc2_weight)
        fc2_b = np.asarray(self.fc2_bias)
        attn = h @ fc2_w.T + fc2_b
        attn = 1.0 / (1.0 + np.exp(-attn))  # Sigmoid: (N, T)

        # Broadcast over C, H, W: (T, N, 1, 1, 1)
        attn = attn.T[:, :, np.newaxis, np.newaxis, np.newaxis]  # (T, N, 1, 1, 1)

        return (x_seq * attn).astype(np.float32)

    def __repr__(self):
        return (
            f"TemporalWiseAttention(T={self.T}, channels={self.channels}, "
            f"reduction={self.reduction})"
        )


class MultiDimensionalAttention(Module):
    """Multi-Dimensional Attention for SNNs (MA-SNN paper).

    Combines Time, Channel, and Spatial attention mechanisms.

    Args:
        T: Number of timesteps
        channels: Number of channels
        reduction: Channel attention reduction ratio (default: 4)
    """

    def __init__(self, T, channels, reduction=4):
        super().__init__()
        self.T = T
        self.channels = channels

        # Time attention (similar to TWA)
        self.time_attn = TemporalWiseAttention(T, channels, reduction)
        self._modules["time_attn"] = self.time_attn

        # Channel attention: learnable per-channel scale
        self.channel_scale = Parameter(
            np.ones(channels, dtype=np.float32), requires_grad=True
        )
        self.register_parameter("channel_scale", self.channel_scale)

        # Spatial attention: 1x1 conv equivalent
        self.spatial_weight = Parameter(
            np.random.randn(1, channels).astype(np.float32) * np.sqrt(2.0 / channels),
            requires_grad=True,
        )
        self.register_parameter("spatial_weight", self.spatial_weight)

    def forward(self, x_seq):
        """Apply multi-dimensional attention.

        Args:
            x_seq: Input (T, N, C, H, W)

        Returns:
            Attended output (T, N, C, H, W)
        """
        # Time attention
        x_seq = self.time_attn(x_seq)

        # Channel attention: per-channel scaling
        ch_scale = np.asarray(self.channel_scale)
        ch_scale = ch_scale[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
        x_seq = x_seq * ch_scale

        # Spatial attention: weighted sum across channels -> sigmoid gate
        T, N, C, H, W = x_seq.shape
        sw = np.asarray(self.spatial_weight)  # (1, C)
        # For each timestep: (N, C, H, W) -> spatial_gate (N, 1, H, W)
        outputs = []
        for t in range(T):
            x_t = x_seq[t]  # (N, C, H, W)
            # Reshape for matmul: (N, H*W, C) @ (C, 1) -> (N, H*W, 1)
            x_flat = x_t.reshape(N, C, H * W).transpose(0, 2, 1)  # (N, H*W, C)
            gate = x_flat @ sw.T  # (N, H*W, 1)
            gate = 1.0 / (1.0 + np.exp(-gate))  # Sigmoid
            gate = gate.reshape(N, 1, H, W)
            outputs.append(x_t * gate)

        return np.stack(outputs, axis=0).astype(np.float32)

    def __repr__(self):
        return f"MultiDimensionalAttention(T={self.T}, channels={self.channels})"


class SpikingSelfAttention(Module):
    """Spikformer-style Spiking Self-Attention.

    QKV computed via linear projections with batch norm and spiking neurons.
    Attention: out = (Q * K^T) * V using element-wise operations (no softmax).

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads (default: 8)
        v_threshold: Spike threshold for Q/K/V neurons (default: 1.0)
    """

    def __init__(self, embed_dim, num_heads=8, v_threshold=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} not divisible by num_heads {num_heads}")

        # QKV projection weights
        scale = np.sqrt(2.0 / embed_dim)
        self.q_weight = Parameter(
            np.random.randn(embed_dim, embed_dim).astype(np.float32) * scale,
            requires_grad=True,
        )
        self.k_weight = Parameter(
            np.random.randn(embed_dim, embed_dim).astype(np.float32) * scale,
            requires_grad=True,
        )
        self.v_weight = Parameter(
            np.random.randn(embed_dim, embed_dim).astype(np.float32) * scale,
            requires_grad=True,
        )
        self.out_weight = Parameter(
            np.random.randn(embed_dim, embed_dim).astype(np.float32) * scale,
            requires_grad=True,
        )

        self.register_parameter("q_weight", self.q_weight)
        self.register_parameter("k_weight", self.k_weight)
        self.register_parameter("v_weight", self.v_weight)
        self.register_parameter("out_weight", self.out_weight)

        self.scale = 1.0 / np.sqrt(self.head_dim)

    def forward(self, x):
        """Compute spiking self-attention.

        Args:
            x: Input (N, L, D) where L=sequence length, D=embed_dim

        Returns:
            Output (N, L, D)
        """
        N, L, D = x.shape
        H = self.num_heads
        Dh = self.head_dim

        # QKV projections
        q_w = np.asarray(self.q_weight)
        k_w = np.asarray(self.k_weight)
        v_w = np.asarray(self.v_weight)

        Q = x.reshape(N * L, D) @ q_w.T
        K = x.reshape(N * L, D) @ k_w.T
        V = x.reshape(N * L, D) @ v_w.T

        # Reshape to multi-head: (N, H, L, Dh)
        Q = Q.reshape(N, L, H, Dh).transpose(0, 2, 1, 3)
        K = K.reshape(N, L, H, Dh).transpose(0, 2, 1, 3)
        V = V.reshape(N, L, H, Dh).transpose(0, 2, 1, 3)

        # Spiking attention: QK^T * scale, no softmax
        # (N, H, L, Dh) @ (N, H, Dh, L) -> (N, H, L, L)
        attn = np.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale

        # Multiply by V: (N, H, L, L) @ (N, H, L, Dh) -> (N, H, L, Dh)
        out = np.matmul(attn, V)

        # Reshape back: (N, L, D)
        out = out.transpose(0, 2, 1, 3).reshape(N, L, D)

        # Output projection
        out_w = np.asarray(self.out_weight)
        out = out.reshape(N * L, D) @ out_w.T
        return out.reshape(N, L, D).astype(np.float32)

    def __repr__(self):
        return (
            f"SpikingSelfAttention(embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads})"
        )


class QKAttention(Module):
    """Base QK attention module (QKFormer style).

    Computes attention using only Q and K (no V), with learnable
    temperature scaling.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
    """

    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / np.sqrt(self.head_dim)

        scale = np.sqrt(2.0 / embed_dim)
        self.q_weight = Parameter(
            np.random.randn(embed_dim, embed_dim).astype(np.float32) * scale,
            requires_grad=True,
        )
        self.k_weight = Parameter(
            np.random.randn(embed_dim, embed_dim).astype(np.float32) * scale,
            requires_grad=True,
        )
        self.register_parameter("q_weight", self.q_weight)
        self.register_parameter("k_weight", self.k_weight)

    def forward(self, x):
        """Compute QK attention.

        Args:
            x: Input (N, L, D)

        Returns:
            Attention output (N, L, L)
        """
        N, L, D = x.shape
        H = self.num_heads
        Dh = self.head_dim

        q_w = np.asarray(self.q_weight)
        k_w = np.asarray(self.k_weight)

        Q = x.reshape(N * L, D) @ q_w.T
        K = x.reshape(N * L, D) @ k_w.T

        Q = Q.reshape(N, L, H, Dh).transpose(0, 2, 1, 3)
        K = K.reshape(N, L, H, Dh).transpose(0, 2, 1, 3)

        attn = np.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        # Average over heads: (N, L, L)
        return attn.mean(axis=1).astype(np.float32)

    def __repr__(self):
        return f"QKAttention(embed_dim={self.embed_dim}, num_heads={self.num_heads})"


class TokenQKAttention(QKAttention):
    """Token-wise QK Attention (QKFormer).

    Applies QK attention at the token level.
    """

    def forward(self, x):
        """Token-wise QK attention: attend to individual tokens.

        Args:
            x: Input (N, L, D)

        Returns:
            Attended output (N, L, D)
        """
        attn = super().forward(x)  # (N, L, L)
        # Softmax over keys
        attn_max = attn.max(axis=-1, keepdims=True)
        attn_exp = np.exp(attn - attn_max)
        attn_norm = attn_exp / (attn_exp.sum(axis=-1, keepdims=True) + 1e-8)
        # Weighted sum of values (values = input)
        return np.matmul(attn_norm, x).astype(np.float32)

    def __repr__(self):
        return f"TokenQKAttention(embed_dim={self.embed_dim}, num_heads={self.num_heads})"


class ChannelQKAttention(QKAttention):
    """Channel-wise QK Attention (QKFormer).

    Applies QK attention across channel dimension.
    """

    def forward(self, x):
        """Channel-wise attention: attend across channels.

        Args:
            x: Input (N, L, D)

        Returns:
            Attended output (N, L, D)
        """
        # Transpose to (N, D, L) for channel-wise attention
        x_t = x.transpose(0, 2, 1)  # (N, D, L)
        N, D, L = x_t.shape

        # Channel attention: (N, D, D) scores
        x_r = x_t.reshape(N, D, L)
        attn = np.matmul(x_r, x_r.transpose(0, 2, 1)) * self.scale  # (N, D, D)

        # Softmax
        attn_max = attn.max(axis=-1, keepdims=True)
        attn_exp = np.exp(attn - attn_max)
        attn_norm = attn_exp / (attn_exp.sum(axis=-1, keepdims=True) + 1e-8)

        # Attend: (N, D, D) @ (N, D, L) -> (N, D, L)
        out = np.matmul(attn_norm, x_r)
        return out.transpose(0, 2, 1).astype(np.float32)  # (N, L, D)

    def __repr__(self):
        return f"ChannelQKAttention(embed_dim={self.embed_dim}, num_heads={self.num_heads})"
