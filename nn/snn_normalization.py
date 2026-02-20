"""
SNN-specific normalization layers.

Threshold-dependent and temporal batch normalization variants
designed for spiking neural networks.
"""

import numpy as np

from .module import Module
from .parameter import Parameter


class NeuNorm(Module):
    """Per-neuron normalization across batch and spatial dimensions.

    Normalizes each neuron's activity independently using a learnable
    scaling factor.

    Args:
        in_channels: Number of input channels
        height: Spatial height
        width: Spatial width
        k: Scaling factor (default: 0.9)
    """

    def __init__(self, in_channels, height=None, width=None, k=0.9):
        super().__init__()
        self.in_channels = in_channels
        self.k = k
        self.w = Parameter(
            np.full(in_channels, k, dtype=np.float32), requires_grad=True
        )
        self.register_parameter("w", self.w)

    def forward(self, x):
        """Normalize input per-channel.

        Args:
            x: Input (N, C, H, W) or (N, C)

        Returns:
            Normalized output (same shape)
        """
        if x.ndim == 4:
            w = np.asarray(self.w)[np.newaxis, :, np.newaxis, np.newaxis]
        elif x.ndim == 2:
            w = np.asarray(self.w)[np.newaxis, :]
        else:
            w = np.asarray(self.w)
        return x * w

    def __repr__(self):
        return f"NeuNorm(in_channels={self.in_channels}, k={self.k})"


class ThresholdDependentBatchNorm2d(Module):
    """Threshold-Dependent Batch Normalization for 2D inputs.

    y = (x - mu) / sigma * (v_threshold * alpha)

    where alpha is a learnable per-channel scaling factor
    and v_threshold is the spiking threshold.

    Args:
        num_features: Number of channels
        v_threshold: Spiking threshold voltage
        eps: Numerical stability (default: 1e-5)
        momentum: Running stats momentum (default: 0.1)
        affine: Learnable parameters (default: True)
    """

    def __init__(self, num_features, v_threshold=1.0, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.v_threshold = v_threshold
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if affine:
            self.alpha = Parameter(
                np.ones(num_features, dtype=np.float32), requires_grad=True
            )
            self.bias = Parameter(
                np.zeros(num_features, dtype=np.float32), requires_grad=True
            )
            self.register_parameter("alpha", self.alpha)
            self.register_parameter("bias", self.bias)

        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input (N, C, H, W)

        Returns:
            Normalized output (N, C, H, W)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input, got {x.ndim}D")

        if self.training:
            mean = np.mean(x, axis=(0, 2, 3))
            var = np.var(x, axis=(0, 2, 3))
            self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
            self.running_var = self.momentum * var + (1 - self.momentum) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        mean_exp = mean[np.newaxis, :, np.newaxis, np.newaxis]
        var_exp = var[np.newaxis, :, np.newaxis, np.newaxis]
        x_norm = (x - mean_exp) / np.sqrt(var_exp + self.eps)

        if self.affine:
            alpha = np.asarray(self.alpha)[np.newaxis, :, np.newaxis, np.newaxis]
            bias = np.asarray(self.bias)[np.newaxis, :, np.newaxis, np.newaxis]
            return x_norm * (self.v_threshold * alpha) + bias

        return x_norm * self.v_threshold

    def __repr__(self):
        return (
            f"ThresholdDependentBatchNorm2d("
            f"{self.num_features}, v_threshold={self.v_threshold})"
        )


class ThresholdDependentBatchNorm1d(Module):
    """Threshold-Dependent Batch Normalization for 1D inputs.

    Args:
        num_features: Number of features
        v_threshold: Spiking threshold voltage
        eps: Numerical stability (default: 1e-5)
        momentum: Running stats momentum (default: 0.1)
        affine: Learnable parameters (default: True)
    """

    def __init__(self, num_features, v_threshold=1.0, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.bn2d = ThresholdDependentBatchNorm2d(
            num_features, v_threshold, eps, momentum, affine
        )
        self.num_features = num_features

    def forward(self, x):
        if x.ndim == 2:
            return self.bn2d(x[:, :, np.newaxis, np.newaxis]).squeeze((2, 3))
        elif x.ndim == 3:
            return self.bn2d(x[:, :, np.newaxis, :]).squeeze(2)
        raise ValueError(f"Expected 2D or 3D input, got {x.ndim}D")

    def __repr__(self):
        return f"ThresholdDependentBatchNorm1d({self.num_features})"


class TemporalEffectiveBatchNorm2d(Module):
    """Temporal Effective Batch Normalization for 2D inputs.

    Standard BN with per-timestep learnable coefficients lambda_t.
    Applied at each timestep with a temporal weighting factor.

    Args:
        T: Number of timesteps
        num_features: Number of channels
        eps: Numerical stability (default: 1e-5)
        momentum: Running stats momentum (default: 0.1)
        affine: Learnable parameters (default: True)
    """

    def __init__(self, T, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.T = T
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        # Per-timestep coefficients
        self.lambda_t = Parameter(
            np.ones(T, dtype=np.float32), requires_grad=True
        )
        self.register_parameter("lambda_t", self.lambda_t)

        if affine:
            self.weight = Parameter(
                np.ones(num_features, dtype=np.float32), requires_grad=True
            )
            self.bias = Parameter(
                np.zeros(num_features, dtype=np.float32), requires_grad=True
            )
            self.register_parameter("weight", self.weight)
            self.register_parameter("bias", self.bias)

        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)

    def forward(self, x_seq):
        """Forward pass over temporal sequence.

        Args:
            x_seq: Input (T, N, C, H, W)

        Returns:
            Normalized output (T, N, C, H, W)
        """
        T = x_seq.shape[0]
        if T > self.T:
            raise ValueError(f"Input T={T} > configured T={self.T}")

        outputs = []
        for t in range(T):
            x = x_seq[t]  # (N, C, H, W)

            if self.training:
                mean = np.mean(x, axis=(0, 2, 3))
                var = np.var(x, axis=(0, 2, 3))
            else:
                mean = self.running_mean
                var = self.running_var

            mean_exp = mean[np.newaxis, :, np.newaxis, np.newaxis]
            var_exp = var[np.newaxis, :, np.newaxis, np.newaxis]
            x_norm = (x - mean_exp) / np.sqrt(var_exp + self.eps)

            lam = float(np.asarray(self.lambda_t)[t])
            if self.affine:
                w = np.asarray(self.weight)[np.newaxis, :, np.newaxis, np.newaxis]
                b = np.asarray(self.bias)[np.newaxis, :, np.newaxis, np.newaxis]
                x_out = lam * (x_norm * w + b)
            else:
                x_out = lam * x_norm

            if self.training:
                self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                self.running_var = self.momentum * var + (1 - self.momentum) * self.running_var

            outputs.append(x_out)

        return np.stack(outputs, axis=0).astype(np.float32)

    def __repr__(self):
        return f"TemporalEffectiveBatchNorm2d(T={self.T}, num_features={self.num_features})"


class TemporalEffectiveBatchNorm1d(Module):
    """Temporal Effective Batch Normalization for 1D inputs.

    Args:
        T: Number of timesteps
        num_features: Number of features
        eps: Numerical stability (default: 1e-5)
        momentum: Running stats momentum (default: 0.1)
        affine: Learnable parameters (default: True)
    """

    def __init__(self, T, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.tebn2d = TemporalEffectiveBatchNorm2d(T, num_features, eps, momentum, affine)
        self.num_features = num_features
        self.T = T

    def forward(self, x_seq):
        """Forward: (T, N, C) -> (T, N, C)"""
        # (T, N, C) -> (T, N, C, 1, 1) -> TEBN2d -> (T, N, C)
        x_4d = x_seq[:, :, :, np.newaxis, np.newaxis]
        out_4d = self.tebn2d(x_4d)
        return out_4d.squeeze((3, 4))

    def __repr__(self):
        return f"TemporalEffectiveBatchNorm1d(T={self.T}, num_features={self.num_features})"


class BatchNormThroughTime2d(Module):
    """Standard Batch Normalization applied independently at each timestep.

    For [T, N, C, H, W] input, applies standard BN at each T independently.

    Args:
        num_features: Number of channels
        eps: Numerical stability (default: 1e-5)
        momentum: Running stats momentum (default: 0.1)
        affine: Learnable parameters (default: True)
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if affine:
            self.weight = Parameter(
                np.ones(num_features, dtype=np.float32), requires_grad=True
            )
            self.bias = Parameter(
                np.zeros(num_features, dtype=np.float32), requires_grad=True
            )
            self.register_parameter("weight", self.weight)
            self.register_parameter("bias", self.bias)

        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)

    def forward(self, x_seq):
        """Forward: (T, N, C, H, W) -> (T, N, C, H, W)"""
        T = x_seq.shape[0]
        outputs = []

        for t in range(T):
            x = x_seq[t]
            if self.training:
                mean = np.mean(x, axis=(0, 2, 3))
                var = np.var(x, axis=(0, 2, 3))
                self.running_mean = (
                    self.momentum * mean + (1 - self.momentum) * self.running_mean
                )
                self.running_var = (
                    self.momentum * var + (1 - self.momentum) * self.running_var
                )
            else:
                mean = self.running_mean
                var = self.running_var

            mean_exp = mean[np.newaxis, :, np.newaxis, np.newaxis]
            var_exp = var[np.newaxis, :, np.newaxis, np.newaxis]
            x_norm = (x - mean_exp) / np.sqrt(var_exp + self.eps)

            if self.affine:
                w = np.asarray(self.weight)[np.newaxis, :, np.newaxis, np.newaxis]
                b = np.asarray(self.bias)[np.newaxis, :, np.newaxis, np.newaxis]
                x_norm = x_norm * w + b

            outputs.append(x_norm)

        return np.stack(outputs, axis=0).astype(np.float32)

    def __repr__(self):
        return f"BatchNormThroughTime2d({self.num_features})"


class BatchNormThroughTime1d(Module):
    """Standard BN applied independently at each timestep for 1D inputs.

    Args:
        num_features: Number of features
        eps: Numerical stability (default: 1e-5)
        momentum: Running stats momentum (default: 0.1)
        affine: Learnable parameters (default: True)
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.bntt2d = BatchNormThroughTime2d(num_features, eps, momentum, affine)
        self.num_features = num_features

    def forward(self, x_seq):
        """Forward: (T, N, C) -> (T, N, C)"""
        x_4d = x_seq[:, :, :, np.newaxis, np.newaxis]
        out_4d = self.bntt2d(x_4d)
        return out_4d.squeeze((3, 4))

    def __repr__(self):
        return f"BatchNormThroughTime1d({self.num_features})"
