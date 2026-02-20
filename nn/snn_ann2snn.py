"""
ANN-to-SNN Conversion utilities.

Converts trained ANN models to equivalent SNNs by:
1. Fusing Conv+BN layers
2. Normalizing weights by activation range
3. Replacing ReLU with IFNode (soft reset)
"""

import numpy as np

from .module import Module
from .parameter import Parameter


class VoltageScaler(Module):
    """Scales voltage/activation after ANN2SNN conversion.

    Applies a per-channel scaling factor to match ANN activation ranges.

    Args:
        scale: Scaling factor (scalar or per-channel array)
    """

    def __init__(self, scale=1.0):
        super().__init__()
        if isinstance(scale, (int, float)):
            self.scale = np.array([scale], dtype=np.float32)
        else:
            self.scale = np.asarray(scale, dtype=np.float32)

    def forward(self, x):
        if x.ndim == 4 and self.scale.ndim == 1 and len(self.scale) > 1:
            return x * self.scale[np.newaxis, :, np.newaxis, np.newaxis]
        return x * self.scale

    def __repr__(self):
        return f"VoltageScaler(scale_shape={self.scale.shape})"


class Converter:
    """ANN-to-SNN converter.

    Converts a trained ANN to an equivalent SNN by:
    1. Fusing BatchNorm into preceding Conv/Linear layers
    2. Normalizing weights based on activation statistics
    3. Replacing ReLU activations with IFNode (integrate-and-fire)

    Args:
        mode: Normalization mode - 'max', '99.9%', or float (manual scale)
        fuse_flag: Whether to fuse Conv+BN layers (default: True)
        T: Number of simulation timesteps (default: 16)

    Example:
        >>> converter = Converter(mode='max', T=16)
        >>> snn_model = converter.convert(ann_model, data_loader)
    """

    def __init__(self, mode="max", fuse_flag=True, T=16):
        self.mode = mode
        self.fuse_flag = fuse_flag
        self.T = T

    def fuse_conv_bn(self, model):
        """Fuse Conv+BatchNorm layers for inference.

        BN fusion: W_new = (gamma/sigma) * W, b_new = (gamma/sigma) * (b - mu) + beta

        Args:
            model: Module with Conv+BN pairs

        Returns:
            Model with fused layers (BN replaced with identity)
        """
        from .conv import Conv2d
        from .normalization import BatchNorm2d

        modules = list(model._modules.items())
        fused_names = set()

        for i in range(len(modules) - 1):
            name_conv, mod_conv = modules[i]
            name_bn, mod_bn = modules[i + 1]

            if isinstance(mod_conv, Conv2d) and isinstance(mod_bn, BatchNorm2d):
                # Fuse BN into Conv
                w = np.asarray(mod_conv.weight, dtype=np.float32)
                if mod_conv.bias is not None:
                    b = np.asarray(mod_conv.bias, dtype=np.float32)
                else:
                    b = np.zeros(mod_conv.out_channels, dtype=np.float32)

                gamma = np.asarray(mod_bn.weight, dtype=np.float32) if mod_bn.affine else np.ones(
                    mod_bn.num_features, dtype=np.float32
                )
                beta = np.asarray(mod_bn.bias, dtype=np.float32) if mod_bn.affine else np.zeros(
                    mod_bn.num_features, dtype=np.float32
                )
                mu = mod_bn.running_mean
                sigma = np.sqrt(mod_bn.running_var + mod_bn.eps)

                # Fuse: W_new = (gamma/sigma) * W
                scale = gamma / sigma
                w_new = w * scale[:, np.newaxis, np.newaxis, np.newaxis]
                b_new = scale * (b - mu) + beta

                # Update conv weights
                mod_conv.weight = Parameter(w_new, requires_grad=True)
                mod_conv.register_parameter("weight", mod_conv.weight)
                if mod_conv.bias is not None:
                    mod_conv.bias = Parameter(b_new, requires_grad=True)
                    mod_conv.register_parameter("bias", mod_conv.bias)
                else:
                    mod_conv.bias = Parameter(b_new, requires_grad=True)
                    mod_conv.register_parameter("bias", mod_conv.bias)

                # Replace BN with identity
                fused_names.add(name_bn)

        # Remove fused BN modules
        for name in fused_names:
            model._modules[name] = _Identity()

        return model

    def _collect_activations(self, model, data_loader, num_batches=10):
        """Collect activation statistics for normalization.

        Args:
            model: The ANN model
            data_loader: DataLoader providing input batches
            num_batches: Number of batches to sample

        Returns:
            Dict mapping module name to activation statistics
        """
        stats = {}

        # Initialize stats for each module
        for name, mod in model._modules.items():
            stats[name] = []

        # Run forward passes
        count = 0
        for batch in data_loader:
            if count >= num_batches:
                break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = np.asarray(x, dtype=np.float32)

            # Forward pass, collecting activations manually
            current = x
            for name, mod in model._modules.items():
                current = mod(current)
                stats[name].append(current.copy())
            count += 1

        # Compute statistics
        act_stats = {}
        for name, activations in stats.items():
            if activations:
                all_acts = np.concatenate([a.flatten() for a in activations])
                act_stats[name] = {
                    "max": float(np.max(np.abs(all_acts))),
                    "p99_9": float(np.percentile(np.abs(all_acts), 99.9)),
                    "mean": float(np.mean(all_acts)),
                }

        return act_stats

    def normalize_model(self, model, data_loader=None, act_stats=None):
        """Normalize model weights based on activation ranges.

        MaxNorm: scale = max(|activation|)
        RobustNorm: scale = 99.9th percentile of |activation|

        Args:
            model: The ANN model
            data_loader: DataLoader for computing activation stats
            act_stats: Pre-computed activation statistics (optional)

        Returns:
            Normalized model
        """
        if act_stats is None and data_loader is not None:
            act_stats = self._collect_activations(model, data_loader)
        elif act_stats is None:
            return model

        for name, mod in model._modules.items():
            if name not in act_stats:
                continue

            if isinstance(self.mode, str):
                if self.mode == "max":
                    scale = act_stats[name]["max"]
                elif self.mode == "99.9%":
                    scale = act_stats[name]["p99_9"]
                else:
                    continue
            elif isinstance(self.mode, (int, float)):
                scale = float(self.mode)
            else:
                continue

            if scale < 1e-8:
                continue

            # Scale weights of the next layer
            if hasattr(mod, "weight") and isinstance(mod.weight, (np.ndarray, Parameter)):
                w = np.asarray(mod.weight, dtype=np.float32)
                mod.weight = Parameter(w / scale, requires_grad=False)
                mod.register_parameter("weight", mod.weight)

        return model

    def replace_relu_with_ifnode(self, model):
        """Replace ReLU activations with IFNode (soft reset).

        Walks the module tree and replaces any ReLU-like module
        with an IFNode using soft reset (v_reset=None).

        Args:
            model: The ANN model

        Returns:
            Model with ReLU replaced by IFNode
        """
        from .snn_neurons import IFNode

        replacements = {}
        for name, mod in model._modules.items():
            mod_class = mod.__class__.__name__
            if mod_class in ("ReLU", "ReLU6"):
                replacements[name] = IFNode(
                    v_threshold=1.0,
                    v_reset=None,  # Soft reset for ANN2SNN
                    step_mode="s",
                )

        for name, new_mod in replacements.items():
            model._modules[name] = new_mod

        # Recurse into sub-modules
        for name, mod in model._modules.items():
            if hasattr(mod, "_modules") and mod._modules:
                self.replace_relu_with_ifnode(mod)

        return model

    def convert(self, model, data_loader=None):
        """Full ANN-to-SNN conversion pipeline.

        1. Fuse Conv+BN (if fuse_flag)
        2. Normalize weights (if data_loader provided)
        3. Replace ReLU with IFNode

        Args:
            model: Trained ANN model
            data_loader: DataLoader for activation statistics (optional)

        Returns:
            Converted SNN model
        """
        model.eval()

        if self.fuse_flag:
            model = self.fuse_conv_bn(model)

        if data_loader is not None:
            model = self.normalize_model(model, data_loader)

        model = self.replace_relu_with_ifnode(model)

        return model


class _Identity(Module):
    """Identity module (replaces fused BatchNorm layers)."""

    def forward(self, x):
        return x

    def __repr__(self):
        return "Identity()"
