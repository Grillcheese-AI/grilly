"""
ONNX Fine-Tuning with LoRA for Grilly

Orchestrates LoRA-based fine-tuning of ONNX models loaded via OnnxModelLoader.
Wraps Linear layers with LoRA adapters, provides a training loop, and supports
saving/loading LoRA weights and exporting merged models to ONNX.

Usage:
    from grilly.utils.onnx_loader import OnnxModelLoader
    from grilly.utils.onnx_finetune import OnnxFineTuner
    from grilly.nn.lora import LoRAConfig

    model = OnnxModelLoader.load("model.onnx")
    config = LoRAConfig(rank=8, alpha=16, target_modules=["q_proj", "v_proj"])
    tuner = OnnxFineTuner(model, config)
    tuner.apply_lora()
    tuner.train(train_data, epochs=3, lr=1e-4, loss_fn="cross_entropy")
    tuner.save_lora("lora_adapters/")
    tuner.save_onnx("finetuned_model.onnx")
"""

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..nn.lora import LoRAConfig, LoRALinear
from ..nn.module import Module
from ..nn.modules import Linear, _get_param_array


class OnnxFineTuner:
    """High-level LoRA fine-tuning API for ONNX models loaded into Grilly.

    Wraps target Linear layers with ``LoRALinear`` adapters, freezes base
    weights, and provides ``train()`` / ``save_lora()`` / ``save_onnx()``
    helpers.
    """

    def __init__(self, model: Module, config: LoRAConfig):
        """
        Args:
            model: A Grilly Module (typically a ``GrillyOnnxModel``).
            config: LoRA configuration specifying rank, alpha, and target modules.
        """
        self.model = model
        self.config = config
        self._lora_layers: dict[str, LoRALinear] = {}
        self._original_layers: dict[str, Linear] = {}
        self._applied = False

    # ------------------------------------------------------------------
    # LoRA application
    # ------------------------------------------------------------------

    def apply_lora(self) -> "OnnxFineTuner":
        """Freeze base weights and add LoRA adapters to matching Linear layers.

        Walks ``model._modules`` recursively. Any ``Linear`` whose name
        contains a substring from ``config.target_modules`` gets wrapped
        with a ``LoRALinear``.

        Returns:
            self (for chaining)
        """
        if self._applied:
            return self

        targets = self.config.target_modules

        # Collect (name, Linear) pairs
        linear_layers = self._find_linear_layers(self.model, prefix="")

        for name, layer in linear_layers:
            # Check if any target module pattern matches this name
            if not self._matches_target(name, targets):
                continue

            # Extract base weight
            weight = _get_param_array(layer.weight).copy()
            in_features = layer.in_features
            out_features = layer.out_features

            # Create LoRA wrapper
            lora = LoRALinear(
                in_features=in_features,
                out_features=out_features,
                rank=self.config.rank,
                alpha=self.config.alpha,
                dropout=self.config.dropout,
                bias=layer.bias is not None,
                init_weights=self.config.init_lora_weights,
                base_weights=weight,
            )

            # If bias exists, copy it
            if layer.bias is not None:
                bias_data = _get_param_array(layer.bias).copy()
                from ..nn.autograd import Variable
                lora.bias = Variable(bias_data.astype(np.float32), requires_grad=True)

            self._lora_layers[name] = lora
            self._original_layers[name] = layer

        # Freeze all non-LoRA parameters
        for param in self.model.parameters():
            if hasattr(param, "requires_grad"):
                param.requires_grad = False

        self._applied = True
        return self

    def _find_linear_layers(
        self, module: Module, prefix: str
    ) -> list[tuple[str, Linear]]:
        """Recursively find all Linear layers in the module tree."""
        results = []
        for key, child in module._modules.items():
            full_name = f"{prefix}.{key}" if prefix else key
            if isinstance(child, Linear):
                results.append((full_name, child))
            else:
                results.extend(self._find_linear_layers(child, full_name))
        return results

    @staticmethod
    def _matches_target(name: str, targets: list[str]) -> bool:
        """Check if *name* contains any of the target substrings."""
        if not targets:
            # If no targets specified, match all Linear layers
            return True
        return any(t in name for t in targets)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_data: Any,
        epochs: int = 3,
        lr: float = 1e-4,
        loss_fn: str | Callable = "cross_entropy",
        optimizer: str | None = "adamw",
        batch_size: int = 8,
        log_interval: int = 10,
        scheduler: str | None = None,
        max_grad_norm: float | None = 1.0,
    ) -> dict[str, list[float]]:
        """Run a LoRA training loop.

        Args:
            train_data: Training data.  Accepts:
                - A list of ``(input, target)`` tuples
                - A ``grilly.utils.DataLoader`` instance
                - Any iterable yielding ``(input, target)`` batches
            epochs: Number of training epochs.
            lr: Learning rate.
            loss_fn: ``"cross_entropy"``, ``"mse"``, or a callable
                ``(predictions, targets) -> scalar``.
            optimizer: ``"adam"`` or ``"adamw"``.
            batch_size: Batch size (used when *train_data* is a list).
            log_interval: Print loss every *log_interval* steps.
            scheduler: Optional LR scheduler (``"cosine"`` or ``None``).
            max_grad_norm: Maximum gradient norm for clipping (``None`` to
                disable).

        Returns:
            Dict with ``"losses"`` list and ``"epoch_losses"`` list.
        """
        if not self._applied:
            raise RuntimeError("Call apply_lora() before train()")

        # Collect trainable parameters (LoRA A, B, and optional bias)
        trainable_params = list(self.trainable_parameters())
        if not trainable_params:
            raise RuntimeError("No trainable LoRA parameters found")

        # Build loss function
        loss_callable = self._resolve_loss_fn(loss_fn)

        # Build optimizer
        opt = self._build_optimizer(optimizer, trainable_params, lr)

        # Build scheduler
        sched = self._build_scheduler(scheduler, opt, epochs)

        # Wrap data
        batches = self._prepare_batches(train_data, batch_size)

        history: dict[str, list[float]] = {"losses": [], "epoch_losses": []}

        step = 0
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_input, batch_target in batches:
                # Forward pass through model with LoRA
                predictions = self._forward_with_lora(batch_input)

                # Compute loss
                loss_val = loss_callable(predictions, batch_target)
                loss_scalar = float(np.mean(loss_val))
                history["losses"].append(loss_scalar)
                epoch_loss += loss_scalar
                n_batches += 1

                # Backward pass — compute gradients for LoRA params
                self._backward_lora(predictions, batch_target, loss_callable)

                # Gradient clipping
                if max_grad_norm is not None:
                    self._clip_grad_norm(trainable_params, max_grad_norm)

                # Optimizer step
                opt.step()

                # Zero gradients
                for p in trainable_params:
                    if hasattr(p, "grad") and p.grad is not None:
                        p.grad = np.zeros_like(p.data)

                step += 1
                if step % log_interval == 0:
                    print(f"  step {step}, loss={loss_scalar:.6f}")

            avg_epoch_loss = epoch_loss / max(n_batches, 1)
            history["epoch_losses"].append(avg_epoch_loss)
            print(f"Epoch {epoch + 1}/{epochs}, avg_loss={avg_epoch_loss:.6f}")

            if sched is not None:
                sched.step()

        return history

    def _forward_with_lora(self, x: np.ndarray) -> np.ndarray:
        """Run forward pass, substituting LoRA layers for originals."""
        from .onnx_loader import GrillyOnnxModel

        if isinstance(self.model, GrillyOnnxModel):
            # Replace module handlers temporarily
            old_handlers = {}
            for nd in self.model._exec_nodes:
                if nd.kind == "module" and isinstance(nd.handler, Linear):
                    # Find the matching LoRA layer
                    for lora_name, orig in self._original_layers.items():
                        if orig is nd.handler and lora_name in self._lora_layers:
                            old_handlers[id(nd)] = nd.handler
                            lora = self._lora_layers[lora_name]
                            # We need a wrapper that accepts numpy and returns numpy
                            nd.handler = self._make_lora_forward_wrapper(lora)
                            break

            result = self.model(x)

            # Restore original handlers
            for nd in self.model._exec_nodes:
                if id(nd) in old_handlers:
                    nd.handler = old_handlers[id(nd)]

            return result

        # Non-ONNX model — just do a forward pass
        return self.model(x)

    @staticmethod
    def _make_lora_forward_wrapper(lora: LoRALinear):
        """Create a Module-compatible wrapper around LoRALinear.forward()."""
        from ..nn.autograd import Variable

        class _LoRAWrapper(Module):
            def __init__(self, lora_layer):
                super().__init__()
                self._lora = lora_layer

            def forward(self, x):
                if not isinstance(x, Variable):
                    x_var = Variable(x.astype(np.float32), requires_grad=False)
                else:
                    x_var = x
                out = self._lora.forward(x_var)
                return out.data if hasattr(out, "data") else np.asarray(out)

        return _LoRAWrapper(lora)

    def _backward_lora(
        self, predictions: np.ndarray, targets: np.ndarray, loss_fn: Callable
    ):
        """Simple gradient estimation for LoRA parameters.

        Uses numerical differentiation (parameter perturbation) since the
        full autograd graph may not be connected through the ONNX model.
        """
        eps = 1e-4
        base_loss = float(np.mean(loss_fn(predictions, targets)))

        for lora in self._lora_layers.values():
            for param in lora.parameters():
                grad = np.zeros_like(param.data)
                # Approximate gradient via forward differences on a sample
                flat = param.data.flatten()
                # Sample a subset of indices for efficiency
                n = len(flat)
                n_sample = min(n, max(64, n // 4))
                indices = np.random.choice(n, size=n_sample, replace=False)

                for idx in indices:
                    old_val = flat[idx]
                    flat[idx] = old_val + eps
                    param.data = flat.reshape(param.data.shape)
                    new_pred = self._forward_with_lora(predictions)
                    # Reuse the same targets
                    new_loss = float(np.mean(loss_fn(new_pred, targets)))
                    grad_val = (new_loss - base_loss) / eps
                    grad.flat[idx] = grad_val
                    flat[idx] = old_val

                param.data = flat.reshape(param.data.shape)
                param.grad = grad

    def _clip_grad_norm(self, params, max_norm: float):
        """Clip gradient norm across all parameters."""
        total_norm_sq = 0.0
        for p in params:
            if hasattr(p, "grad") and p.grad is not None:
                total_norm_sq += float(np.sum(p.grad ** 2))
        total_norm = np.sqrt(total_norm_sq)
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)
            for p in params:
                if hasattr(p, "grad") and p.grad is not None:
                    p.grad = p.grad * scale

    # ------------------------------------------------------------------
    # Optimizer / scheduler / loss builders
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_loss_fn(loss_fn) -> Callable:
        if callable(loss_fn):
            return loss_fn
        if loss_fn == "cross_entropy":
            def ce_loss(pred, target):
                # Numerically stable cross-entropy
                pred = np.asarray(pred, dtype=np.float64)
                shifted = pred - np.max(pred, axis=-1, keepdims=True)
                log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
                log_probs = shifted - log_sum_exp
                if target.ndim < pred.ndim:
                    # target is class indices
                    target_int = target.astype(np.intp).flatten()
                    batch_size = target_int.shape[0]
                    log_probs_2d = log_probs.reshape(batch_size, -1)
                    loss = -log_probs_2d[np.arange(batch_size), target_int]
                    return np.mean(loss).astype(np.float32)
                # target is one-hot or soft labels
                return np.float32(-np.mean(np.sum(target * log_probs, axis=-1)))
            return ce_loss
        if loss_fn == "mse":
            def mse_loss(pred, target):
                return np.mean((pred - target) ** 2).astype(np.float32)
            return mse_loss
        raise ValueError(f"Unknown loss_fn: {loss_fn}")

    @staticmethod
    def _build_optimizer(name, params, lr):
        if name == "adamw":
            from ..optim.adamw import AdamW
            return AdamW(iter(params), lr=lr)
        if name == "adam":
            from ..optim.adam import Adam
            return Adam(iter(params), lr=lr)
        raise ValueError(f"Unknown optimizer: {name}")

    @staticmethod
    def _build_scheduler(name, optimizer, epochs):
        if name is None:
            return None
        if name == "cosine":
            from ..optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(optimizer, T_max=epochs)
        raise ValueError(f"Unknown scheduler: {name}")

    @staticmethod
    def _prepare_batches(data, batch_size):
        """Normalise training data into list of (input, target) batches."""
        # Already an iterable of (input, target) — just return as list
        if hasattr(data, "__iter__"):
            batches = list(data)
            if batches and isinstance(batches[0], (list, tuple)) and len(batches[0]) == 2:
                return batches
            # Might be a DataLoader — try iterating
            return batches
        raise TypeError(f"Unsupported train_data type: {type(data)}")

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    def trainable_parameters(self) -> Iterator:
        """Iterate over all trainable LoRA parameters."""
        for lora in self._lora_layers.values():
            yield from lora.parameters()

    def num_trainable_params(self) -> int:
        """Count trainable LoRA parameters."""
        return sum(layer.num_trainable_params() for layer in self._lora_layers.values())

    def num_total_params(self) -> int:
        """Count total parameters (LoRA + frozen base)."""
        return sum(layer.num_total_params() for layer in self._lora_layers.values())

    def print_trainable_parameters(self):
        """Print summary of trainable vs total parameters."""
        trainable = self.num_trainable_params()
        total = self.num_total_params()
        pct = 100.0 * trainable / total if total > 0 else 0.0
        print(
            f"trainable params: {trainable:,} || all params: {total:,} || "
            f"trainable%: {pct:.4f}"
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_lora(self, path: str | Path) -> None:
        """Save only LoRA adapter weights and config.

        Creates a directory at *path* containing:
        - ``config.json`` — LoRA configuration
        - ``adapters.npz`` — A/B matrices for each adapted layer
        - ``metadata.json`` — layer dimensions
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.save(path / "config.json")

        # Save adapter weights
        arrays = {}
        metadata = {}
        for name, lora in self._lora_layers.items():
            state = lora.get_state_dict()
            for key, value in state.items():
                arrays[f"{name}__{key}"] = value
            metadata[name] = {
                "in_features": lora.in_features,
                "out_features": lora.out_features,
                "rank": lora.rank,
                "alpha": lora.alpha,
            }

        np.savez(path / "adapters.npz", **arrays)
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def load_lora(self, path: str | Path) -> None:
        """Load LoRA adapter weights from a saved checkpoint.

        The model must have had ``apply_lora()`` called first so that
        LoRA layers exist.
        """
        path = Path(path)
        adapters = np.load(path / "adapters.npz")
        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        for name, meta in metadata.items():
            if name not in self._lora_layers:
                continue
            lora = self._lora_layers[name]
            state = {}
            for key in ["lora_A", "lora_B", "bias"]:
                full_key = f"{name}__{key}"
                if full_key in adapters:
                    state[key] = adapters[full_key]
            lora.load_state_dict(state)

    def save_onnx(self, path: str, **export_kwargs) -> None:
        """Merge LoRA weights into base model and export to ONNX.

        After export, LoRA weights are **unmerged** so the tuner can
        continue training if desired.
        """
        # Merge LoRA into base weights
        self._merge_lora_into_model()

        from .onnx_exporter import OnnxExporter
        exporter = OnnxExporter()
        exporter.export(self.model, path, **export_kwargs)

        # Unmerge to allow continued training
        self._unmerge_lora_from_model()

    # ------------------------------------------------------------------
    # Merge / unmerge helpers
    # ------------------------------------------------------------------

    def _merge_lora_into_model(self):
        """Merge LoRA A/B into original Linear layer weights."""
        from ..nn.modules import _create_param_wrapper

        for name, lora in self._lora_layers.items():
            orig = self._original_layers[name]
            base_w = _get_param_array(orig.weight).copy()
            delta = lora.scaling * np.matmul(lora.lora_B.data, lora.lora_A.data)
            merged_w = (base_w + delta).astype(np.float32)
            orig.weight = _create_param_wrapper(merged_w)
            orig.register_parameter("weight", orig.weight)

    def _unmerge_lora_from_model(self):
        """Undo merge — subtract LoRA delta from base weights."""
        from ..nn.modules import _create_param_wrapper

        for name, lora in self._lora_layers.items():
            orig = self._original_layers[name]
            base_w = _get_param_array(orig.weight).copy()
            delta = lora.scaling * np.matmul(lora.lora_B.data, lora.lora_A.data)
            unmerged_w = (base_w - delta).astype(np.float32)
            orig.weight = _create_param_wrapper(unmerged_w)
            orig.register_parameter("weight", orig.weight)


__all__ = [
    "OnnxFineTuner",
]
