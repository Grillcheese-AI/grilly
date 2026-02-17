"""
ONNX LoRA fine-tuning utilities for Grilly.

This module wraps Linear layers in ONNX-loaded Grilly models with LoRA adapters
and provides a training loop with Vulkan-backed LoRA gradients when available.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..nn.lora import LoRAConfig, LoRALinear
from ..nn.module import Module
from ..nn.modules import Linear, _get_param_array

logger = logging.getLogger(__name__)


class OnnxFineTuner:
    """High-level LoRA fine-tuning API for ONNX models loaded into Grilly."""

    def __init__(
        self,
        model: Module,
        config: LoRAConfig,
        gradient_mode: str = "vulkan",
        feedback_alignment: bool = True,
    ):
        """
        Args:
            model: A Grilly Module (typically ``GrillyOnnxModel``).
            config: LoRA configuration.
            gradient_mode: ``"vulkan"`` (default) or ``"finite_diff"``.
            feedback_alignment: Use direct feedback alignment when logits dim
                and adapted layer output dim differ.
        """
        self.model = model
        self.config = config
        self.gradient_mode = gradient_mode
        self.feedback_alignment = feedback_alignment
        self._lora_layers: dict[str, LoRALinear] = {}
        self._original_layers: dict[str, Linear] = {}
        self._forward_cache: dict[str, dict[str, np.ndarray]] = {}
        self._feedback_matrices: dict[str, np.ndarray] = {}
        self._loss_name: str | None = None
        self._compute_backend = None
        self._compute_backend_init = False
        self._applied = False

    # ------------------------------------------------------------------
    # LoRA application
    # ------------------------------------------------------------------

    def apply_lora(self) -> OnnxFineTuner:
        """Freeze base weights and add LoRA adapters to matching Linear layers."""
        if self._applied:
            return self

        targets = self.config.target_modules
        linear_layers = self._find_linear_layers(self.model, prefix="")

        for name, layer in linear_layers:
            if not self._matches_target(name, targets):
                continue

            weight = _get_param_array(layer.weight).copy()
            lora = LoRALinear(
                in_features=layer.in_features,
                out_features=layer.out_features,
                rank=self.config.rank,
                alpha=self.config.alpha,
                dropout=self.config.dropout,
                bias=layer.bias is not None,
                init_weights=self.config.init_lora_weights,
                base_weights=weight,
            )

            if layer.bias is not None:
                from ..nn.autograd import Variable

                bias_data = _get_param_array(layer.bias).copy()
                lora.bias = Variable(bias_data.astype(np.float32), requires_grad=True)

            self._lora_layers[name] = lora
            self._original_layers[name] = layer

        for param in self.model.parameters():
            if hasattr(param, "requires_grad"):
                param.requires_grad = False

        self._applied = True
        return self

    def _find_linear_layers(self, module: Module, prefix: str) -> list[tuple[str, Linear]]:
        """Recursively find all Linear layers in the module tree."""
        results: list[tuple[str, Linear]] = []
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
        """Run LoRA training."""
        if not self._applied:
            raise RuntimeError("Call apply_lora() before train()")

        trainable_params = list(self.trainable_parameters())
        if not trainable_params:
            raise RuntimeError("No trainable LoRA parameters found")

        loss_callable = self._resolve_loss_fn(loss_fn)
        self._loss_name = loss_fn if isinstance(loss_fn, str) else None
        opt = self._build_optimizer(optimizer, trainable_params, lr)
        sched = self._build_scheduler(scheduler, opt, epochs)

        history: dict[str, list[float]] = {"losses": [], "epoch_losses": []}
        step = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_input, batch_target in self._prepare_batches(train_data, batch_size):
                predictions = self._forward_with_lora(batch_input)

                loss_val = loss_callable(predictions, batch_target)
                loss_scalar = float(np.mean(loss_val))
                history["losses"].append(loss_scalar)
                epoch_loss += loss_scalar
                n_batches += 1

                self._backward_lora(batch_input, predictions, batch_target, loss_callable)

                if max_grad_norm is not None:
                    self._clip_grad_norm(trainable_params, max_grad_norm)

                opt.step()

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

    @staticmethod
    def _normalize_model_inputs(x: Any) -> tuple[Any, ...]:
        if isinstance(x, tuple):
            return x
        if isinstance(x, list):
            return tuple(x)
        return (x,)

    @staticmethod
    def _flatten_last_dim(x: Any) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            return arr, arr.reshape(1, arr.shape[0]), ()
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
            return arr, arr, ()
        prefix = arr.shape[:-1]
        return arr, arr.reshape(-1, arr.shape[-1]), prefix

    @staticmethod
    def _restore_last_dim(x_2d: np.ndarray, prefix: tuple[int, ...], out_features: int) -> np.ndarray:
        if not prefix:
            return x_2d.reshape(out_features)
        return x_2d.reshape(*prefix, out_features)

    def _get_compute_backend(self):
        if self.gradient_mode != "vulkan":
            return None
        if self._compute_backend_init:
            return self._compute_backend

        self._compute_backend_init = True
        try:
            from ..backend.compute import VulkanCompute

            self._compute_backend = VulkanCompute()
            logger.info("ONNX LoRA fine-tuner using Vulkan backend")
        except Exception as exc:
            logger.warning("Vulkan LoRA backend unavailable, using CPU LoRA path: %s", exc)
            self._compute_backend = None
        return self._compute_backend

    def _lora_forward_with_cache(self, layer_name: str, lora: LoRALinear, x: Any) -> np.ndarray:
        _, x_2d, prefix = self._flatten_last_dim(x)
        bias_data = _get_param_array(lora.bias) if lora.bias is not None else None

        h = np.matmul(x_2d, lora.lora_A.data.T)
        out_2d = None

        backend = self._get_compute_backend()
        if backend is not None and hasattr(backend, "lora"):
            try:
                out_2d, h = backend.lora.forward_with_intermediate(
                    x_2d,
                    lora.W.data,
                    lora.lora_A.data,
                    lora.lora_B.data,
                    scale=lora.scaling,
                    bias=bias_data,
                )
            except Exception as exc:
                logger.debug("Vulkan LoRA forward failed, falling back to CPU path: %s", exc)
                out_2d = None

        if out_2d is None:
            base = np.matmul(x_2d, lora.W.data.T)
            lora_out = np.matmul(h, lora.lora_B.data.T)
            out_2d = base + lora.scaling * lora_out
            if bias_data is not None:
                out_2d = out_2d + bias_data

        self._forward_cache[layer_name] = {
            "x_2d": np.asarray(x_2d, dtype=np.float32),
            "h": np.asarray(h, dtype=np.float32),
        }
        return self._restore_last_dim(np.asarray(out_2d, dtype=np.float32), prefix, lora.out_features)

    def _forward_with_lora(self, x: Any) -> np.ndarray:
        """Run forward pass, substituting LoRA layers for originals."""
        from .onnx_loader import GrillyOnnxModel

        model_inputs = self._normalize_model_inputs(x)
        self._forward_cache = {}

        if isinstance(self.model, GrillyOnnxModel):
            old_handlers = {}
            for nd in self.model._exec_nodes:
                if nd.kind == "module" and isinstance(nd.handler, Linear):
                    for lora_name, orig in self._original_layers.items():
                        if orig is nd.handler and lora_name in self._lora_layers:
                            old_handlers[id(nd)] = nd.handler
                            nd.handler = self._make_lora_forward_wrapper(
                                lora_name, self._lora_layers[lora_name]
                            )
                            break

            result = self.model(*model_inputs)

            for nd in self.model._exec_nodes:
                if id(nd) in old_handlers:
                    nd.handler = old_handlers[id(nd)]
            return result

        return self.model(*model_inputs)

    def _make_lora_forward_wrapper(self, layer_name: str, lora: LoRALinear):
        """Create a Module-compatible wrapper around LoRA forward."""

        class _LoRAWrapper(Module):
            def __init__(self, fine_tuner: OnnxFineTuner, name: str, lora_layer: LoRALinear):
                super().__init__()
                self._fine_tuner = fine_tuner
                self._name = name
                self._lora = lora_layer

            def forward(self, x):
                return self._fine_tuner._lora_forward_with_cache(self._name, self._lora, x)

        return _LoRAWrapper(self, layer_name, lora)

    def _loss_gradient(self, predictions: Any, targets: Any) -> np.ndarray | None:
        if isinstance(predictions, (tuple, list)):
            if not predictions:
                return None
            predictions = predictions[0]

        pred = np.asarray(predictions, dtype=np.float32)
        target = np.asarray(targets)

        if self._loss_name == "mse":
            grad = (2.0 / max(pred.size, 1)) * (pred - target.astype(np.float32))
            return grad.astype(np.float32)

        if self._loss_name == "cross_entropy":
            logits_2d = pred.reshape(-1, pred.shape[-1]).astype(np.float64)
            shifted = logits_2d - np.max(logits_2d, axis=-1, keepdims=True)
            probs = np.exp(shifted)
            probs /= np.sum(probs, axis=-1, keepdims=True)

            if target.ndim < pred.ndim:
                target_idx = target.astype(np.int64).reshape(-1)
                if target_idx.size == 0:
                    return None
                if target_idx.size != logits_2d.shape[0]:
                    target_idx = np.resize(target_idx, logits_2d.shape[0])
                grad = probs
                grad[np.arange(logits_2d.shape[0]), target_idx] -= 1.0
                grad /= logits_2d.shape[0]
                return grad.astype(np.float32).reshape(pred.shape)

            target_2d = target.reshape(-1, target.shape[-1]).astype(np.float64)
            if target_2d.shape != logits_2d.shape:
                target_2d = np.resize(target_2d, logits_2d.shape)
            grad = (probs - target_2d) / logits_2d.shape[0]
            return grad.astype(np.float32).reshape(pred.shape)

        return None

    @staticmethod
    def _align_rows(matrix: np.ndarray, rows: int) -> np.ndarray:
        if matrix.shape[0] == rows:
            return matrix
        if matrix.shape[0] == 1:
            return np.repeat(matrix, rows, axis=0)
        if rows % matrix.shape[0] == 0:
            return np.repeat(matrix, rows // matrix.shape[0], axis=0)
        if matrix.shape[0] % rows == 0:
            return matrix.reshape(rows, matrix.shape[0] // rows, matrix.shape[1]).mean(axis=1)
        idx = np.linspace(0, matrix.shape[0] - 1, rows).astype(np.int64)
        return matrix[idx]

    def _get_feedback_matrix(self, layer_name: str, in_dim: int, out_dim: int) -> np.ndarray:
        key = f"{layer_name}:{in_dim}->{out_dim}"
        if key not in self._feedback_matrices:
            seed = abs(hash(key)) % (2**32)
            rng = np.random.default_rng(seed)
            mat = rng.standard_normal((in_dim, out_dim), dtype=np.float32)
            mat /= np.sqrt(max(in_dim, 1))
            self._feedback_matrices[key] = mat.astype(np.float32)
        return self._feedback_matrices[key]

    def _prepare_layer_gradients(
        self,
        layer_name: str,
        grad_logits_2d: np.ndarray,
        layer_batch_rows: int,
        layer_out_features: int,
    ) -> np.ndarray | None:
        grad_rows = self._align_rows(grad_logits_2d, layer_batch_rows)
        if grad_rows.shape[1] == layer_out_features:
            return grad_rows.astype(np.float32, copy=False)
        if not self.feedback_alignment:
            return None
        feedback = self._get_feedback_matrix(layer_name, grad_rows.shape[1], layer_out_features)
        projected = np.matmul(grad_rows, feedback)
        return projected.astype(np.float32, copy=False)

    def _compute_lora_grads(
        self,
        grad_output: np.ndarray,
        x_2d: np.ndarray,
        lora: LoRALinear,
        h: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        backend = self._get_compute_backend()
        if backend is not None and hasattr(backend, "lora"):
            try:
                return backend.lora.backward(
                    grad_output=grad_output,
                    x=x_2d,
                    A=lora.lora_A.data,
                    B=lora.lora_B.data,
                    h=h,
                    scale=lora.scaling,
                )
            except Exception as exc:
                logger.debug("Vulkan LoRA backward failed, falling back to CPU path: %s", exc)

        grad_B = lora.scaling * np.matmul(h.T, grad_output).T
        temp = np.matmul(grad_output, lora.lora_B.data)
        grad_A = lora.scaling * np.matmul(temp.T, x_2d)
        return grad_A.astype(np.float32), grad_B.astype(np.float32)

    def _backward_lora(
        self, batch_input: Any, predictions: np.ndarray, targets: np.ndarray, loss_fn: Callable
    ) -> None:
        if self.gradient_mode == "finite_diff":
            self._backward_lora_finite_diff(batch_input, predictions, targets, loss_fn)
            return

        grad_logits = self._loss_gradient(predictions, targets)
        if grad_logits is None:
            self._backward_lora_finite_diff(batch_input, predictions, targets, loss_fn)
            return

        grad_logits = np.asarray(grad_logits, dtype=np.float32)
        grad_logits_2d = (
            grad_logits.reshape(1, -1)
            if grad_logits.ndim == 1
            else grad_logits.reshape(-1, grad_logits.shape[-1])
        )

        updated_layers = 0
        for layer_name, lora in self._lora_layers.items():
            cache = self._forward_cache.get(layer_name)
            if not cache:
                continue

            x_2d = np.asarray(cache["x_2d"], dtype=np.float32)
            h = np.asarray(cache["h"], dtype=np.float32)

            grad_out = self._prepare_layer_gradients(
                layer_name=layer_name,
                grad_logits_2d=grad_logits_2d,
                layer_batch_rows=x_2d.shape[0],
                layer_out_features=lora.out_features,
            )
            if grad_out is None:
                continue

            grad_A, grad_B = self._compute_lora_grads(grad_out, x_2d, lora, h)

            if getattr(lora.lora_A, "grad", None) is None:
                lora.lora_A.grad = grad_A
            else:
                lora.lora_A.grad = lora.lora_A.grad + grad_A

            if getattr(lora.lora_B, "grad", None) is None:
                lora.lora_B.grad = grad_B
            else:
                lora.lora_B.grad = lora.lora_B.grad + grad_B

            if lora.bias is not None:
                bias_grad = np.sum(grad_out, axis=0).astype(np.float32)
                if getattr(lora.bias, "grad", None) is None:
                    lora.bias.grad = bias_grad
                else:
                    lora.bias.grad = lora.bias.grad + bias_grad

            updated_layers += 1

        if updated_layers == 0:
            self._backward_lora_finite_diff(batch_input, predictions, targets, loss_fn)

    def _backward_lora_finite_diff(
        self, batch_input: Any, predictions: np.ndarray, targets: np.ndarray, loss_fn: Callable
    ) -> None:
        """Fallback finite-difference gradient estimation for LoRA parameters."""
        eps = 1e-4
        base_loss = float(np.mean(loss_fn(predictions, targets)))

        for lora in self._lora_layers.values():
            for param in lora.parameters():
                grad = np.zeros_like(param.data)
                flat = param.data.flatten()
                n = len(flat)
                n_sample = min(n, max(64, n // 4))
                indices = np.random.choice(n, size=n_sample, replace=False)

                for idx in indices:
                    old_val = flat[idx]
                    flat[idx] = old_val + eps
                    param.data = flat.reshape(param.data.shape)
                    new_pred = self._forward_with_lora(batch_input)
                    new_loss = float(np.mean(loss_fn(new_pred, targets)))
                    grad.flat[idx] = (new_loss - base_loss) / eps
                    flat[idx] = old_val

                param.data = flat.reshape(param.data.shape)
                param.grad = grad

    def _clip_grad_norm(self, params, max_norm: float) -> None:
        total_norm_sq = 0.0
        for p in params:
            if hasattr(p, "grad") and p.grad is not None:
                total_norm_sq += float(np.sum(p.grad**2))
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
                pred = np.asarray(pred, dtype=np.float64)
                target_arr = np.asarray(target)
                shifted = pred - np.max(pred, axis=-1, keepdims=True)
                log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
                log_probs = shifted - log_sum_exp

                if target_arr.ndim < pred.ndim:
                    target_int = target_arr.astype(np.intp).flatten()
                    if target_int.size != np.prod(pred.shape[:-1]):
                        target_int = np.resize(target_int, int(np.prod(pred.shape[:-1])))
                    log_probs_2d = log_probs.reshape(-1, log_probs.shape[-1])
                    loss = -log_probs_2d[np.arange(log_probs_2d.shape[0]), target_int]
                    return np.mean(loss).astype(np.float32)

                target_2d = target_arr.reshape(-1, target_arr.shape[-1])
                log_probs_2d = log_probs.reshape(-1, log_probs.shape[-1])
                if target_2d.shape != log_probs_2d.shape:
                    target_2d = np.resize(target_2d, log_probs_2d.shape)
                return np.float32(-np.mean(np.sum(target_2d * log_probs_2d, axis=-1)))

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
    def _stack_model_inputs(inputs: list[Any]) -> Any:
        first = inputs[0]
        if isinstance(first, (tuple, list)):
            cols = list(zip(*inputs, strict=False))
            stacked = []
            for col in cols:
                stacked.append(np.stack([np.asarray(v) for v in col], axis=0))
            return tuple(stacked)
        return np.stack([np.asarray(v) for v in inputs], axis=0)

    @staticmethod
    def _stack_targets(targets: list[Any]) -> np.ndarray:
        return np.asarray(targets)

    @staticmethod
    def _prepare_batches(data, batch_size):
        """Normalize training data into (input, target) batches."""
        if isinstance(data, list):
            if not data:
                return
            if not (isinstance(data[0], (list, tuple)) and len(data[0]) == 2):
                raise TypeError("List train_data must contain (input, target) pairs")

            actual_bs = max(int(batch_size), 1)
            for start in range(0, len(data), actual_bs):
                chunk = data[start : start + actual_bs]
                inputs = [item[0] for item in chunk]
                targets = [item[1] for item in chunk]
                yield OnnxFineTuner._stack_model_inputs(inputs), OnnxFineTuner._stack_targets(
                    targets
                )
            return

        if hasattr(data, "__iter__"):
            for item in data:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    yield item[0], item[1]
                else:
                    raise TypeError(
                        f"Iterable train_data must yield (input, target), got {type(item)}"
                    )
            return

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
        print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {pct:.4f}")

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_lora(self, path: str | Path) -> None:
        """Save only LoRA adapter weights and config."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.config.save(path / "config.json")

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
        """Load LoRA adapter weights from a saved checkpoint."""
        path = Path(path)
        adapters = np.load(path / "adapters.npz")
        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        for name, _meta in metadata.items():
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
        """Merge LoRA weights into base model and export to ONNX."""
        self._merge_lora_into_model()

        from .onnx_exporter import OnnxExporter

        exporter = OnnxExporter()
        exporter.export(self.model, path, **export_kwargs)

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
        """Undo merge by subtracting LoRA delta from base weights."""
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

