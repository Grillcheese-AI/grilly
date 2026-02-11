"""
Checkpoint Utilities

Save and load model checkpoints with support for Module objects.
Uses numpy's .npz format for efficient binary storage.
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np


def save_checkpoint(
    model: Any | None = None,
    model_state: dict[str, Any] | None = None,
    optimizer: Any | None = None,
    optimizer_state: dict[str, Any] | None = None,
    epoch: int = 0,
    loss: float | None = None,
    filepath: str = "checkpoint.npz",
    metadata: dict[str, Any] | None = None,
):
    """
    Save model checkpoint.

    Supports both Module objects and state dicts.
    Uses .npz format for efficient binary storage of numpy arrays.

    Args:
        model: Optional Module object (will extract state_dict automatically)
        model_state: Optional model state dict (if model not provided)
        optimizer: Optional optimizer object (will extract state_dict automatically)
        optimizer_state: Optional optimizer state dict (if optimizer not provided)
        epoch: Current epoch
        loss: Optional loss value
        filepath: Path to save checkpoint (.npz or .pth)
        metadata: Optional additional metadata to save
    """
    # Extract state dict from model if provided
    if model is not None:
        if hasattr(model, "state_dict"):
            model_state = model.state_dict()
        elif hasattr(model, "_parameters") or hasattr(model, "_buffers"):
            # Manual state dict extraction
            model_state = {}
            if hasattr(model, "_parameters"):
                for name, param in model._parameters.items():
                    if param is not None:
                        data = param.data if hasattr(param, "data") else param
                        model_state[name] = np.asarray(data, dtype=np.float32)
            if hasattr(model, "_buffers"):
                for name, buffer in model._buffers.items():
                    if buffer is not None:
                        model_state[name] = np.asarray(buffer, dtype=np.float32)
            if hasattr(model, "_modules"):
                for name, module in model._modules.items():
                    if module is not None:
                        module_state = {}
                        if hasattr(module, "_parameters"):
                            for pname, param in module._parameters.items():
                                if param is not None:
                                    data = param.data if hasattr(param, "data") else param
                                    module_state[f"{name}.{pname}"] = np.asarray(
                                        data, dtype=np.float32
                                    )
                        if hasattr(module, "_buffers"):
                            for bname, buffer in module._buffers.items():
                                if buffer is not None:
                                    module_state[f"{name}.{bname}"] = np.asarray(
                                        buffer, dtype=np.float32
                                    )
                        model_state.update(module_state)

    # Extract state dict from optimizer if provided
    if optimizer is not None:
        if hasattr(optimizer, "state_dict"):
            optimizer_state = optimizer.state_dict()
        elif hasattr(optimizer, "state"):
            # Manual optimizer state extraction
            optimizer_state = {}
            for param_id, state in optimizer.state.items():
                optimizer_state[str(param_id)] = {
                    k: np.asarray(v, dtype=np.float32) if isinstance(v, np.ndarray) else v
                    for k, v in state.items()
                }

    # Prepare checkpoint data
    checkpoint_data = {
        "epoch": epoch,
        "loss": loss if loss is not None else np.nan,
    }

    # Add metadata
    if metadata:
        checkpoint_data["metadata"] = metadata

    # Save model state
    if model_state:
        checkpoint_data["model_state"] = model_state

    # Save optimizer state
    if optimizer_state:
        checkpoint_data["optimizer_state"] = optimizer_state

    # Choose format based on file extension
    filepath = Path(filepath)
    if filepath.suffix == ".npz":
        # Flatten model state dict for npz (handle nested dicts)
        def flatten_state_dict(d, prefix=""):
            """Flatten nested state dictionaries to dotted-key mappings."""
            flat = {}
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    flat.update(flatten_state_dict(v, key))
                elif isinstance(v, np.ndarray):
                    flat[key] = v.astype(np.float32)
                else:
                    try:
                        flat[key] = np.asarray(v, dtype=np.float32)
                    except (TypeError, ValueError):
                        # Skip non-numeric values
                        continue
            return flat

        # Save main checkpoint metadata
        metadata_dict = {
            "epoch": np.array([checkpoint_data["epoch"]], dtype=np.int32),
            "loss": np.array([checkpoint_data["loss"]], dtype=np.float32)
            if not np.isnan(checkpoint_data["loss"])
            else np.array([0.0], dtype=np.float32),
        }
        np.savez_compressed(str(filepath), **metadata_dict)

        # Save model state (flattened)
        if model_state:
            flat_model_state = flatten_state_dict(model_state)
            if flat_model_state:
                np.savez_compressed(str(filepath.with_suffix(".model.npz")), **flat_model_state)

        # Save optimizer state as pickle (contains nested dicts)
        if optimizer_state:
            with open(filepath.with_suffix(".optimizer.pkl"), "wb") as f:
                pickle.dump(optimizer_state, f)
    else:
        # Fallback to pickle for .pth files
        with open(filepath, "wb") as f:
            pickle.dump(checkpoint_data, f)

    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: str, model: Any | None = None, optimizer: Any | None = None, strict: bool = True
) -> dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        filepath: Path to checkpoint file
        model: Optional Module object to load state into
        optimizer: Optional optimizer object to load state into
        strict: Whether to strictly enforce that keys match (default: True)

    Returns:
        Checkpoint dictionary with keys: epoch, loss, model_state, optimizer_state
    """
    filepath = Path(filepath)

    # Load based on format
    if filepath.suffix == ".npz":
        # Load main checkpoint
        checkpoint_data = np.load(str(filepath), allow_pickle=True)
        checkpoint = {}

        # Convert numpy scalars to Python types
        if "epoch" in checkpoint_data.files:
            checkpoint["epoch"] = int(checkpoint_data["epoch"][0])
        if "loss" in checkpoint_data.files:
            loss_val = checkpoint_data["loss"][0]
            checkpoint["loss"] = float(loss_val) if not np.isnan(loss_val) else None

        # Load model state (unflatten)
        model_file = filepath.with_suffix(".model.npz")
        if model_file.exists():
            model_data = np.load(str(model_file), allow_pickle=True)

            # Unflatten nested dict structure
            def unflatten_state_dict(flat_dict):
                """Rebuild nested dictionaries from dotted-key mappings."""
                result = {}
                for key, value in flat_dict.items():
                    parts = key.split(".")
                    current = result
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = value
                return result

            flat_dict = {k: model_data[k] for k in model_data.files}
            checkpoint["model_state"] = unflatten_state_dict(flat_dict)

        # Load optimizer state
        optimizer_file = filepath.with_suffix(".optimizer.pkl")
        if optimizer_file.exists():
            with open(optimizer_file, "rb") as f:
                checkpoint["optimizer_state"] = pickle.load(f)
    else:
        # Load from pickle
        with open(filepath, "rb") as f:
            checkpoint = pickle.load(f)

    # Load state into model if provided
    if model is not None and "model_state" in checkpoint:
        if hasattr(model, "load_state_dict"):
            model.load_state_dict(checkpoint["model_state"], strict=strict)
        else:
            # Manual state loading
            model_state = checkpoint["model_state"]
            if hasattr(model, "_parameters"):
                for name, param in model._parameters.items():
                    if name in model_state:
                        data = np.asarray(model_state[name], dtype=np.float32)
                        if hasattr(param, "data"):
                            param.data = data
                        else:
                            model._parameters[name] = data
            if hasattr(model, "_buffers"):
                for name, buffer in model._buffers.items():
                    if name in model_state:
                        model._buffers[name] = np.asarray(model_state[name], dtype=np.float32)

    # Load state into optimizer if provided
    if optimizer is not None and "optimizer_state" in checkpoint:
        if hasattr(optimizer, "load_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        elif hasattr(optimizer, "state"):
            # Manual optimizer state loading
            optimizer.state = checkpoint["optimizer_state"]

    return checkpoint
