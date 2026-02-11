"""Tests for utils.checkpoint module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    from grilly.nn import Linear, Parameter
    from grilly.optim import Adam
    from grilly.utils.checkpoint import load_checkpoint, save_checkpoint
except ImportError:
    pytest.skip("grilly not available", allow_module_level=True)


class TestSaveCheckpoint:
    """Tests for save_checkpoint."""

    def test_save_model_state_dict(self):
        """save_checkpoint with model_state dict."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            model_state = {"weight": np.random.randn(4, 4).astype(np.float32)}
            save_checkpoint(model_state=model_state, filepath=path, epoch=0)
            assert Path(path).exists()
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_with_metadata(self):
        """save_checkpoint with metadata."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            save_checkpoint(
                model_state={"w": np.zeros(4)},
                filepath=path,
                epoch=5,
                loss=0.5,
                metadata={"extra": "value"},
            )
            with np.load(path, allow_pickle=True) as data:
                assert "epoch" in data
                assert data["epoch"] == 5
                assert "loss" in data
        finally:
            Path(path).unlink(missing_ok=True)


class TestLoadCheckpoint:
    """Tests for load_checkpoint."""

    def test_load_checkpoint(self):
        """load_checkpoint returns dict with expected keys."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        model_path = Path(path).with_suffix(".model.npz")
        try:
            save_checkpoint(
                model_state={"weight": np.ones((2, 2), dtype=np.float32)},
                filepath=path,
                epoch=1,
            )
            state = load_checkpoint(path)
            assert "epoch" in state
            assert state["epoch"] == 1
            assert "model_state" in state
            assert "weight" in state["model_state"]
        finally:
            Path(path).unlink(missing_ok=True)
            model_path.unlink(missing_ok=True)

    def test_load_nonexistent_raises(self):
        """load_checkpoint raises for nonexistent file."""
        with pytest.raises((FileNotFoundError, OSError)):
            load_checkpoint("/nonexistent/path/checkpoint.npz")
