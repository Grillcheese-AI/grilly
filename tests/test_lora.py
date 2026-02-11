"""
Tests for LoRA (Low-Rank Adaptation) module.

Tests GPU-accelerated LoRA operations including:
- LoRALinear forward pass
- LoRA parameter counting
- LoRA save/load checkpoints
- LoRAModel wrapper
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from grilly.nn.autograd import Variable
from grilly.nn.lora import (
    LoRAAttention,
    LoRAConfig,
    LoRAEmbedding,
    LoRALinear,
    LoRAModel,
    apply_lora_to_linear,
    calculate_lora_params,
)


class TestLoRAConfig:
    """Tests for LoRAConfig"""

    def test_default_config(self):
        """Test default configuration values"""
        config = LoRAConfig()
        assert config.rank == 8
        assert config.alpha == 16.0
        assert config.dropout == 0.0
        assert config.target_modules == ["q_proj", "v_proj"]

    def test_custom_config(self):
        """Test custom configuration"""
        config = LoRAConfig(
            rank=16,
            alpha=32.0,
            dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        assert config.rank == 16
        assert config.alpha == 32.0
        assert config.dropout == 0.1
        assert len(config.target_modules) == 4

    def test_config_serialization(self):
        """Test config to/from dict"""
        config = LoRAConfig(rank=4, alpha=8.0)
        d = config.to_dict()
        loaded = LoRAConfig.from_dict(d)
        assert loaded.rank == config.rank
        assert loaded.alpha == config.alpha

    def test_config_save_load(self):
        """Test config save and load"""
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["q", "v"])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            config.save(path)

            loaded = LoRAConfig.load(path)
            assert loaded.rank == config.rank
            assert loaded.alpha == config.alpha
            assert loaded.target_modules == config.target_modules


class TestLoRALinear:
    """Tests for LoRALinear layer"""

    def test_basic_forward(self):
        """Test basic forward pass"""
        np.random.seed(42)

        lora = LoRALinear(in_features=128, out_features=64, rank=4)
        x = Variable(np.random.randn(4, 128).astype(np.float32))

        y = lora.forward(x)

        assert y.data.shape == (4, 64)
        assert y.data.dtype == np.float32

    def test_with_base_weights(self):
        """Test forward with pre-trained base weights"""
        np.random.seed(42)

        base_weights = np.random.randn(64, 128).astype(np.float32) * 0.01
        lora = LoRALinear(128, 64, rank=4, base_weights=base_weights)

        x = Variable(np.random.randn(4, 128).astype(np.float32))
        lora.forward(x)

        # Verify base weights are used
        np.testing.assert_array_almost_equal(lora.W.data, base_weights)

    def test_trainable_parameters(self):
        """Test parameter counting"""
        lora = LoRALinear(768, 768, rank=8)

        # Expected: rank * (in + out) = 8 * (768 + 768) = 12,288
        expected = 8 * (768 + 768)
        assert lora.num_trainable_params() == expected

        # Total includes base: 768 * 768 = 589,824
        expected_total = 768 * 768 + expected
        assert lora.num_total_params() == expected_total

    def test_parameter_reduction(self):
        """Test that LoRA significantly reduces trainable parameters"""
        lora = LoRALinear(4096, 4096, rank=8)

        full_params = 4096 * 4096  # 16,777,216
        lora_params = lora.num_trainable_params()  # 65,536

        # LoRA should be < 1% of full params
        ratio = lora_params / full_params
        assert ratio < 0.01

    def test_merge_unmerge(self):
        """Test weight merging and unmerging"""
        np.random.seed(42)

        base_weights = np.random.randn(64, 128).astype(np.float32) * 0.01
        lora = LoRALinear(128, 64, rank=4, base_weights=base_weights)

        # Set non-zero LoRA weights
        lora.lora_A.data = np.random.randn(4, 128).astype(np.float32) * 0.01
        lora.lora_B.data = np.random.randn(64, 4).astype(np.float32) * 0.01

        # Save original base
        original_W = lora.W.data.copy()

        # Test input
        x = Variable(np.random.randn(4, 128).astype(np.float32))

        # Forward before merge
        y_before = lora.forward(x).data.copy()

        # Merge
        lora.merge_weights()
        assert lora._merged

        # Forward after merge should give same result
        y_after_merge = lora.forward(x).data
        np.testing.assert_array_almost_equal(y_before, y_after_merge, decimal=5)

        # Weights should be different after merge
        assert not np.allclose(lora.W.data, original_W)

        # Unmerge
        lora.unmerge_weights()
        assert not lora._merged

        # Weights should be restored
        np.testing.assert_array_almost_equal(lora.W.data, original_W, decimal=5)

        # Forward should still match
        y_after_unmerge = lora.forward(x).data
        np.testing.assert_array_almost_equal(y_before, y_after_unmerge, decimal=5)

    def test_enable_disable(self):
        """Test enabling/disabling LoRA"""
        np.random.seed(42)

        base_weights = np.random.randn(64, 128).astype(np.float32) * 0.01
        lora = LoRALinear(128, 64, rank=4, base_weights=base_weights)

        # Set non-zero LoRA weights
        lora.lora_A.data = np.random.randn(4, 128).astype(np.float32) * 0.1
        lora.lora_B.data = np.random.randn(64, 4).astype(np.float32) * 0.1

        x = Variable(np.random.randn(4, 128).astype(np.float32))

        # With LoRA enabled
        y_enabled = lora.forward(x).data.copy()

        # Disable LoRA
        lora.disable_lora()
        y_disabled = lora.forward(x).data.copy()

        # Outputs should be different
        assert not np.allclose(y_enabled, y_disabled)

        # Re-enable
        lora.enable_lora()
        y_reenabled = lora.forward(x).data
        np.testing.assert_array_almost_equal(y_enabled, y_reenabled)

    def test_state_dict(self):
        """Test state dict save/load"""
        np.random.seed(42)

        lora = LoRALinear(128, 64, rank=4)
        lora.lora_A.data = np.random.randn(4, 128).astype(np.float32)
        lora.lora_B.data = np.random.randn(64, 4).astype(np.float32)

        state = lora.get_state_dict()

        # Create new layer and load
        lora2 = LoRALinear(128, 64, rank=4)
        lora2.load_state_dict(state)

        np.testing.assert_array_equal(lora.lora_A.data, lora2.lora_A.data)
        np.testing.assert_array_equal(lora.lora_B.data, lora2.lora_B.data)

    @pytest.mark.gpu
    def test_gpu_backward(self):
        """Test GPU backward pass for LoRA"""
        np.random.seed(42)

        lora = LoRALinear(in_features=128, out_features=64, rank=4)
        x = Variable(np.random.randn(4, 128).astype(np.float32))

        y = lora.forward(x)
        loss = y.sum()

        # Backward pass
        loss.backward(use_gpu=True)

        # Check gradients exist for LoRA params
        assert lora.lora_A.grad is not None
        assert lora.lora_B.grad is not None

        # Base weights should not have gradients
        assert lora.W.grad is None


class TestLoRAModel:
    """Tests for LoRAModel wrapper"""

    def test_model_creation(self):
        """Test creating LoRA model"""
        config = LoRAConfig(rank=8, alpha=16)
        model = LoRAModel(config)

        assert len(model.lora_layers) == 0
        assert model.config.rank == 8

    def test_add_layers(self):
        """Test adding LoRA layers"""
        config = LoRAConfig(rank=4, alpha=8)
        model = LoRAModel(config)

        model.add_lora_layer("layer.0.q_proj", 768, 768)
        model.add_lora_layer("layer.0.v_proj", 768, 768)
        model.add_lora_layer("layer.1.q_proj", 768, 768)

        assert len(model.lora_layers) == 3
        assert "layer.0.q_proj" in model.lora_layers

    def test_parameter_iteration(self):
        """Test iterating over parameters"""
        config = LoRAConfig(rank=4, alpha=8)
        model = LoRAModel(config)

        model.add_lora_layer("layer.0.q_proj", 256, 256)
        model.add_lora_layer("layer.0.v_proj", 256, 256)

        params = list(model.parameters())
        # Each layer has 2 params (A and B)
        assert len(params) == 4

    def test_checkpoint_save_load(self):
        """Test saving and loading checkpoints"""
        np.random.seed(42)

        config = LoRAConfig(rank=4, alpha=8, target_modules=["q_proj"])
        model = LoRAModel(config)

        model.add_lora_layer("layer.0.q_proj", 256, 256)
        model.add_lora_layer("layer.1.q_proj", 256, 256)

        # Set random weights
        for layer in model.lora_layers.values():
            layer.lora_A.data = np.random.randn(*layer.lora_A.data.shape).astype(np.float32)
            layer.lora_B.data = np.random.randn(*layer.lora_B.data.shape).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "checkpoint"
            model.save_checkpoint(save_path)

            # Load back
            loaded = LoRAModel.load_checkpoint(save_path)

            # Verify
            assert len(loaded.lora_layers) == len(model.lora_layers)
            assert loaded.config.rank == config.rank

            for name in model.lora_layers:
                orig = model.lora_layers[name]
                load = loaded.lora_layers[name]
                np.testing.assert_array_almost_equal(orig.lora_A.data, load.lora_A.data, decimal=5)
                np.testing.assert_array_almost_equal(orig.lora_B.data, load.lora_B.data, decimal=5)

    def test_merge_all(self):
        """Test merging all LoRA weights"""
        config = LoRAConfig(rank=4, alpha=8)
        model = LoRAModel(config)

        model.add_lora_layer("layer.0.q_proj", 256, 256)
        model.add_lora_layer("layer.0.v_proj", 256, 256)

        # Set non-zero weights
        for layer in model.lora_layers.values():
            layer.lora_A.data = np.random.randn(*layer.lora_A.data.shape).astype(np.float32)
            layer.lora_B.data = np.random.randn(*layer.lora_B.data.shape).astype(np.float32)

        model.merge_weights()

        for layer in model.lora_layers.values():
            assert layer._merged

    def test_print_trainable_params(self, capsys):
        """Test parameter summary printing"""
        config = LoRAConfig(rank=8, alpha=16)
        model = LoRAModel(config)

        model.add_lora_layer("layer.0.q_proj", 768, 768)
        model.add_lora_layer("layer.0.v_proj", 768, 768)

        model.print_trainable_parameters()

        captured = capsys.readouterr()
        assert "trainable params" in captured.out
        assert "trainable%" in captured.out


class TestLoRAAttention:
    """Tests for LoRAAttention"""

    def test_attention_creation(self):
        """Test creating LoRA attention"""
        attn = LoRAAttention(embed_dim=512, num_heads=8, rank=4, alpha=8, apply_lora_to=["q", "v"])

        assert attn.q_proj is not None
        assert attn.k_proj is None  # Not in apply_lora_to
        assert attn.v_proj is not None
        assert attn.o_proj is None

    def test_attention_params(self):
        """Test attention parameter counting"""
        attn = LoRAAttention(
            embed_dim=768, num_heads=12, rank=8, alpha=16, apply_lora_to=["q", "v"]
        )

        # Each projection: 8 * (768 + 768) = 12,288
        # 2 projections = 24,576
        expected = 2 * 8 * (768 + 768)
        assert attn.num_trainable_params() == expected


class TestConvenienceFunctions:
    """Tests for convenience functions"""

    def test_apply_lora_to_linear(self):
        """Test applying LoRA to linear weights"""
        np.random.seed(42)

        weight = np.random.randn(256, 128).astype(np.float32)
        config = LoRAConfig(rank=4, alpha=8)

        lora = apply_lora_to_linear(weight, config)

        assert lora.in_features == 128
        assert lora.out_features == 256
        assert lora.rank == 4
        np.testing.assert_array_equal(lora.W.data, weight)

    def test_calculate_lora_params(self):
        """Test parameter calculation"""
        stats = calculate_lora_params(
            model_params=1_000_000_000,  # 1B params
            num_lora_layers=64,  # 32 layers * 2 (q, v)
            in_features=4096,
            out_features=4096,
            rank=8,
        )

        assert "base_params" in stats
        assert "lora_params" in stats
        assert "trainable_ratio" in stats
        assert "total_training_memory_gb" in stats

        # Verify ratio is small
        assert stats["trainable_ratio"] < 0.01

        # Verify memory is reasonable
        assert stats["total_training_memory_gb"] < 1.0


class TestLoRAEmbedding:
    """Tests for LoRAEmbedding"""

    def test_embedding_forward(self):
        """Test embedding forward pass"""
        np.random.seed(42)

        embed = LoRAEmbedding(
            num_embeddings=1000,
            embedding_dim=256,
            rank=4,
            alpha=8,
        )

        input_ids = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        output = embed.forward(input_ids)

        assert output.data.shape == (2, 4, 256)

    def test_embedding_params(self):
        """Test embedding parameter counting"""
        embed = LoRAEmbedding(1000, 256, rank=4)

        params = embed.parameters()
        assert len(params) == 2  # A and B


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
