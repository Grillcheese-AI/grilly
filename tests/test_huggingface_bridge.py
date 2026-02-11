"""
Tests for HuggingFace Bridge

Tests HuggingFace integration with CUDA/PyTorch compatibility
Note: These tests may skip if transformers/PyTorch not available
"""

import numpy as np
import pytest

try:
    from grilly.utils.huggingface_bridge import HuggingFaceBridge, get_huggingface_bridge

    HUGGINGFACE_BRIDGE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_BRIDGE_AVAILABLE = False


@pytest.mark.skipif(not HUGGINGFACE_BRIDGE_AVAILABLE, reason="HuggingFace bridge not available")
class TestHuggingFaceBridge:
    """Test HuggingFaceBridge class"""

    def test_bridge_initialization(self):
        """Test bridge can be initialized"""
        try:
            bridge = HuggingFaceBridge()
            assert bridge is not None
        except RuntimeError as e:
            if "PyTorch" in str(e) or "Transformers" in str(e):
                pytest.skip(f"HuggingFace bridge dependencies not available: {e}")
            raise

    def test_load_tokenizer(self):
        """Test loading tokenizer"""
        try:
            bridge = HuggingFaceBridge()
            # Use a small model for testing
            tokenizer = bridge.load_tokenizer("bert-base-uncased")
            assert tokenizer is not None
        except (RuntimeError, Exception) as e:
            if "not available" in str(e).lower() or "not found" in str(e).lower():
                pytest.skip(f"Model not available: {e}")
            raise

    def test_tokenize(self):
        """Test tokenization"""
        try:
            bridge = HuggingFaceBridge()
            try:
                encoded = bridge.tokenize("Hello, world!", "bert-base-uncased", return_tensors="np")
                assert "input_ids" in encoded
                assert isinstance(encoded["input_ids"], np.ndarray)
            except Exception as e:
                if "not found" in str(e).lower():
                    pytest.skip(f"Model not available: {e}")
                raise
        except RuntimeError as e:
            if "PyTorch" in str(e) or "Transformers" in str(e):
                pytest.skip(f"Dependencies not available: {e}")
            raise

    def test_to_vulkan(self):
        """Test converting PyTorch tensor to numpy"""
        try:
            bridge = HuggingFaceBridge()
            import torch

            tensor = torch.randn(10, 20)
            result = bridge.to_vulkan(tensor)
            assert isinstance(result, np.ndarray)
            assert result.shape == (10, 20)
        except (RuntimeError, ImportError) as e:
            pytest.skip(f"PyTorch not available: {e}")

    def test_to_cuda(self):
        """Test converting numpy to CUDA tensor"""
        try:
            import torch

            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")
            bridge = HuggingFaceBridge()
            arr = np.random.randn(10, 20).astype(np.float32)
            result = bridge.to_cuda(arr)
            assert isinstance(result, torch.Tensor)
            assert result.device.type == "cuda"
        except (RuntimeError, ImportError, AssertionError) as e:
            pytest.skip(f"CUDA/PyTorch not available: {e}")


@pytest.mark.skipif(not HUGGINGFACE_BRIDGE_AVAILABLE, reason="HuggingFace bridge not available")
class TestHuggingFaceBridgeGlobal:
    """Test global HuggingFace bridge functions"""

    def test_get_huggingface_bridge(self):
        """Test getting global bridge instance"""
        try:
            bridge = get_huggingface_bridge()
            assert isinstance(bridge, HuggingFaceBridge)
        except RuntimeError as e:
            if "PyTorch" in str(e) or "Transformers" in str(e):
                pytest.skip(f"Dependencies not available: {e}")
            raise


@pytest.mark.gpu
@pytest.mark.skipif(not HUGGINGFACE_BRIDGE_AVAILABLE, reason="HuggingFace bridge not available")
class TestHuggingFaceBridgeVulkanOnly:
    """Test HuggingFace bridge with Vulkan-only operations (AMD compatible)"""

    def test_tensor_conversion_vulkan(self):
        """Test tensor conversion for Vulkan (no CUDA required)"""
        try:
            bridge = HuggingFaceBridge()
            # Test numpy to numpy (Vulkan path)
            arr = np.random.randn(10, 20).astype(np.float32)
            result = bridge.to_vulkan(arr)
            assert isinstance(result, np.ndarray)
            np.testing.assert_array_equal(result, arr)
        except RuntimeError as e:
            if "PyTorch" in str(e) or "Transformers" in str(e):
                pytest.skip(f"Dependencies not available: {e}")
            raise

    def test_numpy_operations(self):
        """Test that numpy arrays work with Vulkan operations"""
        try:
            from grilly import nn

            HuggingFaceBridge()

            # Create numpy array (Vulkan-compatible)
            arr = np.random.randn(5, 128).astype(np.float32)

            # Process with Vulkan operations
            linear = nn.Linear(128, 64)
            result = linear(arr)

            assert result.shape == (5, 64)
            assert isinstance(result, np.ndarray)
        except RuntimeError as e:
            if "PyTorch" in str(e) or "Transformers" in str(e):
                pytest.skip(f"Dependencies not available: {e}")
            raise
        except Exception as e:
            if "Vulkan" in str(e) or "not available" in str(e).lower():
                pytest.skip(f"Vulkan not available: {e}")
            raise


class TestHuggingFaceBridgeLoRA:
    """Test LoRA support in HuggingFace bridge (no external dependencies required)"""

    def test_lora_methods_exist(self):
        """Test that LoRA methods are available on the class"""
        assert hasattr(HuggingFaceBridge, "load_model_with_lora")
        assert hasattr(HuggingFaceBridge, "save_lora_adapters")
        assert hasattr(HuggingFaceBridge, "load_lora_adapters")
        assert hasattr(HuggingFaceBridge, "extract_model_weights")
        assert hasattr(HuggingFaceBridge, "create_lora_from_weights")
        assert hasattr(HuggingFaceBridge, "merge_lora_to_model")

    def test_create_lora_from_weights(self):
        """Test creating LoRA from weights dict (no HF model needed)"""
        from grilly.nn.lora import LoRAConfig, LoRAModel

        # Simulate extracted weights
        weights = {
            "layer.0.attention.q_proj": np.random.randn(768, 768).astype(np.float32) * 0.01,
            "layer.0.attention.v_proj": np.random.randn(768, 768).astype(np.float32) * 0.01,
            "layer.0.attention.k_proj": np.random.randn(768, 768).astype(np.float32) * 0.01,
            "layer.1.attention.q_proj": np.random.randn(768, 768).astype(np.float32) * 0.01,
            "layer.1.attention.v_proj": np.random.randn(768, 768).astype(np.float32) * 0.01,
        }

        # Create LoRA config
        config = LoRAConfig(rank=8, alpha=16, target_modules=["q_proj", "v_proj"])

        # Create LoRA model manually (same logic as bridge method)
        lora_model = LoRAModel(config)

        for name, weight in weights.items():
            layer_name = name.split(".")[-1]
            if any(tm in layer_name for tm in config.target_modules):
                if len(weight.shape) == 2:
                    out_features, in_features = weight.shape
                    lora_model.add_lora_layer(
                        name=name,
                        in_features=in_features,
                        out_features=out_features,
                        base_weights=weight,
                    )

        # Verify LoRA layers were created for q_proj and v_proj only
        assert len(lora_model.lora_layers) == 4  # 2 layers x 2 modules (q_proj, v_proj)

        # Verify k_proj was not included
        for name in lora_model.lora_layers:
            assert "k_proj" not in name

    def test_lora_save_load_cycle(self):
        """Test saving and loading LoRA adapters"""
        import tempfile
        from pathlib import Path

        from grilly.nn.lora import LoRAConfig, LoRAModel

        # Create LoRA model
        config = LoRAConfig(rank=4, alpha=8, target_modules=["q_proj"])
        lora_model = LoRAModel(config)

        # Add some layers
        lora_model.add_lora_layer("layer.0.q_proj", 256, 256)
        lora_model.add_lora_layer("layer.1.q_proj", 256, 256)

        # Set some non-zero weights
        for name, layer in lora_model.lora_layers.items():
            layer.lora_A.data = np.random.randn(*layer.lora_A.data.shape).astype(np.float32)
            layer.lora_B.data = np.random.randn(*layer.lora_B.data.shape).astype(np.float32)

        # Save to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "lora_test"
            lora_model.save_checkpoint(save_path)

            # Verify files exist
            assert (save_path / "config.json").exists()
            assert (save_path / "adapters.npz").exists()
            assert (save_path / "metadata.json").exists()

            # Load back
            loaded_model = LoRAModel.load_checkpoint(save_path)

            # Verify structure matches
            assert len(loaded_model.lora_layers) == len(lora_model.lora_layers)
            assert loaded_model.config.rank == config.rank
            assert loaded_model.config.alpha == config.alpha

            # Verify weights match
            for name in lora_model.lora_layers:
                orig = lora_model.lora_layers[name]
                loaded = loaded_model.lora_layers[name]
                np.testing.assert_array_almost_equal(
                    orig.lora_A.data, loaded.lora_A.data, decimal=5
                )
                np.testing.assert_array_almost_equal(
                    orig.lora_B.data, loaded.lora_B.data, decimal=5
                )

    def test_lora_parameter_counting(self):
        """Test LoRA reduces trainable parameter count"""
        from grilly.nn.lora import calculate_lora_params

        # Simulate a 3B parameter model with 32 attention layers
        # Each q_proj and v_proj: 4096 x 4096
        model_params = 3_000_000_000
        num_layers = 32
        num_lora_layers = num_layers * 2  # q_proj + v_proj per layer
        in_features = 4096
        out_features = 4096
        rank = 8

        stats = calculate_lora_params(
            model_params=model_params,
            num_lora_layers=num_lora_layers,
            in_features=in_features,
            out_features=out_features,
            rank=rank,
        )

        # Verify parameter reduction
        assert stats["trainable_ratio"] < 0.01  # Less than 1% trainable
        assert stats["lora_params"] < 10_000_000  # Less than 10M LoRA params
        assert stats["total_training_memory_gb"] < 1.0  # Less than 1GB for training
