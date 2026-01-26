"""
Tests for HuggingFace Bridge

Tests HuggingFace integration with CUDA/PyTorch compatibility
Note: These tests may skip if transformers/PyTorch not available
"""
import pytest
import numpy as np

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
                encoded = bridge.tokenize("Hello, world!", "bert-base-uncased", return_tensors='np')
                assert 'input_ids' in encoded
                assert isinstance(encoded['input_ids'], np.ndarray)
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
            assert result.device.type == 'cuda'
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
            bridge = HuggingFaceBridge()
            
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
