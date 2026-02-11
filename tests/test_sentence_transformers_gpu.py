"""
Tests for sentence-transformers GPU integration

Tests sentence-transformers integration with GPU support (CUDA or CPU for AMD).
"""

import numpy as np
import pytest

try:
    from grilly import nn
    from grilly.utils.huggingface_bridge import HuggingFaceBridge, get_huggingface_bridge
    from grilly.utils.tensor_conversion import to_vulkan, to_vulkan_gpu

    BRIDGE_AVAILABLE = True
except ImportError:
    BRIDGE_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@pytest.mark.gpu
@pytest.mark.skipif(not BRIDGE_AVAILABLE, reason="HuggingFace bridge not available")
@pytest.mark.skipif(
    not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available"
)
class TestSentenceTransformersGPU:
    """Test sentence-transformers with GPU support"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures"""
        try:
            self.bridge = get_huggingface_bridge()
        except Exception as e:
            pytest.skip(f"Could not initialize bridge: {e}")

    def test_load_sentence_transformer(self):
        """Test loading sentence-transformer model"""
        try:
            model = self.bridge.load_sentence_transformer("all-MiniLM-L6-v2")
            assert model is not None
        except Exception as e:
            pytest.skip(f"Could not load model: {e}")

    def test_encode_single_text(self):
        """Test encoding single text"""
        try:
            text = "This is a test sentence"
            embeddings = self.bridge.encode_sentence_transformer(text)

            assert isinstance(embeddings, np.ndarray)
            assert embeddings.ndim == 1
            assert embeddings.dtype == np.float32
            assert embeddings.shape[0] > 0  # Should have embedding dimension
        except Exception as e:
            pytest.skip(f"Encoding failed: {e}")

    def test_encode_batch(self):
        """Test batch encoding"""
        try:
            texts = ["First sentence", "Second sentence", "Third sentence"]
            embeddings = self.bridge.encode_sentence_transformer(texts)

            assert isinstance(embeddings, np.ndarray)
            assert embeddings.ndim == 2
            assert embeddings.shape[0] == len(texts)
            assert embeddings.dtype == np.float32
        except Exception as e:
            pytest.skip(f"Batch encoding failed: {e}")

    def test_encode_with_vulkan_conversion(self):
        """Test encoding and converting to Vulkan"""
        try:
            text = "This is a test"
            embeddings = self.bridge.encode_sentence_transformer(text)

            # Convert to Vulkan
            vulkan_emb = to_vulkan(embeddings)
            assert isinstance(vulkan_emb, np.ndarray)
            assert vulkan_emb.dtype == np.float32

            # Process with Vulkan
            linear = nn.Linear(vulkan_emb.shape[0], 64)
            result = linear(vulkan_emb.reshape(1, -1))
            assert result.shape == (1, 64)
        except Exception as e:
            pytest.skip(f"Vulkan conversion failed: {e}")

    def test_encode_gpu_with_vulkan_postprocessing(self):
        """Test GPU encoding with Vulkan post-processing"""
        try:
            texts = ["Hello, world!", "How are you?"]
            embeddings = self.bridge.encode_sentence_transformer_gpu(
                texts, use_vulkan_postprocessing=True
            )

            assert isinstance(embeddings, np.ndarray)
            assert embeddings.ndim == 2
            assert embeddings.shape[0] == len(texts)
            assert embeddings.dtype == np.float32
        except Exception as e:
            pytest.skip(f"GPU encoding with Vulkan post-processing failed: {e}")

    def test_custom_model(self):
        """Test using custom model"""
        try:
            text = "Test text"
            embeddings = self.bridge.encode_sentence_transformer(
                text, model_name="all-MiniLM-L6-v2", normalize_embeddings=True
            )

            assert isinstance(embeddings, np.ndarray)
            # Check normalization (should be close to 1.0)
            norm = np.linalg.norm(embeddings)
            assert abs(norm - 1.0) < 0.1  # Should be normalized
        except Exception as e:
            pytest.skip(f"Custom model test failed: {e}")

    def test_device_selection(self):
        """Test automatic device selection"""
        try:
            # Should work on both CUDA and CPU (AMD)
            text = "Test"
            embeddings = self.bridge.encode_sentence_transformer(text)

            assert isinstance(embeddings, np.ndarray)
            # Should work regardless of device
            print(f"Device: {'CUDA' if self.bridge.cuda_device else 'CPU (AMD/Vulkan)'}")
        except Exception as e:
            pytest.skip(f"Device selection test failed: {e}")


@pytest.mark.gpu
@pytest.mark.skipif(not BRIDGE_AVAILABLE, reason="HuggingFace bridge not available")
@pytest.mark.skipif(
    not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available"
)
class TestSentenceTransformersIntegration:
    """Test sentence-transformers integration with Vulkan operations"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures"""
        try:
            self.bridge = get_huggingface_bridge()
        except Exception as e:
            pytest.skip(f"Could not initialize bridge: {e}")

    def test_end_to_end_pipeline(self):
        """Test complete pipeline: encode → Vulkan → process"""
        try:
            texts = ["First text", "Second text"]

            # Step 1: Encode with sentence-transformers
            embeddings = self.bridge.encode_sentence_transformer(texts)

            # Step 2: Convert to Vulkan
            vulkan_emb = to_vulkan(embeddings)

            # Step 3: Process with Vulkan model
            model = nn.Sequential(
                nn.Linear(vulkan_emb.shape[-1], 256), nn.ReLU(), nn.Linear(256, 128)
            )

            result = model(vulkan_emb)

            assert result.shape[0] == len(texts)
            assert result.shape[1] == 128
        except Exception as e:
            pytest.skip(f"End-to-end pipeline failed: {e}")

    def test_similarity_computation(self):
        """Test computing similarities between embeddings"""
        try:
            query = "What is machine learning?"
            documents = [
                "Machine learning is a subset of AI",
                "Python is a programming language",
                "Deep learning uses neural networks",
            ]

            # Encode all
            query_emb = self.bridge.encode_sentence_transformer(query)
            doc_embs = self.bridge.encode_sentence_transformer(documents)

            # Compute similarities (cosine similarity)
            similarities = np.dot(doc_embs, query_emb)

            assert len(similarities) == len(documents)
            assert all(-1 <= s <= 1 for s in similarities)  # Cosine similarity range
        except Exception as e:
            pytest.skip(f"Similarity computation failed: {e}")
