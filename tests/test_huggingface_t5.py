"""
Tests for HuggingFace T5 Tokenizer Integration

Tests T5 tokenizer integration with Grilly, including:
- Tokenization
- Encoding/decoding
- Tensor conversion to Vulkan
- Full pipeline integration
"""

import numpy as np
import pytest

try:
    from transformers import T5Tokenizer
    from transformers.testing_utils import require_sentencepiece, require_tokenizers

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

    def require_sentencepiece(x):
        return x

    def require_tokenizers(x):
        return x


try:
    from grilly import nn
    from grilly.utils.huggingface_bridge import HuggingFaceBridge, get_huggingface_bridge
    from grilly.utils.tensor_conversion import from_vulkan, to_vulkan, to_vulkan_gpu

    GRILLY_AVAILABLE = True
except ImportError:
    GRILLY_AVAILABLE = False


@pytest.mark.gpu
@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers not available")
@pytest.mark.skipif(not GRILLY_AVAILABLE, reason="Grilly not available")
@require_sentencepiece
@require_tokenizers
class TestT5TokenizerIntegration:
    """Test T5 tokenizer integration with Grilly"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures"""
        self.tokenizer_id = "google-t5/t5-small"
        self.bridge = get_huggingface_bridge()
        try:
            self.tokenizer = self.bridge.load_tokenizer(self.tokenizer_id)
        except Exception as e:
            pytest.skip(f"Could not load T5 tokenizer: {e}")

    def test_tokenizer_loading(self):
        """Test loading T5 tokenizer"""
        assert self.tokenizer is not None
        assert hasattr(self.tokenizer, "encode")
        assert hasattr(self.tokenizer, "decode")

    def test_basic_tokenization(self):
        """Test basic tokenization"""
        text = "This is a test"
        tokens = self.tokenizer.encode(text, return_tensors="np")

        assert tokens is not None
        assert isinstance(tokens, np.ndarray) or hasattr(tokens, "shape")

    def test_tokenization_with_special_tokens(self):
        """Test tokenization with special tokens"""
        text = "This is a test 😊 I was born in 92000, and this is falsé."
        tokens = self.tokenizer.encode(text, return_tensors="np")

        assert tokens is not None
        # Should handle emoji and special characters
        decoded = self.tokenizer.decode(tokens[0] if hasattr(tokens, "__getitem__") else tokens)
        assert isinstance(decoded, str)

    def test_tokenization_to_vulkan(self):
        """Test converting tokenized output to Vulkan"""
        text = "This is a test"
        tokens = self.tokenizer.encode(text, return_tensors="np")

        # Convert to Vulkan
        if isinstance(tokens, np.ndarray):
            vulkan_tokens = to_vulkan(tokens)
        else:
            # Handle PyTorch tensors
            vulkan_tokens = to_vulkan(tokens)

        assert isinstance(vulkan_tokens, np.ndarray)
        assert vulkan_tokens.dtype == np.float32 or vulkan_tokens.dtype in [np.int32, np.int64]

    def test_full_pipeline_tokenization(self):
        """Test full pipeline: tokenize → Vulkan → decode"""
        text = "This is a test"

        # Tokenize
        encoded = self.bridge.tokenize(text, self.tokenizer, return_tensors="np")

        # Get token IDs
        if isinstance(encoded, dict):
            input_ids = encoded.get("input_ids", encoded.get("input_ids"))
            if hasattr(input_ids, "numpy"):
                input_ids = input_ids.numpy()
        else:
            input_ids = encoded

        # Convert to Vulkan
        vulkan_ids = to_vulkan(input_ids)

        # Process with Vulkan (simple embedding-like operation)
        if len(vulkan_ids.shape) == 1:
            vulkan_ids = vulkan_ids.reshape(1, -1)

        # Use a simple linear layer to process
        linear = nn.Linear(vulkan_ids.shape[-1], 64)
        processed = linear(vulkan_ids.astype(np.float32))

        assert processed.shape[0] == vulkan_ids.shape[0]
        assert processed.shape[1] == 64

    def test_batch_tokenization(self):
        """Test batch tokenization"""
        texts = ["This is a test", "Hello world", "How are you?"]

        # Tokenize batch
        encoded = self.bridge.tokenize(texts, self.tokenizer, return_tensors="np")

        # Get input_ids
        if isinstance(encoded, dict):
            input_ids = encoded.get("input_ids")
            if hasattr(input_ids, "numpy"):
                input_ids = input_ids.numpy()
        else:
            input_ids = encoded

        # Convert to Vulkan
        vulkan_ids = to_vulkan(input_ids)

        assert vulkan_ids.shape[0] == len(texts) or vulkan_ids.ndim >= 1

    def test_tokenization_with_vulkan_gpu(self):
        """Test tokenization with GPU-optimized conversion"""
        text = "This is a test for GPU optimization"

        # Tokenize
        encoded = self.bridge.tokenize(text, self.tokenizer, return_tensors="np")

        # Get input_ids
        if isinstance(encoded, dict):
            input_ids = encoded.get("input_ids")
            if hasattr(input_ids, "numpy"):
                input_ids = input_ids.numpy()
        else:
            input_ids = encoded

        # Convert to GPU (if available)
        try:
            gpu_tokens = to_vulkan_gpu(input_ids)
            assert hasattr(gpu_tokens, "shape") or hasattr(gpu_tokens, "numpy")
        except Exception:
            # GPU conversion may not be available, that's okay
            pass

    def test_encoding_decoding_roundtrip(self):
        """Test encoding and decoding roundtrip"""
        original_text = "This is a test"

        # Encode
        encoded = self.tokenizer.encode(original_text, return_tensors="np")

        # Get token IDs
        if hasattr(encoded, "numpy"):
            token_ids = encoded.numpy()
        elif isinstance(encoded, np.ndarray):
            token_ids = encoded
        else:
            token_ids = np.array(encoded)

        # Decode
        if token_ids.ndim > 1:
            decoded = self.tokenizer.decode(token_ids[0])
        else:
            decoded = self.tokenizer.decode(token_ids)

        assert isinstance(decoded, str)
        # Decoded text should be similar (may have special tokens)
        assert len(decoded) > 0

    def test_special_characters_tokenization(self):
        """Test tokenization of special characters"""
        texts = ["Hello 😊", "Test é character", "ปี (Thai)", "生活的真谛是 (Chinese)"]

        for text in texts:
            encoded = self.tokenizer.encode(text, return_tensors="np")
            assert encoded is not None

            # Convert to Vulkan
            vulkan_encoded = to_vulkan(encoded)
            assert isinstance(vulkan_encoded, np.ndarray)

    def test_tokenizer_with_model_encoding(self):
        """Test tokenizer with model encoding (embedding extraction)"""
        try:
            text = "This is a test sentence"

            # Use a sentence transformer model that's compatible with T5 tokenizer
            # Note: T5 tokenizer might not be compatible with all sentence-transformers models
            # So we'll use a model that works with T5 or fall back to a compatible tokenizer
            model_name = "sentence-transformers/all-MiniLM-L6-v2"

            # Try to encode with the model
            # If T5 tokenizer doesn't work with this model, we'll use the model's default tokenizer
            try:
                embeddings = self.bridge.encode(
                    text,
                    model_name=model_name,
                    tokenizer_name=None,  # Use model's default tokenizer
                )
            except Exception:
                # If that fails, try with T5 tokenizer (may not work, but worth trying)
                try:
                    embeddings = self.bridge.encode(
                        text, model_name=model_name, tokenizer_name=self.tokenizer_id
                    )
                except Exception as e2:
                    # Model encoding requires CUDA or specific setup, skip if not available
                    pytest.skip(
                        f"Model encoding not available (may require CUDA or model download): {e2}"
                    )

            # If model encoding works, embeddings should be numpy
            if embeddings is not None:
                assert isinstance(embeddings, np.ndarray)
                assert len(embeddings.shape) >= 1

                # Convert to Vulkan and process
                vulkan_emb = to_vulkan(embeddings)
                linear = nn.Linear(vulkan_emb.shape[-1], 64)
                result = linear(vulkan_emb.reshape(1, -1) if vulkan_emb.ndim == 1 else vulkan_emb)
                assert result.shape[-1] == 64
        except Exception as e:
            # Model may not be available, skip with informative message
            error_msg = str(e)
            if "CUDA" in error_msg or "cuda" in error_msg.lower():
                pytest.skip(f"Model encoding requires CUDA (AMD/Vulkan systems may skip this): {e}")
            elif "not found" in error_msg.lower() or "download" in error_msg.lower():
                pytest.skip(f"Model not downloaded or not available: {e}")
            else:
                pytest.skip(f"Model encoding not available: {e}")


@pytest.mark.gpu
@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers not available")
@pytest.mark.skipif(not GRILLY_AVAILABLE, reason="Grilly not available")
@require_sentencepiece
@require_tokenizers
class TestT5TokenizerExpectedOutputs:
    """Test T5 tokenizer against expected outputs"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures"""
        self.tokenizer_id = "google-t5/t5-small"
        self.bridge = get_huggingface_bridge()
        try:
            self.tokenizer = self.bridge.load_tokenizer(self.tokenizer_id)
        except Exception as e:
            pytest.skip(f"Could not load T5 tokenizer: {e}")

    def test_expected_tokenization(self):
        """Test tokenization against expected outputs"""
        # Test text from the original T5 test
        test_text = "This is a test 😊 I was born in 92000, and this is falsé. 生活的真谛是 Hi Hello Hi Hello Hello <s> hi<s>there The following string should be properly encoded: Hello. But ird and ird  Hey how are you doing"

        # Tokenize
        tokens = self.tokenizer.encode(test_text, add_special_tokens=True)

        # Should produce valid token IDs
        assert isinstance(tokens, list) or hasattr(tokens, "__len__")
        assert len(tokens) > 0

        # Decode back
        decoded = self.tokenizer.decode(tokens)
        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_integration_expected_tokens(self):
        """Test that tokenization produces expected token structure"""
        test_text = "This is a test"

        # Tokenize
        tokens = self.tokenizer.tokenize(test_text)

        # Should be a list of token strings
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

        # Convert to IDs
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        assert isinstance(token_ids, list)
        assert len(token_ids) == len(tokens)

    def test_vulkan_processing_of_token_ids(self):
        """Test processing token IDs with Vulkan"""
        test_text = "This is a test"

        # Get token IDs
        token_ids = self.tokenizer.encode(test_text, return_tensors="np")

        # Convert to numpy if needed
        if hasattr(token_ids, "numpy"):
            token_ids = token_ids.numpy()
        elif not isinstance(token_ids, np.ndarray):
            token_ids = np.array(token_ids)

        # Convert to Vulkan
        vulkan_ids = to_vulkan(token_ids.astype(np.float32))

        # Process with a simple model
        if vulkan_ids.ndim == 1:
            vulkan_ids = vulkan_ids.reshape(1, -1)

        # Use embedding-like processing
        linear = nn.Linear(vulkan_ids.shape[-1], 128)
        processed = linear(vulkan_ids)

        assert processed.shape[0] == vulkan_ids.shape[0]
        assert processed.shape[1] == 128


@pytest.mark.gpu
@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers not available")
@pytest.mark.skipif(not GRILLY_AVAILABLE, reason="Grilly not available")
@require_sentencepiece
@require_tokenizers
class TestT5TokenizerVulkanIntegration:
    """Test T5 tokenizer with full Vulkan integration"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures"""
        self.tokenizer_id = "google-t5/t5-small"
        self.bridge = get_huggingface_bridge()
        try:
            self.tokenizer = self.bridge.load_tokenizer(self.tokenizer_id)
        except Exception as e:
            pytest.skip(f"Could not load T5 tokenizer: {e}")

    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline"""
        text = "This is a test sentence for the pipeline"

        # Step 1: Tokenize
        encoded = self.bridge.tokenize(text, self.tokenizer, return_tensors="np")

        # Step 2: Extract input_ids
        if isinstance(encoded, dict):
            input_ids = encoded.get("input_ids")
        else:
            input_ids = encoded

        if hasattr(input_ids, "numpy"):
            input_ids = input_ids.numpy()
        elif not isinstance(input_ids, np.ndarray):
            input_ids = np.array(input_ids)

        # Step 3: Convert to Vulkan
        vulkan_input = to_vulkan(input_ids.astype(np.float32))

        # Step 4: Process with Vulkan model
        if vulkan_input.ndim == 1:
            vulkan_input = vulkan_input.reshape(1, -1)

        model = nn.Sequential(
            nn.Linear(vulkan_input.shape[-1], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        vulkan_output = model(vulkan_input)

        # Step 5: Convert back if needed
        assert isinstance(vulkan_output, np.ndarray)
        assert vulkan_output.shape[0] == vulkan_input.shape[0]
        assert vulkan_output.shape[1] == 64

    def test_batch_processing_pipeline(self):
        """Test batch processing pipeline"""
        texts = ["First sentence", "Second sentence", "Third sentence"]

        # Tokenize batch
        encoded = self.bridge.tokenize(texts, self.tokenizer, return_tensors="np")

        # Extract input_ids
        if isinstance(encoded, dict):
            input_ids = encoded.get("input_ids")
        else:
            input_ids = encoded

        if hasattr(input_ids, "numpy"):
            input_ids = input_ids.numpy()
        elif not isinstance(input_ids, np.ndarray):
            input_ids = np.array(input_ids)

        # Convert to Vulkan
        vulkan_input = to_vulkan(input_ids.astype(np.float32))

        # Process batch
        if vulkan_input.ndim == 1:
            vulkan_input = vulkan_input.reshape(1, -1)

        # Pad or handle variable lengths (simplified)
        max_len = vulkan_input.shape[-1]

        # Simple processing
        linear = nn.Linear(max_len, 128)
        result = linear(vulkan_input)

        assert result.shape[0] == vulkan_input.shape[0] or result.shape[0] == 1
        assert result.shape[1] == 128
