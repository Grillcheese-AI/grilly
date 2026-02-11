"""
Tests for multimodal AI techniques.

Tests all multimodal fusion architectures:
1. Multimodal Bottleneck Transformer (MBT)
2. Perceiver IO
3. Cross-Attention Fusion
4. ImageBind
5. Perceiver Resampler (Flamingo)
6. Complete VLM
"""

import numpy as np
import pytest

try:
    from grilly import nn
    from grilly.nn.multimodal import (
        BottleneckFusion,
        CrossModalAttentionFusion,
        FlamingoFusion,
        ImageBindFusion,
        PerceiverIO,
        PerceiverResampler,
        VisionLanguageModel,
        VLMLayer,
    )

    MULTIMODAL_AVAILABLE = True
except ImportError as e:
    MULTIMODAL_AVAILABLE = False
    print(f"Multimodal import error: {e}")


@pytest.mark.gpu
@pytest.mark.skipif(not MULTIMODAL_AVAILABLE, reason="Multimodal modules not available")
class TestBottleneckFusion:
    """Test Multimodal Bottleneck Transformer (MBT)"""

    def test_bottleneck_fusion_shapes(self):
        """Test output shapes"""
        batch_size = 4
        d_model = 768
        num_bottlenecks = 64

        model = BottleneckFusion(d_model=d_model, num_bottlenecks=num_bottlenecks, num_heads=8)

        # Video and audio features
        video = np.random.randn(batch_size, 128, d_model).astype(np.float32)
        audio = np.random.randn(batch_size, 64, d_model).astype(np.float32)

        output = model(video, audio)

        assert output.shape == (batch_size, num_bottlenecks, d_model)
        assert np.all(np.isfinite(output))

    def test_bottleneck_fusion_different_seq_lengths(self):
        """Test with different sequence lengths"""
        model = BottleneckFusion(d_model=256, num_bottlenecks=32, num_heads=4)

        # Different sequence lengths
        mod1 = np.random.randn(2, 100, 256).astype(np.float32)
        mod2 = np.random.randn(2, 50, 256).astype(np.float32)

        output = model(mod1, mod2)

        assert output.shape == (2, 32, 256)


@pytest.mark.gpu
@pytest.mark.skipif(not MULTIMODAL_AVAILABLE, reason="Multimodal modules not available")
class TestPerceiverIO:
    """Test Perceiver IO architecture"""

    def test_perceiver_io_basic(self):
        """Test basic Perceiver IO functionality"""
        model = PerceiverIO(
            input_dim=512, latent_dim=1024, num_latents=256, num_heads=8, num_layers=3
        )

        # Large multimodal input
        inputs = np.random.randn(2, 1000, 512).astype(np.float32)

        output = model(inputs)

        # Should compress to fixed latent size
        assert output.shape == (2, 256, 1024)
        assert np.all(np.isfinite(output))

    def test_perceiver_io_with_output_queries(self):
        """Test Perceiver IO with custom output queries"""
        model = PerceiverIO(
            input_dim=256, latent_dim=512, num_latents=128, num_heads=4, num_layers=2
        )

        inputs = np.random.randn(4, 500, 256).astype(np.float32)
        output_queries = np.random.randn(4, 64, 512).astype(np.float32)

        output = model(inputs, output_queries=output_queries)

        # Output should match query shape
        assert output.shape == (4, 64, 512)

    def test_perceiver_io_multimodal_concat(self):
        """Test Perceiver IO with concatenated multimodal inputs"""
        model = PerceiverIO(
            input_dim=512, latent_dim=768, num_latents=128, num_heads=8, num_layers=4
        )

        # Concatenate different modalities
        image_features = np.random.randn(2, 196, 512).astype(np.float32)  # ViT patches
        audio_features = np.random.randn(2, 100, 512).astype(np.float32)  # Audio
        text_features = np.random.randn(2, 77, 512).astype(np.float32)  # Text

        multimodal_input = np.concatenate([image_features, audio_features, text_features], axis=1)

        output = model(multimodal_input)

        assert output.shape == (2, 128, 768)


@pytest.mark.gpu
@pytest.mark.skipif(not MULTIMODAL_AVAILABLE, reason="Multimodal modules not available")
class TestCrossModalAttentionFusion:
    """Test Cross-Modal Attention Fusion"""

    def test_cross_modal_fusion_basic(self):
        """Test basic cross-modal fusion"""
        model = CrossModalAttentionFusion(d_model=768, num_heads=8, num_encoder_layers=2)

        vision = np.random.randn(4, 196, 768).astype(np.float32)
        text = np.random.randn(4, 77, 768).astype(np.float32)

        output = model(vision, text)

        # Output should be pooled representation
        assert output.shape == (4, 768)
        assert np.all(np.isfinite(output))

    def test_cross_modal_fusion_small(self):
        """Test with smaller dimensions"""
        model = CrossModalAttentionFusion(d_model=256, num_heads=4, num_encoder_layers=1)

        vision = np.random.randn(2, 49, 256).astype(np.float32)
        text = np.random.randn(2, 32, 256).astype(np.float32)

        output = model(vision, text)

        assert output.shape == (2, 256)


@pytest.mark.gpu
@pytest.mark.skipif(not MULTIMODAL_AVAILABLE, reason="Multimodal modules not available")
class TestImageBindFusion:
    """Test ImageBind-style joint embedding"""

    def test_imagebind_encoding(self):
        """Test modality encoding"""
        model = ImageBindFusion(
            embed_dim=1024, image_input_dim=2048, text_input_dim=768, audio_input_dim=512
        )

        # Test individual encodings
        image_feats = np.random.randn(8, 2048).astype(np.float32)
        text_feats = np.random.randn(8, 768).astype(np.float32)
        audio_feats = np.random.randn(8, 512).astype(np.float32)

        image_emb = model.encode("image", image_feats)
        text_emb = model.encode("text", text_feats)
        audio_emb = model.encode("audio", audio_feats)

        # All should have same embedding dimension
        assert image_emb.shape == (8, 1024)
        assert text_emb.shape == (8, 1024)
        assert audio_emb.shape == (8, 1024)

        # Should be L2 normalized
        image_norms = np.linalg.norm(image_emb, axis=-1)
        assert np.allclose(image_norms, 1.0, atol=1e-5)

    def test_imagebind_forward(self):
        """Test full forward pass with contrastive loss"""
        model = ImageBindFusion(embed_dim=512)

        image_feats = np.random.randn(16, 2048).astype(np.float32)
        text_feats = np.random.randn(16, 768).astype(np.float32)
        audio_feats = np.random.randn(16, 512).astype(np.float32)

        output = model(image_feats, text_feats, audio_feats)

        # Check outputs
        assert "image_embeddings" in output
        assert "text_embeddings" in output
        assert "audio_embeddings" in output
        assert "loss" in output
        assert "loss_img_text" in output
        assert "loss_img_audio" in output

        # Loss should be positive
        assert output["loss"] >= 0
        assert np.isfinite(output["loss"])

    def test_imagebind_cross_modal_similarity(self):
        """Test cross-modal similarity computation"""
        model = ImageBindFusion(embed_dim=256)

        # Create embeddings
        image_feats = np.random.randn(4, 2048).astype(np.float32)
        text_feats = np.random.randn(4, 768).astype(np.float32)

        image_emb = model.encode("image", image_feats)
        text_emb = model.encode("text", text_feats)

        # Compute similarity matrix
        similarity = image_emb @ text_emb.T

        assert similarity.shape == (4, 4)
        assert np.all(similarity >= -1) and np.all(similarity <= 1)


@pytest.mark.gpu
@pytest.mark.skipif(not MULTIMODAL_AVAILABLE, reason="Multimodal modules not available")
class TestPerceiverResampler:
    """Test Perceiver Resampler (Flamingo)"""

    def test_perceiver_resampler_basic(self):
        """Test basic resampling"""
        model = PerceiverResampler(dim=1024, depth=3, num_latents=64, num_heads=8)

        # Variable length visual features
        visual = np.random.randn(2, 256, 1024).astype(np.float32)

        output = model(visual)

        # Should resample to fixed number of tokens
        assert output.shape == (2, 64, 1024)
        assert np.all(np.isfinite(output))

    def test_perceiver_resampler_different_inputs(self):
        """Test with different input lengths"""
        model = PerceiverResampler(dim=512, num_latents=32, depth=2)

        # Image patches
        image = np.random.randn(4, 196, 512).astype(np.float32)
        output1 = model(image)
        assert output1.shape == (4, 32, 512)

        # Video frames
        video = np.random.randn(4, 1024, 512).astype(np.float32)
        output2 = model(video)
        assert output2.shape == (4, 32, 512)


@pytest.mark.gpu
@pytest.mark.skipif(not MULTIMODAL_AVAILABLE, reason="Multimodal modules not available")
class TestFlamingoFusion:
    """Test Flamingo-style VLM fusion"""

    def test_flamingo_fusion_basic(self):
        """Test basic Flamingo fusion"""
        model = FlamingoFusion(vision_dim=1024, text_dim=2048, num_visual_tokens=64, num_heads=16)

        visual = np.random.randn(2, 256, 1024).astype(np.float32)
        text = np.random.randn(2, 128, 2048).astype(np.float32)

        output = model(visual, text)

        # Output should have same shape as text input
        assert output.shape == (2, 128, 2048)
        assert np.all(np.isfinite(output))

    def test_flamingo_fusion_small(self):
        """Test with smaller dimensions"""
        model = FlamingoFusion(vision_dim=256, text_dim=512, num_visual_tokens=16, num_heads=4)

        visual = np.random.randn(4, 49, 256).astype(np.float32)
        text = np.random.randn(4, 32, 512).astype(np.float32)

        output = model(visual, text)

        assert output.shape == (4, 32, 512)


@pytest.mark.gpu
@pytest.mark.skipif(not MULTIMODAL_AVAILABLE, reason="Multimodal modules not available")
class TestVLMLayer:
    """Test VLM transformer layer"""

    def test_vlm_layer_basic(self):
        """Test single VLM layer"""
        layer = VLMLayer(dim=512, num_heads=8)

        text = np.random.randn(2, 64, 512).astype(np.float32)
        visual = np.random.randn(2, 32, 512).astype(np.float32)

        output = layer(text, visual)

        assert output.shape == text.shape
        assert np.all(np.isfinite(output))


@pytest.mark.gpu
@pytest.mark.skipif(not MULTIMODAL_AVAILABLE, reason="Multimodal modules not available")
class TestVisionLanguageModel:
    """Test complete Vision-Language Model"""

    def test_vlm_forward(self):
        """Test VLM forward pass"""
        model = VisionLanguageModel(
            vision_dim=256,
            text_dim=128,
            hidden_dim=256,
            num_visual_tokens=16,
            num_heads=4,
            num_layers=2,
            vocab_size=1000,
        )

        # Vision features (e.g., from ViT)
        vision = np.random.randn(2, 49, 256).astype(np.float32)

        # Token IDs
        input_ids = np.random.randint(0, 1000, (2, 32))

        logits = model(vision, input_ids)

        # Output should be logits over vocabulary
        assert logits.shape == (2, 32, 1000)
        assert np.all(np.isfinite(logits))

    def test_vlm_generation_simulation(self):
        """Test VLM in generation-like scenario"""
        model = VisionLanguageModel(
            vision_dim=128,
            text_dim=64,
            hidden_dim=128,
            num_visual_tokens=8,
            num_heads=2,
            num_layers=1,
            vocab_size=500,
        )

        vision = np.random.randn(1, 16, 128).astype(np.float32)
        input_ids = np.array([[1, 2, 3]])  # Short prompt

        logits = model(vision, input_ids)

        # Get predicted token
        predicted = np.argmax(logits[0, -1])
        assert 0 <= predicted < 500


@pytest.mark.gpu
@pytest.mark.skipif(not MULTIMODAL_AVAILABLE, reason="Multimodal modules not available")
class TestMultimodalIntegration:
    """Integration tests for multimodal modules"""

    def test_perceiver_to_flamingo_pipeline(self):
        """Test Perceiver IO feeding into Flamingo-style fusion"""
        # First, process multimodal input with Perceiver IO
        perceiver = PerceiverIO(input_dim=512, latent_dim=1024, num_latents=64, num_layers=2)

        # Then use Flamingo fusion
        flamingo = FlamingoFusion(vision_dim=1024, text_dim=2048, num_visual_tokens=32)

        # Multimodal input
        multimodal = np.random.randn(2, 500, 512).astype(np.float32)
        text = np.random.randn(2, 64, 2048).astype(np.float32)

        # Process
        latents = perceiver(multimodal)  # (2, 64, 1024)
        output = flamingo(latents, text)  # (2, 64, 2048)

        assert output.shape == (2, 64, 2048)

    def test_bottleneck_with_imagebind(self):
        """Test combining bottleneck fusion with ImageBind-style training"""
        # Bottleneck fusion for two modalities
        bottleneck = BottleneckFusion(d_model=512, num_bottlenecks=32, num_heads=4)

        # Fuse video and audio
        video = np.random.randn(4, 100, 512).astype(np.float32)
        audio = np.random.randn(4, 50, 512).astype(np.float32)

        fused = bottleneck(video, audio)

        # Pool to get embeddings for contrastive learning
        pooled = fused.mean(axis=1)  # (4, 512)

        assert pooled.shape == (4, 512)
        assert np.all(np.isfinite(pooled))
