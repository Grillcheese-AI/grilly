"""Tests for ProjectionHeads multi-head subspace projection module."""

import numpy as np
import pytest
from grilly.nn.projection_heads import ProjectionHeads


class TestProjectionHeadsForward:
    """Forward pass shape and type checks."""

    def test_default_output_shape_batched(self):
        """Default heads produce (batch, 640) output."""
        heads = ProjectionHeads(768)
        x = np.random.randn(4, 768).astype(np.float32)
        out = heads(x)
        assert out.shape == (4, 640)

    def test_default_output_shape_single(self):
        """Default heads handle a single (768,) vector."""
        heads = ProjectionHeads(768)
        x = np.random.randn(768).astype(np.float32)
        out = heads(x)
        assert out.shape == (640,)

    def test_output_dim_attribute(self):
        """output_dim matches sum of head dims."""
        heads = ProjectionHeads(768)
        assert heads.output_dim == 256 + 128 + 128 + 128  # 640

    def test_output_dtype_float32(self):
        """Output is float32."""
        heads = ProjectionHeads(768)
        x = np.random.randn(2, 768).astype(np.float32)
        out = heads(x)
        assert out.dtype == np.float32


class TestGetSubspace:
    """Tests for subspace slicing."""

    def test_semantic_subspace_shape(self):
        """Semantic head extracts the first 256 dims."""
        heads = ProjectionHeads(768)
        x = np.random.randn(4, 768).astype(np.float32)
        emb = heads(x)
        semantic = heads.get_subspace(emb, "semantic")
        assert semantic.shape == (4, 256)

    def test_emotion_subspace_shape(self):
        """Emotion head extracts 128-dim slice."""
        heads = ProjectionHeads(768)
        x = np.random.randn(4, 768).astype(np.float32)
        emb = heads(x)
        emotion = heads.get_subspace(emb, "emotion")
        assert emotion.shape == (4, 128)

    def test_intent_subspace_shape(self):
        """Intent head extracts 128-dim slice."""
        heads = ProjectionHeads(768)
        x = np.random.randn(4, 768).astype(np.float32)
        emb = heads(x)
        intent = heads.get_subspace(emb, "intent")
        assert intent.shape == (4, 128)

    def test_context_subspace_shape(self):
        """Context head extracts the last 128 dims."""
        heads = ProjectionHeads(768)
        x = np.random.randn(4, 768).astype(np.float32)
        emb = heads(x)
        context = heads.get_subspace(emb, "context")
        assert context.shape == (4, 128)

    def test_subspace_is_correct_slice(self):
        """get_subspace returns the exact slice from the structured embedding."""
        heads = ProjectionHeads(768)
        x = np.random.randn(3, 768).astype(np.float32)
        emb = heads(x)

        # semantic occupies [0:256]
        np.testing.assert_array_equal(heads.get_subspace(emb, "semantic"), emb[:, 0:256])
        # emotion occupies [256:384]
        np.testing.assert_array_equal(heads.get_subspace(emb, "emotion"), emb[:, 256:384])
        # intent occupies [384:512]
        np.testing.assert_array_equal(heads.get_subspace(emb, "intent"), emb[:, 384:512])
        # context occupies [512:640]
        np.testing.assert_array_equal(heads.get_subspace(emb, "context"), emb[:, 512:640])

    def test_unknown_head_raises_key_error(self):
        """get_subspace raises KeyError for unregistered head names."""
        heads = ProjectionHeads(768)
        x = np.random.randn(2, 768).astype(np.float32)
        emb = heads(x)
        with pytest.raises(KeyError, match="unknown_head"):
            heads.get_subspace(emb, "unknown_head")


class TestGetAllSubspaces:
    """Tests for get_all_subspaces."""

    def test_returns_all_heads(self):
        """get_all_subspaces returns a dict with all four default heads."""
        heads = ProjectionHeads(768)
        x = np.random.randn(4, 768).astype(np.float32)
        emb = heads(x)
        subspaces = heads.get_all_subspaces(emb)
        assert set(subspaces.keys()) == {"semantic", "emotion", "intent", "context"}

    def test_all_subspace_shapes(self):
        """Each subspace has the expected dimension."""
        heads = ProjectionHeads(768)
        x = np.random.randn(4, 768).astype(np.float32)
        emb = heads(x)
        subspaces = heads.get_all_subspaces(emb)
        assert subspaces["semantic"].shape == (4, 256)
        assert subspaces["emotion"].shape == (4, 128)
        assert subspaces["intent"].shape == (4, 128)
        assert subspaces["context"].shape == (4, 128)

    def test_all_subspaces_cover_full_embedding(self):
        """Concatenating all subspaces reconstructs the full embedding."""
        heads = ProjectionHeads(768)
        x = np.random.randn(4, 768).astype(np.float32)
        emb = heads(x)
        subspaces = heads.get_all_subspaces(emb)
        reconstructed = np.concatenate(list(subspaces.values()), axis=-1)
        np.testing.assert_array_equal(reconstructed, emb)


class TestSubspaceSimilarity:
    """Tests for cosine similarity in subspaces."""

    def test_self_similarity_is_one(self):
        """Cosine similarity of a vector with itself is 1."""
        heads = ProjectionHeads(768)
        x = np.random.randn(4, 768).astype(np.float32)
        emb = heads(x)
        sims = heads.subspace_similarity(emb, emb, "semantic")
        np.testing.assert_allclose(sims, 1.0, atol=1e-5)

    def test_similarity_range(self):
        """Cosine similarity is in [-1, 1]."""
        np.random.seed(0)
        heads = ProjectionHeads(768)
        a = np.random.randn(16, 768).astype(np.float32)
        b = np.random.randn(16, 768).astype(np.float32)
        ea = heads(a)
        eb = heads(b)
        for head_name in ["semantic", "emotion", "intent", "context"]:
            sims = heads.subspace_similarity(ea, eb, head_name)
            assert sims.shape == (16,)
            assert np.all(sims >= -1.0 - 1e-6)
            assert np.all(sims <= 1.0 + 1e-6)

    def test_similarity_output_shape(self):
        """subspace_similarity returns a (batch,) array."""
        heads = ProjectionHeads(768)
        x = np.random.randn(8, 768).astype(np.float32)
        y = np.random.randn(8, 768).astype(np.float32)
        ex = heads(x)
        ey = heads(y)
        sims = heads.subspace_similarity(ex, ey, "emotion")
        assert sims.shape == (8,)


class TestCustomHeadConfigs:
    """Tests for non-default head configurations."""

    def test_custom_two_heads(self):
        """Custom two-head config produces correct output dim and subspaces."""
        configs = {"meaning": 512, "tone": 64}
        heads = ProjectionHeads(256, head_configs=configs)

        assert heads.output_dim == 576
        x = np.random.randn(3, 256).astype(np.float32)
        emb = heads(x)
        assert emb.shape == (3, 576)
        assert heads.get_subspace(emb, "meaning").shape == (3, 512)
        assert heads.get_subspace(emb, "tone").shape == (3, 64)

    def test_single_head(self):
        """Single-head config works and is a no-op projection."""
        configs = {"all": 128}
        heads = ProjectionHeads(64, head_configs=configs)

        assert heads.output_dim == 128
        x = np.random.randn(5, 64).astype(np.float32)
        emb = heads(x)
        assert emb.shape == (5, 128)
        np.testing.assert_array_equal(heads.get_subspace(emb, "all"), emb)

    def test_custom_configs_preserved(self):
        """head_configs and offsets reflect the custom config."""
        configs = {"a": 32, "b": 64, "c": 16}
        heads = ProjectionHeads(128, head_configs=configs)

        assert heads.head_configs == configs
        assert heads._offsets["a"] == (0, 32)
        assert heads._offsets["b"] == (32, 96)
        assert heads._offsets["c"] == (96, 112)

    def test_repr(self):
        """__repr__ includes input/output dims and head names."""
        heads = ProjectionHeads(768)
        r = repr(heads)
        assert "768" in r
        assert "640" in r
        assert "semantic" in r


class TestModuleIntegration:
    """Checks that ProjectionHeads integrates with the Module base class."""

    def test_parameters_registered(self):
        """All linear head parameters appear in parameters()."""
        heads = ProjectionHeads(768)
        params = list(heads.parameters())
        # 4 heads * 2 params (weight + bias) = 8
        assert len(params) == 8

    def test_submodules_registered(self):
        """Each head is accessible as a submodule."""
        heads = ProjectionHeads(768)
        for name in ["semantic", "emotion", "intent", "context"]:
            assert name in heads._modules

    def test_train_eval_toggle(self):
        """train() and eval() propagate to submodules without error."""
        heads = ProjectionHeads(768)
        heads.eval()
        assert not heads.training
        heads.train()
        assert heads.training


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
