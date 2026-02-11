"""
TDD Tests for RelationalEncoder.

Tests the ability to encode entities, relationships, and extract relations.
"""

import numpy as np


class TestRelationalEncoderBasic:
    """Basic tests for RelationalEncoder initialization."""

    def test_init_default_dimensions(self):
        """Should initialize with default dimensions."""
        from grilly.experimental.moe.relational import RelationalEncoder

        encoder = RelationalEncoder()

        assert encoder.dim > 0
        assert hasattr(encoder, "encode")

    def test_init_custom_dimension(self):
        """Should initialize with custom dimension."""
        from grilly.experimental.moe.relational import RelationalEncoder

        encoder = RelationalEncoder(dim=2048)

        assert encoder.dim == 2048


class TestRelationalEncode:
    """Tests for entity encoding."""

    def test_encode_returns_correct_shape(self, dim):
        """Encode should return vector of correct dimension."""
        from grilly.experimental.moe.relational import RelationalEncoder

        encoder = RelationalEncoder(dim=dim)

        result = encoder.encode("concept_name")

        assert result.shape == (dim,)

    def test_encode_deterministic(self, dim):
        """Same input should produce same encoding."""
        from grilly.experimental.moe.relational import RelationalEncoder

        encoder = RelationalEncoder(dim=dim)

        v1 = encoder.encode("test_concept")
        v2 = encoder.encode("test_concept")

        np.testing.assert_array_equal(v1, v2)

    def test_encode_different_inputs_different_vectors(self, dim):
        """Different inputs should produce different encodings."""
        from grilly.experimental.moe.relational import RelationalEncoder

        encoder = RelationalEncoder(dim=dim)

        v1 = encoder.encode("concept_a")
        v2 = encoder.encode("concept_b")

        assert not np.array_equal(v1, v2)

    def test_encode_with_modality(self, dim):
        """Should encode with modality tag."""
        from grilly.experimental.moe.relational import RelationalEncoder

        encoder = RelationalEncoder(dim=dim)

        v_text = encoder.encode("word", modality="text")
        v_image = encoder.encode("word", modality="image")

        # Same concept, different modalities should be different
        assert not np.array_equal(v_text, v_image)

    def test_encode_with_polarity(self, dim):
        """Should encode with polarity (positive/negative)."""
        from grilly.experimental.moe.relational import RelationalEncoder

        encoder = RelationalEncoder(dim=dim)

        v_pos = encoder.encode("concept", polarity="positive")
        v_neg = encoder.encode("concept", polarity="negative")

        assert not np.array_equal(v_pos, v_neg)


class TestRelationalOpposite:
    """Tests for getting opposite/negated concepts."""

    def test_get_opposite_flips_polarity(self, dim):
        """get_opposite should flip polarity."""
        from grilly.experimental.moe.relational import RelationalEncoder

        encoder = RelationalEncoder(dim=dim)

        v_original = encoder.encode("concept", polarity="positive")
        v_opposite = encoder.get_opposite(v_original)

        # Opposite of positive should be negative
        # They should have negative similarity
        from grilly.experimental.vsa.ops import BinaryOps

        sim = BinaryOps.similarity(v_original, v_opposite)
        assert sim < 0, f"Expected negative similarity, got {sim}"

    def test_double_opposite_returns_original(self, dim):
        """Applying opposite twice should return to original."""
        from grilly.experimental.moe.relational import RelationalEncoder

        encoder = RelationalEncoder(dim=dim)

        v_original = encoder.encode("concept")
        v_opposite = encoder.get_opposite(v_original)
        v_double = encoder.get_opposite(v_opposite)

        np.testing.assert_array_almost_equal(v_original, v_double)


class TestRelationalExtractRelation:
    """Tests for extracting transformation between entities."""

    def test_extract_relation_basic(self, dim):
        """Should extract the transformation from A to B."""
        from grilly.experimental.moe.relational import RelationalEncoder

        encoder = RelationalEncoder(dim=dim)

        v_a = encoder.encode("entity_a")
        v_b = encoder.encode("entity_b")

        relation = encoder.extract_relation(v_a, v_b)

        assert relation.shape == (dim,)

    def test_extract_relation_can_apply(self, dim):
        """Extracted relation when applied to A should give B."""
        from grilly.experimental.moe.relational import RelationalEncoder
        from grilly.experimental.vsa.ops import BinaryOps

        encoder = RelationalEncoder(dim=dim)

        v_a = encoder.encode("entity_a")
        v_b = encoder.encode("entity_b")

        relation = encoder.extract_relation(v_a, v_b)

        # Applying relation to A should give something similar to B
        v_b_recovered = encoder.apply_relation(v_a, relation)

        sim = BinaryOps.similarity(v_b_recovered, v_b)
        assert sim > 0.9, f"Expected high similarity, got {sim}"

    def test_analogy_king_queen_man_woman(self, dim):
        """Should solve analogy: king:queen :: man:? => woman."""
        from grilly.experimental.moe.relational import RelationalEncoder
        from grilly.experimental.vsa.ops import BinaryOps

        encoder = RelationalEncoder(dim=dim)

        king = encoder.encode("king")
        queen = encoder.encode("queen")
        man = encoder.encode("man")
        woman = encoder.encode("woman")

        # Extract king->queen relation
        gender_relation = encoder.extract_relation(king, queen)

        # Apply to man
        predicted = encoder.apply_relation(man, gender_relation)

        # Should be closer to woman than to king or man
        sim_woman = BinaryOps.similarity(predicted, woman)
        sim_king = BinaryOps.similarity(predicted, king)
        sim_man = BinaryOps.similarity(predicted, man)

        assert sim_woman > sim_king
        assert sim_woman > sim_man


class TestRelationalBatch:
    """Tests for batch operations."""

    def test_encode_batch(self, dim):
        """Should encode multiple items efficiently."""
        from grilly.experimental.moe.relational import RelationalEncoder

        encoder = RelationalEncoder(dim=dim)

        items = ["item_a", "item_b", "item_c"]

        result = encoder.encode_batch(items)

        assert result.shape == (3, dim)

    def test_similarity_batch(self, dim):
        """Should compute similarities against codebook."""
        from grilly.experimental.moe.relational import RelationalEncoder

        encoder = RelationalEncoder(dim=dim)

        query = encoder.encode("query_item")
        codebook = encoder.encode_batch(["item_a", "item_b", "item_c"])

        similarities = encoder.similarity_batch(query, codebook)

        assert similarities.shape == (3,)
        assert all(-1 <= s <= 1 for s in similarities)
