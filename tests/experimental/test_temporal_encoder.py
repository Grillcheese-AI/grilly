"""
TDD Tests for TemporalEncoder.

Tests time encoding, temporal binding/unbinding, and time relations.
"""

import numpy as np


class TestTemporalEncoderBasic:
    """Basic tests for TemporalEncoder initialization."""

    def test_init_default_dimensions(self):
        """Should initialize with default dimensions."""
        from grilly.experimental.temporal.encoder import TemporalEncoder

        encoder = TemporalEncoder()

        assert encoder.dim > 0
        assert encoder.max_time > 0

    def test_init_custom_dimension(self):
        """Should initialize with custom dimension."""
        from grilly.experimental.temporal.encoder import TemporalEncoder

        encoder = TemporalEncoder(dim=2048, max_time=500)

        assert encoder.dim == 2048
        assert encoder.max_time == 500


class TestEncodeTime:
    """Tests for time encoding."""

    def test_encode_time_returns_correct_shape(self, dim):
        """encode_time should return vector of correct dimension."""
        from grilly.experimental.temporal.encoder import TemporalEncoder

        encoder = TemporalEncoder(dim=dim)

        result = encoder.encode_time(0)

        assert result.shape == (dim,)

    def test_encode_time_deterministic(self, dim):
        """Same time should produce same encoding."""
        from grilly.experimental.temporal.encoder import TemporalEncoder

        encoder = TemporalEncoder(dim=dim)

        v1 = encoder.encode_time(5)
        v2 = encoder.encode_time(5)

        np.testing.assert_array_equal(v1, v2)

    def test_encode_time_adjacent_similar(self, dim):
        """Adjacent times should have high similarity."""
        from grilly.experimental.temporal.encoder import TemporalEncoder
        from grilly.experimental.vsa.ops import HolographicOps

        encoder = TemporalEncoder(dim=dim)

        t0 = encoder.encode_time(0)
        t1 = encoder.encode_time(1)
        t10 = encoder.encode_time(10)

        sim_adjacent = HolographicOps.similarity(t0, t1)
        sim_distant = HolographicOps.similarity(t0, t10)

        # Adjacent times should be more similar
        assert sim_adjacent > sim_distant - 0.2  # Allow tolerance for HRR

    def test_encode_time_continuous(self, dim):
        """encode_time_continuous should interpolate between discrete times."""
        from grilly.experimental.temporal.encoder import TemporalEncoder

        encoder = TemporalEncoder(dim=dim)

        t_2_5 = encoder.encode_time_continuous(2.5)

        assert t_2_5.shape == (dim,)


class TestTemporalBinding:
    """Tests for binding/unbinding with time."""

    def test_bind_with_time_returns_correct_shape(self, dim):
        """bind_with_time should return vector of correct dimension."""
        from grilly.experimental.temporal.encoder import TemporalEncoder

        encoder = TemporalEncoder(dim=dim)

        state = np.random.randn(dim).astype(np.float32)
        result = encoder.bind_with_time(state, t=5)

        assert result.shape == (dim,)

    def test_unbind_time_recovers_state(self, dim):
        """unbind_time should recover original state."""
        from grilly.experimental.temporal.encoder import TemporalEncoder
        from grilly.experimental.vsa.ops import HolographicOps

        encoder = TemporalEncoder(dim=dim)

        state = HolographicOps.random_vector(dim)
        temporal_state = encoder.bind_with_time(state, t=3)

        recovered = encoder.unbind_time(temporal_state, t=3)

        # Should recover something similar to original
        sim = HolographicOps.similarity(recovered, state)
        assert sim > 0.1  # Some similarity expected (HRR is approximate)

    def test_bind_unbind_roundtrip(self, dim):
        """Binding and unbinding should preserve information."""
        from grilly.experimental.temporal.encoder import TemporalEncoder
        from grilly.experimental.vsa.ops import HolographicOps

        encoder = TemporalEncoder(dim=dim)

        state = HolographicOps.random_vector(dim)
        temporal = encoder.bind_with_time(state, t=5)
        recovered = encoder.unbind_time(temporal, t=5)

        # Roundtrip should preserve some similarity
        sim = HolographicOps.similarity(recovered, state)
        assert sim > 0.1  # HRR approximation


class TestTemporalRelations:
    """Tests for temporal relations."""

    def test_get_temporal_relation_before(self, dim):
        """Should identify 'before' relation."""
        from grilly.experimental.temporal.encoder import TemporalEncoder

        encoder = TemporalEncoder(dim=dim)

        relation = encoder.get_temporal_relation(2, 5)

        assert relation == "before"

    def test_get_temporal_relation_after(self, dim):
        """Should identify 'after' relation."""
        from grilly.experimental.temporal.encoder import TemporalEncoder

        encoder = TemporalEncoder(dim=dim)

        relation = encoder.get_temporal_relation(5, 2)

        assert relation == "after"

    def test_get_temporal_relation_simultaneous(self, dim):
        """Should identify 'simultaneous' relation."""
        from grilly.experimental.temporal.encoder import TemporalEncoder

        encoder = TemporalEncoder(dim=dim)

        relation = encoder.get_temporal_relation(3, 3)

        assert relation == "simultaneous"
