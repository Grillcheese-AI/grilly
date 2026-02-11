"""
TDD Tests for ResonatorMoE.

Tests the resonator-based mixture of experts routing mechanism.
"""

import numpy as np
import pytest


class TestResonatorMoEBasic:
    """Basic tests for ResonatorMoE initialization."""

    def test_init_with_experts(self, dim):
        """Should initialize with expert functions."""
        from grilly.experimental.moe.routing import ResonatorMoE

        # Simple expert functions
        def expert_a(x):
            return x * 2

        def expert_b(x):
            return x + 1

        moe = ResonatorMoE(dim=dim, experts={"expert_a": expert_a, "expert_b": expert_b})

        assert len(moe.experts) == 2

    def test_init_registers_expert_vectors(self, dim):
        """Should create expert vectors for routing."""
        from grilly.experimental.moe.routing import ResonatorMoE

        def expert_a(x):
            return x

        def expert_b(x):
            return x

        moe = ResonatorMoE(dim=dim, experts={"expert_a": expert_a, "expert_b": expert_b})

        assert "expert_a" in moe.expert_vectors
        assert "expert_b" in moe.expert_vectors
        assert moe.expert_vectors["expert_a"].shape == (dim,)


class TestResonatorMoERouting:
    """Tests for expert selection via resonator routing."""

    def test_route_returns_expert_names(self, dim):
        """route should return selected expert names."""
        from grilly.experimental.moe.routing import ResonatorMoE

        def expert_a(x):
            return x

        def expert_b(x):
            return x

        moe = ResonatorMoE(dim=dim, experts={"expert_a": expert_a, "expert_b": expert_b})

        # Create a query
        query = np.random.randn(dim).astype(np.float32)

        selected = moe.route(query, top_k=1)

        assert len(selected) == 1
        assert selected[0] in ["expert_a", "expert_b"]

    def test_route_top_k_experts(self, dim):
        """Should select top_k experts."""
        from grilly.experimental.moe.routing import ResonatorMoE

        experts = {f"expert_{i}": lambda x, i=i: x + i for i in range(5)}

        moe = ResonatorMoE(dim=dim, experts=experts)

        query = np.random.randn(dim).astype(np.float32)

        selected = moe.route(query, top_k=3)

        assert len(selected) == 3
        for name in selected:
            assert name in experts

    def test_route_with_threshold(self, dim):
        """Should only select experts above similarity threshold."""
        from grilly.experimental.moe.routing import ResonatorMoE

        def expert_a(x):
            return x

        def expert_b(x):
            return x

        moe = ResonatorMoE(dim=dim, experts={"expert_a": expert_a, "expert_b": expert_b})

        # Create query similar to expert_a's vector
        query = moe.expert_vectors["expert_a"].copy()

        selected = moe.route(query, threshold=0.9)

        # Should at least select the matching expert
        assert "expert_a" in selected


class TestResonatorMoEForward:
    """Tests for forward pass through MoE."""

    def test_forward_returns_correct_shape(self, dim):
        """Forward should return output of correct shape."""
        from grilly.experimental.moe.routing import ResonatorMoE

        # Experts that preserve shape
        def expert_a(x):
            return x * 2

        def expert_b(x):
            return x + 0.5

        moe = ResonatorMoE(dim=dim, experts={"expert_a": expert_a, "expert_b": expert_b})

        x = np.random.randn(dim).astype(np.float32)
        query = x.copy()

        output = moe.forward(x, query, top_k=1)

        assert output.shape == (dim,)

    def test_forward_uses_selected_expert(self, dim):
        """Forward should apply selected expert."""
        from grilly.experimental.moe.routing import ResonatorMoE

        # Distinguishable experts
        def expert_double(x):
            return x * 2

        def expert_zero(x):
            return np.zeros_like(x)

        moe = ResonatorMoE(dim=dim, experts={"double": expert_double, "zero": expert_zero})

        x = np.ones(dim, dtype=np.float32)

        # Force route to "double" expert
        query = moe.expert_vectors["double"].copy()

        output = moe.forward(x, query, top_k=1)

        # Output should be doubled
        expected = x * 2
        np.testing.assert_array_almost_equal(output, expected)

    def test_forward_combines_multiple_experts(self, dim):
        """Forward with top_k > 1 should combine expert outputs."""
        from grilly.experimental.moe.routing import ResonatorMoE

        def expert_a(x):
            return x * 2

        def expert_b(x):
            return x * 3

        moe = ResonatorMoE(dim=dim, experts={"a": expert_a, "b": expert_b})

        x = np.ones(dim, dtype=np.float32)
        query = np.random.randn(dim).astype(np.float32)

        output = moe.forward(x, query, top_k=2)

        # Output should be some combination of 2x and 3x
        # (exact combination depends on routing weights)
        assert output.shape == (dim,)


class TestResonatorMoECompositional:
    """Tests for compositional/structured routing."""

    def test_compositional_query_selects_multiple(self, dim):
        """Query composed of multiple expert concepts should select both."""
        from grilly.experimental.moe.routing import ResonatorMoE
        from grilly.experimental.vsa.ops import BinaryOps

        def expert_a(x):
            return x

        def expert_b(x):
            return x

        moe = ResonatorMoE(dim=dim, experts={"expert_a": expert_a, "expert_b": expert_b})

        # Bundle both expert vectors
        query = BinaryOps.bundle([moe.expert_vectors["expert_a"], moe.expert_vectors["expert_b"]])

        selected = moe.route(query, top_k=2)

        # Should select both experts
        assert "expert_a" in selected
        assert "expert_b" in selected


class TestResonatorMoEWeights:
    """Tests for expert weighting."""

    def test_get_weights_returns_dict(self, dim):
        """get_weights should return expert->weight mapping."""
        from grilly.experimental.moe.routing import ResonatorMoE

        def expert_a(x):
            return x

        def expert_b(x):
            return x

        moe = ResonatorMoE(dim=dim, experts={"expert_a": expert_a, "expert_b": expert_b})

        query = np.random.randn(dim).astype(np.float32)

        weights = moe.get_weights(query)

        assert isinstance(weights, dict)
        assert "expert_a" in weights
        assert "expert_b" in weights
        # Weights should be non-negative
        assert all(w >= 0 for w in weights.values())

    def test_get_weights_softmax_normalized(self, dim):
        """Weights should be softmax-normalized (sum to 1)."""
        from grilly.experimental.moe.routing import ResonatorMoE

        experts = {f"expert_{i}": lambda x: x for i in range(5)}

        moe = ResonatorMoE(dim=dim, experts=experts)

        query = np.random.randn(dim).astype(np.float32)

        weights = moe.get_weights(query, normalize=True)

        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=1e-5)


class TestResonatorMoECapsuleRouting:
    """Tests for capsule-aware routing."""

    def test_capsule_weight_can_drive_routing(self, dim):
        """Capsule similarity should influence routing when weighted."""
        from grilly.experimental.cognitive.capsule import CapsuleEncoder
        from grilly.experimental.moe.routing import ResonatorMoE

        def expert_a(x):
            return x

        def expert_b(x):
            return x

        encoder = CapsuleEncoder(input_dim=dim)
        query = np.random.randn(dim).astype(np.float32)
        query_capsule = encoder.encode_vector(query)

        expert_capsules = {"expert_a": query_capsule, "expert_b": -query_capsule}

        moe = ResonatorMoE(
            dim=dim,
            experts={"expert_a": expert_a, "expert_b": expert_b},
            expert_capsules=expert_capsules,
            capsule_encoder=encoder,
            capsule_weight=1.0,
        )

        selected = moe.route(query, top_k=1)
        assert selected[0] == "expert_a"


class TestRelationalMoE:
    """Tests for RelationalMoE with structured expert codebook."""

    def test_relational_moe_init(self, dim):
        """Should initialize RelationalMoE with relational expert keys."""
        from grilly.experimental.moe.routing import RelationalMoE

        def expert_add(x):
            return x + 1

        def expert_mult(x):
            return x * 2

        moe = RelationalMoE(
            dim=dim,
            experts={"add": expert_add, "mult": expert_mult},
            expert_relations={"add": ("input", "sum"), "mult": ("input", "product")},
        )

        assert len(moe.experts) == 2

    def test_relational_routing_by_concept(self, dim):
        """Should route based on relational concept queries."""
        from grilly.experimental.moe.relational import RelationalEncoder
        from grilly.experimental.moe.routing import RelationalMoE

        def expert_add(x):
            return x + 1

        def expert_mult(x):
            return x * 2

        moe = RelationalMoE(
            dim=dim,
            experts={"add": expert_add, "mult": expert_mult},
            expert_relations={"add": ("input", "sum"), "mult": ("input", "product")},
        )

        # Query for "product" concept
        encoder = RelationalEncoder(dim=dim)
        query = encoder.encode("product")

        selected = moe.route(query, top_k=1)

        # Should select mult expert (associated with product relation)
        assert "mult" in selected
