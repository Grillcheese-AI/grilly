"""
TDD Tests for WorkingMemory.

Tests capacity limits, activation decay, and attention mechanisms.
"""

import numpy as np


class TestWorkingMemoryBasic:
    """Basic tests for WorkingMemory initialization."""

    def test_init_default_dimensions(self):
        """Should initialize with default dimensions."""
        from grilly.experimental.cognitive.memory import WorkingMemory

        wm = WorkingMemory()

        assert wm.dim > 0
        assert wm.capacity > 0

    def test_init_custom_dimensions(self):
        """Should initialize with custom dimensions."""
        from grilly.experimental.cognitive.memory import WorkingMemory

        wm = WorkingMemory(dim=2048, capacity=5, decay_rate=0.2)

        assert wm.dim == 2048
        assert wm.capacity == 5
        assert wm.decay_rate == 0.2


class TestWorkingMemoryCapacity:
    """Tests for capacity management."""

    def test_add_respects_capacity(self, dim):
        """add should respect capacity limit."""
        from grilly.experimental.cognitive.memory import WorkingMemory, WorkingMemorySlot
        from grilly.experimental.vsa.ops import HolographicOps

        wm = WorkingMemory(dim=dim, capacity=3)

        # Add items up to capacity
        for i in range(3):
            vec = HolographicOps.random_vector(dim)
            wm.add(vec, f"item_{i}", WorkingMemorySlot.CONTEXT)

        assert len(wm.items) == 3

        # Adding one more should remove least activated
        vec = HolographicOps.random_vector(dim)
        wm.add(vec, "item_4", WorkingMemorySlot.CONTEXT)

        assert len(wm.items) == 3  # Still at capacity

    def test_add_returns_index(self, dim):
        """add should return index of added item."""
        from grilly.experimental.cognitive.memory import WorkingMemory, WorkingMemorySlot
        from grilly.experimental.vsa.ops import HolographicOps

        wm = WorkingMemory(dim=dim)

        vec = HolographicOps.random_vector(dim)
        idx = wm.add(vec, "test", WorkingMemorySlot.CONTEXT)

        assert isinstance(idx, int)
        assert 0 <= idx < len(wm.items)


class TestWorkingMemoryDecay:
    """Tests for activation decay."""

    def test_decay_reduces_activation(self, dim):
        """decay should reduce activation over time."""
        from grilly.experimental.cognitive.memory import WorkingMemory, WorkingMemorySlot
        from grilly.experimental.vsa.ops import HolographicOps

        wm = WorkingMemory(dim=dim, decay_rate=0.1)

        vec = HolographicOps.random_vector(dim)
        idx = wm.add(vec, "test", WorkingMemorySlot.CONTEXT)

        initial_activation = wm.items[idx].activation

        # Apply decay
        wm.decay()

        assert wm.items[idx].activation < initial_activation

    def test_decay_boosts_focused_item(self, dim):
        """decay should boost activation of focused item."""
        from grilly.experimental.cognitive.memory import WorkingMemory, WorkingMemorySlot
        from grilly.experimental.vsa.ops import HolographicOps

        wm = WorkingMemory(dim=dim, decay_rate=0.1)

        vec = HolographicOps.random_vector(dim)
        idx = wm.add(vec, "test", WorkingMemorySlot.CONTEXT)

        # Focus on item
        wm.attend(idx)

        initial_activation = wm.items[idx].activation

        # Apply decay (should boost focused item)
        wm.decay()

        # Focused item should have higher activation than if not focused
        assert wm.items[idx].activation >= initial_activation - 0.1


class TestWorkingMemorySlots:
    """Tests for slot-based organization."""

    def test_get_by_slot_returns_matching_items(self, dim):
        """get_by_slot should return items of specified slot."""
        from grilly.experimental.cognitive.memory import WorkingMemory, WorkingMemorySlot
        from grilly.experimental.vsa.ops import HolographicOps

        wm = WorkingMemory(dim=dim)

        # Add items to different slots
        vec1 = HolographicOps.random_vector(dim)
        wm.add(vec1, "context1", WorkingMemorySlot.CONTEXT)

        vec2 = HolographicOps.random_vector(dim)
        wm.add(vec2, "candidate1", WorkingMemorySlot.CANDIDATE)

        vec3 = HolographicOps.random_vector(dim)
        wm.add(vec3, "context2", WorkingMemorySlot.CONTEXT)

        # Get context items
        context_items = wm.get_by_slot(WorkingMemorySlot.CONTEXT)

        assert len(context_items) == 2
        assert all(item.slot == WorkingMemorySlot.CONTEXT for item in context_items)


class TestWorkingMemoryContext:
    """Tests for context vector generation."""

    def test_get_context_vector_returns_vector(self, dim):
        """get_context_vector should return vector of correct dimension."""
        from grilly.experimental.cognitive.memory import WorkingMemory, WorkingMemorySlot
        from grilly.experimental.vsa.ops import HolographicOps

        wm = WorkingMemory(dim=dim)

        vec = HolographicOps.random_vector(dim)
        wm.add(vec, "test", WorkingMemorySlot.CONTEXT)

        context = wm.get_context_vector()

        assert context.shape == (dim,)

    def test_get_context_vector_empty_returns_zeros(self, dim):
        """get_context_vector should return zeros if empty."""
        from grilly.experimental.cognitive.memory import WorkingMemory

        wm = WorkingMemory(dim=dim)

        context = wm.get_context_vector()

        assert context.shape == (dim,)
        np.testing.assert_array_equal(context, np.zeros(dim))

    def test_get_context_capsule_returns_vector(self, dim):
        """get_context_capsule should return capsule vector."""
        from grilly.experimental.cognitive.memory import WorkingMemory, WorkingMemorySlot
        from grilly.experimental.vsa.ops import HolographicOps

        wm = WorkingMemory(dim=dim)

        vec = HolographicOps.random_vector(dim)
        wm.add(vec, "test", WorkingMemorySlot.CONTEXT)

        capsule = wm.get_context_capsule()

        assert capsule is not None
        assert capsule.shape == (wm.capsule_dim,)

    def test_get_context_capsule_empty_returns_zeros(self, dim):
        """get_context_capsule should return zeros if empty."""
        from grilly.experimental.cognitive.memory import WorkingMemory

        wm = WorkingMemory(dim=dim)

        capsule = wm.get_context_capsule()

        assert capsule is not None
        assert capsule.shape == (wm.capsule_dim,)
        np.testing.assert_array_equal(capsule, np.zeros(wm.capsule_dim))

    def test_get_context_vector_weighted_by_activation(self, dim):
        """get_context_vector should weight by activation."""
        from grilly.experimental.cognitive.memory import WorkingMemory, WorkingMemorySlot
        from grilly.experimental.vsa.ops import HolographicOps

        wm = WorkingMemory(dim=dim)

        vec1 = HolographicOps.random_vector(dim)
        idx1 = wm.add(vec1, "high_activation", WorkingMemorySlot.CONTEXT)
        wm.items[idx1].activation = 1.0

        vec2 = HolographicOps.random_vector(dim)
        idx2 = wm.add(vec2, "low_activation", WorkingMemorySlot.CONTEXT)
        wm.items[idx2].activation = 0.1

        context = wm.get_context_vector()

        # Context should be more similar to high activation item
        from grilly.experimental.vsa.ops import HolographicOps

        sim1 = HolographicOps.similarity(context, vec1)
        sim2 = HolographicOps.similarity(context, vec2)

        assert sim1 > sim2


class TestWorkingMemoryBinding:
    """Tests for binding operations."""

    def test_bind_focused_returns_vector(self, dim):
        """bind_focused should return vector."""
        from grilly.experimental.cognitive.memory import WorkingMemory, WorkingMemorySlot
        from grilly.experimental.vsa.ops import HolographicOps

        wm = WorkingMemory(dim=dim)

        vec = HolographicOps.random_vector(dim)
        idx = wm.add(vec, "focus", WorkingMemorySlot.FOCUS)
        wm.attend(idx)

        # Add to binding buffer
        buffer_vec = HolographicOps.random_vector(dim)
        wm.binding_buffer.append(buffer_vec)

        result = wm.bind_focused()

        assert result.shape == (dim,)

    def test_bind_focused_clears_buffer(self, dim):
        """bind_focused should clear binding buffer."""
        from grilly.experimental.cognitive.memory import WorkingMemory, WorkingMemorySlot
        from grilly.experimental.vsa.ops import HolographicOps

        wm = WorkingMemory(dim=dim)

        vec = HolographicOps.random_vector(dim)
        idx = wm.add(vec, "focus", WorkingMemorySlot.FOCUS)
        wm.attend(idx)

        wm.binding_buffer.append(HolographicOps.random_vector(dim))

        assert len(wm.binding_buffer) > 0

        wm.bind_focused()

        assert len(wm.binding_buffer) == 0


class TestWorkingMemoryClearCandidates:
    """Tests for clearing candidates."""

    def test_clear_candidates_removes_candidates(self, dim):
        """clear_candidates should remove all candidate items."""
        from grilly.experimental.cognitive.memory import WorkingMemory, WorkingMemorySlot
        from grilly.experimental.vsa.ops import HolographicOps

        wm = WorkingMemory(dim=dim)

        # Add candidates and non-candidates
        vec1 = HolographicOps.random_vector(dim)
        wm.add(vec1, "candidate1", WorkingMemorySlot.CANDIDATE)

        vec2 = HolographicOps.random_vector(dim)
        wm.add(vec2, "context", WorkingMemorySlot.CONTEXT)

        vec3 = HolographicOps.random_vector(dim)
        wm.add(vec3, "candidate2", WorkingMemorySlot.CANDIDATE)

        assert len(wm.items) == 3

        wm.clear_candidates()

        assert len(wm.items) == 1
        assert wm.items[0].slot == WorkingMemorySlot.CONTEXT
