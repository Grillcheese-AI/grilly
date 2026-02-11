"""
TDD Tests for CounterfactualReasoner.

Tests counterfactual intervention and reasoning.
"""


class TestCounterfactualReasonerBasic:
    """Basic tests for CounterfactualReasoner initialization."""

    def test_init_with_world_model(self, dim):
        """Should initialize with TemporalWorldModel."""
        from grilly.experimental.temporal.counterfactual import CounterfactualReasoner
        from grilly.experimental.temporal.state import TemporalWorldModel

        world = TemporalWorldModel(dim=dim)
        reasoner = CounterfactualReasoner(world)

        assert reasoner.world is world
        assert reasoner.dim == dim


class TestIntervene:
    """Tests for counterfactual intervention."""

    def test_intervene_creates_branch(self, dim):
        """intervene should create a counterfactual branch."""
        from grilly.experimental.temporal.counterfactual import CounterfactualReasoner
        from grilly.experimental.temporal.state import TemporalWorldModel

        world = TemporalWorldModel(dim=dim)
        reasoner = CounterfactualReasoner(world)

        # Set up timeline
        world.set_state(0, {"has_umbrella": True})
        world.set_state(1, {"has_umbrella": False})

        # Intervene: keep umbrella at t=1
        branch_id = reasoner.intervene(1, "has_umbrella", True)

        assert branch_id in world.counterfactual_branches

    def test_intervene_preserves_past(self, dim):
        """intervene should preserve states before intervention."""
        from grilly.experimental.temporal.counterfactual import CounterfactualReasoner
        from grilly.experimental.temporal.state import TemporalWorldModel

        world = TemporalWorldModel(dim=dim)
        reasoner = CounterfactualReasoner(world)

        world.set_state(0, {"has_umbrella": True})
        world.set_state(1, {"has_umbrella": False})

        branch_id = reasoner.intervene(1, "has_umbrella", True)
        cf_timeline = world.counterfactual_branches[branch_id]

        # Past should be preserved
        assert 0 in cf_timeline
        assert cf_timeline[0].variables["has_umbrella"] is True

    def test_intervene_changes_at_intervention_time(self, dim):
        """intervene should change variable at intervention time."""
        from grilly.experimental.temporal.counterfactual import CounterfactualReasoner
        from grilly.experimental.temporal.state import TemporalWorldModel

        world = TemporalWorldModel(dim=dim)
        reasoner = CounterfactualReasoner(world)

        world.set_state(1, {"has_umbrella": False})

        branch_id = reasoner.intervene(1, "has_umbrella", True)
        cf_timeline = world.counterfactual_branches[branch_id]

        # At intervention time, should have new value
        assert cf_timeline[1].variables["has_umbrella"] is True


class TestQueryCounterfactual:
    """Tests for counterfactual queries."""

    def test_query_counterfactual_returns_result(self, dim):
        """query_counterfactual should return CounterfactualResult."""
        from grilly.experimental.temporal.counterfactual import (
            CounterfactualQuery,
            CounterfactualReasoner,
        )
        from grilly.experimental.temporal.state import TemporalWorldModel

        world = TemporalWorldModel(dim=dim)
        reasoner = CounterfactualReasoner(world)

        world.set_state(0, {"has_umbrella": False})
        world.set_state(1, {"has_umbrella": False})

        query = CounterfactualQuery(
            intervention_time=0,
            variable="has_umbrella",
            actual_value=False,
            counterfactual_value=True,
            query_time=1,
            query_variable="has_umbrella",
        )

        result = reasoner.query_counterfactual(query)

        assert result.query is query
        assert "actual_outcome" in result.__dict__
        assert "counterfactual_outcome" in result.__dict__

    def test_query_counterfactual_shows_difference(self, dim):
        """query_counterfactual should show differences between actual and counterfactual."""
        from grilly.experimental.temporal.counterfactual import (
            CounterfactualQuery,
            CounterfactualReasoner,
        )
        from grilly.experimental.temporal.state import TemporalWorldModel

        world = TemporalWorldModel(dim=dim)
        reasoner = CounterfactualReasoner(world)

        world.set_state(0, {"has_umbrella": False})
        world.set_state(1, {"has_umbrella": False})

        query = CounterfactualQuery(
            intervention_time=0,
            variable="has_umbrella",
            actual_value=False,
            counterfactual_value=True,
            query_time=1,
            query_variable="has_umbrella",
        )

        result = reasoner.query_counterfactual(query)

        # Should show difference
        assert isinstance(result.difference, dict)
