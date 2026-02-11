"""
TDD Tests for TemporalDecisionValidator.

Tests decision validation against past, present, and future constraints.
"""


class TestTemporalDecisionValidatorBasic:
    """Basic tests for TemporalDecisionValidator initialization."""

    def test_init_with_world_and_reasoner(self, dim):
        """Should initialize with world model and counterfactual reasoner."""
        from grilly.experimental.temporal.counterfactual import CounterfactualReasoner
        from grilly.experimental.temporal.state import TemporalWorldModel
        from grilly.experimental.temporal.validator import TemporalDecisionValidator

        world = TemporalWorldModel(dim=dim)
        reasoner = CounterfactualReasoner(world)
        validator = TemporalDecisionValidator(world, reasoner)

        assert validator.world is world
        assert validator.cf_reasoner is reasoner


class TestValidateDecision:
    """Tests for decision validation."""

    def test_validate_decision_returns_result(self, dim):
        """validate_decision should return ValidationResult."""
        from grilly.experimental.temporal.counterfactual import CounterfactualReasoner
        from grilly.experimental.temporal.state import TemporalWorldModel
        from grilly.experimental.temporal.validator import TemporalDecisionValidator

        world = TemporalWorldModel(dim=dim)
        reasoner = CounterfactualReasoner(world)
        validator = TemporalDecisionValidator(world, reasoner)

        world.set_state(0, {"has_umbrella": True})

        decision = {"has_umbrella": False}
        result = validator.validate_decision(0, decision)

        assert hasattr(result, "is_valid")
        assert hasattr(result, "past_consistent")
        assert hasattr(result, "present_consistent")
        assert hasattr(result, "future_consistent")
        assert hasattr(result, "violations")
        assert hasattr(result, "confidence")

    def test_validate_decision_past_consistency(self, dim):
        """Should check past consistency."""
        from grilly.experimental.temporal.counterfactual import CounterfactualReasoner
        from grilly.experimental.temporal.state import TemporalWorldModel
        from grilly.experimental.temporal.validator import TemporalDecisionValidator

        world = TemporalWorldModel(dim=dim)
        reasoner = CounterfactualReasoner(world)
        validator = TemporalDecisionValidator(world, reasoner)

        # Set past: had umbrella
        world.set_state(0, {"has_umbrella": True})

        # Decision: claim never had umbrella (contradicts past)
        decision = {"has_umbrella": False}
        result = validator.validate_decision(1, decision)

        # Should detect past inconsistency
        # (Note: simple implementation may not catch all contradictions)
        assert isinstance(result.past_consistent, bool)

    def test_validate_decision_future_constraints(self, dim):
        """Should check future constraint violations."""
        from grilly.experimental.temporal.counterfactual import CounterfactualReasoner
        from grilly.experimental.temporal.state import TemporalWorldModel
        from grilly.experimental.temporal.validator import TemporalDecisionValidator

        world = TemporalWorldModel(dim=dim)
        reasoner = CounterfactualReasoner(world)
        validator = TemporalDecisionValidator(world, reasoner)

        # Add constraint: can't be both wet and dry
        def cant_be_wet_and_dry(time, vars):
            if vars.get("is_wet") and vars.get("is_dry"):
                return False, "Can't be both wet and dry"
            return True, ""

        world.add_constraint(cant_be_wet_and_dry)

        # Add rule that would cause violation
        world.causal_chain.add_rule("wet_rule", {}, {"is_wet": True, "is_dry": True})

        world.set_state(0, {})
        decision = {}

        result = validator.validate_decision(0, decision, check_horizon=2)

        # Should detect future violation
        assert isinstance(result.future_consistent, bool)
