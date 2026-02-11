"""
TDD Tests for CausalChain.

Tests causal rules, state encoding, and forward propagation.
"""


class TestCausalChainBasic:
    """Basic tests for CausalChain initialization."""

    def test_init_default_dimensions(self):
        """Should initialize with default dimensions."""
        from grilly.experimental.temporal.causal import CausalChain

        chain = CausalChain()

        assert chain.dim > 0
        assert len(chain.rules) == 0

    def test_init_custom_dimension(self):
        """Should initialize with custom dimension."""
        from grilly.experimental.temporal.causal import CausalChain

        chain = CausalChain(dim=2048)

        assert chain.dim == 2048


class TestEncodeState:
    """Tests for state encoding."""

    def test_encode_variable_returns_vector(self, dim):
        """encode_variable should return vector."""
        from grilly.experimental.temporal.causal import CausalChain

        chain = CausalChain(dim=dim)

        result = chain.encode_variable("temperature")

        assert result.shape == (dim,)

    def test_encode_value_returns_vector(self, dim):
        """encode_value should return vector."""
        from grilly.experimental.temporal.causal import CausalChain

        chain = CausalChain(dim=dim)

        result = chain.encode_value("temperature", 25)

        assert result.shape == (dim,)

    def test_encode_assignment_returns_vector(self, dim):
        """encode_assignment should return bound vector."""
        from grilly.experimental.temporal.causal import CausalChain

        chain = CausalChain(dim=dim)

        result = chain.encode_assignment("temperature", 25)

        assert result.shape == (dim,)

    def test_encode_state_returns_vector(self, dim):
        """encode_state should return bundled vector."""
        from grilly.experimental.temporal.causal import CausalChain

        chain = CausalChain(dim=dim)

        variables = {"temperature": 25, "humidity": 60}
        result = chain.encode_state(variables)

        assert result.shape == (dim,)


class TestCausalRules:
    """Tests for causal rules."""

    def test_add_rule_stores_rule(self, dim):
        """add_rule should store the rule."""
        from grilly.experimental.temporal.causal import CausalChain

        chain = CausalChain(dim=dim)

        chain.add_rule("rain_causes_wet", {"raining": True}, {"is_wet": True})

        assert len(chain.rules) == 1
        assert chain.rules[0].name == "rain_causes_wet"

    def test_check_condition_matches(self, dim):
        """check_condition should return True when state matches."""
        from grilly.experimental.temporal.causal import CausalChain

        chain = CausalChain(dim=dim)

        state = {"raining": True, "has_umbrella": False}
        condition = {"raining": True}

        result = chain.check_condition(state, condition)

        assert result is True

    def test_check_condition_no_match(self, dim):
        """check_condition should return False when state doesn't match."""
        from grilly.experimental.temporal.causal import CausalChain

        chain = CausalChain(dim=dim)

        state = {"raining": False}
        condition = {"raining": True}

        result = chain.check_condition(state, condition)

        assert result is False


class TestPropagateForward:
    """Tests for forward propagation."""

    def test_propagate_forward_applies_rules(self, dim):
        """propagate_forward should apply matching rules."""
        from grilly.experimental.temporal.causal import CausalChain

        chain = CausalChain(dim=dim)

        # Add rule: if raining -> is_wet
        chain.add_rule("rain_causes_wet", {"raining": True}, {"is_wet": True})

        # Start state: raining
        initial = {"raining": True, "is_wet": False}

        # Propagate forward
        result = chain.propagate_forward(initial, steps=1)

        # Should have is_wet = True
        assert result.get("is_wet") is True

    def test_propagate_forward_multiple_steps(self, dim):
        """propagate_forward should handle multiple steps."""
        from grilly.experimental.temporal.causal import CausalChain

        chain = CausalChain(dim=dim)

        chain.add_rule("wet_causes_cold", {"is_wet": True}, {"is_cold": True})
        chain.add_rule("cold_causes_sick", {"is_cold": True}, {"is_sick": True})

        initial = {"is_wet": True, "is_cold": False, "is_sick": False}

        result = chain.propagate_forward(initial, steps=2)

        # After 2 steps, should be sick
        assert result.get("is_sick") is True

    def test_propagate_forward_no_matching_rules(self, dim):
        """propagate_forward should preserve state when no rules match."""
        from grilly.experimental.temporal.causal import CausalChain

        chain = CausalChain(dim=dim)

        initial = {"temperature": 25}

        result = chain.propagate_forward(initial, steps=1)

        # Should preserve temperature
        assert result.get("temperature") == 25


class TestTraceCauses:
    """Tests for backward reasoning."""

    def test_trace_causes_finds_rules(self, dim):
        """trace_causes should find rules that produce effect."""
        from grilly.experimental.temporal.causal import CausalChain

        chain = CausalChain(dim=dim)

        chain.add_rule("rain_causes_wet", {"raining": True}, {"is_wet": True})
        chain.add_rule("sprinkler_causes_wet", {"sprinkler_on": True}, {"is_wet": True})

        effect_state = {"is_wet": True}

        causes = chain.trace_causes(effect_state, "is_wet")

        # Should find both rules
        assert len(causes) == 2
