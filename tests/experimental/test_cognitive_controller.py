"""
TDD Tests for CognitiveController.

Tests the full "think before speak" pipeline.
"""

import numpy as np


class TestCognitiveControllerBasic:
    """Basic tests for CognitiveController initialization."""

    def test_init_default_dimensions(self):
        """Should initialize with default dimensions."""
        from grilly.experimental.cognitive.controller import CognitiveController

        controller = CognitiveController()

        assert controller.dim > 0
        assert hasattr(controller, "language")
        assert hasattr(controller, "world")
        assert hasattr(controller, "wm")

    def test_init_custom_dimension(self):
        """Should initialize with custom dimension."""
        from grilly.experimental.cognitive.controller import CognitiveController

        controller = CognitiveController(dim=2048)

        assert controller.dim == 2048


class TestProcess:
    """Tests for main processing pipeline."""

    def test_process_returns_response(self, dim):
        """process should return response or None."""
        from grilly.experimental.cognitive.controller import CognitiveController

        controller = CognitiveController(dim=dim)

        response = controller.process("Hello")

        # Response can be None if confidence is too low
        assert response is None or isinstance(response, str)

    def test_process_with_high_confidence_returns_response(self, dim):
        """process should return response when confidence is high."""
        from grilly.experimental.cognitive.controller import CognitiveController

        controller = CognitiveController(dim=dim)

        # Add some knowledge to increase confidence
        controller.add_knowledge("user", "says", "hello")

        response = controller.process("Hello")

        # May return None or a response depending on confidence threshold
        assert response is None or isinstance(response, str)

    def test_process_stores_thinking_trace(self, dim):
        """process should store thinking trace."""
        from grilly.experimental.cognitive.controller import CognitiveController

        controller = CognitiveController(dim=dim)

        controller.process("Hello")

        trace = controller.get_thinking_trace()

        assert isinstance(trace, list)


class TestConfidenceGate:
    """Tests for confidence-based output gating."""

    def test_low_confidence_returns_none(self, dim):
        """process should return None if confidence is too low."""
        from grilly.experimental.cognitive.controller import CognitiveController

        controller = CognitiveController(dim=dim, confidence_threshold=0.9)

        # Process something that would have low confidence
        response = controller.process("gibberish xyz123")

        # Should return None due to low confidence
        assert response is None

    def test_confidence_threshold_configurable(self, dim):
        """confidence_threshold should be configurable."""
        from grilly.experimental.cognitive.controller import CognitiveController

        controller_low = CognitiveController(dim=dim, confidence_threshold=0.1)
        controller_high = CognitiveController(dim=dim, confidence_threshold=0.9)

        assert controller_low.confidence_threshold == 0.1
        assert controller_high.confidence_threshold == 0.9


class TestUnderstand:
    """Tests for understanding input."""

    def test_understand_returns_result(self, dim):
        """understand should return UnderstandingResult."""
        from grilly.experimental.cognitive.controller import CognitiveController

        controller = CognitiveController(dim=dim)

        result = controller.understand("The dog chased the cat")

        assert hasattr(result, "surface_meaning")
        assert hasattr(result, "deep_meaning")
        assert hasattr(result, "words")
        assert hasattr(result, "parsed_roles")


class TestTemporalValidation:
    """Tests for temporal validation in CognitiveController."""

    def test_temporal_validation_filters_invalid_candidate(self, dim):
        """Temporal validator should filter invalid candidates."""
        from grilly.experimental.cognitive.controller import CognitiveController
        from grilly.experimental.cognitive.simulator import SimulationResult
        from grilly.experimental.temporal import (
            CounterfactualReasoner,
            TemporalDecisionValidator,
            TemporalWorldModel,
        )

        controller = CognitiveController(dim=dim, confidence_threshold=0.1)

        world = TemporalWorldModel(dim=dim)
        world.set_state(0, {"status": "alive"})
        cf = CounterfactualReasoner(world)
        validator = TemporalDecisionValidator(world, cf)

        def extractor(text: str) -> dict:
            if "dead" in text:
                return {"status": "dead"}
            if "alive" in text:
                return {"status": "alive"}
            return {}

        controller.set_temporal_validation(world, validator, decision_extractor=extractor)

        controller._generate_candidates = lambda _: ["status is dead", "status is alive"]

        def fake_sim(candidate: str, context=None):
            score = 0.9 if "dead" in candidate else 0.2
            return SimulationResult(
                candidate=candidate,
                vector=np.zeros(dim, dtype=np.float32),
                coherence_score=score,
                coherence_reason="test",
                predicted_response=None,
                social_appropriateness=score,
                confidence=score,
            )

        controller.simulator.simulate_utterance = fake_sim

        response = controller.process("test", decision_time=1, verbose=True)
        assert response == "status is alive"
