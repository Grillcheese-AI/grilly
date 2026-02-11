"""
TDD Tests for InternalSimulator.

Tests simulation of candidate utterances and scoring.
"""


class TestInternalSimulatorBasic:
    """Basic tests for InternalSimulator initialization."""

    def test_init_with_components(self, dim):
        """Should initialize with language, world model, and working memory."""
        from grilly.experimental.cognitive.memory import WorkingMemory
        from grilly.experimental.cognitive.simulator import InternalSimulator
        from grilly.experimental.cognitive.world import WorldModel
        from grilly.experimental.language.system import InstantLanguage

        language = InstantLanguage(dim=dim)
        world = WorldModel(dim=dim)
        wm = WorkingMemory(dim=dim)

        simulator = InternalSimulator(language, world, wm)

        assert simulator.language is language
        assert simulator.world is world
        assert simulator.wm is wm


class TestSimulateUtterance:
    """Tests for utterance simulation."""

    def test_simulate_utterance_returns_result(self, dim):
        """simulate_utterance should return SimulationResult."""
        from grilly.experimental.cognitive.memory import WorkingMemory
        from grilly.experimental.cognitive.simulator import InternalSimulator, SimulationResult
        from grilly.experimental.cognitive.world import WorldModel
        from grilly.experimental.language.system import InstantLanguage

        language = InstantLanguage(dim=dim)
        world = WorldModel(dim=dim)
        wm = WorkingMemory(dim=dim)
        simulator = InternalSimulator(language, world, wm)

        result = simulator.simulate_utterance("The dog is happy")

        assert isinstance(result, SimulationResult)
        assert result.candidate == "The dog is happy"

    def test_simulate_utterance_computes_scores(self, dim):
        """simulate_utterance should compute coherence and social scores."""
        from grilly.experimental.cognitive.memory import WorkingMemory
        from grilly.experimental.cognitive.simulator import InternalSimulator
        from grilly.experimental.cognitive.world import WorldModel
        from grilly.experimental.language.system import InstantLanguage

        language = InstantLanguage(dim=dim)
        world = WorldModel(dim=dim)
        wm = WorkingMemory(dim=dim)
        simulator = InternalSimulator(language, world, wm)

        result = simulator.simulate_utterance("Hello")

        assert hasattr(result, "coherence_score")
        assert hasattr(result, "social_appropriateness")
        assert hasattr(result, "confidence")
        assert hasattr(result, "overall_score")

    def test_simulate_utterance_stores_in_history(self, dim):
        """simulate_utterance should store result in history."""
        from grilly.experimental.cognitive.memory import WorkingMemory
        from grilly.experimental.cognitive.simulator import InternalSimulator
        from grilly.experimental.cognitive.world import WorldModel
        from grilly.experimental.language.system import InstantLanguage

        language = InstantLanguage(dim=dim)
        world = WorldModel(dim=dim)
        wm = WorkingMemory(dim=dim)
        simulator = InternalSimulator(language, world, wm)

        initial_count = len(simulator.simulation_history)

        simulator.simulate_utterance("Test")

        assert len(simulator.simulation_history) == initial_count + 1


class TestSimulationScoring:
    """Tests for scoring mechanisms."""

    def test_overall_score_combines_metrics(self, dim):
        """overall_score should combine coherence, social, and confidence."""
        from grilly.experimental.cognitive.memory import WorkingMemory
        from grilly.experimental.cognitive.simulator import InternalSimulator, SimulationResult
        from grilly.experimental.cognitive.world import WorldModel
        from grilly.experimental.language.system import InstantLanguage
        from grilly.experimental.vsa.ops import HolographicOps

        language = InstantLanguage(dim=dim)
        world = WorldModel(dim=dim)
        wm = WorkingMemory(dim=dim)
        InternalSimulator(language, world, wm)

        result = SimulationResult(
            candidate="test",
            vector=HolographicOps.random_vector(dim),
            coherence_score=0.8,
            coherence_reason="test",
            predicted_response=None,
            social_appropriateness=0.7,
            confidence=0.9,
        )

        overall = result.overall_score

        # Should be average of the three scores
        expected = (0.8 + 0.7 + 0.9) / 3
        assert abs(overall - expected) < 0.01
