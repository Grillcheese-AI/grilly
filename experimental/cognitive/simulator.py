"""
InternalSimulator - "Think before you speak" simulation.

Simulates candidate utterances before outputting to verify coherence and appropriateness.
"""

from dataclasses import dataclass

import numpy as np

from grilly.experimental.cognitive.capsule import cosine_similarity
from grilly.experimental.cognitive.memory import WorkingMemory
from grilly.experimental.cognitive.world import WorldModel
from grilly.experimental.language.system import InstantLanguage
from grilly.experimental.vsa.ops import HolographicOps


@dataclass
class SimulationResult:
    """Result of running an internal simulation."""

    candidate: str
    vector: np.ndarray
    coherence_score: float
    coherence_reason: str
    predicted_response: str | None
    social_appropriateness: float
    confidence: float

    @property
    def overall_score(self) -> float:
        """Execute overall score."""

        return (self.coherence_score + self.social_appropriateness + self.confidence) / 3


class InternalSimulator:
    """
    Simulates "saying" something before actually outputting.

    This implements "think before you speak":
    1. Generate candidate response
    2. Encode it
    3. Check coherence with world model
    4. Predict how it would be received
    5. Score and decide whether to output

    Based on prediction-by-production models in neuroscience.
    """

    def __init__(
        self,
        language_system: InstantLanguage,
        world_model: WorldModel,
        working_memory: WorkingMemory,
    ):
        """Initialize the instance."""

        self.language = language_system
        self.world = world_model
        self.wm = working_memory
        self.dim = language_system.dim

        # Social/pragmatic patterns
        self.social_patterns = {
            "polite": HolographicOps.random_vector(self.dim, seed=7000),
            "rude": HolographicOps.random_vector(self.dim, seed=7001),
            "helpful": HolographicOps.random_vector(self.dim, seed=7002),
            "harmful": HolographicOps.random_vector(self.dim, seed=7003),
            "clear": HolographicOps.random_vector(self.dim, seed=7004),
            "confusing": HolographicOps.random_vector(self.dim, seed=7005),
        }

        # History of simulations for learning
        self.simulation_history: list[SimulationResult] = []

    def simulate_utterance(
        self, candidate: str, context: np.ndarray | None = None
    ) -> SimulationResult:
        """
        Simulate saying something.

        This is the core "think before you speak" operation.
        """
        # 1. Encode the candidate
        words = candidate.lower().split()
        candidate_vec = self.language.sentence_encoder.encode_sentence(words)

        # 2. Check coherence with world model
        is_coherent, coherence_score, reason = self.world.check_coherence(candidate_vec)

        # 3. Predict response (what would happen if I said this?)
        predicted_response = self._predict_response(candidate_vec, context)

        # 4. Check social appropriateness
        social_score = self._check_social(candidate_vec)

        # 5. Compute confidence
        confidence = self._compute_confidence(candidate_vec, context)

        result = SimulationResult(
            candidate=candidate,
            vector=candidate_vec,
            coherence_score=coherence_score if is_coherent else -coherence_score,
            coherence_reason=reason,
            predicted_response=predicted_response,
            social_appropriateness=social_score,
            confidence=confidence,
        )

        self.simulation_history.append(result)
        return result

    def _predict_response(
        self, utterance_vec: np.ndarray, context: np.ndarray | None
    ) -> str | None:
        """Predict how the listener might respond."""
        # Use causal expectations from world model
        # This is simplified - real system would be more sophisticated

        # Check if utterance is a question
        # Questions predict answers
        question_pattern = self.language.word_encoder.encode_word("question")
        is_question = HolographicOps.similarity(utterance_vec, question_pattern) > 0.3

        if is_question:
            return "[expects answer]"

        # Check if utterance is a command
        command_pattern = self.language.word_encoder.encode_word("command")
        is_command = HolographicOps.similarity(utterance_vec, command_pattern) > 0.3

        if is_command:
            return "[expects action]"

        return "[expects acknowledgment]"

    def _check_social(self, utterance_vec: np.ndarray) -> float:
        """Check social appropriateness of utterance."""
        # Compute similarity to good/bad social patterns
        polite_sim = HolographicOps.similarity(utterance_vec, self.social_patterns["polite"])
        rude_sim = HolographicOps.similarity(utterance_vec, self.social_patterns["rude"])
        helpful_sim = HolographicOps.similarity(utterance_vec, self.social_patterns["helpful"])
        harmful_sim = HolographicOps.similarity(utterance_vec, self.social_patterns["harmful"])

        # Positive patterns - negative patterns
        score = (polite_sim + helpful_sim) - (rude_sim + harmful_sim)

        # Normalize to [0, 1]
        return (score + 1) / 2

    def _compute_confidence(self, utterance_vec: np.ndarray, context: np.ndarray | None) -> float:
        """Compute confidence in the utterance."""
        # Base confidence from working memory
        wm_context = self.wm.get_context_vector()

        if context is not None:
            # Check similarity to context
            context_sim = HolographicOps.similarity(utterance_vec, context)
        else:
            context_sim = HolographicOps.similarity(utterance_vec, wm_context)

        vsa_conf = (context_sim + 1) / 2

        capsule_conf = None
        if self.wm.capsule_encoder is not None:
            has_capsule_context = any(item.capsule_vector is not None for item in self.wm.items)
            if has_capsule_context:
                utterance_capsule = self.wm.capsule_encoder.encode_vector(utterance_vec)
                context_capsule = self.wm.get_context_capsule()
                if context_capsule is not None:
                    capsule_sim = cosine_similarity(utterance_capsule, context_capsule)
                    capsule_conf = (capsule_sim + 1) / 2

        if capsule_conf is None:
            return vsa_conf

        return (vsa_conf + capsule_conf) / 2
