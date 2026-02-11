"""
Understander - Deep comprehension beyond surface encoding.

Makes inferences, retrieves knowledge, and builds situation models.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from grilly.experimental.cognitive.capsule import cosine_similarity
from grilly.experimental.cognitive.memory import WorkingMemory, WorkingMemorySlot
from grilly.experimental.cognitive.world import WorldModel
from grilly.experimental.language.system import InstantLanguage
from grilly.experimental.vsa.ops import HolographicOps


@dataclass
class UnderstandingResult:
    """Result of understanding an input."""

    surface_meaning: np.ndarray  # Direct encoding
    deep_meaning: np.ndarray  # After inference
    inferences: list[str]  # What was inferred
    questions: list[str]  # Remaining questions
    confidence: float
    parsed_roles: dict[str, str]  # Extracted semantic roles
    words: list[str]  # Parsed words


class Understander:
    """
    Deep understanding beyond surface encoding.

    Understanding involves:
    1. Parse surface structure
    2. Retrieve relevant knowledge
    3. Make inferences
    4. Identify gaps/questions
    5. Build situation model
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

        # Inference rules (pattern -> inference)
        self.inference_rules: list[tuple[np.ndarray, str, Callable]] = []
        self._init_inference_rules()

    def _init_inference_rules(self):
        """Initialize basic inference rules."""
        # Rule: If X causes Y and X happens, Y will happen
        cause_pattern = self.language.word_encoder.encode_word("causes")

        def causal_inference(context: dict) -> str | None:
            """Execute causal inference."""

            if "cause" in context and "effect" in context:
                return f"{context['effect']} will happen"
            return None

        self.inference_rules.append((cause_pattern, "causal", causal_inference))

        # Rule: If X is_a Y, X has properties of Y
        isa_pattern = self.language.word_encoder.encode_word("is_a")

        def inheritance_inference(context: dict) -> str | None:
            """Execute inheritance inference."""

            if "instance" in context and "category" in context:
                return f"{context['instance']} has properties of {context['category']}"
            return None

        self.inference_rules.append((isa_pattern, "inheritance", inheritance_inference))

    def understand(self, input_text: str, verbose: bool = False) -> UnderstandingResult:
        """
        Deeply understand an input.

        This is more than encoding - it builds a mental model.
        """
        # 1. Parse surface structure
        words = input_text.lower().split()
        surface_vec = self.language.sentence_encoder.encode_sentence(words)
        parsed = self.language.parser.parse(surface_vec, num_slots=len(words))

        # Extract roles
        roles = {word: role for word, role, conf in parsed if conf > 0.3}

        # 2. Add to working memory
        self.wm.add(
            surface_vec, input_text, WorkingMemorySlot.FOCUS, confidence=1.0, source="input"
        )

        # 3. Retrieve relevant knowledge
        retrieved = self._retrieve_relevant(surface_vec)
        for fact_str, fact_vec in retrieved:
            self.wm.add(
                fact_vec,
                fact_str,
                WorkingMemorySlot.RETRIEVED,
                confidence=0.8,
                source="world_model",
            )

        # 4. Make inferences
        inferences = self._make_inferences(surface_vec, roles, retrieved)

        # 5. Identify questions/gaps
        questions = self._identify_gaps(surface_vec, roles)

        # 6. Build deep meaning (surface + inferences)
        inference_vecs = [
            self.language.sentence_encoder.encode_sentence(inf.split()) for inf in inferences
        ]

        if inference_vecs:
            deep_vec = HolographicOps.bundle([surface_vec] + inference_vecs)
        else:
            deep_vec = surface_vec

        # 7. Compute understanding confidence
        confidence = self._compute_understanding_confidence(surface_vec, retrieved, inferences)

        return UnderstandingResult(
            surface_meaning=surface_vec,
            deep_meaning=deep_vec,
            inferences=inferences,
            questions=questions,
            confidence=confidence,
            parsed_roles=roles,
            words=words,
        )

    def _retrieve_relevant(
        self, query_vec: np.ndarray, top_k: int = 3
    ) -> list[tuple[str, np.ndarray]]:
        """Retrieve relevant facts from world model."""
        results = []

        query_capsule = None
        if self.world.capsule_encoder is not None:
            query_capsule = self.world.capsule_encoder.encode_vector(query_vec)

        for fact, fact_capsule in zip(self.world.facts, self.world.fact_capsules):
            sim = HolographicOps.similarity(query_vec, fact.vector)
            if query_capsule is not None and fact_capsule is not None:
                cap_sim = cosine_similarity(query_capsule, fact_capsule)
                sim = 0.7 * sim + 0.3 * cap_sim
            if sim > 0.3:
                fact_str = f"{fact.subject} {fact.relation} {fact.object}"
                results.append((fact_str, fact.vector, sim))

        # Sort by similarity
        results.sort(key=lambda x: x[2], reverse=True)

        return [(r[0], r[1]) for r in results[:top_k]]

    def _make_inferences(
        self,
        surface_vec: np.ndarray,
        roles: dict[str, str],
        retrieved: list[tuple[str, np.ndarray]],
    ) -> list[str]:
        """Make inferences based on input and knowledge."""
        inferences = []

        # Check each inference rule
        for pattern, rule_type, rule_func in self.inference_rules:
            sim = HolographicOps.similarity(surface_vec, pattern)

            if sim > 0.3:
                # Build context for rule
                context = {"roles": roles}

                # Try to apply rule
                inference = rule_func(context)
                if inference:
                    inferences.append(inference)

        # Causal inferences from retrieved facts
        for fact_str, fact_vec in retrieved:
            # Check if retrieved fact enables an inference
            parts = fact_str.split()
            if "causes" in parts:
                idx = parts.index("causes")
                cause = " ".join(parts[:idx])
                effect = " ".join(parts[idx + 1 :])

                # Check if cause is mentioned in input
                cause_vec = self.language.word_encoder.encode_word(cause)
                if HolographicOps.similarity(surface_vec, cause_vec) > 0.3:
                    inferences.append(f"Therefore, {effect} may occur")

        return inferences

    def _identify_gaps(self, surface_vec: np.ndarray, roles: dict[str, str]) -> list[str]:
        """Identify what's not understood / needs clarification."""
        questions = []

        # Check for missing core roles
        core_roles = ["SUBJ", "VERB", "OBJ"]
        found_roles = set(roles.values())

        for role in core_roles:
            if role not in found_roles:
                if role == "SUBJ":
                    questions.append("Who or what is the subject?")
                elif role == "VERB":
                    questions.append("What action or state?")
                elif role == "OBJ":
                    questions.append("What is affected?")

        # Check coherence with world model
        is_coherent, score, reason = self.world.check_coherence(surface_vec)
        if not is_coherent:
            questions.append(f"How can this be true? ({reason})")

        return questions

    def _compute_understanding_confidence(
        self,
        surface_vec: np.ndarray,
        retrieved: list[tuple[str, np.ndarray]],
        inferences: list[str],
    ) -> float:
        """Compute how well we understand the input."""
        # Factors:
        # 1. How much relevant knowledge was retrieved
        retrieval_score = min(1.0, len(retrieved) / 3)

        # 2. How many inferences were made
        inference_score = min(1.0, len(inferences) / 2)

        # 3. Coherence with world model
        is_coherent, coherence_score, _ = self.world.check_coherence(surface_vec)

        # Combine
        confidence = (retrieval_score + inference_score + coherence_score) / 3
        return float(np.clip(confidence, 0.0, 1.0))
