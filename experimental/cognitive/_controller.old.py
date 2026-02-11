"""
CognitiveController - Main "think before speak" controller.

Orchestrates understanding, simulation, and response generation with confidence gating.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from grilly.experimental.cognitive.memory import WorkingMemory
from grilly.experimental.cognitive.simulator import InternalSimulator, SimulationResult
from grilly.experimental.cognitive.understander import Understander, UnderstandingResult
from grilly.experimental.cognitive.world import WorldModel
from grilly.experimental.language.system import InstantLanguage, SVCIngestionResult

if TYPE_CHECKING:
    from grilly.experimental.language.svc_loader import SVCEntry


@dataclass
class CognitiveState:
    """Current state of the cognitive system."""

    understanding: UnderstandingResult | None = None
    candidates: list[tuple[str, SimulationResult]] = field(default_factory=list)
    selected_response: str | None = None
    confidence: float = 0.0
    thinking_steps: list[str] = field(default_factory=list)


class CognitiveController:
    """
    Main controller implementing "think before you speak".

    Process:
    1. RECEIVE: Get input
    2. UNDERSTAND: Deep comprehension
    3. GENERATE: Create candidate responses
    4. SIMULATE: Evaluate each candidate
    5. SELECT: Choose best candidate
    6. VERIFY: Final coherence check
    7. OUTPUT: Return response (if confidence high enough)
    """

    DEFAULT_DIM = 4096
    DEFAULT_CONFIDENCE_THRESHOLD = 0.6

    def __init__(
        self, dim: int = DEFAULT_DIM, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    ):
        """Initialize the instance."""

        self.dim = dim
        self.confidence_threshold = confidence_threshold

        # Core components
        self.language = InstantLanguage(dim=dim)
        self.world = WorldModel(dim=dim)
        self.wm = WorkingMemory(dim=dim)
        self.simulator = InternalSimulator(self.language, self.world, self.wm)
        self.understander = Understander(self.language, self.world, self.wm)

        # State tracking
        self.current_state: CognitiveState | None = None
        self.thinking_trace: list[str] = []

        # Optional temporal validation
        self.temporal_world = None
        self.temporal_validator = None
        self.decision_extractor: Callable[[str], dict[str, object]] | None = None
        self.temporal_check_horizon = 5

    def add_knowledge(self, subject: str, relation: str, object_: str):
        """Add knowledge to world model."""
        self.world.add_fact(subject, relation, object_)

    def ingest_svc(
        self,
        entries: list["SVCEntry"],
        learn_templates: bool = True,
        build_realm_vectors: bool = True,
        verbose: bool = False,
        engine: object | None = None,
    ) -> SVCIngestionResult:
        """
        Ingest SVC data into the full cognitive system.

        When a GPU is available the heavy VSA operations are dispatched
        to Vulkan compute shaders via an ``SVCIngestionEngine``.
        Pass *engine* to control backend selection.

        This does three things:
        1. Feeds entries into InstantLanguage (via the engine) for sentence
           encoding, vocabulary building, template learning, and realm
           vectors.
        2. Adds each SVC triple (s, root_verb, c) as a world model fact
           so the controller can use them for coherence checking.
        3. Adds causal expectations when the root verb implies causality.

        Args:
            entries: List of SVCEntry instances.
            learn_templates: If True, learn sentence templates.
            build_realm_vectors: If True, build realm expert vectors.
            verbose: If True, print progress.
            engine: Optional SVCIngestionEngine (auto-created if None).

        Returns:
            SVCIngestionResult with statistics.
        """
        # 1. Language-level ingestion (GPU-accelerated via engine)
        result = self.language.ingest_svc(
            entries,
            learn_templates=learn_templates,
            build_realm_vectors=build_realm_vectors,
            verbose=verbose,
            engine=engine,
        )

        # 2. World model facts
        causal_verbs = {
            "cause",
            "causes",
            "prevent",
            "prevents",
            "improve",
            "improves",
            "reduce",
            "reduces",
            "increase",
            "increases",
            "enable",
            "enables",
            "lead",
            "leads",
            "result",
            "results",
            "produce",
            "produces",
        }

        for entry in entries:
            subject = entry.svc_s.lower()
            verb = entry.root_verb.lower()
            complement = entry.svc_c.lower()

            # Add as world fact
            self.world.add_fact(
                subject=subject,
                relation=verb,
                object_=complement,
                confidence=min(1.0, 0.5 + entry.complexity),
                source=entry.source,
            )

            # 3. Causal link if the verb implies causation
            if verb in causal_verbs:
                self.world.add_causal_link(
                    cause=subject,
                    effect=complement,
                    strength=0.5 + entry.complexity * 0.5,
                )

        return result

    def understand(self, text: str) -> UnderstandingResult:
        """Just understand without responding."""
        return self.understander.understand(text)

    def set_temporal_validation(
        self,
        world_model,
        validator,
        decision_extractor: Callable[[str], dict[str, object]] | None = None,
        check_horizon: int = 5,
    ) -> None:
        """
        Attach temporal validation for candidate filtering.

        Args:
            world_model: TemporalWorldModel instance
            validator: TemporalDecisionValidator instance
            decision_extractor: Optional function mapping text -> decision dict
            check_horizon: How far into the future to validate
        """
        self.temporal_world = world_model
        self.temporal_validator = validator
        self.decision_extractor = decision_extractor
        self.temporal_check_horizon = check_horizon

    def process(
        self, input_text: str, verbose: bool = False, decision_time: int | None = None
    ) -> str | None:
        """
        Process input and generate response.

        Returns response if confidence is high enough, None otherwise.
        """
        self.thinking_trace = []
        state = CognitiveState()

        # 1. UNDERSTAND
        if verbose:
            self.thinking_trace.append(f"Understanding: {input_text}")

        understanding = self.understander.understand(input_text, verbose=verbose)
        state.understanding = understanding

        if verbose:
            self.thinking_trace.append(f"Inferences: {understanding.inferences}")
            self.thinking_trace.append(f"Confidence: {understanding.confidence:.2f}")

        # 2. GENERATE candidates
        candidates = self._generate_candidates(understanding)

        if verbose:
            self.thinking_trace.append(f"Generated {len(candidates)} candidates")

        # 3. SIMULATE each candidate
        evaluated = []
        for candidate in candidates:
            result = self.simulator.simulate_utterance(
                candidate, context=understanding.deep_meaning
            )
            evaluated.append((candidate, result))

            if verbose:
                self.thinking_trace.append(
                    f"Candidate '{candidate}': score={result.overall_score:.2f}, "
                    f"coherence={result.coherence_score:.2f}"
                )

        # Sort by overall score
        evaluated.sort(key=lambda x: x[1].overall_score, reverse=True)
        state.candidates = evaluated

        # 3.5 TEMPORAL VALIDATION
        evaluated = self._apply_temporal_validation(
            evaluated, decision_time=decision_time, verbose=verbose
        )
        state.candidates = evaluated

        # 4. SELECT best candidate
        if evaluated:
            best_candidate, best_result = evaluated[0]
            state.selected_response = best_candidate
            state.confidence = best_result.overall_score

            # 5. VERIFY final check
            if best_result.overall_score >= self.confidence_threshold:
                if verbose:
                    self.thinking_trace.append(f"Selected: {best_candidate}")
                    self.thinking_trace.append(f"Final confidence: {state.confidence:.2f}")

                self.current_state = state
                return best_candidate

        # Confidence too low - don't output
        if verbose:
            self.thinking_trace.append("Confidence too low - not responding")

        self.current_state = state
        return None

    def _extract_decision(self, text: str) -> dict[str, object]:
        """
        Extract decision variables from text.

        If a custom extractor is set, it is used. Otherwise, a simple
        heuristic handles patterns like "x is y".
        """
        if self.decision_extractor is not None:
            return self.decision_extractor(text)

        parts = text.lower().split()
        if len(parts) >= 3:
            if parts[1] in {"is", "are", "was", "were"}:
                subject = parts[0]
                value = " ".join(parts[2:])
                return {subject: value}

        return {}

    def _apply_temporal_validation(
        self,
        evaluated: list[tuple[str, SimulationResult]],
        decision_time: int | None,
        verbose: bool,
    ) -> list[tuple[str, SimulationResult]]:
        """Execute apply temporal validation."""

        if self.temporal_validator is None or self.temporal_world is None:
            return evaluated

        if decision_time is None:
            decision_time = getattr(self.temporal_world, "current_time", 0)

        filtered: list[tuple[str, SimulationResult]] = []
        for candidate, result in evaluated:
            decision = self._extract_decision(candidate)
            if not decision:
                filtered.append((candidate, result))
                continue

            validation = self.temporal_validator.validate_decision(
                decision_time=decision_time,
                decision=decision,
                check_horizon=self.temporal_check_horizon,
            )

            if validation.is_valid:
                filtered.append((candidate, result))
            elif verbose:
                self.thinking_trace.append(
                    f"Temporal reject '{candidate}': {validation.violations}"
                )

        return filtered

    def _generate_candidates(self, understanding: UnderstandingResult) -> list[str]:
        """Generate candidate responses."""
        candidates = []

        # Simple generation based on understanding
        # In practice, this would be more sophisticated

        # If it's a question, generate answer
        if any("?" in word for word in understanding.words):
            # Try to answer based on retrieved knowledge
            if understanding.inferences:
                candidates.append(understanding.inferences[0])
            else:
                candidates.append("I'm not sure.")

        # Generate acknowledgment
        candidates.append("I understand.")

        # Generate response based on inferences
        if understanding.inferences:
            for inf in understanding.inferences[:2]:
                candidates.append(f"Based on that, {inf.lower()}")

        return candidates

    def get_thinking_trace(self) -> list[str]:
        """Get the thinking trace from last process call."""
        return self.thinking_trace.copy()
