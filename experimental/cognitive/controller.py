"""
CognitiveController - Main "think before speak" controller.

Orchestrates understanding, simulation, and response generation with confidence gating.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from grilly.experimental.language.system import InstantLanguage
from grilly.experimental.cognitive.memory import WorkingMemory
from grilly.experimental.cognitive.world import WorldModel
from grilly.experimental.cognitive.simulator import InternalSimulator
from grilly.experimental.cognitive.understander import Understander
from grilly.experimental.cognitive.understander import UnderstandingResult


@dataclass
class CognitiveState:
    """Current state of the cognitive system."""
    understanding: Optional[UnderstandingResult] = None
    candidates: List[Tuple[str, 'SimulationResult']] = field(default_factory=list)
    selected_response: Optional[str] = None
    confidence: float = 0.0
    thinking_steps: List[str] = field(default_factory=list)


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
        self,
        dim: int = DEFAULT_DIM,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    ):
        self.dim = dim
        self.confidence_threshold = confidence_threshold
        
        # Core components
        self.language = InstantLanguage(dim=dim)
        self.world = WorldModel(dim=dim)
        self.wm = WorkingMemory(dim=dim)
        self.simulator = InternalSimulator(self.language, self.world, self.wm)
        self.understander = Understander(self.language, self.world, self.wm)
        
        # State tracking
        self.current_state: Optional[CognitiveState] = None
        self.thinking_trace: List[str] = []
    
    def add_knowledge(self, subject: str, relation: str, object_: str):
        """Add knowledge to world model."""
        self.world.add_fact(subject, relation, object_)
    
    def understand(self, text: str) -> UnderstandingResult:
        """Just understand without responding."""
        return self.understander.understand(text)
    
    def process(
        self,
        input_text: str,
        verbose: bool = False
    ) -> Optional[str]:
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
                candidate,
                context=understanding.deep_meaning
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
    
    def _generate_candidates(self, understanding: UnderstandingResult) -> List[str]:
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
    
    def get_thinking_trace(self) -> List[str]:
        """Get the thinking trace from last process call."""
        return self.thinking_trace.copy()
