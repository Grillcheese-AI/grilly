"""
grilly.experimental.cognitive - Cognitive controller with "think before speak".

Provides working memory, world model, internal simulation, and cognitive control
for understanding and generating coherent responses.

Submodules:
    - memory: WorkingMemory for internal scratchpad
    - world: WorldModel for knowledge and coherence checking
    - simulator: InternalSimulator for "think before speak"
    - controller: CognitiveController for full pipeline
"""

from .memory import WorkingMemory, WorkingMemorySlot, WorkingMemoryItem
from .world import WorldModel, Fact
from .simulator import InternalSimulator, SimulationResult
from .understander import Understander, UnderstandingResult
from .controller import CognitiveController
from .capsule import CapsuleEncoder, cosine_similarity, batch_cosine_similarity

__all__ = [
    "WorkingMemory",
    "WorkingMemorySlot",
    "WorkingMemoryItem",
    "WorldModel",
    "Fact",
    "InternalSimulator",
    "SimulationResult",
    "Understander",
    "UnderstandingResult",
    "CognitiveController",
    "CapsuleEncoder",
    "cosine_similarity",
    "batch_cosine_similarity",
]
