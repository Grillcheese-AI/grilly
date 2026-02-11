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

from .capsule import CapsuleEncoder, batch_cosine_similarity, cosine_similarity
from .controller import CognitiveController
from .memory import WorkingMemory, WorkingMemoryItem, WorkingMemorySlot
from .simulator import InternalSimulator, SimulationResult
from .understander import Understander, UnderstandingResult
from .world import Fact, WorldModel

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
