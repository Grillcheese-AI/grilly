"""
Example: Cognitive Controller with Temporal Validation

Demonstrates candidate filtering based on temporal constraints.
"""

import numpy as np
from grilly.experimental.cognitive import CognitiveController
from grilly.experimental.cognitive.simulator import SimulationResult
from grilly.experimental.temporal import (
    CounterfactualReasoner,
    TemporalDecisionValidator,
    TemporalWorldModel,
)


def main() -> None:
    dim = 1024
    controller = CognitiveController(dim=dim, confidence_threshold=0.1)

    world = TemporalWorldModel(dim=dim)
    world.set_state(0, {"status": "alive"})
    reasoner = CounterfactualReasoner(world)
    validator = TemporalDecisionValidator(world, reasoner)

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
    print("Selected response:", response)
    print("Thinking trace:")
    for step in controller.thinking_trace:
        print(f"  {step}")


if __name__ == "__main__":
    main()
