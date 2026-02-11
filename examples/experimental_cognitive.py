"""
Example: Cognitive Controller

Demonstrates working memory, world model, internal simulation,
and cognitive control for understanding and generating responses.
"""

from grilly.experimental.cognitive import (
    CognitiveController,
    InternalSimulator,
    WorkingMemory,
    WorkingMemorySlot,
    WorldModel,
)

print("=" * 60)
print("Cognitive Controller Examples")
print("=" * 60)

dim = 2048

# Working Memory
print("\n1. Working Memory")
print("-" * 60)

wm = WorkingMemory(dim=dim, capacity=5)

from grilly.experimental.vsa import HolographicOps

item1 = HolographicOps.random_vector(dim)
item2 = HolographicOps.random_vector(dim)

idx1 = wm.add(item1, "item1", WorkingMemorySlot.CONTEXT)
idx2 = wm.add(item2, "item2", WorkingMemorySlot.CONTEXT)

print(f"Added {len(wm.items)} items to working memory")
print(f"Item 1 index: {idx1}, Item 2 index: {idx2}")

wm.attend(idx1)
print(f"Focused on item {idx1}")

context = wm.get_context_vector()
print(f"Context vector: shape={context.shape}")

# World Model
print("\n2. World Model")
print("-" * 60)

world = WorldModel(dim=dim)

world.add_fact("dog", "is", "animal")
world.add_fact("cat", "is", "animal")
world.add_fact("dog", "has", "fur")

print(f"Added {len(world.facts)} facts to world model")

found = world.query_fact("dog", "is", "animal")
print(f"Query 'dog is animal': {found}")

from grilly.experimental.language import SentenceEncoder, WordEncoder

word_encoder = WordEncoder(dim=dim)
sentence_encoder = SentenceEncoder(word_encoder)

contradiction_words = ["dog", "is", "not", "animal"]
contradiction_vec = sentence_encoder.encode_sentence(contradiction_words)

is_coherent, confidence, reason = world.check_coherence(contradiction_vec)
print("\nCoherence check for contradiction:")
print(f"Coherent: {is_coherent}, Confidence: {confidence:.4f}")
print(f"Reason: {reason}")

# Internal Simulator
print("\n3. Internal Simulator")
print("-" * 60)

from grilly.experimental.language import InstantLanguage

lang = InstantLanguage(dim=dim)
simulator = InternalSimulator(lang, world, wm)

candidate = "The dog is a friendly animal"
result = simulator.simulate_utterance(candidate)

print(f"Candidate utterance: {candidate}")
print(f"Coherence score: {result.coherence_score:.4f}")
print(f"Social appropriateness: {result.social_appropriateness:.4f}")
print(f"Confidence: {result.confidence:.4f}")
print(f"Overall score: {result.overall_score:.4f}")

# Cognitive Controller
print("\n4. Cognitive Controller")
print("-" * 60)

controller = CognitiveController(dim=dim, confidence_threshold=0.5)

controller.world.add_fact("dog", "is", "animal")
controller.world.add_fact("cat", "is", "animal")

input_text = "What is a dog?"
response = controller.process(input_text)

print(f"Input: {input_text}")
print(f"Response: {response}")
print(f"Thinking trace length: {len(controller.thinking_trace)}")

if controller.thinking_trace:
    print("Thinking steps:")
    for i, step in enumerate(controller.thinking_trace[:3], 1):
        print(f"  {i}. {step}")

# Understanding
print("\n5. Understanding")
print("-" * 60)

understanding = controller.understand("The cat sat on the mat")

print("Input: 'The cat sat on the mat'")
print(f"Words: {understanding.words}")
print(f"Parsed roles: {list(understanding.parsed_roles.items())[:3]}")
print(f"Inferences: {len(understanding.inferences)}")
print(f"Questions: {understanding.questions}")
print(f"Confidence: {understanding.confidence:.4f}")
