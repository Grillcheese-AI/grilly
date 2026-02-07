# Experimental Features Examples

This directory contains examples demonstrating all experimental features in Grilly.

## Examples

### VSA Operations
- `experimental_vsa_ops.py` - Binary and holographic VSA operations
- `experimental_vsa_resonator.py` - Resonator network factorization
- `experimental_vsa_batch.py` - Batch bind, bundle, and similarity

### Mixture of Experts
- `experimental_moe.py` - Relational encoding and expert routing
- `experimental_moe_capsule.py` - Capsule-aware expert routing

### Language Learning
- `experimental_language.py` - Instant language learning without training

### Temporal Reasoning
- `experimental_temporal.py` - Temporal encoding, causal chains, counterfactuals

### Cognitive Controller
- `experimental_cognitive.py` - Working memory, world model, cognitive control
- `experimental_chat.py` - Interactive chat interface for responses
- `experimental_cognitive_capsule.py` - Capsule-enhanced working memory and facts
- `experimental_cognitive_temporal_gate.py` - Temporal validation for responses

### GPU Backend
- `experimental_backend_vsa.py` - GPU-accelerated VSA operations

## Running Examples

```bash
# Run individual examples
python examples/experimental_vsa_ops.py
python examples/experimental_vsa_resonator.py
python examples/experimental_vsa_batch.py
python examples/experimental_moe.py
python examples/experimental_moe_capsule.py
python examples/experimental_language.py
python examples/experimental_temporal.py
python examples/experimental_cognitive.py
python examples/experimental_chat.py
python examples/experimental_cognitive_capsule.py
python examples/experimental_cognitive_temporal_gate.py
python examples/experimental_backend_vsa.py

# Run all examples
python -m pytest examples/ -v
```

## Requirements

- Python 3.10+
- numpy
- grilly (with experimental features)
- Vulkan SDK (for GPU examples)

## Notes

- All examples use default dimensions (1024-4096) suitable for demonstration
- GPU examples require compiled shaders in `shaders/experimental/spv/`
- Examples show input/output and key functionality for each module
