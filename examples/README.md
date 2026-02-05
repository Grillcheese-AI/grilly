# Experimental Features Examples

This directory contains examples demonstrating all experimental features in Grilly.

## Examples

### VSA Operations
- `experimental_vsa_ops.py` - Binary and holographic VSA operations
- `experimental_vsa_resonator.py` - Resonator network factorization

### Mixture of Experts
- `experimental_moe.py` - Relational encoding and expert routing

### Language Learning
- `experimental_language.py` - Instant language learning without training

### Temporal Reasoning
- `experimental_temporal.py` - Temporal encoding, causal chains, counterfactuals

### Cognitive Controller
- `experimental_cognitive.py` - Working memory, world model, cognitive control

### GPU Backend
- `experimental_backend_vsa.py` - GPU-accelerated VSA operations

## Running Examples

```bash
# Run individual examples
python examples/experimental_vsa_ops.py
python examples/experimental_vsa_resonator.py
python examples/experimental_moe.py
python examples/experimental_language.py
python examples/experimental_temporal.py
python examples/experimental_cognitive.py
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
