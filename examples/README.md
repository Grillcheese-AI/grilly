# Grilly Examples

## Core Examples

Start here. These show the basics of training and benchmarking with Grilly.

| Script | What it does |
|--------|-------------|
| `hello_grilly.py` | Minimal forward + backward pass with autograd |
| `train_mlp.py` | Full training loop: 3-layer MLP, AdamW, cross-entropy |
| `benchmark_gemm.py` | GPU vs CPU matrix multiply throughput table |

```bash
python examples/hello_grilly.py      # Prints shapes + confirms gradients
python examples/train_mlp.py         # 10 epochs of decreasing loss
python examples/benchmark_gemm.py    # GEMM timing table (requires Vulkan GPU)
```

---

## Experimental Features Examples

14 examples demonstrating advanced research features in Grilly.

### Examples

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
