# Compute Shaders

Every neural network operation in grilly is implemented as a GLSL compute shader compiled to SPIR-V bytecode. The framework ships 194 shaders covering all standard operations.

---

## Shader Organization

```
shaders/
├── activation-relu.glsl          # Forward pass
├── activation-relu-backward.glsl # Backward pass
├── attention-*.glsl              # Attention variants
├── conv-*.glsl                   # Convolution
├── linear-*.glsl                 # Linear algebra
├── normalization-*.glsl          # LayerNorm, RMSNorm, BatchNorm
├── loss-*.glsl                   # Loss functions
├── snn-*.glsl                    # SNN neuron dynamics
├── optimizer-*.glsl              # Adam, SGD update kernels
├── spv/                          # Compiled SPIR-V bytecode
└── experimental/                 # Experimental shaders (VSA, etc.)
```

Each operation has a forward shader and (where needed) a backward shader for gradient computation.

---

## Compilation

Shaders are compiled from GLSL to SPIR-V using `glslc` (part of the Vulkan SDK):

```bash
# Single shader
glslc shader.glsl -o spv/shader.spv

# All shaders (Windows)
.\scripts\compile_all_shaders.ps1
```

Pre-compiled SPIR-V is included in the PyPI package. You only need to recompile if you modify shaders or add new ones.

---

## How Dispatch Works

1. **Loading**: At device initialization, `_bridge.py` calls `device.load_shaders(shader_dir)` to load all `.spv` files from `shaders/spv/`.

2. **Pipeline creation**: `pipelines.py` creates a `VkComputePipeline` for each shader + specialization constant combination. Pipelines are LRU-cached.

3. **Dispatch**: The C++ backend (or Python fallback) creates a command buffer, binds the pipeline, binds descriptor sets with input/output buffers, and dispatches the compute workgroups.

4. **Workgroup sizing**: Each shader declares its local workgroup size (typically 256 or 64 threads). The dispatch calculates the number of workgroups based on the data size.

---

## Shader Registry

`backend/shader_registry.py` selects architecture-specific shader variants with a generic fallback:

```python
# The registry picks the right attention output shader
# based on the model architecture
shader = registry.get_shader("attention-output", arch="gpt")
# Returns: "attention-output-gpt.spv"

shader = registry.get_shader("attention-output", arch="unknown")
# Returns: "attention-output.spv" (generic fallback)
```

Architecture-specific variants exist for BERT, GPT, and T5 attention patterns.

---

## Writing a New Shader

1. Create the GLSL file in `shaders/`:

```glsl
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer InputBuf  { float data_in[];  };
layout(set = 0, binding = 1) buffer OutputBuf { float data_out[]; };

layout(push_constant) uniform Params {
    uint n;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= n) return;
    data_out[idx] = max(data_in[idx], 0.0);  // ReLU
}
```

2. Compile to SPIR-V:

```bash
glslc my-shader.glsl -o spv/my-shader.spv
```

3. Register in the backend and call via the bridge or `VulkanCompute`.

---

## Shader Categories

| Category | Count | Examples |
|----------|-------|---------|
| Activations | 17 | relu, gelu, silu, swiglu, tanh, softmax |
| Attention | 20+ | flash-attention, q-similarity, RoPE, KV-cache |
| Convolution | 8 | conv2d forward/backward, im2col, GEMM |
| Linear | 6 | matmul, linear, bias-add |
| Normalization | 10 | layernorm, rmsnorm, batchnorm |
| Loss | 6 | cross-entropy, MSE, BCE |
| SNN | 19 | LIF, IF, STDP, Hebbian |
| Optimizers | 8 | adam-update, adamw-update, sgd-update |
| Memory | 10+ | memory-read, memory-write, memory-gate |
| HDC / VSA | 9 | block-code, circular-conv, Sanger GHA |
| Other | 80+ | embedding, pooling, dropout, bridge, FFT |

---

## Experimental Shaders

9 experimental shaders live in `shaders/experimental/` for VSA (Vector Symbolic Architecture) operations. These are not part of the stable API.
