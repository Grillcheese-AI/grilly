# Architecture-Specific Shaders

This document explains how to create and use architecture-specific shaders in Grilly.

## Overview

Grilly supports architecture-specific shaders for transformer models. This allows for:
- **Optimized shaders** for specific architectures (e.g., GPT's causal attention)
- **Extensibility** - easy to add support for new architectures
- **Backward compatibility** - generic shaders work for all architectures

## Architecture Support

### Currently Supported Architectures

- **BERT** (`bert`): Bidirectional attention, token type embeddings
- **DistilBERT** (`distilbert`): Bidirectional attention, no token type embeddings
- **RoBERTa** (`roberta`): Similar to BERT, optimized training
- **MPNet** (`mpnet`): Masked and permuted pre-training
- **XLM-RoBERTa** (`xlm-roberta`): Multilingual RoBERTa
- **ALBERT** (`albert`): Factorized embeddings, parameter sharing
- **GPT** (`gpt`): Causal attention, pre-normalization (shader: `attention-output-gpt`)
- **T5** (`t5`): Encoder-decoder with cross-attention (shader: `attention-output-t5`)

### Shader Selection

The shader registry automatically selects the best shader:

1. **Architecture-specific shader** (if available)
2. **Generic shader** (fallback)

Example:
- GPT model → uses `attention-output-gpt` (causal attention optimization)
- Granite model → uses `attention-output-gpt` (same causal attention as GPT)
- BERT model → uses `attention-output` (generic bidirectional)

## Creating Architecture-Specific Shaders

### Step 1: Create the Shader File

Create a new shader file following the naming convention:
```
{shader-name}-{architecture}.glsl
```

Example: `attention-output-gpt.glsl`

### Step 2: Implement Architecture-Specific Logic

The shader should implement the same interface as the generic shader but with architecture-specific optimizations.

**Example: GPT Causal Attention**
```glsl
// attention-output-gpt.glsl
// Optimized for causal attention (can only attend to previous positions)
for (uint k = 0; k <= seq_q && k < seq_len; k++) {
    // Only iterate over positions <= current position
    sum += weights[weight_idx] * V[v_idx];
}
```

### Step 3: Register the Shader

Add the shader to the registry in `grilly/backend/shader_registry.py`:

```python
_registry.register_architecture_specific(
    'attention-output',  # Base shader name
    'gpt',               # Architecture
    'attention-output-gpt'  # Shader file name (without .glsl)
)
```

### Step 4: Compile the Shader

```bash
glslc -fshader-stage=compute shaders/attention-output-gpt.glsl -o shaders/spv/attention-output-gpt.spv
```

## Shader Registry API

### Registering Shaders

```python
from grilly.backend.shader_registry import register_architecture_shader, register_generic_shader

# Register generic shader (works for all architectures)
register_generic_shader('attention-output', 'attention-output')

# Register architecture-specific shader
register_architecture_shader('attention-output', 'gpt', 'attention-output-gpt')
```

### Getting Shaders

```python
from grilly.backend.shader_registry import get_shader

# Get shader for specific architecture
shader_name = get_shader('attention-output', architecture='gpt')
# Returns: 'attention-output-gpt' if available, else 'attention-output'

# Get generic shader
shader_name = get_shader('attention-output')
# Returns: 'attention-output'
```

## When to Create Architecture-Specific Shaders

Create architecture-specific shaders when:

1. **Different attention patterns**: Causal (GPT) vs bidirectional (BERT)
2. **Cross-attention**: Encoder-decoder models (T5)
3. **Performance optimization**: Significant speedup possible
4. **Different normalization**: Pre-norm vs post-norm (if shader-level optimization needed)

**Don't create** architecture-specific shaders for:
- Minor differences handled in Python code
- Differences in weight extraction (handled in `_extract_layer_weights`)
- Differences in token embeddings (handled in `_add_token_type_embeddings`)

## Example: Adding Support for a New Architecture

Let's say you want to add support for **LLaMA** (which uses RoPE and grouped-query attention):

### 1. Create the Shader

```glsl
// attention-output-llama.glsl
// Optimized for LLaMA's grouped-query attention
// (fewer key/value heads than query heads)
```

### 2. Register It

```python
# In shader_registry.py
_registry.register_architecture_specific('attention-output', 'llama', 'attention-output-llama')
```

### 3. Update Model Detection

```python
# In vulkan_sentence_transformer.py
elif 'llama' in model_class_name:
    self.model_type = 'llama'
```

### 4. Compile

```bash
glslc -fshader-stage=compute shaders/attention-output-llama.glsl -o shaders/spv/attention-output-llama.spv
```

## Testing Architecture-Specific Shaders

Test your shader by:

1. **Creating a test model** with the target architecture
2. **Comparing outputs** with the generic shader
3. **Verifying correctness** against reference implementation
4. **Benchmarking performance** to ensure optimization is effective

## Current Architecture-Specific Shaders

| Shader | Architecture | Optimization |
|--------|-------------|--------------|
| `attention-output-gpt` | GPT, Granite | Causal attention (only attend to previous positions) |
| `attention-output-t5` | T5 | Cross-attention (decoder queries attend to encoder values) |

## Best Practices

1. **Always provide a generic fallback** - architecture-specific shaders are optional
2. **Document the optimization** - explain why the architecture-specific shader is needed
3. **Test thoroughly** - ensure correctness matches the generic shader
4. **Benchmark** - verify the optimization actually improves performance
5. **Keep shaders simple** - complex logic should be in Python, not shaders

## Future Enhancements

Potential architecture-specific shaders to add:

- **Flash Attention 2** variants for different architectures
- **Sparse attention** patterns (e.g., Longformer, BigBird)
- **Multi-query attention** (MQA) for faster inference
- **Grouped-query attention** (GQA) for LLaMA-style models
- **LiquidMOE** mixture-of-experts
- **Oja** for neuronal growth / neurogenesis
- **Oja/Sanger/Whitener**
