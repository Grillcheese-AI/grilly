# HuggingFace Integration Guide

This guide explains how to use HuggingFace models (tokenizers, transformers) with Grilly's Vulkan backend.

## Overview

Grilly provides a bridge system that allows you to:
- Run HuggingFace models on CUDA (for compatibility)
- Extract embeddings and process them with Vulkan operations
- Seamlessly convert between PyTorch tensors and numpy arrays
- Use both backends in the same workflow

## Installation

```bash
# Install required dependencies
pip install torch transformers
```

## Quick Start

### Basic Usage

```python
from grilly.utils.huggingface_bridge import get_huggingface_bridge
from grilly import nn, functional

# Initialize HuggingFace bridge (uses CUDA)
hf_bridge = get_huggingface_bridge(cuda_device=0)

# Encode text using HuggingFace model
texts = ["Hello, world!", "How are you?"]
embeddings = hf_bridge.encode(
    texts,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    pool_method='mean'
)

# Process embeddings with Vulkan operations
linear = nn.Linear(embeddings.shape[1], 128)
processed = linear(embeddings)
activated = functional.relu(processed)
```

### Tokenization

```python
# Load tokenizer
tokenizer = hf_bridge.load_tokenizer("bert-base-uncased")

# Tokenize text
encoded = hf_bridge.tokenize(
    "Hello, world!",
    tokenizer,
    return_tensors='np'  # Return numpy arrays for Vulkan
)
```

### Text Generation

```python
# Generate text using a language model
generated = hf_bridge.generate(
    "The future of AI is",
    model_name="gpt2",
    max_length=100,
    do_sample=True,
    temperature=0.7
)
```

### Text Classification

```python
# Classify text
predictions, probabilities = hf_bridge.classify(
    ["I love this!", "This is terrible."],
    model_name="distilbert-base-uncased-finetuned-sst-2-english",
    return_probs=True
)
```

## Tensor Conversion

### PyTorch to Vulkan

```python
import torch

# Create PyTorch tensor on CUDA
torch_tensor = torch.randn(10, 128).cuda()

# Convert to numpy for Vulkan
numpy_array = hf_bridge.to_vulkan(torch_tensor)

# Process with Vulkan
result = nn.Linear(128, 64)(numpy_array)
```

### Vulkan to PyTorch

```python
# Process with Vulkan
numpy_array = np.random.randn(10, 128).astype(np.float32)
processed = nn.Linear(128, 64)(numpy_array)

# Convert back to PyTorch CUDA tensor
torch_tensor = hf_bridge.to_cuda(processed)
```

## Device Management

```python
from grilly.utils.device_manager import get_device_manager

# Get device manager
device_manager = get_device_manager()

# Set device
device_manager.set_device('cuda', cuda_index=0)  # Use CUDA device 0
device_manager.set_device('vulkan')  # Use Vulkan

# Get backends
vulkan_backend = device_manager.vulkan  # Vulkan backend
cuda_backend = device_manager.cuda     # PyTorch CUDA
torch = device_manager.torch            # PyTorch module
```

## PyTorch Compatibility

Grilly provides PyTorch-like operations that use Vulkan backend:

```python
from grilly.utils.pytorch_ops import add, mul, matmul, relu, gelu, softmax
from grilly.utils.pytorch_compat import tensor, zeros, ones

# Create tensors
x = tensor(np.random.randn(10, 128))
y = tensor(np.random.randn(128, 64))

# Operations
z = matmul(x, y)        # Matrix multiplication (Vulkan)
a = relu(z)             # ReLU activation (Vulkan)
b = softmax(a, dim=-1)  # Softmax (Vulkan)
```

## Common PyTorch Operations (80% Coverage)

Grilly implements the most commonly used PyTorch operations:

### Basic Operations
- `add`, `mul`, `matmul`, `bmm` (batch matrix multiplication)

### Activations
- `relu`, `gelu`, `softmax`, `sigmoid`, `tanh`

### Normalization
- `layer_norm`, `batch_norm`

### Convolution & Pooling
- `conv2d`, `max_pool2d`, `avg_pool2d`

### Loss Functions
- `mse_loss`, `cross_entropy_loss`

### Utilities
- `flatten`, `reshape`, `transpose`, `unsqueeze`, `squeeze`

## Example Workflow

```python
from grilly.utils.huggingface_bridge import get_huggingface_bridge
from grilly import nn, functional
from grilly.utils.pytorch_ops import matmul, relu, softmax

# 1. Initialize bridge
hf_bridge = get_huggingface_bridge()

# 2. Get embeddings from HuggingFace (CUDA)
texts = ["Machine learning", "Deep learning", "Neural networks"]
embeddings = hf_bridge.encode(
    texts,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 3. Build model with Vulkan operations
model = nn.Sequential(
    nn.Linear(embeddings.shape[1], 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# 4. Forward pass (Vulkan)
output = model(embeddings)

# 5. Apply softmax (Vulkan)
probs = softmax(output, dim=-1)

print(f"Input shape: {embeddings.shape}")
print(f"Output shape: {output.shape}")
print(f"Probabilities shape: {probs.shape}")
```

## Performance Tips

1. **Batch Processing**: Process multiple texts at once for better GPU utilization
2. **Model Caching**: Models and tokenizers are cached automatically
3. **Mixed Backends**: Use CUDA for HuggingFace, Vulkan for custom operations
4. **Tensor Conversion**: Minimize conversions between PyTorch and numpy

## Limitations

- HuggingFace models run on CUDA (not Vulkan) for compatibility
- Some advanced PyTorch features may not be available
- Gradient computation requires PyTorch tensors

## See Also

- [PyTorch Compatibility Guide](PYTORCH_COMPAT.md)
- [Device Management Guide](DEVICE_MANAGEMENT.md)
- [Examples](../examples/huggingface_integration.py)
