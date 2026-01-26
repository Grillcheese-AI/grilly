# PyTorch to Vulkan Conversion Guide

This guide explains how to convert PyTorch tensors to Vulkan-compatible numpy arrays and use them seamlessly with Grilly operations.

## Quick Answer: Yes! ✅

**Yes, you can convert PyTorch tensors to Vulkan!** Grilly provides multiple ways to do this:

1. **Automatic conversion** - Just pass PyTorch tensors directly to `nn.Module`
2. **Manual conversion** - Use `to_vulkan()` function
3. **GPU optimization** - Use `to_vulkan_gpu()` to keep data on GPU (AMD optimized)
4. **Batch conversion** - Convert multiple tensors at once

## Methods

### Method 1: Automatic Conversion (Easiest)

Grilly's `nn.Module` automatically converts PyTorch tensors to numpy arrays:

```python
import torch
from grilly import nn

# Create PyTorch tensor
x = torch.randn(10, 128)

# Create model
linear = nn.Linear(128, 64)

# Pass PyTorch tensor directly - automatically converted!
result = linear(x)  # x is automatically converted to numpy for Vulkan

print(type(result))  # <class 'numpy.ndarray'>
print(result.shape)  # (10, 64)
```

### Method 2: Manual Conversion

Use the `to_vulkan()` function for explicit conversion:

```python
import torch
from grilly.utils import to_vulkan
from grilly import nn

# Create PyTorch tensor
torch_tensor = torch.randn(10, 128).cuda()  # Can be on CUDA

# Convert to Vulkan
vulkan_array = to_vulkan(torch_tensor)

# Use with Vulkan operations
linear = nn.Linear(128, 64)
result = linear(vulkan_array)
```

### Method 3: GPU-Optimized Conversion (AMD)

For AMD GPUs, use `to_vulkan_gpu()` to keep data on GPU and avoid CPU round-trips:

```python
import torch
from grilly.utils import to_vulkan_gpu
from grilly import nn

# Create PyTorch tensor
torch_tensor = torch.randn(10, 128)

# Convert directly to GPU (stays on GPU, no CPU transfer)
vulkan_gpu_tensor = to_vulkan_gpu(torch_tensor)  # Returns VulkanTensor

# Use with Vulkan operations (automatically handles GPU tensor)
linear = nn.Linear(128, 64)
result = linear(vulkan_gpu_tensor)  # Faster on AMD!
```

Or use the `keep_on_gpu` option:

```python
from grilly.utils import to_vulkan

# Keep on GPU for better performance on AMD
vulkan_tensor = to_vulkan(torch_tensor, keep_on_gpu=True)
result = linear(vulkan_tensor)
```

### Method 4: Batch Conversion

Convert multiple tensors at once:

```python
import torch
from grilly.utils import to_vulkan_batch

tensors = [torch.randn(10, 20), torch.randn(5, 30), torch.randn(8, 15)]
vulkan_arrays = to_vulkan_batch(tensors)  # Returns list of numpy arrays
```

## Converting Back to PyTorch

Convert Vulkan results back to PyTorch tensors:

```python
from grilly.utils import from_vulkan
from grilly import nn
import numpy as np

# Process with Vulkan
x = np.random.randn(10, 128).astype(np.float32)
linear = nn.Linear(128, 64)
result = linear(x)  # Vulkan operation

# Convert back to PyTorch
torch_result = from_vulkan(result, device='cuda')  # or 'cpu'
print(type(torch_result))  # <class 'torch.Tensor'>
```

## Complete Workflow Example

```python
import torch
from grilly import nn
from grilly.utils import to_vulkan, from_vulkan

# Step 1: Create PyTorch tensor (e.g., from HuggingFace model)
torch_input = torch.randn(32, 384).cuda()

# Step 2: Convert to Vulkan (or pass directly - auto-converts!)
vulkan_input = to_vulkan(torch_input)  # or just use torch_input directly

# Step 3: Process with Vulkan
model = nn.Sequential(
    nn.Linear(384, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
vulkan_output = model(vulkan_input)  # Runs on Vulkan!

# Step 4: Convert back to PyTorch (if needed)
torch_output = from_vulkan(vulkan_output, device='cuda')

# Step 5: Continue with PyTorch operations
torch_final = torch.softmax(torch_output, dim=-1)
```

## Features

### ✅ Automatic Conversion
- `nn.Module` automatically converts PyTorch tensors
- No manual conversion needed in most cases
- Works with Sequential, all layers

### ✅ GPU Optimization (AMD)
- **`VulkanTensor`**: GPU-resident tensors that stay on GPU
- Avoids CPU round-trips for better performance
- Use `to_vulkan_gpu()` or `keep_on_gpu=True` option
- Automatically handled by `nn.Module`

### ✅ CUDA Support
- Converts CUDA tensors to CPU numpy (for Vulkan)
- Handles detached tensors properly
- Preserves gradients (detached)

### ✅ Batch Operations
- Convert multiple tensors at once
- Preserves list/tuple structure
- Efficient batch processing

### ✅ Type Safety
- Ensures float32 dtype for Vulkan
- Handles various tensor types (PyTorch, TensorFlow)
- Graceful fallbacks

## API Reference

### `to_vulkan(tensor) -> np.ndarray`
Convert any tensor-like object to numpy array for Vulkan.

**Parameters:**
- `tensor`: PyTorch tensor, numpy array, or array-like

**Returns:**
- `np.ndarray`: float32 numpy array ready for Vulkan

### `from_vulkan(array, device='cuda') -> torch.Tensor`
Convert Vulkan numpy array to PyTorch tensor.

**Parameters:**
- `array`: numpy array from Vulkan operations
- `device`: Target device ('cuda', 'cpu', or PyTorch device)

**Returns:**
- `torch.Tensor`: PyTorch tensor on specified device

### `to_vulkan_batch(tensors) -> List/Tuple`
Convert batch of tensors to numpy arrays.

**Parameters:**
- `tensors`: List or tuple of tensors

**Returns:**
- Same structure with numpy arrays

### `to_vulkan_gpu(tensor) -> VulkanTensor`
Convert tensor directly to GPU buffer (stays on GPU, optimized for AMD).

**Parameters:**
- `tensor`: PyTorch tensor, numpy array, or array-like

**Returns:**
- `VulkanTensor`: GPU-resident tensor wrapper

### `VulkanTensor`
GPU-resident tensor class that keeps data on GPU.

**Methods:**
- `.numpy()`: Download to CPU numpy array
- `.cpu()`: Get CPU copy
- `.shape`: Tensor shape
- `.dtype`: Tensor dtype

### `ensure_vulkan_compatible(data) -> np.ndarray`
Ensure data is Vulkan-compatible (numpy, float32). Handles VulkanTensor automatically.

**Parameters:**
- `data`: Any tensor-like data (including VulkanTensor)

**Returns:**
- `np.ndarray`: float32 numpy array

## Examples

See `examples/pytorch_to_vulkan.py` for complete examples.

## Notes

- **AMD/Vulkan Systems**: Works perfectly on AMD GPUs (no CUDA needed)
- **GPU Optimization**: Use `to_vulkan_gpu()` or `keep_on_gpu=True` to keep data on GPU for better performance
- **Automatic**: Most conversions happen automatically
- **Efficient**: Conversions are lightweight (just memory views when possible)
- **GPU Tensors**: `VulkanTensor` keeps data on GPU, avoiding CPU round-trips
- **Safe**: Handles gradients, CUDA tensors, and various dtypes

## See Also

- [HuggingFace Integration Guide](HUGGINGFACE_INTEGRATION.md)
- [Device Management Guide](DEVICE_MANAGEMENT.md)
- [Examples](../examples/pytorch_to_vulkan.py)
