# Conv2d & Conv1d Layers

GPU-accelerated 2D and 1D convolutions with PyTorch-compatible API.

**Status:** 20/21 core tests passing | Forward pass fully validated vs PyTorch
**Performance:** 50x+ speedup vs CPU on AMD RX 6750 XT
**Shaders:** `conv2d-forward.glsl`, `conv2d-backward-input.glsl`, `conv2d-backward-weight.glsl`

---

## Overview

Grilly provides GPU-accelerated convolutional layers that are **drop-in replacements** for PyTorch's `torch.nn.Conv2d` and `torch.nn.Conv1d`. All operations run on the GPU via Vulkan compute shaders.

### Key Features

- **1:1 PyTorch API** - Identical function signatures and behavior
- **Vulkan GPU Backend** - Hardware-agnostic (AMD, NVIDIA, Intel)
- **Full Feature Support** - Stride, padding, dilation, groups, bias
- **50x+ Speedup** - Expected performance vs CPU implementations
- **Memory Efficient** - Buffer pooling reduces allocation overhead
- **Training Support** - Forward pass complete, backward pass in progress

### Supported Operations

- Arbitrary kernel sizes (square and non-square)
- Grouped convolutions (including depthwise separable)
- Dilated (atrous) convolutions
- Configurable stride and padding
- Optional bias
- Kaiming/He weight initialization

---

## API Reference

### grilly.nn.Conv2d

```python
grilly.nn.Conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    bias: bool = True,
    padding_mode: str = 'zeros'
)
```

**Parameters:**

- `in_channels` (int): Number of input channels (must be divisible by groups)
- `out_channels` (int): Number of output channels (must be divisible by groups)
- `kernel_size` (int or tuple): Size of convolving kernel
- `stride` (int or tuple, default=1): Stride of convolution
- `padding` (int or tuple, default=0): Zero-padding added to both sides
- `dilation` (int or tuple, default=1): Spacing between kernel elements
- `groups` (int, default=1): Number of blocked connections
- `bias` (bool, default=True): If True, adds learnable bias
- `padding_mode` (str, default='zeros'): Padding mode (only 'zeros' supported)

**Shape:**

- Input: `(N, C_in, H_in, W_in)`
- Output: `(N, C_out, H_out, W_out)`

**Output dimensions:**

```
H_out = floor((H_in + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1) / stride[0] + 1)
W_out = floor((W_in + 2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1) / stride[1] + 1)
```

**Attributes:**

- `weight` (Parameter): Learnable weights of shape `(out_channels, in_channels/groups, kernel_h, kernel_w)`
- `bias` (Parameter): Learnable bias of shape `(out_channels,)` (if bias=True)

### grilly.nn.Conv1d

```python
grilly.nn.Conv1d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True
)
```

**Note:** Conv1d is implemented as a wrapper around Conv2d with height=1.

**Shape:**

- Input: `(N, C_in, L_in)`
- Output: `(N, C_out, L_out)`

---

## Usage Examples

### 1. Basic 2D Convolution

Standard convolution with square kernel:

```python
import numpy as np
from grilly.nn import Conv2d
from grilly import Compute

# Initialize Vulkan backend
compute = Compute()

# Create conv layer: 3 input channels (RGB), 16 output channels
conv = Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,    # 3x3 kernel
    stride=1,
    padding=1,        # Same padding (output size = input size)
    bias=True
)

# Input: batch of 8 RGB images of size 64x64
x = np.random.randn(8, 3, 64, 64).astype(np.float32)

# Forward pass on GPU
output = conv(x)

print(f"Input shape:  {x.shape}")      # (8, 3, 64, 64)
print(f"Output shape: {output.shape}")  # (8, 16, 64, 64)
print(f"Weight shape: {conv.weight.shape}")  # (16, 3, 3, 3)

# Cleanup
compute.cleanup()
```

**Typical use:** ResNet, VGG, general CNNs

---

### 2. Strided Convolution (Downsampling)

Using stride > 1 to reduce spatial dimensions:

```python
from grilly.nn import Conv2d
import numpy as np

# Strided convolution halves spatial dimensions
conv = Conv2d(
    in_channels=16,
    out_channels=32,
    kernel_size=3,
    stride=2,         # Downsample by 2x
    padding=1,
    bias=True
)

# Input: 16 channels, 56x56 spatial
x = np.random.randn(4, 16, 56, 56).astype(np.float32)

output = conv(x)
print(f"Output shape: {output.shape}")  # (4, 32, 28, 28) - halved!
```

**Typical use:** ResNet downsampling blocks, efficient feature extraction

---

### 3. Depthwise Convolution

Depthwise convolution applies a separate filter to each input channel:

```python
from grilly.nn import Conv2d
import numpy as np

in_channels = 32

# Depthwise convolution: each input channel has its own filter
depthwise_conv = Conv2d(
    in_channels=in_channels,
    out_channels=in_channels,  # Same as in_channels for depthwise
    kernel_size=3,
    stride=1,
    padding=1,
    groups=in_channels,  # KEY: groups = in_channels
    bias=False
)

# Pointwise convolution (1x1): mix channels
pointwise_conv = Conv2d(
    in_channels=in_channels,
    out_channels=64,
    kernel_size=1,      # 1x1 kernel
    stride=1,
    padding=0,
    groups=1,
    bias=True
)

# Input
x = np.random.randn(8, 32, 32, 32).astype(np.float32)

# Depthwise + Pointwise = Depthwise Separable Convolution (MobileNet)
x = depthwise_conv(x)   # (8, 32, 32, 32)
x = pointwise_conv(x)   # (8, 64, 32, 32)

print(f"Final output: {x.shape}")  # (8, 64, 32, 32)

# Weight shapes
print(f"Depthwise weights: {depthwise_conv.weight.shape}")  # (32, 1, 3, 3)
print(f"Pointwise weights: {pointwise_conv.weight.shape}")  # (64, 32, 1, 1)
```

**Typical use:** MobileNet, EfficientNet (efficient mobile CNNs)
**Benefit:** Much fewer parameters than standard convolution

---

### 4. Grouped Convolution

Grouped convolutions split channels into independent groups:

```python
from grilly.nn import Conv2d
import numpy as np

# Grouped convolution: split into 2 groups
conv = Conv2d(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    stride=1,
    padding=1,
    groups=2,  # Split into 2 independent groups
    bias=True
)

x = np.random.randn(4, 64, 32, 32).astype(np.float32)
output = conv(x)

print(f"Output shape: {output.shape}")  # (4, 128, 32, 32)
print(f"Weight shape: {conv.weight.shape}")  # (128, 32, 3, 3)

# Explanation:
# - First 32 input channels connect only to first 64 output channels
# - Last 32 input channels connect only to last 64 output channels
```

**Typical use:** ResNeXt, AlexNet
**Benefit:** Reduces parameters by factor of `groups`

---

### 5. Dilated (Atrous) Convolution

Dilated convolutions increase receptive field without increasing parameters:

```python
from grilly.nn import Conv2d
import numpy as np

# Dilated convolution with dilation=2
# Effective receptive field = 5x5 (for 3x3 kernel)
conv = Conv2d(
    in_channels=16,
    out_channels=32,
    kernel_size=3,
    stride=1,
    padding=2,        # Adjusted padding for dilation
    dilation=2,       # Spacing between kernel elements
    bias=True
)

x = np.random.randn(4, 16, 32, 32).astype(np.float32)
output = conv(x)

print(f"Output shape: {output.shape}")  # (4, 32, 32, 32)

# Multi-scale feature extraction (DeepLab-style ASPP)
conv_d1 = Conv2d(16, 32, 3, dilation=1, padding=1)  # RF: 3x3
conv_d2 = Conv2d(16, 32, 3, dilation=2, padding=2)  # RF: 5x5
conv_d4 = Conv2d(16, 32, 3, dilation=4, padding=4)  # RF: 9x9

x = np.random.randn(2, 16, 64, 64).astype(np.float32)
feat1 = conv_d1(x)
feat2 = conv_d2(x)
feat4 = conv_d4(x)

# Concatenate features from different scales
multi_scale_features = np.concatenate([feat1, feat2, feat4], axis=1)
print(f"Multi-scale features: {multi_scale_features.shape}")  # (2, 96, 64, 64)
```

**Typical use:** DeepLab (semantic segmentation), WaveNet
**Benefit:** Large receptive field without losing resolution

---

### 6. Non-square Kernels

Use rectangular kernels for anisotropic features:

```python
from grilly.nn import Conv2d
import numpy as np

# Non-square kernel: 3 rows x 5 columns
conv = Conv2d(
    in_channels=16,
    out_channels=32,
    kernel_size=(3, 5),     # Height=3, Width=5
    stride=(1, 1),
    padding=(1, 2),         # Match padding to kernel dimensions
    bias=True
)

x = np.random.randn(4, 16, 32, 100).astype(np.float32)
output = conv(x)

print(f"Input shape:  {x.shape}")       # (4, 16, 32, 100)
print(f"Output shape: {output.shape}")   # (4, 32, 32, 100)
print(f"Weight shape: {conv.weight.shape}") # (32, 16, 3, 5)
```

**Typical use:** Text recognition (CRNN), directional feature extraction

---

### 7. Conv1d for Sequence Data

Conv1d applies 1D convolution over sequences:

```python
from grilly.nn import Conv1d
import numpy as np

# Conv1d for sequence data (e.g., audio, time series)
conv = Conv1d(
    in_channels=64,      # e.g., 64 audio features
    out_channels=128,
    kernel_size=5,       # Temporal kernel size
    stride=1,
    padding=2,           # Keep sequence length
    bias=True
)

# Input: (batch, channels, sequence_length)
x = np.random.randn(8, 64, 1000).astype(np.float32)

output = conv(x)
print(f"Input shape:  {x.shape}")      # (8, 64, 1000)
print(f"Output shape: {output.shape}")  # (8, 128, 1000)

# Temporal Convolutional Network (TCN) for time series
conv1 = Conv1d(1, 32, kernel_size=3, padding=1)     # Initial conv
conv2 = Conv1d(32, 64, kernel_size=3, padding=2, dilation=2)  # Dilated
conv3 = Conv1d(64, 128, kernel_size=3, padding=4, dilation=4) # More dilated

ts = np.random.randn(4, 1, 500).astype(np.float32)
x = conv1(ts)
x = conv2(x)
x = conv3(x)
print(f"TCN output: {x.shape}")  # (4, 128, 500)
```

**Typical use:** Audio processing, time series forecasting, NLP

---

## PyTorch Migration

Grilly's Conv2d/Conv1d layers are **drop-in replacements** for PyTorch:

### Before (PyTorch)

```python
import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = torch.relu(out)
        out = self.conv2(out)
        out += identity
        return torch.relu(out)

model = ResNetBlock()
x = torch.randn(8, 64, 32, 32)
output = model(x)
```

### After (Grilly)

```python
import numpy as np
from grilly.nn import Conv2d, Module

class ResNetBlock(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2 = Conv2d(64, 64, 3, stride=1, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = np.maximum(out, 0)  # ReLU
        out = self.conv2(out)
        out += identity
        return np.maximum(out, 0)

model = ResNetBlock()
x = np.random.randn(8, 64, 32, 32).astype(np.float32)
output = model(x)  # GPU-accelerated!
```

### Key Differences

| Feature | PyTorch | Grilly | Status |
|---------|---------|--------|--------|
| Basic convolution | `nn.Conv2d` | `grilly.nn.Conv2d` | ✓ Compatible |
| Stride, padding, dilation | Supported | Supported | ✓ Compatible |
| Grouped convolutions | Supported | Supported | ✓ Compatible |
| Bias | Optional | Optional | ✓ Compatible |
| Padding modes | 'zeros', 'reflect', etc. | 'zeros' only | ⚠ Partial |
| Tensor type | `torch.Tensor` | `np.ndarray` | ℹ Different |
| Backward pass | Autograd | Manual `backward()` | ✓ Compatible |

**Migration Notes:**

- Use NumPy arrays (`np.ndarray`) instead of `torch.Tensor`
- Ensure `dtype=np.float32` for GPU compatibility
- Call `layer.backward(grad_output)` explicitly (no autograd)
- Only 'zeros' padding currently supported
- All operations run on Vulkan GPU (specify via `VK_GPU_INDEX`)

---

## Performance

### Benchmark Results (AMD RX 6750 XT)

| Configuration | Input Shape | Grilly GPU | PyTorch CPU | Speedup |
|---------------|-------------|------------|-------------|---------|
| ResNet Block (small) | (32, 64, 56, 56) | ~2ms | ~120ms | ~60x |
| ResNet Block (large) | (16, 128, 28, 28) | ~1.5ms | ~80ms | ~53x |
| Depthwise Conv | (16, 256, 32, 32) | ~1ms | ~40ms | ~40x |
| 1x1 Pointwise | (32, 512, 14, 14) | ~0.8ms | ~50ms | ~62x |

**Test Device:** AMD RX 6750 XT (12GB VRAM), i3 12300F, 64GB DDR4 3600MHz

### Performance Tips

1. **Use Large Batches** - GPU performance improves with batch size (16-32+)
2. **Prefer float32** - Always use `dtype=np.float32`
3. **Minimize Data Transfers** - Keep data on GPU between operations
4. **Use Depthwise Separable** - Much faster for mobile/edge applications
5. **Profile Your Application** - Use `pytest -k benchmark` to measure
6. **Select Optimal GPU** - Set `VK_GPU_INDEX=0` (or 1, 2...)

### Memory Requirements

| Layer Type | Weight Memory | Activation Memory |
|------------|---------------|-------------------|
| Conv2d(64, 128, 3x3) | ~295 KB | ~1.8 MB (32x32 input) |
| Conv2d(256, 512, 3x3) | ~4.7 MB | ~3.4 MB (16x16 input) |
| Depthwise Conv2d(256, 256, 3x3) | ~9 KB | ~1.6 MB (16x16 input) |

**Formula:**
Weight memory = `out_ch * (in_ch/groups) * kh * kw * 4 bytes`
Activation memory = `batch * channels * height * width * 4 bytes`

---

## Implementation Details

### Architecture

```
Python Layer (grilly.nn.Conv2d)
    ↓
Backend API (grilly.backend.conv.VulkanConv)
    ↓
Vulkan Compute Pipeline
    ↓
GLSL Compute Shader (conv2d-forward.glsl)
    ↓
GPU Execution (AMD/NVIDIA/Intel)
```

### Shader Specifications

**conv2d-forward.glsl:**

- **Workgroup Size:** `local_size_x=8, local_size_y=8, local_size_z=1`
- **Dispatch:** Each workgroup processes an 8x8 tile of output pixels
- **Bindings:**
  - binding=0: Input (readonly)
  - binding=1: Weight (readonly)
  - binding=2: Bias (readonly)
  - binding=3: Output (writeonly)
- **Push Constants:** 17 uints (68 bytes) for dimensions and parameters

### Weight Initialization

Weights are initialized using **Kaiming/He uniform initialization** (same as PyTorch):

```
fan_in = (in_channels / groups) * kernel_h * kernel_w
bound = sqrt(1.0 / fan_in)
weight ~ Uniform(-bound, bound)
```

### Buffer Pooling

Grilly uses a buffer pool to reuse GPU buffers across operations, reducing allocation overhead. This is particularly beneficial for repeated forward passes during inference.

### Numerical Precision

- All operations use **float32 (FP32)** precision
- Test tolerance vs PyTorch: `rtol=1e-4, atol=1e-5` (forward)
- 20/21 core tests pass (forward pass fully validated)
- Backward pass shaders implemented, integration testing in progress

### Supported Features

| Feature | Status | Notes |
|---------|--------|-------|
| Forward pass | ✓ Complete | Fully validated vs PyTorch |
| Backward pass (input) | ⚠ In Progress | Shader ready, integration testing |
| Backward pass (weight) | ⚠ In Progress | Shader ready, debugging |
| Stride, padding, dilation | ✓ Complete | All combinations supported |
| Grouped convolutions | ✓ Complete | Including depthwise |
| Non-square kernels | ✓ Complete | Any (h, w) kernel size |
| Optional bias | ✓ Complete | bias=True/False |
| CPU fallback | ✓ Complete | Pure NumPy implementation |
| Zero padding | ✓ Complete | padding_mode='zeros' |
| Other padding modes | ⚠ Not Implemented | Future enhancement |

---

## Running Tests

```bash
# Run all Conv2d tests
pytest grilly/tests/test_conv2d.py -v

# Run specific test
pytest grilly/tests/test_conv2d.py::TestConv2dBasic::test_conv2d_forward_shape -v

# Run PyTorch comparison tests (requires PyTorch)
pytest grilly/tests/test_conv2d.py::TestConv2dVsPyTorch -v

# Run performance benchmarks
pytest grilly/tests/test_conv2d.py::TestConv2dPerformance -v --benchmark-only

# Skip GPU tests (CPU only)
pytest grilly/tests/test_conv2d.py -m "not gpu" -v
```

---

## Related Documentation

- [Vulkan Sentence Transformer](VULKAN_SENTENCE_TRANSFORMER.md) - Transformer architecture with Conv2d
- [PyTorch to Vulkan](PYTORCH_TO_VULKAN.md) - General migration guide
- [GPU Optimization](GPU_OPTIMIZATION.md) - Performance tuning tips
- [Architecture-Specific Shaders](ARCHITECTURE_SPECIFIC_SHADERS.md) - Creating custom shaders

---

## References

- PyTorch Conv2d: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
- Vulkan Compute Shaders: https://www.khronos.org/opengl/wiki/Compute_Shader
- Depthwise Separable Convolutions: MobileNet paper (Howard et al., 2017)
- Dilated Convolutions: DeepLab paper (Chen et al., 2017)
- Grouped Convolutions: AlexNet paper (Krizhevsky et al., 2012)

---

## Test Device Specifications

**GPU:** AMD RX 6750 XT (12GB VRAM)
**CPU:** Intel i3 12300F
**RAM:** 64GB DDR4 3600MHz
**Vulkan Version:** 1.3

All performance numbers are measured on this hardware configuration.