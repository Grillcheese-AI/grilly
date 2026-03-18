# API Reference: utils

`grilly.utils` provides data loading, tensor conversion, checkpointing, device management, and HuggingFace integration.

Source: [`utils/`](https://github.com/grillcheese-ai/grilly/tree/main/utils)

---

## Data Loading

| Class | Signature | Description |
|-------|-----------|-------------|
| `Dataset` | `Dataset()` | Abstract base class. Override `__len__` and `__getitem__`. |
| `TensorDataset` | `TensorDataset(*tensors)` | Dataset from numpy arrays. |
| `ArrayDataset` | `ArrayDataset(*arrays)` | Dataset from arrays (alias). |
| `Subset` | `Subset(dataset, indices)` | Subset of a dataset by indices. |
| `ConcatDataset` | `ConcatDataset(datasets)` | Concatenation of datasets. |
| `DataLoader` | `DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)` | Iterable batch loader. |
| `random_split` | `random_split(dataset, lengths)` | Split dataset into random subsets. |

### Samplers

| Class | Description |
|-------|-------------|
| `RandomSampler` | Sample indices randomly. |
| `SequentialSampler` | Sample indices in order. |
| `BatchSampler` | Wrap a sampler to yield batches. |

### Transforms

| Class | Description |
|-------|-------------|
| `Compose` | Chain transforms. |
| `ToFloat32` | Convert to float32. |
| `Normalize` | Normalize with mean and std. |
| `Flatten` | Flatten to 1D. |
| `RandomNoise` | Add random noise (augmentation). |
| `RandomFlip` | Random horizontal flip (augmentation). |
| `OneHot` | One-hot encode labels. |
| `Lambda` | Apply an arbitrary function. |

---

## Tensor Conversion

| Function | Description |
|----------|-------------|
| `to_vulkan(array)` | Upload numpy array to GPU as VulkanTensor. |
| `to_vulkan_batch(arrays)` | Batch upload multiple arrays. |
| `from_vulkan(vt)` | Download VulkanTensor to numpy. |
| `ensure_vulkan_compatible(array)` | Ensure array is float32 and contiguous. |

| Class | Description |
|-------|-------------|
| `VulkanTensor` | Wrapper around a GPU buffer. Supports zero-copy access. |

---

## Checkpointing

| Function | Signature | Description |
|----------|-----------|-------------|
| `save_checkpoint` | `save_checkpoint(model, optimizer, epoch, path)` | Save model + optimizer state. |
| `load_checkpoint` | `load_checkpoint(model, optimizer, path)` | Load model + optimizer state. |

---

## Device Management

| Function | Description |
|----------|-------------|
| `get_device()` | Get the current device name. |
| `set_device(index)` | Set the active GPU by index. |
| `device_count()` | Number of available Vulkan devices. |

| Class | Description |
|-------|-------------|
| `DeviceManager` | Multi-backend device manager (Vulkan + CUDA). |

---

## Weight Initialization

| Function | Description |
|----------|-------------|
| `xavier_uniform_(array)` | Xavier/Glorot uniform initialization. |
| `xavier_normal_(array)` | Xavier/Glorot normal initialization. |
| `kaiming_uniform_(array)` | Kaiming/He uniform initialization. |
| `kaiming_normal_(array)` | Kaiming/He normal initialization. |

---

## HuggingFace Bridge

| Class | Description |
|-------|-------------|
| `HuggingFaceBridge` | Load pretrained HuggingFace weights without PyTorch runtime. |

```python
from grilly.utils import HuggingFaceBridge

bridge = HuggingFaceBridge()
weights = bridge.load_weights("bert-base-uncased")
```

---

## ONNX

| Class | Description |
|-------|-------------|
| `OnnxModelLoader` | Load ONNX models into grilly modules. |
| `OnnxExporter` | Export grilly models to ONNX format. |
| `OnnxFineTuner` | Fine-tune ONNX models with grilly optimizers. |
| `GrillyOnnxModel` | Grilly wrapper for loaded ONNX models. |

---

## PyTorch Compatibility

| Symbol | Description |
|--------|-------------|
| `Tensor` | Drop-in Tensor class with PyTorch-compatible API. |
| `tensor`, `zeros`, `ones`, `randn` | Factory functions matching PyTorch signatures. |

---

## Streaming Pipeline

| Class | Description |
|-------|-------------|
| `StreamingPipeline` | Batched embed + upload pipeline for inference. |

---

## Visualization (Optional)

Requires matplotlib.

| Function | Description |
|----------|-------------|
| `plot_training_history(history)` | Plot loss/accuracy curves. |
| `plot_gradient_flow(model)` | Visualize gradient magnitudes per layer. |
| `print_model_summary(model)` | Print model architecture summary. |
