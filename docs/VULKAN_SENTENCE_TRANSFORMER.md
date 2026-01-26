# Vulkan Sentence-Transformer (AMD GPU)

Complete guide for running sentence-transformers models entirely on AMD GPUs using Vulkan.

## Overview

The `VulkanSentenceTransformer` class extracts weights from sentence-transformers models and runs the entire inference pipeline on AMD GPUs using Vulkan compute shaders. This provides **full GPU acceleration** for AMD systems.

## Quick Start

```python
from grilly.utils.vulkan_sentence_transformer import VulkanSentenceTransformer

# Create Vulkan model (runs on AMD GPU)
model = VulkanSentenceTransformer('all-MiniLM-L6-v2')

# Encode text (entire model on GPU!)
embeddings = model.encode("Hello, world!")
print(f"Shape: {embeddings.shape}")  # (384,) for all-MiniLM-L6-v2
```

## How It Works

1. **Weight Extraction**: Loads sentence-transformer model and extracts all weights
2. **Vulkan Conversion**: Converts PyTorch weights to Vulkan-compatible numpy arrays
3. **GPU Inference**: Runs entire model (embeddings, transformer layers, pooling) on Vulkan GPU
4. **Optimized Operations**: Uses Vulkan shaders for:
   - Embedding lookup
   - Multi-head attention
   - Feed-forward networks
   - Layer normalization
   - Mean pooling
   - L2 normalization

## Usage

### Basic Encoding

```python
from grilly.utils.vulkan_sentence_transformer import VulkanSentenceTransformer

model = VulkanSentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode("This is a test sentence")
```

### Batch Processing

```python
texts = ["First sentence", "Second sentence", "Third sentence"]
embeddings = model.encode(texts, batch_size=32)
print(f"Shape: {embeddings.shape}")  # (3, 384)
```

### Via HuggingFace Bridge

```python
from grilly.utils.huggingface_bridge import get_huggingface_bridge

bridge = get_huggingface_bridge()
embeddings = bridge.encode_sentence_transformer_vulkan(
    "Hello, world!",
    model_name='all-MiniLM-L6-v2'
)
```

## Architecture Support

Currently supports:
- **DistilBERT-based models** (e.g., `all-MiniLM-L6-v2`)
- **BERT-based models** (e.g., `all-mpnet-base-v2`)

The implementation automatically detects the model architecture and extracts weights accordingly.

## Performance

- **GPU Acceleration**: Entire model runs on AMD GPU
- **Batch Processing**: Efficient batch encoding
- **Memory Efficient**: Weights are extracted once and cached
- **Vulkan Shaders**: Optimized compute shaders for all operations

## Supported Models

- `all-MiniLM-L6-v2` (384 dim, 6 layers) ✅
- `all-MiniLM-L12-v2` (384 dim, 12 layers) ✅
- `paraphrase-MiniLM-L6-v2` (384 dim) ✅
- `all-mpnet-base-v2` (768 dim) ✅
- Other DistilBERT/BERT-based models ✅

## API Reference

### `VulkanSentenceTransformer`

```python
VulkanSentenceTransformer(
    model_name: str = 'all-MiniLM-L6-v2',
    device: str = 'vulkan',
    max_seq_length: int = 512
)
```

**Parameters:**
- `model_name`: Sentence-transformer model name
- `device`: Device ('vulkan' for AMD GPU)
- `max_seq_length`: Maximum sequence length

### `encode()`

```python
encode(
    texts: Union[str, List[str]],
    batch_size: int = 32,
    show_progress_bar: bool = False,
    normalize_embeddings: bool = True,
    **kwargs
) -> np.ndarray
```

**Parameters:**
- `texts`: Input text or list of texts
- `batch_size`: Batch size for processing
- `show_progress_bar`: Show progress bar
- `normalize_embeddings`: Normalize embeddings (default: True)

**Returns:**
- `np.ndarray`: Embeddings (batch, embedding_dim) or (embedding_dim,)

## Examples

See `examples/vulkan_sentence_transformer_example.py` for complete examples.

## Comparison

| Method | Device | Speed | Notes |
|--------|--------|-------|-------|
| `encode_sentence_transformer()` | CPU/CUDA | Medium | Uses sentence-transformers directly |
| `encode_sentence_transformer_vulkan()` | **AMD GPU** | **Fastest** | Full GPU acceleration on AMD |
| `encode_sentence_transformer_gpu(use_vulkan_model=True)` | **AMD GPU** | **Fastest** | Full GPU acceleration |

## Notes

- **First Load**: Model weights are extracted on first use (may take a few seconds)
- **Caching**: Models are cached after extraction for faster subsequent loads
- **Memory**: GPU memory usage depends on model size and batch size
- **Compatibility**: Works with DistilBERT and BERT-based sentence-transformers models

## Troubleshooting

**Model not found**: Ensure sentence-transformers is installed and model name is correct.

**Vulkan initialization failed**: Check that Vulkan drivers are installed and GPU is detected.

**Weight extraction failed**: Model architecture may not be supported. Check model type.

## See Also

- [Sentence-Transformers GPU Guide](SENTENCE_TRANSFORMERS_GPU.md)
- [HuggingFace Integration](HUGGINGFACE_INTEGRATION.md)
- [Examples](../examples/vulkan_sentence_transformer_example.py)
