# Grilly Test Suite

Comprehensive unit tests for the Grilly SDK.

## Test Structure

- `test_core.py` - Core functionality and initialization tests
- `test_snn.py` - Spiking Neural Network operations
- `test_gpu_operations.py` - GPU-accelerated operations (LIF, FNN, FAISS)
- `test_memory_operations.py` - Memory read/write operations
- `test_attention.py` - Attention mechanism operations
- `test_learning.py` - Learning algorithms (whitening, NLMS, etc.)
- `test_integration.py` - End-to-end integration tests
- `conftest.py` - Pytest fixtures and configuration

## Running Tests

### Run all tests
```bash
pytest grilly/tests/
```

### Run specific test file
```bash
pytest grilly/tests/test_snn.py
```

### Run with verbose output
```bash
pytest grilly/tests/ -v
```

### Run only CPU tests (skip GPU tests)
```bash
pytest grilly/tests/ -m "not gpu"
```

### Run only GPU tests (requires Vulkan)
```bash
pytest grilly/tests/ -k "gpu"
```

## Test Categories

### CPU Tests
These tests work without GPU and test CPU fallback functionality:
- SNN initialization and basic operations
- SNN forward pass and process method
- Reproducibility tests

### GPU Tests
These tests require Vulkan and test GPU-accelerated operations:
- LIF neuron dynamics
- FNN operations (activations, layer norm)
- FAISS similarity search
- Memory operations
- Attention mechanisms
- Learning algorithms

## Requirements

- Python >= 3.10
- pytest >= 7.4.0
- numpy >= 1.24.0
- vulkan >= 1.3.0 (for GPU tests)

## Notes

- GPU tests will be automatically skipped if Vulkan is not available
- Some tests use random data with fixed seeds for reproducibility
- Integration tests may take longer to run
