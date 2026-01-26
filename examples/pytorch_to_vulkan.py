"""
Example: Converting PyTorch Tensors to Vulkan

This example demonstrates how to seamlessly convert PyTorch tensors
to Vulkan-compatible numpy arrays and use them with Grilly operations.
"""
import numpy as np

# Import Grilly
from grilly import nn, functional
from grilly.utils.tensor_conversion import to_vulkan, from_vulkan, to_vulkan_batch

# Example 1: Basic conversion
def example_basic_conversion():
    """Basic PyTorch tensor to Vulkan conversion"""
    print("Example 1: Basic Conversion")
    
    try:
        import torch
        
        # Create PyTorch tensor
        torch_tensor = torch.randn(10, 128, dtype=torch.float32)
        print(f"PyTorch tensor: shape={torch_tensor.shape}, dtype={torch_tensor.dtype}")
        
        # Convert to Vulkan (numpy)
        vulkan_array = to_vulkan(torch_tensor)
        print(f"Vulkan array: shape={vulkan_array.shape}, dtype={vulkan_array.dtype}")
        
        # Use with Vulkan operations
        linear = nn.Linear(128, 64)
        result = linear(vulkan_array)
        print(f"Vulkan result: shape={result.shape}, dtype={result.dtype}")
        
        # Convert back to PyTorch if needed
        torch_result = from_vulkan(result, device='cpu')
        print(f"Back to PyTorch: shape={torch_result.shape}, dtype={torch_result.dtype}")
        
    except ImportError:
        print("PyTorch not available, skipping PyTorch tensor example")
        # Use numpy directly
        numpy_array = np.random.randn(10, 128).astype(np.float32)
        linear = nn.Linear(128, 64)
        result = linear(numpy_array)
        print(f"Direct numpy: shape={result.shape}")


# Example 2: Automatic conversion in Module
def example_automatic_conversion():
    """Automatic conversion in nn.Module (no manual conversion needed)"""
    print("\nExample 2: Automatic Conversion")
    
    try:
        import torch
        
        # Create PyTorch tensor
        torch_tensor = torch.randn(5, 256, dtype=torch.float32)
        print(f"Input: PyTorch tensor, shape={torch_tensor.shape}")
        
        # Create model
        model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        
        # Pass PyTorch tensor directly - automatically converted!
        result = model(torch_tensor)
        print(f"Output: numpy array, shape={result.shape}")
        print("Note: PyTorch tensor was automatically converted to numpy for Vulkan")
        
    except ImportError:
        print("PyTorch not available, using numpy directly")
        numpy_array = np.random.randn(5, 256).astype(np.float32)
        model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        result = model(numpy_array)
        print(f"Result shape: {result.shape}")


# Example 3: Batch conversion
def example_batch_conversion():
    """Convert multiple PyTorch tensors at once"""
    print("\nExample 3: Batch Conversion")
    
    try:
        import torch
        
        # Create multiple PyTorch tensors
        x = torch.randn(10, 128)
        y = torch.randn(10, 128)
        z = torch.randn(10, 128)
        
        # Convert all at once
        x_vulkan, y_vulkan, z_vulkan = to_vulkan_batch([x, y, z])
        
        print(f"Converted {len([x, y, z])} tensors to Vulkan")
        print(f"All shapes: {[arr.shape for arr in [x_vulkan, y_vulkan, z_vulkan]]}")
        
        # Use with Vulkan operations
        linear = nn.Linear(128, 64)
        results = [linear(arr) for arr in [x_vulkan, y_vulkan, z_vulkan]]
        print(f"Processed {len(results)} arrays with Vulkan")
        
    except ImportError:
        print("PyTorch not available")


# Example 4: Mixed workflow (PyTorch → Vulkan → PyTorch)
def example_mixed_workflow():
    """Complete workflow: PyTorch → Vulkan → PyTorch"""
    print("\nExample 4: Mixed Workflow")
    
    try:
        import torch
        
        # Step 1: Create PyTorch tensor (e.g., from HuggingFace model)
        torch_input = torch.randn(32, 384, dtype=torch.float32)
        print(f"Step 1 - PyTorch input: {torch_input.shape}")
        
        # Step 2: Convert to Vulkan
        vulkan_input = to_vulkan(torch_input)
        print(f"Step 2 - Vulkan input: {vulkan_input.shape}")
        
        # Step 3: Process with Vulkan operations
        model = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        vulkan_output = model(vulkan_input)
        print(f"Step 3 - Vulkan output: {vulkan_output.shape}")
        
        # Step 4: Convert back to PyTorch (if needed for further processing)
        torch_output = from_vulkan(vulkan_output, device='cpu')
        print(f"Step 4 - PyTorch output: {torch_output.shape}")
        
        # Step 5: Use with PyTorch operations
        torch_final = torch.softmax(torch_output, dim=-1)
        print(f"Step 5 - Final PyTorch result: {torch_final.shape}")
        
    except ImportError:
        print("PyTorch not available, using numpy only")
        numpy_input = np.random.randn(32, 384).astype(np.float32)
        model = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        result = model(numpy_input)
        print(f"Result: {result.shape}")


# Example 5: GPU-Optimized Conversion (AMD)
def example_gpu_optimized():
    """GPU-optimized conversion for AMD GPUs"""
    print("\nExample 5: GPU-Optimized Conversion (AMD)")
    
    try:
        import torch
        from grilly.utils import to_vulkan_gpu
        from grilly import nn
        
        # Create PyTorch tensor
        torch_tensor = torch.randn(32, 256, dtype=torch.float32)
        print(f"Input: PyTorch tensor, shape={torch_tensor.shape}")
        
        # Convert directly to GPU (stays on GPU, no CPU round-trip)
        gpu_tensor = to_vulkan_gpu(torch_tensor)
        print(f"GPU tensor: {gpu_tensor}")
        print("Note: Data stays on GPU, avoiding CPU transfer")
        
        # Use with Vulkan operations
        model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Pass GPU tensor directly - automatically handled
        result = model(gpu_tensor)
        print(f"Output: numpy array, shape={result.shape}")
        print("Note: GPU tensor was automatically converted for processing")
        
        # Or use keep_on_gpu option
        gpu_tensor2 = to_vulkan(torch_tensor, keep_on_gpu=True)
        result2 = model(gpu_tensor2)
        print(f"Alternative method result: shape={result2.shape}")
        
    except ImportError:
        print("PyTorch not available")
    except Exception as e:
        print(f"GPU optimization may not be available: {e}")
        # Fall back to regular conversion
        numpy_array = np.random.randn(32, 256).astype(np.float32)
        model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )
        result = model(numpy_array)
        print(f"Fallback result: shape={result.shape}")


# Example 6: Using with HuggingFace
def example_huggingface_integration():
    """Integration with HuggingFace models"""
    print("\nExample 5: HuggingFace Integration")
    
    try:
        from grilly.utils.huggingface_bridge import get_huggingface_bridge
        import torch
        
        # Initialize bridge
        hf_bridge = get_huggingface_bridge()
        
        # Get embeddings from HuggingFace (PyTorch CUDA tensor)
        texts = ["Hello, world!", "How are you?"]
        try:
            embeddings = hf_bridge.encode(
                texts,
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            print(f"HuggingFace embeddings: {embeddings.shape}")
            
            # Embeddings are already numpy (from bridge), but if they were PyTorch:
            # embeddings_vulkan = to_vulkan(embeddings)  # Would convert if needed
            
            # Process with Vulkan
            linear = nn.Linear(embeddings.shape[1], 128)
            result = linear(embeddings)
            print(f"Vulkan processed: {result.shape}")
            
        except Exception as e:
            print(f"HuggingFace model not available: {e}")
            # Use dummy data
            dummy_embeddings = np.random.randn(2, 384).astype(np.float32)
            linear = nn.Linear(384, 128)
            result = linear(dummy_embeddings)
            print(f"Dummy Vulkan result: {result.shape}")
            
    except ImportError:
        print("HuggingFace bridge not available")


if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch to Vulkan Conversion Examples")
    print("=" * 60)
    
    example_basic_conversion()
    example_automatic_conversion()
    example_batch_conversion()
    example_mixed_workflow()
    example_gpu_optimized()
    example_huggingface_integration()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
