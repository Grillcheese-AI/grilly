"""
Example: HuggingFace Integration with Grilly

This example shows how to:
1. Use HuggingFace models (tokenizers, transformers) on CUDA
2. Extract embeddings and process them with Vulkan operations
3. Seamlessly switch between CUDA (for HuggingFace) and Vulkan (for custom ops)
"""
import numpy as np

# Import Grilly utilities
from grilly.utils.huggingface_bridge import get_huggingface_bridge
from grilly.utils.device_manager import get_device_manager
from grilly import nn, functional

# Initialize device manager
device_manager = get_device_manager()

# Initialize HuggingFace bridge (uses CUDA)
hf_bridge = get_huggingface_bridge(cuda_device=0)

# Example 1: Encode text and process with Vulkan
def example_encode_and_process():
    """Encode text with HuggingFace model, then process with Vulkan"""
    print("Example 1: Encode and Process")
    
    # Encode text using HuggingFace model on CUDA
    texts = [
        "Hello, how are you?",
        "I'm doing great, thanks!",
        "That's wonderful to hear."
    ]
    
    # Get embeddings from HuggingFace model (runs on CUDA)
    embeddings = hf_bridge.encode(
        texts,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        pool_method='mean'
    )
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings dtype: {embeddings.dtype}")
    
    # Now process with Vulkan operations
    # Create a linear layer using Vulkan
    linear = nn.Linear(embeddings.shape[1], 128)
    
    # Process embeddings with Vulkan
    processed = linear(embeddings)
    print(f"Processed shape: {processed.shape}")
    
    # Apply activation with Vulkan
    activated = functional.relu(processed)
    print(f"Activated shape: {activated.shape}")
    
    return embeddings, processed, activated


# Example 2: Tokenize and generate
def example_tokenize_and_generate():
    """Tokenize text and generate with language model"""
    print("\nExample 2: Tokenize and Generate")
    
    # Load a small language model
    prompt = "The future of AI is"
    
    # Generate text using HuggingFace model on CUDA
    generated = hf_bridge.generate(
        prompt,
        model_name="gpt2",  # Small model for demo
        max_length=50,
        do_sample=True,
        temperature=0.7
    )
    
    print(f"Generated: {generated[0]}")
    
    return generated


# Example 3: Classify text
def example_classify():
    """Classify text using HuggingFace model"""
    print("\nExample 3: Classify Text")
    
    texts = [
        "I love this product!",
        "This is terrible.",
        "It's okay, nothing special."
    ]
    
    # Classify using HuggingFace model on CUDA
    predictions, probabilities = hf_bridge.classify(
        texts,
        model_name="distilbert-base-uncased-finetuned-sst-2-english",
        return_probs=True
    )
    
    print(f"Predictions: {predictions}")
    print(f"Probabilities shape: {probabilities.shape}")
    
    return predictions, probabilities


# Example 4: Mixed workflow (HuggingFace + Vulkan)
def example_mixed_workflow():
    """Use HuggingFace for embeddings, then Vulkan for custom processing"""
    print("\nExample 4: Mixed Workflow")
    
    # Step 1: Get embeddings from HuggingFace (CUDA)
    texts = [
        "Machine learning is fascinating",
        "Deep learning models are powerful",
        "Neural networks can learn complex patterns"
    ]
    
    embeddings = hf_bridge.encode(
        texts,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Step 2: Process with Vulkan operations
    # Create a small neural network using Vulkan
    model = nn.Sequential(
        nn.Linear(embeddings.shape[1], 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64)
    )
    
    # Forward pass with Vulkan
    output = model(embeddings)
    print(f"Input embeddings shape: {embeddings.shape}")
    print(f"Output shape: {output.shape}")
    
    # Step 3: Compute similarity using Vulkan FAISS
    # (This would use Vulkan FAISS operations)
    
    return embeddings, output


# Example 5: Convert between PyTorch and Vulkan
def example_tensor_conversion():
    """Convert tensors between PyTorch CUDA and Vulkan"""
    print("\nExample 5: Tensor Conversion")
    
    import torch
    
    # Create a PyTorch tensor on CUDA
    torch_tensor = torch.randn(10, 128).cuda()
    print(f"PyTorch tensor shape: {torch_tensor.shape}, device: {torch_tensor.device}")
    
    # Convert to numpy for Vulkan
    numpy_array = hf_bridge.to_vulkan(torch_tensor)
    print(f"Numpy array shape: {numpy_array.shape}, dtype: {numpy_array.dtype}")
    
    # Process with Vulkan
    linear = nn.Linear(128, 64)
    processed = linear(numpy_array)
    print(f"Processed shape: {processed.shape}")
    
    # Convert back to PyTorch CUDA tensor if needed
    torch_result = hf_bridge.to_cuda(processed)
    print(f"Back to PyTorch shape: {torch_result.shape}, device: {torch_result.device}")
    
    return numpy_array, processed, torch_result


if __name__ == "__main__":
    print("HuggingFace Integration Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_encode_and_process()
        # example_tokenize_and_generate()  # Requires GPT-2 model
        # example_classify()  # Requires classification model
        example_mixed_workflow()
        example_tensor_conversion()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: Some examples require specific HuggingFace models.")
        print("Install required models or adjust model names in the examples.")
