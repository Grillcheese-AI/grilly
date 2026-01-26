"""
Example: Using sentence-transformers with GPU acceleration

Demonstrates how to use sentence-transformers on GPU (CUDA) or CPU (AMD/Vulkan)
with Vulkan post-processing for optimal performance.
"""
import numpy as np

try:
    from grilly.utils.huggingface_bridge import get_huggingface_bridge
    from grilly import nn, functional
    from grilly.utils.tensor_conversion import to_vulkan_gpu
    BRIDGE_AVAILABLE = True
except ImportError:
    BRIDGE_AVAILABLE = False
    print("Grilly not available")


def example_basic_encoding():
    """Basic sentence-transformers encoding"""
    print("Example 1: Basic Encoding")
    
    if not BRIDGE_AVAILABLE:
        print("Grilly not available")
        return
    
    try:
        bridge = get_huggingface_bridge()
        
        # Encode single text
        text = "This is a test sentence"
        embeddings = bridge.encode_sentence_transformer(text)
        
        print(f"Text: {text}")
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Embedding dtype: {embeddings.dtype}")
        print(f"Device: {'CUDA' if bridge.cuda_device else 'CPU (AMD/Vulkan compatible)'}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_batch_encoding():
    """Batch encoding with sentence-transformers"""
    print("\nExample 2: Batch Encoding")
    
    if not BRIDGE_AVAILABLE:
        print("Grilly not available")
        return
    
    try:
        bridge = get_huggingface_bridge()
        
        texts = [
            "First sentence",
            "Second sentence",
            "Third sentence"
        ]
        
        embeddings = bridge.encode_sentence_transformer(
            texts,
            batch_size=2,
            show_progress_bar=False
        )
        
        print(f"Number of texts: {len(texts)}")
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Each embedding dimension: {embeddings.shape[1] if embeddings.ndim > 1 else embeddings.shape[0]}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_gpu_encoding():
    """GPU-accelerated encoding with Vulkan post-processing"""
    print("\nExample 3: GPU Encoding with Vulkan Post-processing")
    
    if not BRIDGE_AVAILABLE:
        print("Grilly not available")
        return
    
    try:
        bridge = get_huggingface_bridge()
        
        texts = [
            "This is a test",
            "Another test sentence",
            "Yet another sentence"
        ]
        
        # Encode with GPU acceleration (CUDA if available, CPU on AMD)
        # with Vulkan post-processing
        embeddings = bridge.encode_sentence_transformer_gpu(
            texts,
            use_vulkan_postprocessing=True
        )
        
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Using GPU: {'CUDA' if bridge.cuda_device else 'CPU (AMD/Vulkan)'}")
        print(f"Vulkan post-processing: Enabled")
        
        # Process with Vulkan operations
        linear = nn.Linear(embeddings.shape[-1], 128)
        processed = linear(embeddings)
        
        print(f"Processed shape: {processed.shape}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_custom_model():
    """Using a custom sentence-transformer model"""
    print("\nExample 4: Custom Model")
    
    if not BRIDGE_AVAILABLE:
        print("Grilly not available")
        return
    
    try:
        bridge = get_huggingface_bridge()
        
        # Use a different model
        text = "This is a test"
        embeddings = bridge.encode_sentence_transformer(
            text,
            model_name='all-MiniLM-L6-v2',  # or 'paraphrase-MiniLM-L6-v2', etc.
            normalize_embeddings=True
        )
        
        print(f"Model: all-MiniLM-L6-v2")
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Embedding norm: {np.linalg.norm(embeddings):.4f}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_vulkan_processing():
    """Process sentence-transformer embeddings with Vulkan"""
    print("\nExample 5: Vulkan Processing Pipeline")
    
    if not BRIDGE_AVAILABLE:
        print("Grilly not available")
        return
    
    try:
        bridge = get_huggingface_bridge()
        
        # Get embeddings
        texts = ["Hello, world!", "How are you?"]
        embeddings = bridge.encode_sentence_transformer(texts)
        
        print(f"Step 1 - Embeddings: {embeddings.shape}")
        
        # Convert to Vulkan GPU tensor (if available)
        try:
            gpu_embeddings = to_vulkan_gpu(embeddings)
            print(f"Step 2 - GPU tensor: {gpu_embeddings}")
        except Exception:
            # Fall back to regular numpy
            gpu_embeddings = embeddings
            print(f"Step 2 - Using numpy (GPU tensor not available)")
        
        # Process with Vulkan model
        model = nn.Sequential(
            nn.Linear(embeddings.shape[-1], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Handle GPU tensor or numpy
        if hasattr(gpu_embeddings, 'numpy'):
            processed = model(gpu_embeddings.numpy())
        else:
            processed = model(gpu_embeddings)
        
        print(f"Step 3 - Processed: {processed.shape}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_similarity_search():
    """Use sentence-transformers for similarity search"""
    print("\nExample 6: Similarity Search")
    
    if not BRIDGE_AVAILABLE:
        print("Grilly not available")
        return
    
    try:
        bridge = get_huggingface_bridge()
        
        # Encode query and documents
        query = "What is machine learning?"
        documents = [
            "Machine learning is a subset of artificial intelligence",
            "Python is a programming language",
            "Deep learning uses neural networks",
            "Natural language processing is important"
        ]
        
        # Encode all
        query_emb = bridge.encode_sentence_transformer(query)
        doc_embs = bridge.encode_sentence_transformer(documents)
        
        # Compute similarities (cosine similarity since embeddings are normalized)
        similarities = np.dot(doc_embs, query_emb)
        
        # Sort by similarity
        sorted_indices = np.argsort(similarities)[::-1]
        
        print(f"Query: {query}")
        print("\nMost similar documents:")
        for i, idx in enumerate(sorted_indices[:3]):
            print(f"{i+1}. {documents[idx]} (similarity: {similarities[idx]:.4f})")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Sentence-Transformers GPU Examples")
    print("=" * 60)
    
    example_basic_encoding()
    example_batch_encoding()
    example_gpu_encoding()
    example_custom_model()
    example_vulkan_processing()
    example_similarity_search()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
