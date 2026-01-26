"""
Example: Using sentence-transformers on Vulkan GPU (AMD)

Demonstrates how to run sentence-transformers models entirely on AMD GPU
using Vulkan compute shaders.
"""
import numpy as np

try:
    from grilly.utils.huggingface_bridge import get_huggingface_bridge
    from grilly.utils.vulkan_sentence_transformer import VulkanSentenceTransformer
    BRIDGE_AVAILABLE = True
except ImportError as e:
    BRIDGE_AVAILABLE = False
    print(f"Grilly not available: {e}")


def example_vulkan_sentence_transformer():
    """Use Vulkan sentence-transformer directly"""
    print("Example 1: Direct Vulkan Sentence-Transformer")
    
    if not BRIDGE_AVAILABLE:
        print("Grilly not available")
        return
    
    try:
        # Create Vulkan model
        model = VulkanSentenceTransformer('all-MiniLM-L6-v2')
        
        # Encode text (runs on AMD GPU via Vulkan!)
        text = "This is a test sentence"
        embeddings = model.encode(text)
        
        print(f"Text: {text}")
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Embedding dtype: {embeddings.dtype}")
        print("Running on AMD GPU via Vulkan!")
        
    except Exception as e:
        print(f"Error: {e}")


def example_bridge_vulkan():
    """Use bridge with Vulkan model"""
    print("\nExample 2: Bridge with Vulkan Model")
    
    if not BRIDGE_AVAILABLE:
        print("Grilly not available")
        return
    
    try:
        bridge = get_huggingface_bridge()
        
        texts = ["First sentence", "Second sentence"]
        
        # Use Vulkan for full model inference
        embeddings = bridge.encode_sentence_transformer_vulkan(
            texts,
            model_name='all-MiniLM-L6-v2'
        )
        
        print(f"Texts: {texts}")
        print(f"Embeddings shape: {embeddings.shape}")
        print("Full model running on AMD GPU via Vulkan!")
        
    except Exception as e:
        print(f"Error: {e}")


def example_batch_processing():
    """Batch processing with Vulkan"""
    print("\nExample 3: Batch Processing")
    
    if not BRIDGE_AVAILABLE:
        print("Grilly not available")
        return
    
    try:
        model = VulkanSentenceTransformer('all-MiniLM-L6-v2')
        
        texts = [
            "Machine learning is a subset of AI",
            "Python is a programming language",
            "Deep learning uses neural networks"
        ]
        
        embeddings = model.encode(texts, batch_size=2)
        
        print(f"Number of texts: {len(texts)}")
        print(f"Embeddings shape: {embeddings.shape}")
        print("Batch processing on AMD GPU!")
        
    except Exception as e:
        print(f"Error: {e}")


def example_similarity_search():
    """Similarity search with Vulkan embeddings"""
    print("\nExample 4: Similarity Search")
    
    if not BRIDGE_AVAILABLE:
        print("Grilly not available")
        return
    
    try:
        model = VulkanSentenceTransformer('all-MiniLM-L6-v2')
        
        query = "What is machine learning?"
        documents = [
            "Machine learning is a subset of artificial intelligence",
            "Python is a programming language",
            "Deep learning uses neural networks"
        ]
        
        # Encode all (on GPU!)
        query_emb = model.encode(query)
        doc_embs = model.encode(documents)
        
        # Compute similarities (cosine similarity since embeddings are normalized)
        # Handle single query case
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        similarities = np.dot(doc_embs, query_emb.T).flatten()
        
        # Find most similar
        most_similar_idx = np.argmax(similarities)
        
        print(f"Query: {query}")
        print(f"Most similar: {documents[most_similar_idx]}")
        print(f"Similarity: {similarities[most_similar_idx]:.4f}")
        print("All computation on AMD GPU!")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Vulkan Sentence-Transformer Examples (AMD GPU)")
    print("=" * 60)
    
    example_vulkan_sentence_transformer()
    example_bridge_vulkan()
    example_batch_processing()
    example_similarity_search()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
