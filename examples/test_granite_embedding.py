"""Test Granite Embedding model: ibm-granite/granite-embedding-english-r2

This model is based on ModernBERT architecture (not causal Granite).
Key differences:
- Uses GeGLU activation (not GELU)
- Uses CLS pooling (not mean pooling)
- 22 layers, 12 attention heads
- Embedding size: 768
- Max sequence length: 8192
"""

import numpy as np
from typing import List

print("="*80)
print("GRANITE EMBEDDING MODEL TEST")
print("="*80)
print("Model: ibm-granite/granite-embedding-english-r2")
print("Architecture: ModernBERT (not causal Granite)")
print("="*80)

try:
    from sentence_transformers import SentenceTransformer
    from grilly.utils.vulkan_sentence_transformer import VulkanSentenceTransformer
    
    model_name = "ibm-granite/granite-embedding-english-r2"
    
    print(f"\n[1] Loading original SentenceTransformer model...")
    try:
        st_model = SentenceTransformer(model_name)
        print(f"[OK] Original model loaded")
        print(f"     Model type: {type(st_model).__name__}")
        
        # Check the underlying model
        if hasattr(st_model, '_modules') and '0' in st_model._modules:
            underlying_model = st_model._modules['0']
            print(f"     Underlying model: {type(underlying_model).__name__}")
            if hasattr(underlying_model, 'auto_model'):
                auto_model = underlying_model.auto_model
                print(f"     Auto model: {type(auto_model).__name__}")
                if hasattr(auto_model, 'config'):
                    config = auto_model.config
                    print(f"     Config model_type: {getattr(config, 'model_type', 'N/A')}")
                    print(f"     Hidden size: {getattr(config, 'hidden_size', 'N/A')}")
                    print(f"     Num layers: {getattr(config, 'num_hidden_layers', 'N/A')}")
                    print(f"     Num attention heads: {getattr(config, 'num_attention_heads', 'N/A')}")
    except Exception as e:
        print(f"[ERROR] Failed to load original model: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    print(f"\n[2] Loading VulkanSentenceTransformer model...")
    try:
        vulkan_model = VulkanSentenceTransformer(model_name)
        print(f"[OK] Vulkan model loaded")
        print(f"     Detected model type: {vulkan_model.model_type}")
        print(f"     Backend architecture: {vulkan_model.backend.architecture}")
    except Exception as e:
        print(f"[ERROR] Failed to load Vulkan model: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Test texts from the HuggingFace example
    print(f"\n[3] Testing embeddings...")
    input_queries = [
        ' Who made the song My achy breaky heart? ',
        'summit define'
    ]
    
    input_passages = [
        "Achy Breaky Heart is a country song written by Don Von Tress. Originally titled Don't Tell My Heart and performed by The Marcy Brothers in 1991. ",
        "Definition of summit for English Language Learners. : 1 the highest point of a mountain : the top of a mountain. : 2 the highest level. : 3 a meeting or series of meetings between the leaders of two or more governments."
    ]
    
    print(f"     Queries: {len(input_queries)}")
    print(f"     Passages: {len(input_passages)}")
    
    # Get original embeddings
    print(f"\n[4] Computing original embeddings...")
    try:
        orig_query_embs = st_model.encode(input_queries, normalize_embeddings=True, show_progress_bar=False)
        orig_passage_embs = st_model.encode(input_passages, normalize_embeddings=True, show_progress_bar=False)
        print(f"[OK] Original embeddings computed")
        print(f"     Query embeddings shape: {orig_query_embs.shape}")
        print(f"     Passage embeddings shape: {orig_passage_embs.shape}")
    except Exception as e:
        print(f"[ERROR] Failed to compute original embeddings: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Get Vulkan embeddings
    print(f"\n[5] Computing Vulkan embeddings...")
    try:
        vulkan_query_embs = vulkan_model.encode(input_queries, normalize_embeddings=True)
        vulkan_passage_embs = vulkan_model.encode(input_passages, normalize_embeddings=True)
        print(f"[OK] Vulkan embeddings computed")
        print(f"     Query embeddings shape: {vulkan_query_embs.shape}")
        print(f"     Passage embeddings shape: {vulkan_passage_embs.shape}")
    except Exception as e:
        print(f"[ERROR] Failed to compute Vulkan embeddings: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Compare embeddings
    print(f"\n[6] Comparing embeddings...")
    query_diff = np.linalg.norm(orig_query_embs - vulkan_query_embs)
    passage_diff = np.linalg.norm(orig_passage_embs - vulkan_passage_embs)
    
    print(f"     Query embeddings difference: {query_diff:.6f}")
    print(f"     Passage embeddings difference: {passage_diff:.6f}")
    
    # Compute cosine similarities
    print(f"\n[7] Computing cosine similarities...")
    from sentence_transformers import util
    
    # Original similarities
    orig_sim = util.cos_sim(orig_query_embs, orig_passage_embs).numpy()
    
    # Vulkan similarities
    vulkan_sim = np.dot(vulkan_query_embs, vulkan_passage_embs.T)
    
    print(f"\nOriginal similarities:")
    for i, query in enumerate(input_queries):
        for j, passage in enumerate(input_passages):
            print(f"  Query {i+1} <-> Passage {j+1}: {orig_sim[i, j]:.4f}")
    
    print(f"\nVulkan similarities:")
    for i, query in enumerate(input_queries):
        for j, passage in enumerate(input_passages):
            print(f"  Query {i+1} <-> Passage {j+1}: {vulkan_sim[i, j]:.4f}")
    
    print(f"\nSimilarity differences:")
    for i in range(len(input_queries)):
        for j in range(len(input_passages)):
            diff = abs(orig_sim[i, j] - vulkan_sim[i, j])
            print(f"  Query {i+1} <-> Passage {j+1}: {diff:.6f}")
    
    # Summary
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    max_sim_diff = np.max(np.abs(orig_sim - vulkan_sim))
    avg_sim_diff = np.mean(np.abs(orig_sim - vulkan_sim))
    
    print(f"[OK] Model loaded successfully")
    print(f"[OK] Embeddings computed")
    print(f"[OK] Query embeddings difference: {query_diff:.6f}")
    print(f"[OK] Passage embeddings difference: {passage_diff:.6f}")
    print(f"[OK] Max similarity difference: {max_sim_diff:.6f}")
    print(f"[OK] Avg similarity difference: {avg_sim_diff:.6f}")
    
    if max_sim_diff < 0.01:
        print(f"[OK] Similarities match closely!")
    elif max_sim_diff < 0.1:
        print(f"[WARNING] Similarities have some differences (may need architecture-specific handling)")
    else:
        print(f"[WARNING] Large differences detected - may need ModernBERT-specific implementation")
    
except ImportError as e:
    print(f"[ERROR] Missing dependencies: {e}")
    print("Install with: pip install sentence-transformers")
except Exception as e:
    print(f"[ERROR] Test failed: {e}")
    import traceback
    traceback.print_exc()
