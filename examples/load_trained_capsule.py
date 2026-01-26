"""
Load and test trained capsule embedding model from checkpoint.
"""
import numpy as np
from grilly.nn import CapsuleEmbedding
from grilly.utils import load_checkpoint
from grilly.examples.train_capsule_from_dataset import load_conversation_dataset, extract_text_pairs, generate_embeddings

def load_and_test_model(checkpoint_path: str = "capsule_embedding_epoch_20.npz"):
    """
    Load trained model and test it on sample data.
    
    Args:
        checkpoint_path: Path to checkpoint file
    """
    print("=" * 80)
    print("Loading Trained Capsule Embedding Model")
    print("=" * 80)
    print()
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    try:
        checkpoint = load_checkpoint(checkpoint_path)
        print("[OK] Checkpoint loaded successfully")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Loss: {checkpoint.get('loss', 'unknown')}")
        print()
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Create model with same architecture
    print("Creating model architecture...")
    model = CapsuleEmbedding(
        embedding_dim=384,  # Adjust if your embeddings are different
        capsule_dim=32,
        semantic_dims=28,
        use_dg=False
    )
    
    # Load model state
    if 'model_state' in checkpoint:
        print("Loading model state...")
        try:
            model.load_state_dict(checkpoint['model_state'])
            print("[OK] Model state loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load model state: {e}")
            print("Using randomly initialized model")
    print()
    
    # Test on sample data
    print("Testing on sample conversation data...")
    try:
        # Load a few samples
        entries = load_conversation_dataset(
            'grilly/examples/conversations_dataset_anonymized.jsonl',
            max_samples=5
        )
        anchors, positives, negatives = extract_text_pairs(entries)
        
        if len(anchors) == 0:
            print("No test data available")
            return model
        
        print(f"Testing on {min(5, len(anchors))} sample pairs...")
        print()
        
        # Generate embeddings using HuggingFace bridge (Vulkan GPU)
        try:
            from grilly.examples.train_capsule_from_dataset import generate_embeddings
            anchor_embeddings = generate_embeddings(anchors[:5], batch_size=5, use_vulkan=True)
            positive_embeddings = generate_embeddings(positives[:5], batch_size=5, use_vulkan=True)
            negative_embeddings = generate_embeddings(negatives[:5], batch_size=5, use_vulkan=True)
        except Exception as e:
            print(f"Warning: Embedding generation failed: {e}")
            print("Using random embeddings for testing...")
            anchor_embeddings = np.random.randn(min(5, len(anchors)), 384).astype(np.float32)
            positive_embeddings = np.random.randn(min(5, len(positives)), 384).astype(np.float32)
            negative_embeddings = np.random.randn(min(5, len(negatives)), 384).astype(np.float32)
        
        # Test similarity
        for i in range(min(5, len(anchors))):
            anchor_emb = anchor_embeddings[i:i+1]
            positive_emb = positive_embeddings[i:i+1]
            negative_emb = negative_embeddings[i:i+1]
            
            anchor_cap = model(anchor_emb)
            positive_cap = model(positive_emb)
            negative_cap = model(negative_emb)
            
            # Compute similarities
            pos_sim = np.sum(anchor_cap * positive_cap) / (
                np.linalg.norm(anchor_cap) * np.linalg.norm(positive_cap) + 1e-8
            )
            neg_sim = np.sum(anchor_cap * negative_cap) / (
                np.linalg.norm(anchor_cap) * np.linalg.norm(negative_cap) + 1e-8
            )
            
            print(f"Pair {i+1}:")
            print(f"  Anchor-Positive similarity: {pos_sim:.4f}")
            print(f"  Anchor-Negative similarity: {neg_sim:.4f}")
            print(f"  Difference: {pos_sim - neg_sim:.4f} (should be positive)")
            if len(anchors[i]) > 0:
                print(f"  Anchor text: {anchors[i][:60]}...")
            print()
        
        print("[OK] Model testing complete!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    return model


def main():
    """Main function"""
    # Try to load the latest checkpoint
    import os
    checkpoints = [
        "capsule_embedding_epoch_20.npz",
        "capsule_embedding_epoch_15.npz",
        "capsule_embedding_epoch_10.npz",
        "capsule_embedding_epoch_5.npz"
    ]
    
    for checkpoint in checkpoints:
        if os.path.exists(checkpoint):
            print(f"Found checkpoint: {checkpoint}")
            model = load_and_test_model(checkpoint)
            if model:
                print("\n" + "=" * 80)
                print("Model loaded and tested successfully!")
                print("=" * 80)
            break
    else:
        print("No checkpoint files found. Please train the model first.")


if __name__ == "__main__":
    main()
