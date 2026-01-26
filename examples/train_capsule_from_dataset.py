"""
Train Capsule Embeddings from Conversation Dataset

Uses the conversations_dataset_anonymized.jsonl file to train capsule embeddings
using contrastive learning on real conversation data.
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from grilly.nn import CapsuleEmbedding, ContrastiveLoss
from grilly.optim import Adam
from grilly.utils import save_checkpoint, print_model_summary

# Try to import HuggingFace bridge for GPU-accelerated embeddings
try:
    from grilly.utils.huggingface_bridge import get_huggingface_bridge
    HUGGINGFACE_BRIDGE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_BRIDGE_AVAILABLE = False
    get_huggingface_bridge = None

# Fallback to direct sentence-transformers if bridge not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


def load_conversation_dataset(filepath: str, max_samples: Optional[int] = None) -> List[dict]:
    """
    Load conversation dataset from JSONL file.
    
    Args:
        filepath: Path to JSONL file
        max_samples: Maximum number of samples to load (None for all)
    
    Returns:
        List of conversation entries
    """
    entries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                entry = json.loads(line.strip())
                entries.append(entry)
            except json.JSONDecodeError:
                continue
    return entries


def extract_text_pairs(entries: List[dict]) -> Tuple[List[str], List[str], List[str]]:
    """
    Extract anchor, positive, and negative text pairs from conversation dataset.
    
    For each conversation:
    - Anchor: User message
    - Positive: Assistant response (should be similar context)
    - Negative: Random message from different conversation (should be dissimilar)
    
    Args:
        entries: List of conversation entries
    
    Returns:
        (anchors, positives, negatives) - lists of text strings
    """
    anchors = []
    positives = []
    negatives = []
    
    # Extract all messages for negative sampling
    all_messages = []
    for entry in entries:
        # Handle messages field - could be list of strings or list of dicts
        if 'messages' in entry:
            messages = entry['messages']
        elif 'conversation' in entry:
            messages = entry['conversation']
        elif 'text' in entry:
            messages = [entry['text']]
        else:
            # Try to find any list
            for key, value in entry.items():
                if isinstance(value, list) and len(value) > 0:
                    messages = value
                    break
            else:
                continue
        
        # Convert messages to strings if they're dicts
        message_strings = []
        for msg in messages:
            if isinstance(msg, str):
                message_strings.append(msg)
            elif isinstance(msg, dict):
                # Try common keys like 'content', 'text', 'message'
                found = False
                for key in ['content', 'text', 'message', 'body']:
                    if key in msg and isinstance(msg[key], str) and len(msg[key]) > 0:
                        message_strings.append(msg[key])
                        found = True
                        break
                if not found:
                    # Use first string value found
                    for v in msg.values():
                        if isinstance(v, str) and len(v) > 0:
                            message_strings.append(v)
                            break
        
        # Extract user-assistant pairs (alternating messages)
        for i in range(len(message_strings) - 1):
            user_msg = message_strings[i]
            assistant_msg = message_strings[i + 1]
            
            # Skip if messages are too short
            if len(user_msg) < 10 or len(assistant_msg) < 10:
                continue
            
            anchors.append(user_msg)
            positives.append(assistant_msg)
            all_messages.append(user_msg)
            all_messages.append(assistant_msg)
    
    # Generate negatives (random messages from different conversations)
    np.random.seed(42)
    for i in range(len(anchors)):
        # Pick a random message that's not the current anchor or positive
        neg_candidates = [msg for msg in all_messages 
                         if msg != anchors[i] and msg != positives[i]]
        if neg_candidates:
            negatives.append(np.random.choice(neg_candidates))
        else:
            # Fallback: use a random anchor
            if len(anchors) > 1:
                idx = np.random.randint(0, len(anchors))
                if idx != i:
                    negatives.append(anchors[idx])
                else:
                    negatives.append(anchors[(idx + 1) % len(anchors)])
            else:
                negatives.append("This is a negative example.")
    
    return anchors, positives, negatives


def generate_embeddings(
    texts: List[str],
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 32,
    use_vulkan: bool = True
) -> np.ndarray:
    """
    Generate embeddings from texts using HuggingFace bridge (Vulkan GPU) or sentence-transformers.
    
    Args:
        texts: List of text strings
        model_name: Sentence transformer model name
        batch_size: Batch size for encoding
        use_vulkan: Use Vulkan GPU acceleration (default: True, runs on AMD GPU)
    
    Returns:
        Embeddings array (num_texts, embedding_dim)
    """
    # Try HuggingFace bridge first (Vulkan GPU on AMD)
    if HUGGINGFACE_BRIDGE_AVAILABLE and use_vulkan:
        try:
            print(f"Using HuggingFace bridge with Vulkan GPU acceleration...")
            bridge = get_huggingface_bridge()
            
            # Use Vulkan sentence-transformer (runs entire model on AMD GPU!)
            embeddings = bridge.encode_sentence_transformer_vulkan(
                texts,
                model_name=model_name,
                batch_size=batch_size
            )
            print(f"[OK] Generated {len(texts)} embeddings on Vulkan GPU")
            return embeddings.astype(np.float32)
        except Exception as e:
            print(f"Warning: Vulkan GPU encoding failed: {e}")
            print("Falling back to CPU/CUDA encoding...")
            # Fall through to regular encoding
    
    # Fallback to regular HuggingFace bridge (CUDA or CPU)
    if HUGGINGFACE_BRIDGE_AVAILABLE:
        try:
            print(f"Using HuggingFace bridge (CUDA/CPU)...")
            bridge = get_huggingface_bridge()
            embeddings = bridge.encode_sentence_transformer(
                texts,
                model_name=model_name,
                batch_size=batch_size,
                convert_to_numpy=True
            )
            print(f"[OK] Generated {len(texts)} embeddings")
            return embeddings.astype(np.float32)
        except Exception as e:
            print(f"Warning: HuggingFace bridge encoding failed: {e}")
            print("Falling back to direct sentence-transformers...")
    
    # Fallback to direct sentence-transformers
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            print(f"Using direct sentence-transformers (CPU)...")
            model = SentenceTransformer(model_name)
            embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
            print(f"[OK] Generated {len(texts)} embeddings")
            return embeddings.astype(np.float32)
        except Exception as e:
            print(f"Error: {e}")
            print("Falling back to random embeddings")
    
    # Final fallback: random embeddings
    print("Warning: Using random embeddings (no embedding models available)")
    return np.random.randn(len(texts), 384).astype(np.float32)


def train_capsule_embeddings_from_dataset(
    dataset_path: str,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    max_samples: Optional[int] = None,
    embedding_model: str = 'all-MiniLM-L6-v2',
    checkpoint_path: Optional[str] = None
):
    """
    Train capsule embeddings from conversation dataset.
    
    Args:
        dataset_path: Path to JSONL dataset file
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        max_samples: Maximum number of samples to use (None for all)
        embedding_model: Sentence transformer model name
        checkpoint_path: Optional path to save checkpoints
    """
    print("=" * 80)
    print("Training Capsule Embeddings from Conversation Dataset")
    print("=" * 80)
    print()
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    entries = load_conversation_dataset(dataset_path, max_samples=max_samples)
    print(f"Loaded {len(entries)} conversation entries")
    print()
    
    # Extract text pairs
    print("Extracting text pairs...")
    anchors, positives, negatives = extract_text_pairs(entries)
    print(f"Extracted {len(anchors)} anchor-positive-negative triplets")
    print()
    
    if len(anchors) == 0:
        print("Error: No valid text pairs found in dataset")
        return None
    
    # Generate embeddings
    print("Generating embeddings...")
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("  Warning: sentence-transformers not available, using random embeddings")
        print("  Install with: pip install sentence-transformers")
    print("  This may take a while depending on dataset size...")
    print("  Generating anchor embeddings ({})...".format(len(anchors)))
    anchor_embeddings = generate_embeddings(anchors, embedding_model, batch_size)
    print("  ✓ Anchor embeddings generated")
    print("  Generating positive embeddings ({})...".format(len(positives)))
    positive_embeddings = generate_embeddings(positives, embedding_model, batch_size)
    print("  ✓ Positive embeddings generated")
    print("  Generating negative embeddings ({})...".format(len(negatives)))
    negative_embeddings = generate_embeddings(negatives, embedding_model, batch_size)
    print("  ✓ Negative embeddings generated")
    
    print(f"Anchor embeddings shape: {anchor_embeddings.shape}")
    print(f"Positive embeddings shape: {positive_embeddings.shape}")
    print(f"Negative embeddings shape: {negative_embeddings.shape}")
    print()
    
    # Create model
    embedding_dim = anchor_embeddings.shape[1]
    print(f"Creating capsule embedding model (embedding_dim={embedding_dim})...")
    model = CapsuleEmbedding(
        embedding_dim=embedding_dim,
        capsule_dim=32,
        semantic_dims=28,
        use_dg=False  # Set to True to use DentateGyrus expansion
    )
    print_model_summary(model)
    print()
    
    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = ContrastiveLoss(margin=0.5, distance_metric='cosine')
    
    # Training loop
    num_samples = len(anchors)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print("Starting training...")
    print(f"  Samples: {num_samples}")
    print(f"  Batches per epoch: {num_batches}")
    print(f"  Epochs: {num_epochs}")
    print()
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        # Shuffle data
        indices = np.random.permutation(num_samples)
        anchors_shuffled = anchor_embeddings[indices]
        positives_shuffled = positive_embeddings[indices]
        negatives_shuffled = negative_embeddings[indices]
        
        for batch_idx in range(num_batches):
            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{num_batches}", end='\r')
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            
            batch_anchors = anchors_shuffled[start_idx:end_idx]
            batch_positives = positives_shuffled[start_idx:end_idx]
            batch_negatives = negatives_shuffled[start_idx:end_idx]
            
            # Forward pass
            anchor_capsules = model(batch_anchors)
            positive_capsules = model(batch_positives)
            negative_capsules = model(batch_negatives)
            
            # Compute loss
            loss = loss_fn(anchor_capsules, positive_capsules, negative_capsules)
            epoch_losses.append(float(loss))
            
            # Backward pass
            model.zero_grad()
            grad_anchor, grad_positive, grad_negative = loss_fn.backward(
                1.0, anchor_capsules, positive_capsules, negative_capsules
            )
            
            # Backpropagate through model
            try:
                # Backward through anchor
                model.backward(grad_anchor, batch_anchors)
                
                # Accumulate gradients from positive and negative
                current_grads = {}
                for name, param in model.named_parameters():
                    if hasattr(param, 'grad') and param.grad is not None:
                        current_grads[name] = param.grad.copy()
                
                # Backward through positive
                model.backward(grad_positive, batch_positives)
                for name, param in model.named_parameters():
                    if hasattr(param, 'grad') and param.grad is not None:
                        if name in current_grads:
                            param.grad += current_grads[name]
                        current_grads[name] = param.grad.copy()
                
                # Backward through negative
                model.backward(grad_negative, batch_negatives)
                for name, param in model.named_parameters():
                    if hasattr(param, 'grad') and param.grad is not None:
                        if name in current_grads:
                            param.grad += current_grads[name]
            except Exception as e:
                print(f"Warning: backward pass failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Optimizer step
            optimizer.step()
        
        avg_loss = np.mean(epoch_losses)
        print(f"\nEpoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.6f}")
        
        # Save checkpoint
        if checkpoint_path and (epoch + 1) % 5 == 0:
            from grilly.utils import save_checkpoint
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=avg_loss,
                filepath=f"{checkpoint_path}_epoch_{epoch + 1}.npz"
            )
    
    print("\nTraining complete!")
    
    # Test similarity
    print("\nTesting similarity on sample pairs...")
    test_indices = np.random.choice(len(anchors), min(10, len(anchors)), replace=False)
    
    for idx in test_indices[:5]:
        anchor_emb = anchor_embeddings[idx:idx+1]
        positive_emb = positive_embeddings[idx:idx+1]
        negative_emb = negative_embeddings[idx:idx+1]
        
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
        
        print(f"  Pair {idx+1}:")
        print(f"    Anchor-Positive similarity: {pos_sim:.4f}")
        print(f"    Anchor-Negative similarity: {neg_sim:.4f}")
        print(f"    Difference: {pos_sim - neg_sim:.4f} (should be positive)")
        print(f"    Anchor text: {anchors[idx][:60]}...")
        print()
    
    return model


def main():
    """Main training script"""
    dataset_path = Path(__file__).parent / "conversations_dataset_anonymized.jsonl"
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return
    
    # Train model
    model = train_capsule_embeddings_from_dataset(
        dataset_path=str(dataset_path),
        num_epochs=20,
        batch_size=16,
        learning_rate=1e-3,
        max_samples=100,  # Limit for testing, set to None for full dataset
        embedding_model='all-MiniLM-L6-v2',
        checkpoint_path="capsule_embedding_conversations"
    )
    
    if model:
        print("\nModel training completed successfully!")
        print("You can now use the trained model for capsule embeddings.")


if __name__ == "__main__":
    main()
