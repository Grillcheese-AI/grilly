"""
Train Capsule Embeddings Example

Demonstrates training capsule embeddings using contrastive learning.
"""
import numpy as np
from grilly.nn import CapsuleEmbedding, ContrastiveLoss
from grilly.optim import Adam
from grilly.utils import save_checkpoint, load_checkpoint, print_model_summary


def generate_training_data(num_samples: int = 100, embedding_dim: int = 384):
    """
    Generate synthetic training data for capsule embedding training.
    
    In practice, you would use real embeddings from a sentence transformer.
    
    Args:
        num_samples: Number of training samples
        embedding_dim: Embedding dimension
    
    Returns:
        (anchors, positives, negatives) - all (num_samples, embedding_dim)
    """
    # Generate anchor embeddings
    anchors = np.random.randn(num_samples, embedding_dim).astype(np.float32)
    anchors = anchors / (np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-8)
    
    # Generate positive embeddings (similar to anchors with small noise)
    positives = anchors + np.random.randn(num_samples, embedding_dim).astype(np.float32) * 0.1
    positives = positives / (np.linalg.norm(positives, axis=1, keepdims=True) + 1e-8)
    
    # Generate negative embeddings (different from anchors)
    negatives = np.random.randn(num_samples, embedding_dim).astype(np.float32)
    negatives = negatives / (np.linalg.norm(negatives, axis=1, keepdims=True) + 1e-8)
    
    return anchors, positives, negatives


def train_capsule_embeddings(
    model: CapsuleEmbedding,
    anchors: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    checkpoint_path: Optional[str] = None
):
    """
    Train capsule embeddings using contrastive learning.
    
    Args:
        model: CapsuleEmbedding model
        anchors: Anchor embeddings (num_samples, embedding_dim)
        positives: Positive embeddings (num_samples, embedding_dim)
        negatives: Negative embeddings (num_samples, embedding_dim)
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        checkpoint_path: Optional path to save checkpoints
    """
    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Initialize loss function
    loss_fn = ContrastiveLoss(margin=0.5, distance_metric='cosine')
    
    num_samples = anchors.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"Training capsule embeddings...")
    print(f"  Samples: {num_samples}")
    print(f"  Batches per epoch: {num_batches}")
    print(f"  Epochs: {num_epochs}")
    print()
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        # Shuffle data
        indices = np.random.permutation(num_samples)
        anchors_shuffled = anchors[indices]
        positives_shuffled = positives[indices]
        negatives_shuffled = negatives[indices]
        
        for batch_idx in range(num_batches):
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
            # CapsuleEmbedding.backward() takes (grad_output, embeddings)
            try:
                # Backward through anchor
                model.backward(grad_anchor, batch_anchors)
                
                # Accumulate gradients from positive and negative
                # Note: We need to accumulate gradients, not replace them
                # Store current gradients
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
        print(f"Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.6f}")
        
        # Save checkpoint
        if checkpoint_path and (epoch + 1) % 5 == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=avg_loss,
                filepath=f"{checkpoint_path}_epoch_{epoch + 1}.npz"
            )
    
    print("\nTraining complete!")
    return model


def main():
    """Main training script"""
    # Create model
    model = CapsuleEmbedding(
        embedding_dim=384,
        capsule_dim=32,
        semantic_dims=28,
        use_dg=True  # Set to True to use DentateGyrus expansion
    )
    
    # Print model summary
    print_model_summary(model)
    print()
    
    # Generate training data
    print("Generating training data...")
    anchors, positives, negatives = generate_training_data(num_samples=2000, embedding_dim=384)
    print(f"Generated {len(anchors)} training samples")
    print()
    
    # Train model
    trained_model = train_capsule_embeddings(
        model=model,
        anchors=anchors,
        positives=positives,
        negatives=negatives,
        num_epochs=20,
        batch_size=64,
        learning_rate=1e-3,
        checkpoint_path="capsule_embedding"
    )
    
    # Test similarity
    print("\nTesting similarity...")
    test_anchor = anchors[0:1]
    test_positive = positives[0:1]
    test_negative = negatives[0:1]
    
    anchor_cap = trained_model(test_anchor)
    positive_cap = trained_model(test_positive)
    negative_cap = trained_model(test_negative)
    
    # Compute similarities
    pos_sim = np.sum(anchor_cap * positive_cap) / (
        np.linalg.norm(anchor_cap) * np.linalg.norm(positive_cap) + 1e-8
    )
    neg_sim = np.sum(anchor_cap * negative_cap) / (
        np.linalg.norm(anchor_cap) * np.linalg.norm(negative_cap) + 1e-8
    )
    
    print(f"Anchor-Positive similarity: {pos_sim:.4f}")
    print(f"Anchor-Negative similarity: {neg_sim:.4f}")
    print(f"Difference: {pos_sim - neg_sim:.4f} (should be positive)")


if __name__ == "__main__":
    main()
