"""
Visualization Utilities

Helper functions for visualizing training metrics, model architecture, etc.
"""

import numpy as np

# Try to import matplotlib (optional dependency)
try:
    import matplotlib
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    matplotlib = None


def plot_training_history(
    losses: list[float],
    accuracies: list[float] | None = None,
    val_losses: list[float] | None = None,
    val_accuracies: list[float] | None = None,
    save_path: str | None = None,
    show: bool = True,
):
    """
    Plot training history (loss and accuracy curves).

    Args:
        losses: List of training losses
        accuracies: Optional list of training accuracies
        val_losses: Optional list of validation losses
        val_accuracies: Optional list of validation accuracies
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Install with: pip install matplotlib")
        return

    fig, axes = plt.subplots(1, 2 if accuracies is not None else 1, figsize=(12, 4))
    if accuracies is None:
        axes = [axes]

    # Plot loss
    ax = axes[0]
    ax.plot(losses, label="Train Loss", color="blue")
    if val_losses:
        ax.plot(val_losses, label="Val Loss", color="red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True)

    # Plot accuracy if provided
    if accuracies is not None:
        ax = axes[1]
        ax.plot(accuracies, label="Train Acc", color="blue")
        if val_accuracies:
            ax.plot(val_accuracies, label="Val Acc", color="red")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Training Accuracy")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_gradient_flow(model, save_path: str | None = None, show: bool = True):
    """
    Plot gradient flow through the model (useful for debugging).

    Args:
        model: Module object with parameters
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Install with: pip install matplotlib")
        return

    # Collect gradient norms
    grad_norms = []
    param_names = []

    def collect_grads(module, prefix=""):
        """Run collect grads."""

        for name, param in getattr(module, "_parameters", {}).items():
            if param is not None and hasattr(param, "grad") and param.grad is not None:
                grad = param.grad
                grad_data = grad.data if hasattr(grad, "data") else grad
                grad_norm = np.linalg.norm(grad_data)
                grad_norms.append(grad_norm)
                param_names.append(f"{prefix}.{name}" if prefix else name)

        for name, submodule in getattr(module, "_modules", {}).items():
            collect_grads(submodule, f"{prefix}.{name}" if prefix else name)

    collect_grads(model)

    if not grad_norms:
        print("No gradients found in model")
        return

    # Plot
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(grad_norms)), grad_norms)
    plt.yticks(range(len(grad_norms)), param_names)
    plt.xlabel("Gradient Norm")
    plt.title("Gradient Flow Through Model")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_parameter_distribution(model, save_path: str | None = None, show: bool = True):
    """
    Plot distribution of model parameters.

    Args:
        model: Module object with parameters
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Install with: pip install matplotlib")
        return

    # Collect all parameters
    all_params = []

    def collect_params(module):
        """Run collect params."""

        for param in getattr(module, "_parameters", {}).values():
            if param is not None:
                data = param.data if hasattr(param, "data") else param
                all_params.extend(data.flatten())

        for submodule in getattr(module, "_modules", {}).values():
            if submodule is not None:
                collect_params(submodule)

    collect_params(model)

    if not all_params:
        print("No parameters found in model")
        return

    all_params = np.array(all_params)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(all_params, bins=50, edgecolor="black", alpha=0.7)
    plt.xlabel("Parameter Value")
    plt.ylabel("Frequency")
    plt.title("Parameter Distribution")
    plt.grid(True, alpha=0.3)

    # Add statistics
    mean = np.mean(all_params)
    std = np.std(all_params)
    plt.axvline(mean, color="red", linestyle="--", label=f"Mean: {mean:.4f}")
    plt.axvline(mean + std, color="orange", linestyle="--", label=f"±1 Std: {std:.4f}")
    plt.axvline(mean - std, color="orange", linestyle="--")
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def print_model_summary(model, input_shape: tuple[int, ...] | None = None):
    """
    Print a summary of the model architecture.

    Args:
        model: Module object
        input_shape: Optional input shape for computing output shapes
    """
    print("=" * 80)
    print("Model Summary")
    print("=" * 80)

    total_params = 0
    trainable_params = 0

    def count_params(module, prefix="", depth=0):
        """Run count params."""

        nonlocal total_params, trainable_params

        indent = "  " * depth
        module_name = module.__class__.__name__
        print(f"{indent}{prefix}{module_name}")

        # Count parameters
        for name, param in getattr(module, "_parameters", {}).items():
            if param is not None:
                data = param.data if hasattr(param, "data") else param
                param_count = np.prod(data.shape)
                total_params += param_count
                if hasattr(param, "requires_grad") and param.requires_grad:
                    trainable_params += param_count
                print(f"{indent}  {name}: {data.shape} ({param_count:,} params)")

        # Recurse into submodules
        for name, submodule in getattr(module, "_modules", {}).items():
            if submodule is not None:
                count_params(submodule, f"{name}.", depth + 1)

    count_params(model)

    print("=" * 80)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("=" * 80)


def visualize_attention_weights(
    attention_weights: np.ndarray,
    tokens: list[str] | None = None,
    save_path: str | None = None,
    show: bool = True,
):
    """
    Visualize attention weights as a heatmap.

    Args:
        attention_weights: Attention weights array (seq_len, seq_len) or (num_heads, seq_len, seq_len)
        tokens: Optional list of token strings for labels
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Install with: pip install matplotlib")
        return

    # Handle multi-head attention
    if attention_weights.ndim == 3:
        num_heads, seq_len, _ = attention_weights.shape
        # Average across heads
        attention_weights = attention_weights.mean(axis=0)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights, cmap="Blues", aspect="auto")
    plt.colorbar(label="Attention Weight")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.title("Attention Weights Heatmap")

    # Add token labels if provided
    if tokens and len(tokens) == attention_weights.shape[0]:
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
        plt.yticks(range(len(tokens)), tokens)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
