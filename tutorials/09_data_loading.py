"""
Data Loading Tutorial
=====================

Working with datasets and data loaders in Grilly - similar to PyTorch's
torch.utils.data module.

This tutorial covers:
1. Loading real MNIST dataset
2. Dataset and DataLoader basics
3. Transforms and preprocessing
4. Training a neural network with data loaders
"""

import _path_setup  # noqa: F401 - must be first to ensure grilly is importable

import numpy as np
from pathlib import Path

from grilly import Compute
from grilly.datasets import MNIST, CIFAR10
from grilly.utils.data import (
    Dataset,
    TensorDataset,
    ArrayDataset,
    DataLoader,
    random_split,
    Compose,
    ToFloat32,
    Normalize,
    Flatten,
    OneHot,
)
from grilly.nn import (
    Module,
    Linear,
    ReLU,
    Sequential,
)
from grilly.nn.autograd import (
    Variable,
    tensor,
    cross_entropy,
    no_grad,
)
from grilly.optim import SGD

# Initialize GPU
compute = Compute()
print(f"GPU: {compute.core.device_properties.deviceName}")

print("\n" + "=" * 70)
print("Data Loading Tutorial")
print("=" * 70)

# Find datasets directory
datasets_root = Path(__file__).parent.parent / 'datasets'
print(f"Datasets directory: {datasets_root}")


# ============================================================================
# 1. Loading Real MNIST Dataset
# ============================================================================
print("\n1. Loading Real MNIST Dataset")
print("-" * 70)

# Define transforms for MNIST
# Images are uint8 [0-255], we normalize to float32 [-1, 1]
def normalize_mnist(x):
    """Normalize MNIST images to [-1, 1] and flatten."""
    x = x.astype(np.float32) / 255.0  # Scale to [0, 1]
    x = (x - 0.5) / 0.5  # Normalize to [-1, 1]
    x = x.reshape(-1)  # Flatten to 784
    return x

# Load MNIST dataset
training_data = MNIST(
    root=str(datasets_root),
    train=True,
    transform=normalize_mnist,
)

test_data = MNIST(
    root=str(datasets_root),
    train=False,
    transform=normalize_mnist,
)

print(f"Training: {training_data}")
print(f"Test: {test_data}")

# Check sample
x, y = training_data[0]
print(f"Sample - Shape: {x.shape}, Label: {y}, Min: {x.min():.2f}, Max: {x.max():.2f}")


# ============================================================================
# 2. Basic Dataset and DataLoader
# ============================================================================
print("\n2. Basic Dataset and DataLoader")
print("-" * 70)

# Create a TensorDataset example (for when you have numpy arrays)
np.random.seed(42)
X = np.random.randn(1000, 784).astype(np.float32)
y = np.random.randint(0, 10, 1000)

dataset = TensorDataset(X, y)
print(f"TensorDataset size: {len(dataset)} samples")

x_sample, y_sample = dataset[0]
print(f"Sample shape: {x_sample.shape}, Label: {y_sample}")

# Create a DataLoader
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"DataLoader: {len(dataloader)} batches of size {batch_size}")

# Iterate through one batch
for X_batch, y_batch in dataloader:
    print(f"Batch shapes: X={X_batch.shape}, y={y_batch.shape}")
    break


# ============================================================================
# 3. Creating Data Loaders for MNIST
# ============================================================================
print("\n3. Creating Data Loaders for MNIST")
print("-" * 70)

batch_size = 64

# Create data loaders for real MNIST
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

print(f"Training batches: {len(train_dataloader)}")
print(f"Test batches: {len(test_dataloader)}")

# Show batch info
for X, y in test_dataloader:
    print(f"Shape of X [N, features]: {X.shape}")
    print(f"Shape of y: {y.shape}, dtype: {y.dtype}")
    break


# ============================================================================
# 4. Loading CIFAR-10 Dataset
# ============================================================================
print("\n4. Loading CIFAR-10 Dataset")
print("-" * 70)

def normalize_cifar(x):
    """Normalize CIFAR images."""
    x = x.astype(np.float32) / 255.0
    # CIFAR mean/std per channel
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616]).reshape(3, 1, 1)
    x = (x - mean) / std
    return x

cifar_train = CIFAR10(root=str(datasets_root), train=True, transform=normalize_cifar)
cifar_test = CIFAR10(root=str(datasets_root), train=False, transform=normalize_cifar)

print(f"CIFAR-10 Training: {cifar_train}")
print(f"CIFAR-10 Test: {cifar_test}")
print(f"Classes: {CIFAR10.classes}")

x, y = cifar_train[0]
print(f"Sample - Shape: {x.shape}, Label: {y} ({CIFAR10.classes[y]})")


# ============================================================================
# 5. Splitting Datasets
# ============================================================================
print("\n5. Splitting Datasets")
print("-" * 70)

# Create a dataset and split it
full_dataset = TensorDataset(
    np.random.randn(1000, 784).astype(np.float32),
    np.random.randint(0, 10, 1000)
)

# Split into train/val/test (80/10/10)
train_set, val_set, test_set = random_split(full_dataset, [800, 100, 100])

print(f"Full dataset: {len(full_dataset)} samples")
print(f"Train set: {len(train_set)} samples")
print(f"Val set: {len(val_set)} samples")
print(f"Test set: {len(test_set)} samples")


# ============================================================================
# 6. Training a Neural Network on MNIST (with proper autograd)
# ============================================================================
print("\n6. Training a Neural Network on MNIST")
print("-" * 70)

from grilly.nn.autograd import Variable, matmul, relu, softmax, cross_entropy, no_grad


class SimpleNN:
    """
    Simple 3-layer neural network using autograd Variables.

    Architecture: 784 -> 128 -> 64 -> 10
    """

    def __init__(self):
        # Initialize weights with Xavier initialization
        scale1 = np.sqrt(2.0 / 784)
        scale2 = np.sqrt(2.0 / 128)
        scale3 = np.sqrt(2.0 / 64)

        self.w1 = Variable(np.random.randn(784, 128).astype(np.float32) * scale1, requires_grad=True)
        self.b1 = Variable(np.zeros(128, dtype=np.float32), requires_grad=True)

        self.w2 = Variable(np.random.randn(128, 64).astype(np.float32) * scale2, requires_grad=True)
        self.b2 = Variable(np.zeros(64, dtype=np.float32), requires_grad=True)

        self.w3 = Variable(np.random.randn(64, 10).astype(np.float32) * scale3, requires_grad=True)
        self.b3 = Variable(np.zeros(10, dtype=np.float32), requires_grad=True)

    def forward(self, x):
        """Forward pass using autograd operations."""
        # Layer 1: Linear + ReLU
        h1 = matmul(x, self.w1) + self.b1
        h1 = relu(h1)

        # Layer 2: Linear + ReLU
        h2 = matmul(h1, self.w2) + self.b2
        h2 = relu(h2)

        # Layer 3: Linear (logits)
        logits = matmul(h2, self.w3) + self.b3

        return logits

    def parameters(self):
        """Return list of all parameters."""
        return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

    def zero_grad(self):
        """Zero all gradients."""
        for p in self.parameters():
            p.zero_grad()


# Create model
print("Creating neural network: 784 -> 128 -> 64 -> 10")
model = SimpleNN()

# Training hyperparameters
learning_rate = 0.01
epochs = 3

# Use smaller subset for faster training demo
# Use first 10000 samples for training, 2000 for testing
small_train = training_data
small_test = test_data

# Create data loaders
batch_size = 64
small_train_loader = DataLoader(small_train, batch_size=batch_size, shuffle=True)
small_test_loader = DataLoader(small_test, batch_size=batch_size, shuffle=False)


def train_epoch(dataloader, model, lr):
    """Train for one epoch."""
    total_loss = 0
    n_batches = 0

    for batch_idx, (X, y) in enumerate(dataloader):
        # Flatten and convert to Variable
        X_flat = X.reshape(X.shape[0], -1).astype(np.float32)
        x_var = Variable(X_flat, requires_grad=False)

        # Forward pass
        logits = model.forward(x_var)

        # Compute cross-entropy loss
        y_var = Variable(y.astype(np.int64), requires_grad=False)
        loss = cross_entropy(logits, y_var)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Update weights with SGD
        with no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    param.data -= lr * param.grad

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 200 == 0:
            print(f"  [{batch_idx * batch_size:>5d}/{len(dataloader.dataset):>5d}] loss: {loss.item():.4f}")

    return total_loss / n_batches


def test_model(dataloader, model):
    """Evaluate model accuracy."""
    correct = 0
    total = 0
    total_loss = 0
    n_batches = 0

    with no_grad():
        for X, y in dataloader:
            X_flat = X.reshape(X.shape[0], -1).astype(np.float32)
            x_var = Variable(X_flat, requires_grad=False)

            logits = model.forward(x_var)

            # Compute loss
            y_var = Variable(y.astype(np.int64), requires_grad=False)
            loss = cross_entropy(logits, y_var)
            total_loss += loss.item()
            n_batches += 1

            # Compute accuracy
            predictions = np.argmax(logits.data, axis=1)
            correct += (predictions == y).sum()
            total += len(y)

    accuracy = 100 * correct / total
    avg_loss = total_loss / n_batches
    print(f"Test: Accuracy: {accuracy:.1f}%, Avg loss: {avg_loss:.4f}")
    return accuracy


# Training loop
print(f"\nTraining on {len(small_train)} samples, testing on {len(small_test)} samples")
print(f"Learning rate: {learning_rate}, Batch size: {batch_size}")
print("=" * 60)

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    print("-" * 40)

    avg_loss = train_epoch(small_train_loader, model, learning_rate)
    print(f"  Average training loss: {avg_loss:.4f}")

    accuracy = test_model(small_test_loader, model)

print("\n" + "=" * 60)
print("Training complete!")


# ============================================================================
# 7. Working with ArrayDataset
# ============================================================================
print("\n7. Working with ArrayDataset")
print("-" * 70)

# ArrayDataset is useful when you have numpy arrays and want transforms
data = np.random.randn(500, 28, 28).astype(np.float32)
labels = np.random.randint(0, 10, 500)

# Create dataset with transforms
dataset = ArrayDataset(
    data=data,
    labels=labels,
    transform=Compose([
        lambda x: x.flatten(),  # Flatten
        Normalize(mean=0.0, std=1.0),  # Normalize
    ]),
    target_transform=OneHot(num_classes=10),  # One-hot encode labels
)

# Create dataloader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Get a batch
X_batch, y_batch = next(iter(loader))
print(f"Input shape: {X_batch.shape}")
print(f"Target shape (one-hot): {y_batch.shape}")


# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("""
Key concepts:

1. Built-in Datasets:
   - MNIST: 60k train + 10k test, 28x28 grayscale digits
   - CIFAR10: 50k train + 10k test, 32x32 color images (10 classes)

2. Dataset: Abstract base class for datasets
   - __len__(): Returns number of samples
   - __getitem__(idx): Returns sample at index

3. TensorDataset: Wraps numpy arrays
   - TensorDataset(X, y) for paired data

4. ArrayDataset: Adds transform support
   - transform: Applied to data
   - target_transform: Applied to labels

5. DataLoader: Batches and shuffles data
   - batch_size: Number of samples per batch
   - shuffle: Randomize order each epoch
   - drop_last: Drop incomplete final batch

6. Transforms: Preprocessing functions
   - Compose: Chain multiple transforms
   - Normalize: Standardize data
   - Flatten: Reshape to 1D
   - OneHot: Convert labels to one-hot

7. random_split: Split dataset into subsets
   - random_split(dataset, [train_size, val_size, test_size])
""")

print("Tutorial completed successfully!")
