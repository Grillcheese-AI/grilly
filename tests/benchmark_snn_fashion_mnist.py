"""
Fashion-MNIST Benchmark for Convolutional SNNs

Compares IFNode CSNN vs LIFNode CSNN on real Fashion-MNIST data.
Tests both manual SGD and AutoHypergradientAdamW optimizer.

Usage:
    python tests/benchmark_snn_fashion_mnist.py
    uv run python tests/benchmark_snn_fashion_mnist.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grilly.functional.snn import reset_net
from grilly.nn.conv import Conv2d
from grilly.nn.module import Module
from grilly.nn.modules import Linear, Sequential
from grilly.nn.normalization import BatchNorm2d
from grilly.nn.pooling import MaxPool2d
from grilly.nn.snn_containers import Flatten, MultiStepContainer, SeqToANNContainer
from grilly.nn.snn_neurons import IFNode, LIFNode
from grilly.nn.snn_surrogate import ATan
from grilly.optim.hypergradient import AutoHypergradientAdamW
from grilly.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def load_fashion_mnist():
    """Download Fashion-MNIST via sklearn and return numpy arrays."""
    from sklearn.datasets import fetch_openml

    print("[DATA] Fetching Fashion-MNIST from OpenML (may download on first run)...")
    t0 = time.time()
    mnist = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="liac-arff")
    data = mnist.data.astype(np.float32) / 255.0
    labels = mnist.target.astype(np.int64)
    print(f"[DATA] Loaded {len(data)} samples in {time.time() - t0:.1f}s")

    data = data.reshape(-1, 1, 28, 28)
    x_train, y_train = data[:60000], labels[:60000]
    x_test, y_test = data[60000:], labels[60000:]
    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class CSNN_IF(Module):
    """Convolutional SNN with IFNode neurons."""

    def __init__(self, T=4, channels=64):
        super().__init__()
        self.T = T
        self.conv_fc = Sequential(
            SeqToANNContainer(
                Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
                BatchNorm2d(channels),
            ),
            IFNode(surrogate_function=ATan(), step_mode="m"),
            MultiStepContainer(MaxPool2d(2, 2)),
            SeqToANNContainer(
                Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                BatchNorm2d(channels),
            ),
            IFNode(surrogate_function=ATan(), step_mode="m"),
            MultiStepContainer(MaxPool2d(2, 2)),
            MultiStepContainer(Flatten(start_dim=1)),
            SeqToANNContainer(Linear(channels * 7 * 7, channels * 4 * 4, bias=False)),
            IFNode(surrogate_function=ATan(), step_mode="m"),
            SeqToANNContainer(Linear(channels * 4 * 4, 10, bias=False)),
            IFNode(surrogate_function=ATan(), step_mode="m"),
        )
        self._modules["conv_fc"] = self.conv_fc

    def forward(self, x):
        x_seq = np.stack([x] * self.T, axis=0)
        out_seq = self.conv_fc(x_seq)
        return out_seq.mean(axis=0)

    def backward(self, grad_output):
        """Backward through the SNN (sets .grad on parameters)."""
        T = self.T
        grad_seq = np.stack([grad_output / T] * T, axis=0)
        self.conv_fc.backward(grad_seq)

    def backward_and_update(self, grad_output, lr=0.01):
        """Backward + manual SGD update (legacy)."""
        self.backward(grad_output)
        for param in self.parameters():
            p = np.asarray(param)
            if hasattr(param, "grad") and param.grad is not None:
                p -= lr * param.grad
                param.grad = None


class CSNN_LIF(Module):
    """Convolutional SNN with LIFNode neurons (varying tau)."""

    def __init__(self, T=4, channels=64):
        super().__init__()
        self.T = T
        self.conv_fc = Sequential(
            SeqToANNContainer(
                Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
                BatchNorm2d(channels),
            ),
            LIFNode(tau=2.0, surrogate_function=ATan(), step_mode="m"),
            MultiStepContainer(MaxPool2d(2, 2)),
            SeqToANNContainer(
                Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                BatchNorm2d(channels),
            ),
            LIFNode(tau=4.0, surrogate_function=ATan(), step_mode="m"),
            MultiStepContainer(MaxPool2d(2, 2)),
            MultiStepContainer(Flatten(start_dim=1)),
            SeqToANNContainer(Linear(channels * 7 * 7, channels * 4 * 4, bias=False)),
            LIFNode(tau=2.0, surrogate_function=ATan(), step_mode="m"),
            SeqToANNContainer(Linear(channels * 4 * 4, 10, bias=False)),
            LIFNode(tau=2.0, surrogate_function=ATan(), step_mode="m"),
        )
        self._modules["conv_fc"] = self.conv_fc

    def forward(self, x):
        x_seq = np.stack([x] * self.T, axis=0)
        out_seq = self.conv_fc(x_seq)
        return out_seq.mean(axis=0)

    def backward(self, grad_output):
        """Backward through the SNN (sets .grad on parameters)."""
        T = self.T
        grad_seq = np.stack([grad_output / T] * T, axis=0)
        self.conv_fc.backward(grad_seq)

    def backward_and_update(self, grad_output, lr=0.01):
        """Backward + manual SGD update (legacy)."""
        self.backward(grad_output)
        for param in self.parameters():
            p = np.asarray(param)
            if hasattr(param, "grad") and param.grad is not None:
                p -= lr * param.grad
                param.grad = None


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def cross_entropy_loss(logits, labels):
    """Returns (loss, grad_logits)."""
    N = logits.shape[0]
    probs = softmax(logits)
    log_probs = np.log(probs + 1e-8)
    loss = -np.mean(log_probs[np.arange(N), labels])
    grad = probs.copy()
    grad[np.arange(N), labels] -= 1.0
    grad /= N
    return float(loss), grad


def train_epoch(model, dataloader, lr=0.01, optimizer=None, surprise_gain=0.0):
    """Train one epoch with surrogate gradient backprop.

    If optimizer is provided, uses it for parameter updates.
    Otherwise falls back to manual SGD via backward_and_update.

    If surprise_gain > 0 and the optimizer exposes current_surprise,
    the input is scaled by (1 + surprise_gain * surprise) before the
    forward pass. This implements input-level surprise modulation:
    high gradient prediction error → amplified input → more neuron firing.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    batch_count = 0

    for batch in dataloader:
        x_batch, y_batch = batch
        x_batch = x_batch.astype(np.float32)
        y_batch = y_batch.astype(np.int64)
        N = x_batch.shape[0]

        # Input-level surprise: amplify input when optimizer detects
        # gradient prediction error (landscape shift / phase transition).
        # Uses the inverted-U gain (Yerkes-Dodson) which self-limits
        # at high surprise to prevent trauma/fixation.
        if surprise_gain > 0 and optimizer is not None:
            s_gain = getattr(optimizer, "current_surprise_gain", 0.0)
            gain = 1.0 + surprise_gain * s_gain
            x_input = x_batch * gain
        else:
            x_input = x_batch

        reset_net(model)
        out = model(x_input)  # (N, 10) firing rates

        loss, grad = cross_entropy_loss(out, y_batch)
        total_loss += loss

        preds = np.argmax(out, axis=1)
        correct += np.sum(preds == y_batch)
        total += N
        batch_count += 1

        if optimizer is not None:
            # Optimizer-based update
            model.backward(grad)
            optimizer.step()
            optimizer.zero_grad()
        else:
            # Legacy SGD update
            model.backward_and_update(grad, lr=lr)

    avg_loss = total_loss / max(batch_count, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def evaluate(model, dataloader):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0

    for batch in dataloader:
        x_batch, y_batch = batch
        x_batch = x_batch.astype(np.float32)
        y_batch = y_batch.astype(np.int64)

        reset_net(model)
        out = model(x_batch)
        preds = np.argmax(out, axis=1)
        correct += np.sum(preds == y_batch)
        total += y_batch.shape[0]

    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def benchmark_model(name, model, train_loader, test_loader,
                    epochs=3, lr=0.01, optimizer=None, surprise_gain=0.0):
    """Run benchmark for a single model."""
    opt_name = repr(optimizer) if optimizer else f"SGD(lr={lr})"
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {name}")
    print(f"  T={model.T}, epochs={epochs}, optimizer={opt_name}")
    print(f"{'='*60}")

    param_count = sum(np.prod(p.shape) for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    # Warmup
    print("  Warmup...", end=" ", flush=True)
    x_warmup = np.random.randn(2, 1, 28, 28).astype(np.float32)
    reset_net(model)
    t0 = time.time()
    _ = model(x_warmup)
    print(f"{time.time() - t0:.3f}s")

    # Throughput benchmark
    print("  Benchmarking throughput...", end=" ", flush=True)
    batch_times = []
    for i, batch in enumerate(train_loader):
        if i >= 5:
            break
        x_batch = batch[0].astype(np.float32)
        reset_net(model)
        t0 = time.time()
        _ = model(x_batch)
        batch_times.append(time.time() - t0)

    avg_fwd = np.mean(batch_times)
    bs = train_loader.batch_size
    print(f"{avg_fwd:.3f}s/batch ({bs/avg_fwd:.1f} samples/s fwd)")

    # Training
    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, lr=lr, optimizer=optimizer,
            surprise_gain=surprise_gain,
        )
        epoch_time = time.time() - t0
        lr_info = ""
        if optimizer is not None and hasattr(optimizer, "current_lr"):
            lr_info = f", lr={optimizer.current_lr:.6f}"
        print(
            f"  Epoch {epoch+1}/{epochs}: "
            f"loss={train_loss:.4f}, train_acc={train_acc:.2%}, "
            f"time={epoch_time:.1f}s{lr_info}"
        )

    # Evaluation
    print("  Evaluating...", end=" ", flush=True)
    t0 = time.time()
    test_acc = evaluate(model, test_loader)
    print(f"test_acc={test_acc:.2%} ({time.time() - t0:.1f}s)")

    return {
        "name": name,
        "params": param_count,
        "avg_fwd_ms": avg_fwd * 1000,
        "fwd_samples_per_sec": bs / avg_fwd,
        "final_train_loss": train_loss,
        "final_train_acc": train_acc,
        "test_acc": test_acc,
    }


def main():
    print("=" * 60)
    print("  Grilly SNN Fashion-MNIST Benchmark")
    print("=" * 60)

    # Configuration
    T = 4
    CHANNELS = 32
    BATCH_SIZE = 64
    EPOCHS = 5
    LR = 0.005
    TRAIN_SUBSET = 10000
    TEST_SUBSET = 2000

    x_train, y_train, x_test, y_test = load_fashion_mnist()

    if TRAIN_SUBSET:
        x_train, y_train = x_train[:TRAIN_SUBSET], y_train[:TRAIN_SUBSET]
    if TEST_SUBSET:
        x_test, y_test = x_test[:TEST_SUBSET], y_test[:TEST_SUBSET]

    print(f"[DATA] Train: {x_train.shape}, Test: {x_test.shape}")
    print(f"[CONFIG] T={T}, ch={CHANNELS}, bs={BATCH_SIZE}, epochs={EPOCHS}, lr={LR}")

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    results = []

    # --- 1. CSNN-IF with manual SGD (baseline) ---
    np.random.seed(42)
    model_if = CSNN_IF(T=T, channels=CHANNELS)
    r = benchmark_model(
        "CSNN-IF", model_if, train_loader, test_loader,
        epochs=EPOCHS, lr=LR,
    )
    results.append(r)

    # --- 2. CSNN-LIF with manual SGD (baseline) ---
    np.random.seed(42)
    model_lif = CSNN_LIF(T=T, channels=CHANNELS)
    r = benchmark_model(
        "CSNN-LIF", model_lif, train_loader, test_loader,
        epochs=EPOCHS, lr=LR,
    )
    results.append(r)

    # --- 3. CSNN-LIF with AutoHypergradientAdamW ---
    np.random.seed(42)
    model_lif_auto = CSNN_LIF(T=T, channels=CHANNELS)
    optimizer_auto = AutoHypergradientAdamW(
        model_lif_auto.parameters(),
        lr=5e-4,
        weight_decay=0.005,
        hyper_lr=0.001,
        warmup_steps=50,
        lr_min=1e-5,
        lr_max=0.002,
        use_gpu=False,
    )
    r = benchmark_model(
        "CSNN-LIF+Auto", model_lif_auto, train_loader, test_loader,
        epochs=EPOCHS, optimizer=optimizer_auto,
    )
    results.append(r)

    # --- 4. CSNN-LIF with AutoHypergradientAdamW + Surprise input gain ---
    np.random.seed(42)
    model_lif_surprise = CSNN_LIF(T=T, channels=CHANNELS)
    optimizer_surprise = AutoHypergradientAdamW(
        model_lif_surprise.parameters(),
        lr=5e-4,
        weight_decay=0.005,
        hyper_lr=0.001,
        warmup_steps=50,
        lr_min=1e-5,
        lr_max=0.002,
        track_surprise=True,
        surprise_gamma=0.95,
        surprise_alpha=0.05,
        trauma_threshold=0.3,
        use_gpu=False,
    )
    SURPRISE_GAIN = 0.1  # moderate input-level surprise gain
    r = benchmark_model(
        "CSNN-LIF+Surprise", model_lif_surprise, train_loader, test_loader,
        epochs=EPOCHS, optimizer=optimizer_surprise,
        surprise_gain=SURPRISE_GAIN,
    )
    results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(
        f"{'Model':<20} {'Params':>10} {'Fwd ms/batch':>14} "
        f"{'Samples/s':>11} {'Train Acc':>10} {'Test Acc':>10}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['name']:<20} {r['params']:>10,} "
            f"{r['avg_fwd_ms']:>11.1f}ms "
            f"{r['fwd_samples_per_sec']:>11.1f} "
            f"{r['final_train_acc']:>9.2%} "
            f"{r['test_acc']:>9.2%}"
        )
    print("-" * 80)

    best_acc = max(results, key=lambda r: r["test_acc"])
    best_speed = max(results, key=lambda r: r["fwd_samples_per_sec"])
    print(f"\n  Best accuracy:   {best_acc['name']} ({best_acc['test_acc']:.2%})")
    print(f"  Best throughput: {best_speed['name']} ({best_speed['fwd_samples_per_sec']:.0f} samples/s)")

    # Print LR trajectories
    for name, opt in [("Auto", optimizer_auto), ("Surprise", optimizer_surprise)]:
        if hasattr(opt, "lr_history") and len(opt.lr_history) > 1:
            hist = opt.lr_history
            print(f"\n  {name} LR trajectory:")
            print(f"    Start: {hist[0]:.6f}")
            print(f"    End:   {hist[-1]:.6f}")
            print(f"    Min:   {min(hist):.6f}")
            print(f"    Max:   {max(hist):.6f}")
            print(f"    Steps: {len(hist)}")

    # Print surprise trajectory
    if hasattr(optimizer_surprise, "surprise_history") and optimizer_surprise.surprise_history:
        sh = optimizer_surprise.surprise_history
        sb = optimizer_surprise.s_bar_history
        print("\n  Surprise trajectory:")
        print(f"    Instant - Mean: {np.mean(sh):.6f}, Max: {max(sh):.6f}")
        print(f"    S_bar   - Mean: {np.mean(sb):.6f}, Max: {max(sb):.6f}")
        print(f"    Final gain: {optimizer_surprise.current_surprise_gain:.6f}")
        print(f"    Trauma threshold: {optimizer_surprise.trauma_threshold}")
        print(f"    Steps: {len(sh)}")


if __name__ == "__main__":
    main()
