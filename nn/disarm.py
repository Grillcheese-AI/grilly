"""DisARM: Discrete Antithetic Gradient Estimator.

Low-variance gradient estimation through discrete operations via
antithetic sampling. Enables training through block-code discretization
without continuous relaxations (Gumbel-softmax).

From the HMM-VSA paper: DisARM provides gradient flow through
the block-code sign/argmax operations that are non-differentiable.
It uses paired antithetic samples (u, 1-u) to cancel out
common noise, achieving ~7.5x lower variance than REINFORCE.

Reference: Dong et al. (2020) "DisARM: An Antithetic Gradient
Estimator for Binary Latent Variables"
"""

import numpy as np

from .module import Module


def disarm_gradient(
    logits: np.ndarray,
    loss_fn,
    num_samples: int = 1,
    temperature: float = 1.0,
) -> np.ndarray:
    """Estimate gradients through discrete sampling using DisARM.

    Uses antithetic coupling: for each uniform sample u,
    also evaluates at 1-u. The gradient estimate is:

        grad ≈ (f(b(u)) - f(b(1-u))) * (u - 0.5) / (temperature * sigmoid_derivative)

    where b(u) = (logits + log(u/(1-u)) > 0) is the discrete sample.

    Args:
        logits: Pre-sigmoid logits of shape (batch, ..., dim)
        loss_fn: callable that takes discrete samples and returns scalar loss
        num_samples: Number of antithetic pairs (default: 1)
        temperature: Sampling temperature (default: 1.0)

    Returns:
        grad_logits: Gradient estimate of same shape as logits
    """
    grad_accum = np.zeros_like(logits)

    for _ in range(num_samples):
        # Sample uniform
        u = np.random.uniform(0.01, 0.99, size=logits.shape).astype(np.float32)

        # Gumbel-like noise: log(u / (1-u))
        noise = np.log(u / (1.0 - u))

        # Discrete samples via antithetic coupling
        z_pos = (logits / temperature + noise > 0).astype(np.float32)
        z_neg = (logits / temperature - noise > 0).astype(np.float32)  # antithetic: 1-u → -noise

        # Evaluate loss for both samples
        loss_pos = loss_fn(z_pos)
        loss_neg = loss_fn(z_neg)

        # DisARM gradient: difference times the coupling term
        sigmoid = 1.0 / (1.0 + np.exp(-logits / temperature))
        coupling = u - 0.5

        # Gradient estimate
        grad_accum += (loss_pos - loss_neg) * coupling / (sigmoid * (1 - sigmoid) + 1e-8)

    return grad_accum / (num_samples * temperature)


class DisARMSampler(Module):
    """Module wrapper for DisARM gradient estimation through discrete ops.

    Wraps a block-code discretization step so gradients can flow through it.

    Usage:
        sampler = DisARMSampler(dim=1000, block_size=50)
        logits = linear(x)                    # continuous
        discrete, grad_est = sampler(logits)  # discrete block codes + gradient
    """

    def __init__(self, dim: int, block_size: int, temperature: float = 1.0, num_samples: int = 1):
        super().__init__()
        self.dim = dim
        self.block_size = block_size
        self.num_blocks = dim // block_size
        self.temperature = temperature
        self.num_samples = num_samples

    def forward(self, logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sample discrete block codes with DisARM gradient estimate.

        Args:
            logits: (batch, dim) continuous logits

        Returns:
            (discrete, grad_estimate):
                discrete: (batch, dim) one-hot block codes
                grad_estimate: (batch, dim) gradient through discretization
        """
        batch_size = logits.shape[0]

        # Reshape to blocks: (batch, num_blocks, block_size)
        block_logits = logits.reshape(batch_size, self.num_blocks, self.block_size)

        # Sample discrete block codes via Gumbel-argmax
        u = np.random.uniform(0.01, 0.99, size=block_logits.shape).astype(np.float32)
        gumbel = -np.log(-np.log(u))
        perturbed = block_logits / self.temperature + gumbel

        # Argmax per block → one-hot
        indices = perturbed.argmax(axis=-1)  # (batch, num_blocks)
        discrete = np.zeros_like(block_logits)
        for b in range(batch_size):
            for k in range(self.num_blocks):
                discrete[b, k, indices[b, k]] = 1.0

        # Antithetic sample
        gumbel_anti = -np.log(-np.log(1.0 - u))
        perturbed_anti = block_logits / self.temperature + gumbel_anti
        indices_anti = perturbed_anti.argmax(axis=-1)
        discrete_anti = np.zeros_like(block_logits)
        for b in range(batch_size):
            for k in range(self.num_blocks):
                discrete_anti[b, k, indices_anti[b, k]] = 1.0

        # Gradient estimate via straight-through + antithetic correction
        softmax_probs = np.exp(block_logits / self.temperature)
        softmax_probs = softmax_probs / softmax_probs.sum(axis=-1, keepdims=True)

        # DisARM gradient: difference between samples projected through softmax Jacobian
        diff = discrete - discrete_anti
        grad_estimate = diff * softmax_probs  # approximate Jacobian

        # Reshape back
        discrete_flat = discrete.reshape(batch_size, self.dim)
        grad_flat = grad_estimate.reshape(batch_size, self.dim)

        return discrete_flat, grad_flat

    def __repr__(self):
        return (f"DisARMSampler(dim={self.dim}, block_size={self.block_size}, "
                f"num_blocks={self.num_blocks}, temp={self.temperature})")
