"""VQ-VSA Quantizer — Straight-Through Estimator for block-codes.

Bridges continuous CNN features to discrete VSA block-codes:
1. Argmax per block to get hard one-hot codes (forward)
2. Straight-Through Estimator copies gradients past the argmax (backward)
3. Commitment loss pulls continuous output toward discrete codes

Works with both grilly Variable autograd and numpy.
"""

from __future__ import annotations

import numpy as np


def vsa_quantize(continuous_logits: np.ndarray, k: int = 8, l: int = 64):
    """Quantize continuous logits to discrete block-codes with STE.

    Forward: argmax per block → hard one-hot
    Backward: straight-through (gradient bypasses argmax)

    Args:
        continuous_logits: (d,) or (n, d) continuous CNN output.
        k: Number of blocks.
        l: Block length.

    Returns:
        (quantized, commitment_loss, grad_fn)
        quantized: discrete one-hot block-codes, same shape as input
        commitment_loss: MSE between continuous and discrete (scalar)
    """
    orig_shape = continuous_logits.shape
    logits = continuous_logits.reshape(-1, k, l)
    batch = logits.shape[0]

    # Hard quantization: argmax per block
    indices = np.argmax(logits, axis=-1)  # (batch, k)
    hard = np.zeros_like(logits)
    for b in range(batch):
        for j in range(k):
            hard[b, j, indices[b, j]] = 1.0

    # Straight-Through Estimator:
    # Forward: use hard codes
    # Backward: gradient flows through as if hard = continuous
    # Implemented as: quantized = continuous + (hard - continuous).detach()
    # In numpy: just return hard, but provide gradient w.r.t. continuous
    quantized = hard.reshape(orig_shape)

    # Commitment loss: pull continuous toward discrete
    commitment_loss = float(0.25 * np.mean((continuous_logits - quantized) ** 2))

    return quantized, commitment_loss


def vsa_vq_loss(
    continuous_logits: np.ndarray,
    target_attrs: dict,
    codebook,
    bc,
    k: int = 8,
    l: int = 64,
    beta: float = 0.25,
) -> tuple[float, np.ndarray]:
    """VQ-VSA Loss: semantic unbinding loss + commitment loss.

    1. Quantize CNN output to discrete block-codes
    2. Unbind spatial role from discrete codes
    3. Compare unbound entity to clean target attribute codes
    4. Commitment loss pulls continuous output toward discrete space

    Args:
        continuous_logits: Raw CNN output (d,).
        target_attrs: Dict with Type, Size, Color, Angle (int 0-9).
        codebook: VSACodebook with frozen attribute codes.
        bc: BlockCodes instance.
        k, l: Block dimensions.
        beta: Commitment loss weight.

    Returns:
        (total_loss, gradient w.r.t. continuous_logits)
    """
    d = k * l

    # 1. Quantize
    quantized, commit_loss = vsa_quantize(continuous_logits, k, l)

    # 2. Compute cosine similarity to each target attribute code
    # (without unbinding — just direct similarity to bundled target)
    target_indices = codebook.target_indices(target_attrs)
    q_flat = continuous_logits.ravel()
    q_norm = np.linalg.norm(q_flat) + 1e-8
    q_hat = q_flat / q_norm

    W_hat = codebook.W_hat  # (n, d) normalized codebook
    sims = W_hat @ q_hat  # (n,)

    # Additive CE-style loss on target indices
    sims_clipped = np.clip(sims, -0.999, 0.999)
    temperature = 5.0
    logits_scaled = np.clip(temperature * sims_clipped, -50.0, 50.0)
    logits_shifted = logits_scaled - logits_scaled.max()
    exp_l = np.exp(logits_shifted)
    probs = exp_l / (exp_l.sum() + 1e-8)

    target_probs = [max(probs[idx], 1e-8) for idx in target_indices]
    semantic_loss = -np.mean([np.log(p) for p in target_probs])

    # 3. Gradient (STE: gradient flows through as if quantized = continuous)
    expected_w = probs @ W_hat
    target_w = np.mean([W_hat[idx] for idx in target_indices], axis=0)
    grad_semantic = temperature * (expected_w - target_w) / q_norm

    # Commitment gradient: d/dx MSE(x, quantized.detach()) = 2*beta*(x - quantized)
    grad_commit = 2 * beta * (continuous_logits.ravel() - quantized.ravel())

    # Total
    total_loss = semantic_loss + commit_loss
    total_grad = np.clip(grad_semantic + grad_commit, -1.0, 1.0).astype(np.float32)

    return total_loss, total_grad.reshape(continuous_logits.shape)


if __name__ == "__main__":
    # Quick test
    from cubemind.ops.block_codes import BlockCodes
    from cubemind.perception.additive_ce import VSACodebook

    bc = BlockCodes(8, 64)
    codebook = VSACodebook(bc)

    # Simulate CNN output
    logits = np.random.randn(512).astype(np.float32)
    target = {"Type": 3, "Size": 2, "Color": 5, "Angle": 1}

    loss, grad = vsa_vq_loss(logits, target, codebook, bc)
    print(f"VQ-VSA loss: {loss:.4f}")
    print(f"grad norm: {np.linalg.norm(grad):.4f}")

    # SGD step
    logits -= 0.01 * grad
    loss2, _ = vsa_vq_loss(logits, target, codebook, bc)
    print(f"after step: {loss2:.4f}, decreased: {loss2 < loss}")
