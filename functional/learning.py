"""
Functional Learning Operations

Uses: fisher-info.glsl, fisher-ewc-penalty.glsl, fisher-natural-gradient.glsl,
      fisher-normalize.glsl, nlms-predict.glsl, nlms-update.glsl, nlms-ensemble.glsl,
      nlms-metrics.glsl, whitening-transform.glsl, whitening-apply.glsl,
      whitening-batch-stats.glsl
"""

import numpy as np


def _get_backend():
    """Get backend instance"""
    try:
        from ..backend.compute import Compute

        return Compute()
    except Exception:
        return None


def fisher_info(
    gradients: np.ndarray,
    fisher: np.ndarray,
    momentum: float = 0.9,
    use_ema: bool = True,
    reset: bool = False,
) -> np.ndarray:
    """
    Update Fisher information estimate from gradients.

    Uses: fisher-info.glsl

    Args:
        gradients: Parameter gradients (num_params,)
        fisher: Current Fisher information (num_params,)
        momentum: EMA momentum for running estimate
        use_ema: Use exponential moving average
        reset: Reset Fisher before accumulation

    Returns:
        Updated Fisher information
    """
    from grilly import Compute

    backend = Compute()
    return backend.fisher_info_update(
        gradients, fisher, momentum=momentum, use_ema=use_ema, reset=reset
    )


def ewc_penalty(
    current_params: np.ndarray,
    important_params: np.ndarray,
    fisher: np.ndarray,
    lambda_ewc: float = 1.0,
) -> float:
    """
    Compute EWC (Elastic Weight Consolidation) penalty.

    Uses: fisher-ewc-penalty.glsl

    Args:
        current_params: Current parameter values
        important_params: Important parameter values (from previous task)
        fisher: Fisher information matrix
        lambda_ewc: EWC regularization strength

    Returns:
        EWC penalty value
    """
    from grilly import Compute

    backend = Compute()
    return backend.ewc_penalty(current_params, important_params, fisher, lambda_ewc=lambda_ewc)


def natural_gradient(gradients: np.ndarray, fisher: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Apply natural gradient scaling using Fisher information.

    Uses: fisher-natural-gradient.glsl

    Args:
        gradients: Parameter gradients
        fisher: Fisher information matrix
        eps: Small constant for numerical stability

    Returns:
        Natural gradient: F^(-1) * grad
    """
    from grilly import Compute

    backend = Compute()
    return backend.natural_gradient(gradients, fisher, eps=eps)


def fisher_normalize(fisher: np.ndarray) -> np.ndarray:
    """
    Normalize Fisher information matrix.

    Uses: fisher-normalize.glsl

    Args:
        fisher: Fisher information matrix

    Returns:
        Normalized Fisher information
    """
    backend = _get_backend()

    # Try GPU shader if available
    if backend and hasattr(backend, "shaders") and "fisher-normalize" in backend.shaders:
        try:
            # GPU Fisher normalization would go here
            # For now, use CPU fallback
            pass
        except Exception:
            pass  # Fall back to CPU

    # CPU fallback
    fisher_sum = np.sum(fisher)
    if fisher_sum > 0:
        return fisher / fisher_sum
    return fisher


def nlms_predict(x: np.ndarray, w: np.ndarray, bias: float = 0.0) -> float:
    """
    NLMS prediction.

    Uses: nlms-predict.glsl

    Args:
        x: Input features (n_features,)
        w: Weight vector (n_features,)
        bias: Bias term

    Returns:
        Predicted value
    """
    from grilly import Compute

    backend = Compute()
    return backend.nlms_predict(x, w, bias)


def nlms_update(
    x: np.ndarray, y_true: float, w: np.ndarray, bias: float, mu: float = 0.5, eps: float = 1e-6
) -> tuple[np.ndarray, float]:
    """
    NLMS weight update.

    Uses: nlms-update.glsl

    Args:
        x: Input features (n_features,)
        y_true: True target value
        w: Current weight vector (n_features,)
        bias: Current bias
        mu: Learning rate
        eps: Small constant for numerical stability

    Returns:
        (updated_weights, updated_bias)
    """
    from grilly import Compute

    backend = Compute()
    return backend.nlms_update(x, y_true, w, bias, mu=mu, eps=eps)


def nlms_ensemble(x: np.ndarray, weights_list: list, biases_list: list) -> np.ndarray:
    """
    NLMS ensemble prediction.

    Uses: nlms-ensemble.glsl

    Args:
        x: Input features (n_features,)
        weights_list: List of weight vectors
        biases_list: List of bias values

    Returns:
        Ensemble predictions (num_experts,)
    """
    backend = _get_backend()

    # Try GPU shader if available
    if backend and hasattr(backend, "shaders") and "nlms-ensemble" in backend.shaders:
        try:
            # GPU NLMS ensemble would go here
            # For now, use CPU fallback
            pass
        except Exception:
            pass  # Fall back to CPU

    # CPU fallback
    predictions = []
    for w, b in zip(weights_list, biases_list):
        pred = np.dot(x, w) + b
        predictions.append(pred)
    return np.array(predictions, dtype=np.float32)


def nlms_metrics(errors: np.ndarray, update_count: int) -> dict:
    """
    Compute NLMS metrics.

    Uses: nlms-metrics.glsl

    Args:
        errors: Prediction errors (num_updates,)
        update_count: Number of updates

    Returns:
        Dictionary with metrics (rmse, mae, etc.)
    """
    backend = _get_backend()

    # Try GPU shader if available
    if backend and hasattr(backend, "shaders") and "nlms-metrics" in backend.shaders:
        try:
            # GPU NLMS metrics would go here
            # For now, use CPU fallback
            pass
        except Exception:
            pass  # Fall back to CPU

    # CPU fallback
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    return {"rmse": float(rmse), "mae": float(mae), "update_count": update_count}


def whitening_transform(
    data: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply whitening transform to data.

    Uses: whitening-transform.glsl

    Args:
        data: Input data (batch, features)
        mean: Optional precomputed mean (features,)
        std: Optional precomputed std (features,)

    Returns:
        (whitened_data, mean, std)
    """
    from grilly import Compute

    backend = Compute()
    return backend.whitening_transform(data, mean=mean, std=std)


def whitening_apply(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Apply precomputed whitening to data.

    Uses: whitening-apply.glsl

    Args:
        data: Input data (batch, features)
        mean: Mean vector (features,)
        std: Std vector (features,)

    Returns:
        Whitened data
    """
    backend = _get_backend()

    # Try GPU shader if available
    if backend and hasattr(backend, "shaders") and "whitening-apply" in backend.shaders:
        try:
            # GPU whitening apply would go here
            # For now, use CPU fallback
            pass
        except Exception:
            pass  # Fall back to CPU

    # CPU fallback
    return (data - mean) / (std + 1e-8)


def whitening_batch_stats(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute batch statistics for whitening.

    Uses: whitening-batch-stats.glsl

    Args:
        data: Input data (batch, features)

    Returns:
        (mean, std)
    """
    backend = _get_backend()

    # Try GPU shader if available
    if backend and hasattr(backend, "shaders") and "whitening-batch-stats" in backend.shaders:
        try:
            # GPU whitening batch stats would go here
            # For now, use CPU fallback
            pass
        except Exception:
            pass  # Fall back to CPU

    # CPU fallback
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return mean, std
