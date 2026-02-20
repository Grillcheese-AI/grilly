"""
Functional API (PyTorch-like)
Similar to torch.nn.functional
"""

from .activations import (
    gelu,
    relu,
    silu,
    softmax,
    softplus,
)
from .attention import (
    attention,
    flash_attention2,
)
from .bridge import (
    bridge_temporal_weights,
    continuous_to_spikes,
    spikes_to_continuous,
)
from .cells import (
    place_cell,
    theta_gamma_encoding,
    time_cell,
)
from .dropout import (
    dropout,
)
from .embedding import (
    embedding_attention,
    embedding_ffn,
    embedding_lookup,
    embedding_normalize,
    embedding_pool,
    embedding_position,
)
from .faiss import (
    faiss_distance,
    faiss_ivf_filter,
    faiss_kmeans_update,
    faiss_quantize,
    faiss_topk,
)
from .fft import (
    fft,
    fft_magnitude,
    fft_power_spectrum,
    ifft,
)
from .learning import (
    ewc_penalty,
    fisher_info,
    fisher_normalize,
    natural_gradient,
    nlms_ensemble,
    nlms_metrics,
    nlms_predict,
    nlms_update,
    whitening_apply,
    whitening_batch_stats,
    whitening_transform,
)
from .linear import (
    linear,
)
from .loss import (
    binary_cross_entropy,
    cross_entropy,
)
from .memory import (
    memory_context_aggregate,
    memory_inject_concat,
    memory_inject_gate,
    memory_inject_residual,
    memory_query_pooling,
    memory_read,
    memory_write,
)
from .normalization import (
    layer_norm,
)
from .snn import (
    if_step,
    lif_step,
    multi_step_forward,
    reset_net,
    seq_to_ann_forward,
    set_step_mode,
)

__all__ = [
    # Activations
    "relu",
    "gelu",
    "silu",
    "softmax",
    "softplus",
    # Linear
    "linear",
    # Normalization
    "layer_norm",
    # Dropout
    "dropout",
    # Attention
    "attention",
    "flash_attention2",
    # Loss
    "cross_entropy",
    "binary_cross_entropy",
    # FFT
    "fft",
    "ifft",
    "fft_magnitude",
    "fft_power_spectrum",
    # Memory
    "memory_read",
    "memory_write",
    "memory_context_aggregate",
    "memory_query_pooling",
    "memory_inject_concat",
    "memory_inject_gate",
    "memory_inject_residual",
    # Cells
    "place_cell",
    "time_cell",
    "theta_gamma_encoding",
    # Learning
    "fisher_info",
    "ewc_penalty",
    "natural_gradient",
    "fisher_normalize",
    "nlms_predict",
    "nlms_update",
    "nlms_ensemble",
    "nlms_metrics",
    "whitening_transform",
    "whitening_apply",
    "whitening_batch_stats",
    # Bridge
    "continuous_to_spikes",
    "spikes_to_continuous",
    "bridge_temporal_weights",
    # Embedding
    "embedding_lookup",
    "embedding_normalize",
    "embedding_position",
    "embedding_pool",
    "embedding_ffn",
    "embedding_attention",
    # FAISS
    "faiss_distance",
    "faiss_topk",
    "faiss_ivf_filter",
    "faiss_kmeans_update",
    "faiss_quantize",
    # SNN
    "lif_step",
    "if_step",
    "multi_step_forward",
    "seq_to_ann_forward",
    "reset_net",
    "set_step_mode",
]
