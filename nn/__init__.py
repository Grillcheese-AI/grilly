"""Neural network module namespace for the Grilly framework."""

from .affect import (
    AffectMLP,
)

# Autograd utilities
from .autograd import (
    Function,
    FunctionCtx,
    GradFn,
    Variable,
    abs,
    acos,
    # Arithmetic
    add,
    arange,
    asin,
    atan,
    atan2,
    bce_loss,
    bce_with_logits_loss,
    clamp,
    clone,
    concat,
    contiguous,
    cos,
    # Loss functions
    cross_entropy,
    div,
    elu,
    # Context managers
    enable_grad,
    # Comparisons
    eq,
    exp,
    expand,
    eye,
    flatten,
    full,
    ge,
    gelu,
    gt,
    index,
    is_grad_enabled,
    kl_div_loss,
    l1_loss,
    le,
    leaky_relu,
    linspace,
    log,
    lt,
    matmul,
    max,
    mean,
    min,
    mse_loss,
    mul,
    ne,
    neg,
    nll_loss,
    no_grad,
    norm,
    ones,
    permute,
    pow,
    rand,
    randn,
    # Activations
    relu,
    repeat,
    # Shapes
    reshape,
    sigmoid,
    silu,
    # Trigonometric
    sin,
    smooth_l1_loss,
    softmax,
    softplus,
    sqrt,
    squeeze,
    stack,
    std,
    sub,
    # Reductions
    sum,
    tan,
    tanh,
    # Factory functions
    tensor,
    transpose,
    unsqueeze,
    var,
    view,
    where,
    zeros,
)
from .capsule import (
    CapsuleProject,
    DentateGyrus,
    SemanticEncoder,
)
from .capsule_embedding import (
    CapsuleEmbedding,
    ContrastiveLoss,
)
from .cells import (
    PlaceCell,
    ThetaGammaEncoder,
    TimeCell,
)

# Convolutional layers
from .conv import (
    Conv1d,
    Conv2d,
)
from .decoding import (
    GreedyDecoder,
    SampleDecoder,
)
from .hippocampal import (
    HippocampalTransformerLayer,
)
from .loss import (
    BCELoss,
    CrossEntropyLoss,
    MSELoss,
)

# Placeholders for modules to be created
# These will be imported when the respective modules are implemented
from .memory import (
    MemoryContextAggregate,
    MemoryInject,
    MemoryInjectConcat,
    MemoryInjectGate,
    MemoryInjectResidual,
    MemoryQueryPooling,
    MemoryRead,
    MemoryWrite,
)
from .module import Module
from .modules import (
    GCU,
    GELU,
    Dropout,
    Embedding,
    FlashAttention2,
    LayerNorm,
    Linear,
    MultiheadAttention,
    ReLU,
    Residual,
    RoSwish,
    Sequential,
    SiLU,
    Softmax,
    Softplus,
    SwiGLU,
)

# Normalization layers
from .normalization import (
    BatchNorm1d,
    BatchNorm2d,
)
from .parameter import Parameter, parameter

# Pooling layers
from .pooling import (
    AdaptiveAvgPool2d,
    AdaptiveMaxPool2d,
    AvgPool2d,
    MaxPool2d,
)

# Recurrent layers
from .rnn import (
    GRU,
    LSTM,
    GRUCell,
    LSTMCell,
)
from .routing import (
    DomainClassifier,
    DomainPredictor,
    DomainRouter,
    ExpertCombiner,
)
from .snn import (
    GIFNeuron,
    HebbianLayer,
    LIFNeuron,
    SNNLayer,
    SNNMatMul,
    SNNReadout,
    SNNRMSNorm,
    SNNSoftmax,
    STDPLayer,
    Synapse,
)

# New SNN framework (surrogate gradients, nodes, containers, etc.)
from .snn_ann2snn import Converter, VoltageScaler
from .snn_attention import (
    ChannelQKAttention,
    MultiDimensionalAttention,
    QKAttention,
    SpikingSelfAttention,
    TemporalWiseAttention,
    TokenQKAttention,
)
from .snn_base import BaseNode, MemoryModule
from .snn_containers import Flatten, MultiStepContainer, SeqToANNContainer
from .snn_monitor import Monitor
from .snn_neurons import IFNode, LIFNode, ParametricLIFNode
from .snn_normalization import (
    BatchNormThroughTime1d,
    BatchNormThroughTime2d,
    NeuNorm,
    TemporalEffectiveBatchNorm1d,
    TemporalEffectiveBatchNorm2d,
    ThresholdDependentBatchNorm1d,
    ThresholdDependentBatchNorm2d,
)
from .snn_surrogate import ATan, FastSigmoid, Sigmoid, SurrogateFunction
from .snn_synapses import (
    DualTimescaleSynapse,
    ElementWiseRecurrentContainer,
    STPSynapse,
    SynapseFilter,
)
from .transformer import (
    ProsodyModulatedAttention,
    RoPE,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)

# Backend autograd integration
try:
    from ..backend.autograd_core import (
        ComputationNode,
        GradientTape,
        ModuleTracer,
        TrainingContext,
        backward,
        backward_ops,
        is_grad_enabled,
    )
    from ..backend.autograd_core import (
        enable_grad as autograd_enable_grad,
    )
    from ..backend.autograd_core import (
        no_grad as autograd_no_grad,
    )

    AUTOGRAD_CORE_AVAILABLE = True
except ImportError:
    AUTOGRAD_CORE_AVAILABLE = False

# Multimodal techniques
from .multimodal import (
    BottleneckFusion,
    CrossModalAttentionFusion,
    FlamingoFusion,
    ImageBindFusion,
    PerceiverIO,
    PerceiverResampler,
    VisionLanguageModel,
    VLMLayer,
)

__all__ = [
    # Base class
    "Module",
    # Standard layers
    "Linear",
    "LayerNorm",
    "Dropout",
    "ReLU",
    "GELU",
    "SiLU",
    "GCU",
    "RoSwish",
    "SwiGLU",
    "Softmax",
    "Softplus",
    "MultiheadAttention",
    "FlashAttention2",
    "Embedding",
    "Sequential",
    "Residual",
    # SNN layers (legacy)
    "LIFNeuron",
    "SNNLayer",
    "HebbianLayer",
    "STDPLayer",
    "GIFNeuron",
    "SNNMatMul",
    "SNNSoftmax",
    "SNNRMSNorm",
    "SNNReadout",
    "Synapse",
    # SNN framework (new)
    "SurrogateFunction",
    "ATan",
    "Sigmoid",
    "FastSigmoid",
    "MemoryModule",
    "BaseNode",
    "IFNode",
    "LIFNode",
    "ParametricLIFNode",
    "MultiStepContainer",
    "SeqToANNContainer",
    "Flatten",
    "NeuNorm",
    "ThresholdDependentBatchNorm1d",
    "ThresholdDependentBatchNorm2d",
    "TemporalEffectiveBatchNorm1d",
    "TemporalEffectiveBatchNorm2d",
    "BatchNormThroughTime1d",
    "BatchNormThroughTime2d",
    "DualTimescaleSynapse",
    "ElementWiseRecurrentContainer",
    "STPSynapse",
    "SynapseFilter",
    "TemporalWiseAttention",
    "MultiDimensionalAttention",
    "SpikingSelfAttention",
    "QKAttention",
    "TokenQKAttention",
    "ChannelQKAttention",
    "Converter",
    "VoltageScaler",
    "Monitor",
    # Memory layers (when implemented)
    "MemoryRead",
    "MemoryWrite",
    "MemoryContextAggregate",
    "MemoryQueryPooling",
    "MemoryInject",
    "MemoryInjectConcat",
    "MemoryInjectGate",
    "MemoryInjectResidual",
    # Cell layers (when implemented)
    "PlaceCell",
    "TimeCell",
    "ThetaGammaEncoder",
    # Transformer layers (when implemented)
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    "RoPE",
    "ProsodyModulatedAttention",
    # Hippocampal transformer (when implemented)
    "HippocampalTransformerLayer",
    # Routing layers (when implemented)
    "DomainRouter",
    "DomainPredictor",
    "DomainClassifier",
    "ExpertCombiner",
    # Affect layers (when implemented)
    "AffectMLP",
    # Capsule layers
    "CapsuleProject",
    "SemanticEncoder",
    "DentateGyrus",
    "CapsuleEmbedding",
    "ContrastiveLoss",
    # Decoding layers (when implemented)
    "GreedyDecoder",
    "SampleDecoder",
    # Loss functions
    "MSELoss",
    "CrossEntropyLoss",
    "BCELoss",
    # Convolutional layers
    "Conv1d",
    "Conv2d",
    # Normalization layers
    "BatchNorm1d",
    "BatchNorm2d",
    # Pooling layers
    "MaxPool2d",
    "AvgPool2d",
    "AdaptiveMaxPool2d",
    "AdaptiveAvgPool2d",
    # Recurrent layers
    "LSTM",
    "LSTMCell",
    "GRU",
    "GRUCell",
    # Autograd - Core
    "Variable",
    "GradFn",
    "Function",
    "FunctionCtx",
    # Arithmetic
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "pow",
    "matmul",
    # Reductions
    "sum",
    "mean",
    "max",
    "min",
    "var",
    "std",
    "norm",
    # Activations
    "relu",
    "sigmoid",
    "tanh",
    "exp",
    "log",
    "sqrt",
    "abs",
    "clamp",
    "gelu",
    "silu",
    "leaky_relu",
    "elu",
    "softplus",
    "softmax",
    # Trigonometric
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    # Shapes
    "reshape",
    "transpose",
    "squeeze",
    "unsqueeze",
    "index",
    "flatten",
    "view",
    "expand",
    "repeat",
    "permute",
    "contiguous",
    "clone",
    "concat",
    "stack",
    "where",
    # Comparisons
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    # Loss functions
    "cross_entropy",
    "mse_loss",
    "l1_loss",
    "smooth_l1_loss",
    "bce_loss",
    "bce_with_logits_loss",
    "nll_loss",
    "kl_div_loss",
    # Context managers
    "enable_grad",
    "no_grad",
    "is_grad_enabled",
    # Factory functions
    "tensor",
    "zeros",
    "ones",
    "randn",
    "rand",
    "linspace",
    "arange",
    "eye",
    "full",
]

# Add autograd core exports if available
if AUTOGRAD_CORE_AVAILABLE:
    __all__.extend(
        [
            "GradientTape",
            "ComputationNode",
            "ModuleTracer",
            "TrainingContext",
            "backward_ops",
            "is_grad_enabled",
            "backward",
        ]
    )

# Multimodal techniques
__all__.extend(
    [
        "BottleneckFusion",
        "PerceiverIO",
        "CrossModalAttentionFusion",
        "ImageBindFusion",
        "PerceiverResampler",
        "FlamingoFusion",
        "VisionLanguageModel",
        "VLMLayer",
    ]
)

# LoRA (Low-Rank Adaptation) for efficient fine-tuning
from .lora import (
    LoRAAttention,
    LoRAConfig,
    LoRAEmbedding,
    LoRALinear,
    LoRAModel,
    apply_lora_to_linear,
    calculate_lora_params,
)

__all__.extend(
    [
        "LoRAConfig",
        "LoRALinear",
        "LoRAEmbedding",
        "LoRAAttention",
        "LoRAModel",
        "apply_lora_to_linear",
        "calculate_lora_params",
    ]
)
