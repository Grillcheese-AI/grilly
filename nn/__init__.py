"""
Neural Network Modules (PyTorch-like API)

This module provides a PyTorch-like interface for GPU-accelerated neural network operations
using Vulkan compute shaders.
"""
from .module import Module
from .parameter import Parameter, parameter
from .modules import (
    Linear,
    LayerNorm,
    Dropout,
    ReLU,
    GELU,
    SiLU,
    Softmax,
    Softplus,
    MultiheadAttention,
    FlashAttention2,
    Embedding,
    Sequential,
    Residual,
)
from .snn import (
    LIFNeuron,
    SNNLayer,
    HebbianLayer,
    STDPLayer,
    GIFNeuron,
    SNNMatMul,
    SNNSoftmax,
    SNNRMSNorm,
    SNNReadout,
    Synapse,
)

# Placeholders for modules to be created
# These will be imported when the respective modules are implemented
from .memory import (
    MemoryRead,
    MemoryWrite,
    MemoryContextAggregate,
    MemoryQueryPooling,
    MemoryInject,
    MemoryInjectConcat,
    MemoryInjectGate,
    MemoryInjectResidual,
)

from .cells import (
    PlaceCell,
    TimeCell,
    ThetaGammaEncoder,
)

from .transformer import (
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    RoPE,
    ProsodyModulatedAttention,
)

from .hippocampal import (
    HippocampalTransformerLayer,
)

from .routing import (
    DomainRouter,
    DomainPredictor,
    DomainClassifier,
    ExpertCombiner,
)

from .affect import (
    AffectMLP,
)

from .capsule import (
    CapsuleProject,
    SemanticEncoder,
    DentateGyrus,
)

from .capsule_embedding import (
    CapsuleEmbedding,
    ContrastiveLoss,
)

from .decoding import (
    GreedyDecoder,
    SampleDecoder,
)

from .loss import (
    MSELoss,
    CrossEntropyLoss,
    BCELoss,
)

# Autograd utilities
from .autograd import (
    Variable,
    GradFn,
    Function,
    FunctionCtx,
    # Arithmetic
    add,
    sub,
    mul,
    div,
    neg,
    pow,
    matmul,
    # Reductions
    sum,
    mean,
    max,
    min,
    var,
    std,
    norm,
    # Activations
    relu,
    sigmoid,
    tanh,
    exp,
    log,
    sqrt,
    abs,
    clamp,
    gelu,
    silu,
    leaky_relu,
    elu,
    softplus,
    softmax,
    # Trigonometric
    sin,
    cos,
    tan,
    asin,
    acos,
    atan,
    atan2,
    # Shapes
    reshape,
    transpose,
    squeeze,
    unsqueeze,
    index,
    flatten,
    view,
    expand,
    repeat,
    permute,
    contiguous,
    clone,
    concat,
    stack,
    where,
    # Comparisons
    eq,
    ne,
    lt,
    le,
    gt,
    ge,
    # Loss functions
    cross_entropy,
    mse_loss,
    l1_loss,
    smooth_l1_loss,
    bce_loss,
    bce_with_logits_loss,
    nll_loss,
    kl_div_loss,
    # Context managers
    enable_grad,
    no_grad,
    is_grad_enabled,
    # Factory functions
    tensor,
    zeros,
    ones,
    randn,
    rand,
    linspace,
    arange,
    eye,
    full,
)

# Backend autograd integration
try:
    from ..backend.autograd_core import (
        GradientTape,
        ComputationNode,
        ModuleTracer,
        TrainingContext,
        backward_ops,
        no_grad as autograd_no_grad,
        enable_grad as autograd_enable_grad,
        is_grad_enabled,
        backward,
    )
    AUTOGRAD_CORE_AVAILABLE = True
except ImportError:
    AUTOGRAD_CORE_AVAILABLE = False

# Multimodal techniques
from .multimodal import (
    BottleneckFusion,
    PerceiverIO,
    CrossModalAttentionFusion,
    ImageBindFusion,
    PerceiverResampler,
    FlamingoFusion,
    VisionLanguageModel,
    VLMLayer,
)

__all__ = [
    # Base class
    'Module',
    
    # Standard layers
    'Linear',
    'LayerNorm',
    'Dropout',
    'ReLU',
    'GELU',
    'SiLU',
    'Softmax',
    'Softplus',
    'MultiheadAttention',
    'FlashAttention2',
    'Embedding',
    'Sequential',
    'Residual',
    
    # SNN layers
    'LIFNeuron',
    'SNNLayer',
    'HebbianLayer',
    'STDPLayer',
    'GIFNeuron',
    'SNNMatMul',
    'SNNSoftmax',
    'SNNRMSNorm',
    'SNNReadout',
    'Synapse',
    
    # Memory layers (when implemented)
    'MemoryRead',
    'MemoryWrite',
    'MemoryContextAggregate',
    'MemoryQueryPooling',
    'MemoryInject',
    'MemoryInjectConcat',
    'MemoryInjectGate',
    'MemoryInjectResidual',
    
    # Cell layers (when implemented)
    'PlaceCell',
    'TimeCell',
    'ThetaGammaEncoder',
    
    # Transformer layers (when implemented)
    'TransformerEncoderLayer',
    'TransformerDecoderLayer',
    'RoPE',
    'ProsodyModulatedAttention',
    
    # Hippocampal transformer (when implemented)
    'HippocampalTransformerLayer',
    
    # Routing layers (when implemented)
    'DomainRouter',
    'DomainPredictor',
    'DomainClassifier',
    'ExpertCombiner',
    
    # Affect layers (when implemented)
    'AffectMLP',
    
    # Capsule layers
    'CapsuleProject',
    'SemanticEncoder',
    'DentateGyrus',
    'CapsuleEmbedding',
    'ContrastiveLoss',
    
    # Decoding layers (when implemented)
    'GreedyDecoder',
    'SampleDecoder',
    
    # Loss functions
    'MSELoss',
    'CrossEntropyLoss',
    'BCELoss',

    # Autograd - Core
    'Variable',
    'GradFn',
    'Function',
    'FunctionCtx',
    # Arithmetic
    'add',
    'sub',
    'mul',
    'div',
    'neg',
    'pow',
    'matmul',
    # Reductions
    'sum',
    'mean',
    'max',
    'min',
    'var',
    'std',
    'norm',
    # Activations
    'relu',
    'sigmoid',
    'tanh',
    'exp',
    'log',
    'sqrt',
    'abs',
    'clamp',
    'gelu',
    'silu',
    'leaky_relu',
    'elu',
    'softplus',
    'softmax',
    # Trigonometric
    'sin',
    'cos',
    'tan',
    'asin',
    'acos',
    'atan',
    'atan2',
    # Shapes
    'reshape',
    'transpose',
    'squeeze',
    'unsqueeze',
    'index',
    'flatten',
    'view',
    'expand',
    'repeat',
    'permute',
    'contiguous',
    'clone',
    'concat',
    'stack',
    'where',
    # Comparisons
    'eq',
    'ne',
    'lt',
    'le',
    'gt',
    'ge',
    # Loss functions
    'cross_entropy',
    'mse_loss',
    'l1_loss',
    'smooth_l1_loss',
    'bce_loss',
    'bce_with_logits_loss',
    'nll_loss',
    'kl_div_loss',
    # Context managers
    'enable_grad',
    'no_grad',
    'is_grad_enabled',
    # Factory functions
    'tensor',
    'zeros',
    'ones',
    'randn',
    'rand',
    'linspace',
    'arange',
    'eye',
    'full',
]

# Add autograd core exports if available
if AUTOGRAD_CORE_AVAILABLE:
    __all__.extend([
        'GradientTape',
        'ComputationNode',
        'ModuleTracer',
        'TrainingContext',
        'backward_ops',
        'is_grad_enabled',
        'backward',
    ])

# Multimodal techniques
__all__.extend([
    'BottleneckFusion',
    'PerceiverIO',
    'CrossModalAttentionFusion',
    'ImageBindFusion',
    'PerceiverResampler',
    'FlamingoFusion',
    'VisionLanguageModel',
    'VLMLayer',
])
