"""
Utilities Module

Helper functions for data loading, visualization, checkpointing, device management, etc.
"""
from .data import (
    # Dataset classes
    Dataset,
    TensorDataset,
    ArrayDataset,
    Subset,
    ConcatDataset,
    # Samplers
    RandomSampler,
    SequentialSampler,
    BatchSampler,
    # DataLoader
    DataLoader,
    default_collate,
    random_split,
    # Transforms
    Compose,
    ToFloat32,
    Normalize,
    Flatten,
    RandomNoise,
    RandomFlip,
    OneHot,
    Lambda,
)
from .checkpoint import save_checkpoint, load_checkpoint
from .device import get_device, set_device, device_count
from .initialization import xavier_uniform_, xavier_normal_, kaiming_uniform_, kaiming_normal_

# Visualization utilities (optional dependency)
try:
    from .visualization import (
        plot_training_history,
        plot_gradient_flow,
        plot_parameter_distribution,
        print_model_summary,
        visualize_attention_weights
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    plot_training_history = None
    plot_gradient_flow = None
    plot_parameter_distribution = None
    print_model_summary = None
    visualize_attention_weights = None

# Multi-backend device management
try:
    from .device_manager import (
        DeviceManager, get_device_manager, get_vulkan_backend,
        get_cuda_backend, get_torch
    )
    DEVICE_MANAGER_AVAILABLE = True
except Exception:
    DEVICE_MANAGER_AVAILABLE = False
    DeviceManager = None
    get_device_manager = None
    get_vulkan_backend = None
    get_cuda_backend = None
    get_torch = None

# HuggingFace bridge
try:
    from .huggingface_bridge import HuggingFaceBridge, get_huggingface_bridge
    HUGGINGFACE_BRIDGE_AVAILABLE = True
except Exception:
    HUGGINGFACE_BRIDGE_AVAILABLE = False
    HuggingFaceBridge = None
    get_huggingface_bridge = None

# PyTorch compatibility
try:
    from .pytorch_compat import (
        Tensor, tensor, zeros, ones, randn, arange, cat, stack,
        from_numpy, to_numpy
    )
    PYTORCH_COMPAT_AVAILABLE = True
except Exception:
    PYTORCH_COMPAT_AVAILABLE = False
    Tensor = None
    tensor = None
    zeros = None
    ones = None
    randn = None
    arange = None
    cat = None
    stack = None
    from_numpy = None
    to_numpy = None

__all__ = [
    # Dataset classes
    'Dataset',
    'TensorDataset',
    'ArrayDataset',
    'Subset',
    'ConcatDataset',
    # Samplers
    'RandomSampler',
    'SequentialSampler',
    'BatchSampler',
    # DataLoader
    'DataLoader',
    'default_collate',
    'random_split',
    # Transforms
    'Compose',
    'ToFloat32',
    'Normalize',
    'Flatten',
    'RandomNoise',
    'RandomFlip',
    'OneHot',
    'Lambda',
    # Checkpoint
    'save_checkpoint',
    'load_checkpoint',
    # Device
    'get_device',
    'set_device',
    'device_count',
    # Initialization
    'xavier_uniform_',
    'xavier_normal_',
    'kaiming_uniform_',
    'kaiming_normal_',
    # Visualization
    'plot_training_history',
    'plot_gradient_flow',
    'plot_parameter_distribution',
    'print_model_summary',
    'visualize_attention_weights',
]

# Conditionally add device manager exports
if DEVICE_MANAGER_AVAILABLE:
    __all__.extend([
        'DeviceManager',
        'get_device_manager',
        'get_vulkan_backend',
        'get_cuda_backend',
        'get_torch',
    ])

# Conditionally add HuggingFace bridge exports
if HUGGINGFACE_BRIDGE_AVAILABLE:
    __all__.extend([
        'HuggingFaceBridge',
        'get_huggingface_bridge',
    ])

# Conditionally add PyTorch compat exports
if PYTORCH_COMPAT_AVAILABLE:
    __all__.extend([
        'Tensor',
        'tensor',
        'zeros',
        'ones',
        'randn',
        'arange',
        'cat',
        'stack',
        'from_numpy',
        'to_numpy',
    ])

# Tensor conversion utilities
try:
    from .tensor_conversion import (
        to_vulkan, to_vulkan_batch, from_vulkan,
        ensure_vulkan_compatible, convert_module_inputs,
        auto_convert_to_vulkan
    )
    TENSOR_CONVERSION_AVAILABLE = True
except Exception:
    TENSOR_CONVERSION_AVAILABLE = False

if TENSOR_CONVERSION_AVAILABLE:
    from .tensor_conversion import VulkanTensor
    __all__.extend([
        'to_vulkan',
        'to_vulkan_batch',
        'to_vulkan_gpu',
        'from_vulkan',
        'ensure_vulkan_compatible',
        'convert_module_inputs',
        'auto_convert_to_vulkan',
        'VulkanTensor',
    ])

# PyTorch operations
try:
    from .pytorch_ops import (
        add, mul, matmul, bmm, relu, gelu, softmax, sigmoid, tanh,
        layer_norm, batch_norm, dropout, conv2d, max_pool2d, avg_pool2d,
        mse_loss, cross_entropy_loss, flatten, reshape, transpose,
        unsqueeze, squeeze
    )
    PYTORCH_OPS_AVAILABLE = True
except Exception:
    PYTORCH_OPS_AVAILABLE = False

# Vulkan Sentence Transformer
try:
    from .vulkan_sentence_transformer import VulkanSentenceTransformer
    VULKAN_SENTENCE_TRANSFORMER_AVAILABLE = True
except Exception:
    VULKAN_SENTENCE_TRANSFORMER_AVAILABLE = False
    VulkanSentenceTransformer = None

if PYTORCH_OPS_AVAILABLE:
    __all__.extend([
        'add', 'mul', 'matmul', 'bmm', 'relu', 'gelu', 'softmax', 'sigmoid', 'tanh',
        'layer_norm', 'batch_norm', 'dropout', 'conv2d', 'max_pool2d', 'avg_pool2d',
        'mse_loss', 'cross_entropy_loss', 'flatten', 'reshape', 'transpose',
        'unsqueeze', 'squeeze',
    ])

if VULKAN_SENTENCE_TRANSFORMER_AVAILABLE:
    __all__.append('VulkanSentenceTransformer')
