"""
Main VulkanCompute class that composes all operation modules.
"""

import numpy as np
from .core import VulkanCore
from .pipelines import VulkanPipelines
from .snn import VulkanSNN
from .faiss import VulkanFAISS
from .fnn import VulkanFNN
from .attention import VulkanAttention
from .memory import VulkanMemory
from .cells import VulkanCells
from .affect import VulkanAffect
from .learning import VulkanLearning
from .fft import VulkanFFT
from .contrastive import VulkanContrastive
from .pooling import VulkanPooling


class VulkanCompute:
    """Complete Vulkan compute backend with GPU dispatch"""
    
    def __init__(self, shader_dir: str = None):
        """Initialize Vulkan compute backend"""
        # Initialize core
        self.core = VulkanCore(shader_dir)
        
        # Initialize pipelines
        self.pipelines = VulkanPipelines(self.core)
        
        # Initialize operation modules
        # Note: architecture can be set later via set_architecture() method
        self.architecture = None
        self.snn = VulkanSNN(self.core, self.pipelines)
        self.faiss = VulkanFAISS(self.core, self.pipelines)
        self.fnn = VulkanFNN(self.core, self.pipelines, self.core.shaders)
        self.attention = VulkanAttention(self.core, self.pipelines, self.core.shaders, architecture=self.architecture)
        self.memory = VulkanMemory(self.core, self.pipelines, self.core.shaders, self.fnn)
        self.cells = VulkanCells(self.core, self.pipelines, self.core.shaders)
        self.affect = VulkanAffect(self.core, self.pipelines, self.core.shaders)
        self.learning = VulkanLearning(self.core, self.pipelines, self.core.shaders)
        self.fft = VulkanFFT(self.core, self.pipelines, self.core.shaders)
        self.contrastive = VulkanContrastive(self.core, self.pipelines, self.core.shaders)
        self.pooling = VulkanPooling(self.core, self.pipelines, self.core.shaders)

        # Expose core device/queue for convenience
        self.device = self.core.device
        self.queue = self.core.queue
        self.shaders = self.core.shaders

    def cleanup(self):
        """Clean up Vulkan resources"""
        # Clear buffer pools first (before device is destroyed)
        if hasattr(self, 'fnn') and self.fnn._pool is not None:
            try:
                self.fnn._pool.clear()
                self.fnn._pool = None
            except Exception:
                pass
        if hasattr(self, 'pipelines'):
            self.pipelines.cleanup()
        if hasattr(self, 'core'):
            self.core.cleanup()
    
    def set_architecture(self, architecture: str):
        """
        Set the model architecture for architecture-specific shader selection.
        
        Args:
            architecture: Model architecture (e.g., 'bert', 'gpt', 't5', 'distilbert')
        """
        self.architecture = architecture
        # Update attention module with new architecture
        self.attention.architecture = architecture
        
        # Expose core methods for backward compatibility (private)
        self._create_buffer = self.core._create_buffer
        self._upload_buffer = self.core._upload_buffer
        self._download_buffer = self.core._download_buffer
        self._dispatch_compute = self.core._dispatch_compute
        self.device = self.core.device
        self.queue = self.core.queue  # Expose compute queue for callers/tests
        self.descriptor_pool = self.core.descriptor_pool
        self.shaders = self.core.shaders
        
        # Public convenience methods
        self.create_pipeline = self.pipelines.get_or_create_pipeline
        self.dispatch = self.core._dispatch_compute
        
        # Expose tiling info
        self.get_tiling_info = self.core.get_tiling_info
        
        # Expose flash attention 2
        self.flash_attention2 = self.attention.flash_attention2
        
        # Expose affect operations
        self.affect_mlp_forward = self.affect.affect_mlp_forward
        self.affect_adam_update = self.affect.affect_adam_update
        
        # Expose optimizer operations
        self.adam_update = self.learning.adam_update
        
        # Expose backward passes for training
        self.activation_gelu_backward = self.fnn.activation_gelu_backward
        self.layernorm_backward = self.fnn.layernorm_backward
        self.softmax_backward = self.fnn.softmax_backward

        # Expose all activation functions
        self.activation_relu = self.fnn.activation_relu
        self.activation_gelu = self.fnn.activation_gelu
        self.activation_silu = self.fnn.activation_silu
        self.activation_softmax = self.fnn.activation_softmax
        self.layernorm = self.fnn.layernorm
        self.linear = self.fnn.linear
        self.linear_backward = self.fnn.linear_backward
        self.residual = self.fnn.residual
        self.dropout = self.fnn.dropout
    
    # SNN operations - delegate to snn module
    def lif_step(self, *args, **kwargs):
        return self.snn.lif_step(*args, **kwargs)
    
    def hebbian_learning(self, *args, **kwargs):
        return self.snn.hebbian_learning(*args, **kwargs)
    
    def stdp_learning(self, *args, **kwargs):
        return self.snn.stdp_learning(*args, **kwargs)
    
    def gif_neuron_step(self, *args, **kwargs):
        return self.snn.gif_neuron_step(*args, **kwargs)
    
    # FAISS operations - delegate to faiss module
    def faiss_compute_distances(self, *args, **kwargs):
        """Compute pairwise distances between query and database vectors"""
        return self.faiss.compute_distances(*args, **kwargs)
    
    def faiss_topk(self, *args, **kwargs):
        """Select top-k smallest distances for each query"""
        return self.faiss.topk(*args, **kwargs)
    
    # FNN operations - delegate to fnn module
    def activation_relu(self, *args, **kwargs):
        """Apply ReLU activation: max(0, x)"""
        return self.fnn.activation_relu(*args, **kwargs)
    
    def activation_gelu(self, *args, **kwargs):
        """Apply GELU activation"""
        return self.fnn.activation_gelu(*args, **kwargs)
    
    def activation_silu(self, *args, **kwargs):
        """Apply SiLU (Swish) activation: x * sigmoid(x)"""
        return self.fnn.activation_silu(*args, **kwargs)
    
    def activation_softmax(self, *args, **kwargs):
        """Apply softmax activation"""
        return self.fnn.activation_softmax(*args, **kwargs)
    
    def xavier_init(self, *args, **kwargs):
        """GPU-accelerated Xavier weight initialization"""
        return self.fnn.xavier_init(*args, **kwargs)

    def layernorm(self, *args, **kwargs):
        """GPU-accelerated layer normalization"""
        return self.fnn.layernorm(*args, **kwargs)

    def linear(self, *args, **kwargs):
        """GPU-accelerated linear projection"""
        return self.fnn.linear(*args, **kwargs)

    def linear_backward(self, *args, **kwargs):
        """GPU-accelerated linear backward pass"""
        return self.fnn.linear_backward(*args, **kwargs)

    # Memory operations - delegate to memory module
    def memory_read(self, *args, **kwargs):
        """Retrieve memories using attention mechanism"""
        return self.memory.memory_read(*args, **kwargs)
    
    def memory_write(self, *args, **kwargs):
        """Write key-value pair to memory"""
        return self.memory.memory_write(*args, **kwargs)
    
    def memory_query_pooling(self, *args, **kwargs):
        """Pool sequence representations into memory queries (CPU fallback)."""
        return self.memory.memory_query_pooling(*args, **kwargs)
    
    def memory_inject_gate(self, *args, **kwargs):
        """Gate between attention output and memory context (CPU fallback)."""
        return self.memory.memory_inject_gate(*args, **kwargs)
    
    # Attention operations - delegate to attention module
    def attention_scores(self, *args, **kwargs):
        """Compute attention scores: Q @ K^T / sqrt(head_dim)"""
        return self.attention.attention_scores(*args, **kwargs)
    
    def attention_mask(self, *args, **kwargs):
        """Apply causal mask to attention scores"""
        return self.attention.attention_mask(*args, **kwargs)
    
    def attention_output(self, *args, **kwargs):
        """Compute attention output: weights @ values"""
        return self.attention.attention_output(*args, **kwargs)
    
    def attention_concat_heads(self, *args, **kwargs):
        """Concatenate attention heads"""
        return self.attention.attention_concat_heads(*args, **kwargs)
    
    # Cell operations - delegate to cells module
    def place_cell(self, *args, **kwargs):
        """Generate place cell firing rates based on agent position"""
        return self.cells.place_cell(*args, **kwargs)
    
    def time_cell(self, *args, **kwargs):
        """Generate time cell firing rates based on elapsed time"""
        return self.cells.time_cell(*args, **kwargs)
    
    # Public buffer management methods
    def create_buffer(self, data_or_size, usage='storage'):
        """
        Create a Vulkan buffer (public API)
        
        Args:
            data_or_size: Either numpy array data, bytes, or integer size in bytes
            usage: Buffer usage ('storage', 'uniform', etc.) or VK_BUFFER_USAGE flag
            
        Returns:
            Tuple of (buffer, memory) handles
        """
        from .base import VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
        
        # Handle usage parameter
        if isinstance(usage, str):
            usage_flag = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT if usage == 'storage' else VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
        else:
            # Assume it's already a Vulkan flag
            usage_flag = usage
        
        # Determine size
        if isinstance(data_or_size, (int, np.integer)):
            size = int(data_or_size)
        elif isinstance(data_or_size, bytes):
            size = len(data_or_size)
        elif hasattr(data_or_size, 'nbytes'):
            # Numpy array
            size = data_or_size.nbytes
        elif isinstance(data_or_size, (list, tuple)):
            # Assume list/tuple of numbers, estimate size
            size = len(data_or_size) * 4  # Assume float32
        else:
            raise ValueError(f"Unsupported data type for buffer creation: {type(data_or_size)}")
        
        return self.core._create_buffer(size, usage_flag)
    
    def upload_buffer(self, buffer, memory, data):
        """Upload data to GPU buffer"""
        return self.core._upload_buffer(buffer, memory, data)
    
    def read_buffer(self, memory, size=None, dtype=np.float32):
        """Read data from GPU buffer"""
        if size is None:
            raise ValueError("Size must be provided for read_buffer")
        return self.core._download_buffer(memory, size, dtype=dtype)
    
    # Learning operations - delegate to learning module
    def fisher_info_update(self, *args, **kwargs):
        """Update Fisher information from gradients"""
        return self.learning.fisher_info_update(*args, **kwargs)
    
    def ewc_penalty(self, *args, **kwargs):
        """Compute EWC penalty for continual learning"""
        return self.learning.ewc_penalty(*args, **kwargs)
    
    def natural_gradient(self, *args, **kwargs):
        """Apply natural gradient scaling"""
        return self.learning.natural_gradient(*args, **kwargs)
    
    def nlms_predict(self, *args, **kwargs):
        """NLMS adaptive filter prediction"""
        return self.learning.nlms_predict(*args, **kwargs)
    
    def nlms_update(self, *args, **kwargs):
        """NLMS adaptive filter weight update"""
        return self.learning.nlms_update(*args, **kwargs)
    
    def whitening_transform(self, *args, **kwargs):
        """Apply whitening transform"""
        return self.learning.whitening_transform(*args, **kwargs)
    
    def continuous_to_spikes(self, *args, **kwargs):
        """Convert continuous features to spike trains"""
        return self.learning.continuous_to_spikes(*args, **kwargs)
    
    def spikes_to_continuous(self, *args, **kwargs):
        """Convert spike trains to continuous features"""
        return self.learning.spikes_to_continuous(*args, **kwargs)
    
    def domain_route(self, *args, **kwargs):
        """Route inputs to experts based on domain"""
        return self.learning.domain_route(*args, **kwargs)
    
    def embedding_lookup(self, *args, **kwargs):
        """GPU-accelerated embedding lookup"""
        return self.learning.embedding_lookup(*args, **kwargs)
    
    # FFT operations - delegate to fft module
    def fft(self, *args, **kwargs):
        """Compute FFT"""
        return self.fft.fft(*args, **kwargs)
    
    def fft_magnitude(self, *args, **kwargs):
        """Compute FFT magnitude"""
        return self.fft.fft_magnitude(*args, **kwargs)
    
    def fft_power_spectrum(self, *args, **kwargs):
        """Compute FFT power spectrum"""
        return self.fft.fft_power_spectrum(*args, **kwargs)
    
    def fft_normalize(self, *args, **kwargs):
        """Normalize FFT output"""
        return self.fft.fft_normalize(*args, **kwargs)
    
    # Contrastive learning operations - delegate to contrastive module
    def contrastive_loss(self, *args, **kwargs):
        """Compute contrastive loss"""
        return self.contrastive.contrastive_loss(*args, **kwargs)
    
    def contrastive_gradient(self, *args, **kwargs):
        """Compute contrastive loss gradients"""
        return self.contrastive.contrastive_gradient(*args, **kwargs)
    
    def __del__(self):
        """Cleanup Vulkan resources"""
        if hasattr(self, 'pipelines'):
            self.pipelines.cleanup()
        if hasattr(self, 'core'):
            self.core.cleanup()
