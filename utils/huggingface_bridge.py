"""
HuggingFace Bridge for GPU Compatibility

Provides a wrapper to run HuggingFace models (tokenizers, transformers) on CUDA
while using Vulkan for custom operations. Handles seamless tensor conversion
between PyTorch CUDA tensors and numpy arrays for Vulkan.
"""
from typing import Optional, Union, Dict, Any, List, Tuple
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForCausalLM,
        AutoModelForSequenceClassification, PreTrainedTokenizer,
        PreTrainedModel
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModel = None
    AutoModelForCausalLM = None
    AutoModelForSequenceClassification = None
    PreTrainedTokenizer = None
    PreTrainedModel = None

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

from .device_manager import get_device_manager
import logging

logger = logging.getLogger(__name__)


class HuggingFaceBridge:
    """
    Bridge for running HuggingFace models on CUDA while using Vulkan for
    custom operations.
    
    Handles:
    - Tokenizer operations (CPU or CUDA)
    - Model inference on CUDA
    - Tensor conversion between PyTorch and numpy
    - Embedding extraction for Vulkan operations
    """
    
    def __init__(self, cuda_device: Optional[Union[str, int]] = None):
        """
        Initialize HuggingFace bridge.
        
        Args:
            cuda_device: CUDA device ('cuda:0', 'cuda:1', or device index)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for HuggingFace bridge. Install with: pip install torch")
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers is required. Install with: pip install transformers")
        
        self.device_manager = get_device_manager()
        
        # Set CUDA device (only if CUDA is available)
        try:
            torch = self.device_manager.torch
            if torch.cuda.is_available():
                if cuda_device is not None:
                    if isinstance(cuda_device, int):
                        self.device_manager.set_device('cuda', cuda_device)
                    else:
                        self.device_manager.set_device('cuda')
                        self.device_manager._cuda_device = torch.device(cuda_device)
                else:
                    self.device_manager.set_device('cuda')
                
                self.cuda_device = self.device_manager.get_cuda_device()
            else:
                # CUDA not available - use CPU/Vulkan fallback for AMD systems
                self.device_manager.set_device('cpu')
                self.cuda_device = None
        except (RuntimeError, AssertionError, AttributeError) as e:
            # CUDA not available - use CPU/Vulkan fallback
            if "CUDA" in str(e) or "not compiled" in str(e) or "is_available" in str(e):
                # For AMD/Vulkan-only systems, we can still use the bridge for tokenization
                # but model inference will need to be handled differently
                self.device_manager.set_device('cpu')
                self.cuda_device = None
            else:
                raise
        
        self.torch = self.device_manager.torch
        
        # Cache for loaded models
        self._tokenizers: Dict[str, PreTrainedTokenizer] = {}
        self._models: Dict[str, PreTrainedModel] = {}
        self._sentence_models: Dict[str, Any] = {}  # Cache for sentence-transformers models
    
    def load_tokenizer(self, model_name: str, **kwargs) -> PreTrainedTokenizer:
        """
        Load a HuggingFace tokenizer.
        
        Args:
            model_name: Model name or path
            **kwargs: Additional arguments for AutoTokenizer
        
        Returns:
            Tokenizer instance
        """
        if model_name in self._tokenizers:
            return self._tokenizers[model_name]
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self._tokenizers[model_name] = tokenizer
        return tokenizer
    
    def load_model(
        self,
        model_name: str,
        model_type: str = 'auto',
        **kwargs
    ) -> PreTrainedModel:
        """
        Load a HuggingFace model and move it to CUDA.
        
        Args:
            model_name: Model name or path
            model_type: Model type ('auto', 'causal_lm', 'sequence_classification')
            **kwargs: Additional arguments for model loading
        
        Returns:
            Model instance (on CUDA)
        """
        cache_key = f"{model_name}_{model_type}"
        if cache_key in self._models:
            return self._models[cache_key]
        
        # Load model based on type
        if model_type == 'causal_lm':
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        elif model_type == 'sequence_classification':
            model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
        else:
            model = AutoModel.from_pretrained(model_name, **kwargs)
        
        # Move to CUDA
        model = model.to(self.cuda_device)
        model.eval()  # Set to evaluation mode
        
        self._models[cache_key] = model
        return model
    
    def tokenize(
        self,
        text: Union[str, List[str]],
        tokenizer: Union[str, PreTrainedTokenizer],
        return_tensors: str = 'pt',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Tokenize text using HuggingFace tokenizer.
        
        Args:
            text: Input text or list of texts
            tokenizer: Tokenizer instance or model name
            return_tensors: Return format ('pt' for PyTorch, 'np' for numpy)
            **kwargs: Additional tokenizer arguments
        
        Returns:
            Tokenized output
        """
        if isinstance(tokenizer, str):
            tokenizer = self.load_tokenizer(tokenizer)
        
        encoded = tokenizer(text, return_tensors=return_tensors, **kwargs)
        
        # Convert to numpy if requested
        if return_tensors == 'np':
            encoded = {
                k: v.numpy() if isinstance(v, torch.Tensor) else v
                for k, v in encoded.items()
            }
        
        return encoded
    
    def encode(
        self,
        text: Union[str, List[str]],
        model_name: str,
        tokenizer_name: Optional[str] = None,
        extract_layer: Optional[int] = None,
        pool_method: str = 'mean'
    ) -> np.ndarray:
        """
        Encode text to embeddings using a HuggingFace model.
        
        Args:
            text: Input text or list of texts
            model_name: Model name or path
            tokenizer_name: Optional tokenizer name (defaults to model_name)
            extract_layer: Optional layer index to extract (None = last layer)
            pool_method: Pooling method ('mean', 'cls', 'max')
        
        Returns:
            Embeddings as numpy array (ready for Vulkan operations)
        """
        # Load tokenizer and model
        tokenizer = self.load_tokenizer(tokenizer_name or model_name)
        model = self.load_model(model_name)
        
        # Tokenize
        encoded = self.tokenize(text, tokenizer, return_tensors='pt')
        
        # Move inputs to CUDA
        inputs = {k: v.to(self.cuda_device) for k, v in encoded.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
            # Extract embeddings
            if extract_layer is not None:
                hidden_states = outputs.hidden_states[extract_layer]
            else:
                # Use last hidden state
                if hasattr(outputs, 'last_hidden_state'):
                    hidden_states = outputs.last_hidden_state
                elif hasattr(outputs, 'hidden_states'):
                    hidden_states = outputs.hidden_states[-1]
                else:
                    # Fallback to pooler output if available
                    hidden_states = outputs.pooler_output.unsqueeze(1)
            
            # Pool embeddings
            if pool_method == 'mean':
                # Mean pooling (excluding padding)
                attention_mask = inputs.get('attention_mask', None)
                if attention_mask is not None:
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                else:
                    embeddings = hidden_states.mean(dim=1)
            elif pool_method == 'cls':
                # Use [CLS] token
                embeddings = hidden_states[:, 0, :]
            elif pool_method == 'max':
                # Max pooling
                embeddings = hidden_states.max(dim=1)[0]
            else:
                embeddings = hidden_states.mean(dim=1)
        
        # Convert to numpy for Vulkan
        return embeddings.cpu().numpy().astype(np.float32)
    
    def load_sentence_transformer(
        self,
        model_name: str,
        device: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Load a sentence-transformers model.
        
        Args:
            model_name: Model name (e.g., 'all-MiniLM-L6-v2')
            device: Device to use ('cuda', 'cpu', or None for auto)
            **kwargs: Additional arguments for SentenceTransformer
        
        Returns:
            SentenceTransformer model instance
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )
        
        # Check cache
        cache_key = f"{model_name}_{device}"
        if cache_key in self._sentence_models:
            return self._sentence_models[cache_key]
        
        # Determine device
        if device is None:
            # Auto-select: CUDA if available, otherwise CPU (works on AMD)
            if self.cuda_device is not None:
                device = 'cuda'
            else:
                device = 'cpu'
        
        # Load model
        model = SentenceTransformer(model_name, device=device, **kwargs)
        
        # Cache it
        self._sentence_models[cache_key] = model
        
        return model
    
    def encode_sentence_transformer(
        self,
        texts: Union[str, List[str]],
        model_name: str = 'all-MiniLM-L6-v2',
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        batch_size: int = 32,
        use_gpu: Optional[bool] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Encode text(s) using sentence-transformers with GPU support.
        
        On AMD systems: Uses CPU for model inference, then converts to Vulkan-compatible numpy.
        On CUDA systems: Uses CUDA for model inference, then converts to numpy.
        
        Args:
            texts: Input text or list of texts
            model_name: Sentence-transformer model name
            convert_to_numpy: Convert to numpy array (default True for Vulkan compatibility)
            normalize_embeddings: Normalize embeddings (default True)
            show_progress_bar: Show progress bar for batch processing
            batch_size: Batch size for encoding
            use_gpu: Force GPU usage (None = auto-detect)
            **kwargs: Additional arguments for SentenceTransformer.encode()
        
        Returns:
            Embeddings as numpy array (ready for Vulkan operations)
        
        Examples:
            >>> bridge = HuggingFaceBridge()
            >>> embeddings = bridge.encode_sentence_transformer("Hello, world!")
            >>> # Works on AMD (CPU) and NVIDIA (CUDA)
            >>> embeddings.shape  # (384,) for all-MiniLM-L6-v2
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )
        
        # Load model
        if use_gpu is None:
            # Auto-detect: use CUDA if available, otherwise CPU (AMD compatible)
            device = None  # Let load_sentence_transformer decide
        elif use_gpu:
            device = 'cuda' if self.cuda_device is not None else 'cpu'
        else:
            device = 'cpu'
        
        model = self.load_sentence_transformer(model_name, device=device)
        
        # Encode
        embeddings = model.encode(
            texts,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
            batch_size=batch_size,
            **kwargs
        )
        
        # Ensure numpy and float32 for Vulkan
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        return embeddings
    
    def encode_sentence_transformer_vulkan(
        self,
        texts: Union[str, List[str]],
        model_name: str = 'all-MiniLM-L6-v2',
        use_vulkan: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Encode text(s) using sentence-transformers on Vulkan GPU (AMD).
        
        This method runs the entire model on Vulkan GPU, not just post-processing.
        Extracts weights from sentence-transformers and runs inference on GPU.
        
        Args:
            texts: Input text or list of texts
            model_name: Sentence-transformer model name
            use_vulkan: Use Vulkan for full model inference (default: True)
            **kwargs: Additional arguments for encoding
        
        Returns:
            Embeddings as numpy array (ready for Vulkan operations)
        
        Examples:
            >>> bridge = HuggingFaceBridge()
            >>> embeddings = bridge.encode_sentence_transformer_vulkan("Hello, world!")
            >>> # Runs entirely on AMD GPU via Vulkan!
        """
        if use_vulkan:
            try:
                from .vulkan_sentence_transformer import VulkanSentenceTransformer
                
                # Create or get cached Vulkan model
                cache_key = f"vulkan_{model_name}"
                if cache_key not in self._sentence_models:
                    logger.info(f"Creating Vulkan sentence-transformer: {model_name}")
                    vulkan_model = VulkanSentenceTransformer(model_name)
                    self._sentence_models[cache_key] = vulkan_model
                else:
                    vulkan_model = self._sentence_models[cache_key]
                
                # Encode using Vulkan
                embeddings = vulkan_model.encode(texts, **kwargs)
                return embeddings
                
            except Exception as e:
                logger.warning(f"Vulkan sentence-transformer failed: {e}, falling back to CPU")
                # Fall back to regular encoding
                return self.encode_sentence_transformer(texts, model_name=model_name, **kwargs)
        else:
            # Use regular encoding (CPU or CUDA)
            return self.encode_sentence_transformer(texts, model_name=model_name, **kwargs)
    
    def encode_sentence_transformer_gpu(
        self,
        texts: Union[str, List[str]],
        model_name: str = 'all-MiniLM-L6-v2',
        use_vulkan_postprocessing: bool = True,
        use_vulkan_model: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Encode text(s) using sentence-transformers with GPU acceleration.
        
        This method:
        1. Uses Vulkan for full model inference on AMD (if use_vulkan_model=True)
        2. Uses CUDA for model inference on NVIDIA (if available)
        3. Falls back to CPU on AMD systems (if use_vulkan_model=False)
        4. Optionally uses Vulkan for post-processing (normalization, etc.)
        
        Args:
            texts: Input text or list of texts
            model_name: Sentence-transformer model name
            use_vulkan_postprocessing: Use Vulkan for normalization/post-processing
            use_vulkan_model: Use Vulkan for full model inference (AMD GPU)
            **kwargs: Additional arguments for encoding
        
        Returns:
            Embeddings as numpy array (ready for Vulkan operations)
        """
        # Try Vulkan model first if requested (for AMD GPUs)
        if use_vulkan_model:
            try:
                return self.encode_sentence_transformer_vulkan(
                    texts,
                    model_name=model_name,
                    **kwargs
                )
            except Exception as e:
                logger.warning(f"Vulkan model failed: {e}, falling back")
        
        # Encode with sentence-transformers (CPU or CUDA)
        embeddings = self.encode_sentence_transformer(
            texts,
            model_name=model_name,
            **kwargs
        )
        
        # Optional Vulkan post-processing (normalization, etc.)
        if use_vulkan_postprocessing:
            try:
                from grilly import functional
                # Normalize using Vulkan if available
                original_shape = embeddings.shape
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(1, -1)
                
                # L2 normalize using Vulkan
                embeddings = functional.embedding_normalize(embeddings)
                
                # Restore original shape
                if len(original_shape) == 1:
                    embeddings = embeddings[0]  # Back to 1D for single text
            except Exception:
                # Vulkan post-processing not available, use numpy normalization
                # (normalization is already done by sentence-transformers if normalize_embeddings=True)
                pass
        
        return embeddings
    
    def generate(
        self,
        text: Union[str, List[str]],
        model_name: str,
        tokenizer_name: Optional[str] = None,
        max_length: int = 512,
        **kwargs
    ) -> List[str]:
        """
        Generate text using a causal language model.
        
        Args:
            text: Input text or list of texts
            model_name: Model name or path
            tokenizer_name: Optional tokenizer name
            max_length: Maximum generation length
            **kwargs: Additional generation arguments
        
        Returns:
            Generated text
        """
        # Load tokenizer and model
        tokenizer = self.load_tokenizer(tokenizer_name or model_name)
        model = self.load_model(model_name, model_type='causal_lm')
        
        # Tokenize
        encoded = self.tokenize(text, tokenizer, return_tensors='pt')
        inputs = {k: v.to(self.cuda_device) for k, v in encoded.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                **kwargs
            )
        
        # Decode
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return generated_texts
    
    def classify(
        self,
        text: Union[str, List[str]],
        model_name: str,
        tokenizer_name: Optional[str] = None,
        return_probs: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Classify text using a sequence classification model.
        
        Args:
            text: Input text or list of texts
            model_name: Model name or path
            tokenizer_name: Optional tokenizer name
            return_probs: Whether to return probabilities
        
        Returns:
            Predictions (and optionally probabilities)
        """
        # Load tokenizer and model
        tokenizer = self.load_tokenizer(tokenizer_name or model_name)
        model = self.load_model(model_name, model_type='sequence_classification')
        
        # Tokenize
        encoded = self.tokenize(text, tokenizer, return_tensors='pt')
        inputs = {k: v.to(self.cuda_device) for k, v in encoded.items()}
        
        # Classify
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
        
        # Convert to numpy
        predictions_np = predictions.cpu().numpy()
        probs_np = probs.cpu().numpy().astype(np.float32)
        
        if return_probs:
            return predictions_np, probs_np
        return predictions_np
    
    def to_vulkan(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor to numpy for Vulkan operations"""
        return self.device_manager.to_vulkan(tensor)
    
    def to_cuda(self, array: np.ndarray, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Convert numpy array to PyTorch CUDA tensor"""
        return self.device_manager.to_cuda(array, dtype)


def get_huggingface_bridge(cuda_device: Optional[Union[str, int]] = None) -> HuggingFaceBridge:
    """
    Get or create HuggingFace bridge instance.
    
    Args:
        cuda_device: CUDA device specification
    
    Returns:
        HuggingFaceBridge instance
    """
    return HuggingFaceBridge(cuda_device)
