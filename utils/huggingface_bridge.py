"""
HuggingFace Bridge for GPU Compatibility

Provides a wrapper to run HuggingFace models (tokenizers, transformers) on CUDA
while using Vulkan for custom operations. Handles seamless tensor conversion
between PyTorch CUDA tensors and numpy arrays for Vulkan.

Also provides LoRA (Low-Rank Adaptation) support for efficient fine-tuning:
- Load pre-trained models with LoRA adapters
- Save/load LoRA adapters independently
- Apply LoRA to specific model layers
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from grilly.nn.lora import LoRAConfig, LoRAModel

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizer,
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

import logging

from .device_manager import get_device_manager

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

    def __init__(
        self,
        cuda_device: str | int | None = None,
        model_name: str | None = None,
        **_kwargs,
    ):
        """
        Initialize HuggingFace bridge.

        Args:
            cuda_device: CUDA device ('cuda:0', 'cuda:1', or device index)
            model_name: Optional default model identifier.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is required for HuggingFace bridge. Install with: pip install torch"
            )
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers is required. Install with: pip install transformers")

        self.device_manager = get_device_manager()

        # Set CUDA device (only if CUDA is available)
        try:
            torch = self.device_manager.torch
            if torch.cuda.is_available():
                if cuda_device is not None:
                    if isinstance(cuda_device, int):
                        self.device_manager.set_device("cuda", cuda_device)
                    else:
                        self.device_manager.set_device("cuda")
                        self.device_manager._cuda_device = torch.device(cuda_device)
                else:
                    self.device_manager.set_device("cuda")

                self.cuda_device = self.device_manager.get_cuda_device()
            else:
                # CUDA not available - use CPU/Vulkan fallback for AMD systems
                self.device_manager.set_device("cpu")
                self.cuda_device = None
        except (RuntimeError, AssertionError, AttributeError) as e:
            # CUDA not available - use CPU/Vulkan fallback
            if "CUDA" in str(e) or "not compiled" in str(e) or "is_available" in str(e):
                # For AMD/Vulkan-only systems, we can still use the bridge for tokenization
                # but model inference will need to be handled differently
                self.device_manager.set_device("cpu")
                self.cuda_device = None
            else:
                raise

        self.torch = self.device_manager.torch
        self.default_model_name = model_name

        # Cache for loaded models
        self._tokenizers: dict[str, PreTrainedTokenizer] = {}
        self._models: dict[str, PreTrainedModel] = {}
        self._sentence_models: dict[str, Any] = {}  # Cache for sentence-transformers models

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

    def load_model(self, model_name: str, model_type: str = "auto", **kwargs) -> PreTrainedModel:
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
        if model_type == "causal_lm":
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        elif model_type == "sequence_classification":
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
        text: str | list[str],
        tokenizer: str | PreTrainedTokenizer,
        return_tensors: str = "pt",
        **kwargs,
    ) -> dict[str, Any]:
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
        if return_tensors == "np":
            encoded = {
                k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in encoded.items()
            }

        return encoded

    def encode(
        self,
        text: str | list[str],
        model_name: str,
        tokenizer_name: str | None = None,
        extract_layer: int | None = None,
        pool_method: str = "mean",
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
        encoded = self.tokenize(text, tokenizer, return_tensors="pt")

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
                if hasattr(outputs, "last_hidden_state"):
                    hidden_states = outputs.last_hidden_state
                elif hasattr(outputs, "hidden_states"):
                    hidden_states = outputs.hidden_states[-1]
                else:
                    # Fallback to pooler output if available
                    hidden_states = outputs.pooler_output.unsqueeze(1)

            # Pool embeddings
            if pool_method == "mean":
                # Mean pooling (excluding padding)
                attention_mask = inputs.get("attention_mask", None)
                if attention_mask is not None:
                    mask_expanded = (
                        attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    )
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                else:
                    embeddings = hidden_states.mean(dim=1)
            elif pool_method == "cls":
                # Use [CLS] token
                embeddings = hidden_states[:, 0, :]
            elif pool_method == "max":
                # Max pooling
                embeddings = hidden_states.max(dim=1)[0]
            else:
                embeddings = hidden_states.mean(dim=1)

        # Convert to numpy for Vulkan
        return embeddings.cpu().numpy().astype(np.float32)

    def load_sentence_transformer(
        self, model_name: str, device: str | None = None, **kwargs
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
                device = "cuda"
            else:
                device = "cpu"

        # Load model
        model = SentenceTransformer(model_name, device=device, **kwargs)

        # Cache it
        self._sentence_models[cache_key] = model

        return model

    def encode_sentence_transformer(
        self,
        texts: str | list[str],
        model_name: str = "all-MiniLM-L6-v2",
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        batch_size: int = 32,
        use_gpu: bool | None = None,
        **kwargs,
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
            device = "cuda" if self.cuda_device is not None else "cpu"
        else:
            device = "cpu"

        model = self.load_sentence_transformer(model_name, device=device)

        # Encode
        embeddings = model.encode(
            texts,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
            batch_size=batch_size,
            **kwargs,
        )

        # Ensure numpy and float32 for Vulkan
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        return embeddings

    def encode_sentence_transformer_vulkan(
        self,
        texts: str | list[str],
        model_name: str = "all-MiniLM-L6-v2",
        use_vulkan: bool = True,
        **kwargs,
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
        texts: str | list[str],
        model_name: str = "all-MiniLM-L6-v2",
        use_vulkan_postprocessing: bool = True,
        use_vulkan_model: bool = False,
        **kwargs,
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
                    texts, model_name=model_name, **kwargs
                )
            except Exception as e:
                logger.warning(f"Vulkan model failed: {e}, falling back")

        # Encode with sentence-transformers (CPU or CUDA)
        embeddings = self.encode_sentence_transformer(texts, model_name=model_name, **kwargs)

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
        text: str | list[str],
        model_name: str,
        tokenizer_name: str | None = None,
        max_length: int = 512,
        **kwargs,
    ) -> list[str]:
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
        model = self.load_model(model_name, model_type="causal_lm")

        # Tokenize
        encoded = self.tokenize(text, tokenizer, return_tensors="pt")
        inputs = {k: v.to(self.cuda_device) for k, v in encoded.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=max_length, **kwargs)

        # Decode
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return generated_texts

    def classify(
        self,
        text: str | list[str],
        model_name: str,
        tokenizer_name: str | None = None,
        return_probs: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
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
        model = self.load_model(model_name, model_type="sequence_classification")

        # Tokenize
        encoded = self.tokenize(text, tokenizer, return_tensors="pt")
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

    def to_cuda(self, array: np.ndarray, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Convert numpy array to PyTorch CUDA tensor"""
        return self.device_manager.to_cuda(array, dtype)

    # ============== LoRA Support ==============

    def load_model_with_lora(
        self,
        model_name: str,
        lora_config: dict[str, Any] | None = None,
        lora_path: str | Path | None = None,
        model_type: str = "causal_lm",
        target_modules: list[str] | None = None,
        rank: int = 8,
        alpha: float = 16.0,
        **kwargs,
    ) -> tuple[Any, LoRAModel]:
        """
        Load a HuggingFace model with LoRA adapters for fine-tuning.

        This method:
        1. Loads the base HuggingFace model
        2. Extracts weights from target layers
        3. Creates Grilly LoRA adapters for those layers
        4. Optionally loads pre-trained LoRA weights

        Args:
            model_name: HuggingFace model name or path
            lora_config: Optional LoRAConfig dict (rank, alpha, target_modules, etc.)
            lora_path: Optional path to pre-trained LoRA weights
            model_type: Model type ('causal_lm', 'auto', 'sequence_classification')
            target_modules: List of module names to apply LoRA to
                Default for LLMs: ['q_proj', 'v_proj']
            rank: LoRA rank (default: 8)
            alpha: LoRA scaling factor (default: 16.0)
            **kwargs: Additional arguments for model loading

        Returns:
            Tuple of (base_model, lora_model):
            - base_model: HuggingFace model (frozen)
            - lora_model: Grilly LoRAModel with adapters

        Example:
            >>> bridge = HuggingFaceBridge()
            >>> model, lora = bridge.load_model_with_lora(
            ...     "meta-llama/Llama-3.2-3B-Instruct",
            ...     rank=8, alpha=16,
            ...     target_modules=['q_proj', 'v_proj']
            ... )
            >>> # Train with lora.parameters()
            >>> # Save with bridge.save_lora_adapters(lora, "path/to/lora")
        """
        from grilly.nn.lora import LoRAConfig, LoRAModel

        # Load base model
        base_model = self.load_model(model_name, model_type=model_type, **kwargs)

        # Freeze base model
        for param in base_model.parameters():
            param.requires_grad = False

        # Create LoRA config
        if lora_config is not None:
            config = LoRAConfig.from_dict(lora_config)
        else:
            if target_modules is None:
                # Default target modules for common architectures
                target_modules = self._get_default_target_modules(base_model)
            config = LoRAConfig(
                rank=rank,
                alpha=alpha,
                target_modules=target_modules,
            )

        # Create LoRA model
        lora_model = LoRAModel(config)

        # Find and wrap target modules
        self._apply_lora_to_model(base_model, lora_model, config)

        # Load pre-trained LoRA weights if provided
        if lora_path is not None:
            lora_path = Path(lora_path)
            if lora_path.exists():
                logger.info(f"Loading LoRA weights from {lora_path}")
                lora_model = LoRAModel.load_checkpoint(lora_path)

        # Print parameter summary
        lora_model.print_trainable_parameters()

        return base_model, lora_model

    def _get_default_target_modules(self, model: Any) -> list[str]:
        """Get default target modules for LoRA based on model architecture."""
        model_name = model.__class__.__name__.lower()

        # Common attention projection names
        if "llama" in model_name or "mistral" in model_name:
            return ["q_proj", "v_proj"]
        elif "gpt" in model_name or "gpt2" in model_name:
            return ["c_attn"]  # GPT-2 uses combined QKV
        elif "bert" in model_name:
            return ["query", "value"]
        elif "t5" in model_name:
            return ["q", "v"]
        else:
            # Default: try common projection names
            return ["q_proj", "v_proj", "query", "value"]

    def _apply_lora_to_model(
        self, model: Any, lora_model: LoRAModel, config: LoRAConfig
    ) -> None:
        """
        Apply LoRA adapters to target modules in the model.

        Extracts weights from PyTorch layers and creates corresponding
        Grilly LoRA layers.
        """

        for name, module in model.named_modules():
            # Check if this module should have LoRA
            module_name = name.split(".")[-1]
            if module_name in config.target_modules:
                # Check if it's a Linear layer
                if hasattr(module, "weight") and hasattr(module, "in_features"):
                    # Extract weight as float32 (NumPy does not support BF16 tensors directly).
                    weight_tensor = module.weight.data
                    if hasattr(self.torch, "bfloat16") and weight_tensor.dtype == self.torch.bfloat16:
                        weight_tensor = weight_tensor.to(dtype=self.torch.float32)
                    weight = weight_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
                    in_features = module.in_features
                    out_features = module.out_features

                    # Create LoRA layer
                    lora_model.add_lora_layer(
                        name=name,
                        in_features=in_features,
                        out_features=out_features,
                        base_weights=weight,
                    )

                    logger.debug(f"Added LoRA to {name}: {in_features} -> {out_features}")

    def save_lora_adapters(
        self,
        lora_model: LoRAModel,
        save_path: str | Path,
        model_name: str | None = None,
    ) -> None:
        """
        Save LoRA adapters to disk.

        Saves:
        - config.json: LoRA configuration
        - adapters.npz: LoRA A and B matrices
        - metadata.json: Layer information and model name

        Args:
            lora_model: LoRAModel instance
            save_path: Directory to save adapters
            model_name: Optional base model name for reference

        Example:
            >>> bridge.save_lora_adapters(lora, "my_lora_adapters")
            >>> # Later: bridge.load_lora_adapters("my_lora_adapters")
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save using LoRAModel's method
        lora_model.save_checkpoint(save_path)

        # Add model reference if provided
        if model_name is not None:
            meta_path = save_path / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    metadata = json.load(f)
            else:
                metadata = {}

            metadata["base_model"] = model_name

            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

        logger.info(f"LoRA adapters saved to {save_path}")

    def load_lora_adapters(
        self,
        load_path: str | Path,
    ) -> LoRAModel:
        """
        Load LoRA adapters from disk.

        Args:
            load_path: Directory containing saved adapters

        Returns:
            LoRAModel instance with loaded weights

        Example:
            >>> lora = bridge.load_lora_adapters("my_lora_adapters")
            >>> # Use lora layers for inference
        """
        from grilly.nn.lora import LoRAModel

        load_path = Path(load_path)

        if not load_path.exists():
            raise FileNotFoundError(f"LoRA checkpoint not found: {load_path}")

        lora_model = LoRAModel.load_checkpoint(load_path)
        logger.info(f"LoRA adapters loaded from {load_path}")

        return lora_model

    def extract_model_weights(
        self,
        model_name: str,
        layer_names: list[str] | None = None,
        model_type: str = "causal_lm",
    ) -> dict[str, np.ndarray]:
        """
        Extract weights from a HuggingFace model for Vulkan inference.

        Args:
            model_name: Model name or path
            layer_names: Optional list of layer names to extract
                        (None = extract all linear layers)
            model_type: Model type

        Returns:
            Dictionary mapping layer names to numpy weight arrays

        Example:
            >>> weights = bridge.extract_model_weights("meta-llama/Llama-3.2-3B-Instruct")
            >>> # Use weights with Vulkan compute backend
        """
        model = self.load_model(model_name, model_type=model_type)

        weights = {}

        for name, module in model.named_modules():
            # Filter by layer names if specified
            if layer_names is not None:
                if not any(ln in name for ln in layer_names):
                    continue

            # Extract linear layer weights
            if hasattr(module, "weight"):
                weight = module.weight.data.cpu().numpy().astype(np.float32)
                weights[name] = weight

                # Also extract bias if present
                if hasattr(module, "bias") and module.bias is not None:
                    bias = module.bias.data.cpu().numpy().astype(np.float32)
                    weights[f"{name}.bias"] = bias

        return weights

    def create_lora_from_weights(
        self,
        weights: dict[str, np.ndarray],
        target_modules: list[str],
        rank: int = 8,
        alpha: float = 16.0,
    ) -> LoRAModel:
        """
        Create LoRA model from extracted weights.

        This is useful when you want to create LoRA adapters without
        loading the full HuggingFace model (e.g., for Vulkan-only inference).

        Args:
            weights: Dictionary of layer weights (from extract_model_weights)
            target_modules: Layer name patterns to apply LoRA to
            rank: LoRA rank
            alpha: LoRA scaling factor

        Returns:
            LoRAModel with adapters for target layers

        Example:
            >>> weights = bridge.extract_model_weights("model_name")
            >>> lora = bridge.create_lora_from_weights(
            ...     weights,
            ...     target_modules=['q_proj', 'v_proj'],
            ...     rank=8
            ... )
        """
        from grilly.nn.lora import LoRAConfig, LoRAModel

        config = LoRAConfig(
            rank=rank,
            alpha=alpha,
            target_modules=target_modules,
        )

        lora_model = LoRAModel(config)

        for name, weight in weights.items():
            # Skip bias weights
            if name.endswith(".bias"):
                continue

            # Check if this layer should have LoRA
            layer_name = name.split(".")[-1]
            if any(tm in layer_name for tm in target_modules):
                if len(weight.shape) == 2:
                    out_features, in_features = weight.shape
                    lora_model.add_lora_layer(
                        name=name,
                        in_features=in_features,
                        out_features=out_features,
                        base_weights=weight,
                    )

        return lora_model

    def merge_lora_to_model(
        self,
        model: Any,
        lora_model: LoRAModel,
    ) -> Any:
        """
        Merge LoRA weights into the base model.

        After merging, the model can be used without LoRA overhead.
        Note: This modifies the model in-place.

        Args:
            model: HuggingFace model
            lora_model: LoRAModel with trained adapters

        Returns:
            Model with merged weights
        """
        for name, module in model.named_modules():
            if name in lora_model.lora_layers:
                lora_layer = lora_model.lora_layers[name]

                # Merge LoRA into base weights
                lora_layer.merge_weights()

                # Update PyTorch module weight
                if hasattr(module, "weight"):
                    merged_weight = self.torch.from_numpy(lora_layer.W.data)
                    module.weight.data = merged_weight.to(module.weight.device)

        return model


def get_huggingface_bridge(cuda_device: str | int | None = None) -> HuggingFaceBridge:
    """
    Get or create HuggingFace bridge instance.

    Args:
        cuda_device: CUDA device specification

    Returns:
        HuggingFaceBridge instance
    """
    return HuggingFaceBridge(cuda_device)
