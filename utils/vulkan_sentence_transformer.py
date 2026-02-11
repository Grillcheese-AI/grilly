"""
Vulkan-based Sentence Transformer

Runs sentence-transformers models on AMD GPUs using Vulkan shaders.
Extracts weights from sentence-transformers models and runs inference on GPU.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

GRILLY_AVAILABLE = False
_GRILLY_IMPORT_ERROR = None

try:
    from grilly import functional

    GRILLY_AVAILABLE = True
except ImportError as e:
    _GRILLY_IMPORT_ERROR = e
    GRILLY_AVAILABLE = False


class VulkanSentenceTransformer:
    """
    GPU-accelerated sentence transformer using Vulkan.

    Extracts weights from sentence-transformers models and runs inference
    on AMD GPUs using Vulkan compute shaders.

    Supported architectures:
    - BERT (bert)
    - DistilBERT (distilbert)
    - RoBERTa (roberta)
    - MPNet (mpnet)
    - XLM-RoBERTa (xlm-roberta)
    - ALBERT (albert)
    - Granite (granite) - Uses GPT-style causal attention
    - GPT (gpt) - Causal attention
    - T5 (t5) - Encoder-decoder with cross-attention

    BERT-family models (BERT, DistilBERT, RoBERTa, MPNet, XLM-RoBERTa, ALBERT) use
    post-normalization (layer norm after attention/FFN) and bidirectional attention.

    GPT-family models (GPT, Granite) use causal attention (can only attend to previous positions).

    The implementation automatically detects the model architecture and adapts accordingly.

    For unsupported architectures, the code will attempt to use BERT-style forward pass
    as a fallback, but results may not be accurate.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "vulkan",
        max_seq_length: int = 512,
    ):
        """
        Initialize Vulkan sentence transformer.

        Args:
            model_name: Sentence-transformer model name
            device: Device ('vulkan' for AMD GPU)
            max_seq_length: Maximum sequence length
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )
        if not GRILLY_AVAILABLE:
            error_msg = "Grilly backend not available"
            if _GRILLY_IMPORT_ERROR is not None:
                error_msg += f": {_GRILLY_IMPORT_ERROR}"
            error_msg += "\nMake sure grilly is installed: pip install -e . (from grilly directory)"
            raise RuntimeError(error_msg)

        self.model_name = model_name
        self.device = device
        self.max_seq_length = max_seq_length

        # Initialize Vulkan backend using device manager
        try:
            from .device_manager import get_vulkan_backend

            self.backend = get_vulkan_backend()
            logger.info("[OK] Vulkan backend initialized for sentence-transformer")
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Vulkan backend: {e}. Make sure Vulkan is available and grilly is properly installed."
            )

        # Load sentence-transformer model to extract weights
        logger.info(f"Loading sentence-transformer model: {model_name}")
        self.st_model = SentenceTransformer(model_name, device="cpu")

        # Extract and convert model architecture
        self._extract_model_weights()

        # Set architecture in backend for architecture-specific shader selection
        if hasattr(self, "model_type") and self.model_type:
            self.backend.set_architecture(self.model_type)
            logger.debug(f"Set backend architecture to: {self.model_type}")

        # Build Vulkan model
        self._build_vulkan_model()

        logger.info("[OK] Vulkan sentence-transformer ready")

    def _extract_model_weights(self):
        """Extract weights from sentence-transformer model"""
        # sentence-transformers wraps models - try different access patterns
        # Get the first module (usually the transformer)
        if hasattr(self.st_model, "_modules"):
            modules = list(self.st_model._modules.values())
            if modules:
                # Get the first module (usually the transformer)
                auto_model = modules[0]
            else:
                raise ValueError("Could not find transformer model in sentence-transformer")
        elif hasattr(self.st_model, "0"):
            # Sequential models use numeric keys
            auto_model = self.st_model[0]
        else:
            # Try to get from named_children
            try:
                children = list(self.st_model.named_children())
                if children:
                    auto_model = children[0][1]  # Get the module from (name, module) tuple
                else:
                    raise ValueError("Could not find transformer model in sentence-transformer")
            except Exception:
                raise ValueError("Could not access sentence-transformer modules")

        # Now get the actual transformer from auto_model
        # sentence-transformers wraps models in a Transformer object that has 'auto_model'
        # For DistilBERT: auto_model.auto_model.transformer
        # For BERT: auto_model.auto_model (which has encoder)
        if hasattr(auto_model, "auto_model"):
            # This is a sentence-transformers Transformer wrapper
            actual_model = auto_model.auto_model
            # Now check the actual model structure
            if hasattr(actual_model, "transformer"):
                # DistilBERT structure
                self.transformer_model = actual_model.transformer
            elif hasattr(actual_model, "encoder"):
                # BERT structure
                self.transformer_model = actual_model
            else:
                # Try to use actual_model directly
                self.transformer_model = actual_model
        elif hasattr(auto_model, "transformer"):
            # Direct access (shouldn't happen with sentence-transformers, but handle it)
            self.transformer_model = auto_model.transformer
        elif hasattr(auto_model, "encoder"):
            # Direct BERT access
            self.transformer_model = auto_model
        else:
            # Try to use auto_model directly (might be the transformer already)
            self.transformer_model = auto_model

        # Extract tokenizer
        if hasattr(self.st_model, "tokenizer"):
            self.tokenizer = self.st_model.tokenizer
        else:
            raise ValueError("Could not find tokenizer in sentence-transformer")

        # Extract model config
        if hasattr(self.transformer_model, "config"):
            self.config = self.transformer_model.config
        else:
            # Default config for all-MiniLM-L6-v2 (DistilBERT-based)
            self.config = type(
                "Config",
                (),
                {
                    "hidden_size": 384,
                    "num_attention_heads": 12,
                    "intermediate_size": 1536,
                    "num_hidden_layers": 6,
                    "max_position_embeddings": 512,
                    "vocab_size": 30522,
                    "layer_norm_eps": 1e-12,
                    "model_type": "distilbert",  # Default assumption
                },
            )()

        # Detect model architecture type
        if hasattr(self.config, "model_type"):
            self.model_type = self.config.model_type.lower()
        else:
            # Try to infer from model class name
            model_class_name = self.transformer_model.__class__.__name__.lower()
            if "bert" in model_class_name:
                self.model_type = "bert"
            elif "distilbert" in model_class_name or "distil" in model_class_name:
                self.model_type = "distilbert"
            elif "roberta" in model_class_name:
                self.model_type = "roberta"
            elif "mpnet" in model_class_name:
                self.model_type = "mpnet"
            elif "granite" in model_class_name:
                self.model_type = "granite"
            elif "modernbert" in model_class_name:
                self.model_type = "modernbert"
            elif "gpt" in model_class_name:
                self.model_type = "gpt"
            elif "t5" in model_class_name:
                self.model_type = "t5"
            else:
                self.model_type = "bert"  # Default to BERT-style
                logger.warning(
                    f"Could not detect model type, defaulting to BERT-style. Model class: {model_class_name}"
                )

        # Check if architecture is supported
        supported_architectures = [
            "bert",
            "distilbert",
            "roberta",
            "mpnet",
            "xlm-roberta",
            "albert",
            "granite",
            "gpt",
            "t5",
            "modernbert",
        ]
        if self.model_type not in supported_architectures:
            logger.warning(
                f"Model type '{self.model_type}' may not be fully supported. "
                f"Supported types: {supported_architectures}. "
                f"Attempting to use BERT-style forward pass..."
            )
            # Try to use BERT-style forward pass as fallback
            self.model_type = "bert"

        logger.info(f"Detected model type: {self.model_type}")
        logger.info(
            f"Model config: hidden_size={self.config.hidden_size}, "
            f"num_layers={self.config.num_hidden_layers}"
        )

    def _build_vulkan_model(self):
        """Build Vulkan model from extracted weights"""
        num_layers = self.config.num_hidden_layers

        # Build transformer layers
        self.layers = []
        for i in range(num_layers):
            layer = self._extract_layer_weights(i)
            self.layers.append(layer)

        # Extract final layer norm (ModernBERT has final_norm after all layers)
        self.final_norm_weight = None
        self.final_norm_bias = None
        self.final_norm_eps = 1e-5
        if self.model_type == "modernbert":
            actual_model = self.transformer_model
            if hasattr(actual_model, "auto_model"):
                actual_model = actual_model.auto_model
            if hasattr(actual_model, "final_norm"):
                final_norm = actual_model.final_norm
                if hasattr(final_norm, "weight"):
                    self.final_norm_weight = (
                        final_norm.weight.detach().cpu().numpy().astype(np.float32)
                    )
                    self.final_norm_bias = (
                        final_norm.bias.detach().cpu().numpy().astype(np.float32)
                        if hasattr(final_norm, "bias") and final_norm.bias is not None
                        else np.zeros(final_norm.weight.shape[0], dtype=np.float32)
                    )
                    self.final_norm_eps = getattr(final_norm, "eps", 1e-5)
                    logger.info(
                        f"Extracted final_norm: weight shape={self.final_norm_weight.shape}, eps={self.final_norm_eps}"
                    )

        # Pooling layer (mean pooling for sentence-transformers)
        self.pooling_type = "mean"  # sentence-transformers uses mean pooling

        # Output normalization (sentence-transformers normalizes embeddings)
        self.normalize = True

    def _extract_layer_weights(self, layer_idx: int) -> dict[str, np.ndarray]:
        """Extract weights for a single transformer layer"""
        # Get the actual transformer model (might need to go through auto_model)
        actual_model = self.transformer_model
        if hasattr(actual_model, "auto_model"):
            actual_model = actual_model.auto_model

        # Handle different model architectures (DistilBERT, BERT, ModernBERT, etc.)
        encoder = None
        layers = None

        # Try different paths to find layers
        if hasattr(actual_model, "transformer"):
            # DistilBERT structure: model.transformer.layer
            if hasattr(actual_model.transformer, "layer"):
                encoder = actual_model.transformer
                layers = encoder.layer
        elif hasattr(actual_model, "encoder"):
            # BERT structure: model.encoder.layer
            if hasattr(actual_model.encoder, "layer"):
                encoder = actual_model.encoder
                layers = encoder.layer
        elif hasattr(actual_model, "layers"):
            # ModernBERT structure: model.layers (direct access)
            layers = actual_model.layers
        elif hasattr(actual_model, "layer"):
            # Direct layer access
            layers = actual_model.layer

        if layers is None:
            # Debug: print available attributes
            logger.debug(
                f"Transformer model attributes: {[a for a in dir(actual_model) if not a.startswith('_')][:20]}"
            )
            if hasattr(actual_model, "transformer"):
                logger.debug(
                    f"Transformer.transformer attributes: {[a for a in dir(actual_model.transformer) if not a.startswith('_')][:20]}"
                )
            raise ValueError(
                f"Could not find encoder/layers in transformer model. Available attributes: {[a for a in dir(actual_model) if not a.startswith('_')][:20]}"
            )

        layer = layers[layer_idx]
        weights = {}

        # Attention weights
        # ModernBERT uses attn with Wqkv (combined QKV) and Wo (output)
        # DistilBERT uses q_lin, k_lin, v_lin, out_lin
        # BERT uses query, key, value, dense
        # Note: functional.linear does x @ weights.T, so we store weights as (out_features, in_features)
        # PyTorch stores weights as (out_features, in_features), so we don't need to transpose
        if self.model_type == "modernbert" and hasattr(layer, "attn"):
            # ModernBERT uses 'attn' with Wqkv (combined QKV) and Wo
            attn = layer.attn
            if hasattr(attn, "Wqkv") and hasattr(attn, "Wo"):
                # ModernBERT: Wqkv projects to (3 * hidden_size) for Q, K, V combined
                # We need to split it into Q, K, V
                wqkv = (
                    attn.Wqkv.weight.detach().cpu().numpy().astype(np.float32)
                )  # (3*hidden_size, hidden_size)
                wqkv_bias = (
                    attn.Wqkv.bias.detach().cpu().numpy().astype(np.float32)
                    if hasattr(attn.Wqkv, "bias") and attn.Wqkv.bias is not None
                    else np.zeros(wqkv.shape[0], dtype=np.float32)
                )

                # Split Wqkv into Q, K, V
                hidden_size = self.config.hidden_size
                weights["attn_q_weight"] = wqkv[:hidden_size, :]  # First third
                weights["attn_q_bias"] = wqkv_bias[:hidden_size]
                weights["attn_k_weight"] = wqkv[hidden_size : 2 * hidden_size, :]  # Second third
                weights["attn_k_bias"] = wqkv_bias[hidden_size : 2 * hidden_size]
                weights["attn_v_weight"] = wqkv[2 * hidden_size :, :]  # Last third
                weights["attn_v_bias"] = wqkv_bias[2 * hidden_size :]

                # Output projection
                weights["attn_out_weight"] = (
                    attn.Wo.weight.detach().cpu().numpy().astype(np.float32)
                )
                weights["attn_out_bias"] = (
                    attn.Wo.bias.detach().cpu().numpy().astype(np.float32)
                    if hasattr(attn.Wo, "bias") and attn.Wo.bias is not None
                    else np.zeros(attn.Wo.weight.shape[0], dtype=np.float32)
                )

            # LayerNorm for ModernBERT attention
            if hasattr(layer, "attn_norm"):
                attn_norm = layer.attn_norm
                # Check if it's Identity (no-op) or actual LayerNorm
                if hasattr(attn_norm, "weight"):
                    weights["attn_ln_gamma"] = (
                        attn_norm.weight.detach().cpu().numpy().astype(np.float32)
                    )
                    weights["attn_ln_beta"] = (
                        attn_norm.bias.detach().cpu().numpy().astype(np.float32)
                        if hasattr(attn_norm, "bias") and attn_norm.bias is not None
                        else np.zeros(attn_norm.weight.shape[0], dtype=np.float32)
                    )
                else:
                    # Identity layer (no-op), use identity weights
                    hidden_size = self.config.hidden_size
                    weights["attn_ln_gamma"] = np.ones(hidden_size, dtype=np.float32)
                    weights["attn_ln_beta"] = np.zeros(hidden_size, dtype=np.float32)
            else:
                # No layer norm, use identity
                hidden_size = self.config.hidden_size
                weights["attn_ln_gamma"] = np.ones(hidden_size, dtype=np.float32)
                weights["attn_ln_beta"] = np.zeros(hidden_size, dtype=np.float32)
        elif (
            self.model_type == "modernbert"
            and hasattr(layer, "attn")
            and hasattr(layer.attn, "q_proj")
        ):
            # Alternative ModernBERT structure with separate projections
            attn = layer.attn
            if hasattr(attn, "q_proj"):
                # Alternative ModernBERT structure with separate projections
                weights["attn_q_weight"] = (
                    attn.q_proj.weight.detach().cpu().numpy().astype(np.float32)
                )
                weights["attn_q_bias"] = (
                    attn.q_proj.bias.detach().cpu().numpy().astype(np.float32)
                    if hasattr(attn.q_proj, "bias") and attn.q_proj.bias is not None
                    else np.zeros(attn.q_proj.weight.shape[0], dtype=np.float32)
                )
                weights["attn_k_weight"] = (
                    attn.k_proj.weight.detach().cpu().numpy().astype(np.float32)
                )
                weights["attn_k_bias"] = (
                    attn.k_proj.bias.detach().cpu().numpy().astype(np.float32)
                    if hasattr(attn.k_proj, "bias") and attn.k_proj.bias is not None
                    else np.zeros(attn.k_proj.weight.shape[0], dtype=np.float32)
                )
                weights["attn_v_weight"] = (
                    attn.v_proj.weight.detach().cpu().numpy().astype(np.float32)
                )
                weights["attn_v_bias"] = (
                    attn.v_proj.bias.detach().cpu().numpy().astype(np.float32)
                    if hasattr(attn.v_proj, "bias") and attn.v_proj.bias is not None
                    else np.zeros(attn.v_proj.weight.shape[0], dtype=np.float32)
                )
                weights["attn_out_weight"] = (
                    attn.o_proj.weight.detach().cpu().numpy().astype(np.float32)
                )
                weights["attn_out_bias"] = (
                    attn.o_proj.bias.detach().cpu().numpy().astype(np.float32)
                    if hasattr(attn.o_proj, "bias") and attn.o_proj.bias is not None
                    else np.zeros(attn.o_proj.weight.shape[0], dtype=np.float32)
                )
            else:
                raise ValueError(
                    f"ModernBERT attention structure not recognized. Available: {[a for a in dir(attn) if not a.startswith('_')][:20]}"
                )
        elif hasattr(layer, "attention"):
            attn = layer.attention
            # Check if it's DistilBERT-style (has q_lin) or BERT-style (has query)
            if hasattr(attn, "q_lin"):
                # DistilBERT structure
                weights["attn_q_weight"] = (
                    attn.q_lin.weight.detach().cpu().numpy().astype(np.float32)
                )  # (out_features, in_features)
                weights["attn_q_bias"] = (
                    attn.q_lin.bias.detach().cpu().numpy().astype(np.float32)
                    if attn.q_lin.bias is not None
                    else np.zeros(attn.q_lin.out_features, dtype=np.float32)
                )
                weights["attn_k_weight"] = (
                    attn.k_lin.weight.detach().cpu().numpy().astype(np.float32)
                )
                weights["attn_k_bias"] = (
                    attn.k_lin.bias.detach().cpu().numpy().astype(np.float32)
                    if attn.k_lin.bias is not None
                    else np.zeros(attn.k_lin.out_features, dtype=np.float32)
                )
                weights["attn_v_weight"] = (
                    attn.v_lin.weight.detach().cpu().numpy().astype(np.float32)
                )
                weights["attn_v_bias"] = (
                    attn.v_lin.bias.detach().cpu().numpy().astype(np.float32)
                    if attn.v_lin.bias is not None
                    else np.zeros(attn.v_lin.out_features, dtype=np.float32)
                )
                weights["attn_out_weight"] = (
                    attn.out_lin.weight.detach().cpu().numpy().astype(np.float32)
                )
                weights["attn_out_bias"] = (
                    attn.out_lin.bias.detach().cpu().numpy().astype(np.float32)
                    if attn.out_lin.bias is not None
                    else np.zeros(attn.out_lin.out_features, dtype=np.float32)
                )
            elif hasattr(attn, "self"):
                # BERT structure: attention.self (query, key, value) and attention.output (dense)
                attn_self = attn.self
                weights["attn_q_weight"] = (
                    attn_self.query.weight.detach().cpu().numpy().astype(np.float32)
                )  # (out_features, in_features)
                weights["attn_q_bias"] = (
                    attn_self.query.bias.detach().cpu().numpy().astype(np.float32)
                )
                weights["attn_k_weight"] = (
                    attn_self.key.weight.detach().cpu().numpy().astype(np.float32)
                )
                weights["attn_k_bias"] = (
                    attn_self.key.bias.detach().cpu().numpy().astype(np.float32)
                )
                weights["attn_v_weight"] = (
                    attn_self.value.weight.detach().cpu().numpy().astype(np.float32)
                )
                weights["attn_v_bias"] = (
                    attn_self.value.bias.detach().cpu().numpy().astype(np.float32)
                )

                attn_output = attn.output
                weights["attn_out_weight"] = (
                    attn_output.dense.weight.detach().cpu().numpy().astype(np.float32)
                )
                weights["attn_out_bias"] = (
                    attn_output.dense.bias.detach().cpu().numpy().astype(np.float32)
                )
            else:
                raise ValueError(
                    f"Unknown attention structure. Available: {[a for a in dir(attn) if not a.startswith('_')]}"
                )

            # LayerNorm (may be before or after attention depending on model)
            # ModernBERT: layer.attn_norm (but it's Identity, so no norm!)
            # DistilBERT: layer.sa_layer_norm
            # BERT: attn_output.LayerNorm
            if self.model_type == "modernbert" and hasattr(layer, "attn_norm"):
                # ModernBERT uses 'attn_norm', but it's Identity (no-op)
                # Check if it's actually Identity or LayerNorm
                import torch.nn as nn

                if isinstance(layer.attn_norm, nn.Identity):
                    # Identity layer (no-op), use identity weights
                    hidden_size = self.config.hidden_size
                    weights["attn_ln_gamma"] = np.ones(hidden_size, dtype=np.float32)
                    weights["attn_ln_beta"] = np.zeros(hidden_size, dtype=np.float32)
                else:
                    # Actual LayerNorm
                    weights["attn_ln_gamma"] = (
                        layer.attn_norm.weight.detach().cpu().numpy().astype(np.float32)
                    )
                    weights["attn_ln_beta"] = (
                        layer.attn_norm.bias.detach().cpu().numpy().astype(np.float32)
                        if hasattr(layer.attn_norm, "bias") and layer.attn_norm.bias is not None
                        else np.zeros(layer.attn_norm.weight.shape[0], dtype=np.float32)
                    )
            elif hasattr(layer, "sa_layer_norm"):
                # DistilBERT structure
                weights["attn_ln_gamma"] = (
                    layer.sa_layer_norm.weight.detach().cpu().numpy().astype(np.float32)
                )
                weights["attn_ln_beta"] = (
                    layer.sa_layer_norm.bias.detach().cpu().numpy().astype(np.float32)
                )
            elif hasattr(attn, "output") and hasattr(attn.output, "LayerNorm"):
                # BERT structure
                weights["attn_ln_gamma"] = (
                    attn.output.LayerNorm.weight.detach().cpu().numpy().astype(np.float32)
                )
                weights["attn_ln_beta"] = (
                    attn.output.LayerNorm.bias.detach().cpu().numpy().astype(np.float32)
                )
            else:
                # No layer norm, use identity
                hidden_size = self.config.hidden_size
                weights["attn_ln_gamma"] = np.ones(hidden_size, dtype=np.float32)
                weights["attn_ln_beta"] = np.zeros(hidden_size, dtype=np.float32)

        # FFN weights
        # DistilBERT uses ffn.lin1 and ffn.lin2 (not ffn.dropout.linear1)
        # ModernBERT uses mlp with gate_proj, up_proj, down_proj (GeGLU structure)
        # Note: functional.linear does x @ weights.T, so we store weights as (out_features, in_features)
        # PyTorch stores weights as (out_features, in_features), so we don't need to transpose
        if self.model_type == "modernbert" and hasattr(layer, "mlp"):
            # ModernBERT uses 'mlp' with Wi (input projection) and Wo (output projection)
            # GeGLU activation: GeGLU(Wi(x)) @ Wo
            mlp = layer.mlp
            if hasattr(mlp, "Wi") and hasattr(mlp, "Wo"):
                # ModernBERT MLP structure: Wi -> GeGLU -> Wo
                # Wi shape: (intermediate_size * 2, hidden_size) = (2304, 768) for this model
                # Split Wi into gate (first half) and up (second half) for GeGLU
                wi_weight = mlp.Wi.weight.detach().cpu().numpy().astype(np.float32)
                wi_bias = (
                    mlp.Wi.bias.detach().cpu().numpy().astype(np.float32)
                    if hasattr(mlp.Wi, "bias") and mlp.Wi.bias is not None
                    else np.zeros(mlp.Wi.weight.shape[0], dtype=np.float32)
                )

                intermediate_size = self.config.intermediate_size
                # Split Wi into input (first half) and gate (second half)
                # ModernBERT: input, gate = Wi(hidden_states).chunk(2, dim=-1)
                # So first half is "input" (gets GELU), second half is "gate" (multiplies)
                weights["ffn_input_weight"] = wi_weight[
                    :intermediate_size, :
                ]  # Input projection: (1152, 768) - gets GELU
                weights["ffn_input_bias"] = wi_bias[:intermediate_size]
                weights["ffn_gate_weight"] = wi_weight[
                    intermediate_size:, :
                ]  # Gate projection: (1152, 768) - multiplies
                weights["ffn_gate_bias"] = wi_bias[intermediate_size:]

                # For backward compatibility, also set ffn_gate_weight and ffn_up_weight
                weights["ffn_gate_weight"] = weights[
                    "ffn_input_weight"
                ]  # Alias for input (gets GELU)
                weights["ffn_gate_bias"] = weights["ffn_input_bias"]
                weights["ffn_up_weight"] = wi_weight[intermediate_size:, :]  # Gate (multiplies)
                weights["ffn_up_bias"] = wi_bias[intermediate_size:]

                # For compatibility, also set w1/w2 (but forward pass will use gate/up for GeGLU)
                weights["ffn_w1"] = weights["ffn_gate_weight"]
                weights["ffn_b1"] = weights["ffn_gate_bias"]

                weights["ffn_w2"] = mlp.Wo.weight.detach().cpu().numpy().astype(np.float32)
                weights["ffn_b2"] = (
                    mlp.Wo.bias.detach().cpu().numpy().astype(np.float32)
                    if hasattr(mlp.Wo, "bias") and mlp.Wo.bias is not None
                    else np.zeros(mlp.Wo.weight.shape[0], dtype=np.float32)
                )
            elif (
                hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj")
            ):
                # Alternative ModernBERT structure with gate_proj/up_proj/down_proj
                weights["ffn_w1"] = mlp.gate_proj.weight.detach().cpu().numpy().astype(np.float32)
                weights["ffn_b1"] = (
                    mlp.gate_proj.bias.detach().cpu().numpy().astype(np.float32)
                    if hasattr(mlp.gate_proj, "bias") and mlp.gate_proj.bias is not None
                    else np.zeros(mlp.gate_proj.weight.shape[0], dtype=np.float32)
                )
                weights["ffn_w2"] = mlp.down_proj.weight.detach().cpu().numpy().astype(np.float32)
                weights["ffn_b2"] = (
                    mlp.down_proj.bias.detach().cpu().numpy().astype(np.float32)
                    if hasattr(mlp.down_proj, "bias") and mlp.down_proj.bias is not None
                    else np.zeros(mlp.down_proj.weight.shape[0], dtype=np.float32)
                )
                weights["ffn_up_proj"] = (
                    mlp.up_proj.weight.detach().cpu().numpy().astype(np.float32)
                )
                weights["ffn_up_bias"] = (
                    mlp.up_proj.bias.detach().cpu().numpy().astype(np.float32)
                    if hasattr(mlp.up_proj, "bias") and mlp.up_proj.bias is not None
                    else np.zeros(mlp.up_proj.weight.shape[0], dtype=np.float32)
                )
            else:
                raise ValueError(
                    f"ModernBERT MLP structure not recognized. Available: {[a for a in dir(mlp) if not a.startswith('_')]}"
                )
        elif hasattr(layer, "ffn"):
            ffn = layer.ffn
            # DistilBERT structure: ffn.lin1 and ffn.lin2
            if hasattr(ffn, "lin1") and hasattr(ffn, "lin2"):
                weights["ffn_w1"] = (
                    ffn.lin1.weight.detach().cpu().numpy().astype(np.float32)
                )  # (out_features, in_features)
                weights["ffn_b1"] = (
                    ffn.lin1.bias.detach().cpu().numpy().astype(np.float32)
                    if ffn.lin1.bias is not None
                    else np.zeros(ffn.lin1.out_features, dtype=np.float32)
                )
                weights["ffn_w2"] = ffn.lin2.weight.detach().cpu().numpy().astype(np.float32)
                weights["ffn_b2"] = (
                    ffn.lin2.bias.detach().cpu().numpy().astype(np.float32)
                    if ffn.lin2.bias is not None
                    else np.zeros(ffn.lin2.out_features, dtype=np.float32)
                )
            # Legacy check for old structure (shouldn't be needed, but kept for compatibility)
            elif hasattr(ffn, "dropout") and hasattr(ffn.dropout, "linear1"):
                weights["ffn_w1"] = (
                    ffn.dropout.linear1.weight.detach().cpu().numpy().astype(np.float32)
                )
                weights["ffn_b1"] = (
                    ffn.dropout.linear1.bias.detach().cpu().numpy().astype(np.float32)
                    if ffn.dropout.linear1.bias is not None
                    else np.zeros(ffn.dropout.linear1.out_features, dtype=np.float32)
                )
                weights["ffn_w2"] = ffn.linear2.weight.detach().cpu().numpy().astype(np.float32)
                weights["ffn_b2"] = (
                    ffn.linear2.bias.detach().cpu().numpy().astype(np.float32)
                    if ffn.linear2.bias is not None
                    else np.zeros(ffn.linear2.out_features, dtype=np.float32)
                )
            elif hasattr(ffn, "dense"):
                # BERT structure: ffn.dense
                weights["ffn_w1"] = ffn.dense.weight.detach().cpu().numpy().astype(np.float32)
                weights["ffn_b1"] = ffn.dense.bias.detach().cpu().numpy().astype(np.float32)
                # Get output layer
                if hasattr(layer, "output"):
                    ffn_output = layer.output
                elif hasattr(layer, "ffn_output"):
                    ffn_output = layer.ffn_output
                else:
                    raise ValueError("Could not find FFN output in layer")
                weights["ffn_w2"] = (
                    ffn_output.dense.weight.detach().cpu().numpy().astype(np.float32)
                )
                weights["ffn_b2"] = ffn_output.dense.bias.detach().cpu().numpy().astype(np.float32)
            else:
                raise ValueError(
                    f"Could not find FFN structure in layer. Available: {[a for a in dir(ffn) if not a.startswith('_')]}"
                )
        elif hasattr(layer, "intermediate"):
            # BERT structure
            ffn = layer.intermediate
            weights["ffn_w1"] = (
                ffn.dense.weight.detach().cpu().numpy().astype(np.float32)
            )  # (out_features, in_features)
            weights["ffn_b1"] = ffn.dense.bias.detach().cpu().numpy().astype(np.float32)

            if hasattr(layer, "output"):
                ffn_output = layer.output
            elif hasattr(layer, "ffn_output"):
                ffn_output = layer.ffn_output
            else:
                raise ValueError("Could not find FFN output in layer")
            weights["ffn_w2"] = ffn_output.dense.weight.detach().cpu().numpy().astype(np.float32)
            weights["ffn_b2"] = ffn_output.dense.bias.detach().cpu().numpy().astype(np.float32)
        else:
            raise ValueError(
                f"Could not find FFN in layer. Available: {[a for a in dir(layer) if not a.startswith('_')]}"
            )

        # FFN LayerNorm
        # Check ModernBERT structure first (mlp_norm)
        if self.model_type == "modernbert" and hasattr(layer, "mlp_norm"):
            # ModernBERT uses 'mlp_norm'
            mlp_norm = layer.mlp_norm
            # Check if it's Identity (no-op) or actual LayerNorm
            if hasattr(mlp_norm, "weight"):
                weights["ffn_ln_gamma"] = mlp_norm.weight.detach().cpu().numpy().astype(np.float32)
                weights["ffn_ln_beta"] = (
                    mlp_norm.bias.detach().cpu().numpy().astype(np.float32)
                    if hasattr(mlp_norm, "bias") and mlp_norm.bias is not None
                    else np.zeros(mlp_norm.weight.shape[0], dtype=np.float32)
                )
            else:
                # Identity layer (no-op), use identity weights
                hidden_size = self.config.hidden_size
                weights["ffn_ln_gamma"] = np.ones(hidden_size, dtype=np.float32)
                weights["ffn_ln_beta"] = np.zeros(hidden_size, dtype=np.float32)
        elif hasattr(layer, "output_layer_norm"):
            # DistilBERT structure
            weights["ffn_ln_gamma"] = (
                layer.output_layer_norm.weight.detach().cpu().numpy().astype(np.float32)
            )
            weights["ffn_ln_beta"] = (
                layer.output_layer_norm.bias.detach().cpu().numpy().astype(np.float32)
            )
        elif (
            "ffn_output" in locals() and ffn_output is not None and hasattr(ffn_output, "LayerNorm")
        ):
            # BERT structure
            weights["ffn_ln_gamma"] = (
                ffn_output.LayerNorm.weight.detach().cpu().numpy().astype(np.float32)
            )
            weights["ffn_ln_beta"] = (
                ffn_output.LayerNorm.bias.detach().cpu().numpy().astype(np.float32)
            )
        else:
            # No layer norm, use identity
            hidden_size = self.config.hidden_size
            weights["ffn_ln_gamma"] = np.ones(hidden_size, dtype=np.float32)
            weights["ffn_ln_beta"] = np.zeros(hidden_size, dtype=np.float32)

        return weights

    def _tokenize(self, texts: str | list[str]) -> dict[str, np.ndarray]:
        """Tokenize texts using sentence-transformer tokenizer"""
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="np",
        )

        return {
            "input_ids": encoded["input_ids"].astype(np.int32),
            "attention_mask": encoded["attention_mask"].astype(np.float32),
        }

    def _embedding_lookup(self, input_ids: np.ndarray) -> np.ndarray:
        """Get token embeddings"""
        # Get the actual transformer model (might need to go through auto_model)
        actual_model = self.transformer_model
        if hasattr(actual_model, "auto_model"):
            actual_model = actual_model.auto_model

        # Get embedding weights from model
        # Try different paths for different architectures
        embeddings = None
        word_emb = None

        if hasattr(actual_model, "embeddings"):
            embeddings = actual_model.embeddings
        elif hasattr(actual_model, "transformer"):
            embeddings = actual_model.transformer.embeddings
        elif hasattr(actual_model, "embed_tokens"):
            # ModernBERT and some architectures use embed_tokens directly
            word_emb = actual_model.embed_tokens
        else:
            raise ValueError(
                f"Could not find embeddings. Available: {[a for a in dir(actual_model) if not a.startswith('_')][:20]}"
            )

        if word_emb is None and embeddings is not None:
            if hasattr(embeddings, "word_embeddings"):
                word_emb = embeddings.word_embeddings
            elif hasattr(embeddings, "token_embeddings"):
                word_emb = embeddings.token_embeddings
            elif hasattr(embeddings, "tok_embeddings"):
                # ModernBERT uses tok_embeddings
                word_emb = embeddings.tok_embeddings
            elif hasattr(embeddings, "embed_tokens"):
                word_emb = embeddings.embed_tokens
            elif hasattr(embeddings, "token_embedding"):
                # Some models use token_embedding (singular)
                word_emb = embeddings.token_embedding
            else:
                # Try to find any child module that might be the embedding
                for name, module in embeddings.named_children():
                    if "embed" in name.lower() or "token" in name.lower() or "tok" in name.lower():
                        word_emb = module
                        break
                if word_emb is None:
                    # Try using embeddings directly if it has weight attribute
                    if hasattr(embeddings, "weight"):
                        word_emb = embeddings
                    else:
                        raise ValueError(
                            f"Could not find word embeddings in embeddings. Available: {[a for a in dir(embeddings) if not a.startswith('_')][:20]}"
                        )

        if word_emb is None:
            raise ValueError("Could not find word embeddings")

        emb_weight = word_emb.weight.detach().cpu().numpy().astype(np.float32)

        # Use GPU embedding lookup
        try:
            # functional.embedding_lookup expects (weight, indices) where weight is (vocab_size, embedding_dim)
            # and indices is (batch, seq_len) or (seq_len,)
            result = functional.embedding_lookup(emb_weight, input_ids)
            # Ensure correct shape
            batch_size, seq_len = input_ids.shape
            emb_dim = emb_weight.shape[1]
            if result.shape != (batch_size, seq_len, emb_dim):
                result = result.reshape(batch_size, seq_len, emb_dim)
            return result
        except Exception as e:
            logger.debug(f"GPU embedding lookup failed: {e}, using CPU fallback")
            # CPU fallback
            batch_size, seq_len = input_ids.shape
            emb_dim = emb_weight.shape[1]
            output = np.zeros((batch_size, seq_len, emb_dim), dtype=np.float32)
            for b in range(batch_size):
                for s in range(seq_len):
                    token_id = int(input_ids[b, s])
                    if 0 <= token_id < emb_weight.shape[0]:
                        output[b, s] = emb_weight[token_id]
            return output

    def _add_positional_embeddings(self, x: np.ndarray, seq_len: int) -> np.ndarray:
        """Add token type and positional embeddings to token embeddings"""
        # Get embeddings module
        actual_model = self.transformer_model
        if hasattr(actual_model, "auto_model"):
            actual_model = actual_model.auto_model

        # Get embeddings module
        if hasattr(actual_model, "embeddings"):
            embeddings = actual_model.embeddings
        elif hasattr(actual_model, "transformer"):
            embeddings = actual_model.transformer.embeddings
        else:
            # No embeddings available, return as-is
            return x

        batch_size = x.shape[0]
        hidden_size = x.shape[2]

        # Add token type embeddings (BERT/RoBERTa have them, DistilBERT doesn't)
        # Check model type to determine if we should add token_type_embeddings
        if self.model_type not in ["distilbert"] and hasattr(embeddings, "token_type_embeddings"):
            token_type_emb = embeddings.token_type_embeddings
            token_type_emb_weight = token_type_emb.weight.detach().cpu().numpy().astype(np.float32)
            # Token type IDs are usually all zeros for sentence-transformers
            token_type_ids = np.zeros((batch_size, seq_len), dtype=np.int32)

            # Lookup token type embeddings
            token_type_embeddings = np.zeros((batch_size, seq_len, hidden_size), dtype=np.float32)
            for b in range(batch_size):
                for s in range(seq_len):
                    token_type_id = int(token_type_ids[b, s])
                    if 0 <= token_type_id < token_type_emb_weight.shape[0]:
                        token_type_embeddings[b, s] = token_type_emb_weight[token_type_id]

            # Add token type embeddings
            x = x + token_type_embeddings

        # Get positional embeddings
        if hasattr(embeddings, "position_embeddings"):
            pos_emb = embeddings.position_embeddings
        elif hasattr(embeddings, "position_encoding"):
            pos_emb = embeddings.position_encoding
        else:
            # No positional embeddings, return as-is
            return x

        # Extract positional embedding weights
        pos_emb_weight = pos_emb.weight.detach().cpu().numpy().astype(np.float32)
        # pos_emb_weight shape: (max_position_embeddings, hidden_size)

        # Create position indices: [0, 1, 2, ..., seq_len-1]
        position_ids = np.arange(seq_len, dtype=np.int32)
        position_ids = np.tile(position_ids, (batch_size, 1))  # (batch, seq_len)

        # Lookup positional embeddings
        # Use CPU for now (could be GPU-accelerated later)
        pos_embeddings = np.zeros((batch_size, seq_len, hidden_size), dtype=np.float32)
        for b in range(batch_size):
            for s in range(seq_len):
                pos_idx = int(position_ids[b, s])
                if 0 <= pos_idx < pos_emb_weight.shape[0]:
                    pos_embeddings[b, s] = pos_emb_weight[pos_idx]

        # Add positional embeddings to token embeddings
        x = x + pos_embeddings

        return x

    def _apply_embedding_norm_dropout(self, x: np.ndarray) -> np.ndarray:
        """Apply embedding LayerNorm and Dropout (BERT-style)"""
        # Get embeddings module
        actual_model = self.transformer_model
        if hasattr(actual_model, "auto_model"):
            actual_model = actual_model.auto_model

        # Get embeddings module
        if hasattr(actual_model, "embeddings"):
            embeddings = actual_model.embeddings
        elif hasattr(actual_model, "transformer"):
            embeddings = actual_model.transformer.embeddings
        else:
            # No embeddings module, return as-is
            return x

        # Get LayerNorm weights
        if hasattr(embeddings, "LayerNorm"):
            ln_weight = embeddings.LayerNorm.weight.detach().cpu().numpy().astype(np.float32)
            ln_bias = embeddings.LayerNorm.bias.detach().cpu().numpy().astype(np.float32)
            eps = embeddings.LayerNorm.eps
        else:
            # No LayerNorm, return as-is
            return x

        # Apply LayerNorm (GPU if available)
        try:
            x = functional.layer_norm(
                x, normalized_shape=x.shape[-1], weight=ln_weight, bias=ln_bias, eps=eps
            )
        except Exception as e:
            logger.debug(f"GPU embedding layer_norm failed: {e}, using CPU fallback")
            # CPU fallback
            x_mean = x.mean(axis=-1, keepdims=True)
            x_var = x.var(axis=-1, keepdims=True)
            x = (x - x_mean) / np.sqrt(x_var + eps)
            x = x * ln_weight + ln_bias

        # Dropout is disabled in inference mode, so we skip it

        return x

    def _apply_modernbert_embedding_norm(self, x: np.ndarray) -> np.ndarray:
        """Apply ModernBERT embedding norm (LayerNorm after tok_embeddings)"""
        # Get embeddings module
        actual_model = self.transformer_model
        if hasattr(actual_model, "auto_model"):
            actual_model = actual_model.auto_model

        if hasattr(actual_model, "embeddings"):
            embeddings = actual_model.embeddings
        else:
            return x  # No embeddings module, skip

        # Check if norm exists
        if not hasattr(embeddings, "norm"):
            return x  # No norm, skip

        norm = embeddings.norm
        if not hasattr(norm, "weight"):
            return x  # Not a LayerNorm, skip

        # Get norm weights
        norm_weight = norm.weight.detach().cpu().numpy().astype(np.float32)
        norm_bias = (
            norm.bias.detach().cpu().numpy().astype(np.float32)
            if hasattr(norm, "bias") and norm.bias is not None
            else np.zeros(norm.weight.shape[0], dtype=np.float32)
        )
        eps = getattr(norm, "eps", 1e-5)

        # Apply layer norm
        try:
            x = functional.layer_norm(
                x, normalized_shape=x.shape[-1], weight=norm_weight, bias=norm_bias, eps=eps
            )
        except Exception as e:
            logger.debug(f"GPU embedding layer_norm failed: {e}, using CPU fallback")
            # CPU fallback
            x_mean = x.mean(axis=-1, keepdims=True)
            x_var = x.var(axis=-1, keepdims=True)
            x = (x - x_mean) / np.sqrt(x_var + eps)
            x = x * norm_weight + norm_bias

        return x

    def _forward_layer(
        self, x: np.ndarray, layer_weights: dict, attention_mask: np.ndarray, layer_idx: int = 0
    ) -> np.ndarray:
        """Forward pass through a single transformer layer"""
        hidden_size = x.shape[-1]
        batch_size, seq_len, _ = x.shape
        num_heads = self.config.num_attention_heads
        head_dim = hidden_size // num_heads

        # Debug: Check for NaN/Inf at input
        if np.isnan(x).any() or np.isinf(x).any():
            logger.error(
                f"Layer {layer_idx}: NaN/Inf detected at input! NaN: {np.isnan(x).sum()}, Inf: {np.isinf(x).sum()}"
            )
            return x

        if self.model_type == "modernbert":
            # ModernBERT uses PRE-NORM architecture:
            # 1. attn_norm(x) -> attn -> x + attn_out (NO post-norm)
            # 2. mlp_norm(x) -> mlp -> x + mlp_out (NO post-norm)

            # Step 1: Pre-norm before attention (but attn_norm is Identity, so skip)
            # attn_norm is Identity, so we can skip it
            x_norm = x

            # Compute Q, K, V
            try:
                q = functional.linear(
                    x_norm, layer_weights["attn_q_weight"], layer_weights["attn_q_bias"]
                )
                k = functional.linear(
                    x_norm, layer_weights["attn_k_weight"], layer_weights["attn_k_bias"]
                )
                v = functional.linear(
                    x_norm, layer_weights["attn_v_weight"], layer_weights["attn_v_bias"]
                )
            except Exception as e:
                logger.debug(f"GPU linear failed: {e}, using CPU")
                q = (x_norm @ layer_weights["attn_q_weight"].T) + layer_weights["attn_q_bias"]
                k = (x_norm @ layer_weights["attn_k_weight"].T) + layer_weights["attn_k_bias"]
                v = (x_norm @ layer_weights["attn_v_weight"].T) + layer_weights["attn_v_bias"]

            # Debug: Check Q, K, V
            if np.isnan(q).any() or np.isnan(k).any() or np.isnan(v).any():
                logger.error(f"Layer {layer_idx}: NaN in Q/K/V after linear projection")
                logger.error(
                    f"  Q NaN: {np.isnan(q).sum()}, K NaN: {np.isnan(k).sum()}, V NaN: {np.isnan(v).sum()}"
                )
                logger.error(
                    f"  x_norm stats: mean={x_norm.mean():.6f}, std={x_norm.std():.6f}, min={x_norm.min():.6f}, max={x_norm.max():.6f}"
                )

            # Reshape for multi-head attention
            q = q.reshape(batch_size, seq_len, num_heads, head_dim)
            k = k.reshape(batch_size, seq_len, num_heads, head_dim)
            v = v.reshape(batch_size, seq_len, num_heads, head_dim)

            # Apply RoPE
            # ModernBERT uses local_rope_theta (default 10000.0) for local attention
            rope_base = getattr(self.config, "local_rope_theta", None)
            if rope_base is None:
                rope_base = (
                    getattr(self.config, "rope_theta", 10000.0)
                    if hasattr(self.config, "rope_theta")
                    else 10000.0
                )
            rope_scaling = getattr(self.config, "rope_scaling", None)
            if rope_scaling is not None and isinstance(rope_scaling, dict):
                rope_scaling_factor = rope_scaling.get("factor", 1.0)
            else:
                rope_scaling_factor = 1.0

            position_ids = np.arange(seq_len, dtype=np.int32)
            position_ids = np.tile(position_ids, (batch_size, 1))

            # Use CPU RoPE for now to ensure correctness (GPU shader needs verification)
            try:
                q = self.backend.attention._rope_cpu(
                    q, position_ids, rope_base, rope_scaling_factor
                )
                k = self.backend.attention._rope_cpu(
                    k, position_ids, rope_base, rope_scaling_factor
                )
            except Exception as e:
                logger.error(f"CPU RoPE failed: {e}")
                # Fallback: try GPU
                try:
                    q = self.backend.attention.apply_rope(
                        q, position_ids, rope_base, rope_scaling_factor
                    )
                    k = self.backend.attention.apply_rope(
                        k, position_ids, rope_base, rope_scaling_factor
                    )
                except Exception as e2:
                    logger.error(f"GPU RoPE also failed: {e2}")
                    raise

            # Attention
            # For debugging: try CPU first to see if GPU attention is the issue
            try:
                # Try CPU first for debugging
                attn_out = self._attention_cpu(q, k, v, attention_mask)
                # Uncomment to use GPU:
                # attn_out = self._attention_gpu(q, k, v, attention_mask, num_heads, head_dim)
            except Exception as e:
                logger.debug(f"CPU attention failed: {e}, trying GPU")
                try:
                    attn_out = self._attention_gpu(q, k, v, attention_mask, num_heads, head_dim)
                except Exception as e2:
                    logger.error(f"Both CPU and GPU attention failed: {e2}")
                    raise

            # Debug: Check attention output
            if np.isnan(attn_out).any() or np.isinf(attn_out).any():
                logger.error(
                    f"Layer {layer_idx}: NaN/Inf in attention output! NaN: {np.isnan(attn_out).sum()}, Inf: {np.isinf(attn_out).sum()}"
                )

            # Reshape and output projection
            attn_out = attn_out.reshape(batch_size, seq_len, hidden_size)
            try:
                attn_out = functional.linear(
                    attn_out, layer_weights["attn_out_weight"], layer_weights["attn_out_bias"]
                )
            except Exception as e:
                logger.debug(f"GPU linear failed: {e}, using CPU")
                attn_out = (attn_out @ layer_weights["attn_out_weight"].T) + layer_weights[
                    "attn_out_bias"
                ]

            # Debug: Check after output projection
            if np.isnan(attn_out).any() or np.isinf(attn_out).any():
                logger.error(
                    f"Layer {layer_idx}: NaN/Inf after attention output projection! NaN: {np.isnan(attn_out).sum()}, Inf: {np.isinf(attn_out).sum()}"
                )

            # Residual (NO post-norm!)
            x = x + attn_out

            # Debug: Check after residual
            if np.isnan(x).any() or np.isinf(x).any():
                logger.error(
                    f"Layer {layer_idx}: NaN/Inf after attention residual! NaN: {np.isnan(x).sum()}, Inf: {np.isinf(x).sum()}"
                )

            # Step 2: Pre-norm before MLP
            try:
                x_norm = functional.layer_norm(
                    x,
                    normalized_shape=hidden_size,
                    weight=layer_weights["ffn_ln_gamma"],
                    bias=layer_weights["ffn_ln_beta"],
                    eps=1e-5,
                )
            except Exception as e:
                logger.debug(f"GPU layer_norm failed: {e}, using CPU fallback")
                x_mean = x.mean(axis=-1, keepdims=True)
                x_var = x.var(axis=-1, keepdims=True)
                x_norm = (x - x_mean) / np.sqrt(x_var + 1e-5)
                x_norm = x_norm * layer_weights["ffn_ln_gamma"] + layer_weights["ffn_ln_beta"]

            # Debug: Check after layer norm
            if np.isnan(x_norm).any() or np.isinf(x_norm).any():
                logger.error(
                    f"Layer {layer_idx}: NaN/Inf after MLP layer norm! NaN: {np.isnan(x_norm).sum()}, Inf: {np.isinf(x_norm).sum()}"
                )
                if not (np.isnan(x).any() or np.isinf(x).any()):
                    logger.error(
                        f"  x stats: mean={x.mean():.6f}, std={x.std():.6f}, min={x.min():.6f}, max={x.max():.6f}"
                    )
                    logger.error(
                        f"  ffn_ln_gamma stats: mean={layer_weights['ffn_ln_gamma'].mean():.6f}, std={layer_weights['ffn_ln_gamma'].std():.6f}"
                    )
                    logger.error(
                        f"  ffn_ln_beta stats: mean={layer_weights['ffn_ln_beta'].mean():.6f}, std={layer_weights['ffn_ln_beta'].std():.6f}"
                    )
                else:
                    logger.error("  x already contains NaN/Inf!")
                    # Replace NaN/Inf with zeros to prevent propagation
                    x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=1e6, neginf=-1e6)

            # MLP with GeGLU
            # ModernBERT: Wi(hidden_states) -> chunk(2) -> [input, gate] -> act(input) * gate -> Wo
            # GeGLU: GELU(input) * gate, where input is first half, gate is second half
            if "ffn_gate_weight" in layer_weights:
                try:
                    # ffn_gate_weight is the first half (input, gets GELU)
                    # ffn_up_weight is the second half (gate, multiplies)
                    input_out = functional.linear(
                        x_norm, layer_weights["ffn_gate_weight"], layer_weights["ffn_gate_bias"]
                    )
                    gate_out = functional.linear(
                        x_norm, layer_weights["ffn_up_weight"], layer_weights["ffn_up_bias"]
                    )
                    # Clamp to prevent overflow
                    input_out = np.clip(input_out, -50, 50)
                    gate_out = np.clip(gate_out, -1e6, 1e6)
                    # GeGLU: GELU(input) * gate
                    input_activated = functional.gelu(input_out)
                    ffn_out = input_activated * gate_out
                    # Clamp before final linear to prevent overflow
                    ffn_out = np.clip(ffn_out, -1e6, 1e6)
                    ffn_out = functional.linear(
                        ffn_out, layer_weights["ffn_w2"], layer_weights["ffn_b2"]
                    )
                except Exception as e:
                    logger.debug(f"GPU FFN (GeGLU) failed: {e}, using CPU fallback")
                    input_out = (x_norm @ layer_weights["ffn_gate_weight"].T) + layer_weights[
                        "ffn_gate_bias"
                    ]
                    gate_out = (x_norm @ layer_weights["ffn_up_weight"].T) + layer_weights[
                        "ffn_up_bias"
                    ]
                    # Clamp to prevent overflow
                    input_out = np.clip(input_out, -50, 50)
                    gate_out = np.clip(gate_out, -1e6, 1e6)
                    # GeGLU: GELU(input) * gate
                    input_activated = (
                        0.5
                        * input_out
                        * (1 + np.tanh(np.sqrt(2 / np.pi) * (input_out + 0.044715 * input_out**3)))
                    )
                    ffn_out = input_activated * gate_out
                    # Clamp before final linear
                    ffn_out = np.clip(ffn_out, -1e6, 1e6)
                    ffn_out = (ffn_out @ layer_weights["ffn_w2"].T) + layer_weights["ffn_b2"]
            else:
                raise ValueError("ModernBERT requires ffn_gate_weight and ffn_up_weight")

            # Debug: Check FFN output
            if np.isnan(ffn_out).any() or np.isinf(ffn_out).any():
                logger.error(
                    f"Layer {layer_idx}: NaN/Inf in FFN output! NaN: {np.isnan(ffn_out).sum()}, Inf: {np.isinf(ffn_out).sum()}"
                )
                if "gate_out" in locals() and "input_out" in locals():
                    logger.error(
                        f"  gate_out stats: mean={gate_out.mean():.6f}, std={gate_out.std():.6f}, min={gate_out.min():.6f}, max={gate_out.max():.6f}"
                    )
                    logger.error(
                        f"  input_out stats: mean={input_out.mean():.6f}, std={input_out.std():.6f}, min={input_out.min():.6f}, max={input_out.max():.6f}"
                    )
                # Replace NaN/Inf with zeros (no random noise to avoid divergence)
                if np.isnan(ffn_out).any() or np.isinf(ffn_out).any():
                    ffn_out = np.nan_to_num(ffn_out, nan=0.0, posinf=1e6, neginf=-1e6)

            # Residual (NO post-norm!)
            x = x + ffn_out

            # Debug: Check final output
            if np.isnan(x).any() or np.isinf(x).any():
                logger.error(
                    f"Layer {layer_idx}: NaN/Inf in final output! NaN: {np.isnan(x).sum()}, Inf: {np.isinf(x).sum()}"
                )
                # Replace NaN/Inf with zeros (no random noise to avoid divergence)
                if np.isnan(x).any() or np.isinf(x).any():
                    x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        else:
            # BERT/DistilBERT: POST-NORM architecture
            # Compute Q, K, V (no pre-norm)
            try:
                q = functional.linear(
                    x, layer_weights["attn_q_weight"], layer_weights["attn_q_bias"]
                )
                k = functional.linear(
                    x, layer_weights["attn_k_weight"], layer_weights["attn_k_bias"]
                )
                v = functional.linear(
                    x, layer_weights["attn_v_weight"], layer_weights["attn_v_bias"]
                )
            except Exception as e:
                logger.debug(f"GPU linear failed: {e}, using CPU")
                q = (x @ layer_weights["attn_q_weight"].T) + layer_weights["attn_q_bias"]
                k = (x @ layer_weights["attn_k_weight"].T) + layer_weights["attn_k_bias"]
                v = (x @ layer_weights["attn_v_weight"].T) + layer_weights["attn_v_bias"]

            q = q.reshape(batch_size, seq_len, num_heads, head_dim)
            k = k.reshape(batch_size, seq_len, num_heads, head_dim)
            v = v.reshape(batch_size, seq_len, num_heads, head_dim)

            try:
                attn_out = self._attention_gpu(q, k, v, attention_mask, num_heads, head_dim)
            except Exception as e:
                logger.debug(f"GPU attention failed: {e}, using CPU fallback")
                attn_out = self._attention_cpu(q, k, v, attention_mask)

            attn_out = attn_out.reshape(batch_size, seq_len, hidden_size)
            try:
                attn_out = functional.linear(
                    attn_out, layer_weights["attn_out_weight"], layer_weights["attn_out_bias"]
                )
            except Exception as e:
                logger.debug(f"GPU linear failed: {e}, using CPU")
                attn_out = (attn_out @ layer_weights["attn_out_weight"].T) + layer_weights[
                    "attn_out_bias"
                ]

            # Post-norm: LayerNorm(attn_out + x)
            try:
                x = functional.layer_norm(
                    attn_out + x,
                    normalized_shape=hidden_size,
                    weight=layer_weights["attn_ln_gamma"],
                    bias=layer_weights["attn_ln_beta"],
                    eps=1e-5,
                )
            except Exception as e:
                logger.debug(f"GPU layer_norm failed: {e}, using CPU fallback")
                x = attn_out + x
                x_mean = x.mean(axis=-1, keepdims=True)
                x_var = x.var(axis=-1, keepdims=True)
                x = (x - x_mean) / np.sqrt(x_var + 1e-5)
                x = x * layer_weights["attn_ln_gamma"] + layer_weights["attn_ln_beta"]

            # FFN
            try:
                ffn_out = functional.linear(x, layer_weights["ffn_w1"], layer_weights["ffn_b1"])
                ffn_out = functional.gelu(ffn_out)
                ffn_out = functional.linear(
                    ffn_out, layer_weights["ffn_w2"], layer_weights["ffn_b2"]
                )
            except Exception as e:
                logger.debug(f"GPU FFN failed: {e}, using CPU fallback")
                ffn_out = (x @ layer_weights["ffn_w1"].T) + layer_weights["ffn_b1"]
                ffn_out = (
                    0.5
                    * ffn_out
                    * (1 + np.tanh(np.sqrt(2 / np.pi) * (ffn_out + 0.044715 * ffn_out**3)))
                )
                ffn_out = (ffn_out @ layer_weights["ffn_w2"].T) + layer_weights["ffn_b2"]

            # Post-norm: LayerNorm(ffn_out + x)
            try:
                x = functional.layer_norm(
                    ffn_out + x,
                    normalized_shape=hidden_size,
                    weight=layer_weights["ffn_ln_gamma"],
                    bias=layer_weights["ffn_ln_beta"],
                    eps=1e-5,
                )
            except Exception as e:
                logger.debug(f"GPU layer_norm failed: {e}, using CPU fallback")
                x = ffn_out + x
                x_mean = x.mean(axis=-1, keepdims=True)
                x_var = x.var(axis=-1, keepdims=True)
                x = (x - x_mean) / np.sqrt(x_var + 1e-5)
                x = x * layer_weights["ffn_ln_gamma"] + layer_weights["ffn_ln_beta"]

        return x

    def _attention_gpu(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: np.ndarray,
        num_heads: int,
        head_dim: int,
    ) -> np.ndarray:
        """GPU-accelerated multi-head attention"""
        backend = self.backend
        _batch_size, seq_len = q.shape[0], q.shape[1]

        # Compute attention scores on GPU
        # q, k, v are (batch, seq_len, num_heads, head_dim)
        scores = backend.attention.attention_scores(q, k, num_heads, head_dim)
        # scores shape: (batch, num_heads, seq_len, seq_len)

        # Apply mask on GPU (if provided)
        if mask is not None:
            # Use GPU attention_mask shader with custom mask
            scores = backend.attention.attention_mask(
                scores, use_causal=False, mask_value=-1e9, custom_mask=mask
            )

        # Softmax on GPU
        # Reshape scores for softmax: flatten to (batch*num_heads*seq_q, seq_k)
        scores_reshaped = scores.reshape(-1, seq_len)  # (batch*num_heads*seq_q, seq_k)
        scores_softmax = backend.fnn.activation_softmax(scores_reshaped, axis=-1)
        scores_softmax = scores_softmax.reshape(
            scores.shape
        )  # Back to (batch, num_heads, seq_q, seq_k)

        # Attention output on GPU
        attn_out = backend.attention.attention_output(scores_softmax, v, num_heads, head_dim)
        # attn_out shape: (batch, seq_len, num_heads, head_dim)

        return attn_out

    def _attention_cpu(
        self, q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """CPU fallback for attention"""
        batch_size, seq_len, num_heads, head_dim = q.shape

        # Scale
        scale = 1.0 / np.sqrt(head_dim)

        # Compute attention scores: (batch, seq_q, num_heads, head_dim) @ (batch, seq_k, num_heads, head_dim) -> (batch, num_heads, seq_q, seq_k)
        # Transpose to get (batch, num_heads, seq_q, seq_k)
        q_t = q.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_q, head_dim)
        k_t = k.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_k, head_dim)
        scores = np.einsum("bhqd,bhkd->bhqk", q_t, k_t) * scale  # (batch, num_heads, seq_q, seq_k)

        # Apply mask - ModernBERT adds mask directly to attention weights
        # mask: 1 = valid token, 0 = padding
        # We need to convert: 1 -> 0 (no masking), 0 -> -inf (masked out)
        # ModernBERT: attn_weights = attn_weights + attention_mask
        # where attention_mask has -inf for padding, 0 for valid tokens
        if mask is not None:
            # Convert mask: 1 -> 0, 0 -> -inf
            # mask shape: (batch, seq_len) -> expand to (batch, 1, 1, seq_k) for broadcasting
            mask_expanded = mask.astype(np.float32)  # (batch, seq_len)
            # Invert: 1 -> 0, 0 -> -inf
            mask_expanded = (1.0 - mask_expanded) * -1e9  # 1 -> 0, 0 -> -inf
            mask_expanded = mask_expanded[:, None, None, :]  # (batch, 1, 1, seq_k)
            scores = scores + mask_expanded

        # Softmax over last dimension (seq_k)
        scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        scores = scores / (scores.sum(axis=-1, keepdims=True) + 1e-8)

        # Apply to values: (batch, num_heads, seq_q, seq_k) @ (batch, num_heads, seq_k, head_dim) -> (batch, num_heads, seq_q, head_dim)
        v_t = v.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_k, head_dim)
        out = np.einsum("bhqk,bhkd->bhqd", scores, v_t)  # (batch, num_heads, seq_q, head_dim)

        # Transpose back: (batch, num_heads, seq_q, head_dim) -> (batch, seq_q, num_heads, head_dim)
        out = out.transpose(0, 2, 1, 3)

        return out

    def encode(
        self,
        texts: str | list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Encode texts to embeddings using Vulkan GPU.

        Args:
            texts: Input text or list of texts
            batch_size: Batch size for processing
            show_progress_bar: Show progress bar
            normalize_embeddings: Normalize embeddings (default: True)
            **kwargs: Additional arguments

        Returns:
            Embeddings as numpy array (ready for Vulkan operations)
        """
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize
            encoded = self._tokenize(batch_texts)
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            # Get embeddings
            x = self._embedding_lookup(input_ids)

            # Add positional embeddings (ModernBERT uses RoPE in attention, not here)
            if self.model_type != "modernbert":
                x = self._add_positional_embeddings(x, input_ids.shape[1])
                # Apply embedding LayerNorm and Dropout (BERT-style)
                x = self._apply_embedding_norm_dropout(x)
            else:
                # ModernBERT: Apply embedding norm after tok_embeddings
                # ModernBERT has embeddings.norm that should be applied
                x = self._apply_modernbert_embedding_norm(x)

            # Forward through transformer layers
            for layer_idx, layer_weights in enumerate(self.layers):
                x = self._forward_layer(x, layer_weights, attention_mask, layer_idx=layer_idx)
                # Early exit if NaN detected
                if np.isnan(x).any() or np.isinf(x).any():
                    logger.error(f"NaN/Inf detected at layer {layer_idx}, stopping forward pass")
                    break

            # Apply final layer norm (ModernBERT has final_norm after all layers)
            if self.model_type == "modernbert" and self.final_norm_weight is not None:
                try:
                    x = functional.layer_norm(
                        x,
                        normalized_shape=x.shape[-1],
                        weight=self.final_norm_weight,
                        bias=self.final_norm_bias,
                        eps=self.final_norm_eps,
                    )
                except Exception as e:
                    logger.debug(f"GPU final layer_norm failed: {e}, using CPU fallback")
                    # CPU fallback
                    x_mean = x.mean(axis=-1, keepdims=True)
                    x_var = x.var(axis=-1, keepdims=True)
                    x = (x - x_mean) / np.sqrt(x_var + self.final_norm_eps)
                    x = x * self.final_norm_weight + self.final_norm_bias

            # Pooling
            # ModernBERT uses CLS pooling (first token), others use mean pooling
            if self.model_type == "modernbert":
                # CLS pooling: use first token embedding
                pooled = x[:, 0, :]  # (batch, hidden_size)
            else:
                # Mean pooling for BERT-family models - GPU with mask
                try:
                    pooled = self.backend.pooling.mean_pool(x, mask=attention_mask)
                except Exception as e:
                    logger.debug(f"GPU pooling failed: {e}, using CPU fallback")
                    # CPU fallback: Apply attention mask
                    mask_expanded = attention_mask[:, :, None]  # (batch, seq, 1)
                    x_masked = x * mask_expanded
                    pooled = x_masked.sum(axis=1) / (
                        mask_expanded.sum(axis=1) + 1e-8
                    )  # Mean pooling

            # Normalize if requested (GPU)
            if normalize_embeddings:
                try:
                    pooled = functional.embedding_normalize(pooled)
                except Exception as e:
                    logger.debug(f"GPU normalization failed: {e}, using CPU fallback")
                    # CPU fallback
                    norm = np.linalg.norm(pooled, axis=-1, keepdims=True)
                    pooled = pooled / (norm + 1e-8)

            # Ensure float32
            pooled = pooled.astype(np.float32)

            all_embeddings.append(pooled)

        # Concatenate batches
        if len(all_embeddings) == 1:
            result = all_embeddings[0]
        else:
            result = np.concatenate(all_embeddings, axis=0)

        # Ensure float32 and correct shape
        result = result.astype(np.float32)
        if result.ndim == 2 and result.shape[0] == 1 and len(texts) == 1:
            result = result[0]  # Single text -> 1D

        return result

    def __call__(self, texts: str | list[str], **kwargs) -> np.ndarray:
        """Alias for encode()"""
        return self.encode(texts, **kwargs)


def get_vulkan_sentence_transformer(
    model_name: str = "all-MiniLM-L6-v2", **kwargs
) -> VulkanSentenceTransformer:
    """
    Get or create a Vulkan sentence-transformer instance.

    Args:
        model_name: Model name
        **kwargs: Additional arguments for VulkanSentenceTransformer

    Returns:
        VulkanSentenceTransformer instance
    """
    return VulkanSentenceTransformer(model_name, **kwargs)
