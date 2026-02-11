"""Multimodal fusion modules for Grilly neural network workflows."""

import numpy as np

from .module import Module
from .modules import GELU, Dropout, LayerNorm, Linear, MultiheadAttention, Sequential

# =============================================================================
# 1. Multimodal Bottleneck Transformer (MBT)
# =============================================================================


class BottleneckFusion(Module):
    """Fuse modalities through a bottleneck-token attention mechanism."""

    def __init__(
        self,
        d_model: int = 768,
        num_bottlenecks: int = 64,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """Initialize the instance."""

        super().__init__()
        self.d_model = d_model
        self.num_bottlenecks = num_bottlenecks
        self.num_heads = num_heads

        # Learnable bottleneck tokens
        self.bottleneck_tokens = (
            np.random.randn(1, num_bottlenecks, d_model).astype(np.float32) * 0.02
        )
        self.register_parameter("bottleneck_tokens", self.bottleneck_tokens)

        # Cross-attention from bottleneck to modality 1
        self.cross_attn_1 = MultiheadAttention(d_model, num_heads, dropout=dropout)

        # Cross-attention from bottleneck to modality 2
        self.cross_attn_2 = MultiheadAttention(d_model, num_heads, dropout=dropout)

        # Self-attention on bottleneck
        self.self_attn = MultiheadAttention(d_model, num_heads, dropout=dropout)

        # Layer norms
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self._modules = {
            "cross_attn_1": self.cross_attn_1,
            "cross_attn_2": self.cross_attn_2,
            "self_attn": self.self_attn,
            "norm1": self.norm1,
            "norm2": self.norm2,
            "norm3": self.norm3,
        }

    def forward(self, modality1: np.ndarray, modality2: np.ndarray) -> np.ndarray:
        """
        Fuse two modalities through bottleneck attention.

        Args:
            modality1: First modality features (batch, seq1, d_model)
            modality2: Second modality features (batch, seq2, d_model)

        Returns:
            Fused representation (batch, num_bottlenecks, d_model)
        """
        batch_size = modality1.shape[0]

        # Expand bottleneck tokens for batch
        bottleneck = np.tile(self.bottleneck_tokens, (batch_size, 1, 1))

        # Cross-attend from bottleneck to modality 1
        attn_out1, _ = self.cross_attn_1(query=bottleneck, key=modality1, value=modality1)
        bottleneck = self.norm1(bottleneck + attn_out1)

        # Cross-attend from bottleneck to modality 2
        attn_out2, _ = self.cross_attn_2(query=bottleneck, key=modality2, value=modality2)
        bottleneck = self.norm2(bottleneck + attn_out2)

        # Self-attention on fused bottleneck
        attn_out3, _ = self.self_attn(query=bottleneck, key=bottleneck, value=bottleneck)
        fused = self.norm3(bottleneck + attn_out3)

        return fused


# =============================================================================
# 2. Perceiver IO - Modality-Agnostic Architecture
# =============================================================================


class PerceiverIO(Module):
    """
    Perceiver IO for modality-agnostic multimodal processing.

    Handles arbitrary input/output modalities (images, audio, video, text,
    point clouds, etc.) using cross-attention to map variable-length inputs
    to fixed latent representations.

    Key Innovation: Asymmetric attention bottleneck achieving O(M×L) complexity
    instead of O(M²) where M is input size.

    Reference: DeepMind - "Perceiver IO: A General Architecture for
    Structured Inputs & Outputs"

    Args:
        input_dim: Input feature dimension
        latent_dim: Latent space dimension
        num_latents: Number of latent tokens (bottleneck size)
        num_heads: Number of attention heads
        num_layers: Number of latent processing layers
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int = 512,
        latent_dim: int = 1024,
        num_latents: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        """Initialize the instance."""

        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_latents = num_latents

        # Learnable latent array (bottleneck)
        self.latents = np.random.randn(1, num_latents, latent_dim).astype(np.float32) * 0.02
        self.register_parameter("latents", self.latents)

        # Input projection
        self.input_proj = Linear(input_dim, latent_dim)

        # Input cross-attention (encode)
        self.input_cross_attn = MultiheadAttention(latent_dim, num_heads, dropout=dropout)
        self.input_norm = LayerNorm(latent_dim)

        # Latent self-attention processor layers
        self.latent_layers = []
        for i in range(num_layers):
            layer = {
                "self_attn": MultiheadAttention(latent_dim, num_heads, dropout=dropout),
                "norm1": LayerNorm(latent_dim),
                "ffn": Sequential(
                    Linear(latent_dim, latent_dim * 4),
                    GELU(),
                    Dropout(dropout),
                    Linear(latent_dim * 4, latent_dim),
                ),
                "norm2": LayerNorm(latent_dim),
            }
            self.latent_layers.append(layer)

        # Output cross-attention (decode)
        self.output_cross_attn = MultiheadAttention(latent_dim, num_heads, dropout=dropout)
        self.output_norm = LayerNorm(latent_dim)

        # Register submodules
        self._modules = {
            "input_proj": self.input_proj,
            "input_cross_attn": self.input_cross_attn,
            "input_norm": self.input_norm,
            "output_cross_attn": self.output_cross_attn,
            "output_norm": self.output_norm,
        }
        for i, layer in enumerate(self.latent_layers):
            for name, module in layer.items():
                self._modules[f"latent_{i}_{name}"] = module

    def forward(self, inputs: np.ndarray, output_queries: np.ndarray | None = None) -> np.ndarray:
        """
        Process inputs through Perceiver IO.

        Args:
            inputs: Input features (batch, input_seq, input_dim)
            output_queries: Optional output query tokens (batch, output_seq, latent_dim)
                          If None, returns the latent representation

        Returns:
            outputs: (batch, output_seq, latent_dim) or (batch, num_latents, latent_dim)
        """
        batch_size = inputs.shape[0]

        # Project inputs to latent dimension
        inputs_proj = self.input_proj(inputs)

        # Expand latents for batch
        latents = np.tile(self.latents, (batch_size, 1, 1))

        # Cross-attend from latents to inputs (encoding)
        encoded, _ = self.input_cross_attn(query=latents, key=inputs_proj, value=inputs_proj)
        latents = self.input_norm(latents + encoded)

        # Process latents with self-attention layers
        for layer in self.latent_layers:
            # Self-attention
            attn_out, _ = layer["self_attn"](query=latents, key=latents, value=latents)
            latents = layer["norm1"](latents + attn_out)

            # Feed-forward
            ffn_out = layer["ffn"](latents)
            latents = layer["norm2"](latents + ffn_out)

        # Decode: cross-attend from output queries to latents
        if output_queries is None:
            outputs = latents
        else:
            decoded, _ = self.output_cross_attn(query=output_queries, key=latents, value=latents)
            outputs = self.output_norm(output_queries + decoded)

        return outputs


# =============================================================================
# 3. Cross-Attention Fusion with Modality-Specific Encoders
# =============================================================================


class CrossModalAttentionFusion(Module):
    """
    Cross-attention based multimodal fusion with modality-specific
    and modality-agnostic representations.

    Uses dual-space learning: Models learn both unique modality features
    (specific) and shared cross-modal patterns (agnostic).

    Reference: "Modality-Specific and Agnostic Representations for
    Multimodal Understanding"

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        num_encoder_layers: Number of modality-specific encoder layers
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 8,
        num_encoder_layers: int = 3,
        dropout: float = 0.1,
    ):
        """Initialize the instance."""

        super().__init__()
        self.d_model = d_model

        # Modality-specific encoders
        self.vision_encoder_layers = []
        self.text_encoder_layers = []

        for i in range(num_encoder_layers):
            # Vision encoder layer
            vision_layer = {
                "self_attn": MultiheadAttention(d_model, num_heads, dropout=dropout),
                "norm1": LayerNorm(d_model),
                "ffn": Sequential(
                    Linear(d_model, d_model * 4),
                    GELU(),
                    Dropout(dropout),
                    Linear(d_model * 4, d_model),
                ),
                "norm2": LayerNorm(d_model),
            }
            self.vision_encoder_layers.append(vision_layer)

            # Text encoder layer
            text_layer = {
                "self_attn": MultiheadAttention(d_model, num_heads, dropout=dropout),
                "norm1": LayerNorm(d_model),
                "ffn": Sequential(
                    Linear(d_model, d_model * 4),
                    GELU(),
                    Dropout(dropout),
                    Linear(d_model * 4, d_model),
                ),
                "norm2": LayerNorm(d_model),
            }
            self.text_encoder_layers.append(text_layer)

        # Cross-modal attention (text to vision)
        self.cross_attn_t2v = MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm_t2v = LayerNorm(d_model)

        # Cross-modal attention (vision to text)
        self.cross_attn_v2t = MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm_v2t = LayerNorm(d_model)

        # Fusion FFN
        self.fusion_ffn = Sequential(
            Linear(d_model * 2, d_model * 4),
            GELU(),
            Dropout(dropout),
            Linear(d_model * 4, d_model),
        )

        # Register modules
        self._modules = {
            "cross_attn_t2v": self.cross_attn_t2v,
            "cross_attn_v2t": self.cross_attn_v2t,
            "norm_t2v": self.norm_t2v,
            "norm_v2t": self.norm_v2t,
            "fusion_ffn": self.fusion_ffn,
        }

    def _encode_modality(self, x: np.ndarray, layers: list[dict]) -> np.ndarray:
        """Encode through modality-specific layers."""
        for layer in layers:
            # Self-attention
            attn_out, _ = layer["self_attn"](query=x, key=x, value=x)
            x = layer["norm1"](x + attn_out)

            # Feed-forward
            ffn_out = layer["ffn"](x)
            x = layer["norm2"](x + ffn_out)
        return x

    def forward(self, vision_input: np.ndarray, text_input: np.ndarray) -> np.ndarray:
        """
        Fuse vision and text modalities.

        Args:
            vision_input: Vision features (batch, vision_seq, d_model)
            text_input: Text features (batch, text_seq, d_model)

        Returns:
            Fused representation (batch, d_model)
        """
        # Modality-specific encoding
        vision_specific = self._encode_modality(vision_input, self.vision_encoder_layers)
        text_specific = self._encode_modality(text_input, self.text_encoder_layers)

        # Cross-modal attention: text queries attend to vision
        text_cross, _ = self.cross_attn_t2v(
            query=text_specific, key=vision_specific, value=vision_specific
        )
        text_enriched = self.norm_t2v(text_specific + text_cross)

        # Cross-modal attention: vision queries attend to text
        vision_cross, _ = self.cross_attn_v2t(
            query=vision_specific, key=text_specific, value=text_specific
        )
        vision_enriched = self.norm_v2t(vision_specific + vision_cross)

        # Pool modality-specific representations (mean pooling)
        vision_pooled = vision_enriched.mean(axis=1)  # (batch, d_model)
        text_pooled = text_enriched.mean(axis=1)  # (batch, d_model)

        # Concatenate and fuse
        concatenated = np.concatenate([vision_pooled, text_pooled], axis=-1)
        fused = self.fusion_ffn(concatenated)

        return fused


# =============================================================================
# 4. ImageBind: Joint Embedding Across Multiple Modalities
# =============================================================================


class ImageBindFusion(Module):
    """
    ImageBind-style joint embedding for multiple modalities.

    Uses contrastive learning with image as anchor modality to create
    a single unified embedding space for images, video, text, audio,
    depth, thermal, and IMU data.

    Key Innovation: Image-paired training creates emergent joint space
    across all modalities without requiring paired data for all combinations.

    Reference: Meta AI - "ImageBind: One Embedding Space To Bind Them All"

    Args:
        embed_dim: Embedding dimension
        image_input_dim: Image encoder output dimension
        text_input_dim: Text encoder output dimension
        audio_input_dim: Audio encoder output dimension
        temperature: Contrastive learning temperature
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        image_input_dim: int = 2048,
        text_input_dim: int = 768,
        audio_input_dim: int = 512,
        temperature: float = 0.07,
    ):
        """Initialize the instance."""

        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature

        # Modality encoders (project to joint embedding space)
        self.image_encoder = Sequential(
            Linear(image_input_dim, embed_dim * 2),
            GELU(),
            Linear(embed_dim * 2, embed_dim),
        )

        self.text_encoder = Sequential(
            Linear(text_input_dim, embed_dim),
            GELU(),
            Linear(embed_dim, embed_dim),
        )

        self.audio_encoder = Sequential(
            Linear(audio_input_dim, embed_dim),
            GELU(),
            Linear(embed_dim, embed_dim),
        )

        self._modules = {
            "image_encoder": self.image_encoder,
            "text_encoder": self.text_encoder,
            "audio_encoder": self.audio_encoder,
        }

    def encode(self, modality_type: str, features: np.ndarray) -> np.ndarray:
        """
        Encode features from any modality to joint embedding space.

        Args:
            modality_type: 'image', 'text', or 'audio'
            features: Input features

        Returns:
            L2-normalized embeddings (batch, embed_dim)
        """
        if modality_type == "image":
            embeddings = self.image_encoder(features)
        elif modality_type == "text":
            embeddings = self.text_encoder(features)
        elif modality_type == "audio":
            embeddings = self.audio_encoder(features)
        else:
            raise ValueError(f"Unknown modality: {modality_type}")

        # L2 normalize
        norm = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        embeddings = embeddings / (norm + 1e-8)

        return embeddings

    def contrastive_loss(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> float:
        """
        Compute symmetric contrastive loss between two modality embeddings.

        Args:
            embeddings1: First modality embeddings (batch, embed_dim)
            embeddings2: Second modality embeddings (batch, embed_dim)

        Returns:
            Contrastive loss value
        """
        # Compute similarity matrix
        logits = (embeddings1 @ embeddings2.T) / self.temperature

        batch_size = embeddings1.shape[0]
        labels = np.arange(batch_size)

        # Cross-entropy loss for both directions
        def softmax_cross_entropy(logits, labels):
            # Softmax
            """Execute softmax cross entropy."""

            exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
            # Cross-entropy
            loss = -np.log(probs[np.arange(len(labels)), labels] + 1e-8)
            return loss.mean()

        loss_i2t = softmax_cross_entropy(logits, labels)
        loss_t2i = softmax_cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2

    def forward(
        self,
        image_features: np.ndarray,
        text_features: np.ndarray,
        audio_features: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Joint training with image as binding modality.

        Args:
            image_features: Image features (batch, image_input_dim)
            text_features: Text features (batch, text_input_dim)
            audio_features: Optional audio features (batch, audio_input_dim)

        Returns:
            Dictionary with embeddings and losses
        """
        # Encode all modalities
        image_emb = self.encode("image", image_features)
        text_emb = self.encode("text", text_features)

        result = {
            "image_embeddings": image_emb,
            "text_embeddings": text_emb,
            "loss_img_text": self.contrastive_loss(image_emb, text_emb),
        }

        if audio_features is not None:
            audio_emb = self.encode("audio", audio_features)
            result["audio_embeddings"] = audio_emb
            result["loss_img_audio"] = self.contrastive_loss(image_emb, audio_emb)
            result["loss"] = result["loss_img_text"] + result["loss_img_audio"]
        else:
            result["loss"] = result["loss_img_text"]

        return result


# =============================================================================
# 5. Perceiver Resampler (Flamingo Architecture)
# =============================================================================


class PerceiverResampler(Module):
    """
    Perceiver Resampler from Flamingo architecture.

    Converts variable-length visual sequences (images/videos) into a fixed
    number of tokens for efficient cross-attention with language models.

    Key Innovation: Reduces visual tokens from thousands (e.g., 1024 ViT patches)
    to fixed 64 tokens using learned latent queries.

    Reference: DeepMind - "Flamingo: a Visual Language Model for Few-Shot Learning"

    Args:
        dim: Feature dimension
        depth: Number of resampler layers
        num_latents: Number of output latent tokens (default: 64)
        num_heads: Number of attention heads
        ff_mult: Feed-forward multiplier
    """

    def __init__(
        self,
        dim: int = 1024,
        depth: int = 6,
        num_latents: int = 64,
        num_heads: int = 8,
        ff_mult: int = 4,
    ):
        """Initialize the instance."""

        super().__init__()
        self.dim = dim
        self.num_latents = num_latents

        # Learned latent queries
        self.latents = np.random.randn(num_latents, dim).astype(np.float32) * 0.02
        self.register_parameter("latents", self.latents)

        # Perceiver layers
        self.layers = []
        for i in range(depth):
            layer = {
                # Cross-attention: latents attend to visual features
                "cross_attn": MultiheadAttention(dim, num_heads),
                "cross_norm": LayerNorm(dim),
                # Self-attention: latents attend to each other
                "self_attn": MultiheadAttention(dim, num_heads),
                "self_norm": LayerNorm(dim),
                # Feed-forward
                "ffn": Sequential(
                    Linear(dim, dim * ff_mult),
                    GELU(),
                    Linear(dim * ff_mult, dim),
                ),
                "ff_norm": LayerNorm(dim),
            }
            self.layers.append(layer)

        # Register modules
        self._modules = {}
        for i, layer in enumerate(self.layers):
            for name, module in layer.items():
                self._modules[f"layer_{i}_{name}"] = module

    def forward(self, visual_features: np.ndarray) -> np.ndarray:
        """
        Resample visual features to fixed number of tokens.

        Args:
            visual_features: Variable-length visual features
                           (batch, variable_seq, dim)
                           Can be images (49-196 patches) or videos (many frames)

        Returns:
            resampled: Fixed tokens (batch, num_latents, dim)
        """
        batch_size = visual_features.shape[0]

        # Expand latents for batch
        latents = np.tile(self.latents[np.newaxis, :, :], (batch_size, 1, 1))

        # Apply Perceiver layers
        for layer in self.layers:
            # Cross-attention: latents attend to visual features
            cross_out, _ = layer["cross_attn"](
                query=latents, key=visual_features, value=visual_features
            )
            latents = layer["cross_norm"](latents + cross_out)

            # Self-attention: latents attend to each other
            self_out, _ = layer["self_attn"](query=latents, key=latents, value=latents)
            latents = layer["self_norm"](latents + self_out)

            # Feed-forward
            ff_out = layer["ffn"](latents)
            latents = layer["ff_norm"](latents + ff_out)

        return latents


# =============================================================================
# 6. Flamingo-Style Vision-Language Fusion
# =============================================================================


class FlamingoFusion(Module):
    """
    Vision-Language Model fusion using Perceiver Resampler.

    Combines:
    - Perceiver Resampler for variable visual input compression
    - Gated cross-attention for vision-language fusion

    Args:
        vision_dim: Vision feature dimension
        text_dim: Text/language model dimension
        num_visual_tokens: Number of visual tokens after resampling
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        vision_dim: int = 1024,
        text_dim: int = 2048,
        num_visual_tokens: int = 64,
        num_heads: int = 16,
    ):
        """Initialize the instance."""

        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim

        # Perceiver Resampler for vision
        self.resampler = PerceiverResampler(dim=vision_dim, num_latents=num_visual_tokens)

        # Project visual tokens to text dimension
        self.vision_proj = Linear(vision_dim, text_dim)

        # Gated cross-attention
        self.cross_attn = MultiheadAttention(text_dim, num_heads)
        self.cross_norm = LayerNorm(text_dim)

        # Gating mechanism
        self.gate = Sequential(
            Linear(text_dim, 1),
        )

        self._modules = {
            "resampler": self.resampler,
            "vision_proj": self.vision_proj,
            "cross_attn": self.cross_attn,
            "cross_norm": self.cross_norm,
            "gate": self.gate,
        }

    def forward(self, visual_features: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
        """
        Fuse visual and text features.

        Args:
            visual_features: Variable-length visual features (batch, var_seq, vision_dim)
            text_embeddings: Text embeddings (batch, text_seq, text_dim)

        Returns:
            Fused text embeddings (batch, text_seq, text_dim)
        """
        # Resample visual features to fixed tokens
        visual_tokens = self.resampler(visual_features)  # (batch, 64, vision_dim)

        # Project to text dimension
        visual_tokens = self.vision_proj(visual_tokens)  # (batch, 64, text_dim)

        # Gated cross-attention: text attends to vision
        attended, _ = self.cross_attn(query=text_embeddings, key=visual_tokens, value=visual_tokens)

        # Compute gate values (tanh for range [-1, 1])
        gate_values = np.tanh(self.gate(text_embeddings))

        # Fuse with gating
        fused_text = text_embeddings + gate_values * attended
        fused_text = self.cross_norm(fused_text)

        return fused_text


# =============================================================================
# 7. Complete Vision-Language Model
# =============================================================================


class VisionLanguageModel(Module):
    """
    Complete multimodal Vision-Language Model combining:
    - Perceiver Resampler for vision
    - Cross-Attention for fusion
    - Transformer layers for processing

    Args:
        vision_dim: Vision encoder output dimension
        text_dim: Text embedding dimension
        hidden_dim: Hidden dimension for transformer
        num_visual_tokens: Number of visual tokens after resampling
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        vocab_size: Vocabulary size for language model
    """

    def __init__(
        self,
        vision_dim: int = 1024,
        text_dim: int = 768,
        hidden_dim: int = 2048,
        num_visual_tokens: int = 64,
        num_heads: int = 16,
        num_layers: int = 12,
        vocab_size: int = 50257,
    ):
        """Initialize the instance."""

        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Visual resampler
        self.visual_resampler = PerceiverResampler(
            dim=vision_dim, num_latents=num_visual_tokens, depth=6
        )

        # Projections
        self.vision_proj = Linear(vision_dim, hidden_dim)
        self.text_proj = Linear(text_dim, hidden_dim)

        # Text embedding (simplified - in practice use pretrained)
        self.text_embed_weight = np.random.randn(vocab_size, text_dim).astype(np.float32) * 0.02
        self.register_parameter("text_embed_weight", self.text_embed_weight)

        # VLM layers
        self.layers = []
        for i in range(num_layers):
            layer = VLMLayer(hidden_dim, num_heads)
            self.layers.append(layer)

        # Output head
        self.lm_head = Linear(hidden_dim, vocab_size)

        # Register modules
        self._modules = {
            "visual_resampler": self.visual_resampler,
            "vision_proj": self.vision_proj,
            "text_proj": self.text_proj,
            "lm_head": self.lm_head,
        }
        for i, layer in enumerate(self.layers):
            self._modules[f"layer_{i}"] = layer

    def forward(self, vision_features: np.ndarray, input_ids: np.ndarray) -> np.ndarray:
        """
        Forward pass through VLM.

        Args:
            vision_features: Visual features (batch, patches, vision_dim)
            input_ids: Token IDs (batch, seq_len)

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        # Encode vision
        visual_tokens = self.visual_resampler(vision_features)
        visual_tokens = self.vision_proj(visual_tokens)

        # Embed text
        text_embeds = self.text_embed_weight[input_ids]  # (batch, seq_len, text_dim)
        text_tokens = self.text_proj(text_embeds)

        # Process through VLM layers
        for layer in self.layers:
            text_tokens = layer(text_tokens, visual_tokens)

        # Generate logits
        logits = self.lm_head(text_tokens)

        return logits


class VLMLayer(Module):
    """Single VLM transformer layer with vision cross-attention."""

    def __init__(self, dim: int = 2048, num_heads: int = 16):
        """Initialize the instance."""

        super().__init__()
        self.dim = dim

        # Self-attention on text
        self.self_attn = MultiheadAttention(dim, num_heads)
        self.norm1 = LayerNorm(dim)

        # Cross-attention: text queries attend to vision
        self.cross_attn = MultiheadAttention(dim, num_heads)
        self.norm2 = LayerNorm(dim)

        # Feed-forward
        self.ffn = Sequential(
            Linear(dim, dim * 4),
            GELU(),
            Linear(dim * 4, dim),
        )
        self.norm3 = LayerNorm(dim)

        # Gating for cross-attention
        self.gate = np.zeros((1,), dtype=np.float32)
        self.register_parameter("gate", self.gate)

        self._modules = {
            "self_attn": self.self_attn,
            "norm1": self.norm1,
            "cross_attn": self.cross_attn,
            "norm2": self.norm2,
            "ffn": self.ffn,
            "norm3": self.norm3,
        }

    def forward(self, text_tokens: np.ndarray, visual_tokens: np.ndarray) -> np.ndarray:
        """
        Process text tokens with self-attention and vision cross-attention.

        Args:
            text_tokens: (batch, text_seq, dim)
            visual_tokens: (batch, visual_seq, dim)

        Returns:
            Processed text tokens (batch, text_seq, dim)
        """
        # Self-attention
        attn_out, _ = self.self_attn(query=text_tokens, key=text_tokens, value=text_tokens)
        text_tokens = self.norm1(text_tokens + attn_out)

        # Cross-attention with gating
        cross_out, _ = self.cross_attn(query=text_tokens, key=visual_tokens, value=visual_tokens)
        gate_value = np.tanh(self.gate)
        text_tokens = self.norm2(text_tokens + gate_value * cross_out)

        # Feed-forward
        ffn_out = self.ffn(text_tokens)
        text_tokens = self.norm3(text_tokens + ffn_out)

        return text_tokens


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # 1. Multimodal Bottleneck Transformer
    "BottleneckFusion",
    # 2. Perceiver IO
    "PerceiverIO",
    # 3. Cross-Attention Fusion
    "CrossModalAttentionFusion",
    # 4. ImageBind
    "ImageBindFusion",
    # 5. Perceiver Resampler (Flamingo)
    "PerceiverResampler",
    "FlamingoFusion",
    # 6. Complete VLM
    "VisionLanguageModel",
    "VLMLayer",
]
