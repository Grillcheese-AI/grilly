"""
Shader Registry for Architecture-Specific Optimizations

This module provides a registry system for selecting architecture-specific shaders
when available, falling back to generic shaders otherwise.

Supported architectures:
- BERT (bert): Bidirectional attention, token type embeddings
- DistilBERT (distilbert): Bidirectional attention, no token type embeddings
- RoBERTa (roberta): Similar to BERT, optimized training
- GPT (gpt): Causal attention, pre-normalization
- T5 (t5): Encoder-decoder with cross-attention
- MPNet (mpnet): Masked and permuted pre-training
- XLM-RoBERTa (xlm-roberta): Multilingual RoBERTa
- ALBERT (albert): Factorized embeddings, parameter sharing
"""

import logging

logger = logging.getLogger(__name__)


class ShaderRegistry:
    """Registry for architecture-specific shaders"""

    def __init__(self):
        # Map: (shader_name, architecture) -> shader_file
        """Initialize the instance."""

        self._registry: dict[tuple, str] = {}
        self._generic_shaders: dict[str, str] = {}

    def register_generic(self, shader_name: str, shader_file: str):
        """Register a generic shader (works for all architectures)"""
        self._generic_shaders[shader_name] = shader_file

    def register_architecture_specific(self, shader_name: str, architecture: str, shader_file: str):
        """Register an architecture-specific shader"""
        key = (shader_name, architecture.lower())
        self._registry[key] = shader_file

    def get_shader(self, shader_name: str, architecture: str | None = None) -> str | None:
        """
        Get shader file name for given architecture.

        Args:
            shader_name: Base shader name (e.g., 'attention-output')
            architecture: Model architecture (e.g., 'bert', 'gpt', 't5')

        Returns:
            Shader file name (without .glsl extension) or None if not found
        """
        # Try architecture-specific first
        if architecture:
            key = (shader_name, architecture.lower())
            if key in self._registry:
                logger.debug(f"Using architecture-specific shader: {shader_name}-{architecture}")
                return self._registry[key]

        # Fall back to generic
        if shader_name in self._generic_shaders:
            logger.debug(f"Using generic shader: {shader_name}")
            return self._generic_shaders[shader_name]

        # Try architecture-specific without architecture (for backwards compatibility)
        if architecture:
            # Check if there's a variant like 'attention-output-bert'
            variant_name = f"{shader_name}-{architecture.lower()}"
            if variant_name in self._generic_shaders:
                logger.debug(f"Using variant shader: {variant_name}")
                return self._generic_shaders[variant_name]

        logger.warning(f"Shader not found: {shader_name} (architecture: {architecture})")
        return None

    def list_shaders(self, architecture: str | None = None) -> list[str]:
        """List all available shaders for an architecture"""
        shaders = set()

        if architecture:
            # Add architecture-specific shaders
            for (name, arch), _ in self._registry.items():
                if arch == architecture.lower():
                    shaders.add(name)

        # Add generic shaders
        shaders.update(self._generic_shaders.keys())

        return sorted(shaders)


# Global registry instance
_registry = ShaderRegistry()


def get_registry() -> ShaderRegistry:
    """Get the global shader registry"""
    return _registry


def register_generic_shader(shader_name: str, shader_file: str):
    """Register a generic shader"""
    _registry.register_generic(shader_name, shader_file)


def register_architecture_shader(shader_name: str, architecture: str, shader_file: str):
    """Register an architecture-specific shader"""
    _registry.register_architecture_specific(shader_name, architecture, shader_file)


def get_shader(shader_name: str, architecture: str | None = None) -> str | None:
    """Get shader file name for given architecture"""
    return _registry.get_shader(shader_name, architecture)


# Initialize with known generic shaders
def _initialize_registry():
    """Initialize the registry with default shaders"""
    # Generic shaders (work for all architectures)
    generic_shaders = [
        "activation-gelu",
        "activation-relu",
        "activation-silu",
        "activation-softmax",
        "attention-scores",
        "attention-mask",
        "attention-output",
        "attention-concat-heads",
        "embedding-lookup",
        "embedding-lookup-tiled",
        "embedding-pool-mask",
        "fnn-linear",
        "fnn-layernorm",
        "fnn-dropout",
        "fnn-residual",
        "rope",  # RoPE (Rotary Position Embeddings)
    ]

    for shader in generic_shaders:
        _registry.register_generic(shader, shader)

    # Register architecture-specific shaders
    # These are optimized variants for specific architectures
    _registry.register_architecture_specific("attention-output", "gpt", "attention-output-gpt")
    _registry.register_architecture_specific(
        "attention-output", "granite", "attention-output-gpt"
    )  # Granite uses GPT-style causal attention
    _registry.register_architecture_specific("attention-output", "t5", "attention-output-t5")

    # Note: BERT, DistilBERT, RoBERTa, MPNet, XLM-RoBERTa, ALBERT all use the generic
    # 'attention-output' shader since they share the same bidirectional attention pattern.
    # Architecture-specific shaders are only created when there's a significant
    # optimization opportunity (e.g., causal attention for GPT/Granite, cross-attention for T5).


# Initialize on import
_initialize_registry()
