"""Network architecture visualizer for the Grilly framework.

Renders neural network architectures as clean diagrams in both terminal-friendly
ASCII art and self-contained HTML/SVG output. Works with grilly.nn modules by
introspecting .parameters(), layer names, and input/output shapes.

Usage:
    from grilly.utils.visualizer import visualize

    # Terminal output (default)
    visualize(model)

    # HTML output
    visualize(model, output="html", save_path="model.html")

    # Return the rendered string
    text = visualize(model, output="ascii", return_str=True)
"""

from __future__ import annotations

import html as html_mod
import textwrap
from dataclasses import dataclass, field
from io import StringIO
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Layer type classification for coloring and grouping
_LAYER_CATEGORIES: dict[str, str] = {
    # Embeddings
    "Embedding": "embedding",
    "CapsuleEmbedding": "embedding",
    # Linear / projection
    "Linear": "linear",
    "LoRALinear": "linear",
    # Activations
    "ReLU": "activation",
    "GELU": "activation",
    "SiLU": "activation",
    "GCU": "activation",
    "RoSwish": "activation",
    "SwiGLU": "activation",
    "Softmax": "activation",
    "Softplus": "activation",
    # Normalization
    "LayerNorm": "norm",
    "BatchNorm1d": "norm",
    "BatchNorm2d": "norm",
    "RMSNorm": "norm",
    "SNNRMSNorm": "norm",
    "_NormAdapter": "norm",
    # Attention
    "MultiheadAttention": "attention",
    "FlashAttention2": "attention",
    "ProsodyModulatedAttention": "attention",
    # Recurrent / SSM
    "LSTM": "recurrent",
    "LSTMCell": "recurrent",
    "GRU": "recurrent",
    "GRUCell": "recurrent",
    "_SelectiveScanBlock": "ssm",
    # SNN
    "LIFNeuron": "snn",
    "GIFNeuron": "snn",
    "SNNLayer": "snn",
    "HebbianLayer": "snn",
    "STDPLayer": "snn",
    "Synapse": "snn",
    # Containers
    "Sequential": "container",
    "Residual": "container",
    # Memory
    "MemoryRead": "memory",
    "MemoryWrite": "memory",
    "MemoryInjectGate": "memory",
    "MemoryQueryPooling": "memory",
    # Capsule
    "CapsuleProject": "capsule",
    "DentateGyrus": "capsule",
    "SemanticEncoder": "capsule",
    # Routing / MoE
    "DomainRouter": "routing",
    "ExpertCombiner": "routing",
    # Convolutional
    "Conv1d": "conv",
    "Conv2d": "conv",
    # Pooling
    "MaxPool2d": "pooling",
    "AvgPool2d": "pooling",
    "AdaptiveMaxPool2d": "pooling",
    "AdaptiveAvgPool2d": "pooling",
    # Dropout
    "Dropout": "regularization",
    # Loss
    "MSELoss": "loss",
    "CrossEntropyLoss": "loss",
    "BCELoss": "loss",
    "ContrastiveLoss": "loss",
    # VSA head
    "VSAReasoningHead": "vsa",
    # NLMS
    "_NLMSResidualHead": "nlms",
    # Transformer
    "TransformerEncoderLayer": "transformer",
    "TransformerDecoderLayer": "transformer",
    # Multimodal
    "VisionLanguageModel": "multimodal",
    "PerceiverIO": "multimodal",
    "PerceiverResampler": "multimodal",
}

# Color palette for HTML/SVG mode
_CATEGORY_COLORS: dict[str, str] = {
    "embedding": "#4A90D9",     # blue
    "linear": "#5CB85C",        # green
    "activation": "#F0AD4E",    # orange
    "norm": "#9B59B6",          # purple
    "attention": "#E74C3C",     # red
    "recurrent": "#1ABC9C",     # teal
    "ssm": "#2C3E50",           # dark blue
    "snn": "#E67E22",           # dark orange
    "container": "#95A5A6",     # grey
    "memory": "#8E44AD",        # dark purple
    "capsule": "#16A085",       # dark teal
    "routing": "#D35400",       # rust
    "conv": "#27AE60",          # emerald
    "pooling": "#2980B9",       # bright blue
    "regularization": "#BDC3C7",# light grey
    "loss": "#C0392B",          # dark red
    "vsa": "#2ECC71",           # bright green
    "nlms": "#F39C12",          # amber
    "transformer": "#E74C3C",   # red
    "multimodal": "#3498DB",    # sky blue
    "unknown": "#7F8C8D",       # grey
}

_CATEGORY_LABELS: dict[str, str] = {
    "embedding": "EMB",
    "linear": "LIN",
    "activation": "ACT",
    "norm": "NRM",
    "attention": "ATT",
    "recurrent": "RNN",
    "ssm": "SSM",
    "snn": "SNN",
    "container": "SEQ",
    "memory": "MEM",
    "capsule": "CAP",
    "routing": "MoE",
    "conv": "CNN",
    "pooling": "POL",
    "regularization": "REG",
    "loss": "LOS",
    "vsa": "VSA",
    "nlms": "NLM",
    "transformer": "TRN",
    "multimodal": "MML",
    "unknown": "???",
}


# ---------------------------------------------------------------------------
# Layer info extraction
# ---------------------------------------------------------------------------

@dataclass
class LayerInfo:
    """Extracted metadata about a single layer."""
    name: str
    type_name: str
    category: str
    param_count: int
    param_shapes: list[tuple[str, tuple[int, ...]]]  # (param_name, shape)
    input_shape: str
    output_shape: str
    children: list[LayerInfo] = field(default_factory=list)
    extra_info: dict[str, str] = field(default_factory=dict)
    repeat_count: int = 1
    is_frozen: bool = False
    has_gpu: bool = False


def _classify_layer(type_name: str) -> str:
    """Return the category string for a layer type name."""
    return _LAYER_CATEGORIES.get(type_name, "unknown")


def _count_params(obj: Any) -> int:
    """Count total parameters in an object."""
    total = 0
    params = _get_parameters(obj)
    for p in params:
        arr = _to_array(p)
        if arr is not None:
            total += int(np.prod(arr.shape))
    return total


def _get_parameters(obj: Any) -> list[Any]:
    """Extract parameter list from various module types."""
    if hasattr(obj, "parameters"):
        result = obj.parameters()
        if hasattr(result, "__iter__") and not isinstance(result, np.ndarray):
            return list(result)
        return [result]
    return []


def _to_array(p: Any) -> np.ndarray | None:
    """Safely convert a parameter-like object to a numpy array for shape inspection."""
    if isinstance(p, np.ndarray):
        return p
    if hasattr(p, "data") and isinstance(getattr(p, "data", None), np.ndarray):
        return p.data
    if hasattr(p, "shape"):
        # Duck-type: has shape, likely array-like
        try:
            return np.asarray(p)
        except Exception:
            pass
    return None


def _get_param_shapes(obj: Any) -> list[tuple[str, tuple[int, ...]]]:
    """Extract named parameter shapes from an object."""
    shapes: list[tuple[str, tuple[int, ...]]] = []

    # Try named_parameters first (grilly.nn.Module interface)
    if hasattr(obj, "named_parameters"):
        try:
            for name, param in obj.named_parameters():
                arr = _to_array(param)
                if arr is not None:
                    shapes.append((name, tuple(arr.shape)))
        except Exception:
            pass
        if shapes:
            return shapes

    # Try _parameters dict
    if hasattr(obj, "_parameters") and isinstance(obj._parameters, dict):
        for name, param in obj._parameters.items():
            if param is not None:
                arr = _to_array(param)
                if arr is not None:
                    shapes.append((name, tuple(arr.shape)))

    # Fallback: look for weight and bias attributes directly
    if not shapes:
        for attr_name in ("weight", "bias", "decay_logits"):
            attr = getattr(obj, attr_name, None)
            if attr is not None:
                arr = _to_array(attr)
                if arr is not None:
                    shapes.append((attr_name, tuple(arr.shape)))

    return shapes


def _infer_shapes(obj: Any) -> tuple[str, str]:
    """Infer input and output shapes from layer attributes."""
    type_name = type(obj).__name__

    # Linear
    if hasattr(obj, "in_features") and hasattr(obj, "out_features"):
        return f"(..., {obj.in_features})", f"(..., {obj.out_features})"

    # Embedding
    if hasattr(obj, "num_embeddings") and hasattr(obj, "embedding_dim"):
        return "(B, T) int", f"(B, T, {obj.embedding_dim})"

    # LayerNorm / RMSNorm
    if hasattr(obj, "normalized_shape"):
        dim = obj.normalized_shape
        return f"(..., {dim})", f"(..., {dim})"

    # Conv layers
    if hasattr(obj, "in_channels") and hasattr(obj, "out_channels"):
        return f"(..., {obj.in_channels}, *)", f"(..., {obj.out_channels}, *)"

    # Attention
    if hasattr(obj, "embed_dim") and hasattr(obj, "num_heads"):
        d = obj.embed_dim
        return f"(B, T, {d})", f"(B, T, {d})"

    # _SelectiveScanBlock
    if hasattr(obj, "d_model") and type_name == "_SelectiveScanBlock":
        d = obj.d_model
        return f"(B, T, {d})", f"(B, T, {d})"

    # _NormAdapter
    if type_name == "_NormAdapter" and hasattr(obj, "layer"):
        inner = obj.layer
        if hasattr(inner, "normalized_shape"):
            dim = inner.normalized_shape
            return f"(..., {dim})", f"(..., {dim})"

    # LSTM/GRU
    if hasattr(obj, "input_size") and hasattr(obj, "hidden_size"):
        return f"(B, T, {obj.input_size})", f"(B, T, {obj.hidden_size})"

    # VSAReasoningHead
    if hasattr(obj, "d_model") and hasattr(obj, "vsa_dim"):
        return f"(B, {obj.d_model})", f"(B, {obj.vsa_dim})"

    # Default: activations and parameterless layers
    return "(...)", "(...)"


def _is_frozen(obj: Any) -> bool:
    """Check if a module has frozen (non-trainable) parameters."""
    params = _get_parameters(obj)
    if not params:
        return False
    for p in params:
        if hasattr(p, "requires_grad") and p.requires_grad:
            return False
    return True


def _has_gpu(obj: Any) -> bool:
    """Check if a module is in GPU mode."""
    if hasattr(obj, "_return_gpu_tensor") and obj._return_gpu_tensor:
        return True
    if hasattr(obj, "use_vulkan_tensor") and obj.use_vulkan_tensor:
        return True
    return False


# ---------------------------------------------------------------------------
# Model introspection: discover layers
# ---------------------------------------------------------------------------

def _discover_layers_grilly_ssm(model: Any) -> list[LayerInfo]:
    """Specialized introspection for GrillySSMLM architecture."""
    layers: list[LayerInfo] = []
    config = getattr(model, "config", None)
    d_model = getattr(config, "d_model", 768) if config else 768
    vocab_size = getattr(config, "vocab_size", 32768) if config else 32768
    n_layers = getattr(config, "n_layers", 12) if config else 12

    # Embedding
    tok_embed = getattr(model, "tok_embed", None)
    if tok_embed is not None:
        in_s, out_s = _infer_shapes(tok_embed)
        layers.append(LayerInfo(
            name="tok_embed",
            type_name="Embedding",
            category="embedding",
            param_count=_count_params(tok_embed),
            param_shapes=_get_param_shapes(tok_embed),
            input_shape=in_s,
            output_shape=out_s,
            has_gpu=_has_gpu(tok_embed),
            extra_info={"vocab_size": str(vocab_size), "d_model": str(d_model)},
        ))

    # SSM blocks
    blocks = getattr(model, "blocks", [])
    if blocks:
        # Inspect the first block for children detail
        block = blocks[0]
        block_children: list[LayerInfo] = []

        # Norm inside block
        block_norm = getattr(block, "norm", None)
        if block_norm is not None:
            norm_layer = getattr(block_norm, "layer", block_norm)
            norm_type = type(norm_layer).__name__
            uses_snn = getattr(block_norm, "uses_snn", False)
            label = "SNNRMSNorm" if uses_snn else norm_type
            dim_val = getattr(norm_layer, "normalized_shape", d_model)
            block_children.append(LayerInfo(
                name="norm",
                type_name=label,
                category="norm",
                param_count=_count_params(block_norm),
                param_shapes=_get_param_shapes(block_norm),
                input_shape=f"(B, T, {d_model})",
                output_shape=f"(B, T, {d_model})",
                extra_info={"dim": str(dim_val)},
            ))

        # in_proj
        in_proj = getattr(block, "in_proj", None)
        if in_proj is not None:
            in_s, out_s = _infer_shapes(in_proj)
            block_children.append(LayerInfo(
                name="in_proj",
                type_name="Linear",
                category="linear",
                param_count=_count_params(in_proj),
                param_shapes=_get_param_shapes(in_proj),
                input_shape=f"(B, T, {d_model})",
                output_shape=f"(B, T, {2 * d_model})",
                has_gpu=_has_gpu(in_proj),
            ))

        # Gate + value split (not a parameterized layer)
        gate_act = getattr(block, "gate_activation", "sigmoid_tanh")
        block_children.append(LayerInfo(
            name="gate_value_split",
            type_name="Split",
            category="activation",
            param_count=0,
            param_shapes=[],
            input_shape=f"(B, T, {2 * d_model})",
            output_shape=f"gate: (B,T,{d_model}), value: (B,T,{d_model})",
            extra_info={"activation": gate_act},
        ))

        # Selective scan
        block_children.append(LayerInfo(
            name="selective_scan",
            type_name="SelectiveScan",
            category="ssm",
            param_count=d_model,  # decay_logits
            param_shapes=[("decay_logits", (d_model,))],
            input_shape=f"gate, value: (B, T, {d_model})",
            output_shape=f"(B, T, {d_model})",
            extra_info={"scan_impl": getattr(block, "scan_impl", "vectorized")},
        ))

        # out_proj
        out_proj = getattr(block, "out_proj", None)
        if out_proj is not None:
            block_children.append(LayerInfo(
                name="out_proj",
                type_name="Linear",
                category="linear",
                param_count=_count_params(out_proj),
                param_shapes=_get_param_shapes(out_proj),
                input_shape=f"(B, T, {d_model})",
                output_shape=f"(B, T, {d_model})",
                has_gpu=_has_gpu(out_proj),
            ))

        # Residual add
        block_children.append(LayerInfo(
            name="residual",
            type_name="ResidualAdd",
            category="container",
            param_count=0,
            param_shapes=[],
            input_shape=f"x, scan_out: (B, T, {d_model})",
            output_shape=f"(B, T, {d_model})",
        ))

        total_block_params = sum(c.param_count for c in block_children)
        total_all_blocks = total_block_params * n_layers

        layers.append(LayerInfo(
            name="blocks",
            type_name="_SelectiveScanBlock",
            category="ssm",
            param_count=total_all_blocks,
            param_shapes=[],
            input_shape=f"(B, T, {d_model})",
            output_shape=f"(B, T, {d_model})",
            children=block_children,
            repeat_count=n_layers,
            has_gpu=_has_gpu(block),
            extra_info={"per_block_params": str(total_block_params)},
        ))

    # Final norm
    final_norm = getattr(model, "norm", None)
    if final_norm is not None:
        norm_layer = getattr(final_norm, "layer", final_norm)
        norm_type = type(norm_layer).__name__
        uses_snn = getattr(final_norm, "uses_snn", False)
        label = "SNNRMSNorm" if uses_snn else norm_type
        layers.append(LayerInfo(
            name="norm",
            type_name=label,
            category="norm",
            param_count=_count_params(final_norm),
            param_shapes=_get_param_shapes(final_norm),
            input_shape=f"(B, T, {d_model})",
            output_shape=f"(B, T, {d_model})",
        ))

    # Branching heads: LM head and optional VSA head
    lm_head = getattr(model, "lm_head", None)
    vsa_head = getattr(model, "vsa_head", None)
    use_vsa = getattr(model, "use_vsa_head", False) and vsa_head is not None

    if lm_head is not None:
        layers.append(LayerInfo(
            name="lm_head",
            type_name="Linear",
            category="linear",
            param_count=_count_params(lm_head),
            param_shapes=_get_param_shapes(lm_head),
            input_shape=f"(B, T, {d_model})",
            output_shape=f"(B, T, {vocab_size})",
            has_gpu=_has_gpu(lm_head),
            extra_info={"role": "LM Head"},
        ))

    if use_vsa and vsa_head is not None:
        vsa_dim = getattr(vsa_head, "vsa_dim", 10000)
        layers.append(LayerInfo(
            name="vsa_head",
            type_name="VSAReasoningHead",
            category="vsa",
            param_count=_count_params(vsa_head),
            param_shapes=_get_param_shapes(vsa_head),
            input_shape=f"(B, {d_model})",
            output_shape=f"(B, {vsa_dim})",
            has_gpu=_has_gpu(vsa_head) if hasattr(vsa_head, "use_vulkan_tensor") else False,
            extra_info={
                "role": "VSA Head",
                "pipeline": f"Linear {d_model}->{vsa_dim} -> tanh -> sign",
            },
        ))

    # NLMS head (non-parametric, metadata only)
    nlms = getattr(model, "nlms_head", None)
    if nlms is not None and getattr(nlms, "enabled", False):
        layers.append(LayerInfo(
            name="nlms_head",
            type_name="_NLMSResidualHead",
            category="nlms",
            param_count=0,
            param_shapes=[],
            input_shape=f"hidden: (T, {d_model}), logits: (T, {vocab_size})",
            output_shape=f"logits: (T, {vocab_size})",
            extra_info={
                "topk": str(getattr(nlms, "topk", "?")),
                "scale": str(getattr(nlms, "scale", "?")),
                "role": "NLMS Residual",
            },
        ))

    return layers


def _discover_layers_generic(model: Any, name: str = "model") -> list[LayerInfo]:
    """Generic layer discovery via attribute introspection."""
    layers: list[LayerInfo] = []

    # If it has _modules (grilly.nn.Module), iterate those
    if hasattr(model, "_modules") and isinstance(model._modules, dict):
        for mod_name, mod in model._modules.items():
            if mod is None:
                continue
            type_name = type(mod).__name__
            category = _classify_layer(type_name)
            in_s, out_s = _infer_shapes(mod)
            children = []
            if hasattr(mod, "_modules") and mod._modules:
                children = _discover_layers_generic(mod, mod_name)
            layers.append(LayerInfo(
                name=mod_name,
                type_name=type_name,
                category=category,
                param_count=_count_params(mod),
                param_shapes=_get_param_shapes(mod),
                input_shape=in_s,
                output_shape=out_s,
                children=children,
                is_frozen=_is_frozen(mod),
                has_gpu=_has_gpu(mod),
            ))
        return layers

    # Fallback: scan all attributes for module-like objects
    for attr_name in sorted(dir(model)):
        if attr_name.startswith("_"):
            continue
        try:
            attr = getattr(model, attr_name)
        except Exception:
            continue
        if attr is None or callable(attr) and not hasattr(attr, "parameters"):
            continue
        if hasattr(attr, "parameters") and not isinstance(attr, (int, float, str, bool, np.ndarray)):
            type_name = type(attr).__name__
            if type_name in ("method", "builtin_function_or_method", "function"):
                continue
            category = _classify_layer(type_name)
            in_s, out_s = _infer_shapes(attr)
            layers.append(LayerInfo(
                name=attr_name,
                type_name=type_name,
                category=category,
                param_count=_count_params(attr),
                param_shapes=_get_param_shapes(attr),
                input_shape=in_s,
                output_shape=out_s,
                is_frozen=_is_frozen(attr),
                has_gpu=_has_gpu(attr),
            ))

    # Also handle list attributes (like blocks)
    for attr_name in sorted(dir(model)):
        if attr_name.startswith("_"):
            continue
        try:
            attr = getattr(model, attr_name)
        except Exception:
            continue
        if isinstance(attr, (list, tuple)) and len(attr) > 0:
            first = attr[0]
            if hasattr(first, "parameters"):
                type_name = type(first).__name__
                category = _classify_layer(type_name)
                in_s, out_s = _infer_shapes(first)
                per_elem_params = _count_params(first)
                layers.append(LayerInfo(
                    name=attr_name,
                    type_name=type_name,
                    category=category,
                    param_count=per_elem_params * len(attr),
                    param_shapes=_get_param_shapes(first),
                    input_shape=in_s,
                    output_shape=out_s,
                    repeat_count=len(attr),
                    has_gpu=_has_gpu(first),
                ))

    return layers


def discover_layers(model: Any) -> list[LayerInfo]:
    """Auto-discover layers from a model, using specialized logic when available."""
    type_name = type(model).__name__

    # GrillySSMLM specialization
    if type_name == "GrillySSMLM" or (
        hasattr(model, "blocks")
        and hasattr(model, "tok_embed")
        and hasattr(model, "lm_head")
    ):
        return _discover_layers_grilly_ssm(model)

    return _discover_layers_generic(model)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_param_count(n: int) -> str:
    """Format a parameter count as a human-readable string."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _format_memory(n_params: int, bytes_per_param: int = 4) -> str:
    """Format memory footprint."""
    total_bytes = n_params * bytes_per_param
    if total_bytes >= 1_073_741_824:
        return f"{total_bytes / 1_073_741_824:.1f} GB"
    if total_bytes >= 1_048_576:
        return f"{total_bytes / 1_048_576:.1f} MB"
    if total_bytes >= 1_024:
        return f"{total_bytes / 1_024:.1f} KB"
    return f"{total_bytes} B"


# ---------------------------------------------------------------------------
# ASCII / Terminal renderer
# ---------------------------------------------------------------------------

_BOX_TL = "\u250c"   # top-left
_BOX_TR = "\u2510"   # top-right
_BOX_BL = "\u2514"   # bottom-left
_BOX_BR = "\u2518"   # bottom-right
_BOX_H = "\u2500"    # horizontal
_BOX_V = "\u2502"    # vertical
_BOX_TJ = "\u252c"   # top junction
_BOX_BJ = "\u2534"   # bottom junction
_BOX_LJ = "\u251c"   # left junction
_ARROW_DOWN = "\u25bc"
_TREE_BRANCH = "\u251c\u2500 "
_TREE_LAST = "\u2514\u2500 "


def _draw_box(lines: list[str], width: int) -> list[str]:
    """Wrap lines in a box of the given width (content width, excludes borders)."""
    result = []
    result.append(f"  {_BOX_TL}{_BOX_H * (width + 2)}{_BOX_TR}")
    for line in lines:
        padded = line.ljust(width)
        result.append(f"  {_BOX_V} {padded} {_BOX_V}")
    result.append(f"  {_BOX_BL}{_BOX_H * (width + 2)}{_BOX_BR}")
    return result


def _draw_connector(label: str, total_width: int) -> list[str]:
    """Draw a downward connector with a shape annotation."""
    mid = total_width // 2
    lines = []
    line1 = " " * (mid + 2) + _BOX_V
    lines.append(line1)
    if label:
        # Center the label
        label_str = f" {label}"
        line2 = " " * (mid + 2) + _BOX_V + label_str
        lines.append(line2)
    line3 = " " * (mid + 2) + _ARROW_DOWN
    lines.append(line3)
    return lines


def _draw_branch(labels: list[tuple[str, list[str]]], total_width: int) -> list[str]:
    """Draw a branching point with multiple output paths."""
    mid = total_width // 2
    result = []
    result.append(" " * (mid + 2) + _BOX_V)

    n = len(labels)
    if n < 2:
        return result

    # Compute positions for each branch
    segment = total_width // (n + 1)
    positions = [(i + 1) * segment for i in range(n)]

    # Draw the horizontal branch line
    branch_line = list(" " * (total_width + 6))
    min_pos = positions[0] + 2
    max_pos = positions[-1] + 2
    for i in range(min_pos, max_pos + 1):
        if i < len(branch_line):
            branch_line[i] = _BOX_H
    # Place the junction from the top
    center = mid + 2
    if center < len(branch_line):
        branch_line[center] = _BOX_BJ
    for pos in positions:
        p = pos + 2
        if p < len(branch_line):
            branch_line[p] = _BOX_TJ
    result.append("".join(branch_line))

    # Arrows down from each position
    arrow_line = list(" " * (total_width + 6))
    for pos in positions:
        p = pos + 2
        if p < len(arrow_line):
            arrow_line[p] = _ARROW_DOWN
    result.append("".join(arrow_line))

    # Draw boxes side by side
    box_width = max(12, segment - 4)
    all_box_lines: list[list[str]] = []
    max_box_height = 0

    for _, box_content in labels:
        box = _draw_box(box_content, box_width)
        all_box_lines.append(box)
        max_box_height = max(max_box_height, len(box))

    # Pad all boxes to same height
    for box in all_box_lines:
        while len(box) < max_box_height:
            box.append(" " * (box_width + 6))

    # Merge boxes horizontally
    for row_idx in range(max_box_height):
        merged = ""
        for col_idx, box in enumerate(all_box_lines):
            line = box[row_idx] if row_idx < len(box) else ""
            target_start = positions[col_idx] - box_width // 2
            # Pad to position
            needed = max(0, target_start - len(merged))
            merged += " " * needed + line.strip()
        result.append(merged)

    return result


def render_ascii(
    model: Any,
    layers: list[LayerInfo] | None = None,
) -> str:
    """Render a model architecture as terminal-friendly ASCII art."""
    if layers is None:
        layers = discover_layers(model)

    total_params = sum(layer.param_count for layer in layers)
    model_name = type(model).__name__

    out = StringIO()
    content_width = 45
    outer_width = content_width + 8

    # Header
    header = f"  {model_name} ({_format_param_count(total_params)} params, {_format_memory(total_params)})"
    out.write(f"{_BOX_TL}{_BOX_H * (outer_width)}{_BOX_TR}\n")
    out.write(f"{_BOX_V}{header.ljust(outer_width)}{_BOX_V}\n")
    out.write(f"{_BOX_LJ}{_BOX_H * (outer_width)}{_BOX_V}\n")
    out.write(f"{_BOX_V}{' ' * outer_width}{_BOX_V}\n")

    # Detect branching (LM head + VSA head at the end)
    main_layers = []
    branch_layers = []
    for layer in layers:
        if layer.extra_info.get("role") in ("LM Head", "VSA Head", "NLMS Residual"):
            branch_layers.append(layer)
        else:
            main_layers.append(layer)

    # Render main sequential layers
    for i, layer in enumerate(main_layers):
        # Build box content
        box_lines = []
        repeat_str = f" x {layer.repeat_count}" if layer.repeat_count > 1 else ""
        type_label = layer.type_name
        if layer.category == "norm" and layer.type_name not in ("LayerNorm", "BatchNorm1d", "BatchNorm2d"):
            type_label = layer.type_name

        title = f"{type_label}{repeat_str}"

        # First line: type and shape info
        if layer.param_shapes:
            ", ".join(
                f"{s[0]}x{s[1]}" if len(s) == 2 else str(s)
                for _, s in layer.param_shapes[:2]
            )
            title_line = f"{title}"
            if len(title_line) < content_width:
                box_lines.append(title_line)
            else:
                box_lines.append(title_line[:content_width])
        else:
            box_lines.append(title)

        # Children (for SSM blocks)
        if layer.children:
            for ci, child in enumerate(layer.children):
                prefix = _TREE_BRANCH if ci < len(layer.children) - 1 else _TREE_LAST
                child_info = child.name
                if child.type_name == "Linear" and child.param_shapes:
                    for _, shape in child.param_shapes:
                        if len(shape) == 2:
                            child_info = f"{child.name} ({shape[1]} -> {shape[0]})"
                            break
                elif child.type_name in ("SelectiveScan",):
                    child_info = f"selective_scan ({layer.extra_info.get('per_block_params', '?')} features)"
                elif child.type_name == "Split":
                    child_info = f"gate + value split ({child.extra_info.get('activation', '?')})"
                elif child.type_name == "ResidualAdd":
                    child_info = "residual add"
                else:
                    dims = child.extra_info.get("dim", "")
                    if dims:
                        child_info = f"{child.type_name} ({dims})"
                    else:
                        child_info = child.type_name
                box_lines.append(f"{prefix}{child_info}")

        # Param count
        if layer.param_count > 0:
            count_str = f"{_format_param_count(layer.param_count)} params"
            if layer.is_frozen:
                count_str += " [frozen]"
            if layer.has_gpu:
                count_str += " [GPU]"
            box_lines.append(count_str)

        # Extra info
        for key, val in layer.extra_info.items():
            if key in ("role", "per_block_params"):
                continue
            box_lines.append(f"{key}: {val}")

        # Draw the box
        box_w = max(len(line) for line in box_lines) + 2
        box_w = max(box_w, 30)
        box_w = min(box_w, content_width)
        box = _draw_box(box_lines, box_w)

        for line in box:
            out.write(f"{_BOX_V}{line.ljust(outer_width)}{_BOX_V}\n")

        # Draw connector to next layer (except before branch)
        is_last_main = (i == len(main_layers) - 1)
        if not is_last_main:
            connector = _draw_connector(layer.output_shape, box_w)
            for line in connector:
                out.write(f"{_BOX_V}{line.ljust(outer_width)}{_BOX_V}\n")
        elif branch_layers:
            # Draw connector then branch
            connector = _draw_connector(layer.output_shape, box_w)
            for line in connector:
                out.write(f"{_BOX_V}{line.ljust(outer_width)}{_BOX_V}\n")

    # Render branch layers
    if len(branch_layers) >= 2:
        branch_data: list[tuple[str, list[str]]] = []
        for bl in branch_layers:
            box_lines = []
            box_lines.append(bl.extra_info.get("role", bl.type_name))

            # Shape info
            if bl.type_name == "Linear" and bl.param_shapes:
                for _, shape in bl.param_shapes:
                    if len(shape) == 2:
                        box_lines.append(f"{shape[1]}->{shape[0]}")
                        break
            elif "pipeline" in bl.extra_info:
                pipeline_str = bl.extra_info["pipeline"]
                # Wrap long pipeline strings
                if len(pipeline_str) > 20:
                    parts = pipeline_str.split(" -> ")
                    for part in parts:
                        box_lines.append(part.strip())
                else:
                    box_lines.append(pipeline_str)

            if bl.param_count > 0:
                box_lines.append(f"{_format_param_count(bl.param_count)}")

            branch_data.append((bl.name, box_lines))

        branch_lines = _draw_branch(branch_data, content_width)
        for line in branch_lines:
            out.write(f"{_BOX_V}{line.ljust(outer_width)}{_BOX_V}\n")

    elif len(branch_layers) == 1:
        # Single head, no branching
        bl = branch_layers[0]
        box_lines = []
        box_lines.append(bl.extra_info.get("role", bl.type_name))
        if bl.param_shapes:
            for _, shape in bl.param_shapes:
                if len(shape) == 2:
                    box_lines.append(f"{shape[1]}->{shape[0]}")
                    break
        if bl.param_count > 0:
            box_lines.append(f"{_format_param_count(bl.param_count)} params")
        box_w = max(len(line) for line in box_lines) + 2
        box_w = max(box_w, 20)
        box = _draw_box(box_lines, box_w)
        for line in box:
            out.write(f"{_BOX_V}{line.ljust(outer_width)}{_BOX_V}\n")

    # Footer with summary
    out.write(f"{_BOX_V}{' ' * outer_width}{_BOX_V}\n")
    summary_line = f"  Total: {_format_param_count(total_params)} params | {_format_memory(total_params)}"
    out.write(f"{_BOX_V}{summary_line.ljust(outer_width)}{_BOX_V}\n")

    # Layer breakdown
    param_by_category: dict[str, int] = {}
    for layer in layers:
        cat = layer.category
        param_by_category[cat] = param_by_category.get(cat, 0) + layer.param_count

    breakdown_parts = []
    for cat, count in sorted(param_by_category.items(), key=lambda x: -x[1]):
        if count > 0:
            label = _CATEGORY_LABELS.get(cat, cat[:3].upper())
            breakdown_parts.append(f"{label}:{_format_param_count(count)}")

    if breakdown_parts:
        breakdown = "  " + " | ".join(breakdown_parts[:5])
        out.write(f"{_BOX_V}{breakdown.ljust(outer_width)}{_BOX_V}\n")

    out.write(f"{_BOX_BL}{_BOX_H * (outer_width)}{_BOX_BR}\n")

    return out.getvalue()


# ---------------------------------------------------------------------------
# HTML / SVG renderer
# ---------------------------------------------------------------------------

def _svg_rect(x: float, y: float, w: float, h: float, fill: str,
              rx: float = 8, stroke: str = "#333", stroke_width: float = 1.5) -> str:
    """Generate an SVG rect element."""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
        f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
    )


def _svg_text(x: float, y: float, text: str, font_size: int = 14,
              fill: str = "#fff", anchor: str = "middle",
              weight: str = "normal", font_family: str = "monospace") -> str:
    """Generate an SVG text element."""
    escaped = html_mod.escape(text)
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-size="{font_size}" fill="{fill}" font-weight="{weight}" '
        f'font-family="{font_family}">{escaped}</text>'
    )


def _svg_line(x1: float, y1: float, x2: float, y2: float,
              stroke: str = "#555", width: float = 2) -> str:
    """Generate an SVG line element."""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{stroke}" stroke-width="{width}"/>'
    )


def _svg_arrow(x1: float, y1: float, x2: float, y2: float,
               stroke: str = "#555", width: float = 2) -> str:
    """Generate an SVG line with arrowhead."""
    arrow_size = 8
    elements = [_svg_line(x1, y1, x2, y2 - arrow_size, stroke, width)]
    # Triangle arrowhead
    elements.append(
        f'<polygon points="{x2},{y2} {x2-arrow_size/2},{y2-arrow_size} '
        f'{x2+arrow_size/2},{y2-arrow_size}" fill="{stroke}"/>'
    )
    return "\n".join(elements)


def render_html(
    model: Any,
    layers: list[LayerInfo] | None = None,
    title: str | None = None,
) -> str:
    """Render a model architecture as a self-contained HTML file with inline SVG."""
    if layers is None:
        layers = discover_layers(model)

    total_params = sum(layer.param_count for layer in layers)
    model_name = title or type(model).__name__

    # Layout constants
    canvas_width = 900
    box_width = 340
    box_x = (canvas_width - box_width) / 2
    box_padding = 16
    line_height = 20
    box_gap = 50
    child_indent = 20
    child_line_height = 18

    # Separate main layers and branch layers
    main_layers = []
    branch_layers = []
    for layer in layers:
        if layer.extra_info.get("role") in ("LM Head", "VSA Head", "NLMS Residual"):
            branch_layers.append(layer)
        else:
            main_layers.append(layer)

    # First pass: compute heights
    def _box_height(layer: LayerInfo) -> float:
        n_lines = 1  # title line
        if layer.children:
            n_lines += len(layer.children)
        if layer.param_count > 0:
            n_lines += 1  # param count
        for key in layer.extra_info:
            if key not in ("role", "per_block_params", "pipeline"):
                n_lines += 1
        return box_padding * 2 + n_lines * line_height

    y_cursor = 80.0  # start below title
    positions: list[tuple[float, float, float]] = []  # (x, y, height) for main layers

    for layer in main_layers:
        h = _box_height(layer)
        positions.append((box_x, y_cursor, h))
        y_cursor += h + box_gap

    # Branch positions
    branch_y = y_cursor
    branch_height = 0.0
    branch_positions: list[tuple[float, float, float]] = []
    if branch_layers:
        n_branches = len(branch_layers)
        branch_box_width = min(box_width, 240)
        total_branch_width = n_branches * branch_box_width + (n_branches - 1) * 40
        start_x = (canvas_width - total_branch_width) / 2
        for i, bl in enumerate(branch_layers):
            bx = start_x + i * (branch_box_width + 40)
            bh = _box_height(bl) + (
                len(bl.extra_info.get("pipeline", "").split(" -> ")) * child_line_height
                if "pipeline" in bl.extra_info else 0
            )
            branch_positions.append((bx, branch_y + 30, bh))
            branch_height = max(branch_height, bh)

    total_height = y_cursor + branch_height + 120  # extra space for summary

    # Build SVG elements
    svg_elements: list[str] = []

    # Background
    svg_elements.append(
        f'<rect width="{canvas_width}" height="{total_height}" '
        f'fill="#1a1a2e" rx="12"/>'
    )

    # Title
    svg_elements.append(_svg_text(
        canvas_width / 2, 35, model_name,
        font_size=22, fill="#e0e0e0", weight="bold",
    ))
    svg_elements.append(_svg_text(
        canvas_width / 2, 58,
        f"{_format_param_count(total_params)} params | {_format_memory(total_params)}",
        font_size=14, fill="#aaa",
    ))

    # Render main layers
    for idx, layer in enumerate(main_layers):
        x, y, h = positions[idx]
        color = _CATEGORY_COLORS.get(layer.category, _CATEGORY_COLORS["unknown"])

        # Box
        svg_elements.append(_svg_rect(x, y, box_width, h, color, rx=10))

        # Title
        repeat_str = f" x {layer.repeat_count}" if layer.repeat_count > 1 else ""
        title_text = f"{layer.type_name}{repeat_str}"
        ty = y + box_padding + 14
        svg_elements.append(_svg_text(
            x + box_width / 2, ty, title_text,
            font_size=16, fill="#fff", weight="bold",
        ))
        ty += line_height

        # Children
        if layer.children:
            for child in layer.children:
                child_text = ""
                if child.type_name == "Linear" and child.param_shapes:
                    for _, shape in child.param_shapes:
                        if len(shape) == 2:
                            child_text = f"{child.name} ({shape[1]} -> {shape[0]})"
                            break
                elif child.type_name == "Split":
                    child_text = "gate + value split"
                elif child.type_name == "SelectiveScan":
                    child_text = "selective_scan"
                elif child.type_name == "ResidualAdd":
                    child_text = "residual add"
                else:
                    dims = child.extra_info.get("dim", "")
                    child_text = f"{child.type_name}({dims})" if dims else child.type_name

                svg_elements.append(_svg_text(
                    x + child_indent + 10, ty, f"  {child_text}",
                    font_size=12, fill="#ddd", anchor="start",
                ))
                ty += child_line_height

        # Param count
        if layer.param_count > 0:
            count_str = f"{_format_param_count(layer.param_count)} params"
            flags = []
            if layer.is_frozen:
                flags.append("frozen")
            if layer.has_gpu:
                flags.append("GPU")
            if flags:
                count_str += f" [{', '.join(flags)}]"
            svg_elements.append(_svg_text(
                x + box_width / 2, ty, count_str,
                font_size=12, fill="#ccc",
            ))
            ty += line_height

        # Connector arrow to next layer
        if idx < len(main_layers) - 1:
            next_y = positions[idx + 1][1]
            cx = x + box_width / 2
            svg_elements.append(_svg_arrow(cx, y + h, cx, next_y, stroke="#888"))

            # Shape label on connector
            mid_y = (y + h + next_y) / 2
            svg_elements.append(_svg_text(
                cx + box_width / 2 + 20, mid_y, layer.output_shape,
                font_size=11, fill="#999", anchor="start",
            ))
        elif branch_layers:
            # Connector to branching point
            cx = x + box_width / 2
            fork_y = y + h + 15
            svg_elements.append(_svg_line(cx, y + h, cx, fork_y, stroke="#888"))

            # Shape label
            mid_y = y + h + 8
            svg_elements.append(_svg_text(
                cx + box_width / 2 + 20, mid_y, layer.output_shape,
                font_size=11, fill="#999", anchor="start",
            ))

            # Horizontal fork line
            if len(branch_positions) > 0:
                min(bp[0] + (branch_box_width if len(branch_layers) > 1 else box_width) / 2
                             for bp, branch_box_width_dummy in
                             [(bp, min(box_width, 240)) for bp in branch_positions])
                max(bp[0] + min(box_width, 240) / 2 for bp in branch_positions)
                branch_box_width = min(box_width, 240)

                fork_y2 = branch_y + 10
                svg_elements.append(_svg_line(cx, fork_y, cx, fork_y2, stroke="#888"))

                if len(branch_positions) > 1:
                    left_cx = branch_positions[0][0] + branch_box_width / 2
                    right_cx = branch_positions[-1][0] + branch_box_width / 2
                    svg_elements.append(_svg_line(
                        left_cx, fork_y2, right_cx, fork_y2, stroke="#888",
                    ))

                for bp in branch_positions:
                    bcx = bp[0] + branch_box_width / 2
                    svg_elements.append(_svg_arrow(
                        bcx, fork_y2, bcx, bp[1], stroke="#888",
                    ))

    # Render branch layers
    branch_box_width = min(box_width, 240)
    for idx, bl in enumerate(branch_layers):
        if idx >= len(branch_positions):
            break
        bx, by, bh = branch_positions[idx]
        color = _CATEGORY_COLORS.get(bl.category, _CATEGORY_COLORS["unknown"])

        svg_elements.append(_svg_rect(bx, by, branch_box_width, bh, color, rx=10))

        ty = by + box_padding + 14
        role_text = bl.extra_info.get("role", bl.type_name)
        svg_elements.append(_svg_text(
            bx + branch_box_width / 2, ty, role_text,
            font_size=14, fill="#fff", weight="bold",
        ))
        ty += line_height

        # Shape info
        if bl.type_name == "Linear" and bl.param_shapes:
            for _, shape in bl.param_shapes:
                if len(shape) == 2:
                    svg_elements.append(_svg_text(
                        bx + branch_box_width / 2, ty, f"{shape[1]} -> {shape[0]}",
                        font_size=12, fill="#ddd",
                    ))
                    ty += child_line_height
                    break
        elif "pipeline" in bl.extra_info:
            for part in bl.extra_info["pipeline"].split(" -> "):
                svg_elements.append(_svg_text(
                    bx + branch_box_width / 2, ty, part.strip(),
                    font_size=12, fill="#ddd",
                ))
                ty += child_line_height

        if bl.param_count > 0:
            svg_elements.append(_svg_text(
                bx + branch_box_width / 2, ty,
                f"{_format_param_count(bl.param_count)} params",
                font_size=12, fill="#ccc",
            ))

    # Summary bar at the bottom
    summary_y = total_height - 55
    svg_elements.append(_svg_rect(
        30, summary_y, canvas_width - 60, 40,
        "#16213e", rx=8, stroke="#444",
    ))

    # Layer breakdown
    param_by_category: dict[str, int] = {}
    for layer in layers:
        cat = layer.category
        param_by_category[cat] = param_by_category.get(cat, 0) + layer.param_count

    breakdown_parts = []
    for cat, count in sorted(param_by_category.items(), key=lambda x: -x[1]):
        if count > 0:
            label = _CATEGORY_LABELS.get(cat, cat[:3].upper())
            breakdown_parts.append(f"{label}: {_format_param_count(count)}")

    summary_text = " | ".join(breakdown_parts[:6])
    svg_elements.append(_svg_text(
        canvas_width / 2, summary_y + 25, summary_text,
        font_size=13, fill="#bbb",
    ))

    # Legend
    legend_y = total_height - 15
    legend_categories = list(set(layer.category for layer in layers))
    legend_x_start = 40
    lx = legend_x_start
    for cat in sorted(legend_categories):
        color = _CATEGORY_COLORS.get(cat, _CATEGORY_COLORS["unknown"])
        label = _CATEGORY_LABELS.get(cat, cat[:3].upper())
        svg_elements.append(
            f'<rect x="{lx}" y="{legend_y - 10}" width="12" height="12" '
            f'rx="3" fill="{color}"/>'
        )
        svg_elements.append(_svg_text(
            lx + 18, legend_y, label,
            font_size=11, fill="#888", anchor="start",
        ))
        lx += len(label) * 8 + 30

    # Assemble SVG
    svg_content = "\n".join(svg_elements)
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {canvas_width} {total_height}">\n{svg_content}\n</svg>'

    # Wrap in HTML
    html = textwrap.dedent(f"""\
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{html_mod.escape(model_name)} - Architecture Diagram</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                background: #0f0f23;
                color: #e0e0e0;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 20px;
                min-height: 100vh;
            }}
            h1 {{
                font-size: 1.8rem;
                margin-bottom: 8px;
                color: #fff;
            }}
            .subtitle {{
                font-size: 0.95rem;
                color: #888;
                margin-bottom: 20px;
            }}
            .diagram-container {{
                max-width: 960px;
                width: 100%;
                overflow-x: auto;
            }}
            svg {{
                width: 100%;
                height: auto;
            }}
            .info-table {{
                margin-top: 24px;
                border-collapse: collapse;
                width: 100%;
                max-width: 800px;
            }}
            .info-table th, .info-table td {{
                padding: 8px 14px;
                text-align: left;
                border-bottom: 1px solid #333;
                font-size: 0.9rem;
            }}
            .info-table th {{
                color: #aaa;
                font-weight: 600;
            }}
            .info-table td {{
                color: #ddd;
            }}
            .info-table tr:hover td {{
                background: #1a1a2e;
            }}
            .category-badge {{
                display: inline-block;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 0.75rem;
                font-weight: 600;
                color: #fff;
            }}
        </style>
    </head>
    <body>
        <h1>{html_mod.escape(model_name)}</h1>
        <p class="subtitle">{_format_param_count(total_params)} parameters | {_format_memory(total_params)} (fp32)</p>

        <div class="diagram-container">
            {svg}
        </div>

        <table class="info-table">
            <thead>
                <tr>
                    <th>Layer</th>
                    <th>Type</th>
                    <th>Category</th>
                    <th>Input Shape</th>
                    <th>Output Shape</th>
                    <th>Params</th>
                    <th>Memory</th>
                </tr>
            </thead>
            <tbody>
    """)

    for layer in layers:
        color = _CATEGORY_COLORS.get(layer.category, _CATEGORY_COLORS["unknown"])
        cat_label = _CATEGORY_LABELS.get(layer.category, layer.category)
        repeat_str = f" x {layer.repeat_count}" if layer.repeat_count > 1 else ""
        flags = []
        if layer.is_frozen:
            flags.append("frozen")
        if layer.has_gpu:
            flags.append("GPU")
        flag_str = f' [{", ".join(flags)}]' if flags else ""

        html += f"""\
                <tr>
                    <td>{html_mod.escape(layer.name)}</td>
                    <td>{html_mod.escape(layer.type_name)}{html_mod.escape(repeat_str)}</td>
                    <td><span class="category-badge" style="background:{color}">{html_mod.escape(cat_label)}</span></td>
                    <td><code>{html_mod.escape(layer.input_shape)}</code></td>
                    <td><code>{html_mod.escape(layer.output_shape)}</code></td>
                    <td>{_format_param_count(layer.param_count)}{html_mod.escape(flag_str)}</td>
                    <td>{_format_memory(layer.param_count)}</td>
                </tr>
    """

    html += textwrap.dedent("""\
            </tbody>
        </table>
    </body>
    </html>
    """)

    return html


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def visualize(
    model: Any,
    *,
    output: str = "ascii",
    save_path: str | None = None,
    return_str: bool = False,
    title: str | None = None,
) -> str | None:
    """Visualize a neural network architecture.

    Works with any object that exposes a grilly.nn-compatible interface:
    - .parameters() -> iterable of arrays
    - Nested module attributes (weight, bias, _modules, etc.)
    - Known layer types (Linear, Embedding, etc.)

    Also provides specialized rendering for GrillySSMLM architectures,
    showing the full SSM block internal structure, branching heads, and
    VSA reasoning pipeline.

    Args:
        model: The model / module to visualize. Can be a grilly.nn.Module,
               a GrillySSMLM, or any object with discoverable layers.
        output: Rendering mode. One of:
            - "ascii" (default): Terminal-friendly box-drawing characters.
            - "html": Self-contained HTML file with inline SVG diagram
                      and a detailed layer table. Color-coded by layer type.
            - "svg": Raw SVG markup (no HTML wrapper).
        save_path: If provided, write the output to this file path.
        return_str: If True, return the rendered string instead of printing.
                    When save_path is set, the string is always returned.
        title: Override the model name shown in the diagram header.

    Returns:
        The rendered string if return_str is True or save_path is set,
        otherwise None (output is printed to stdout).

    Examples:
        >>> from grilly.nn import Linear, Sequential, ReLU
        >>> model = Sequential(Linear(784, 256), ReLU(), Linear(256, 10))
        >>> visualize(model)

        >>> from grillcheese.language.grilly_ssm import GrillySSMLM, GrillySSMConfig
        >>> config = GrillySSMConfig(d_model=768, n_layers=12, vocab_size=32768)
        >>> model = GrillySSMLM(config)
        >>> visualize(model, output="html", save_path="ssm_arch.html")
    """
    layers = discover_layers(model)

    if output == "ascii":
        result = render_ascii(model, layers)
    elif output == "html":
        result = render_html(model, layers, title=title)
    elif output == "svg":
        # Extract just the SVG from the HTML renderer
        html_full = render_html(model, layers, title=title)
        # Find the SVG tag
        start = html_full.find("<svg")
        end = html_full.find("</svg>") + len("</svg>")
        if start >= 0 and end > start:
            result = html_full[start:end]
        else:
            result = render_html(model, layers, title=title)
    else:
        raise ValueError(f"Unknown output mode: {output!r}. Use 'ascii', 'html', or 'svg'.")

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(result)

    if return_str or save_path is not None:
        return result

    print(result)
    return None


def summary(model: Any) -> str:
    """Print a compact parameter summary table.

    This is a lighter-weight alternative to visualize() that produces a
    tabular summary similar to PyTorch's torchsummary, but works with
    grilly.nn modules.

    Args:
        model: The model to summarize.

    Returns:
        The formatted summary string.
    """
    layers = discover_layers(model)
    total_params = sum(layer.param_count for layer in layers)
    model_name = type(model).__name__

    lines: list[str] = []
    sep = "=" * 80
    lines.append(sep)
    lines.append(f"  {model_name} Summary")
    lines.append(sep)
    lines.append(
        f"  {'Layer':<20} {'Type':<20} {'Output Shape':<20} {'Params':>10}"
    )
    lines.append("-" * 80)

    for layer in layers:
        repeat_str = f" x{layer.repeat_count}" if layer.repeat_count > 1 else ""
        type_str = f"{layer.type_name}{repeat_str}"
        flags = ""
        if layer.is_frozen:
            flags += " [F]"
        if layer.has_gpu:
            flags += " [G]"
        lines.append(
            f"  {layer.name:<20} {type_str:<20} {layer.output_shape:<20} "
            f"{_format_param_count(layer.param_count):>8}{flags}"
        )

        # Show children indented
        for child in layer.children:
            c_repeat = f" x{child.repeat_count}" if child.repeat_count > 1 else ""
            c_type = f"{child.type_name}{c_repeat}"
            lines.append(
                f"    {child.name:<18} {c_type:<20} {child.output_shape:<20} "
                f"{_format_param_count(child.param_count):>8}"
            )

    lines.append(sep)
    lines.append(f"  Total params: {total_params:>12,}")
    lines.append(f"  Memory (fp32): {_format_memory(total_params):>10}")
    lines.append(sep)

    result = "\n".join(lines)
    print(result)
    return result
