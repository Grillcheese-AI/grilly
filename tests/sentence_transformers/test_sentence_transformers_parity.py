"""Sentence-transformers adaptive parity tests against reference implementation."""

from __future__ import annotations

import numpy as np
import pytest

sentence_transformers = pytest.importorskip("sentence_transformers")


def _load_reference_model(model_name: str):
    try:
        return sentence_transformers.SentenceTransformer(model_name, device="cpu")
    except Exception as exc:  # pragma: no cover - env/model cache dependent
        pytest.skip(f"Reference sentence-transformers model unavailable: {exc}")


def _load_grilly_model(model_name: str):
    try:
        from grilly.utils.vulkan_sentence_transformer import VulkanSentenceTransformer
    except Exception as exc:
        pytest.skip(f"Grilly Vulkan sentence transformer unavailable: {exc}")

    try:
        return VulkanSentenceTransformer(model_name=model_name)
    except Exception as exc:  # pragma: no cover - depends on Vulkan/model assets
        pytest.skip(f"Could not initialize VulkanSentenceTransformer: {exc}")


def _as_float32(x):
    arr = np.asarray(x)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr


@pytest.mark.gpu
def test_sentence_transformer_encode_shape_dtype_parity():
    """Embedding shape/dtype should align with reference model outputs."""
    model_name = "all-MiniLM-L6-v2"
    texts = [
        "Grilly runs on Vulkan across vendors.",
        "Tokenizer and encoder should stay GPU-resident.",
    ]

    ref = _load_reference_model(model_name)
    gr = _load_grilly_model(model_name)

    ref_emb = _as_float32(ref.encode(texts, normalize_embeddings=True))
    gr_emb = _as_float32(gr.encode(texts, normalize_embeddings=True))

    assert ref_emb.shape == gr_emb.shape
    assert gr_emb.dtype == np.float32
    assert np.all(np.isfinite(gr_emb))


@pytest.mark.gpu
def test_sentence_transformer_similarity_top1_matches_reference():
    """Top-1 nearest sentence should match reference similarity ranking."""
    model_name = "all-MiniLM-L6-v2"
    texts = [
        "How do I speed up GPU dispatch batching?",
        "Use one submit for many kernels with command recording.",
        "Bananas are yellow and sweet.",
        "Fused loss kernels reduce synchronization overhead.",
    ]

    ref = _load_reference_model(model_name)
    gr = _load_grilly_model(model_name)

    ref_emb = _as_float32(ref.encode(texts, normalize_embeddings=True))
    gr_emb = _as_float32(gr.encode(texts, normalize_embeddings=True))

    # Query = first sentence, compare against candidates [1:]
    ref_sims = ref_emb[1:] @ ref_emb[0]
    gr_sims = gr_emb[1:] @ gr_emb[0]

    assert int(np.argmax(gr_sims)) == int(np.argmax(ref_sims))

