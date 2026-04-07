"""SentencePiece compatibility tests (adaptive to in-progress tokenizer API)."""

import pytest

sentencepiece = pytest.importorskip("sentencepiece")
from tests._tokenizer_parity_helpers import (
    encode_ids,
    extract_input_ids,
    load_grilly_tokenizer,
    load_hf_tokenizer,
)


@pytest.mark.parametrize(
    "model_id,text",
    [
        ("t5-small", "Translate English to German: A tiny test sentence."),
        ("google/mt5-small", "A multilingual sentencepiece parity check."),
    ],
)
def test_sentencepiece_ids_match_hf_reference(model_id: str, text: str):
    """SentencePiece-backed token IDs should match HF reference on supported models."""
    # Keep explicit import usage for environment signaling.
    assert sentencepiece.__name__ == "sentencepiece"

    hf_tok = load_hf_tokenizer(model_id)
    gr_tok = load_grilly_tokenizer(model_id)

    hf_ids = extract_input_ids(hf_tok(text))
    gr_ids = encode_ids(gr_tok, text)

    assert list(gr_ids) == list(hf_ids)


def test_sentencepiece_special_tokens_alignment():
    """Special token insertion should align with HF on canonical T5 tokenizer."""
    model_id = "t5-small"
    text = "summarize: Grilly targets Vulkan GPUs."

    hf_tok = load_hf_tokenizer(model_id)
    gr_tok = load_grilly_tokenizer(model_id)

    hf_with = extract_input_ids(hf_tok(text, add_special_tokens=True))
    hf_without = extract_input_ids(hf_tok(text, add_special_tokens=False))
    gr_with = encode_ids(gr_tok, text, add_special_tokens=True)
    gr_without = encode_ids(gr_tok, text, add_special_tokens=False)

    assert list(gr_with) == list(hf_with)
    assert list(gr_without) == list(hf_without)

