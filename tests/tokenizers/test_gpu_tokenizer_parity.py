"""Tokenizer parity tests (first real pass, adaptive to in-progress API)."""

import pytest

from tests._tokenizer_parity_helpers import (
    encode_ids,
    extract_input_ids,
    load_grilly_tokenizer,
    load_hf_tokenizer,
)


@pytest.mark.parametrize(
    "model_id,text",
    [
        ("bert-base-uncased", "Hello from Grilly GPU tokenization."),
        ("distilbert-base-uncased", "Tokenizer parity check with punctuation: !?.,"),
    ],
)
def test_tokenizer_ids_match_hf_reference(model_id: str, text: str):
    """Token IDs should match Hugging Face for covered model/tokenizer assets."""
    hf_tok = load_hf_tokenizer(model_id)
    gr_tok = load_grilly_tokenizer(model_id)

    hf_ids = extract_input_ids(hf_tok(text))
    gr_ids = encode_ids(gr_tok, text)

    assert list(gr_ids) == list(hf_ids)


def test_batch_encode_matches_hf_reference():
    """Batch tokenization should match Hugging Face input_ids for canonical uncased BERT."""
    model_id = "bert-base-uncased"
    texts = [
        "Grilly runs on Vulkan.",
        "Any GPU, one backend.",
        "Parity matters before v1.0.",
    ]
    hf_tok = load_hf_tokenizer(model_id)
    gr_tok = load_grilly_tokenizer(model_id)

    hf_batch = hf_tok(texts, padding=False, truncation=False)
    hf_ids = hf_batch["input_ids"]

    if hasattr(gr_tok, "batch_encode"):
        gr_batch = gr_tok.batch_encode(texts)
    elif hasattr(gr_tok, "__call__"):
        gr_batch = gr_tok(texts)
    else:
        pytest.skip("Tokenizer missing batch_encode/__call__ batch path")

    gr_ids = extract_input_ids(gr_batch)
    assert [list(x) for x in gr_ids] == [list(x) for x in hf_ids]

