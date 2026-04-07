"""GRL (.grl) checkpoint format — roundtrip and magic/version checks."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest
from grilly.utils.grl_checkpoint import (
    FORMAT_VERSION,
    HEADER_SIZE,
    MAGIC,
    load_grl,
    save_grl,
)


def test_grl_roundtrip_nested_state():
    state = {
        "embed": {"weight": np.random.randn(10, 8).astype(np.float32)},
        "layers": {
            "0": {"w": np.ones((4, 4), dtype=np.float32)},
        },
    }
    meta = {"training_step": 42, "best_ppl": 3.14}

    with tempfile.NamedTemporaryFile(suffix=".grl", delete=False) as f:
        path = f.name
    try:
        save_grl(path, state, metadata=meta)
        out = load_grl(path)
        assert "metadata" in out
        assert out["metadata"]["schema"] == "grilly.checkpoint.v1"
        assert out["training_step"] == 42
        m = out["model"]
        assert np.allclose(m["embed"]["weight"], state["embed"]["weight"])
        assert np.allclose(m["layers"]["0"]["w"], state["layers"]["0"]["w"])
    finally:
        import os

        os.unlink(path)


def test_grl_rejects_bad_magic():
    with tempfile.NamedTemporaryFile(suffix=".grl", delete=False) as f:
        f.write(b"XXXX")
        path = f.name
    try:
        with pytest.raises(ValueError, match="GRL"):
            load_grl(path)
    finally:
        import os

        os.unlink(path)


def test_header_constants_match_cpp_layout():
    assert len(MAGIC) == 4
    assert HEADER_SIZE == 64
    assert FORMAT_VERSION == 1
