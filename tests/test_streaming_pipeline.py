"""Tests for utils/streaming_pipeline.py.

All tests use mock embedder and uploader — no real GPU or Qdrant required.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from grilly.utils.streaming_pipeline import StreamingPipeline


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

EMBED_DIM = 64


def _make_embedder(embed_dim: int = EMBED_DIM):
    """Return a mock embedder whose encode() returns (N, embed_dim) float32."""
    embedder = MagicMock()

    def _encode(texts, batch_size=512):
        n = len(texts)
        return np.random.randn(n, embed_dim).astype(np.float32)

    embedder.encode.side_effect = _encode
    return embedder


def _make_uploader():
    """Return a mock uploader that records calls."""
    return MagicMock()


def _make_pipeline(batch_size=4, num_workers=2, queue_maxsize=8, uploader=None):
    embedder = _make_embedder()
    if uploader is None:
        uploader = _make_uploader()
    return StreamingPipeline(
        embedder=embedder,
        uploader=uploader,
        batch_size=batch_size,
        num_upload_workers=num_workers,
        queue_maxsize=queue_maxsize,
    ), embedder, uploader


# --------------------------------------------------------------------------- #
# Test: basic processing                                                       #
# --------------------------------------------------------------------------- #


class TestStreamingPipelineBasic:
    def test_processes_all_texts(self):
        """All texts are embedded and uploaded."""
        texts = [f"sentence {i}" for i in range(20)]
        pipeline, embedder, uploader = _make_pipeline(batch_size=5)

        stats = pipeline.process(texts, show_progress=False)

        assert stats["embedded"] == 20
        assert stats["uploaded"] == 20
        assert stats["errors"] == 0

    def test_uploader_called_for_each_batch(self):
        """uploader is called once per batch (ceil(N / batch_size) times)."""
        texts = [f"t{i}" for i in range(10)]
        pipeline, _, uploader = _make_pipeline(batch_size=3)

        pipeline.process(texts, show_progress=False)

        # ceil(10 / 3) = 4 batches
        assert uploader.call_count == 4

    def test_uploader_receives_texts_and_embeddings(self):
        """Each uploader call receives (list[str], np.ndarray)."""
        texts = ["alpha", "beta", "gamma"]
        pipeline, _, uploader = _make_pipeline(batch_size=2)

        pipeline.process(texts, show_progress=False)

        for args, _ in uploader.call_args_list:
            batch_texts, batch_embeddings = args
            assert isinstance(batch_texts, list)
            assert all(isinstance(t, str) for t in batch_texts)
            assert isinstance(batch_embeddings, np.ndarray)
            assert batch_embeddings.ndim == 2
            assert batch_embeddings.shape == (len(batch_texts), EMBED_DIM)

    def test_returns_stats_dict(self):
        """process() returns a dict with embedded/uploaded/errors keys."""
        texts = ["a", "b", "c"]
        pipeline, _, _ = _make_pipeline(batch_size=2)

        result = pipeline.process(texts, show_progress=False)

        assert isinstance(result, dict)
        assert set(result.keys()) >= {"embedded", "uploaded", "errors"}


# --------------------------------------------------------------------------- #
# Test: batching                                                               #
# --------------------------------------------------------------------------- #


class TestBatching:
    def test_exact_batch_boundary(self):
        """N divisible by batch_size → every batch is full-size."""
        texts = [f"s{i}" for i in range(12)]
        pipeline, embedder, uploader = _make_pipeline(batch_size=4)

        pipeline.process(texts, show_progress=False)

        # 3 batches of 4
        assert embedder.encode.call_count == 3
        for call_args in embedder.encode.call_args_list:
            batch_texts = call_args[0][0]
            assert len(batch_texts) == 4

    def test_partial_last_batch(self):
        """Remaining texts form a smaller final batch."""
        texts = [f"s{i}" for i in range(11)]
        pipeline, embedder, uploader = _make_pipeline(batch_size=4)

        pipeline.process(texts, show_progress=False)

        # 2 full batches + 1 partial (3)
        assert embedder.encode.call_count == 3
        # Last call batch_size arg should equal the remainder
        last_call_batch_texts = embedder.encode.call_args_list[-1][0][0]
        assert len(last_call_batch_texts) == 3

    def test_single_text_single_batch(self):
        """A single text forms exactly one batch."""
        pipeline, embedder, uploader = _make_pipeline(batch_size=10)

        pipeline.process(["only one"], show_progress=False)

        assert embedder.encode.call_count == 1
        assert uploader.call_count == 1

    def test_batch_size_one(self):
        """batch_size=1 → one encode call and one upload per text."""
        texts = ["x", "y", "z"]
        pipeline, embedder, uploader = _make_pipeline(batch_size=1)

        pipeline.process(texts, show_progress=False)

        assert embedder.encode.call_count == 3
        assert uploader.call_count == 3


# --------------------------------------------------------------------------- #
# Test: stats tracking                                                         #
# --------------------------------------------------------------------------- #


class TestStatsTracking:
    def test_stats_property_reflects_state(self):
        """stats property returns a copy of internal counters."""
        pipeline, _, _ = _make_pipeline(batch_size=5)
        pipeline.process(["a", "b", "c", "d", "e"], show_progress=False)

        s = pipeline.stats
        assert s["embedded"] == 5
        assert s["uploaded"] == 5
        assert s["errors"] == 0

    def test_stats_reset_between_runs(self):
        """Each call to process() resets stats."""
        pipeline, _, _ = _make_pipeline(batch_size=3)

        pipeline.process(["a", "b", "c"], show_progress=False)
        first = pipeline.stats.copy()

        pipeline.process(["x", "y"], show_progress=False)
        second = pipeline.stats.copy()

        assert first["embedded"] == 3
        assert second["embedded"] == 2  # not cumulative

    def test_stats_copy_is_independent(self):
        """Mutating the returned stats dict does not affect pipeline state."""
        pipeline, _, _ = _make_pipeline(batch_size=2)
        pipeline.process(["p", "q"], show_progress=False)

        s = pipeline.stats
        s["embedded"] = 9999

        assert pipeline.stats["embedded"] == 2


# --------------------------------------------------------------------------- #
# Test: empty input                                                            #
# --------------------------------------------------------------------------- #


class TestEmptyInput:
    def test_empty_list_returns_zero_stats(self):
        pipeline, embedder, uploader = _make_pipeline()

        stats = pipeline.process([], show_progress=False)

        assert stats["embedded"] == 0
        assert stats["uploaded"] == 0
        assert stats["errors"] == 0
        embedder.encode.assert_not_called()
        uploader.assert_not_called()

    def test_empty_generator_returns_zero_stats(self):
        pipeline, _, _ = _make_pipeline()

        stats = pipeline.process(iter([]), show_progress=False)

        assert stats["embedded"] == 0


# --------------------------------------------------------------------------- #
# Test: error handling                                                         #
# --------------------------------------------------------------------------- #


class TestErrorHandling:
    def test_upload_error_increments_error_counter(self):
        """When uploader raises, errors are counted and pipeline continues."""
        texts = [f"t{i}" for i in range(6)]

        uploader = MagicMock(side_effect=RuntimeError("qdrant unavailable"))
        pipeline, _, _ = _make_pipeline(batch_size=2, uploader=uploader)

        stats = pipeline.process(texts, show_progress=False)

        # All texts were embedded
        assert stats["embedded"] == 6
        # All failed to upload
        assert stats["errors"] == 6
        assert stats["uploaded"] == 0

    def test_partial_upload_error(self):
        """First batch succeeds, second fails → partial upload count."""
        texts = [f"s{i}" for i in range(4)]
        call_counter = {"n": 0}

        def flaky_uploader(batch_texts, batch_embeddings):
            call_counter["n"] += 1
            if call_counter["n"] == 2:
                raise ConnectionError("timeout")

        pipeline = StreamingPipeline(
            embedder=_make_embedder(),
            uploader=flaky_uploader,
            batch_size=2,
            num_upload_workers=1,  # serial — easier to reason about
            queue_maxsize=4,
        )

        stats = pipeline.process(texts, show_progress=False)

        assert stats["embedded"] == 4
        assert stats["uploaded"] == 2
        assert stats["errors"] == 2

    def test_pipeline_continues_after_error(self):
        """Even if every upload fails, process() returns normally (no exception)."""
        texts = ["a", "b", "c"]
        uploader = MagicMock(side_effect=Exception("always fails"))
        pipeline, _, _ = _make_pipeline(batch_size=1, uploader=uploader)

        # Should not raise
        stats = pipeline.process(texts, show_progress=False)
        assert stats["embedded"] == 3


# --------------------------------------------------------------------------- #
# Test: constructor / attribute access                                         #
# --------------------------------------------------------------------------- #


class TestConstructor:
    def test_default_parameters(self):
        embedder = _make_embedder()
        uploader = _make_uploader()
        p = StreamingPipeline(embedder=embedder, uploader=uploader)

        assert p.batch_size == 1024
        assert p.num_upload_workers == 4
        assert p.queue_maxsize == 8

    def test_custom_parameters(self):
        embedder = _make_embedder()
        uploader = _make_uploader()
        p = StreamingPipeline(
            embedder=embedder,
            uploader=uploader,
            batch_size=256,
            num_upload_workers=8,
            queue_maxsize=16,
        )

        assert p.batch_size == 256
        assert p.num_upload_workers == 8
        assert p.queue_maxsize == 16
