"""Streaming GPU embedding + upload pipeline.

Producer-consumer pattern:
- Producer: GPU batch encodes sentences → puts (texts, embeddings) in queue
- Consumer: parallel uploads embeddings to vector store

Usage:
    pipeline = StreamingPipeline(
        embedder=BatchEmbedder("model.onnx"),
        uploader=qdrant_upload_fn,
        batch_size=1024,
        num_upload_workers=4,
    )
    pipeline.process(all_sentences)  # pipelined embed + upload
"""

import logging
import queue
import threading
import time
from collections.abc import Callable

logger = logging.getLogger("grilly.streaming_pipeline")


class StreamingPipeline:
    """Producer-consumer pipeline for batch embedding + upload.

    Args:
        embedder: Object with encode(texts, batch_size) method
        uploader: Callable(texts, embeddings) that uploads a batch
        batch_size: Sentences per GPU batch (default: 1024)
        num_upload_workers: Parallel upload threads (default: 4)
        queue_maxsize: Max batches in flight (default: 8, backpressure)
    """

    def __init__(
        self,
        embedder,
        uploader: Callable,
        batch_size: int = 1024,
        num_upload_workers: int = 4,
        queue_maxsize: int = 8,
    ):
        self.embedder = embedder
        self.uploader = uploader
        self.batch_size = batch_size
        self.num_upload_workers = num_upload_workers
        self.queue_maxsize = queue_maxsize
        self._stats = {"embedded": 0, "uploaded": 0, "errors": 0}

    def process(self, texts, show_progress: bool = True) -> dict:
        """Process all texts through the embed→upload pipeline.

        Args:
            texts: iterable of strings
            show_progress: log progress every 10 batches

        Returns:
            dict with keys ``embedded``, ``uploaded``, ``errors``
        """
        # Reset stats for this run
        self._stats = {"embedded": 0, "uploaded": 0, "errors": 0}

        texts = list(texts)
        if not texts:
            logger.info("Pipeline called with empty input — nothing to do.")
            return self._stats.copy()

        q: queue.Queue = queue.Queue(maxsize=self.queue_maxsize)
        stop_event = threading.Event()

        # ------------------------------------------------------------------ #
        # Upload workers                                                       #
        # ------------------------------------------------------------------ #
        def upload_worker() -> None:
            while not stop_event.is_set() or not q.empty():
                try:
                    batch_texts, batch_embeddings = q.get(timeout=1.0)
                except queue.Empty:
                    continue
                try:
                    self.uploader(batch_texts, batch_embeddings)
                    self._stats["uploaded"] += len(batch_texts)
                except Exception as e:
                    self._stats["errors"] += len(batch_texts)
                    logger.warning("Upload error: %s", e)
                finally:
                    q.task_done()

        workers = [
            threading.Thread(target=upload_worker, daemon=True)
            for _ in range(self.num_upload_workers)
        ]
        for w in workers:
            w.start()

        # ------------------------------------------------------------------ #
        # Producer: GPU batch embed                                            #
        # ------------------------------------------------------------------ #
        t0 = time.perf_counter()
        batch_count = 0
        batch: list[str] = []

        for text in texts:
            batch.append(text)
            if len(batch) >= self.batch_size:
                embeddings = self.embedder.encode(batch, batch_size=self.batch_size)
                self._stats["embedded"] += len(batch)
                q.put((batch, embeddings))
                batch_count += 1
                if show_progress and batch_count % 10 == 0:
                    elapsed = time.perf_counter() - t0
                    rate = self._stats["embedded"] / max(elapsed, 1e-9)
                    logger.info(
                        "Embedded %d, uploaded %d (%.0f/s)",
                        self._stats["embedded"],
                        self._stats["uploaded"],
                        rate,
                    )
                batch = []

        # Final partial batch
        if batch:
            embeddings = self.embedder.encode(batch, batch_size=len(batch))
            self._stats["embedded"] += len(batch)
            q.put((batch, embeddings))

        # Wait for all uploads to finish
        q.join()
        stop_event.set()
        for w in workers:
            w.join(timeout=5.0)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Pipeline complete: %d embedded, %d uploaded, %d errors in %.1fs (%.0f/s)",
            self._stats["embedded"],
            self._stats["uploaded"],
            self._stats["errors"],
            elapsed,
            self._stats["embedded"] / max(elapsed, 0.01),
        )

        return self._stats.copy()

    @property
    def stats(self) -> dict:
        """Return a snapshot of current pipeline statistics."""
        return self._stats.copy()


__all__ = ["StreamingPipeline"]
