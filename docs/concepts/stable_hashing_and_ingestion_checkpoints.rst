Stable Hashing and Ingestion Checkpoints
========================================

Why stable hashing exists
-------------------------

Python's built-in `hash()` is process-randomized, which breaks deterministic
seeding when used directly. Grilly provides stable hashing utilities for
reproducible vector generation and indexing.

Stable hash utilities
---------------------

From `grilly.utils.stable_hash`:

- `stable_u32(...)`
- `stable_u64(...)`
- `stable_bytes(...)`
- `bipolar_from_key(...)`

These functions prefer BLAKE3 and fall back deterministically when BLAKE3 is
unavailable.

Design choices
--------------

Stable hashing and checkpointing were added to solve reproducibility and scale:

1. Avoid process-randomized `hash()` for seed derivation in vector pipelines.
2. Prefer BLAKE3 for speed and deterministic byte output.
3. Store ingestion checkpoints in compact array formats (`npz`) with an explicit
   manifest for forward compatibility.
4. Support compressed sentence memory modes to control disk and RAM usage.

Ingestion checkpoint system
---------------------------

`grilly.utils.ingest_checkpoint` provides a compact checkpoint format for
experimental language/cognition ingestion states.

Key capabilities:

- save/load ingestion state (`save_ingest_checkpoint`, `load_ingest_checkpoint`)
- compressed sentence memory
- compact token id storage
- lightweight view (`CheckpointView`) for inspection

Checkpoint flow example
-----------------------

.. code-block:: python

   from grilly.experimental.cognitive.controller import CognitiveController
   from grilly.utils.ingest_checkpoint import (
       save_ingest_checkpoint,
       load_ingest_checkpoint,
       CheckpointView,
   )

   controller = CognitiveController(dim=1024, word_use_ngrams=False)
   # ... ingest entries ...

   save_ingest_checkpoint(
       "checkpoints/ingest_v2.npz",
       controller,
       include_sentence_memory=True,
       sentence_compress="auto",
       fp16=True,
   )

   view = CheckpointView("checkpoints/ingest_v2.npz")
   print(view.sentence_count())

   restored = CognitiveController(dim=1024, word_use_ngrams=False)
   manifest = load_ingest_checkpoint("checkpoints/ingest_v2.npz", restored)
   print(manifest["format"])

When to use it
--------------

- long-running ingestion jobs
- reproducible experiments
- fast restarts for iterative development
