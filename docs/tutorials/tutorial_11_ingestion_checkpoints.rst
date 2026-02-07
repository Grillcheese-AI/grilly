Tutorial 11: Ingestion Checkpoints and Reproducibility
=======================================================

Goal: save and reload experimental ingestion state with compressed sentence
memory and stable hashing utilities.

Step 1: Build a controller
--------------------------

.. code-block:: python

   from grilly.experimental.cognitive.controller import CognitiveController

   controller = CognitiveController(dim=512, word_use_ngrams=False)

Step 2: Ingest a tiny SVC batch
-------------------------------

.. code-block:: python

   from grilly.experimental.language.svc_loader import load_svc_entries_from_dicts

   entries = load_svc_entries_from_dicts([
       {
           "id": "c0",
           "text": "Exercise improves health.",
           "svc": {"s": "Exercise", "v": "improves", "c": "health"},
           "pos": ["NOUN", "VERB", "NOUN", "PUNCT"],
           "deps": ["nsubj", "ROOT", "dobj", "punct"],
           "lemmas": ["exercise", "improve", "health", "."],
           "root_verb": "improve",
           "realm": "health",
           "source": "manual",
           "complexity": 0.3,
       }
   ])
   controller.ingest_svc(entries, verbose=False)

Step 3: Save checkpoint
-----------------------

.. code-block:: python

   from grilly.utils.ingest_checkpoint import save_ingest_checkpoint

   save_ingest_checkpoint(
       "checkpoints/ingest_v2.npz",
       controller,
       include_sentence_memory=True,
       sentence_compress="auto",
       fp16=True,
   )

Step 4: Inspect without full restore
------------------------------------

.. code-block:: python

   from grilly.utils.ingest_checkpoint import CheckpointView

   view = CheckpointView("checkpoints/ingest_v2.npz")
   print("sentences:", view.sentence_count())

Step 5: Load into a fresh controller
------------------------------------

.. code-block:: python

   from grilly.utils.ingest_checkpoint import load_ingest_checkpoint

   restored = CognitiveController(dim=512, word_use_ngrams=False)
   manifest = load_ingest_checkpoint("checkpoints/ingest_v2.npz", restored)
   print(manifest["format"])

Step 6: Stable hash usage
-------------------------

.. code-block:: python

   from grilly.utils.stable_hash import stable_u32
   print(stable_u32("realm:health", domain="routing"))
