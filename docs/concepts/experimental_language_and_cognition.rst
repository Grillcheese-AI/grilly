Experimental Language and Cognition
===================================

Scope
-----

Grilly includes an experimental cognitive stack centered around vector symbolic
representations and fast ingestion.

Primary components
------------------

- `experimental.language`:
  `WordEncoder`, `SentenceEncoder`, `InstantLanguage`, SVC ingestion pipeline.
- `experimental.cognitive`:
  `WorldModel`, `CognitiveController`, simulation and understanding helpers.
- `experimental.vsa`:
  binary/holographic ops and GPU-backed VSA acceleration.

Instant language model
----------------------

`InstantLanguage` provides:

- on-the-fly word encoding
- sentence memory
- template learning
- similarity search over stored sentence vectors
- SVC data ingestion with optional GPU acceleration through
  `SVCIngestionEngine`

Cognitive controller
--------------------

`CognitiveController` orchestrates:

1. language understanding
2. world-model consistency checks
3. candidate generation and simulation
4. confidence-gated response selection

Minimal ingestion example
-------------------------

.. code-block:: python

   from grilly.experimental.language.svc_loader import load_svc_entries_from_dicts
   from grilly.experimental.cognitive.controller import CognitiveController

   entries = load_svc_entries_from_dicts([
       {
           "id": "x0",
           "text": "Gravity attracts mass.",
           "svc": {"s": "Gravity", "v": "attracts", "c": "mass"},
           "pos": ["NOUN", "VERB", "NOUN", "PUNCT"],
           "deps": ["nsubj", "ROOT", "dobj", "punct"],
           "lemmas": ["gravity", "attract", "mass", "."],
           "root_verb": "attract",
           "realm": "science",
           "source": "manual",
           "complexity": 0.3,
       }
   ])

   controller = CognitiveController(dim=1024)
   result = controller.ingest_svc(entries, verbose=False)
   print(result.summary())

Operational guidance
--------------------

- Use smaller dimensions during iteration (`dim=512` or `1024`) for faster tests.
- Turn off expensive n-gram paths for large ingestion runs when needed.
- Persist ingestion state with checkpoint utilities for resumable workflows.
