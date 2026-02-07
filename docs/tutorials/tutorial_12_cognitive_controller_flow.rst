Tutorial 12: Cognitive Controller End-to-End
============================================

Goal: run the full experimental cognitive flow from ingestion to response.

Step 1: Initialize controller
-----------------------------

.. code-block:: python

   from grilly.experimental.cognitive.controller import CognitiveController

   controller = CognitiveController(dim=1024)

Step 2: Add baseline world knowledge
------------------------------------

.. code-block:: python

   controller.add_knowledge("vaccines", "prevent", "infection")
   controller.add_knowledge("sleep", "improves", "recovery")

Step 3: Ingest structured language data
---------------------------------------

.. code-block:: python

   from grilly.experimental.language.svc_loader import load_svc_entries_from_dicts

   entries = load_svc_entries_from_dicts([
       {
           "id": "k0",
           "text": "Vaccines prevent disease.",
           "svc": {"s": "Vaccines", "v": "prevent", "c": "disease"},
           "pos": ["NOUN", "VERB", "NOUN", "PUNCT"],
           "deps": ["nsubj", "ROOT", "dobj", "punct"],
           "lemmas": ["vaccine", "prevent", "disease", "."],
           "root_verb": "prevent",
           "realm": "health",
           "source": "manual",
           "complexity": 0.2,
       }
   ])

   result = controller.ingest_svc(entries, verbose=False)
   print(result.sentences_learned)

Step 4: Understand input
------------------------

.. code-block:: python

   understanding = controller.understand("How do vaccines help?")
   print(understanding.confidence)

Step 5: Generate a response
---------------------------

.. code-block:: python

   response = controller.process("How do vaccines help?", verbose=True)
   print(response)

Step 6: Inspect reasoning trace
-------------------------------

.. code-block:: python

   for line in controller.thinking_trace:
       print(line)

This pattern gives you a reusable scaffold for experimental assistants,
reasoning simulations, and safety/coherence experiments.
