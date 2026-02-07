Tutorial 05: Experimental Language Ingestion
============================================

Goal: ingest SVC-style sentence data and query semantic memory.

Step 1: Create SVC entries
--------------------------

.. code-block:: python

   from grilly.experimental.language.svc_loader import load_svc_entries_from_dicts

   raw_entries = [
       {
           "id": "e0",
           "text": "Vaccines prevent infectious diseases.",
           "svc": {"s": "Vaccines", "v": "prevent", "c": "infectious diseases"},
           "pos": ["NOUN", "VERB", "ADJ", "NOUN", "PUNCT"],
           "deps": ["nsubj", "ROOT", "amod", "dobj", "punct"],
           "lemmas": ["vaccine", "prevent", "infectious", "disease", "."],
           "root_verb": "prevent",
           "realm": "health",
           "source": "manual",
           "complexity": 0.4,
       }
   ]
   entries = load_svc_entries_from_dicts(raw_entries)

Step 2: Initialize InstantLanguage
----------------------------------

.. code-block:: python

   from grilly.experimental.language.system import InstantLanguage

   lang = InstantLanguage(dim=1024)

Step 3: Ingest entries
----------------------

.. code-block:: python

   result = lang.ingest_svc(entries)
   print("sentences learned:", result.sentences_learned)
   print("realms:", result.realm_counts)

Step 4: Query similar sentences
-------------------------------

.. code-block:: python

   query = "Vaccines are effective."
   similar = lang.find_similar_sentences(query, top_k=3)
   print(similar)

Step 5: Integrate with controller
---------------------------------

.. code-block:: python

   from grilly.experimental.cognitive.controller import CognitiveController

   controller = CognitiveController(dim=1024)
   controller.ingest_svc(entries)
   response = controller.process("What helps prevent diseases?")
   print(response)

This workflow gives you a full path from structured sentence ingestion to cognitive querying.
