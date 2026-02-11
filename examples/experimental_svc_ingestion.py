"""
Example: SVC Data Ingestion Pipeline

Demonstrates the full SVC ingestion flow:
1. Load SVC entries (from inline data or JSONL files)
2. Ingest into InstantLanguage (vocabulary, sentences, templates, realm vectors)
3. Ingest into CognitiveController (world facts, causal links)
4. Build realm-routed ResonatorMoE
5. Query the system with natural language

This is part of the challenge - showing how
structured SVC data powers instant language understanding without
any gradient descent.
"""

import numpy as np
from grilly.experimental.cognitive import CognitiveController
from grilly.experimental.language import (
    InstantLanguage,
    SVCIngestionEngine,
    load_svc_entries_from_dicts,
)
from grilly.experimental.moe import ResonatorMoE
from grilly.experimental.vsa import BinaryOps

# =============================================================================
# Inline SVC Data (mirrors real dataset schema)
# =============================================================================

SVC_DATA = [
    {
        "id": "demo_h0",
        "text": "Exercise is crucial for maintaining good health.",
        "svc": {"s": "Exercise", "v": "is", "c": "crucial for maintaining good health"},
        "pos": ["NOUN", "AUX", "ADJ", "ADP", "VERB", "ADJ", "NOUN", "PUNCT"],
        "lemmas": ["exercise", "be", "crucial", "for", "maintain", "good", "health", "."],
        "deps": ["nsubj", "ROOT", "acomp", "prep", "pcomp", "amod", "dobj", "punct"],
        "root_verb": "be",
        "realm": "health",
        "source": "instruct",
        "complexity": 0.4,
    },
    {
        "id": "demo_h1",
        "text": "Vaccines prevent many infectious diseases effectively.",
        "svc": {"s": "Vaccines", "v": "prevent", "c": "many infectious diseases effectively"},
        "pos": ["NOUN", "VERB", "ADJ", "ADJ", "NOUN", "ADV", "PUNCT"],
        "lemmas": ["vaccine", "prevent", "many", "infectious", "disease", "effectively", "."],
        "deps": ["nsubj", "ROOT", "amod", "amod", "dobj", "advmod", "punct"],
        "root_verb": "prevent",
        "realm": "health",
        "source": "instruct",
        "complexity": 0.5,
    },
    {
        "id": "demo_h2",
        "text": "Regular sleep improves overall health significantly.",
        "svc": {"s": "Regular sleep", "v": "improves", "c": "overall health significantly"},
        "pos": ["ADJ", "NOUN", "VERB", "ADJ", "NOUN", "ADV", "PUNCT"],
        "lemmas": ["regular", "sleep", "improve", "overall", "health", "significantly", "."],
        "deps": ["amod", "nsubj", "ROOT", "amod", "dobj", "advmod", "punct"],
        "root_verb": "improve",
        "realm": "health",
        "source": "instruct",
        "complexity": 0.45,
    },
    {
        "id": "demo_s0",
        "text": "Photosynthesis converts sunlight into chemical energy.",
        "svc": {"s": "Photosynthesis", "v": "converts", "c": "sunlight into chemical energy"},
        "pos": ["NOUN", "VERB", "NOUN", "ADP", "ADJ", "NOUN", "PUNCT"],
        "lemmas": ["photosynthesis", "convert", "sunlight", "into", "chemical", "energy", "."],
        "deps": ["nsubj", "ROOT", "dobj", "prep", "amod", "pobj", "punct"],
        "root_verb": "convert",
        "realm": "science",
        "source": "instruct",
        "complexity": 0.55,
    },
    {
        "id": "demo_s1",
        "text": "Gravity attracts objects toward the center of mass.",
        "svc": {"s": "Gravity", "v": "attracts", "c": "objects toward the center of mass"},
        "pos": ["NOUN", "VERB", "NOUN", "ADP", "DET", "NOUN", "ADP", "NOUN", "PUNCT"],
        "lemmas": ["gravity", "attract", "object", "toward", "the", "center", "of", "mass", "."],
        "deps": ["nsubj", "ROOT", "dobj", "prep", "det", "pobj", "prep", "pobj", "punct"],
        "root_verb": "attract",
        "realm": "science",
        "source": "instruct",
        "complexity": 0.6,
    },
    {
        "id": "demo_g0",
        "text": "The weather changes throughout the seasons.",
        "svc": {"s": "The weather", "v": "changes", "c": "throughout the seasons"},
        "pos": ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN", "PUNCT"],
        "lemmas": ["the", "weather", "change", "throughout", "the", "season", "."],
        "deps": ["det", "nsubj", "ROOT", "prep", "det", "pobj", "punct"],
        "root_verb": "change",
        "realm": "general",
        "source": "instruct",
        "complexity": 0.35,
    },
    {
        "id": "demo_g1",
        "text": "Communication builds stronger relationships over time.",
        "svc": {"s": "Communication", "v": "builds", "c": "stronger relationships over time"},
        "pos": ["NOUN", "VERB", "ADJ", "NOUN", "ADP", "NOUN", "PUNCT"],
        "lemmas": ["communication", "build", "strong", "relationship", "over", "time", "."],
        "deps": ["nsubj", "ROOT", "amod", "dobj", "prep", "pobj", "punct"],
        "root_verb": "build",
        "realm": "general",
        "source": "instruct",
        "complexity": 0.45,
    },
]


def main():
    dim = 2048

    print("=" * 60)
    print("SVC Ingestion Pipeline Demo")
    print("Beat DeepSeek V4 Challenge")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 0: Create GPU-aware ingestion engine
    # ------------------------------------------------------------------
    engine = SVCIngestionEngine(dim=dim)
    print(f"\nEngine: {engine.status()}")

    # ------------------------------------------------------------------
    # Step 1: Load SVC entries
    # ------------------------------------------------------------------
    print("\n1. Loading SVC entries...")
    entries = load_svc_entries_from_dicts(SVC_DATA)
    print(f"   Loaded {len(entries)} entries")
    for e in entries:
        print(f"   [{e.realm:>8}] {e.text[:50]}...")

    # ------------------------------------------------------------------
    # Step 2: Ingest into InstantLanguage (GPU-accelerated)
    # ------------------------------------------------------------------
    print("\n2. Ingesting into InstantLanguage...")
    lang = InstantLanguage(dim=dim)
    lang_result = lang.ingest_svc(entries, verbose=False, engine=engine)

    print(f"   Sentences learned: {lang_result.sentences_learned}")
    print(f"   New words encoded: {lang_result.words_encoded}")
    print(f"   Templates learned: {lang_result.templates_learned}")
    print(f"   Realm vectors:     {sorted(lang_result.realm_vectors.keys())}")
    print(f"   Backend:           {lang_result.backend}")

    # ------------------------------------------------------------------
    # Step 3: Ingest into CognitiveController (GPU-accelerated)
    # ------------------------------------------------------------------
    print("\n3. Ingesting into CognitiveController...")
    controller = CognitiveController(dim=dim, confidence_threshold=0.0)
    controller.ingest_svc(entries, engine=engine)

    print(f"   World facts:    {len(controller.world.facts)}")
    print(f"   Causal links:   {len(controller.world.expectations)}")
    print(f"   Sentence memory: {len(controller.language.sentence_memory)}")

    # ------------------------------------------------------------------
    # Step 4: Build realm-routed MoE
    # ------------------------------------------------------------------
    print("\n4. Building realm-routed MoE...")
    realms = sorted(lang_result.realm_vectors.keys())
    realm_fns = {r: (lambda x, _r=r: x) for r in realms}
    moe = ResonatorMoE.from_realm_vectors(dim=dim, realm_expert_fns=realm_fns)

    print(f"   Experts: {realms}")
    for realm in realms:
        indicator = BinaryOps.hash_to_bipolar(realm, dim)
        routed = moe.route(indicator, top_k=1)
        weights = moe.get_weights(indicator, normalize=True)
        top_weight = weights[routed[0]]
        print(f"   Route '{realm}' -> {routed[0]} (weight={top_weight:.3f})")

    # ------------------------------------------------------------------
    # Step 5: Query the system
    # ------------------------------------------------------------------
    print("\n5. Querying the cognitive system...")

    queries = [
        "Exercise is crucial for health",
        "Gravity attracts objects",
        "Weather changes with the seasons",
        "Vaccines prevent diseases",
    ]

    for query in queries:
        print(f'\n   Query: "{query}"')

        # Understand
        understanding = controller.understand(query)
        print(f"   Confidence: {understanding.confidence:.3f}")
        print(f"   Words: {understanding.words}")
        if understanding.inferences:
            print(f"   Inferences: {understanding.inferences[:2]}")

        # Process (generate response)
        response = controller.process(query)
        print(f"   Response: {response}")

    # ------------------------------------------------------------------
    # Step 6: Similarity search
    # ------------------------------------------------------------------
    print("\n6. Similarity search after ingestion...")

    search_query = "health and exercise"
    similar = lang.find_similar_sentences(search_query, top_k=3)
    print(f'   Query: "{search_query}"')
    for words, sim in similar:
        print(f"   {sim:.4f}: {' '.join(words)}")

    # ------------------------------------------------------------------
    # Step 7: World model coherence check
    # ------------------------------------------------------------------
    print("\n7. World model coherence checks...")

    checks = [
        ("exercise", "be", "crucial for maintaining good health"),
        ("vaccines", "prevent", "many infectious diseases effectively"),
        ("unicorns", "fly", "to the moon"),
    ]

    for subj, rel, obj in checks:
        is_known, conf = controller.world.query_fact(subj, rel, obj)
        print(f"   '{subj} {rel} {obj}' -> known={is_known}, conf={conf:.3f}")

    # ------------------------------------------------------------------
    # Step 8: GPU-accelerated similarity search via engine
    # ------------------------------------------------------------------
    print("\n8. Engine-accelerated batch similarity search...")
    all_vecs = np.array([v for v, _ in lang.sentence_memory])
    query_vec = lang.sentence_encoder.encode_sentence(["exercise", "improves", "health"])
    top_results = engine.batch_similarity_search(query_vec, all_vecs, top_k=3)
    print("   Query: 'exercise improves health'")
    for idx, sim in top_results:
        words = lang.sentence_memory[idx][1]
        print(f"   {sim:.4f}: {' '.join(words)}")

    print(f"\n{'=' * 60}")
    print("Pipeline complete! Zero gradient descent. Instant learning.")
    print(f"Backend used: {engine.status()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
