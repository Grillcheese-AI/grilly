"""
InstantLanguage - Complete instant language learning system.

Combines word encoding, sentence encoding, parsing, and generation.
Uses VulkanVSA GPU shaders when available for batch VSA operations.
"""

import re
from collections import defaultdict
from typing import TYPE_CHECKING, Optional

import numpy as np

from grilly.experimental.language.encoder import SentenceEncoder, WordEncoder
from grilly.experimental.language.generator import SentenceGenerator
from grilly.experimental.language.parser import ResonatorParser
from grilly.experimental.vsa.ops import BinaryOps, HolographicOps

if TYPE_CHECKING:
    from grilly.experimental.language.svc_loader import SVCEntry, SVCIngestionEngine


class SVCIngestionResult:
    """Result of ingesting SVC data into the language system."""

    def __init__(self) -> None:
        """Initialize the instance."""

        self.sentences_learned: int = 0
        self.words_encoded: int = 0
        self.templates_learned: int = 0
        self.realm_counts: dict[str, int] = defaultdict(int)
        self.verb_counts: dict[str, int] = defaultdict(int)
        self.realm_vectors: dict[str, np.ndarray] = {}
        self.backend: str = "cpu"

    def summary(self) -> str:
        """Execute summary."""

        lines = [
            f"SVC Ingestion: {self.sentences_learned} sentences, {self.words_encoded} unique words",
            f"  Templates learned: {self.templates_learned}",
            f"  Realms: {dict(self.realm_counts)}",
            f"  Top verbs: {dict(sorted(self.verb_counts.items(), key=lambda x: -x[1])[:10])}",
            f"  Backend: {self.backend}",
        ]
        return "\n".join(lines)


class InstantLanguage:
    """
    Complete instant language learning system.

    Combines:
    - Word encoding (n-grams, no training)
    - Relation extraction (O(1))
    - Sentence encoding (composition)
    - Parsing (resonator factorization)
    - Generation (template filling)

    Everything is instant - no gradient descent!
    """

    DEFAULT_DIM = 4096

    def __init__(self, dim: int = DEFAULT_DIM, *, word_use_ngrams: bool = True):
        """Initialize the instance."""

        self.dim = dim

        # Components
        # N-gram HRR word vectors are *very* expensive to build at scale.
        # For large ingestion runs you can disable them (word_use_ngrams=False)
        # to switch to fast hashed vectors.
        self.word_encoder = WordEncoder(dim=dim, use_ngrams=word_use_ngrams)
        self.sentence_encoder = SentenceEncoder(self.word_encoder)
        self.generator = SentenceGenerator(self.sentence_encoder)
        self.parser = ResonatorParser(self.sentence_encoder)

        # Memory
        self.sentence_memory: list[tuple[np.ndarray, list[str]]] = []
        self.relation_memory: dict[str, list[tuple[str, str]]] = defaultdict(list)

        # Realm vectors built from SVC data
        self.realm_vectors: dict[str, np.ndarray] = {}
        # Realm sentence accumulators (bundled sentence vectors per realm)
        self._realm_accumulators: dict[str, list[np.ndarray]] = defaultdict(list)

    def learn_sentence(self, sentence: str) -> np.ndarray:
        """
        Learn a sentence instantly.

        Encodes and stores in memory. No training loop!
        """
        words = self._tokenize(sentence)

        # Encode all words (builds vocabulary on the fly)
        for word in words:
            self.word_encoder.encode_word(word)

        # Encode sentence
        sent_vec = self.sentence_encoder.encode_sentence(words)

        # Store
        self.sentence_memory.append((sent_vec, words))

        return sent_vec

    def learn_relation(self, word_a: str, word_b: str, relation: str):
        """
        Learn a word relation from a single example.
        """
        self.relation_memory[relation].append((word_a, word_b))

        # If we have enough examples, create a relation prototype
        if len(self.relation_memory[relation]) >= 2:
            pairs = self.relation_memory[relation]
            self.word_encoder.learn_relation(pairs, relation)

    def query_relation(self, word: str, relation: str) -> list[tuple[str, float]]:
        """
        Query: "What is the [relation] of [word]?"

        E.g., "What is the antonym of hot?" -> cold
        """
        if relation not in self.word_encoder.relations:
            return []

        rel_vec = self.word_encoder.relations[relation]
        result_vec = self.word_encoder.apply_relation(word, rel_vec)
        return self.word_encoder.find_closest(result_vec)

    def express_relation(self, word_a: str, relation: str, word_b: str) -> str:
        """
        Generate a sentence expressing a relation.
        """
        words = self.generator.generate_from_relation(word_a, relation, word_b)
        return " ".join(words)

    def parse_sentence(self, sentence: str) -> list[tuple[str, str, float]]:
        """
        Parse a sentence into word-role pairs.
        """
        words = self._tokenize(sentence)

        # First encode
        for word in words:
            self.word_encoder.encode_word(word)

        sent_vec = self.sentence_encoder.encode_sentence(words)

        return self.parser.parse(sent_vec, num_slots=len(words))

    def find_similar_sentences(self, query: str, top_k: int = 5) -> list[tuple[list[str], float]]:
        """
        Find similar sentences in memory.
        """
        query_words = self._tokenize(query)
        query_vec = self.sentence_encoder.encode_sentence(query_words)

        results = []
        for sent_vec, words in self.sentence_memory:
            sim = HolographicOps.similarity(query_vec, sent_vec)
            results.append((words, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def complete(self, partial: str, role: str = "OBJ") -> list[tuple[str, float]]:
        """
        Complete a partial sentence.

        "The dog chased the ___" -> find best OBJ
        """
        words = self._tokenize(partial)

        # Encode what we have
        for word in words:
            self.word_encoder.encode_word(word)

        sent_vec = self.sentence_encoder.encode_sentence(words)

        return self.sentence_encoder.find_role_filler(sent_vec, role)

    def analogy(self, word_a: str, word_b: str, word_c: str) -> list[tuple[str, float]]:
        """
        Solve analogy: A is to B as C is to ?

        E.g., king:queen :: man:? -> woman
        """
        # Extract A:B relation
        relation = self.word_encoder.extract_relation(word_a, word_b)

        # Apply to C
        c_vec = self.word_encoder.encode_word(word_c)
        d_vec = HolographicOps.convolve(c_vec, relation)

        return self.word_encoder.find_closest(d_vec)

    def _get_engine(
        self,
        engine: Optional["SVCIngestionEngine"] = None,
    ) -> "SVCIngestionEngine":
        """Return the provided engine or create a default one."""
        if engine is not None:
            return engine
        from grilly.experimental.language.svc_loader import SVCIngestionEngine

        return SVCIngestionEngine(dim=self.dim)

    def ingest_svc(
        self,
        entries: list["SVCEntry"],
        learn_templates: bool = True,
        build_realm_vectors: bool = True,
        verbose: bool = False,
        engine: Optional["SVCIngestionEngine"] = None,
        progress_every: int = 10_000,
        max_templates: int = 512,
        template_examples_per_key: int = 32,
    ) -> SVCIngestionResult:
        """
        Ingest SVC data into the language system.

        When a GPU is available the heavy VSA operations (convolve,
        bundle, similarity_batch) are dispatched to Vulkan compute
        shaders via :class:`SVCIngestionEngine`.  Pass ``engine`` to
        control backend selection, or leave ``None`` for auto-detect.

        Pipeline per entry:
        1. Encode all words (builds vocabulary).
        2. Bind word ⊗ role ⊗ position via ``engine.convolve`` (GPU).
        3. Bundle components into sentence vector via ``engine.bundle`` (GPU).
        4. Store in sentence memory.
        5. Accumulate realm vectors.
        6. Learn sentence templates from dependency patterns.
        7. Bundle realm vectors via ``engine.batch_build_realm_vectors`` (GPU).

        Args:
            entries: List of SVCEntry instances.
            learn_templates: If True, learn sentence templates from dep patterns.
            build_realm_vectors: If True, build realm expert vectors.
            verbose: If True, print progress.
            engine: Optional SVCIngestionEngine (auto-created if None).

        Returns:
            SVCIngestionResult with statistics.
        """
        eng = self._get_engine(engine)
        result = SVCIngestionResult()
        result.backend = "VulkanVSA GPU" if eng.using_gpu else "CPU"
        words_before = len(self.word_encoder.word_vectors)

        if verbose:
            print(f"  Backend: {eng.status()}")

        # -- Step 1: Batch encode sentences via engine --------------------
        if verbose:
            print(f"  Encoding {len(entries)} sentences...")
        sentence_vecs, word_lists = eng.batch_encode_sentences(
            entries,
            self.word_encoder,
            self.sentence_encoder,
        )
        if verbose:
            print(f"  Encoded {len(sentence_vecs)} sentences")

        # -- Step 2: Store + accumulate stats ----------------------------
        if verbose:
            print(f"  Storing {len(entries)} sentences...")

        # Template learning can explode in memory on large corpora.
        # We keep *counts* for all keys, but only store a bounded number
        # of examples per key for template induction.
        template_counts: dict[str, int] = defaultdict(int)
        template_examples: dict[str, list[list[str]]] = defaultdict(list)

        for i, entry in enumerate(entries):
            sent_vec = sentence_vecs[i]
            words = word_lists[i]

            self.sentence_memory.append((sent_vec, words))
            result.sentences_learned += 1
            result.realm_counts[entry.realm] += 1
            result.verb_counts[entry.root_verb] += 1

            if build_realm_vectors and entry.realm:
                self._realm_accumulators[entry.realm].append(sent_vec)

            if learn_templates and entry.deps:
                tkey = entry.template_key()
                template_counts[tkey] += 1
                if len(template_examples[tkey]) < template_examples_per_key:
                    template_examples[tkey].append(words)

            if verbose and progress_every and (i + 1) % progress_every == 0:
                print(f"  Ingested {i + 1}/{len(entries)} entries...")

        # -- Step 3: Learn templates ------------------------------------
        if learn_templates:
            # Choose the most frequent template keys, but learn from a
            # bounded sample to keep runtime and memory predictable.
            keys = sorted(template_counts.keys(), key=lambda k: -template_counts[k])
            keys = keys[:max_templates] if max_templates else keys
            if verbose:
                print(f"  Learning {len(keys)} templates (of {len(template_counts)})...")
            for tkey in keys:
                wlists = template_examples.get(tkey, [])
                if len(wlists) >= 2:
                    self.generator.learn_svc_templates(wlists, tkey)
                    result.templates_learned += 1

        # -- Step 4: Build realm vectors (GPU bundle) -------------------
        if build_realm_vectors:
            if verbose:
                print(f"  Building {len(self._realm_accumulators)} realm vectors...")
            self.realm_vectors.update(eng.batch_build_realm_vectors(dict(self._realm_accumulators)))
            result.realm_vectors = dict(self.realm_vectors)
            if verbose:
                print(f"  Built {len(result.realm_vectors)} realm vectors")

        result.words_encoded = len(self.word_encoder.word_vectors) - words_before

        if verbose:
            print(result.summary())

        return result

    def get_realm_vector(self, realm: str) -> np.ndarray | None:
        """Get the bundled prototype vector for a realm.

        Returns None if no entries for that realm have been ingested.
        """
        return self.realm_vectors.get(realm)

    def get_realm_indicator(self, realm: str) -> np.ndarray:
        """Get a deterministic bipolar indicator vector for a realm name.

        This uses hash-based generation so the same realm always
        produces the same vector, even before any data is ingested.
        """
        return BinaryOps.hash_to_bipolar(realm, self.dim)

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        # Remove punctuation, split on whitespace
        text = re.sub(r"[^\w\s]", "", text.lower())
        return text.split()
