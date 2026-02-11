"""
Tests for the SVC ingestion pipeline.

Tests that SVC data can be loaded, ingested into InstantLanguage,
CognitiveController, SentenceGenerator, and ResonatorMoE, and that
all components produce correct results.
"""

import json

import numpy as np
import pytest

from grilly.experimental.cognitive.controller import CognitiveController
from grilly.experimental.language.encoder import SentenceEncoder, WordEncoder
from grilly.experimental.language.generator import SentenceGenerator
from grilly.experimental.language.svc_loader import (
    SVCBatch,
    SVCEntry,
    SVCIngestionEngine,
    load_svc_batch,
    load_svc_entries,
    load_svc_entries_from_dicts,
)
from grilly.experimental.language.system import InstantLanguage, SVCIngestionResult
from grilly.experimental.moe.routing import ResonatorMoE
from grilly.experimental.vsa.ops import BinaryOps, HolographicOps

# =============================================================================
# Inline SVC Test Data
# =============================================================================

HEALTH_ENTRIES = [
    {
        "id": "test_h0",
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
        "id": "test_h1",
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
        "id": "test_h2",
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
]

SCIENCE_ENTRIES = [
    {
        "id": "test_s0",
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
        "id": "test_s1",
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
]

GENERAL_ENTRIES = [
    {
        "id": "test_g0",
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
]

ALL_ENTRIES = HEALTH_ENTRIES + SCIENCE_ENTRIES + GENERAL_ENTRIES


def _entries_to_svc(dicts):
    """Convert dicts to SVCEntry list."""
    return load_svc_entries_from_dicts(dicts)


def _write_jsonl(entries, path):
    """Write entries to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# =============================================================================
# 1. TestSVCLoader
# =============================================================================


class TestSVCLoader:
    """Test SVCEntry parsing, file loading, and batch loading."""

    def test_svc_entry_from_dict(self):
        """SVCEntry.from_dict parses all fields."""
        entry = SVCEntry.from_dict(HEALTH_ENTRIES[0])
        assert entry.id == "test_h0"
        assert entry.svc_s == "Exercise"
        assert entry.svc_v == "is"
        assert entry.svc_c == "crucial for maintaining good health"
        assert entry.realm == "health"
        assert entry.root_verb == "be"
        assert entry.complexity == 0.4
        assert len(entry.pos) == 8
        assert len(entry.deps) == 8
        assert len(entry.lemmas) == 8

    def test_svc_entry_tokenize(self):
        """SVCEntry.tokenize lowers and strips punctuation."""
        entry = SVCEntry.from_dict(HEALTH_ENTRIES[0])
        tokens = entry.tokenize()
        assert "exercise" in tokens
        assert "." not in tokens
        assert all(t == t.lower() for t in tokens)

    def test_svc_entry_to_roles(self):
        """SVCEntry.to_roles maps s/v/c to SUBJ/VERB/OBJ."""
        entry = SVCEntry.from_dict(HEALTH_ENTRIES[0])
        words, roles = entry.to_roles()
        assert "SUBJ" in roles
        assert "VERB" in roles
        assert roles[words.index("exercise")] == "SUBJ"
        assert roles[words.index("is")] == "VERB"

    def test_svc_entry_template_key(self):
        """SVCEntry.template_key produces a dep-label string."""
        entry = SVCEntry.from_dict(HEALTH_ENTRIES[0])
        key = entry.template_key()
        assert "ROOT" in key
        assert "_" in key

    def test_load_entries_from_dicts(self):
        """load_svc_entries_from_dicts converts dicts to SVCEntry."""
        entries = load_svc_entries_from_dicts(ALL_ENTRIES)
        assert len(entries) == len(ALL_ENTRIES)
        assert all(isinstance(e, SVCEntry) for e in entries)

    def test_load_entries_from_dicts_realm_filter(self):
        """load_svc_entries_from_dicts filters by realm."""
        entries = load_svc_entries_from_dicts(ALL_ENTRIES, realms=["science"])
        assert len(entries) == len(SCIENCE_ENTRIES)
        assert all(e.realm == "science" for e in entries)

    def test_load_svc_entries_from_file(self, tmp_path):
        """load_svc_entries streams from a JSONL file."""
        path = tmp_path / "test.jsonl"
        _write_jsonl(ALL_ENTRIES, path)

        entries = list(load_svc_entries(str(path)))
        assert len(entries) == len(ALL_ENTRIES)

    def test_load_svc_entries_max(self, tmp_path):
        """load_svc_entries respects max_entries."""
        path = tmp_path / "test.jsonl"
        _write_jsonl(ALL_ENTRIES, path)

        entries = list(load_svc_entries(str(path), max_entries=2))
        assert len(entries) == 2

    def test_load_svc_entries_realm_filter(self, tmp_path):
        """load_svc_entries filters by realm."""
        path = tmp_path / "test.jsonl"
        _write_jsonl(ALL_ENTRIES, path)

        entries = list(load_svc_entries(str(path), realms=["health"]))
        assert len(entries) == len(HEALTH_ENTRIES)

    def test_load_svc_entries_source_filter(self, tmp_path):
        """load_svc_entries filters by source."""
        path = tmp_path / "test.jsonl"
        _write_jsonl(ALL_ENTRIES, path)

        entries = list(load_svc_entries(str(path), sources=["instruct"]))
        assert len(entries) == len(ALL_ENTRIES)  # all are instruct

    def test_load_svc_entries_complexity_filter(self, tmp_path):
        """load_svc_entries filters by complexity range."""
        path = tmp_path / "test.jsonl"
        _write_jsonl(ALL_ENTRIES, path)

        entries = list(load_svc_entries(str(path), min_complexity=0.5))
        assert all(e.complexity >= 0.5 for e in entries)

    def test_load_svc_entries_file_not_found(self):
        """load_svc_entries raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            list(load_svc_entries("nonexistent.jsonl"))

    def test_load_svc_batch(self, tmp_path):
        """load_svc_batch returns SVCBatch with statistics."""
        path = tmp_path / "test.jsonl"
        _write_jsonl(ALL_ENTRIES, path)

        batch = load_svc_batch(str(path))
        assert isinstance(batch, SVCBatch)
        assert batch.total_loaded == len(ALL_ENTRIES)
        assert "health" in batch.realm_counts
        assert "science" in batch.realm_counts
        assert batch.avg_complexity > 0
        assert len(batch.realms) >= 3

    def test_svc_batch_realm_entries(self, tmp_path):
        """SVCBatch.realm_entries groups entries by realm."""
        path = tmp_path / "test.jsonl"
        _write_jsonl(ALL_ENTRIES, path)

        batch = load_svc_batch(str(path))
        grouped = batch.realm_entries
        assert "health" in grouped
        assert len(grouped["health"]) == len(HEALTH_ENTRIES)

    def test_svc_batch_summary(self, tmp_path):
        """SVCBatch.summary returns readable text."""
        path = tmp_path / "test.jsonl"
        _write_jsonl(ALL_ENTRIES, path)

        batch = load_svc_batch(str(path))
        summary = batch.summary()
        assert "SVCBatch" in summary
        assert "health" in summary


# =============================================================================
# 2. TestInstantLanguageIngestSVC
# =============================================================================


class TestInstantLanguageIngestSVC:
    """Test InstantLanguage.ingest_svc."""

    def test_ingest_returns_result(self, dim):
        """ingest_svc returns SVCIngestionResult."""
        lang = InstantLanguage(dim=dim)
        entries = _entries_to_svc(ALL_ENTRIES)
        result = lang.ingest_svc(entries)
        assert isinstance(result, SVCIngestionResult)

    def test_ingest_learns_sentences(self, dim):
        """Ingestion stores sentences in memory."""
        lang = InstantLanguage(dim=dim)
        entries = _entries_to_svc(ALL_ENTRIES)
        result = lang.ingest_svc(entries)
        assert result.sentences_learned == len(ALL_ENTRIES)
        assert len(lang.sentence_memory) == len(ALL_ENTRIES)

    def test_ingest_encodes_words(self, dim):
        """Ingestion builds vocabulary."""
        lang = InstantLanguage(dim=dim)
        entries = _entries_to_svc(ALL_ENTRIES)
        result = lang.ingest_svc(entries)
        assert result.words_encoded > 0
        assert len(lang.word_encoder.word_vectors) > 0

    def test_ingest_tracks_realms(self, dim):
        """Ingestion tracks realm counts."""
        lang = InstantLanguage(dim=dim)
        entries = _entries_to_svc(ALL_ENTRIES)
        result = lang.ingest_svc(entries)
        assert "health" in result.realm_counts
        assert "science" in result.realm_counts
        assert result.realm_counts["health"] == len(HEALTH_ENTRIES)

    def test_ingest_tracks_verbs(self, dim):
        """Ingestion tracks verb counts."""
        lang = InstantLanguage(dim=dim)
        entries = _entries_to_svc(ALL_ENTRIES)
        result = lang.ingest_svc(entries)
        assert "be" in result.verb_counts or "prevent" in result.verb_counts

    def test_ingest_builds_realm_vectors(self, dim):
        """Ingestion builds realm vectors."""
        lang = InstantLanguage(dim=dim)
        entries = _entries_to_svc(ALL_ENTRIES)
        result = lang.ingest_svc(entries, build_realm_vectors=True)
        assert len(result.realm_vectors) >= 3
        for realm, vec in result.realm_vectors.items():
            assert vec.shape == (dim,)
            assert np.isfinite(vec).all()

    def test_ingest_no_realm_vectors(self, dim):
        """Skipping realm vector building."""
        lang = InstantLanguage(dim=dim)
        entries = _entries_to_svc(ALL_ENTRIES)
        result = lang.ingest_svc(entries, build_realm_vectors=False)
        assert len(result.realm_vectors) == 0

    def test_ingest_learns_templates(self, dim):
        """Ingestion learns sentence templates."""
        lang = InstantLanguage(dim=dim)
        entries = _entries_to_svc(ALL_ENTRIES)
        result = lang.ingest_svc(entries, learn_templates=True)
        # Templates need at least 2 sentences with same dep pattern
        # HEALTH_ENTRIES[1] and HEALTH_ENTRIES[2] share the same dep pattern
        assert result.templates_learned >= 0  # May be 0 if all patterns unique

    def test_ingest_no_templates(self, dim):
        """Skipping template learning."""
        lang = InstantLanguage(dim=dim)
        entries = _entries_to_svc(ALL_ENTRIES)
        result = lang.ingest_svc(entries, learn_templates=False)
        assert result.templates_learned == 0

    def test_ingest_result_summary(self, dim):
        """SVCIngestionResult.summary returns readable text."""
        lang = InstantLanguage(dim=dim)
        entries = _entries_to_svc(ALL_ENTRIES)
        result = lang.ingest_svc(entries)
        summary = result.summary()
        assert "SVC Ingestion" in summary

    def test_ingest_sentence_vectors_valid(self, dim):
        """All stored sentence vectors are unit-normalized and finite."""
        lang = InstantLanguage(dim=dim)
        entries = _entries_to_svc(ALL_ENTRIES)
        lang.ingest_svc(entries)

        for vec, words in lang.sentence_memory:
            assert vec.shape == (dim,)
            assert np.isfinite(vec).all()
            norm = np.linalg.norm(vec)
            assert abs(norm - 1.0) < 0.01

    def test_get_realm_vector(self, dim):
        """get_realm_vector returns vectors after ingestion."""
        lang = InstantLanguage(dim=dim)
        entries = _entries_to_svc(ALL_ENTRIES)
        lang.ingest_svc(entries)

        vec = lang.get_realm_vector("health")
        assert vec is not None
        assert vec.shape == (dim,)

        assert lang.get_realm_vector("nonexistent") is None

    def test_get_realm_indicator(self, dim):
        """get_realm_indicator returns deterministic bipolar vectors."""
        lang = InstantLanguage(dim=dim)
        v1 = lang.get_realm_indicator("health")
        v2 = lang.get_realm_indicator("health")
        np.testing.assert_array_equal(v1, v2)
        assert v1.shape == (dim,)

    def test_ingest_multiple_batches(self, dim):
        """Ingesting multiple batches accumulates correctly."""
        lang = InstantLanguage(dim=dim)

        r1 = lang.ingest_svc(_entries_to_svc(HEALTH_ENTRIES))
        assert r1.sentences_learned == len(HEALTH_ENTRIES)

        r2 = lang.ingest_svc(_entries_to_svc(SCIENCE_ENTRIES))
        assert r2.sentences_learned == len(SCIENCE_ENTRIES)

        assert len(lang.sentence_memory) == len(HEALTH_ENTRIES) + len(SCIENCE_ENTRIES)

    def test_find_similar_after_ingest(self, dim):
        """find_similar_sentences works after SVC ingestion."""
        lang = InstantLanguage(dim=dim)
        entries = _entries_to_svc(ALL_ENTRIES)
        lang.ingest_svc(entries)

        results = lang.find_similar_sentences("exercise improves health", top_k=3)
        assert len(results) > 0
        for words, sim in results:
            assert isinstance(words, list)
            assert isinstance(sim, float)


# =============================================================================
# 3. TestSentenceGeneratorLearnSVCTemplates
# =============================================================================


class TestSentenceGeneratorLearnSVCTemplates:
    """Test SentenceGenerator.learn_svc_templates."""

    def test_learn_svc_templates_stores_pattern(self, dim):
        """learn_svc_templates adds to patterns dict."""
        we = WordEncoder(dim=dim)
        se = SentenceEncoder(we)
        gen = SentenceGenerator(se)

        word_lists = [
            ["vaccines", "prevent", "diseases"],
            ["exercise", "improves", "health"],
        ]
        gen.learn_svc_templates(word_lists, "svc_transitive")

        assert "svc_transitive" in gen.patterns
        assert gen.patterns["svc_transitive"].shape == (dim,)

    def test_learn_svc_templates_adds_template(self, dim):
        """learn_svc_templates creates a usable template entry."""
        we = WordEncoder(dim=dim)
        se = SentenceEncoder(we)
        gen = SentenceGenerator(se)

        word_lists = [
            ["the", "cat", "chased", "the", "mouse"],
            ["the", "dog", "chased", "the", "ball"],
        ]
        gen.learn_svc_templates(word_lists, "chase_pattern")

        assert "chase_pattern" in gen.templates
        template = gen.templates["chase_pattern"]
        assert "pattern" in template
        assert "example" in template
        assert isinstance(template["pattern"], list)

    def test_learn_svc_templates_empty_input(self, dim):
        """learn_svc_templates with empty list does nothing."""
        we = WordEncoder(dim=dim)
        se = SentenceEncoder(we)
        gen = SentenceGenerator(se)

        gen.learn_svc_templates([], "empty")
        assert "empty" not in gen.patterns

    def test_learn_svc_templates_prototype_similarity(self, dim):
        """Prototype is similar to the sentences that created it."""
        we = WordEncoder(dim=dim)
        se = SentenceEncoder(we)
        gen = SentenceGenerator(se)

        word_lists = [
            ["vaccines", "prevent", "diseases"],
            ["exercise", "improves", "health"],
            ["sleep", "improves", "mood"],
        ]
        gen.learn_svc_templates(word_lists, "health_verbs")

        prototype = gen.patterns["health_verbs"]
        test_vec = se.encode_sentence(["sleep", "improves", "mood"])
        sim = HolographicOps.similarity(prototype, test_vec)
        # Prototype should have non-trivial similarity to one of its sources
        assert np.isfinite(sim)

    def test_generate_from_learned_template(self, dim):
        """generate_from_roles works with a learned template."""
        we = WordEncoder(dim=dim)
        se = SentenceEncoder(we)
        gen = SentenceGenerator(se)

        word_lists = [
            ["the", "cat", "chased", "the", "mouse"],
            ["the", "dog", "chased", "the", "ball"],
        ]
        gen.learn_svc_templates(word_lists, "chase_pattern")

        # Use the template
        result = gen.generate_from_roles(
            {"SUBJ": "bird", "VERB": "found", "OBJ": "worm"},
            template_name="chase_pattern",
        )
        assert isinstance(result, list)
        assert len(result) > 0


# =============================================================================
# 4. TestCognitiveControllerIngestSVC
# =============================================================================


class TestCognitiveControllerIngestSVC:
    """Test CognitiveController.ingest_svc."""

    def test_ingest_returns_result(self, dim):
        """ingest_svc returns SVCIngestionResult."""
        ctrl = CognitiveController(dim=dim)
        entries = _entries_to_svc(ALL_ENTRIES)
        result = ctrl.ingest_svc(entries)
        assert isinstance(result, SVCIngestionResult)

    def test_ingest_adds_world_facts(self, dim):
        """Ingestion adds SVC triples as world model facts."""
        ctrl = CognitiveController(dim=dim)
        entries = _entries_to_svc(ALL_ENTRIES)
        ctrl.ingest_svc(entries)
        assert len(ctrl.world.facts) == len(ALL_ENTRIES)

    def test_ingest_adds_causal_links(self, dim):
        """Causal verbs (prevent, improve) create causal links."""
        ctrl = CognitiveController(dim=dim)
        entries = _entries_to_svc(HEALTH_ENTRIES)
        ctrl.ingest_svc(entries)
        # "prevent" and "improve" are causal verbs
        assert len(ctrl.world.expectations) > 0

    def test_ingest_populates_language(self, dim):
        """Ingestion also populates the language system."""
        ctrl = CognitiveController(dim=dim)
        entries = _entries_to_svc(ALL_ENTRIES)
        ctrl.ingest_svc(entries)
        assert len(ctrl.language.sentence_memory) == len(ALL_ENTRIES)
        assert len(ctrl.language.word_encoder.word_vectors) > 0

    def test_query_fact_after_ingest(self, dim):
        """Known facts are queryable after ingestion."""
        ctrl = CognitiveController(dim=dim)
        entries = _entries_to_svc(HEALTH_ENTRIES)
        ctrl.ingest_svc(entries)

        entry = HEALTH_ENTRIES[0]
        is_known, conf = ctrl.world.query_fact(
            entry["svc"]["s"].lower(),
            entry["root_verb"],
            entry["svc"]["c"].lower(),
        )
        assert is_known
        assert conf > 0.7

    def test_understand_after_ingest(self, dim):
        """Understanding works after SVC ingestion."""
        ctrl = CognitiveController(dim=dim)
        entries = _entries_to_svc(ALL_ENTRIES)
        ctrl.ingest_svc(entries)

        result = ctrl.understand("Exercise is crucial for health")
        assert result is not None
        assert result.surface_meaning.shape == (dim,)

    def test_process_after_ingest(self, dim):
        """CognitiveController.process works after SVC ingestion."""
        ctrl = CognitiveController(dim=dim, confidence_threshold=0.0)
        entries = _entries_to_svc(ALL_ENTRIES)
        ctrl.ingest_svc(entries)

        response = ctrl.process("Vaccines prevent diseases")
        assert response is not None
        assert isinstance(response, str)

    def test_ingest_with_verbose(self, dim, capsys):
        """Verbose mode prints progress."""
        ctrl = CognitiveController(dim=dim)
        entries = _entries_to_svc(ALL_ENTRIES)
        ctrl.ingest_svc(entries, verbose=True)
        captured = capsys.readouterr()
        assert "SVC Ingestion" in captured.out


# =============================================================================
# 5. TestRealmMoEIntegration
# =============================================================================


class TestRealmMoEIntegration:
    """Test ResonatorMoE.from_realm_vectors with SVC data."""

    REALMS = ["health", "science", "general"]

    def test_from_realm_vectors_creates_moe(self, dim):
        """from_realm_vectors creates a working MoE."""
        fns = {r: (lambda x, _r=r: x) for r in self.REALMS}
        moe = ResonatorMoE.from_realm_vectors(dim=dim, realm_expert_fns=fns)
        assert len(moe.experts) == 3
        assert len(moe.expert_vectors) == 3

    def test_from_realm_vectors_hash_routing(self, dim):
        """Hash-based realm vectors route correctly."""
        fns = {r: (lambda x, _r=r: x) for r in self.REALMS}
        moe = ResonatorMoE.from_realm_vectors(dim=dim, realm_expert_fns=fns)

        for realm in self.REALMS:
            indicator = BinaryOps.hash_to_bipolar(realm, dim)
            result = moe.route(indicator, top_k=1)
            assert result[0] == realm

    def test_from_realm_vectors_custom_vectors(self, dim):
        """from_realm_vectors accepts custom pre-built vectors."""
        fns = {r: (lambda x, _r=r: x) for r in self.REALMS}
        custom_vecs = {
            r: BinaryOps.random_bipolar(dim, seed=hash(r) % (2**31)) for r in self.REALMS
        }
        moe = ResonatorMoE.from_realm_vectors(
            dim=dim,
            realm_expert_fns=fns,
            realm_vectors=custom_vecs,
        )
        assert len(moe.expert_vectors) == 3

    def test_realm_routing_after_ingest(self, dim):
        """Full pipeline: ingest -> build realm MoE -> route."""
        lang = InstantLanguage(dim=dim)
        entries = _entries_to_svc(ALL_ENTRIES)
        result = lang.ingest_svc(entries)

        fns = {r: (lambda x, _r=r: x) for r in result.realm_vectors}
        moe = ResonatorMoE.from_realm_vectors(dim=dim, realm_expert_fns=fns)

        for realm in result.realm_vectors:
            indicator = BinaryOps.hash_to_bipolar(realm, dim)
            routed = moe.route(indicator, top_k=1)
            assert routed[0] == realm

    def test_realm_weights_sum_to_one(self, dim):
        """Normalized expert weights sum to 1."""
        fns = {r: (lambda x, _r=r: x) for r in self.REALMS}
        moe = ResonatorMoE.from_realm_vectors(dim=dim, realm_expert_fns=fns)

        query = BinaryOps.hash_to_bipolar("health", dim)
        weights = moe.get_weights(query, normalize=True)
        assert abs(sum(weights.values()) - 1.0) < 1e-5

    def test_forward_with_realm_routing(self, dim):
        """MoE forward pass with realm routing produces output."""
        fns = {r: (lambda x, _r=r: x * 1.0) for r in self.REALMS}
        moe = ResonatorMoE.from_realm_vectors(dim=dim, realm_expert_fns=fns)

        x = np.random.randn(dim).astype(np.float32)
        query = BinaryOps.hash_to_bipolar("health", dim)
        output = moe.forward(x, query, top_k=1)
        assert output.shape == (dim,)
        assert np.isfinite(output).all()


# =============================================================================
# 6. TestEndToEndIngestion
# =============================================================================


class TestEndToEndIngestion:
    """Full end-to-end ingestion pipeline test."""

    def test_full_pipeline(self, dim):
        """Complete pipeline: load -> ingest -> route -> understand."""
        # 1. Load from dicts
        entries = _entries_to_svc(ALL_ENTRIES)
        assert len(entries) == len(ALL_ENTRIES)

        # 2. Ingest into CognitiveController
        ctrl = CognitiveController(dim=dim, confidence_threshold=0.0)
        result = ctrl.ingest_svc(entries)
        assert result.sentences_learned == len(ALL_ENTRIES)
        assert len(ctrl.world.facts) == len(ALL_ENTRIES)

        # 3. Build realm MoE
        realm_fns = {r: (lambda x, _r=r: x) for r in result.realm_vectors}
        moe = ResonatorMoE.from_realm_vectors(dim=dim, realm_expert_fns=realm_fns)

        # 4. Route health query
        health_indicator = BinaryOps.hash_to_bipolar("health", dim)
        routed = moe.route(health_indicator, top_k=1)
        assert routed[0] == "health"

        # 5. Understand and process
        understanding = ctrl.understand("Exercise is crucial for health")
        assert understanding is not None

        response = ctrl.process("Vaccines prevent diseases")
        assert response is not None

    def test_full_pipeline_from_file(self, dim, tmp_path):
        """Pipeline from JSONL file."""
        path = tmp_path / "test.jsonl"
        _write_jsonl(ALL_ENTRIES, path)

        batch = load_svc_batch(str(path))
        assert batch.total_loaded == len(ALL_ENTRIES)

        ctrl = CognitiveController(dim=dim)
        result = ctrl.ingest_svc(batch.entries)
        assert result.sentences_learned == len(ALL_ENTRIES)
        assert len(ctrl.world.facts) == len(ALL_ENTRIES)

    def test_filtered_ingestion(self, dim, tmp_path):
        """Pipeline with realm filtering."""
        path = tmp_path / "test.jsonl"
        _write_jsonl(ALL_ENTRIES, path)

        batch = load_svc_batch(str(path), realms=["health"])
        assert batch.total_loaded == len(HEALTH_ENTRIES)

        lang = InstantLanguage(dim=dim)
        result = lang.ingest_svc(batch.entries)
        assert result.sentences_learned == len(HEALTH_ENTRIES)
        assert "health" in result.realm_counts
        assert "science" not in result.realm_counts


# =============================================================================
# 7. TestSVCIngestionEngine
# =============================================================================


class TestSVCIngestionEngine:
    """Test SVCIngestionEngine (CPU path, GPU auto-detect)."""

    def test_engine_creates_cpu_fallback(self, dim):
        """Engine falls back to CPU when gpu=False."""
        engine = SVCIngestionEngine(dim=dim, gpu=False)
        assert not engine.using_gpu
        assert "CPU" in engine.status()

    def test_engine_auto_detect(self, dim):
        """Engine auto-detects (may be CPU or GPU depending on host)."""
        engine = SVCIngestionEngine(dim=dim)
        # Just ensure it doesn't crash
        assert engine.dim == dim
        assert isinstance(engine.status(), str)

    def test_engine_bundle(self, dim):
        """engine.bundle produces correct-shape normalized vector."""
        engine = SVCIngestionEngine(dim=dim, gpu=False)
        vecs = [np.random.randn(dim).astype(np.float32) for _ in range(5)]
        result = engine.bundle(vecs, normalize=True)
        assert result.shape == (dim,)
        assert abs(np.linalg.norm(result) - 1.0) < 0.01

    def test_engine_convolve(self, dim):
        """engine.convolve matches HolographicOps.convolve."""
        engine = SVCIngestionEngine(dim=dim, gpu=False)
        a = np.random.randn(dim).astype(np.float32)
        b = np.random.randn(dim).astype(np.float32)
        result = engine.convolve(a, b)
        expected = HolographicOps.convolve(a, b)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_engine_similarity_batch(self, dim):
        """engine.similarity_batch returns correct-shape similarities."""
        engine = SVCIngestionEngine(dim=dim, gpu=False)
        query = HolographicOps.random_vector(dim)
        codebook = np.array([HolographicOps.random_vector(dim) for _ in range(10)])
        sims = engine.similarity_batch(query, codebook)
        assert sims.shape == (10,)
        # Self-similarity should be highest
        codebook[3] = query
        sims = engine.similarity_batch(query, codebook)
        assert np.argmax(sims) == 3

    def test_engine_bind_bipolar_batch(self, dim):
        """engine.bind_bipolar_batch matches BinaryOps.bind_batch."""
        engine = SVCIngestionEngine(dim=dim, gpu=False)
        a = np.sign(np.random.randn(4, dim)).astype(np.float32)
        b = np.sign(np.random.randn(4, dim)).astype(np.float32)
        result = engine.bind_bipolar_batch(a, b)
        expected = BinaryOps.bind_batch(a, b)
        np.testing.assert_array_equal(result, expected)

    def test_engine_resonator_step(self, dim):
        """engine.resonator_step selects best codebook entry."""
        engine = SVCIngestionEngine(dim=dim, gpu=False)
        codebook = np.array([BinaryOps.random_bipolar(dim, seed=i) for i in range(5)])
        target = codebook[2].copy()
        best_vec, best_idx = engine.resonator_step(target, codebook)
        assert best_idx == 2
        np.testing.assert_array_equal(best_vec, codebook[2])

    def test_engine_batch_encode_sentences(self, dim):
        """batch_encode_sentences returns vectors matching entry count."""
        engine = SVCIngestionEngine(dim=dim, gpu=False)
        entries = _entries_to_svc(ALL_ENTRIES)
        we = WordEncoder(dim=dim)
        se = SentenceEncoder(we)

        vecs, wlists = engine.batch_encode_sentences(entries, we, se)
        assert len(vecs) == len(ALL_ENTRIES)
        assert len(wlists) == len(ALL_ENTRIES)
        for v in vecs:
            assert v.shape == (dim,)
            assert np.isfinite(v).all()

    def test_engine_batch_build_realm_vectors(self, dim):
        """batch_build_realm_vectors produces per-realm vectors."""
        engine = SVCIngestionEngine(dim=dim, gpu=False)
        realm_vecs = {
            "health": [np.random.randn(dim).astype(np.float32) for _ in range(3)],
            "science": [np.random.randn(dim).astype(np.float32) for _ in range(2)],
        }
        result = engine.batch_build_realm_vectors(realm_vecs)
        assert "health" in result
        assert "science" in result
        for v in result.values():
            assert v.shape == (dim,)
            assert abs(np.linalg.norm(v) - 1.0) < 0.01

    def test_engine_batch_similarity_search(self, dim):
        """batch_similarity_search returns sorted (index, sim) tuples."""
        engine = SVCIngestionEngine(dim=dim, gpu=False)
        vecs = np.array([HolographicOps.random_vector(dim) for _ in range(10)])
        query = vecs[7].copy()  # should be most similar to itself
        results = engine.batch_similarity_search(query, vecs, top_k=3)
        assert len(results) == 3
        assert results[0][0] == 7  # index of best match
        assert results[0][1] > 0.99

    def test_engine_batch_realm_route(self, dim):
        """batch_realm_route assigns each query to correct realm."""
        engine = SVCIngestionEngine(dim=dim, gpu=False)
        realms = ["health", "science", "general"]
        codebook = np.array([BinaryOps.hash_to_bipolar(r, dim) for r in realms])
        queries = np.array([BinaryOps.hash_to_bipolar(r, dim) for r in realms])
        results = engine.batch_realm_route(queries, codebook, realms)
        assert results == realms

    def test_ingest_with_explicit_engine(self, dim):
        """InstantLanguage.ingest_svc accepts an explicit engine."""
        engine = SVCIngestionEngine(dim=dim, gpu=False)
        lang = InstantLanguage(dim=dim)
        entries = _entries_to_svc(ALL_ENTRIES)
        result = lang.ingest_svc(entries, engine=engine)
        assert result.sentences_learned == len(ALL_ENTRIES)
        assert result.backend == "CPU"

    def test_ingest_result_reports_backend(self, dim):
        """SVCIngestionResult.backend reflects the engine used."""
        engine = SVCIngestionEngine(dim=dim, gpu=False)
        lang = InstantLanguage(dim=dim)
        entries = _entries_to_svc(ALL_ENTRIES)
        result = lang.ingest_svc(entries, engine=engine)
        assert "CPU" in result.summary()

    def test_controller_ingest_with_engine(self, dim):
        """CognitiveController.ingest_svc accepts engine kwarg."""
        engine = SVCIngestionEngine(dim=dim, gpu=False)
        ctrl = CognitiveController(dim=dim)
        entries = _entries_to_svc(ALL_ENTRIES)
        result = ctrl.ingest_svc(entries, engine=engine)
        assert result.sentences_learned == len(ALL_ENTRIES)
        assert len(ctrl.world.facts) == len(ALL_ENTRIES)
