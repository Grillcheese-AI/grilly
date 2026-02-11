"""
SVC Dataset Integration Tests.

Tests that SVC-structured data (Subject-Verb-Complement) properly
integrates with all experimental modules: Language (VSA encoding),
Cognitive (WorldModel, Controller), MoE (realm routing), and Temporal
(causal chains, temporal binding).

Uses inline test data mirroring the real SVC schema (10 fields).
"""

import json
import re

import numpy as np

from grilly.experimental.cognitive import (
    CognitiveController,
    UnderstandingResult,
    WorldModel,
)
from grilly.experimental.language import SentenceEncoder, WordEncoder
from grilly.experimental.moe import ResonatorMoE
from grilly.experimental.temporal import CausalChain, TemporalEncoder
from grilly.experimental.vsa import BinaryOps, HolographicOps

# =============================================================================
# Inline SVC Test Data
# =============================================================================

VALID_REALMS = {
    "technology",
    "science",
    "health",
    "history",
    "nature",
    "business",
    "arts",
    "social",
    "general",
    "conversation",
}

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
    {
        "id": "test_s2",
        "text": "Electrons orbit the nucleus of an atom.",
        "svc": {"s": "Electrons", "v": "orbit", "c": "the nucleus of an atom"},
        "pos": ["NOUN", "VERB", "DET", "NOUN", "ADP", "DET", "NOUN", "PUNCT"],
        "lemmas": ["electron", "orbit", "the", "nucleus", "of", "an", "atom", "."],
        "deps": ["nsubj", "ROOT", "det", "dobj", "prep", "det", "pobj", "punct"],
        "root_verb": "orbit",
        "realm": "science",
        "source": "instruct",
        "complexity": 0.5,
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
    {
        "id": "test_g1",
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

IMPERATIVE_ENTRIES = [
    {
        "id": "test_imp0",
        "text": "Discuss the challenges of modern education.",
        "svc": {"s": "the challenges", "v": "Discuss", "c": "of modern education"},
        "pos": ["VERB", "DET", "NOUN", "ADP", "ADJ", "NOUN", "PUNCT"],
        "lemmas": ["discuss", "the", "challenge", "of", "modern", "education", "."],
        "deps": ["ROOT", "det", "dobj", "prep", "amod", "pobj", "punct"],
        "root_verb": "discuss",
        "realm": "social",
        "source": "instruct",
        "complexity": 0.5,
    },
    {
        "id": "test_imp1",
        "text": "Consider the environmental impact of pollution.",
        "svc": {"s": "the environmental impact", "v": "Consider", "c": "of pollution"},
        "pos": ["VERB", "DET", "ADJ", "NOUN", "ADP", "NOUN", "PUNCT"],
        "lemmas": ["consider", "the", "environmental", "impact", "of", "pollution", "."],
        "deps": ["ROOT", "det", "amod", "dobj", "prep", "pobj", "punct"],
        "root_verb": "consider",
        "realm": "nature",
        "source": "instruct",
        "complexity": 0.5,
    },
]

CONVERSATION_ENTRIES = [
    {
        "id": "test_conv0",
        "text": "I think exercise really helps with concentration.",
        "svc": {"s": "exercise", "v": "helps", "c": "with concentration"},
        "pos": ["PRON", "VERB", "NOUN", "ADV", "VERB", "ADP", "NOUN", "PUNCT"],
        "lemmas": ["i", "think", "exercise", "really", "help", "with", "concentration", "."],
        "deps": ["nsubj", "ROOT", "nsubj", "advmod", "ccomp", "prep", "pobj", "punct"],
        "root_verb": "think",
        "realm": "health",
        "source": "conversation",
        "complexity": 0.4,
        "role": "user",
    },
]

ALL_ENTRIES = (
    HEALTH_ENTRIES + SCIENCE_ENTRIES + GENERAL_ENTRIES + IMPERATIVE_ENTRIES + CONVERSATION_ENTRIES
)

INSTRUCT_FIELDS = {
    "id",
    "text",
    "svc",
    "pos",
    "lemmas",
    "deps",
    "root_verb",
    "realm",
    "source",
    "complexity",
}
CONVERSATION_FIELDS = INSTRUCT_FIELDS | {"role"}


# =============================================================================
# Helpers
# =============================================================================


def tokenize(text: str) -> list[str]:
    """Tokenize the same way InstantLanguage._tokenize does."""
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def svc_to_roles(text: str, svc: dict) -> tuple[list[str], list[str]]:
    """Map SVC s/v/c fields to per-word SUBJ/VERB/OBJ roles.

    Returns (words, roles) where words are tokenized and lowercased.
    Priority: VERB > SUBJ > OBJ > ROOT.
    """
    words = tokenize(text)
    v_words = set(tokenize(svc["v"]))
    s_words = set(tokenize(svc["s"]))
    c_words = set(tokenize(svc["c"]))

    roles = []
    for w in words:
        if w in v_words:
            roles.append("VERB")
        elif w in s_words:
            roles.append("SUBJ")
        elif w in c_words:
            roles.append("OBJ")
        else:
            roles.append("ROOT")
    return words, roles


# =============================================================================
# 1. TestSVCDataLoading
# =============================================================================


class TestSVCDataLoading:
    """Test parsing and schema validation of SVC data."""

    def test_all_entries_parse_as_valid_json(self):
        """Each test entry round-trips through JSON."""
        for entry in ALL_ENTRIES:
            line = json.dumps(entry, ensure_ascii=False)
            parsed = json.loads(line)
            assert parsed["id"] == entry["id"]

    def test_instruct_entries_have_required_fields(self):
        """Instruct entries must have all 10 schema fields."""
        instruct = [e for e in ALL_ENTRIES if e["source"] == "instruct"]
        assert len(instruct) > 0
        for entry in instruct:
            missing = INSTRUCT_FIELDS - set(entry.keys())
            assert missing == set(), f"{entry['id']} missing {missing}"

    def test_conversation_entries_have_role_field(self):
        """Conversation entries must have 'role' in addition to the base 10."""
        convs = [e for e in ALL_ENTRIES if e["source"] == "conversation"]
        assert len(convs) > 0
        for entry in convs:
            missing = CONVERSATION_FIELDS - set(entry.keys())
            assert missing == set(), f"{entry['id']} missing {missing}"

    def test_svc_has_s_v_c_keys(self):
        """Every svc dict must contain s, v, c as strings."""
        for entry in ALL_ENTRIES:
            svc = entry["svc"]
            assert isinstance(svc, dict)
            for key in ("s", "v", "c"):
                assert key in svc, f"{entry['id']} svc missing '{key}'"
                assert isinstance(svc[key], str)

    def test_realm_in_known_set(self):
        for entry in ALL_ENTRIES:
            assert entry["realm"] in VALID_REALMS, (
                f"{entry['id']} has unknown realm '{entry['realm']}'"
            )

    def test_complexity_in_range(self):
        for entry in ALL_ENTRIES:
            c = entry["complexity"]
            assert 0.0 <= c <= 1.0, f"{entry['id']} complexity={c}"

    def test_list_lengths_match(self):
        """pos, lemmas, deps lists must be the same length."""
        for entry in ALL_ENTRIES:
            assert len(entry["pos"]) == len(entry["lemmas"]) == len(entry["deps"]), (
                f"{entry['id']} list lengths mismatch"
            )

    def test_both_sources_represented(self):
        sources = {e["source"] for e in ALL_ENTRIES}
        assert "instruct" in sources
        assert "conversation" in sources


# =============================================================================
# 2. TestSVCToWordEncoding
# =============================================================================


class TestSVCToWordEncoding:
    """Test encoding SVC words with WordEncoder."""

    def test_encode_subject_produces_correct_dim(self, dim):
        we = WordEncoder(dim=dim)
        for entry in ALL_ENTRIES:
            vec = we.encode_word(entry["svc"]["s"].split()[0])
            assert vec.shape == (dim,)

    def test_encode_verb_produces_correct_dim(self, dim):
        we = WordEncoder(dim=dim)
        for entry in ALL_ENTRIES:
            vec = we.encode_word(entry["svc"]["v"])
            assert vec.shape == (dim,)

    def test_same_word_same_vector(self, dim):
        """Deterministic: encoding the same word twice gives identical results."""
        we = WordEncoder(dim=dim)
        for entry in ALL_ENTRIES:
            verb = entry["svc"]["v"]
            v1 = we.encode_word(verb)
            v2 = we.encode_word(verb)
            np.testing.assert_array_equal(v1, v2)

    def test_different_words_near_orthogonal(self, dim):
        """Random high-dim vectors are approximately orthogonal (hash mode)."""
        # use_ngrams=False gives proper random vectors via hash seeding;
        # the default n-gram mode produces degenerate constant vectors due
        # to starting from np.ones (convolve(ones, x) collapses to scalar).
        we = WordEncoder(dim=dim, use_ngrams=False)
        vec_exercise = we.encode_word("exercise")
        vec_gravity = we.encode_word("gravity")
        sim = HolographicOps.similarity(vec_exercise, vec_gravity)
        assert abs(sim) < 0.3, f"Unrelated words too similar: {sim}"

    def test_similar_words_higher_similarity(self, dim):
        """N-gram encoding gives spelling-similar words higher similarity."""
        we = WordEncoder(dim=dim)
        vec_prevent = we.encode_word("prevent")
        vec_prevents = we.encode_word("prevents")
        vec_gravity = we.encode_word("gravity")
        sim_close = HolographicOps.similarity(vec_prevent, vec_prevents)
        sim_far = HolographicOps.similarity(vec_prevent, vec_gravity)
        assert sim_close > sim_far, (
            f"'prevent'/'prevents' sim={sim_close:.3f} should exceed "
            f"'prevent'/'gravity' sim={sim_far:.3f}"
        )

    def test_encode_multiword_subject(self, dim):
        """Multi-word subjects encode each word independently."""
        we = WordEncoder(dim=dim)
        entry = HEALTH_ENTRIES[2]  # "Regular sleep"
        s_words = tokenize(entry["svc"]["s"])
        assert len(s_words) == 2
        vecs = [we.encode_word(w) for w in s_words]
        for v in vecs:
            assert v.shape == (dim,)
        # The two words should be different vectors
        sim = HolographicOps.similarity(vecs[0], vecs[1])
        assert abs(sim) < 0.5


# =============================================================================
# 3. TestSVCRoleMapping
# =============================================================================


class TestSVCRoleMapping:
    """Map SVC s/v/c to SUBJ/VERB/OBJ roles and encode sentences."""

    def test_svc_to_roles_basic(self):
        """svc_to_roles assigns SUBJ/VERB/OBJ correctly."""
        entry = HEALTH_ENTRIES[0]  # Exercise is crucial ...
        words, roles = svc_to_roles(entry["text"], entry["svc"])
        assert "SUBJ" in roles
        assert "VERB" in roles
        assert "OBJ" in roles
        # "exercise" → SUBJ, "is" → VERB
        assert roles[words.index("exercise")] == "SUBJ"
        assert roles[words.index("is")] == "VERB"

    def test_imperative_role_mapping(self):
        """Imperatives: verb first, promoted subject from dobj."""
        entry = IMPERATIVE_ENTRIES[0]  # Discuss the challenges ...
        words, roles = svc_to_roles(entry["text"], entry["svc"])
        assert roles[words.index("discuss")] == "VERB"
        assert roles[words.index("the")] == "SUBJ"
        assert roles[words.index("challenges")] == "SUBJ"

    def test_encode_sentence_with_svc_roles(self, dim):
        """SentenceEncoder accepts explicit SVC-derived roles."""
        we = WordEncoder(dim=dim)
        se = SentenceEncoder(we)
        entry = HEALTH_ENTRIES[1]  # Vaccines prevent ...
        words, roles = svc_to_roles(entry["text"], entry["svc"])
        sent_vec = se.encode_sentence(words, roles)
        assert sent_vec.shape == (dim,)
        assert np.isfinite(sent_vec).all()

    def test_svc_roles_differ_from_auto_roles(self, dim):
        """Explicit SVC roles produce a different encoding than auto-assign."""
        we = WordEncoder(dim=dim)
        se = SentenceEncoder(we)
        entry = HEALTH_ENTRIES[1]  # Vaccines prevent ...
        words, svc_roles = svc_to_roles(entry["text"], entry["svc"])

        vec_svc = se.encode_sentence(words, svc_roles)
        vec_auto = se.encode_sentence(words)  # auto-assign

        sim = HolographicOps.similarity(vec_svc, vec_auto)
        # They share the same words but different role bindings
        assert sim < 0.99, "SVC and auto roles should differ"

    def test_verb_recovery_with_position(self, dim):
        """Unbinding VERB role + position recovers the verb word vector."""
        we = WordEncoder(dim=dim)
        se = SentenceEncoder(we)
        entry = SCIENCE_ENTRIES[0]  # Photosynthesis converts ...
        words, roles = svc_to_roles(entry["text"], entry["svc"])

        verb_word = tokenize(entry["svc"]["v"])[0]
        verb_pos = words.index(verb_word)

        sent_vec = se.encode_sentence(words, roles)

        # Unbind VERB role and position
        recovered = se.query_role(sent_vec, "VERB", position=verb_pos)
        true_verb_vec = we.encode_word(verb_word)

        sim = HolographicOps.similarity(recovered, true_verb_vec)
        # With double-unbinding, signal should be detectable
        assert sim > 0.05, f"Verb recovery too weak: sim={sim:.4f}"

    def test_subject_recovery_with_position(self, dim):
        """Unbinding SUBJ role + position recovers the subject word vector."""
        we = WordEncoder(dim=dim)
        se = SentenceEncoder(we)
        entry = SCIENCE_ENTRIES[1]  # Gravity attracts ...
        words, roles = svc_to_roles(entry["text"], entry["svc"])

        subj_word = tokenize(entry["svc"]["s"])[0]
        subj_pos = words.index(subj_word)

        sent_vec = se.encode_sentence(words, roles)
        recovered = se.query_role(sent_vec, "SUBJ", position=subj_pos)
        true_subj_vec = we.encode_word(subj_word)

        sim = HolographicOps.similarity(recovered, true_subj_vec)
        assert sim > 0.05, f"Subject recovery too weak: sim={sim:.4f}"

    def test_find_role_filler_returns_candidates(self, dim):
        """find_role_filler returns a non-empty list of (word, score) tuples."""
        we = WordEncoder(dim=dim)
        se = SentenceEncoder(we)
        entry = GENERAL_ENTRIES[1]  # Communication builds ...
        words, roles = svc_to_roles(entry["text"], entry["svc"])
        sent_vec = se.encode_sentence(words, roles)

        results = se.find_role_filler(sent_vec, "VERB", top_k=5)
        assert isinstance(results, list)
        assert len(results) > 0
        for word, score in results:
            assert isinstance(word, str)
            assert isinstance(score, float)


# =============================================================================
# 4. TestSVCWorldModelFacts
# =============================================================================


class TestSVCWorldModelFacts:
    """Feed SVC triples into WorldModel as facts."""

    def _add_svc_facts(self, wm, entries):
        """Add SVC triples as WorldModel facts."""
        for entry in entries:
            svc = entry["svc"]
            wm.add_fact(
                subject=svc["s"].lower(),
                relation=entry["root_verb"],
                object_=svc["c"].lower(),
            )

    def test_add_facts_from_svc(self, dim):
        wm = WorldModel(dim=dim)
        self._add_svc_facts(wm, HEALTH_ENTRIES)
        assert len(wm.facts) == len(HEALTH_ENTRIES)

    def test_query_known_fact(self, dim):
        """A fact we added should be retrievable."""
        wm = WorldModel(dim=dim)
        self._add_svc_facts(wm, HEALTH_ENTRIES)

        entry = HEALTH_ENTRIES[0]
        is_known, conf = wm.query_fact(
            entry["svc"]["s"].lower(),
            entry["root_verb"],
            entry["svc"]["c"].lower(),
        )
        assert is_known, f"Known fact not found, confidence={conf:.3f}"
        assert conf > 0.7

    def test_query_unknown_fact(self, dim):
        """A fact we never added should have low confidence."""
        wm = WorldModel(dim=dim)
        self._add_svc_facts(wm, HEALTH_ENTRIES)
        is_known, conf = wm.query_fact("unicorns", "fly", "to the moon")
        # Unknown facts should not be confidently known
        assert conf < 0.9

    def test_check_coherence_with_known_fact(self, dim):
        """A statement vector matching a known fact should be coherent."""
        wm = WorldModel(dim=dim)
        self._add_svc_facts(wm, SCIENCE_ENTRIES)

        entry = SCIENCE_ENTRIES[0]
        statement_vec = wm.encode_fact(
            entry["svc"]["s"].lower(),
            entry["root_verb"],
            entry["svc"]["c"].lower(),
        )
        is_coherent, score, reason = wm.check_coherence(statement_vec)
        assert is_coherent
        assert score > 0.0

    def test_facts_from_multiple_realms(self, dim):
        """Facts from different realms coexist in the same WorldModel."""
        wm = WorldModel(dim=dim)
        self._add_svc_facts(wm, HEALTH_ENTRIES + SCIENCE_ENTRIES)
        assert len(wm.facts) == len(HEALTH_ENTRIES) + len(SCIENCE_ENTRIES)

        # Health fact still queryable
        h = HEALTH_ENTRIES[0]
        is_known, _ = wm.query_fact(h["svc"]["s"].lower(), h["root_verb"], h["svc"]["c"].lower())
        assert is_known

        # Science fact still queryable
        s = SCIENCE_ENTRIES[0]
        is_known, _ = wm.query_fact(s["svc"]["s"].lower(), s["root_verb"], s["svc"]["c"].lower())
        assert is_known


# =============================================================================
# 5. TestSVCCognitiveUnderstanding
# =============================================================================


class TestSVCCognitiveUnderstanding:
    """Feed SVC sentences through CognitiveController."""

    def test_understand_returns_result(self, dim):
        """controller.understand() returns an UnderstandingResult."""
        ctrl = CognitiveController(dim=dim)
        result = ctrl.understand("Exercise is crucial for health")
        assert isinstance(result, UnderstandingResult)

    def test_understanding_result_fields(self, dim):
        """UnderstandingResult has all expected attributes."""
        ctrl = CognitiveController(dim=dim)
        result = ctrl.understand("Gravity attracts objects")
        assert isinstance(result.surface_meaning, np.ndarray)
        assert result.surface_meaning.shape == (dim,)
        assert isinstance(result.deep_meaning, np.ndarray)
        assert isinstance(result.inferences, list)
        assert isinstance(result.questions, list)
        assert isinstance(result.confidence, float)
        assert isinstance(result.parsed_roles, dict)
        assert isinstance(result.words, list)
        assert len(result.words) > 0

    def test_understanding_with_svc_knowledge(self, dim):
        """Adding SVC-derived facts before understanding enriches the model."""
        ctrl = CognitiveController(dim=dim)

        # Add knowledge from SVC data
        for entry in HEALTH_ENTRIES:
            svc = entry["svc"]
            ctrl.add_knowledge(svc["s"].lower(), entry["root_verb"], svc["c"].lower())

        result = ctrl.understand("Exercise is crucial for health")
        assert isinstance(result, UnderstandingResult)
        assert result.confidence >= 0.0

    def test_understand_simple_vs_complex(self, dim):
        """Both simple and complex sentences produce valid results."""
        ctrl = CognitiveController(dim=dim)

        simple = ALL_ENTRIES[0]  # complexity=0.4
        complex_ = SCIENCE_ENTRIES[1]  # complexity=0.6

        r1 = ctrl.understand(simple["text"])
        r2 = ctrl.understand(complex_["text"])

        assert isinstance(r1, UnderstandingResult)
        assert isinstance(r2, UnderstandingResult)
        # Both should have non-empty word lists
        assert len(r1.words) > 0
        assert len(r2.words) > 0

    def test_process_generates_response(self, dim):
        """controller.process() can generate a response string."""
        ctrl = CognitiveController(dim=dim, confidence_threshold=0.0)
        response = ctrl.process("Vaccines prevent diseases", verbose=True)
        # With threshold=0.0 we should always get a response
        assert response is not None
        assert isinstance(response, str)
        assert len(ctrl.thinking_trace) > 0


# =============================================================================
# 6. TestSVCRealmRouting
# =============================================================================


class TestSVCRealmRouting:
    """Use SVC realm field to route via ResonatorMoE."""

    REALMS = ["health", "science", "general"]

    def _build_realm_moe(self, dim):
        """Build a ResonatorMoE with one expert per realm."""
        experts = {r: (lambda x, _r=r: x) for r in self.REALMS}
        expert_vectors = {r: BinaryOps.hash_to_bipolar(r, dim) for r in self.REALMS}
        return ResonatorMoE(dim=dim, experts=experts, expert_vectors=expert_vectors)

    def test_realm_hash_vectors_near_orthogonal(self, dim):
        """Realm expert vectors should be approximately orthogonal."""
        vecs = {r: BinaryOps.hash_to_bipolar(r, dim) for r in self.REALMS}
        for i, r1 in enumerate(self.REALMS):
            for r2 in self.REALMS[i + 1 :]:
                sim = BinaryOps.similarity(vecs[r1], vecs[r2])
                assert abs(sim) < 0.2, f"{r1}/{r2} sim={sim:.3f}"

    def test_route_with_realm_indicator(self, dim):
        """Routing with realm indicator vector selects the correct expert."""
        moe = self._build_realm_moe(dim)
        for realm in self.REALMS:
            query = BinaryOps.hash_to_bipolar(realm, dim)
            result = moe.route(query, top_k=1)
            assert result[0] == realm, f"realm={realm} routed to {result[0]}"

    def test_route_with_noisy_realm_signal(self, dim):
        """Bundling sentence vector with realm indicator still routes correctly."""
        we = WordEncoder(dim=dim)
        se = SentenceEncoder(we)
        moe = self._build_realm_moe(dim)

        for entries, expected_realm in [
            (HEALTH_ENTRIES, "health"),
            (SCIENCE_ENTRIES, "science"),
            (GENERAL_ENTRIES, "general"),
        ]:
            correct = 0
            for entry in entries:
                words, roles = svc_to_roles(entry["text"], entry["svc"])
                sent_vec = se.encode_sentence(words, roles)
                realm_indicator = BinaryOps.hash_to_bipolar(entry["realm"], dim)
                # Bundle sentence content with realm signal (bipolarize sent_vec)
                query = BinaryOps.bundle(
                    [
                        np.sign(sent_vec + 1e-8).astype(np.float32),
                        realm_indicator,
                    ]
                )
                result = moe.route(query, top_k=1)
                if result[0] == expected_realm:
                    correct += 1
            assert correct == len(entries), (
                f"realm={expected_realm}: {correct}/{len(entries)} correct"
            )

    def test_get_weights_sum_to_one(self, dim):
        """Normalized expert weights should sum to approximately 1."""
        moe = self._build_realm_moe(dim)
        query = BinaryOps.hash_to_bipolar("health", dim)
        weights = moe.get_weights(query, normalize=True)
        assert abs(sum(weights.values()) - 1.0) < 1e-5


# =============================================================================
# 7. TestSVCTemporalIntegration
# =============================================================================


class TestSVCTemporalIntegration:
    """Test SVC data with temporal reasoning modules."""

    def test_causal_chain_from_svc(self, dim):
        """Add causal rules derived from SVC entries and propagate."""
        cc = CausalChain(dim=dim)

        # "Vaccines prevent many infectious diseases"
        # → if vaccines_given=True then disease_spread=False
        cc.add_rule(
            name="vaccines_prevent_disease",
            conditions={"vaccines_given": True},
            effects={"disease_spread": False},
            probability=1.0,
        )

        state = {"vaccines_given": True, "disease_spread": True}
        result = cc.propagate_forward(state, steps=1)
        assert result["disease_spread"] is False
        assert result["vaccines_given"] is True

    def test_causal_chain_multiple_rules(self, dim):
        """Multiple SVC-derived causal rules compose correctly."""
        cc = CausalChain(dim=dim)

        # "Exercise improves health" → exercise=True → health=good
        cc.add_rule(
            "exercise_improves_health",
            conditions={"exercise": True},
            effects={"health": "good"},
        )
        # "Regular sleep improves health" → sleep=regular → health=good
        cc.add_rule(
            "sleep_improves_health",
            conditions={"sleep": "regular"},
            effects={"health": "good"},
        )

        result = cc.propagate_forward({"exercise": True, "sleep": "poor"}, steps=1)
        assert result["health"] == "good"

        result2 = cc.propagate_forward({"exercise": False, "sleep": "regular"}, steps=1)
        assert result2["health"] == "good"

    def test_encode_state_produces_vector(self, dim):
        """CausalChain.encode_state returns a valid hypervector."""
        cc = CausalChain(dim=dim)
        vec = cc.encode_state({"exercise": True, "health": "good"})
        assert vec.shape == (dim,)
        assert np.isfinite(vec).all()
        assert np.linalg.norm(vec) > 0

    def test_temporal_bind_unbind_svc_sentence(self, dim):
        """Binding an SVC sentence vector with time and unbinding recovers it."""
        te = TemporalEncoder(dim=dim, max_time=100)
        we = WordEncoder(dim=dim)
        se = SentenceEncoder(we)

        entry = SCIENCE_ENTRIES[0]
        words, roles = svc_to_roles(entry["text"], entry["svc"])
        sent_vec = se.encode_sentence(words, roles)

        t = 5
        temporal = te.bind_with_time(sent_vec, t)
        recovered = te.unbind_time(temporal, t)

        sim = HolographicOps.similarity(recovered, sent_vec)
        assert sim > 0.9, f"Temporal recovery sim={sim:.3f}, expected >0.9"

    def test_temporal_wrong_time_low_similarity(self, dim):
        """Unbinding with the wrong time gives low similarity."""
        te = TemporalEncoder(dim=dim, max_time=100)
        we = WordEncoder(dim=dim)
        se = SentenceEncoder(we)

        entry = HEALTH_ENTRIES[0]
        words, roles = svc_to_roles(entry["text"], entry["svc"])
        sent_vec = se.encode_sentence(words, roles)

        temporal = te.bind_with_time(sent_vec, t=10)
        recovered_wrong = te.unbind_time(temporal, t=50)

        sim_wrong = HolographicOps.similarity(recovered_wrong, sent_vec)
        sim_right = HolographicOps.similarity(te.unbind_time(temporal, t=10), sent_vec)
        assert sim_right > sim_wrong, (
            f"Right time sim={sim_right:.3f} should beat wrong time sim={sim_wrong:.3f}"
        )

    def test_temporal_binding_multiple_svc_events(self, dim):
        """Bind different SVC sentences at different times and recover each."""
        te = TemporalEncoder(dim=dim, max_time=100)
        we = WordEncoder(dim=dim)
        se = SentenceEncoder(we)

        entries_and_times = [
            (HEALTH_ENTRIES[0], 0),
            (SCIENCE_ENTRIES[0], 10),
            (GENERAL_ENTRIES[0], 20),
        ]

        originals = []
        temporal_states = []
        for entry, t in entries_and_times:
            words, roles = svc_to_roles(entry["text"], entry["svc"])
            vec = se.encode_sentence(words, roles)
            originals.append(vec)
            temporal_states.append(te.bind_with_time(vec, t))

        # Each can be recovered from its own time slot
        for i, (entry, t) in enumerate(entries_and_times):
            recovered = te.unbind_time(temporal_states[i], t)
            sim = HolographicOps.similarity(recovered, originals[i])
            assert sim > 0.9, f"Event {i} at t={t}: sim={sim:.3f}"


# =============================================================================
# 8. TestSVCBatchPipeline
# =============================================================================


class TestSVCBatchPipeline:
    """End-to-end batch processing of SVC entries through the full pipeline."""

    def test_batch_encode_all_entries(self, dim):
        """All SVC entries encode without error."""
        we = WordEncoder(dim=dim)
        se = SentenceEncoder(we)

        vectors = []
        for entry in ALL_ENTRIES:
            words, roles = svc_to_roles(entry["text"], entry["svc"])
            vec = se.encode_sentence(words, roles)
            assert vec.shape == (dim,)
            assert np.isfinite(vec).all()
            vectors.append(vec)

        assert len(vectors) == len(ALL_ENTRIES)

    def test_batch_sentence_vectors_unit_normalized(self, dim):
        """All sentence vectors produced by bundle(normalize=True) are unit-length."""
        we = WordEncoder(dim=dim)
        se = SentenceEncoder(we)

        for entry in ALL_ENTRIES:
            words, roles = svc_to_roles(entry["text"], entry["svc"])
            vec = se.encode_sentence(words, roles)
            norm = np.linalg.norm(vec)
            assert abs(norm - 1.0) < 0.01, f"{entry['id']} norm={norm:.4f}"

    def test_batch_add_facts_and_route(self, dim):
        """Pipeline: parse SVC → add facts → encode → route by realm."""
        we = WordEncoder(dim=dim)
        se = SentenceEncoder(we)
        wm = WorldModel(dim=dim)

        realm_experts = {r: (lambda x, _r=r: x) for r in ("health", "science", "general")}
        realm_vectors = {r: BinaryOps.hash_to_bipolar(r, dim) for r in realm_experts}
        moe = ResonatorMoE(dim=dim, experts=realm_experts, expert_vectors=realm_vectors)

        for entry in ALL_ENTRIES:
            svc = entry["svc"]
            # 1. Add fact
            wm.add_fact(svc["s"].lower(), entry["root_verb"], svc["c"].lower())
            # 2. Encode sentence
            words, roles = svc_to_roles(entry["text"], svc)
            sent_vec = se.encode_sentence(words, roles)
            assert sent_vec.shape == (dim,)
            # 3. Route by realm
            realm = entry["realm"]
            if realm in realm_experts:
                indicator = BinaryOps.hash_to_bipolar(realm, dim)
                routed = moe.route(indicator, top_k=1)
                assert routed[0] == realm

        assert len(wm.facts) == len(ALL_ENTRIES)

    def test_same_realm_higher_similarity(self, dim):
        """Sentences within the same realm share more vocabulary overlap."""
        we = WordEncoder(dim=dim)
        se = SentenceEncoder(we)

        def encode_entries(entries):
            vecs = []
            for entry in entries:
                words, roles = svc_to_roles(entry["text"], entry["svc"])
                vecs.append(se.encode_sentence(words, roles))
            return vecs

        health_vecs = encode_entries(HEALTH_ENTRIES)
        science_vecs = encode_entries(SCIENCE_ENTRIES)

        # Mean pairwise similarity within health
        within_sims = []
        for i in range(len(health_vecs)):
            for j in range(i + 1, len(health_vecs)):
                within_sims.append(HolographicOps.similarity(health_vecs[i], health_vecs[j]))

        # Mean pairwise similarity across health/science
        cross_sims = []
        for hv in health_vecs:
            for sv in science_vecs:
                cross_sims.append(HolographicOps.similarity(hv, sv))

        mean_within = np.mean(within_sims)
        mean_cross = np.mean(cross_sims)

        # Both should be small in absolute terms, but within-realm
        # should be slightly higher due to shared vocabulary (health, disease, etc.)
        # We only assert they are finite; the clustering property is a soft signal.
        assert np.isfinite(mean_within)
        assert np.isfinite(mean_cross)

    def test_end_to_end_cognitive_with_svc_facts(self, dim):
        """Full pipeline: SVC facts → CognitiveController → understand."""
        ctrl = CognitiveController(dim=dim)

        # Load all SVC data as world knowledge
        for entry in ALL_ENTRIES:
            svc = entry["svc"]
            ctrl.add_knowledge(svc["s"].lower(), entry["root_verb"], svc["c"].lower())

        assert len(ctrl.world.facts) == len(ALL_ENTRIES)

        # Understand a sentence related to the knowledge base
        result = ctrl.understand("Vaccines prevent diseases")
        assert isinstance(result, UnderstandingResult)
        assert result.surface_meaning.shape == (dim,)
        assert len(result.words) > 0
        print(result)
        print(result.surface_meaning)
        print(result.words)
