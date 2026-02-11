"""
WordEncoder and SentenceEncoder for instant language learning.

Encodes words using n-grams and sentences using role-bound composition.
"""

import numpy as np

# Stable hashing (BLAKE3) for deterministic word seeding
try:
    from utils.stable_hash import stable_u32
except ModuleNotFoundError:
    try:
        from grilly.utils.stable_hash import stable_u32  # type: ignore
    except Exception:
        stable_u32 = None  # type: ignore

from grilly.experimental.vsa.ops import HolographicOps


def _unitary(vec: np.ndarray) -> np.ndarray:
    """Project vector to unitary form in frequency domain."""
    fft = np.fft.fft(vec)
    mag = np.abs(fft)
    fft = fft / (mag + 1e-8)
    return np.real(np.fft.ifft(fft)).astype(np.float32)


class WordEncoder:
    """
    Encodes words as hypervectors with:
    - Semantic similarity preserved (similar words = similar vectors)
    - Instant relationship extraction via correlation
    - No training needed for basic operations

    Uses character n-gram composition for OOV handling.
    """

    DEFAULT_DIM = 4096

    def __init__(self, dim: int = DEFAULT_DIM, use_ngrams: bool = True):
        """Initialize the instance."""

        self.dim = dim
        self.use_ngrams = use_ngrams

        # Core vocabulary
        self.word_vectors: dict[str, np.ndarray] = {}

        # Character vectors for n-gram encoding
        self.char_vectors: dict[str, np.ndarray] = {}
        self._init_char_vectors()

        # Position vectors for n-gram positions
        self.position_vectors: list[np.ndarray] = [
            HolographicOps.random_vector(dim, seed=10000 + i) for i in range(20)
        ]

        # Relation primitives
        self.relations: dict[str, np.ndarray] = {}
        self._init_relations()

    def _init_char_vectors(self):
        """Initialize character-level vectors."""
        chars = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?'-"
        for i, char in enumerate(chars):
            self.char_vectors[char] = HolographicOps.random_vector(self.dim, seed=1000 + i)
        # Unknown char
        self.char_vectors["<UNK>"] = HolographicOps.random_vector(self.dim, seed=999)

    def _init_relations(self):
        """Initialize relation vectors for word relationships."""
        self.relations = {
            "subject_of": HolographicOps.random_vector(self.dim, seed=2000),
            "object_of": HolographicOps.random_vector(self.dim, seed=2001),
            "modifier_of": HolographicOps.random_vector(self.dim, seed=2002),
            "head_of": HolographicOps.random_vector(self.dim, seed=2003),
            "synonym": HolographicOps.random_vector(self.dim, seed=2010),
            "antonym": -HolographicOps.random_vector(self.dim, seed=2010),
            "hypernym": HolographicOps.random_vector(self.dim, seed=2011),
            "hyponym": HolographicOps.random_vector(self.dim, seed=2012),
            "meronym": HolographicOps.random_vector(self.dim, seed=2013),
            "before": HolographicOps.random_vector(self.dim, seed=2020),
            "after": HolographicOps.random_vector(self.dim, seed=2021),
            "causes": HolographicOps.random_vector(self.dim, seed=2022),
        }

    def encode_word(self, word: str) -> np.ndarray:
        """
        Encode a word as a hypervector.

        Uses cached vector if available, otherwise builds from n-grams.
        """
        word = word.lower().strip()

        if word in self.word_vectors:
            return self.word_vectors[word]

        if self.use_ngrams:
            vec = self._encode_ngrams(word)
        else:
            # Deterministic random from hash
            vec = HolographicOps.random_vector(
                self.dim,
                seed=(stable_u32(word, domain="grilly.word") % (2**31) if stable_u32 else 0),
            )

        self.word_vectors[word] = vec
        return vec

    def _encode_ngrams(self, word: str, n: int = 3) -> np.ndarray:
        """
        Encode word using character n-grams.

        Similar spelling = similar vectors.
        """
        # Pad word
        padded = f"#{word}#"

        ngrams = []
        for i in range(len(padded) - n + 1):
            ngram = padded[i : i + n]

            # Encode n-gram as bound character sequence
            ngram_vec = np.zeros(self.dim, dtype=np.float32)
            ngram_vec[0] = 1.0
            for j, char in enumerate(ngram):
                char_vec = self.char_vectors.get(char, self.char_vectors["<UNK>"])
                pos_vec = self.position_vectors[j % len(self.position_vectors)]
                # Bind character with position
                ngram_vec = HolographicOps.convolve(
                    ngram_vec, HolographicOps.convolve(char_vec, pos_vec)
                )

            ngrams.append(ngram_vec)

        # Bundle all n-grams and project to unitary form
        return _unitary(HolographicOps.bundle(ngrams))

    def extract_relation(self, word_a: str, word_b: str) -> np.ndarray:
        """
        Extract the relational transformation between two words.

        relation = correlate(encode(b), encode(a))
        """
        vec_a = self.encode_word(word_a)
        vec_b = self.encode_word(word_b)
        return HolographicOps.correlate(vec_b, vec_a)

    def apply_relation(self, word: str, relation: np.ndarray) -> np.ndarray:
        """
        Apply a relation vector to get a new word vector.

        result = convolve(encode(word), relation)
        """
        word_vec = self.encode_word(word)
        return HolographicOps.convolve(word_vec, relation)

    def find_closest(self, vec: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """Find closest words to a vector."""
        results = []
        for word, word_vec in self.word_vectors.items():
            sim = HolographicOps.similarity(vec, word_vec)
            results.append((word, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def learn_relation(self, pairs: list[tuple[str, str]], relation_name: str) -> np.ndarray:
        """
        Learn a relation from word pairs.

        Given: [(king, queen), (man, woman)]
        Learn: the "gender" relation
        """
        relation_samples = []
        for word_a, word_b in pairs:
            rel = self.extract_relation(word_a, word_b)
            relation_samples.append(rel)

        # Average the relation patterns
        learned_relation = HolographicOps.bundle(relation_samples)
        self.relations[relation_name] = learned_relation

        return learned_relation


class SentenceEncoder:
    """
    Encodes sentences as compositional hypervectors.

    Sentence = Σ (word_i ⊗ role_i ⊗ position_i)

    Allows instant parsing via unbinding and role queries.
    """

    def __init__(self, word_encoder: WordEncoder):
        """Initialize the instance."""

        self.word_encoder = word_encoder
        self.dim = word_encoder.dim

        # Syntactic role vectors
        self.roles = {
            "SUBJ": _unitary(HolographicOps.random_vector(self.dim, seed=3000)),
            "VERB": _unitary(HolographicOps.random_vector(self.dim, seed=3001)),
            "OBJ": _unitary(HolographicOps.random_vector(self.dim, seed=3002)),
            "IOBJ": _unitary(HolographicOps.random_vector(self.dim, seed=3003)),
            "ADJ": _unitary(HolographicOps.random_vector(self.dim, seed=3010)),
            "ADV": _unitary(HolographicOps.random_vector(self.dim, seed=3011)),
            "DET": _unitary(HolographicOps.random_vector(self.dim, seed=3012)),
            "PREP": _unitary(HolographicOps.random_vector(self.dim, seed=3013)),
            "POBJ": _unitary(HolographicOps.random_vector(self.dim, seed=3014)),
            "AUX": _unitary(HolographicOps.random_vector(self.dim, seed=3020)),
            "COMP": _unitary(HolographicOps.random_vector(self.dim, seed=3021)),
            "REL": _unitary(HolographicOps.random_vector(self.dim, seed=3022)),
            "ROOT": _unitary(HolographicOps.random_vector(self.dim, seed=3030)),
            "PUNCT": _unitary(HolographicOps.random_vector(self.dim, seed=3031)),
        }

        # Position encoding vectors
        self.position_vectors = [
            _unitary(HolographicOps.random_vector(self.dim, seed=4000 + i)) for i in range(100)
        ]

        # Phrase structure vectors
        self.phrase_types = {
            "NP": _unitary(HolographicOps.random_vector(self.dim, seed=5000)),
            "VP": _unitary(HolographicOps.random_vector(self.dim, seed=5001)),
            "PP": _unitary(HolographicOps.random_vector(self.dim, seed=5002)),
            "AP": _unitary(HolographicOps.random_vector(self.dim, seed=5003)),
            "S": _unitary(HolographicOps.random_vector(self.dim, seed=5004)),
        }

    def encode_sentence(
        self, words: list[str], roles: list[str] | None = None, return_components: bool = False
    ) -> np.ndarray | tuple[np.ndarray, list[np.ndarray]]:
        """
        Encode a sentence as a holographic vector.

        Args:
            words: List of words in order
            roles: Optional syntactic roles (auto-assigned if None)
            return_components: Whether to return individual word-role bindings

        Returns:
            Sentence vector (and optionally component vectors)
        """
        if roles is None:
            roles = self._auto_assign_roles(words)

        components = []

        for i, (word, role) in enumerate(zip(words, roles)):
            # Get vectors
            word_vec = self.word_encoder.encode_word(word)
            role_vec = self.roles.get(role, self.roles["ROOT"])
            pos_vec = self.position_vectors[i % len(self.position_vectors)]

            # Bind word with role and position
            component = HolographicOps.convolve(word_vec, role_vec)
            component = HolographicOps.convolve(component, pos_vec)

            components.append(component)

        # Bundle all components
        sentence_vec = HolographicOps.bundle(components, normalize=True)

        if return_components:
            return sentence_vec, components
        return sentence_vec

    def _auto_assign_roles(self, words: list[str]) -> list[str]:
        """
        Simple heuristic role assignment.

        For production, use a proper POS tagger or dependency parser.
        """
        roles = []

        determiners = {
            "the",
            "a",
            "an",
            "this",
            "that",
            "these",
            "those",
            "my",
            "your",
            "his",
            "her",
        }
        prepositions = {"in", "on", "at", "to", "for", "with", "by", "from", "of", "about"}
        auxiliaries = {
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
        }

        found_verb = False
        found_subj = False

        for i, word in enumerate(words):
            word_lower = word.lower()

            if word_lower in determiners:
                roles.append("DET")
            elif word_lower in prepositions:
                roles.append("PREP")
            elif word_lower in auxiliaries:
                roles.append("AUX")
                found_verb = True
            elif not found_verb:
                if not found_subj:
                    if i > 0 and len(roles) > 0 and roles[i - 1] == "DET":
                        roles.append("SUBJ")
                        found_subj = True
                    elif word[0].isupper() or word_lower in {
                        "i",
                        "he",
                        "she",
                        "it",
                        "they",
                        "we",
                        "you",
                    }:
                        roles.append("SUBJ")
                        found_subj = True
                    else:
                        roles.append("ADJ")
                else:
                    roles.append("ADJ")
            elif word_lower.endswith(("ly",)):
                roles.append("ADV")
            elif not found_verb or (found_verb and word_lower.endswith(("s", "ed", "ing"))):
                roles.append("VERB")
                found_verb = True
            else:
                if i > 0 and len(roles) > 0 and roles[i - 1] == "PREP":
                    roles.append("POBJ")
                else:
                    roles.append("OBJ")

        # Ensure we have at least ROOT roles
        if not roles:
            roles = ["ROOT"] * len(words)
        elif len(roles) < len(words):
            roles.extend(["ROOT"] * (len(words) - len(roles)))

        return roles[: len(words)]

    def query_role(
        self, sentence_vec: np.ndarray, role: str, position: int | None = None
    ) -> np.ndarray:
        """
        Query: "What fills this role in the sentence?"

        Unbind the role (and optionally position) to recover the word.
        """
        role_vec = self.roles.get(role, self.roles["ROOT"])

        # Unbind role
        result = HolographicOps.correlate(sentence_vec, role_vec)

        # Optionally unbind position too
        if position is not None:
            pos_vec = self.position_vectors[position % len(self.position_vectors)]
            result = HolographicOps.correlate(result, pos_vec)

        return result

    def find_role_filler(
        self, sentence_vec: np.ndarray, role: str, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """
        Find what word fills a role in the sentence.

        "Who is the subject?" -> query SUBJ role -> find closest word
        """
        query_result = self.query_role(sentence_vec, role)
        return self.word_encoder.find_closest(query_result, top_k)

    def sentence_similarity(
        self, sent_a: str | list[str] | np.ndarray, sent_b: str | list[str] | np.ndarray
    ) -> float:
        """Compare two sentences semantically."""
        if isinstance(sent_a, str):
            sent_a = sent_a.split()
        if isinstance(sent_b, str):
            sent_b = sent_b.split()

        if isinstance(sent_a, list):
            vec_a = self.encode_sentence(sent_a)
        else:
            vec_a = sent_a

        if isinstance(sent_b, list):
            vec_b = self.encode_sentence(sent_b)
        else:
            vec_b = sent_b

        return HolographicOps.similarity(vec_a, vec_b)
