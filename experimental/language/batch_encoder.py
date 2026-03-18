"""Batch VSA text encoding with GPU acceleration.

Encodes large numbers of sentences into hypervector representations
using VulkanVSA for the binding and bundling operations.

Usage:
    encoder = BatchVSAEncoder(dim=4096)
    vectors = encoder.encode_sentences(["hello world", "foo bar"])
    # vectors: (2, 4096) bipolar hypervectors
"""

import numpy as np

from grilly.experimental.language.encoder import SentenceEncoder, WordEncoder
from grilly.experimental.vsa.ops import HolographicOps


def _try_init_gpu():
    """Attempt to initialise a VulkanVSA instance. Returns None on failure."""
    try:
        from grilly.backend.core import VulkanCore
        from grilly.backend.experimental.vsa import VulkanVSA

        core = VulkanCore()
        return VulkanVSA(core)
    except Exception:
        return None


class BatchVSAEncoder:
    """
    Batch-encode text into hypervector representations.

    Wraps WordEncoder + SentenceEncoder and batches the bind/bundle
    operations so that 10K+ sentences can be encoded efficiently using
    GPU-accelerated VulkanVSA when available, falling back to CPU numpy
    batch ops automatically.

    Args:
        dim: Hypervector dimension (default 4096).
        use_gpu: Whether to attempt GPU acceleration via VulkanVSA.
    """

    def __init__(self, dim: int = 4096, use_gpu: bool = True):
        self.dim = dim
        self.word_encoder = WordEncoder(dim=dim)
        self.sentence_encoder = SentenceEncoder(self.word_encoder)

        self.vsa_gpu = _try_init_gpu() if use_gpu else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_sentences(self, texts: list[str], batch_size: int = 1000) -> np.ndarray:
        """
        Batch-encode sentences into hypervectors.

        Args:
            texts: List of sentences (plain text; split on whitespace).
            batch_size: Number of sentences to process per GPU batch.

        Returns:
            Float32 array of shape (N, dim) — one row per sentence.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        results = np.empty((len(texts), self.dim), dtype=np.float32)

        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            chunk_vecs = self._encode_sentence_chunk(chunk)
            results[start : start + len(chunk)] = chunk_vecs

        return results

    def encode_words(self, words: list[str]) -> np.ndarray:
        """
        Batch-encode words into hypervectors.

        Args:
            words: List of word strings.

        Returns:
            Float32 array of shape (N, dim).
        """
        if not words:
            return np.zeros((0, self.dim), dtype=np.float32)

        result = np.empty((len(words), self.dim), dtype=np.float32)
        for i, word in enumerate(words):
            result[i] = self.word_encoder.encode_word(word)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_sentence_chunk(self, texts: list[str]) -> np.ndarray:
        """
        Encode a chunk of sentences using batched bind + bundle.

        Pipeline per sentence:
            1. Tokenise (whitespace split).
            2. Encode each token → word vector.
            3. Bind each word vector with its role vector and position vector.
            4. Bundle all bound components into a single sentence vector.

        Steps 3 and 4 are batched across all sentences in the chunk using
        VulkanVSA (GPU) when available, or numpy batch ops otherwise.
        """
        batch_n = len(texts)
        tokenised = [text.lower().split() for text in texts]

        # Maximum number of tokens across all sentences in this chunk.
        # Used to pad the 3D arrays for batch ops.
        max_len = max((len(t) for t in tokenised), default=1)

        # Arrays to accumulate bound components: shape (batch_n, max_len, dim)
        bound_components = np.zeros((batch_n, max_len, self.dim), dtype=np.float32)
        # Mask: 1.0 where a real token exists, 0.0 for padding slots
        token_mask = np.zeros((batch_n, max_len), dtype=np.float32)

        # --- Step 1–3: encode words, bind with role + position ---
        # Collect flat pairs for batch bind: word ⊗ role, then ⊗ position.
        # We do two successive batch-bind passes.

        roles_map = self.sentence_encoder.roles
        pos_vecs = self.sentence_encoder.position_vectors

        # Flat arrays for batch bind pass 1: word ⊗ role
        all_word_vecs: list[np.ndarray] = []
        all_role_vecs: list[np.ndarray] = []
        # Flat arrays for batch bind pass 2: (word⊗role) ⊗ position
        all_pos_vecs: list[np.ndarray] = []
        # (sentence_idx, token_idx) for each flat entry
        flat_indices: list[tuple[int, int]] = []

        for sent_idx, words in enumerate(tokenised):
            auto_roles = self.sentence_encoder._auto_assign_roles(words)
            for tok_idx, (word, role) in enumerate(zip(words, auto_roles)):
                word_vec = self.word_encoder.encode_word(word)
                role_vec = roles_map.get(role, roles_map["ROOT"])
                pos_vec = pos_vecs[tok_idx % len(pos_vecs)]

                all_word_vecs.append(word_vec)
                all_role_vecs.append(role_vec)
                all_pos_vecs.append(pos_vec)
                flat_indices.append((sent_idx, tok_idx))
                token_mask[sent_idx, tok_idx] = 1.0

        if not flat_indices:
            # All texts were empty strings
            return np.zeros((batch_n, self.dim), dtype=np.float32)

        word_arr = np.array(all_word_vecs, dtype=np.float32)  # (F, dim)
        role_arr = np.array(all_role_vecs, dtype=np.float32)  # (F, dim)
        pos_arr = np.array(all_pos_vecs, dtype=np.float32)  # (F, dim)

        # Batch bind pass 1: word ⊗ role
        wr_arr = self._batch_bind(word_arr, role_arr)  # (F, dim)

        # Batch bind pass 2: (word⊗role) ⊗ position
        wrp_arr = self._batch_bind(wr_arr, pos_arr)  # (F, dim)

        # Scatter back into (batch_n, max_len, dim)
        for flat_i, (sent_idx, tok_idx) in enumerate(flat_indices):
            bound_components[sent_idx, tok_idx] = wrp_arr[flat_i]

        # --- Step 4: bundle per sentence ---
        # bundle_batch expects (batch_n, max_len, dim) but padding slots should
        # contribute zero, so we zero them out (already zero by np.zeros init
        # except we must account for the mask).
        # Apply mask so padded slots don't affect majority-vote sign.
        # We scale padding to 0 and real tokens to 1 (no-op since already 0).
        # bundle_batch sums over axis=1 then normalises.
        sentence_vecs = self._batch_bundle(bound_components, token_mask)  # (batch_n, dim)

        return sentence_vecs.astype(np.float32)

    # ------------------------------------------------------------------
    # GPU / CPU dispatch
    # ------------------------------------------------------------------

    def _batch_bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Element-wise multiply two (F, dim) arrays — GPU when available.

        VulkanVSA.bind_bipolar_batch does element-wise multiplication which
        is mathematically identical for both bipolar and continuous vectors.
        """
        if self.vsa_gpu is not None:
            try:
                return self.vsa_gpu.bind_bipolar_batch(a, b)
            except Exception:
                pass  # fall through to CPU
        # CPU fallback: numpy element-wise multiply
        return (a * b).astype(np.float32)

    def _batch_bundle(self, components: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Bundle (N, T, dim) → (N, dim) with masking, GPU when available.

        Masked-out positions (padding) contribute 0 to the sum, so they
        don't distort the majority-vote / normalisation.

        Args:
            components: Shape (batch_n, max_len, dim).
            mask: Shape (batch_n, max_len) — 1 for real tokens, 0 for padding.
        """
        # Apply mask: multiply each token vector by its 0/1 mask weight
        masked = components * mask[:, :, np.newaxis]  # (N, T, dim)

        if self.vsa_gpu is not None:
            try:
                # VulkanVSA.bundle_batch sums axis=1 with majority-vote normalisation
                return self.vsa_gpu.bundle_batch(masked)
            except Exception:
                pass  # fall through to CPU

        # CPU fallback: HolographicOps.bundle_batch (normalised sum)
        return HolographicOps.bundle_batch(masked, normalize=True)
