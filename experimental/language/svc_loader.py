"""SVC data loader and GPU-aware ingestion engine.

This module provides JSONL loaders for SVC data, batch-level statistics,
and an ingestion engine that dispatches VSA operations to Vulkan when
available and falls back to CPU implementations otherwise.
"""

import json
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from grilly.experimental.vsa.ops import BinaryOps, HolographicOps

if TYPE_CHECKING:
    from grilly.backend.experimental.vsa import VulkanVSA
    from grilly.experimental.language.encoder import SentenceEncoder, WordEncoder


# ---------------------------------------------------------------------------
# Data layer - parsing and filtering
# ---------------------------------------------------------------------------


@dataclass
class SVCEntry:
    """A parsed SVC entry from the training data."""

    id: str
    text: str
    svc_s: str
    svc_v: str
    svc_c: str
    pos: list[str]
    deps: list[str]
    lemmas: list[str]
    root_verb: str
    realm: str
    source: str
    complexity: float

    @classmethod
    def from_dict(cls, data: dict) -> "SVCEntry":
        """Create SVCEntry from JSON dict."""
        return cls(
            id=data.get("id", ""),
            text=data.get("text", ""),
            svc_s=data.get("svc", {}).get("s", ""),
            svc_v=data.get("svc", {}).get("v", ""),
            svc_c=data.get("svc", {}).get("c", ""),
            pos=data.get("pos", []),
            deps=data.get("deps", []),
            lemmas=data.get("lemmas", []),
            root_verb=data.get("root_verb", ""),
            realm=data.get("realm", ""),
            source=data.get("source", ""),
            complexity=data.get("complexity", 0.0),
        )

    def tokenize(self) -> list[str]:
        """Tokenize text the same way InstantLanguage does."""
        text = re.sub(r"[^\w\s]", "", self.text.lower())
        return text.split()

    def to_roles(self) -> tuple[list[str], list[str]]:
        """Map SVC s/v/c fields to per-word SUBJ/VERB/OBJ roles.

        Returns:
            (words, roles) where words are tokenized and lowercased.
        """
        words = self.tokenize()
        v_words = set(re.sub(r"[^\w\s]", "", self.svc_v.lower()).split())
        s_words = set(re.sub(r"[^\w\s]", "", self.svc_s.lower()).split())
        c_words = set(re.sub(r"[^\w\s]", "", self.svc_c.lower()).split())

        roles: list[str] = []
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

    def template_key(self) -> str:
        """Return a template key from the dependency pattern."""
        return "_".join(self.deps) if self.deps else "unknown"


@dataclass
class SVCBatch:
    """A batch of loaded SVC entries with statistics."""

    entries: list[SVCEntry]
    realm_counts: dict[str, int] = field(default_factory=dict)
    source_counts: dict[str, int] = field(default_factory=dict)
    verb_counts: dict[str, int] = field(default_factory=dict)
    avg_complexity: float = 0.0
    total_loaded: int = 0
    total_skipped: int = 0

    @property
    def realms(self) -> list[str]:
        """Execute realms."""

        return sorted(self.realm_counts.keys())

    @property
    def realm_entries(self) -> dict[str, list[SVCEntry]]:
        """Execute realm entries."""

        grouped: dict[str, list[SVCEntry]] = defaultdict(list)
        for entry in self.entries:
            grouped[entry.realm].append(entry)
        return dict(grouped)

    def summary(self) -> str:
        """Execute summary."""

        lines = [
            f"SVCBatch: {len(self.entries)} entries loaded ({self.total_skipped} skipped)",
            f"  Realms: {self.realm_counts}",
            f"  Sources: {self.source_counts}",
            f"  Top verbs: {dict(sorted(self.verb_counts.items(), key=lambda x: -x[1])[:10])}",
            f"  Avg complexity: {self.avg_complexity:.3f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------


def load_svc_entries(
    path: str,
    max_entries: int | None = None,
    realms: list[str] | None = None,
    min_complexity: float | None = None,
    max_complexity: float | None = None,
    sources: list[str] | None = None,
) -> Iterator[SVCEntry]:
    """Load SVC entries from JSONL file with optional filtering."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"SVC data file not found: {path}")

    count = 0
    with open(path_obj, encoding="utf-8") as f:
        for line in f:
            if max_entries is not None and count >= max_entries:
                break
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                entry = SVCEntry.from_dict(data)
                if realms is not None and entry.realm not in realms:
                    continue
                if sources is not None and entry.source not in sources:
                    continue
                if min_complexity is not None and entry.complexity < min_complexity:
                    continue
                if max_complexity is not None and entry.complexity > max_complexity:
                    continue
                yield entry
                count += 1
            except (json.JSONDecodeError, Exception):
                continue


def load_svc_batch(
    path: str,
    max_entries: int | None = None,
    realms: list[str] | None = None,
    min_complexity: float | None = None,
    max_complexity: float | None = None,
    sources: list[str] | None = None,
) -> SVCBatch:
    """Load SVC entries into a batch with computed statistics."""
    entries: list[SVCEntry] = []
    realm_counts: dict[str, int] = defaultdict(int)
    source_counts: dict[str, int] = defaultdict(int)
    verb_counts: dict[str, int] = defaultdict(int)
    total_complexity = 0.0
    total_skipped = 0

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"SVC data file not found: {path}")

    with open(path_obj, encoding="utf-8") as f:
        for line in f:
            if max_entries is not None and len(entries) >= max_entries:
                break
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                entry = SVCEntry.from_dict(data)
                if realms is not None and entry.realm not in realms:
                    total_skipped += 1
                    continue
                if sources is not None and entry.source not in sources:
                    total_skipped += 1
                    continue
                if min_complexity is not None and entry.complexity < min_complexity:
                    total_skipped += 1
                    continue
                if max_complexity is not None and entry.complexity > max_complexity:
                    total_skipped += 1
                    continue
                entries.append(entry)
                realm_counts[entry.realm] += 1
                source_counts[entry.source] += 1
                verb_counts[entry.root_verb] += 1
                total_complexity += entry.complexity
            except (json.JSONDecodeError, Exception):
                total_skipped += 1
                continue

    avg_complexity = total_complexity / len(entries) if entries else 0.0
    return SVCBatch(
        entries=entries,
        realm_counts=dict(realm_counts),
        source_counts=dict(source_counts),
        verb_counts=dict(verb_counts),
        avg_complexity=avg_complexity,
        total_loaded=len(entries),
        total_skipped=total_skipped,
    )


def load_svc_entries_from_dicts(
    data: list[dict],
    realms: list[str] | None = None,
) -> list[SVCEntry]:
    """Load SVC entries from in-memory dicts (for testing)."""
    entries = []
    for d in data:
        entry = SVCEntry.from_dict(d)
        if realms is not None and entry.realm not in realms:
            continue
        entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# GPU-aware Ingestion Engine
# ---------------------------------------------------------------------------


def _try_get_vulkan_vsa() -> Optional["VulkanVSA"]:
    """Attempt to create a VulkanVSA instance.  Returns None on failure."""
    try:
        from grilly.backend.core import VulkanCore
        from grilly.backend.experimental.vsa import VulkanVSA

        core = VulkanCore()
        return VulkanVSA(core)
    except Exception:
        return None


class SVCIngestionEngine:
    """GPU-aware SVC ingestion engine.

    The engine wraps sentence encoding, bundling, similarity search, and
    routing operations. It uses Vulkan-backed ``VulkanVSA`` kernels when
    available, and CPU implementations as fallback.

    Usage::

        engine = SVCIngestionEngine(dim=2048)  # auto-detect GPU
        engine = SVCIngestionEngine(dim=2048, gpu=my_vulkan_vsa)
        engine = SVCIngestionEngine(dim=2048, gpu=False)  # force CPU
    """

    def __init__(
        self,
        dim: int,
        gpu: object | None = None,
    ):
        """
        Args:
            dim: Hypervector dimension.
            gpu: One of:
                 - ``None``: auto-detect VulkanVSA.
                 - ``False``: force CPU path.
                 - A ``VulkanVSA`` instance: use it directly.
        """
        self.dim = dim

        if gpu is False:
            self._gpu: VulkanVSA | None = None
        elif gpu is None:
            self._gpu = _try_get_vulkan_vsa()
        else:
            self._gpu = gpu  # type: ignore[assignment]

        self.using_gpu = self._gpu is not None

    # -- core ops (GPU or CPU) ----------------------------------------

    def bundle(
        self,
        vectors: list[np.ndarray],
        normalize: bool = True,
    ) -> np.ndarray:
        """Bundle (superpose) a list of vectors.

        GPU: ``VulkanVSA.bundle`` -> ``vsa-bundle.spv``
        CPU: ``HolographicOps.bundle``
        """
        if self._gpu is not None:
            try:
                result = self._gpu.bundle(vectors)
                if normalize:
                    norm = np.linalg.norm(result)
                    if norm > 0:
                        result = result / norm
                return result.astype(np.float32)
            except Exception:
                pass
        return HolographicOps.bundle(vectors, normalize=normalize)

    def similarity_batch(
        self,
        query: np.ndarray,
        codebook: np.ndarray,
    ) -> np.ndarray:
        """Batch cosine similarity: query vs every row in codebook.

        GPU: ``VulkanVSA.similarity_batch`` -> ``vsa-similarity-batch.spv``
        CPU: ``HolographicOps.similarity_batch``
        """
        if self._gpu is not None:
            try:
                return self._gpu.similarity_batch(query, codebook)
            except Exception:
                pass
        return HolographicOps.similarity_batch(query, codebook)

    def bind_bipolar_batch(
        self,
        a_batch: np.ndarray,
        b_batch: np.ndarray,
    ) -> np.ndarray:
        """Batch element-wise bipolar binding.

        GPU: ``VulkanVSA.bind_bipolar_batch`` -> ``vsa-bind-batch.spv``
        CPU: ``BinaryOps.bind_batch``
        """
        if self._gpu is not None:
            try:
                return self._gpu.bind_bipolar_batch(a_batch, b_batch)
            except Exception:
                pass
        return BinaryOps.bind_batch(a_batch, b_batch)

    def convolve(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Circular convolution (HRR binding).

        GPU: ``VulkanVSA.circular_convolve`` -> ``vsa-fft-convolve.spv``
        CPU: ``HolographicOps.convolve``
        """
        if self._gpu is not None:
            try:
                return self._gpu.circular_convolve(a, b)
            except Exception:
                pass
        return HolographicOps.convolve(a, b)

    def resonator_step(
        self,
        composite: np.ndarray,
        codebook: np.ndarray,
        other_estimates: list[np.ndarray] | None = None,
    ) -> tuple[np.ndarray, int]:
        """One resonator projection step.

        GPU: ``VulkanVSA.resonator_step`` -> ``vsa-resonator-step.spv``
        CPU: codebook dot-product + argmax
        """
        if self._gpu is not None:
            try:
                return self._gpu.resonator_step(composite, codebook, other_estimates)
            except Exception:
                pass
        # CPU fallback
        unbound = composite.copy()
        if other_estimates:
            for est in other_estimates:
                unbound = BinaryOps.unbind(unbound, est)
        sims = (codebook @ unbound) / float(self.dim)
        best_idx = int(np.argmax(sims))
        return codebook[best_idx].copy(), best_idx

    # -- high-level batch operations ----------------------------------

    def batch_encode_sentences(
        self,
        entries: list[SVCEntry],
        word_encoder: "WordEncoder",
        sentence_encoder: "SentenceEncoder",
    ) -> tuple[list[np.ndarray], list[list[str]]]:
        """Encode a batch of SVC entries into sentence vectors.

        For each entry this:
        1. Tokenizes and maps to SVC roles (SUBJ/VERB/OBJ).
        2. Encodes every word (populates word_encoder vocabulary).
        3. Binds word, role, and position per slot using ``self.convolve``.
        4. Bundles slot vectors via ``self.bundle`` into a sentence vector.

        Returns:
            ``(sentence_vectors, word_lists)`` as parallel lists.
        """
        sentence_vecs: list[np.ndarray] = []
        word_lists: list[list[str]] = []

        # Cache of (role, pos_mod) -> (role bound with position) to reduce convolves.
        # For long corpora this cuts per-token binding cost roughly in half.
        rolepos_cache: dict[tuple[str, int], np.ndarray] = {}
        pos_mod = len(sentence_encoder.position_vectors)

        def _rolepos(role: str, i: int) -> np.ndarray:
            """Execute rolepos."""

            key = (role, i % pos_mod)
            cached = rolepos_cache.get(key)
            if cached is not None:
                return cached
            role_vec = sentence_encoder.roles.get(role, sentence_encoder.roles["ROOT"])
            pos_vec = sentence_encoder.position_vectors[key[1]]
            rp = self.convolve(role_vec, pos_vec)
            rolepos_cache[key] = rp
            return rp

        for entry in entries:
            words, roles = entry.to_roles()

            # ensure vocabulary is populated
            for w in words:
                word_encoder.encode_word(w)

            components: list[np.ndarray] = []
            for i, (word, role) in enumerate(zip(words, roles)):
                word_vec = word_encoder.encode_word(word)
                # Bind word with role-position slot (GPU convolve when available).
                comp = self.convolve(word_vec, _rolepos(role, i))
                components.append(comp)

            # Bundle components into sentence vector (GPU bundle when available).
            sent_vec = self.bundle(components, normalize=True)
            sentence_vecs.append(sent_vec)
            word_lists.append(words)

        return sentence_vecs, word_lists

    def batch_build_realm_vectors(
        self,
        realm_sentence_vecs: dict[str, list[np.ndarray]],
    ) -> dict[str, np.ndarray]:
        """Bundle sentence vectors per realm into prototype vectors.

        Uses ``self.bundle`` (GPU ``vsa-bundle.spv`` when available).
        """
        realm_vectors: dict[str, np.ndarray] = {}
        for realm, vecs in realm_sentence_vecs.items():
            if vecs:
                realm_vectors[realm] = self.bundle(vecs, normalize=True)
        return realm_vectors

    def batch_similarity_search(
        self,
        query_vec: np.ndarray,
        sentence_vecs: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """Find top-k most similar sentences to *query_vec*.

        Uses ``self.similarity_batch`` (GPU ``vsa-similarity-batch.spv``
        when available).

        Args:
            query_vec: Query vector (dim,).
            sentence_vecs: Matrix of sentence vectors (N, dim).
            top_k: Number of results to return.

        Returns:
            List of (index, similarity) sorted descending.
        """
        sims = self.similarity_batch(query_vec, sentence_vecs)
        indices = np.argsort(sims)[::-1][:top_k]
        return [(int(idx), float(sims[idx])) for idx in indices]

    def batch_realm_route(
        self,
        queries: np.ndarray,
        realm_codebook: np.ndarray,
        realm_names: list[str],
    ) -> list[str]:
        """Route each query to its best-matching realm.

        Uses ``self.similarity_batch`` per query (GPU accelerated).

        Args:
            queries: (N, dim) batch of query vectors.
            realm_codebook: (R, dim) realm expert vectors.
            realm_names: Ordered list of realm names matching codebook rows.

        Returns:
            List of realm names (length N).
        """
        results: list[str] = []
        for i in range(queries.shape[0]):
            sims = self.similarity_batch(queries[i], realm_codebook)
            best_idx = int(np.argmax(sims))
            results.append(realm_names[best_idx])
        return results

    def status(self) -> str:
        """Human-readable backend status."""
        if self.using_gpu:
            return f"SVCIngestionEngine(dim={self.dim}, backend=VulkanVSA GPU)"
        return f"SVCIngestionEngine(dim={self.dim}, backend=CPU)"
