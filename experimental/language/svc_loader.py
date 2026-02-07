"""SVC Data Loader & GPU-Accelerated Ingestion Engine.

Provides:
- SVCEntry dataclass for parsed entries
- Streaming JSONL loader with realm filtering
- Batch loading with statistics
- SVCIngestionEngine that uses VulkanVSA GPU shaders when available
  (bundle, bundle_batch, similarity_batch, bind_bipolar_batch)
  and falls back to CPU ops (HolographicOps, BinaryOps) otherwise
"""

import json
import re
import time
import numpy as np
from typing import Dict, List, Optional, Iterator, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

from grilly.experimental.vsa.ops import HolographicOps, BinaryOps

if TYPE_CHECKING:
    from grilly.backend.experimental.vsa import VulkanVSA
    from grilly.experimental.language.encoder import WordEncoder, SentenceEncoder
    from grilly.experimental.language.generator import SentenceGenerator


# ---------------------------------------------------------------------------
# Data layer – parsing and filtering
# ---------------------------------------------------------------------------

@dataclass
class SVCEntry:
    """A parsed SVC entry from the training data."""
    id: str
    text: str
    svc_s: str
    svc_v: str
    svc_c: str
    pos: List[str]
    deps: List[str]
    lemmas: List[str]
    root_verb: str
    realm: str
    source: str
    complexity: float

    @classmethod
    def from_dict(cls, data: Dict) -> 'SVCEntry':
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

    def tokenize(self) -> List[str]:
        """Tokenize text the same way InstantLanguage does."""
        text = re.sub(r'[^\w\s]', '', self.text.lower())
        return text.split()

    def to_roles(self) -> Tuple[List[str], List[str]]:
        """Map SVC s/v/c fields to per-word SUBJ/VERB/OBJ roles.

        Returns:
            (words, roles) where words are tokenized and lowercased.
        """
        words = self.tokenize()
        v_words = set(re.sub(r'[^\w\s]', '', self.svc_v.lower()).split())
        s_words = set(re.sub(r'[^\w\s]', '', self.svc_s.lower()).split())
        c_words = set(re.sub(r'[^\w\s]', '', self.svc_c.lower()).split())

        roles: List[str] = []
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
    entries: List[SVCEntry]
    realm_counts: Dict[str, int] = field(default_factory=dict)
    source_counts: Dict[str, int] = field(default_factory=dict)
    verb_counts: Dict[str, int] = field(default_factory=dict)
    avg_complexity: float = 0.0
    total_loaded: int = 0
    total_skipped: int = 0

    @property
    def realms(self) -> List[str]:
        return sorted(self.realm_counts.keys())

    @property
    def realm_entries(self) -> Dict[str, List[SVCEntry]]:
        grouped: Dict[str, List[SVCEntry]] = defaultdict(list)
        for entry in self.entries:
            grouped[entry.realm].append(entry)
        return dict(grouped)

    def summary(self) -> str:
        lines = [
            f"SVCBatch: {len(self.entries)} entries loaded "
            f"({self.total_skipped} skipped)",
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
    max_entries: Optional[int] = None,
    realms: Optional[List[str]] = None,
    min_complexity: Optional[float] = None,
    max_complexity: Optional[float] = None,
    sources: Optional[List[str]] = None,
) -> Iterator[SVCEntry]:
    """Load SVC entries from JSONL file with optional filtering."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"SVC data file not found: {path}")

    count = 0
    with open(path_obj, 'r', encoding='utf-8') as f:
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
    max_entries: Optional[int] = None,
    realms: Optional[List[str]] = None,
    min_complexity: Optional[float] = None,
    max_complexity: Optional[float] = None,
    sources: Optional[List[str]] = None,
) -> SVCBatch:
    """Load SVC entries into a batch with computed statistics."""
    entries: List[SVCEntry] = []
    realm_counts: Dict[str, int] = defaultdict(int)
    source_counts: Dict[str, int] = defaultdict(int)
    verb_counts: Dict[str, int] = defaultdict(int)
    total_complexity = 0.0
    total_skipped = 0

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"SVC data file not found: {path}")

    with open(path_obj, 'r', encoding='utf-8') as f:
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
    data: List[Dict],
    realms: Optional[List[str]] = None,
) -> List[SVCEntry]:
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

def _try_get_vulkan_vsa() -> Optional['VulkanVSA']:
    """Attempt to create a VulkanVSA instance.  Returns None on failure."""
    try:
        from grilly.backend.experimental.vsa import VulkanVSA
        from grilly.backend.core import VulkanCore
        core = VulkanCore()
        return VulkanVSA(core)
    except Exception:
        return None


class SVCIngestionEngine:
    """GPU-accelerated SVC ingestion engine.

    Wraps the full encode → store → bundle → route pipeline and
    dispatches to VulkanVSA GPU shaders when available:

    ┌──────────────────────────────────────────────────────────────┐
    │  GPU Path (VulkanVSA)             CPU Fallback              │
    │  ─────────────────────            ────────────              │
    │  bind_bipolar_batch  ←──or──→  BinaryOps.bind_batch        │
    │  bundle / bundle_batch ←or──→  HolographicOps.bundle       │
    │  similarity_batch    ←──or──→  HolographicOps.similarity   │
    │  resonator_step      ←──or──→  codebook @ query            │
    │  circular_convolve   ←──or──→  HolographicOps.convolve     │
    └──────────────────────────────────────────────────────────────┘

    Usage::

        engine = SVCIngestionEngine(dim=2048)   # auto-detects GPU
        engine = SVCIngestionEngine(dim=2048, gpu=my_vulkan_vsa)
        engine = SVCIngestionEngine(dim=2048, gpu=False)  # force CPU
    """

    def __init__(
        self,
        dim: int,
        gpu: Optional[object] = None,
    ):
        """
        Args:
            dim: Hypervector dimension.
            gpu: One of:
                 - ``None``  → auto-detect VulkanVSA
                 - ``False`` → force CPU path
                 - A ``VulkanVSA`` instance → use it directly
        """
        self.dim = dim

        if gpu is False:
            self._gpu: Optional['VulkanVSA'] = None
        elif gpu is None:
            self._gpu = _try_get_vulkan_vsa()
        else:
            self._gpu = gpu  # type: ignore[assignment]

        self.using_gpu = self._gpu is not None

    # -- core ops (GPU or CPU) ----------------------------------------

    def bundle(
        self,
        vectors: List[np.ndarray],
        normalize: bool = True,
    ) -> np.ndarray:
        """Bundle (superpose) a list of vectors.

        GPU: ``VulkanVSA.bundle`` → ``vsa-bundle.spv``
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

        GPU: ``VulkanVSA.similarity_batch`` → ``vsa-similarity-batch.spv``
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

        GPU: ``VulkanVSA.bind_bipolar_batch`` → ``vsa-bind-batch.spv``
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

        GPU: ``VulkanVSA.circular_convolve`` → ``vsa-fft-convolve.spv``
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
        other_estimates: Optional[List[np.ndarray]] = None,
    ) -> Tuple[np.ndarray, int]:
        """One resonator projection step.

        GPU: ``VulkanVSA.resonator_step`` → ``vsa-resonator-step.spv``
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

    def bundle_batch(self, vectors: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Bundle batch [B, L, D] vectors along axis 1."""
        if self._gpu is not None:
            try:
                return self._gpu.bundle_batch(vectors, normalize=normalize)
            except Exception:
                pass
        return HolographicOps.bundle_batch(vectors, normalize=normalize)

    def batch_encode_sentences(
        self,
        entries: List[SVCEntry],
        word_encoder: 'WordEncoder',
        sentence_encoder: 'SentenceEncoder',
    ) -> Tuple[List[np.ndarray], List[List[str]]]:
        """Encode a batch of SVC entries into sentence vectors (GPU-friendly)."""
        sentence_vecs: List[np.ndarray] = []
        word_lists: List[List[str]] = []

        if not entries:
            return sentence_vecs, word_lists

        # RDNA2-friendly GPU path: bipolar bind + bundle_batch
        if self._gpu is not None:
            try:
                dim = self.dim
                B = len(entries)

                words_per: List[List[str]] = []
                roles_per: List[List[str]] = []
                lengths: List[int] = []

                max_len = 0
                for entry in entries:
                    words, roles = entry.to_roles()
                    # ensure vocabulary is populated / stable vectors
                    for w in words:
                        word_encoder.encode_word(w)
                    words_per.append(words)
                    roles_per.append(roles)
                    lengths.append(len(words))
                    max_len = max(max_len, len(words))

                if max_len == 0:
                    return [np.zeros(dim, dtype=np.float32) for _ in entries], words_per

                # Build padded tensors [B, L, D]
                W = np.zeros((B, max_len, dim), dtype=np.float32)
                R = np.zeros((B, max_len, dim), dtype=np.float32)
                P = np.zeros((B, max_len, dim), dtype=np.float32)

                pos_vecs = sentence_encoder.position_vectors
                pos_mod = len(pos_vecs)

                for b in range(B):
                    words = words_per[b]
                    roles = roles_per[b]
                    for i, (word, role) in enumerate(zip(words, roles)):
                        wv = word_encoder.encode_word(word)
                        W[b, i] = np.where(wv >= 0.0, 1.0, -1.0)
                        rv = sentence_encoder.roles.get(role, sentence_encoder.roles["ROOT"])
                        R[b, i] = np.where(rv >= 0.0, 1.0, -1.0)
                        pv = pos_vecs[i % pos_mod]
                        P[b, i] = np.where(pv >= 0.0, 1.0, -1.0)

                # Flatten to [B*L, D] for bind_batch
                flatW = W.reshape(B * max_len, dim)
                flatR = R.reshape(B * max_len, dim)
                flatP = P.reshape(B * max_len, dim)

                comp = self.bind_bipolar_batch(flatW, flatR)
                comp = self.bind_bipolar_batch(comp, flatP)

                comp3 = comp.reshape(B, max_len, dim)

                summed = self.bundle_batch(comp3, normalize=False)  # [B, D]

                # Per-sentence normalization using true lengths (avoid padding bias)
                for b in range(B):
                    L = lengths[b]
                    if L > 0:
                        sent = summed[b] / float(np.sqrt(L))
                    else:
                        sent = np.zeros(dim, dtype=np.float32)
                    norm = np.linalg.norm(sent)
                    if norm > 0:
                        sent = sent / norm
                    sentence_vecs.append(sent.astype(np.float32, copy=False))
                    word_lists.append(words_per[b])

                return sentence_vecs, word_lists

            except Exception:
                # fall through to CPU loop
                pass

        # CPU / legacy path (HRR-style convolve)
        for entry in entries:
            words, roles = entry.to_roles()

            for w in words:
                word_encoder.encode_word(w)

            components: List[np.ndarray] = []
            for i, (word, role) in enumerate(zip(words, roles)):
                word_vec = word_encoder.encode_word(word)
                role_vec = sentence_encoder.roles.get(role, sentence_encoder.roles["ROOT"])
                pos_vec = sentence_encoder.position_vectors[i % len(sentence_encoder.position_vectors)]
                comp = self.convolve(word_vec, role_vec)
                comp = self.convolve(comp, pos_vec)
                components.append(comp)

            sent_vec = self.bundle(components, normalize=True)
            sentence_vecs.append(sent_vec)
            word_lists.append(words)

        return sentence_vecs, word_lists

    def batch_build_realm_vectors(
        self,
        realm_sentence_vecs: Dict[str, List[np.ndarray]],
    ) -> Dict[str, np.ndarray]:
        """Bundle sentence vectors per realm into prototype vectors."""
        realm_vectors: Dict[str, np.ndarray] = {}
        for realm, vecs in realm_sentence_vecs.items():
            if vecs:
                realm_vectors[realm] = self.bundle(vecs, normalize=True)
        return realm_vectors

    def batch_similarity_search(
        self,
        query_vec: np.ndarray,
        sentence_vecs: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """Find top-k most similar sentences to *query_vec*."""
        sims = self.similarity_batch(query_vec, sentence_vecs)
        indices = np.argsort(sims)[::-1][:top_k]
        return [(int(idx), float(sims[idx])) for idx in indices]

    def batch_realm_route(
        self,
        queries: np.ndarray,
        realm_codebook: np.ndarray,
        realm_names: List[str],
    ) -> List[str]:
        """Route each query to its best-matching realm."""
        results: List[str] = []
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
