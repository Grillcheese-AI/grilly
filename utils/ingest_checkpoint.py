"""Checkpoint helpers for SVC ingestion snapshots and restore flows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _as_obj_array(items):
    """Run as obj array."""

    return np.asarray(list(items), dtype=object)


def _stack_or_empty(vecs, dim: int, dtype=np.float32):
    """Run stack or empty."""

    if not vecs:
        return np.zeros((0, dim), dtype=dtype)
    return np.stack(vecs).astype(dtype, copy=False)


def _is_bipolar_matrix(x: np.ndarray, tol: float = 1e-3) -> bool:
    """Run is bipolar matrix."""

    if x.size == 0:
        return False
    a = np.abs(x)
    return bool(np.all(np.abs(a - 1.0) <= tol))


def _pack_bipolar(mat: np.ndarray) -> np.ndarray:
    # mat: (N, D) float32/float16 with values near {-1,+1}
    """Run pack bipolar."""

    bits = (mat > 0).astype(np.uint8)
    return np.packbits(bits, axis=1, bitorder="little")


def _unpack_bipolar(packed: np.ndarray, dim: int, dtype=np.float32) -> np.ndarray:
    """Run unpack bipolar."""

    bits = np.unpackbits(packed, axis=1, count=dim, bitorder="little")
    return (bits.astype(np.int8) * 2 - 1).astype(dtype)


def save_ingest_checkpoint(
    path: str,
    controller: Any,
    *,
    fp16: bool = True,
    include_fact_vectors: bool = True,
    include_sentence_memory: bool = False,
    max_sentences: int = 0,
    sentence_compress: str = "auto",  # auto|fp16|bitpack
    metadata: dict[str, Any] | None = None,
) -> str:
    """Run save ingest checkpoint."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    dim = int(getattr(controller, "dim", 0) or getattr(controller.language, "dim", 0))
    lang = controller.language
    world = controller.world

    we = lang.word_encoder

    vocab_words = sorted(getattr(we, "word_vectors", {}).keys())
    vocab_vecs = _stack_or_empty([we.word_vectors[w] for w in vocab_words], dim, np.float32)

    rel_names = sorted(getattr(we, "relations", {}).keys())
    rel_vecs = _stack_or_empty([we.relations[n] for n in rel_names], dim, np.float32)

    realm_vectors = getattr(lang, "realm_vectors", {}) or {}
    realm_names = sorted(realm_vectors.keys())
    realm_vecs = _stack_or_empty([realm_vectors[n] for n in realm_names], dim, np.float32)

    patterns = getattr(getattr(lang, "generator", None), "patterns", {}) or {}
    pattern_names = sorted(patterns.keys())
    pattern_vecs = _stack_or_empty([patterns[n] for n in pattern_names], dim, np.float32)

    facts = getattr(world, "facts", []) or []
    fact_s = _as_obj_array([f.subject for f in facts])
    fact_r = _as_obj_array([f.relation for f in facts])
    fact_o = _as_obj_array([f.object for f in facts])
    fact_conf = np.asarray([float(getattr(f, "confidence", 1.0)) for f in facts], dtype=np.float32)
    fact_src = _as_obj_array([getattr(f, "source", "observed") for f in facts])

    if include_fact_vectors:
        fact_vecs = _stack_or_empty([f.vector for f in facts], dim, np.float32)
        # capsule vectors are optional per fact
        cap_dim = int(getattr(world, "capsule_dim", 0) or 0)
        cap_dim = max(1, cap_dim)
        caps = []
        has_caps = []
        for f in facts:
            cv = getattr(f, "capsule_vector", None)
            if cv is None:
                has_caps.append(False)
            else:
                has_caps.append(True)
                caps.append(cv)
        fact_caps = _stack_or_empty(caps, cap_dim, np.float32)
        has_caps = np.asarray(has_caps, dtype=np.bool_)
    else:
        fact_vecs = np.zeros((0, dim), dtype=np.float32)
        fact_caps = np.zeros((0, 1), dtype=np.float32)
        has_caps = np.zeros((len(facts),), dtype=np.bool_)

    expectations = getattr(world, "expectations", {}) or {}
    expectations_json = json.dumps(expectations)

    constraints = getattr(world, "constraints", []) or []
    if constraints:
        c_a = _stack_or_empty([a for (a, b) in constraints], dim, np.float32)
        c_b = _stack_or_empty([b for (a, b) in constraints], dim, np.float32)
    else:
        c_a = np.zeros((0, dim), dtype=np.float32)
        c_b = np.zeros((0, dim), dtype=np.float32)

    # --- sentence memory (compressed) ---
    sent_vecs = np.zeros((0, dim), dtype=np.float32)
    sent_packed = np.zeros((0, (dim + 7) // 8), dtype=np.uint8)
    sent_pack_mode = "none"

    sent_token_vocab = np.zeros((0,), dtype=object)
    sent_token_ids = np.zeros((0,), dtype=np.int32)
    sent_offsets = np.zeros((0,), dtype=np.int64)

    if include_sentence_memory:
        mem: list[tuple[np.ndarray, list[str]]] = getattr(lang, "sentence_memory", []) or []
        if max_sentences and max_sentences > 0:
            mem = mem[:max_sentences]

        # vectors
        sent_vecs = _stack_or_empty([v for (v, toks) in mem], dim, np.float32)

        # token vocabulary + ids
        token_to_id: dict[str, int] = {}
        vocab_list: list[str] = []
        ids: list[int] = []
        offsets: list[int] = [0]

        for _, toks in mem:
            for t in toks:
                t = str(t)
                tid = token_to_id.get(t)
                if tid is None:
                    tid = len(vocab_list)
                    token_to_id[t] = tid
                    vocab_list.append(t)
                ids.append(tid)
            offsets.append(len(ids))

        sent_token_vocab = _as_obj_array(vocab_list)
        sent_token_ids = np.asarray(ids, dtype=np.int32)
        sent_offsets = np.asarray(offsets, dtype=np.int64)

        # vector compression decision
        mode = (sentence_compress or "auto").lower()
        if mode not in ("auto", "fp16", "bitpack"):
            mode = "auto"

        if mode == "bitpack" or (mode == "auto" and _is_bipolar_matrix(sent_vecs)):
            sent_packed = _pack_bipolar(sent_vecs)
            sent_pack_mode = "bitpack"
            # keep sent_vecs empty to avoid double storage
            sent_vecs = np.zeros((0, dim), dtype=np.float32)
        else:
            sent_pack_mode = "fp16"

    vec_dtype = np.float16 if fp16 else np.float32
    vocab_vecs = vocab_vecs.astype(vec_dtype, copy=False)
    rel_vecs = rel_vecs.astype(vec_dtype, copy=False)
    realm_vecs = realm_vecs.astype(vec_dtype, copy=False)
    pattern_vecs = pattern_vecs.astype(vec_dtype, copy=False)
    fact_vecs = fact_vecs.astype(vec_dtype, copy=False)
    fact_caps = fact_caps.astype(vec_dtype, copy=False)
    c_a = c_a.astype(vec_dtype, copy=False)
    c_b = c_b.astype(vec_dtype, copy=False)
    if sent_pack_mode == "fp16":
        sent_vecs = sent_vecs.astype(vec_dtype, copy=False)

    manifest = {
        "format": "grilly.ingest_checkpoint.v2",
        "saved_utc": __import__("datetime").datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dim": dim,
        "fp16": bool(fp16),
        "word_use_ngrams": bool(getattr(we, "use_ngrams", True)),
        "include_fact_vectors": bool(include_fact_vectors),
        "include_sentence_memory": bool(include_sentence_memory),
        "sentence_pack": sent_pack_mode,
        "sentence_token_encoding": "vocab_ids" if include_sentence_memory else "none",
        "max_sentences": int(max_sentences) if max_sentences else 0,
        "counts": {
            "vocab_words": int(len(vocab_words)),
            "relations": int(len(rel_names)),
            "realms": int(len(realm_names)),
            "patterns": int(len(pattern_names)),
            "facts": int(len(facts)),
            "constraints": int(len(constraints)),
            "sentences_saved": int(
                0
                if not include_sentence_memory
                else (sent_packed.shape[0] if sent_pack_mode == "bitpack" else sent_vecs.shape[0])
            ),
            "sentence_token_vocab": int(len(sent_token_vocab)),
            "sentence_total_tokens": int(len(sent_token_ids)),
        },
    }
    if metadata:
        manifest.update(metadata)

    # Provide a quick "index" section for readability in downstream tools
    manifest["index"] = {
        "arrays": [
            "vocab_words",
            "vocab_vecs",
            "relation_names",
            "relation_vecs",
            "realm_names",
            "realm_vecs",
            "pattern_names",
            "pattern_vecs",
            "fact_s",
            "fact_r",
            "fact_o",
            "fact_conf",
            "fact_src",
            "fact_vecs",
            "fact_caps",
            "fact_has_caps",
            "expectations_json",
            "constraint_a",
            "constraint_b",
            "sent_vecs",
            "sent_packed",
            "sent_token_vocab",
            "sent_token_ids",
            "sent_offsets",
        ]
    }

    np.savez_compressed(
        str(out),
        manifest_json=np.asarray([json.dumps(manifest)], dtype=object),
        vocab_words=_as_obj_array(vocab_words),
        vocab_vecs=vocab_vecs,
        relation_names=_as_obj_array(rel_names),
        relation_vecs=rel_vecs,
        realm_names=_as_obj_array(realm_names),
        realm_vecs=realm_vecs,
        pattern_names=_as_obj_array(pattern_names),
        pattern_vecs=pattern_vecs,
        fact_s=fact_s,
        fact_r=fact_r,
        fact_o=fact_o,
        fact_conf=fact_conf,
        fact_src=fact_src,
        fact_vecs=fact_vecs,
        fact_caps=fact_caps,
        fact_has_caps=has_caps,
        expectations_json=np.asarray([expectations_json], dtype=object),
        constraint_a=c_a,
        constraint_b=c_b,
        sent_vecs=sent_vecs,
        sent_packed=sent_packed,
        sent_token_vocab=sent_token_vocab,
        sent_token_ids=sent_token_ids,
        sent_offsets=sent_offsets,
    )
    return str(out)


class CheckpointView:
    """
    Lightweight accessor that avoids expanding everything into python objects.
    """

    def __init__(self, path: str):
        """Initialize the instance."""

        self.path = path
        self._data = np.load(path, allow_pickle=True)
        self.manifest = json.loads(str(self._data["manifest_json"][0]))
        self.dim = int(self.manifest.get("dim", 0))

    def vocab(self):
        """Execute vocab."""

        return list(self._data["vocab_words"]), self._data["vocab_vecs"]

    def facts(self):
        """Execute facts."""

        return (
            self._data["fact_s"],
            self._data["fact_r"],
            self._data["fact_o"],
            self._data["fact_conf"],
        )

    def sentence_count(self) -> int:
        """Execute sentence count."""

        if not self.manifest.get("include_sentence_memory", False):
            return 0
        mode = self.manifest.get("sentence_pack", "none")
        if mode == "bitpack":
            return int(self._data["sent_packed"].shape[0])
        if mode == "fp16":
            return int(self._data["sent_vecs"].shape[0])
        return 0

    def get_sentence_vector(self, i: int, dtype=np.float32) -> np.ndarray:
        """Execute get sentence vector."""

        mode = self.manifest.get("sentence_pack", "none")
        if mode == "bitpack":
            packed = self._data["sent_packed"][i : i + 1]
            return _unpack_bipolar(packed, self.dim, dtype=dtype)[0]
        return self._data["sent_vecs"][i].astype(dtype, copy=False)

    def get_sentence_tokens(self, i: int) -> list[str]:
        """Execute get sentence tokens."""

        offsets = self._data["sent_offsets"].astype(int, copy=False)
        a, b = int(offsets[i]), int(offsets[i + 1])
        ids = self._data["sent_token_ids"][a:b].astype(int, copy=False)
        vocab = self._data["sent_token_vocab"]
        return [str(vocab[j]) for j in ids]


def load_ingest_checkpoint(
    path: str, controller: Any, *, strict_dim: bool = True
) -> dict[str, Any]:
    """Run load ingest checkpoint."""

    p = Path(path)
    data = np.load(str(p), allow_pickle=True)
    manifest = json.loads(str(data["manifest_json"][0]))
    dim = int(manifest.get("dim", 0))

    if strict_dim and int(getattr(controller, "dim", dim)) != dim:
        raise ValueError(f"Checkpoint dim={dim} does not match controller dim={controller.dim}")

    lang = controller.language
    we = lang.word_encoder

    vocab_words = list(data["vocab_words"])
    vocab_vecs = data["vocab_vecs"].astype(np.float32, copy=False)
    we.word_vectors = {w: vocab_vecs[i] for i, w in enumerate(vocab_words)}

    rel_names = list(data["relation_names"])
    rel_vecs = data["relation_vecs"].astype(np.float32, copy=False)
    we.relations = {n: rel_vecs[i] for i, n in enumerate(rel_names)}

    realm_names = list(data["realm_names"])
    realm_vecs = data["realm_vecs"].astype(np.float32, copy=False)
    lang.realm_vectors = {n: realm_vecs[i] for i, n in enumerate(realm_names)}

    pat_names = list(data["pattern_names"])
    pat_vecs = data["pattern_vecs"].astype(np.float32, copy=False)
    if getattr(lang, "generator", None) is not None:
        lang.generator.patterns = {n: pat_vecs[i] for i, n in enumerate(pat_names)}

    world = controller.world
    try:
        from grilly.experimental.cognitive.world import Fact
    except ModuleNotFoundError:
        from experimental.cognitive.world import Fact

    world.facts.clear()
    world.fact_vectors.clear()
    world.fact_capsules.clear()

    fact_s = list(data["fact_s"])
    fact_r = list(data["fact_r"])
    fact_o = list(data["fact_o"])
    fact_conf = list(data["fact_conf"])
    fact_src = list(data["fact_src"])

    fact_vecs = data["fact_vecs"].astype(np.float32, copy=False)
    has_caps = data["fact_has_caps"].astype(bool, copy=False)

    fact_caps = data["fact_caps"].astype(np.float32, copy=False)
    cap_i = 0

    for i in range(len(fact_s)):
        vec = (
            fact_vecs[i]
            if fact_vecs.shape[0] == len(fact_s)
            else np.zeros((dim,), dtype=np.float32)
        )
        capsule_vec = None
        if has_caps[i]:
            capsule_vec = fact_caps[cap_i]
            cap_i += 1
        f = Fact(
            subject=str(fact_s[i]),
            relation=str(fact_r[i]),
            object=str(fact_o[i]),
            vector=vec,
            capsule_vector=capsule_vec,
            confidence=float(fact_conf[i]),
            source=str(fact_src[i]),
        )
        world.facts.append(f)
        world.fact_vectors.append(f.vector)
        world.fact_capsules.append(f.capsule_vector)

    try:
        expectations = json.loads(str(data["expectations_json"][0]))
    except Exception:
        expectations = {}
    world.expectations = expectations

    c_a = data["constraint_a"].astype(np.float32, copy=False)
    c_b = data["constraint_b"].astype(np.float32, copy=False)
    world.constraints = [(c_a[i], c_b[i]) for i in range(c_a.shape[0])]

    if manifest.get("include_sentence_memory", False):
        mode = manifest.get("sentence_pack", "none")
        vocab = data["sent_token_vocab"]
        ids = data["sent_token_ids"].astype(int, copy=False)
        offsets = data["sent_offsets"].astype(int, copy=False)

        if mode == "bitpack":
            packed = data["sent_packed"]
            sent_vecs = _unpack_bipolar(packed, dim, dtype=np.float32)
        else:
            sent_vecs = data["sent_vecs"].astype(np.float32, copy=False)

        mem = []
        for i in range(len(offsets) - 1):
            a, b = int(offsets[i]), int(offsets[i + 1])
            toks = [str(vocab[j]) for j in ids[a:b]]
            mem.append((sent_vecs[i], toks))
        lang.sentence_memory = mem

    return manifest
