"""Hippocampal checkpoint format (.hippo.npz).

Arrays are prefixed by hippocampal circuit stage:
  EC  (Entorhinal Cortex)  – input representations (vocab, relations)
  DG  (Dentate Gyrus)      – pattern separation (realms, patterns, templates)
  CA3 (Associative memory) – facts, expectations, constraints
  CA1 (Consolidated output) – episodic sentence memory
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Shared helpers (same logic as ingest_checkpoint.py)
# ---------------------------------------------------------------------------


def _as_obj_array(items) -> np.ndarray:
    """Create a numpy object array from an iterable."""
    return np.asarray(list(items), dtype=object)


def _stack_or_empty(vecs, dim: int, dtype=np.float32) -> np.ndarray:
    """Stack vectors or return an empty (0, dim) array."""
    if not vecs:
        return np.zeros((0, dim), dtype=dtype)
    return np.stack(vecs).astype(dtype, copy=False)


def _is_bipolar_matrix(x: np.ndarray, tol: float = 1e-3) -> bool:
    """Return True when every element is near +1 or -1."""
    if x.size == 0:
        return False
    a = np.abs(x)
    return bool(np.all(np.abs(a - 1.0) <= tol))


def _pack_bipolar(mat: np.ndarray) -> np.ndarray:
    """Bitpack a bipolar {-1,+1} matrix to uint8."""
    bits = (mat > 0).astype(np.uint8)
    return np.packbits(bits, axis=1, bitorder="little")


def _unpack_bipolar(packed: np.ndarray, dim: int, dtype=np.float32) -> np.ndarray:
    """Unpack bitpacked uint8 back to bipolar {-1,+1}."""
    bits = np.unpackbits(packed, axis=1, count=dim, bitorder="little")
    return (bits.astype(np.int8) * 2 - 1).astype(dtype)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_hippocampal_checkpoint(
    path: str,
    controller: Any,
    *,
    fp16: bool = True,
    include_fact_vectors: bool = True,
    include_sentence_memory: bool = False,
    max_sentences: int = 0,
    sentence_compress: str = "auto",
    metadata: dict[str, Any] | None = None,
) -> str:
    """Save a hippocampal checkpoint (.hippo.npz).

    Returns the resolved output path as a string.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    dim = int(getattr(controller, "dim", 0) or getattr(controller.language, "dim", 0))
    lang = controller.language
    world = controller.world
    we = lang.word_encoder

    # ---- EC: vocab & relations ----
    vocab_words = sorted(getattr(we, "word_vectors", {}).keys())
    vocab_vecs = _stack_or_empty([we.word_vectors[w] for w in vocab_words], dim)

    rel_names = sorted(getattr(we, "relations", {}).keys())
    rel_vecs = _stack_or_empty([we.relations[n] for n in rel_names], dim)

    # relation_memory: Dict[str, List[Tuple[str, str]]]
    raw_rm = getattr(lang, "relation_memory", {}) or {}
    relation_memory_pairs = 0
    relation_memory_dict: dict[str, list] = {}
    for rname, pairs in raw_rm.items():
        pair_list = [[a, b] for (a, b) in pairs]
        relation_memory_dict[rname] = pair_list
        relation_memory_pairs += len(pair_list)

    # ---- DG: realms, patterns, templates ----
    realm_vectors = getattr(lang, "realm_vectors", {}) or {}
    realm_names = sorted(realm_vectors.keys())
    realm_vecs = _stack_or_empty([realm_vectors[n] for n in realm_names], dim)

    patterns = getattr(getattr(lang, "generator", None), "patterns", {}) or {}
    pattern_names = sorted(patterns.keys())
    pattern_vecs = _stack_or_empty([patterns[n] for n in pattern_names], dim)

    # templates: Dict[str, dict] from SentenceGenerator
    raw_templates = getattr(getattr(lang, "generator", None), "templates", {}) or {}
    templates_dict: dict[str, dict] = {}
    for tname, tval in raw_templates.items():
        templates_dict[tname] = {
            "pattern": list(tval.get("pattern", [])),
            "example": list(tval.get("example", [])),
        }

    # ---- CA3: facts, expectations, constraints ----
    facts = getattr(world, "facts", []) or []
    fact_s = _as_obj_array([f.subject for f in facts])
    fact_r = _as_obj_array([f.relation for f in facts])
    fact_o = _as_obj_array([f.object for f in facts])
    fact_conf = np.asarray([float(getattr(f, "confidence", 1.0)) for f in facts], dtype=np.float32)
    fact_src = _as_obj_array([getattr(f, "source", "observed") for f in facts])

    capsule_dim = int(getattr(world, "capsule_dim", 0) or 0)
    semantic_dims = int(getattr(world, "semantic_dims", 0) or 0)

    if include_fact_vectors:
        fact_vecs = _stack_or_empty([f.vector for f in facts], dim)
        cap_dim = max(1, capsule_dim)
        caps: list = []
        has_caps_list: list = []
        for f in facts:
            cv = getattr(f, "capsule_vector", None)
            if cv is None:
                has_caps_list.append(False)
            else:
                has_caps_list.append(True)
                caps.append(cv)
        fact_caps = _stack_or_empty(caps, cap_dim)
        has_caps = np.asarray(has_caps_list, dtype=np.bool_)
    else:
        fact_vecs = np.zeros((0, dim), dtype=np.float32)
        fact_caps = np.zeros((0, 1), dtype=np.float32)
        has_caps = np.zeros((len(facts),), dtype=np.bool_)

    expectations = getattr(world, "expectations", {}) or {}
    expectations_json = json.dumps(expectations)

    constraints = getattr(world, "constraints", []) or []
    if constraints:
        c_a = _stack_or_empty([a for (a, _) in constraints], dim)
        c_b = _stack_or_empty([b for (_, b) in constraints], dim)
    else:
        c_a = np.zeros((0, dim), dtype=np.float32)
        c_b = np.zeros((0, dim), dtype=np.float32)

    # ---- CA1: sentence memory ----
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

        sent_vecs = _stack_or_empty([v for (v, _) in mem], dim)

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

        mode = (sentence_compress or "auto").lower()
        if mode not in ("auto", "fp16", "bitpack"):
            mode = "auto"

        if mode == "bitpack" or (mode == "auto" and _is_bipolar_matrix(sent_vecs)):
            sent_packed = _pack_bipolar(sent_vecs)
            sent_pack_mode = "bitpack"
            sent_vecs = np.zeros((0, dim), dtype=np.float32)
        else:
            sent_pack_mode = "fp16"

    # ---- fp16 downcast ----
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

    # ---- manifest ----
    n_sentences = sent_packed.shape[0] if sent_pack_mode == "bitpack" else sent_vecs.shape[0]
    manifest = {
        "format": "grilly.hippocampal.v1",
        "saved_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dim": dim,
        "capsule_dim": capsule_dim,
        "semantic_dims": semantic_dims,
        "fp16": bool(fp16),
        "word_use_ngrams": bool(getattr(we, "use_ngrams", True)),
        "sentence_pack": sent_pack_mode,
        "counts": {
            "vocab": len(vocab_words),
            "relations": len(rel_names),
            "relation_memory_pairs": relation_memory_pairs,
            "realms": len(realm_names),
            "patterns": len(pattern_names),
            "templates": len(templates_dict),
            "facts": len(facts),
            "constraints": len(constraints),
            "sentences": int(n_sentences),
        },
        "index": {
            "arrays": [
                "ec_vocab_words",
                "ec_vocab_vecs",
                "ec_relation_names",
                "ec_relation_vecs",
                "ec_relation_memory_json",
                "dg_realm_names",
                "dg_realm_vecs",
                "dg_pattern_names",
                "dg_pattern_vecs",
                "dg_templates_json",
                "ca3_fact_s",
                "ca3_fact_r",
                "ca3_fact_o",
                "ca3_fact_conf",
                "ca3_fact_src",
                "ca3_fact_vecs",
                "ca3_fact_caps",
                "ca3_fact_has_caps",
                "ca3_expectations_json",
                "ca3_constraint_a",
                "ca3_constraint_b",
                "ca1_sent_vecs",
                "ca1_sent_packed",
                "ca1_sent_token_vocab",
                "ca1_sent_token_ids",
                "ca1_sent_offsets",
            ],
        },
    }
    if metadata:
        manifest.update(metadata)

    # ---- write archive ----
    np.savez_compressed(
        str(out),
        manifest_json=np.asarray([json.dumps(manifest)], dtype=object),
        # EC
        ec_vocab_words=_as_obj_array(vocab_words),
        ec_vocab_vecs=vocab_vecs,
        ec_relation_names=_as_obj_array(rel_names),
        ec_relation_vecs=rel_vecs,
        ec_relation_memory_json=np.asarray([json.dumps(relation_memory_dict)], dtype=object),
        # DG
        dg_realm_names=_as_obj_array(realm_names),
        dg_realm_vecs=realm_vecs,
        dg_pattern_names=_as_obj_array(pattern_names),
        dg_pattern_vecs=pattern_vecs,
        dg_templates_json=np.asarray([json.dumps(templates_dict)], dtype=object),
        # CA3
        ca3_fact_s=fact_s,
        ca3_fact_r=fact_r,
        ca3_fact_o=fact_o,
        ca3_fact_conf=fact_conf,
        ca3_fact_src=fact_src,
        ca3_fact_vecs=fact_vecs,
        ca3_fact_caps=fact_caps,
        ca3_fact_has_caps=has_caps,
        ca3_expectations_json=np.asarray([expectations_json], dtype=object),
        ca3_constraint_a=c_a,
        ca3_constraint_b=c_b,
        # CA1
        ca1_sent_vecs=sent_vecs,
        ca1_sent_packed=sent_packed,
        ca1_sent_token_vocab=sent_token_vocab,
        ca1_sent_token_ids=sent_token_ids,
        ca1_sent_offsets=sent_offsets,
    )
    return str(out)


# ---------------------------------------------------------------------------
# Read-only view
# ---------------------------------------------------------------------------


class HippoCheckpointView:
    """Lightweight read-only accessor for .hippo.npz files."""

    def __init__(self, path: str):
        self.path = path
        self._data = np.load(path, allow_pickle=True)
        self.manifest: dict = json.loads(str(self._data["manifest_json"][0]))
        self.dim: int = int(self.manifest.get("dim", 0))

    # -- EC --
    def vocab(self) -> tuple[list, np.ndarray]:
        """Return (word_list, vectors)."""
        return list(self._data["ec_vocab_words"]), self._data["ec_vocab_vecs"]

    def relation_memory(self) -> dict[str, list[list[str]]]:
        """Return relation_memory dict: {relation: [[word_a, word_b], ...]}."""
        try:
            return json.loads(str(self._data["ec_relation_memory_json"][0]))
        except Exception:
            return {}

    # -- DG --
    def templates(self) -> dict[str, dict]:
        """Return templates dict: {name: {pattern: [...], example: [...]}}."""
        try:
            return json.loads(str(self._data["dg_templates_json"][0]))
        except Exception:
            return {}

    # -- CA3 --
    def facts(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (subjects, relations, objects, confidences)."""
        return (
            self._data["ca3_fact_s"],
            self._data["ca3_fact_r"],
            self._data["ca3_fact_o"],
            self._data["ca3_fact_conf"],
        )

    # -- CA1 --
    def sentence_count(self) -> int:
        mode = self.manifest.get("sentence_pack", "none")
        if mode == "bitpack":
            return int(self._data["ca1_sent_packed"].shape[0])
        if mode == "fp16":
            return int(self._data["ca1_sent_vecs"].shape[0])
        return 0

    def get_sentence_vector(self, i: int, dtype=np.float32) -> np.ndarray:
        mode = self.manifest.get("sentence_pack", "none")
        if mode == "bitpack":
            packed = self._data["ca1_sent_packed"][i : i + 1]
            return _unpack_bipolar(packed, self.dim, dtype=dtype)[0]
        return self._data["ca1_sent_vecs"][i].astype(dtype, copy=False)

    def get_sentence_tokens(self, i: int) -> list[str]:
        offsets = self._data["ca1_sent_offsets"].astype(int, copy=False)
        a, b = int(offsets[i]), int(offsets[i + 1])
        ids = self._data["ca1_sent_token_ids"][a:b].astype(int, copy=False)
        vocab = self._data["ca1_sent_token_vocab"]
        return [str(vocab[j]) for j in ids]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_hippocampal_checkpoint(
    path: str,
    controller: Any,
    *,
    strict_dim: bool = True,
) -> dict:
    """Load a .hippo.npz checkpoint into *controller*.

    Returns the manifest dict.
    """
    data = np.load(str(path), allow_pickle=True)
    manifest = json.loads(str(data["manifest_json"][0]))
    dim = int(manifest.get("dim", 0))

    if strict_dim and int(getattr(controller, "dim", dim)) != dim:
        raise ValueError(f"Checkpoint dim={dim} does not match controller dim={controller.dim}")

    lang = controller.language
    world = controller.world
    we = lang.word_encoder

    # ---- EC: vocab & relations ----
    vocab_words = list(data["ec_vocab_words"])
    vocab_vecs = data["ec_vocab_vecs"].astype(np.float32, copy=False)
    we.word_vectors = {w: vocab_vecs[i] for i, w in enumerate(vocab_words)}

    rel_names = list(data["ec_relation_names"])
    rel_vecs = data["ec_relation_vecs"].astype(np.float32, copy=False)
    we.relations = {n: rel_vecs[i] for i, n in enumerate(rel_names)}

    # relation_memory
    try:
        rm_raw = json.loads(str(data["ec_relation_memory_json"][0]))
    except Exception:
        rm_raw = {}
    lang.relation_memory = defaultdict(list)
    for rname, pairs in rm_raw.items():
        for pair in pairs:
            lang.relation_memory[rname].append(tuple(pair))

    # ---- DG: realms, patterns, templates ----
    realm_names = list(data["dg_realm_names"])
    realm_vecs = data["dg_realm_vecs"].astype(np.float32, copy=False)
    lang.realm_vectors = {n: realm_vecs[i] for i, n in enumerate(realm_names)}

    pat_names = list(data["dg_pattern_names"])
    pat_vecs = data["dg_pattern_vecs"].astype(np.float32, copy=False)
    if getattr(lang, "generator", None) is not None:
        lang.generator.patterns = {n: pat_vecs[i] for i, n in enumerate(pat_names)}

    # templates
    try:
        templates_raw = json.loads(str(data["dg_templates_json"][0]))
    except Exception:
        templates_raw = {}
    if getattr(lang, "generator", None) is not None and templates_raw:
        lang.generator.templates = templates_raw

    # ---- CA3: facts ----
    # capsule config
    ckpt_capsule_dim = int(manifest.get("capsule_dim", 0))
    ckpt_semantic_dims = int(manifest.get("semantic_dims", 0))
    if ckpt_capsule_dim and hasattr(world, "capsule_dim"):
        world.capsule_dim = ckpt_capsule_dim
    if ckpt_semantic_dims and hasattr(world, "semantic_dims"):
        world.semantic_dims = ckpt_semantic_dims

    try:
        from grilly.experimental.cognitive.world import Fact
    except ModuleNotFoundError:
        from experimental.cognitive.world import Fact  # type: ignore[no-redef]

    world.facts.clear()
    world.fact_vectors.clear()
    world.fact_capsules.clear()

    fact_s = list(data["ca3_fact_s"])
    fact_r = list(data["ca3_fact_r"])
    fact_o = list(data["ca3_fact_o"])
    fact_conf = list(data["ca3_fact_conf"])
    fact_src = list(data["ca3_fact_src"])

    fact_vecs = data["ca3_fact_vecs"].astype(np.float32, copy=False)
    has_caps = data["ca3_fact_has_caps"].astype(bool, copy=False)
    fact_caps = data["ca3_fact_caps"].astype(np.float32, copy=False)
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
        expectations = json.loads(str(data["ca3_expectations_json"][0]))
    except Exception:
        expectations = {}
    world.expectations = expectations

    c_a = data["ca3_constraint_a"].astype(np.float32, copy=False)
    c_b = data["ca3_constraint_b"].astype(np.float32, copy=False)
    world.constraints = [(c_a[i], c_b[i]) for i in range(c_a.shape[0])]

    # ---- CA1: sentence memory ----
    if manifest.get("sentence_pack", "none") != "none":
        mode = manifest["sentence_pack"]
        vocab = data["ca1_sent_token_vocab"]
        ids = data["ca1_sent_token_ids"].astype(int, copy=False)
        offsets = data["ca1_sent_offsets"].astype(int, copy=False)

        if mode == "bitpack":
            packed = data["ca1_sent_packed"]
            s_vecs = _unpack_bipolar(packed, dim, dtype=np.float32)
        else:
            s_vecs = data["ca1_sent_vecs"].astype(np.float32, copy=False)

        mem = []
        for i in range(len(offsets) - 1):
            a, b = int(offsets[i]), int(offsets[i + 1])
            toks = [str(vocab[j]) for j in ids[a:b]]
            mem.append((s_vecs[i], toks))
        lang.sentence_memory = mem

    return manifest
