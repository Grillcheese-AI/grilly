"""
Grilly checkpoint format (.grl) — GRL v1.

Binary layout:
  - Magic ``GRLY`` (4 bytes)
  - uint16 format version (1)
  - uint16 flags (reserved, 0)
  - uint32 reserved
  - uint64 metadata_json_offset, metadata_json_length
  - uint64 tensor_index_offset, tensor_index_length
  - uint64 payload_offset, payload_length
  - padding to 64-byte header

Followed by:
  - UTF-8 JSON metadata blob
  - UTF-8 JSON tensor index (array of tensor descriptors)
  - payload (concatenated tensor bytes, C-contiguous row-major)

Tensor index entry::
  {"name": str, "dtype": "f32"|"f16"|"i64"|"i32"|"u8", "shape": [int,...],
   "offset": int, "length": int}

``offset`` / ``length`` are byte ranges relative to **start of payload section**.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

try:
    import grilly_core as _grl_core

    _HAS_CPP_GRL = hasattr(_grl_core, "grl_write_file")
except ImportError:
    _grl_core = None
    _HAS_CPP_GRL = False

MAGIC = b"GRLY"
FORMAT_VERSION = 1
HEADER_SIZE = 64
_FLAG_NONE = 0

_DTYPE_TO_STR = {
    np.dtype("float32"): "f32",
    np.dtype("float16"): "f16",
    np.dtype("int64"): "i64",
    np.dtype("int32"): "i32",
    np.dtype("uint8"): "u8",
}
_STR_TO_DTYPE = {v: k for k, v in _DTYPE_TO_STR.items()}


def _flatten_state_dict(d: dict[str, Any], prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten nested dict to dotted keys -> ndarray."""
    out: dict[str, np.ndarray] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_state_dict(v, key))
        elif isinstance(v, np.ndarray):
            out[key] = np.ascontiguousarray(v)
        else:
            try:
                out[key] = np.asarray(v)
            except Exception:
                continue
    return out


def _unflatten_state_dict(flat: dict[str, np.ndarray]) -> dict[str, Any]:
    """Rebuild nested dict from dotted keys."""
    root: dict[str, Any] = {}
    for key, arr in flat.items():
        parts = key.split(".")
        cur = root
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = arr
    return root


def save_grl(
    filepath: str | Path,
    state_dict: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write a GRL v1 checkpoint from a (possibly nested) state dict of numpy arrays."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    flat = _flatten_state_dict(state_dict)
    meta = {
        "schema": "grilly.checkpoint.v1",
        "framework": "grilly",
        **(metadata or {}),
    }
    meta_bytes = json.dumps(meta, separators=(",", ":"), sort_keys=True).encode("utf-8")

    payload_parts: list[bytes] = []
    index_entries: list[dict[str, Any]] = []
    offset = 0
    for name in sorted(flat.keys()):
        arr = flat[name]
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        dt = arr.dtype
        if dt not in _DTYPE_TO_STR:
            arr = arr.astype(np.float32)
            dt = arr.dtype
        raw = arr.tobytes(order="C")
        dtype_str = _DTYPE_TO_STR.get(dt, "f32")
        index_entries.append(
            {
                "name": name,
                "dtype": dtype_str,
                "shape": list(arr.shape),
                "offset": offset,
                "length": len(raw),
            }
        )
        payload_parts.append(raw)
        offset += len(raw)

    payload = b"".join(payload_parts)
    index_bytes = json.dumps(index_entries, separators=(",", ":")).encode("utf-8")

    # Layout: header | meta | index | payload
    meta_off = HEADER_SIZE
    meta_len = len(meta_bytes)
    idx_off = meta_off + meta_len
    idx_len = len(index_bytes)
    pay_off = idx_off + idx_len
    pay_len = len(payload)

    if _HAS_CPP_GRL:
        _grl_core.grl_write_file(
            str(path),
            meta_bytes.decode("utf-8"),
            index_bytes.decode("utf-8"),
            payload,
        )
        return

    header = bytearray(HEADER_SIZE)
    header[0:4] = MAGIC
    struct.pack_into("<HHI", header, 4, FORMAT_VERSION, _FLAG_NONE, 0)
    struct.pack_into("<QQ", header, 12, meta_off, meta_len)
    struct.pack_into("<QQ", header, 28, idx_off, idx_len)
    struct.pack_into("<QQ", header, 44, pay_off, pay_len)

    with open(path, "wb") as f:
        f.write(header)
        f.write(meta_bytes)
        f.write(index_bytes)
        f.write(payload)


def load_grl(filepath: str | Path, *, map_location: Any = None) -> dict[str, Any]:
    """
    Load GRL v1 checkpoint. Returns a dict with ``metadata``, ``model`` (nested state_dict),
    and any extra keys from metadata.

    ``map_location`` is accepted for torch API compatibility; ``\"cpu\"`` keeps arrays
    on host; ``\"vulkan\"`` / default leaves numpy arrays (caller uploads to GPU).
    """
    _ = map_location
    path = Path(filepath)

    if _HAS_CPP_GRL:
        meta_json, index_json, pay_bytes = _grl_core.grl_read_file(str(path))
        metadata = json.loads(meta_json)
        index_entries = json.loads(index_json)
        payload = memoryview(pay_bytes)
        flat: dict[str, np.ndarray] = {}
        for ent in index_entries:
            name = ent["name"]
            dtype_s = ent["dtype"]
            shape = tuple(ent["shape"])
            off = int(ent["offset"])
            ln = int(ent["length"])
            dt = _STR_TO_DTYPE.get(dtype_s, np.float32)
            buf = payload[off : off + ln].tobytes()
            arr = np.frombuffer(buf, dtype=dt).reshape(shape)
            flat[name] = np.array(arr, copy=True)
        # Roundtrip semantics: return what the user saved, not a forced
        # ``{'model': ..., 'metadata': ...}`` wrapper. ``torch.save`` /
        # ``torch.load`` users expect ``ck == original_payload``.
        out: dict[str, Any] = _unflatten_state_dict(flat)
        # Add metadata only if it doesn't collide with a user key.
        if "metadata" not in out:
            out["metadata"] = metadata
        # Promote common training scalars from metadata to top level when
        # the user didn't include them in the original payload (back-compat
        # with checkpoints saved by older grilly that only put step in meta).
        for k in ("step", "training_step", "best_ppl", "epoch"):
            if k in metadata and k not in out:
                out[k] = metadata[k]
        return out

    data = path.read_bytes()
    if len(data) < HEADER_SIZE or data[0:4] != MAGIC:
        raise ValueError(f"Not a GRL file or corrupt magic: {path}")

    version = struct.unpack_from("<H", data, 4)[0]
    if version != FORMAT_VERSION:
        raise ValueError(f"Unsupported GRL format version {version}")

    meta_off, meta_len = struct.unpack_from("<QQ", data, 12)
    idx_off, idx_len = struct.unpack_from("<QQ", data, 28)
    pay_off, pay_len = struct.unpack_from("<QQ", data, 44)

    meta_json = data[meta_off : meta_off + meta_len].decode("utf-8")
    index_json = data[idx_off : idx_off + idx_len].decode("utf-8")
    payload = memoryview(data)[pay_off : pay_off + pay_len]

    metadata = json.loads(meta_json)
    index_entries = json.loads(index_json)

    flat: dict[str, np.ndarray] = {}
    for ent in index_entries:
        name = ent["name"]
        dtype_s = ent["dtype"]
        shape = tuple(ent["shape"])
        off = int(ent["offset"])
        ln = int(ent["length"])
        dt = _STR_TO_DTYPE.get(dtype_s, np.float32)
        buf = payload[off : off + ln].tobytes()
        arr = np.frombuffer(buf, dtype=dt).reshape(shape)
        flat[name] = np.array(arr, copy=True)

    nested = _unflatten_state_dict(flat)
    out: dict[str, Any] = {"metadata": metadata, "model": nested}
    # Promote common training keys from metadata to top level for torch-style access
    for k in ("step", "training_step", "best_ppl", "epoch"):
        if k in metadata:
            out[k] = metadata[k]
    return out


def state_dict_to_torch_style(grl_dict: dict[str, Any]) -> dict[str, Any]:
    """Map load_grl output to a torch-like checkpoint dict (``model``, ``step``, ...)."""
    d = dict(grl_dict)
    if "model" in d and isinstance(d["model"], dict):
        # torch often uses 'model' key for state_dict
        pass
    return d
