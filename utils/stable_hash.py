"""Deterministic hashing helpers used for reproducible vector seeding."""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

BytesLike = bytes | bytearray | memoryview
Part = str | int | float | bytes

try:
    from blake3 import blake3 as _blake3  # type: ignore

    _USING_BLAKE3 = True
except Exception:
    _blake3 = None
    _USING_BLAKE3 = False
    _warned = False

    def _warn_once():
        """Emit a single warning about the hashing fallback."""
        global _warned
        if not _warned:
            warnings.warn(
                "blake3 not installed; falling back to hashlib.blake2s for stable hashing. "
                "Install blake3 for canonical outputs: `pip install blake3`.",
                RuntimeWarning,
            )
            _warned = True


def using_blake3() -> bool:
    """Run using blake3."""

    return _USING_BLAKE3


def _to_bytes(x: Part) -> bytes:
    """Run to bytes."""

    if isinstance(x, bytes):
        return x
    if isinstance(x, str):
        return x.encode("utf-8", errors="surrogatepass")
    # ints/floats: stable textual form
    return str(x).encode("utf-8", errors="surrogatepass")


def _join_parts(parts: Iterable[Part]) -> bytes:
    # \x1f = ASCII unit separator (rare in text)
    """Run join parts."""

    return b"\x1f".join(_to_bytes(p) for p in parts)


def digest(parts: Iterable[Part], *, domain: str = "grilly", out_len: int = 32) -> bytes:
    """Run digest."""

    msg = _join_parts([domain] + list(parts))
    if _USING_BLAKE3:
        return _blake3(msg).digest(out_len)  # type: ignore
    _warn_once()
    # BLAKE2s output is max 32 bytes; chain if longer requested
    if out_len <= 32:
        return hashlib.blake2s(msg).digest(out_len)
    out = bytearray()
    ctr = 0
    while len(out) < out_len:
        h = hashlib.blake2s(msg + ctr.to_bytes(4, "little")).digest()
        out.extend(h)
        ctr += 1
    return bytes(out[:out_len])


def stable_u32(*parts: Part, domain: str = "grilly") -> int:
    """Run stable u32."""

    d = digest(parts, domain=domain, out_len=4)
    return int.from_bytes(d, "little", signed=False)


def stable_u64(*parts: Part, domain: str = "grilly") -> int:
    """Run stable u64."""

    d = digest(parts, domain=domain, out_len=8)
    return int.from_bytes(d, "little", signed=False)


def stable_bytes(*parts: Part, domain: str = "grilly", out_len: int = 32) -> bytes:
    """Run stable bytes."""

    return digest(parts, domain=domain, out_len=out_len)


def bipolar_from_key(key: str, dim: int, *, domain: str = "grilly.bipolar") -> np.ndarray:
    """
    Deterministically generate a bipolar (+1/-1) vector using a hash-stream.
    This avoids RNG differences across numpy versions.

    Requires numpy at runtime but is kept here to centralize the logic.
    """
    import numpy as np

    if dim <= 0:
        return np.zeros((0,), dtype=np.float32)

    # Need dim bits
    nbytes = (dim + 7) // 8
    # stream bytes from hash(domain||key||counter)
    out = bytearray()
    ctr = 0
    while len(out) < nbytes:
        out.extend(digest((key, ctr), domain=domain, out_len=32))
        ctr += 1
    buf = bytes(out[:nbytes])

    bits = np.unpackbits(np.frombuffer(buf, dtype=np.uint8), bitorder="little")[:dim]
    # 0 -> -1, 1 -> +1
    return (bits.astype(np.int8) * 2 - 1).astype(np.float32)
