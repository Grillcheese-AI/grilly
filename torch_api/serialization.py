"""``torch.save`` / ``torch.load`` facades for GRL checkpoints (``.grl``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from grilly.utils.grl_checkpoint import load_grl, save_grl


def save(
    obj: object,
    f: str | Path,
    *,
    pickle_module: object | None = None,
    pickle_protocol: int | None = None,
    _use_new_zipfile_serialization: bool = True,
) -> None:
    del pickle_module, pickle_protocol, _use_new_zipfile_serialization
    path = Path(f)
    if path.suffix.lower() not in (".grl",):
        path = path.with_suffix(".grl")
    meta: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k in ("step", "training_step", "best_ppl", "epoch"):
            if k in obj and not isinstance(obj[k], dict):
                try:
                    meta[k] = obj[k]
                except TypeError:
                    pass
    save_grl(path, obj, metadata=meta or None)


def load(
    f: str | Path,
    map_location: Any = None,
    pickle_module: object | None = None,
    *,
    weights_only: bool = False,
    **kwargs: Any,
) -> Any:
    del pickle_module, weights_only, kwargs
    path = Path(f)
    if path.suffix.lower() != ".grl":
        raise ValueError(
            f"Grilly torch.load only supports .grl checkpoints (got {path.suffix!r}). "
            "Export PyTorch .pt to .grl with a one-time migration script."
        )
    return load_grl(path, map_location=map_location)
