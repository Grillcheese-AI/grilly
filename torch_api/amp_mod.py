"""``torch.amp``-compatible module (Vulkan / numpy path; no CUDA)."""

from __future__ import annotations

from typing import Any

from grilly.backend import amp as _amp


class autocast:
    """Autocast context (delegates to :mod:`grilly.backend.amp`). Accepts ``device_type`` / ``dtype`` for API parity."""

    def __init__(
        self,
        device_type: str | None = None,
        enabled: bool = True,
        dtype: Any = None,
        cache_enabled: bool = True,
    ) -> None:
        del device_type, dtype, cache_enabled
        self._inner = _amp.autocast(enabled=enabled)

    def __enter__(self) -> autocast:
        self._inner.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self._inner.__exit__(*args)


class GradScaler:
    """GradScaler shim; first positional arg may be ``'vulkan'`` / ``'cuda'`` (ignored) or ``init_scale`` (float)."""

    def __init__(
        self,
        device_type: Any = None,
        enabled: bool = True,
        *,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        **kwargs: Any,
    ) -> None:
        del kwargs
        scale = init_scale
        if isinstance(device_type, (int, float)) and not isinstance(device_type, bool):
            scale = float(device_type)
        self._inner = _amp.GradScaler(
            init_scale=scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            enabled=enabled,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
