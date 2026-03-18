"""
Automatic Mixed Precision (AMP) for Grilly

Provides autocast context manager and GradScaler, mirroring the API of
torch.cuda.amp.autocast and torch.cuda.amp.GradScaler.

When autocast is active, eligible ops (linear, conv2d, attention) are
flagged to use float16 compute with float32 accumulation. The GradScaler
scales loss values to prevent float16 gradient underflow and unscales
before the optimizer step.

Usage::

    scaler = GradScaler()
    with autocast():
        output = model(input_data)
        loss = criterion(output, target)
    scaled_loss = scaler.scale(loss)
    scaled_loss.backward()
    scaler.step(optimizer)
    scaler.update()
"""

import threading
import warnings
from contextlib import contextmanager
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Thread-local autocast state
# ---------------------------------------------------------------------------

_autocast_local = threading.local()

# Default scale configuration
_DEFAULT_INIT_SCALE: float = 65536.0  # 2^16, same default as PyTorch
_DEFAULT_GROWTH_FACTOR: float = 2.0
_DEFAULT_BACKOFF_FACTOR: float = 0.5
_DEFAULT_GROWTH_INTERVAL: int = 2000
_DEFAULT_MIN_SCALE: float = 1.0


def is_autocast_enabled() -> bool:
    """Return True if autocast is currently active on this thread."""
    return getattr(_autocast_local, "enabled", False)


def _set_autocast_enabled(value: bool) -> None:
    """Set the autocast flag for this thread."""
    _autocast_local.enabled = value


# ---------------------------------------------------------------------------
# autocast context manager
# ---------------------------------------------------------------------------


class autocast:
    """Enable automatic mixed precision for eligible operations.

    When active, the global flag ``is_autocast_enabled()`` returns True.
    Bridge functions may check this flag and dispatch float16 compute paths
    instead of float32 when it is safe to do so.

    Autocast contexts can be nested; the innermost ``enabled`` value wins.

    Args:
        enabled: Set to False to temporarily disable autocast inside a nested
                 ``with autocast(enabled=False):`` block.  Defaults to True.

    Example::

        with autocast():
            output = model(input_data)
            loss = criterion(output, target)
    """

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._prev_enabled: bool | None = None

    def __enter__(self) -> "autocast":
        self._prev_enabled = getattr(_autocast_local, "enabled", False)
        _set_autocast_enabled(self._enabled)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # Restore the previous state (supports nesting)
        _set_autocast_enabled(self._prev_enabled if self._prev_enabled is not None else False)

    # Support use as a decorator
    def __call__(self, func):  # type: ignore[override]
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):  # type: ignore[misc]
            with self.__class__(enabled=self._enabled):
                return func(*args, **kwargs)

        return wrapper


@contextmanager
def autocast_context(enabled: bool = True):
    """Functional form of the autocast context manager.

    Equivalent to ``with autocast(enabled=enabled):``.
    """
    with autocast(enabled=enabled):
        yield


# ---------------------------------------------------------------------------
# GradScaler
# ---------------------------------------------------------------------------


class GradScaler:
    """Scales loss to prevent float16 gradient underflow.

    Maintains a dynamic scale factor.  Before each optimizer step the scaler
    checks for infs/NaNs in the gradients:

    * If clean → call ``optimizer.step()`` and increase the scale after
      ``growth_interval`` consecutive clean steps.
    * If inf/NaN found → skip the optimizer step and decrease the scale by
      ``backoff_factor``.

    Args:
        init_scale: Initial scale factor. Default 65536.0 (2**16).
        growth_factor: Multiplicative factor applied to the scale on a
            successful growth event. Default 2.0.
        backoff_factor: Multiplicative factor applied to the scale when
            inf/NaN is detected. Default 0.5.
        growth_interval: Number of consecutive unscaled steps before the
            scale is increased. Default 2000.
        enabled: If False the scaler is a no-op. Useful for easily toggling
            AMP without changing other code. Default True.

    Example::

        scaler = GradScaler()
        with autocast():
            output = model(input_data)
            loss = criterion(output, target)
        scaler.scale(loss).backward()   # or: (scaler.scale(loss)).backward()
        scaler.step(optimizer)
        scaler.update()
    """

    def __init__(
        self,
        init_scale: float = _DEFAULT_INIT_SCALE,
        growth_factor: float = _DEFAULT_GROWTH_FACTOR,
        backoff_factor: float = _DEFAULT_BACKOFF_FACTOR,
        growth_interval: int = _DEFAULT_GROWTH_INTERVAL,
        enabled: bool = True,
    ) -> None:
        if growth_factor <= 1.0:
            raise ValueError(f"growth_factor must be > 1.0, got {growth_factor}")
        if not (0.0 < backoff_factor < 1.0):
            raise ValueError(f"backoff_factor must be in (0, 1), got {backoff_factor}")
        if growth_interval < 1:
            raise ValueError(f"growth_interval must be >= 1, got {growth_interval}")

        self._enabled = enabled
        self._scale: float = float(init_scale)
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval

        # Consecutive clean update counter
        self._growth_tracker: int = 0
        # Whether the last step was skipped due to inf/NaN
        self._last_step_was_skipped: bool = False

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """True if this scaler is active."""
        return self._enabled

    def get_scale(self) -> float:
        """Return the current scale factor.  Returns 1.0 when disabled."""
        return self._scale if self._enabled else 1.0

    def get_growth_factor(self) -> float:
        return self._growth_factor

    def get_backoff_factor(self) -> float:
        return self._backoff_factor

    def get_growth_interval(self) -> int:
        return self._growth_interval

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def scale(self, loss: Any) -> Any:
        """Multiply *loss* by the current scale factor.

        Works with numpy scalars, 0-d arrays, and any object that supports
        in-place ``*=`` or regular ``*`` with a float.

        Args:
            loss: The loss value (numpy scalar, float, or array).

        Returns:
            Scaled loss (same type as input).
        """
        if not self._enabled:
            return loss

        if isinstance(loss, np.ndarray):
            return loss * np.float32(self._scale)
        if isinstance(loss, (int, float)):
            return loss * self._scale
        # Fallback: try generic multiplication (works with Variable / Tensor)
        try:
            return loss * self._scale
        except TypeError:
            warnings.warn(
                f"GradScaler.scale(): unsupported loss type {type(loss)!r}. "
                "Returning loss unchanged.",
                stacklevel=2,
            )
            return loss

    def unscale_(self, optimizer: Any) -> None:
        """Unscale gradients in-place by dividing by the current scale.

        Grilly parameters store gradients as numpy arrays in ``param.grad``.
        This method iterates all parameters in the optimizer's param_groups
        and divides each gradient by the scale factor.

        Args:
            optimizer: A grilly ``Optimizer`` instance whose gradients will
                       be unscaled.
        """
        if not self._enabled:
            return

        inv_scale = 1.0 / self._scale
        for group in optimizer.param_groups:
            for param in group["params"]:
                if hasattr(param, "grad") and param.grad is not None:
                    if isinstance(param.grad, np.ndarray):
                        param.grad = param.grad * inv_scale
                    else:
                        try:
                            param.grad = param.grad * inv_scale
                        except Exception:
                            pass

    def _check_inf_nan(self, optimizer: Any) -> bool:
        """Return True if any gradient contains inf or NaN."""
        for group in optimizer.param_groups:
            for param in group["params"]:
                if hasattr(param, "grad") and param.grad is not None:
                    grad = param.grad
                    if isinstance(grad, np.ndarray):
                        if np.any(~np.isfinite(grad)):
                            return True
                    else:
                        try:
                            arr = np.asarray(grad, dtype=np.float32)
                            if np.any(~np.isfinite(arr)):
                                return True
                        except Exception:
                            pass
        return False

    def step(self, optimizer: Any, *args: Any, **kwargs: Any) -> Any | None:
        """Unscale gradients and call ``optimizer.step()``.

        If inf or NaN values are found in the unscaled gradients the
        optimizer step is skipped and ``_last_step_was_skipped`` is set.

        Args:
            optimizer: The optimizer whose ``.step()`` should be called.
            *args, **kwargs: Forwarded verbatim to ``optimizer.step()``.

        Returns:
            The return value of ``optimizer.step()``, or None if skipped.
        """
        if not self._enabled:
            return optimizer.step(*args, **kwargs)

        self.unscale_(optimizer)

        if self._check_inf_nan(optimizer):
            self._last_step_was_skipped = True
            return None

        self._last_step_was_skipped = False
        return optimizer.step(*args, **kwargs)

    def update(self, new_scale: float | None = None) -> None:
        """Update the scale factor after each training iteration.

        * If the last step was skipped (inf/NaN detected), multiply the
          scale by ``backoff_factor``.
        * Otherwise, increment the growth tracker; when it reaches
          ``growth_interval`` multiply the scale by ``growth_factor`` and
          reset the tracker.

        Args:
            new_scale: If provided, override the automatic update and set the
                       scale to this value directly.
        """
        if not self._enabled:
            return

        if new_scale is not None:
            self._scale = max(float(new_scale), _DEFAULT_MIN_SCALE)
            self._growth_tracker = 0
            return

        if self._last_step_was_skipped:
            self._scale = max(self._scale * self._backoff_factor, _DEFAULT_MIN_SCALE)
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                self._scale *= self._growth_factor
                self._growth_tracker = 0

    def state_dict(self) -> dict:
        """Return scaler state for checkpointing."""
        return {
            "scale": self._scale,
            "growth_factor": self._growth_factor,
            "backoff_factor": self._backoff_factor,
            "growth_interval": self._growth_interval,
            "growth_tracker": self._growth_tracker,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore scaler state from a checkpoint."""
        self._scale = state_dict["scale"]
        self._growth_factor = state_dict["growth_factor"]
        self._backoff_factor = state_dict["backoff_factor"]
        self._growth_interval = state_dict["growth_interval"]
        self._growth_tracker = state_dict["growth_tracker"]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"scale={self._scale}, "
            f"growth_factor={self._growth_factor}, "
            f"backoff_factor={self._backoff_factor}, "
            f"growth_interval={self._growth_interval}, "
            f"enabled={self._enabled})"
        )
