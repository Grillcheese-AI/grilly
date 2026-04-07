"""grilly.backend.channels — Python-side channel interface.

Wraps the C++ InProcessChannel with Pythonic API.
Falls back to a pure-Python implementation if C++ not available.

Usage:
    from grilly.backend.channels import Channel, MessageType

    ch = Channel("brain")
    ch.send_tensor(numpy_array, sender="vision")
    data = ch.receive_tensor()

    ch.send_spikes(spike_array, n_neurons=64, n_timesteps=10)

    # Subscribe to events
    ch.on(MessageType.TELEMETRY_EVENT, lambda msg: print(msg))
"""

from __future__ import annotations

import struct
import time
from collections import defaultdict
from collections.abc import Callable
from enum import IntEnum
from queue import Empty, Queue
from threading import Lock
from typing import Any

import numpy as np


class MessageType(IntEnum):
    """Message types matching C++ enum."""
    TENSOR_DATA = 0
    SPIKE_TRAIN = 1
    EXPERT_WEIGHTS = 2
    EXPERT_UPDATE = 3
    ROUTE_REQUEST = 4
    ROUTE_RESPONSE = 5
    MEMORY_CAPSULE = 6
    MEMORY_QUERY = 7
    MEMORY_RESULT = 8
    TELEMETRY_EVENT = 9
    NEUROCHEM_STATE = 10
    TRAIN_STEP_REQUEST = 11
    TRAIN_STEP_RESULT = 12


class Message:
    """A channel message with type, payload, and metadata."""

    __slots__ = ("type", "payload", "sender_id", "timestamp_ns", "metadata")

    def __init__(self, msg_type: MessageType, payload: bytes = b"",
                 sender_id: str = "", metadata: dict[str, Any] | None = None):
        self.type = msg_type
        self.payload = payload
        self.sender_id = sender_id
        self.timestamp_ns = time.time_ns()
        self.metadata = metadata or {}

    @property
    def size(self) -> int:
        return len(self.payload)


class Channel:
    """High-level channel with C++ backend fallback.

    Tries to use grilly_core.InProcessChannel (C++, zero-copy).
    Falls back to pure-Python thread-safe queue.

    Args:
        name: Channel name for debugging.
        max_queue_size: Maximum queued messages before dropping oldest.
    """

    def __init__(self, name: str = "default", max_queue_size: int = 10000):
        self.name = name
        self._cpp_channel = None
        self._py_queue: Queue = Queue(maxsize=max_queue_size)
        self._listeners: dict[MessageType, list[Callable]] = defaultdict(list)
        self._lock = Lock()

        # Try C++ channel
        try:
            from grilly import _core
            if hasattr(_core, "InProcessChannel"):
                self._cpp_channel = _core.InProcessChannel(name, max_queue_size)
        except Exception:
            pass

    @property
    def backend(self) -> str:
        return "cpp" if self._cpp_channel else "python"

    def send(self, msg: Message) -> None:
        """Send a message. Notifies listeners synchronously."""
        # Notify listeners
        for callback in self._listeners.get(msg.type, []):
            try:
                callback(msg)
            except Exception:
                pass

        if self._cpp_channel:
            # TODO: convert Message → C++ MessageEnvelope
            pass

        # Python fallback
        if self._py_queue.full():
            try:
                self._py_queue.get_nowait()  # Drop oldest
            except Empty:
                pass
        self._py_queue.put_nowait(msg)

    def receive(self, timeout: float | None = None) -> Message | None:
        """Receive next message. Returns None if empty."""
        if self._cpp_channel:
            # TODO: receive from C++ channel
            pass

        try:
            return self._py_queue.get(timeout=timeout)
        except Empty:
            return None

    def has_messages(self) -> bool:
        if self._cpp_channel:
            return self._cpp_channel.has_messages()
        return not self._py_queue.empty()

    def on(self, msg_type: MessageType, callback: Callable) -> None:
        """Subscribe to a message type."""
        self._listeners[msg_type].append(callback)

    def queue_size(self) -> int:
        if self._cpp_channel:
            return self._cpp_channel.queue_size()
        return self._py_queue.qsize()

    def clear(self) -> None:
        if self._cpp_channel:
            self._cpp_channel.clear()
        while not self._py_queue.empty():
            try:
                self._py_queue.get_nowait()
            except Empty:
                break

    # ── Convenience methods ──────────────────────────────────────────────

    def send_tensor(self, arr: np.ndarray, sender: str = "python") -> None:
        """Send a numpy array as a TENSOR_DATA message."""
        msg = Message(
            msg_type=MessageType.TENSOR_DATA,
            payload=arr.astype(np.float32).tobytes(),
            sender_id=sender,
            metadata={"shape": list(arr.shape), "dtype": str(arr.dtype)},
        )
        self.send(msg)

    def receive_tensor(self, shape: tuple | None = None) -> np.ndarray | None:
        """Receive a TENSOR_DATA message as numpy array."""
        msg = self.receive()
        if msg is None or msg.type != MessageType.TENSOR_DATA:
            return None
        arr = np.frombuffer(msg.payload, dtype=np.float32)
        if shape:
            arr = arr.reshape(shape)
        elif "shape" in msg.metadata:
            arr = arr.reshape(msg.metadata["shape"])
        return arr

    def send_spikes(self, spikes: np.ndarray, n_neurons: int,
                     n_timesteps: int, sender: str = "python") -> None:
        """Send spike train: (timesteps, neurons) flattened."""
        header = struct.pack("<II", n_neurons, n_timesteps)
        msg = Message(
            msg_type=MessageType.SPIKE_TRAIN,
            payload=header + spikes.astype(np.float32).tobytes(),
            sender_id=sender,
            metadata={"n_neurons": n_neurons, "n_timesteps": n_timesteps},
        )
        self.send(msg)

    def receive_spikes(self) -> tuple[np.ndarray, int, int] | None:
        """Receive spike train → (spikes, n_neurons, n_timesteps)."""
        msg = self.receive()
        if msg is None or msg.type != MessageType.SPIKE_TRAIN:
            return None
        n_neurons, n_timesteps = struct.unpack("<II", msg.payload[:8])
        spikes = np.frombuffer(msg.payload[8:], dtype=np.float32)
        return spikes.reshape(n_timesteps, n_neurons), n_neurons, n_timesteps

    def send_telemetry(self, component_id: str, event_type: str,
                        metrics: dict[str, float] | None = None,
                        step: int = 0) -> None:
        """Send a telemetry event."""
        import json
        payload = json.dumps({
            "component_id": component_id,
            "event_type": event_type,
            "metrics": metrics or {},
            "step": step,
        }).encode()
        msg = Message(
            msg_type=MessageType.TELEMETRY_EVENT,
            payload=payload,
            sender_id=component_id,
        )
        self.send(msg)

    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "queue_size": self.queue_size(),
            "listeners": {t.name: len(cbs) for t, cbs in self._listeners.items()},
        }
