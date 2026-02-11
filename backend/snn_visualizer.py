"""
Real-time SNN (Spiking Neural Network) Visualization
Provides WebSocket-based real-time visualization of SNN activity
"""

import json
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SNNState:
    """Snapshot of SNN state at a point in time"""

    timestamp: float
    neuron_potentials: list[float]
    spike_events: list[int]  # Indices of neurons that spiked
    synaptic_weights: list[list[float]] | None = None
    stdp_trace: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "timestamp": self.timestamp,
            "neuron_potentials": self.neuron_potentials[:100],  # Limit to first 100 for bandwidth
            "spike_events": self.spike_events,
            "spike_count": len(self.spike_events),
            "avg_potential": float(np.mean(self.neuron_potentials)),
            "max_potential": float(np.max(self.neuron_potentials)),
            "stdp_trace": self.stdp_trace[:20] if self.stdp_trace else None,
        }


class SNNVisualizer:
    """
    Real-time SNN visualization manager
    Captures SNN state and broadcasts to WebSocket clients
    """

    def __init__(self, max_history: int = 100):
        """
        Initialize SNN visualizer

        Args:
            max_history: Maximum number of state snapshots to keep
        """
        self.max_history = max_history
        self.state_history: list[SNNState] = []
        self.active_clients = set()
        self.enabled = False

        logger.info("SNN Visualizer initialized")

    def capture_state(
        self,
        neuron_potentials: np.ndarray,
        spike_events: np.ndarray,
        synaptic_weights: np.ndarray | None = None,
        stdp_trace: np.ndarray | None = None,
    ) -> SNNState:
        """
        Capture current SNN state

        Args:
            neuron_potentials: Array of neuron membrane potentials
            spike_events: Array of neuron indices that spiked
            synaptic_weights: Optional weight matrix
            stdp_trace: Optional STDP learning trace

        Returns:
            SNNState snapshot
        """
        import time

        state = SNNState(
            timestamp=time.time(),
            neuron_potentials=neuron_potentials.tolist(),
            spike_events=spike_events.tolist()
            if isinstance(spike_events, np.ndarray)
            else spike_events,
            synaptic_weights=synaptic_weights.tolist() if synaptic_weights is not None else None,
            stdp_trace=stdp_trace.tolist() if stdp_trace is not None else None,
        )

        # Add to history
        self.state_history.append(state)

        # Trim history if needed
        if len(self.state_history) > self.max_history:
            self.state_history = self.state_history[-self.max_history :]

        return state

    async def broadcast_state(self, state: SNNState):
        """
        Broadcast SNN state to all connected WebSocket clients

        Args:
            state: SNN state to broadcast
        """
        if not self.active_clients or not self.enabled:
            return

        message = json.dumps({"type": "snn_state", "data": state.to_dict()})

        # Broadcast to all clients
        disconnected = set()
        for client in self.active_clients:
            try:
                await client.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                disconnected.add(client)

        # Remove disconnected clients
        self.active_clients -= disconnected

    def register_client(self, websocket):
        """Register a new WebSocket client"""
        self.active_clients.add(websocket)
        logger.info(f"Client registered. Active clients: {len(self.active_clients)}")

    def unregister_client(self, websocket):
        """Unregister a WebSocket client"""
        self.active_clients.discard(websocket)
        logger.info(f"Client unregistered. Active clients: {len(self.active_clients)}")

    def enable(self):
        """Enable real-time visualization"""
        self.enabled = True
        logger.info("SNN visualization enabled")

    def disable(self):
        """Disable real-time visualization"""
        self.enabled = False
        logger.info("SNN visualization disabled")

    def get_history(self, n: int | None = None) -> list[dict[str, Any]]:
        """
        Get recent state history

        Args:
            n: Number of recent states to return (None = all)

        Returns:
            List of state dictionaries
        """
        history = self.state_history[-n:] if n else self.state_history
        return [state.to_dict() for state in history]

    def get_statistics(self) -> dict[str, Any]:
        """
        Get aggregated statistics from history

        Returns:
            Dictionary of statistics
        """
        if not self.state_history:
            return {"error": "No history available"}

        recent_states = self.state_history[-20:]

        avg_spike_rate = np.mean([len(s.spike_events) for s in recent_states])
        avg_potential = np.mean([np.mean(s.neuron_potentials) for s in recent_states])

        return {
            "total_snapshots": len(self.state_history),
            "avg_spike_rate": float(avg_spike_rate),
            "avg_membrane_potential": float(avg_potential),
            "active_clients": len(self.active_clients),
            "visualization_enabled": self.enabled,
        }


# Global instance
_visualizer: SNNVisualizer | None = None


def get_visualizer() -> SNNVisualizer:
    """Get or create global SNN visualizer instance"""
    global _visualizer
    if _visualizer is None:
        _visualizer = SNNVisualizer()
    return _visualizer
