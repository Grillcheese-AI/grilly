"""
Base Optimizer class (PyTorch-like)

Similar to torch.optim.Optimizer
"""

from collections.abc import Iterator
from typing import Any

import numpy as np


class Optimizer:
    """
    Base class for all optimizers.

    Similar to torch.optim.Optimizer, but works with numpy arrays
    and GPU-accelerated operations via Vulkan shaders.
    """

    def __init__(self, params: Iterator[np.ndarray], defaults: dict[str, Any]):
        """
        Initialize optimizer.

        Args:
            params: Iterator of parameter arrays to optimize
            defaults: Dictionary of default hyperparameter values
        """
        self.defaults = defaults
        self.state: dict[int, dict[str, Any]] = {}
        self.param_groups: list = []

        # Convert params to list of parameter groups
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("Optimizer got an empty parameter list")

        # If first element is a dict, it's a parameter group
        if isinstance(param_groups[0], dict):
            self.param_groups = param_groups
        else:
            # Single parameter group
            self.param_groups = [{"params": param_groups}]

        # Copy defaults to each param_group (PyTorch behavior)
        for group in self.param_groups:
            for key, value in self.defaults.items():
                if key not in group:
                    group[key] = value

        # Initialize state for each parameter
        for group in self.param_groups:
            for p in group["params"]:
                # Accept both numpy arrays and Variable objects
                if hasattr(p, "data") and hasattr(p, "grad"):
                    # Variable from autograd - this is fine
                    pass
                elif not isinstance(p, np.ndarray):
                    raise TypeError("Optimizer can only optimize numpy arrays or Variable objects")
                # Create state entry for this parameter
                param_id = id(p)
                if param_id not in self.state:
                    self.state[param_id] = {}

    def zero_grad(self):
        """
        Clear gradients for all parameters.

        Note: In this implementation, gradients are expected to be
        stored in a separate structure (e.g., in the model's backward pass).
        This method is provided for API compatibility.
        """
        pass

    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure: Optional closure that reevaluates the model and returns loss

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        """
        Return the state of the optimizer as a dict.

        Returns:
            Dictionary containing optimizer state
        """
        return {
            "state": self.state,
            "param_groups": self.param_groups,
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        """
        Load optimizer state from state_dict.

        Args:
            state_dict: Dictionary containing optimizer state
        """
        self.state = state_dict.get("state", {})
        self.param_groups = state_dict.get("param_groups", [])

    def __repr__(self):
        """Return a debug representation."""

        return f"{self.__class__.__name__}(lr={self.defaults.get('lr', 'N/A')})"
