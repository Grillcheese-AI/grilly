"""Random Fourier Features + Contrastive Value Learning (CVL) readout.

CVL (Mazoure, Eysenbach, Nachum, Tompson — Contrastive Value Learning, NeurIPS-W 2022) estimates a
**TD-free** value by *propagating forward*: it models the discounted future-state occupancy with a
contrastive (inner-product) critic and reads the value as a reward-weighted average of future-state
likelihoods. The RBF kernel e^{x·y} is linearized with **Random Fourier Features** (Rahimi & Recht
2007): F(x) = sqrt(2e/d)·cos(Wx + b), W ~ N(0,I), b ~ U(0,2π), so that E[F(x)·F(y)] = e^{x·y} for
unit x, y. This factors the exponential, so one keeps a single reward-weighted future-feature vector
ξ (EMA) and reads Q(s,a) = 1/(1-γ) · F(φ(s,a))·ξ in O(1) — no temporal-difference back-up.

General-purpose + dependency-light (numpy): inputs are real feature vectors. The caller maps its own
representations (e.g. VSA hypervectors) to these features. Vulkan/GLSL acceleration can replace the
numpy core later without changing this interface.
"""
from __future__ import annotations

import numpy as np


class RandomFourierFeatures:
    """F(x) = sqrt(2e/d)·cos(Wx + b); for unit x,y, E[F(x)·F(y)] = e^{x·y} (Rahimi & Recht 2007)."""

    def __init__(self, in_dim: int, n_features: int = 4096, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((in_dim, n_features))
        self.b = rng.uniform(0.0, 2.0 * np.pi, n_features)
        self.n_features = n_features
        self._scale = np.sqrt(2.0 * np.e / n_features)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return self._scale * np.cos(x @ self.W + self.b)


class CVLValue:
    """TD-free CVL value: a reward-weighted future-feature EMA ξ, read as Q = 1/(1-γ)·F(φ)·ξ.

    - observe(future_feat, reward): fold (RFF feature of a future state × reward) into ξ (EMA).
    - value(phi_feat): forward readout 1/(1-γ)·⟨F(φ), ξ⟩ — proportional to Qπ, so it ranks actions
      correctly (CVL recovers Q up to a positive constant). No bootstrapping.
    """

    def __init__(self, gamma: float = 0.95, beta: float = 0.99) -> None:
        self.gamma = gamma
        self.beta = beta
        self._xi: np.ndarray | None = None

    @property
    def xi(self) -> np.ndarray | None:
        return self._xi

    def observe(self, future_feat: np.ndarray, reward: float) -> None:
        f = np.asarray(future_feat, dtype=float) * float(reward)
        if self._xi is None:
            self._xi = np.zeros_like(f)
        self._xi = self.beta * self._xi + (1.0 - self.beta) * f

    def value(self, phi_feat: np.ndarray) -> float:
        if self._xi is None:
            return 0.0
        return float(np.dot(np.asarray(phi_feat, dtype=float), self._xi)) / (1.0 - self.gamma)
