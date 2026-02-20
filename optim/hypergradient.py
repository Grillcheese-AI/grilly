"""
Hypergradient Descent Optimizers

Implements online learning rate adaptation via hypergradient descent.

HypergradientAdamW: Basic hypergradient (Baydin et al. 2018).
    Fixed beta_hyper. Simple but requires tuning beta_hyper.

AutoHypergradientAdamW: OSGM-style auto adjustment (arXiv:2502.11229).
    Self-tuning via AdaGrad-stabilized hypergradients with gradient-norm
    normalization. No manual hypergradient LR tuning needed.
    Optional surprise signal: gradient prediction error exposed as
    `current_surprise` for input-level gain modulation. The model
    scales inputs by (1 + gain * surprise), amplifying signals when
    the optimization landscape shifts (e.g., SNN phase transitions).

The core idea: the learning rate is treated as a learnable parameter.
At each step, the hypergradient h = -g_k . d_{k-1} / ||g_{k-1}||^2
tells us whether to increase or decrease the learning rate based on
gradient agreement with the previous update direction.

References:
    [1] Baydin et al. "Online Learning Rate Adaptation with Hypergradient
        Descent" (ICLR 2018)
    [2] "Provable and Practical Online Learning Rate Adaptation with
        Hypergradient Descent" (arXiv:2502.11229)
    [3] "Gradient Methods with Online Scaling" (arXiv:2505.23081, 2509.11007)

Uses: adamw-update.glsl (via AdamW base class)
"""

from collections.abc import Iterator

import numpy as np

from .adamw import AdamW


def _collect_grads(param_groups, gradients=None):
    """Collect gradients from param groups into a dict keyed by param id."""
    grads = {}
    for group in param_groups:
        for p in group["params"]:
            if p is None:
                continue
            param_id = id(p)
            grad = None
            if gradients is not None:
                grad = gradients.get(param_id, None)
            if grad is None:
                grad = getattr(p, "grad", None)
            if grad is None:
                continue
            if hasattr(grad, "data"):
                grad = grad.data
            if not isinstance(grad, np.ndarray):
                grad = np.array(grad, dtype=np.float32)
            grads[param_id] = grad
    return grads


def _compute_update_directions(param_groups, state, step_count, betas, eps):
    """Compute Adam update directions d = m_hat / (sqrt(v_hat) + eps)."""
    beta1, beta2 = betas
    directions = {}
    for group in param_groups:
        for p in group["params"]:
            if p is None:
                continue
            param_id = id(p)
            s = state.get(param_id, {})
            if "exp_avg" not in s or "exp_avg_sq" not in s:
                continue
            sc = s.get("step", step_count)
            if sc == 0:
                continue
            m_hat = s["exp_avg"] / (1.0 - beta1 ** sc)
            v_hat = s["exp_avg_sq"] / (1.0 - beta2 ** sc)
            directions[param_id] = m_hat / (np.sqrt(v_hat) + eps)
    return directions


class HypergradientAdamW(AdamW):
    """AdamW with hypergradient-based online learning rate adaptation.

    Basic version from Baydin et al. (2018). Uses a fixed hypergradient
    learning rate beta_hyper. Simple but requires manual tuning of
    beta_hyper. For a self-tuning version, use AutoHypergradientAdamW.

    Update rule:
        alpha_{t+1} = alpha_t + beta_hyper * sum(g_t * d_{t-1})

    Args:
        params: Iterator of parameter arrays to optimize
        lr: Initial learning rate (default: 1e-3)
        betas: Coefficients for running averages (default: (0.9, 0.999))
        eps: Numerical stability term (default: 1e-8)
        weight_decay: Decoupled weight decay (default: 0.01)
        beta_hyper: Hypergradient learning rate (default: 1e-7)
        lr_min: Minimum learning rate clamp (default: 1e-6)
        lr_max: Maximum learning rate clamp (default: 1.0)
        log_scale: If True, adapt log(lr) instead of lr (default: False)
        use_gpu: Whether to use GPU acceleration (default: True)
    """

    def __init__(
        self,
        params: Iterator[np.ndarray],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        beta_hyper: float = 1e-7,
        lr_min: float = 1e-6,
        lr_max: float = 1.0,
        log_scale: bool = False,
        use_gpu: bool = True,
    ):
        super().__init__(
            params, lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, use_gpu=use_gpu,
        )
        self.beta_hyper = beta_hyper
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.log_scale = log_scale
        self._prev_directions = {}
        self._lr_history = [lr]

    @property
    def current_lr(self):
        return self.defaults["lr"]

    @property
    def lr_history(self):
        return self._lr_history

    def step(self, closure=None, gradients=None):
        current_grads = _collect_grads(self.param_groups, gradients)

        # Compute hypergradient: sum(g_t . d_{t-1})
        hypergradient = 0.0
        n_active = 0
        for pid, grad in current_grads.items():
            if pid in self._prev_directions:
                hypergradient += np.sum(grad * self._prev_directions[pid])
                n_active += 1

        if n_active > 0:
            if self.log_scale:
                log_lr = np.log(self.defaults["lr"])
                log_lr += self.beta_hyper * hypergradient
                new_lr = float(np.exp(np.clip(
                    log_lr, np.log(self.lr_min), np.log(self.lr_max)
                )))
            else:
                new_lr = self.defaults["lr"] + self.beta_hyper * hypergradient
                new_lr = float(np.clip(new_lr, self.lr_min, self.lr_max))
            self.defaults["lr"] = new_lr
            for group in self.param_groups:
                group["lr"] = new_lr
            self._lr_history.append(new_lr)

        loss = super().step(closure=closure, gradients=gradients)

        self._prev_directions = _compute_update_directions(
            self.param_groups, self.state, self._step_count,
            self.defaults["betas"], self.defaults["eps"],
        )
        return loss

    def __repr__(self):
        return (
            f"HypergradientAdamW(lr={self.defaults['lr']:.6f}, "
            f"beta_hyper={self.beta_hyper}, "
            f"lr_range=[{self.lr_min}, {self.lr_max}])"
        )


class AutoHypergradientAdamW(AdamW):
    """AdamW with OSGM-style auto hypergradient adjustment.

    Self-tuning optimizer that automatically adapts the learning rate
    (and optionally momentum beta1) using online hypergradient descent
    with AdaGrad-stabilized updates. No manual hypergradient LR tuning
    needed — the AdaGrad accumulator self-adjusts the meta-learning rate.

    Based on the OSGM/HDM algorithm:

    Step size hypergradient (how lr should change):
        h_lr = -g_k . d_{k-1} / (||g_{k-1}||^2 + eps)
        G_lr += h_lr^2
        lr -= hyper_lr * h_lr / (sqrt(G_lr) + eps)

    Momentum hypergradient (how beta1 should change):
        h_beta = g_k . m_{k-1} / (||g_{k-1}||^2 + eps)
        G_beta += h_beta^2
        beta1 -= hyper_lr_beta * h_beta / (sqrt(G_beta) + eps)

    The gradient-norm normalization (/ ||g||^2) makes the algorithm
    scale-invariant, and the AdaGrad accumulator makes the meta-LR
    self-adjusting — larger past hypergradients automatically slow
    down future adaptation, preventing oscillation.

    Particularly effective for SNN training where surrogate gradients
    are noisy and the optimal learning rate shifts during training.

    Surprise signal (optional, input-level):
        Tracks gradient prediction error as a "surprise" signal and
        exposes it for the model to use as input gain modulation.
        Unlike backprop-level momentum changes, this acts at the
        forward-pass level — amplifying input signals when the
        optimization landscape shifts unexpectedly.

        Instant surprise (gradient prediction error):
            S_instant = tanh(||g_k - EMA(g)||^2 / (EMA(||g||^2) + eps))

        Accumulated surprise (biological momentum / S_bar):
            S_bar = alpha * S_instant + (1-alpha) * S_bar_prev

        Inverted-U gain (Yerkes-Dodson / trauma protection):
            gain = S_bar * exp(-S_bar / trauma_threshold)

        The inverted-U curve implements the biological stress response:
            - Low S_bar  → low gain (nothing interesting)
            - Moderate S_bar → peak gain (optimal learning zone)
            - High S_bar → gain drops (trauma protection)

        This prevents "unerasable events" — if surprise stays high
        for many consecutive steps (chronic stress), the gain suppresses
        instead of amplifying, protecting the model from fixating on
        a single extreme event. Mirrors the HPA axis: acute stress
        enhances encoding, chronic stress impairs plasticity.

        The model reads `current_surprise_gain` for input scaling:
            x_effective = x * (1 + scale * optimizer.current_surprise_gain)

    Args:
        params: Iterator of parameter arrays to optimize
        lr: Initial learning rate (default: 1e-3)
        betas: Coefficients for running averages (default: (0.9, 0.999))
        eps: Numerical stability term (default: 1e-8)
        weight_decay: Decoupled weight decay (default: 0.01)
        hyper_lr: Meta-learning rate for step size adaptation (default: 0.01).
            This is automatically modulated by the AdaGrad accumulator,
            so it's much less sensitive than HypergradientAdamW's beta_hyper.
        hyper_lr_beta: Meta-learning rate for momentum adaptation
            (default: 1.0). Only used when adapt_momentum=True.
        lr_min: Minimum learning rate clamp (default: 1e-6)
        lr_max: Maximum learning rate clamp (default: 1.0)
        adapt_momentum: If True, also adapt beta1 via hypergradient
            (default: False)
        track_surprise: If True, compute and expose gradient surprise
            signal via `current_surprise_gain` (default: False). The model's
            forward pass should read this to modulate input gain.
        surprise_gamma: EMA decay for gradient tracking (default: 0.9).
            Higher = smoother baseline, slower to detect change.
        surprise_alpha: EMA decay for surprise accumulation S_bar
            (default: 0.1). Controls how fast accumulated surprise
            builds up and decays. Lower = longer memory of surprise.
        trauma_threshold: S_bar level where gain peaks before suppression
            (default: 0.5). The inverted-U gain = S_bar * exp(-S_bar/T)
            peaks at S_bar = T. Above this, gain decreases (protection).
        beta_min: Minimum beta1 clamp (default: 0.5)
        beta_max: Maximum beta1 clamp (default: 0.9995)
        warmup_steps: Steps before starting adaptation (default: 10).
            Lets Adam moments initialize before adapting LR.
        use_gpu: Whether to use GPU acceleration (default: True)
    """

    def __init__(
        self,
        params: Iterator[np.ndarray],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        hyper_lr: float = 0.01,
        hyper_lr_beta: float = 1.0,
        lr_min: float = 1e-6,
        lr_max: float = 1.0,
        adapt_momentum: bool = False,
        track_surprise: bool = False,
        surprise_gamma: float = 0.9,
        surprise_alpha: float = 0.1,
        trauma_threshold: float = 0.5,
        beta_min: float = 0.5,
        beta_max: float = 0.9995,
        warmup_steps: int = 10,
        use_gpu: bool = True,
    ):
        super().__init__(
            params, lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, use_gpu=use_gpu,
        )
        self.hyper_lr = hyper_lr
        self.hyper_lr_beta = hyper_lr_beta
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.adapt_momentum = adapt_momentum
        self.track_surprise = track_surprise
        self.surprise_gamma = surprise_gamma
        self.surprise_alpha = surprise_alpha
        self.trauma_threshold = trauma_threshold
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.warmup_steps = warmup_steps

        # AdaGrad accumulators for hypergradient stabilization.
        # Seeded to 1.0 (not 0.0) so the first adaptive step doesn't apply
        # the full hyper_lr as lr_delta. With G_lr=0 and hyper_lr=0.01,
        # the first step would be lr_delta=±0.01, a 10x jump when lr=0.001.
        # Seeding to 1.0 bounds the first step to hyper_lr*h/(sqrt(1+h^2)).
        self._G_lr = 1.0
        self._G_beta = 1.0
        self._adagrad_eps = 1e-12
        self._meta_decay = 0.99  # RMSProp-style decay for meta-accumulator

        # Previous step state for hypergradient computation
        self._prev_directions = {}   # d_{k-1}: Adam update directions
        self._prev_grad_norm_sq = 0.0  # ||g_{k-1}||^2
        self._prev_first_moments = {}  # m_{k-1}: for momentum adaptation

        # Surprise signal state (input-level, not backprop-level).
        # Tracks gradient prediction error as a neuromodulatory signal.
        # The optimizer computes surprise; the model reads current_surprise_gain
        # for input scaling via the inverted-U (Yerkes-Dodson) curve.
        self._grad_ema = {}          # EMA of gradients (per-param)
        self._grad_var_ema = 0.0     # EMA of ||g||^2 (scalar)
        self._current_surprise = 0.0  # instant surprise [0, 1]
        self._s_bar = 0.0            # accumulated surprise (biological momentum)
        self._current_gain = 0.0     # inverted-U modulated gain
        self._surprise_history = []  # instant surprise history
        self._s_bar_history = []     # accumulated surprise history

        # History for monitoring / plotting
        self._lr_history = [lr]
        self._beta1_history = [betas[0]]

    @property
    def current_lr(self):
        return self.defaults["lr"]

    @property
    def current_surprise(self):
        """Instant surprise signal [0, 1]. Raw gradient prediction error."""
        return self._current_surprise

    @property
    def accumulated_surprise(self):
        """Accumulated surprise S_bar. Biological momentum of surprise."""
        return self._s_bar

    @property
    def current_surprise_gain(self):
        """Inverted-U gain signal for input-level modulation.

        Implements the Yerkes-Dodson curve / trauma protection:
            gain = S_bar * exp(-S_bar / trauma_threshold)

        - Low S_bar → low gain (nothing interesting happening)
        - Moderate S_bar → peak gain (optimal learning zone)
        - High S_bar → gain drops (trauma protection, don't fixate)

        Read this after each optimizer step and pass to the model:
            x_effective = x * (1 + scale * optimizer.current_surprise_gain)

        Returns 0.0 when surprise tracking is off or during warmup.
        """
        return self._current_gain

    @property
    def lr_history(self):
        return self._lr_history

    @property
    def beta1_history(self):
        return self._beta1_history

    @property
    def surprise_history(self):
        return self._surprise_history

    @property
    def s_bar_history(self):
        return self._s_bar_history

    def step(self, closure=None, gradients=None):
        """Perform optimization step with OSGM-style auto LR adaptation.

        1. Collect current gradients g_k
        2. Compute surprise signal (if track_surprise=True)
        3. Compute normalized hypergradients (after warmup):
           h_lr  = -g_k . d_{k-1} / ||g_{k-1}||^2
           h_beta = g_k . m_{k-1} / ||g_{k-1}||^2
        4. Update AdaGrad accumulators and adjust lr (and beta1)
        5. Run standard AdamW step with adapted hyperparameters
        6. Store d_k, ||g_k||^2, m_k for next step
        """
        current_grads = _collect_grads(self.param_groups, gradients)

        # --- Surprise signal computation (input-level, not backprop) ---
        # Computes gradient prediction error and exposes it via
        # current_surprise. The model reads this to scale inputs.
        if self.track_surprise and self._step_count >= self.warmup_steps:
            gamma = self.surprise_gamma

            # Compute current gradient norm squared
            current_norm_sq = 0.0
            for grad in current_grads.values():
                current_norm_sq += np.sum(grad * grad)

            # Update gradient variance EMA: EMA(||g||^2)
            if self._grad_var_ema == 0.0 and self._step_count == self.warmup_steps:
                self._grad_var_ema = current_norm_sq
            else:
                self._grad_var_ema = (gamma * self._grad_var_ema
                                      + (1.0 - gamma) * current_norm_sq)

            # Compute surprise: ||g_k - EMA(g)||^2 / (EMA(||g||^2) + eps)
            prediction_error_sq = 0.0
            for pid, grad in current_grads.items():
                if pid in self._grad_ema:
                    diff = grad - self._grad_ema[pid]
                    prediction_error_sq += np.sum(diff * diff)
                else:
                    prediction_error_sq += np.sum(grad * grad)

            raw_surprise = prediction_error_sq / (self._grad_var_ema + self._adagrad_eps)
            # Squash instant surprise to [0, 1] via tanh
            self._current_surprise = float(np.tanh(raw_surprise))
            self._surprise_history.append(self._current_surprise)

            # Accumulate surprise: S_bar = alpha * S_instant + (1-alpha) * S_bar
            # This is the biological momentum — tracks sustained surprise.
            alpha = self.surprise_alpha
            self._s_bar = alpha * self._current_surprise + (1.0 - alpha) * self._s_bar
            self._s_bar_history.append(self._s_bar)

            # Inverted-U gain (Yerkes-Dodson / trauma protection):
            # gain = S_bar * exp(-S_bar / trauma_threshold)
            # Peaks at S_bar = trauma_threshold, suppresses above.
            # This prevents "unerasable events" — chronic high surprise
            # (trauma) reduces gain instead of amplifying it.
            T = self.trauma_threshold
            self._current_gain = float(self._s_bar * np.exp(-self._s_bar / T))

            # Update gradient EMA
            for pid, grad in current_grads.items():
                if pid in self._grad_ema:
                    self._grad_ema[pid] = (gamma * self._grad_ema[pid]
                                           + (1.0 - gamma) * grad.copy())
                else:
                    self._grad_ema[pid] = grad.copy()

        # --- Hypergradient-based adaptation (after warmup) ---
        if (self._step_count >= self.warmup_steps
                and self._prev_grad_norm_sq > self._adagrad_eps
                and self._prev_directions):

            norm_sq = self._prev_grad_norm_sq

            # Hypergradient for learning rate:
            # h_lr = -sum(g_k * d_{k-1}) / ||g_{k-1}||^2
            h_lr = 0.0
            for pid, grad in current_grads.items():
                if pid in self._prev_directions:
                    h_lr -= np.sum(grad * self._prev_directions[pid])
            h_lr /= norm_sq

            # Clip hypergradient to prevent outliers from poisoning accumulator.
            # Without clipping, a single step with tiny ||g_{k-1}||^2 (common
            # in LIF silent phases) can produce h_lr >> 1, permanently inflating
            # G_lr and freezing the meta-LR near zero.
            h_lr = float(np.clip(h_lr, -1.0, 1.0))

            # RMSProp-style decay for meta-accumulator (replaces pure AdaGrad).
            # AdaGrad's "never forget" property means one outlier permanently
            # poisons G_lr. Decaying lets the optimizer recover from early
            # instability (especially important for LIF's noisy gradient start).
            self._G_lr = self._meta_decay * self._G_lr + (1.0 - self._meta_decay) * h_lr * h_lr
            lr_delta = self.hyper_lr * h_lr / (np.sqrt(self._G_lr) + self._adagrad_eps)

            # Rate-limit LR changes: max 10% relative change per step.
            # 50% allows 0.0005 to reach 0.01 in 7 steps (too fast for LIF).
            # 10% means 0.0005 reaches 0.005 in ~24 steps — gradual enough
            # for the model to signal back if the LR is getting too high.
            current_lr = self.defaults["lr"]
            target_lr = current_lr - lr_delta
            max_change = 0.1 * current_lr
            if abs(target_lr - current_lr) > max_change:
                direction = 1.0 if target_lr > current_lr else -1.0
                target_lr = current_lr + direction * max_change

            new_lr = float(np.clip(target_lr, self.lr_min, self.lr_max))
            self.defaults["lr"] = new_lr
            for group in self.param_groups:
                group["lr"] = new_lr
            self._lr_history.append(new_lr)

            # Hypergradient for momentum (beta1):
            if self.adapt_momentum and self._prev_first_moments:
                h_beta = 0.0
                for pid, grad in current_grads.items():
                    if pid in self._prev_first_moments:
                        h_beta += np.sum(grad * self._prev_first_moments[pid])
                h_beta /= norm_sq

                h_beta = float(np.clip(h_beta, -1.0, 1.0))
                self._G_beta = (self._meta_decay * self._G_beta
                                + (1.0 - self._meta_decay) * h_beta * h_beta)
                beta_delta = (self.hyper_lr_beta * h_beta
                              / (np.sqrt(self._G_beta) + self._adagrad_eps))

                beta1, beta2 = self.defaults["betas"]
                new_beta1 = float(np.clip(
                    beta1 - beta_delta,
                    self.beta_min, self.beta_max,
                ))
                self.defaults["betas"] = (new_beta1, beta2)
                for group in self.param_groups:
                    group["betas"] = (new_beta1, beta2)
                self._beta1_history.append(new_beta1)

        # --- Compute current gradient norm for next step ---
        grad_norm_sq = 0.0
        for grad in current_grads.values():
            grad_norm_sq += np.sum(grad * grad)

        # --- Run standard AdamW step ---
        loss = super().step(closure=closure, gradients=gradients)

        # --- Store state for next step's hypergradient ---
        self._prev_directions = _compute_update_directions(
            self.param_groups, self.state, self._step_count,
            self.defaults["betas"], self.defaults["eps"],
        )
        self._prev_grad_norm_sq = grad_norm_sq

        if self.adapt_momentum:
            self._prev_first_moments = {}
            for group in self.param_groups:
                for p in group["params"]:
                    if p is None:
                        continue
                    pid = id(p)
                    s = self.state.get(pid, {})
                    if "exp_avg" in s:
                        self._prev_first_moments[pid] = s["exp_avg"].copy()

        return loss

    def __repr__(self):
        beta1 = self.defaults["betas"][0]
        parts = [
            f"AutoHypergradientAdamW(lr={self.defaults['lr']:.6f}",
            f"beta1={beta1:.4f}",
            f"hyper_lr={self.hyper_lr}",
            f"lr_range=[{self.lr_min}, {self.lr_max}]",
            f"adapt_momentum={self.adapt_momentum}",
            f"track_surprise={self.track_surprise}",
        ]
        return ", ".join(parts) + ")"
