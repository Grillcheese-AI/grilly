"""
VsaLmModel — VSA Language Model with AdditionLinear FFN layers.

Wraps the fused C++ forward/backward (``grilly_core.vsa_lm_*``) for
GPU-accelerated training, with transparent NumPy CPU fallback.

Usage::

    model = VsaLmModel(vocab=1000, d_model=256, d_ffn=512, max_seq=512, n_layers=6)
    logits = model(input_ids)                       # (seq, vocab)
    grads  = model.backward(input_ids, grad_logits) # dict of gradients
    model.step(grads, lr=1e-3)                      # SGD update
"""

from __future__ import annotations

import numpy as np

from ._helpers import _get_param_array, _create_param_wrapper
from .module import Module
from .addition_linear import AdditionLinear


def _try_bridge():
    try:
        from ..backend import _bridge
        if _bridge.is_available():
            return _bridge
    except Exception:
        pass
    return None


class VsaLmModel(Module):
    """Full VSA-LM: Embedding → [LayerNorm → AdditionLinear FFN → Sign → AdditionLinear → Residual] × L → Output Proj.

    When the C++ fused kernel is available, ``forward``/``backward`` dispatch to
    a single ``grilly_core`` call (GPU forward, Eigen backward).  Otherwise
    falls back to pure NumPy.
    """

    def __init__(
        self,
        vocab: int,
        d_model: int,
        d_ffn: int,
        max_seq: int,
        n_layers: int,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.max_seq = max_seq
        self.n_layers = n_layers

        rng = np.random.default_rng()

        self.embed_w = _create_param_wrapper(
            rng.standard_normal((vocab, d_model)).astype(np.float32) * init_std
        )
        self.pos_w = _create_param_wrapper(
            rng.standard_normal((max_seq, d_model)).astype(np.float32) * init_std
        )
        self.out_w = _create_param_wrapper(
            rng.standard_normal((vocab, d_model)).astype(np.float32) * init_std
        )

        self.register_parameter("embed_w", self.embed_w)
        self.register_parameter("pos_w", self.pos_w)
        self.register_parameter("out_w", self.out_w)

        self.ffn_up: list[AdditionLinear] = []
        self.ffn_down: list[AdditionLinear] = []
        self.ln_gammas: list = []
        self.ln_betas: list = []

        for l in range(n_layers):
            up = AdditionLinear(d_model, d_ffn, bias=True)
            down = AdditionLinear(d_ffn, d_model, bias=True)
            self.ffn_up.append(up)
            self.ffn_down.append(down)
            self._modules[f"ffn_up_{l}"] = up
            self._modules[f"ffn_down_{l}"] = down

            gamma = _create_param_wrapper(np.ones(d_model, dtype=np.float32))
            beta = _create_param_wrapper(np.zeros(d_model, dtype=np.float32))
            self.ln_gammas.append(gamma)
            self.ln_betas.append(beta)
            self.register_parameter(f"ln_gamma_{l}", gamma)
            self.register_parameter(f"ln_beta_{l}", beta)

        self._handle: int | None = None
        self._bridge = _try_bridge()

    # ── GPU handle lifecycle ────────────────────────────────────────────

    def _upload(self):
        """Upload all weights to GPU via bridge (idempotent)."""
        if self._handle is not None:
            return
        br = self._bridge
        if br is None:
            return

        h = br.vsa_lm_upload(
            _get_param_array(self.embed_w),
            _get_param_array(self.pos_w),
            [_get_param_array(u.weight) for u in self.ffn_up],
            [_get_param_array(u.bias_param) for u in self.ffn_up],
            [_get_param_array(d.weight) for d in self.ffn_down],
            [_get_param_array(d.bias_param) for d in self.ffn_down],
            [_get_param_array(g) for g in self.ln_gammas],
            [_get_param_array(b) for b in self.ln_betas],
            _get_param_array(self.out_w),
            self.n_layers, self.d_model, self.d_ffn,
        )
        self._handle = h

    def _sync_weights(self):
        """Re-upload updated weights to existing GPU handle."""
        br = self._bridge
        if br is None or self._handle is None:
            return
        br.vsa_lm_update_weights(
            self._handle,
            _get_param_array(self.embed_w),
            _get_param_array(self.pos_w),
            [_get_param_array(u.weight) for u in self.ffn_up],
            [_get_param_array(u.bias_param) for u in self.ffn_up],
            [_get_param_array(d.weight) for d in self.ffn_down],
            [_get_param_array(d.bias_param) for d in self.ffn_down],
            [_get_param_array(g) for g in self.ln_gammas],
            [_get_param_array(b) for b in self.ln_betas],
            _get_param_array(self.out_w),
        )

    def release(self):
        """Free GPU resources."""
        if self._handle is not None and self._bridge is not None:
            self._bridge.vsa_lm_release(self._handle)
        self._handle = None

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """Run forward pass; returns logits (seq_len, vocab).

        Tries the fused GPU path first, falls back to NumPy.
        """
        ids = np.ascontiguousarray(input_ids, dtype=np.int32)

        # GPU path
        self._upload()
        if self._handle is not None and self._bridge is not None:
            result = self._bridge.vsa_lm_forward(self._handle, ids)
            if result is not None:
                return result

        # NumPy fallback
        return self._forward_numpy(ids)

    def _forward_numpy(self, ids: np.ndarray) -> np.ndarray:
        S = ids.shape[0]
        d = self.d_model
        embed = _get_param_array(self.embed_w)
        pos = _get_param_array(self.pos_w)
        out_w = _get_param_array(self.out_w)

        x = embed[ids.astype(np.int64)] + pos[:S]

        for l in range(self.n_layers):
            gamma = _get_param_array(self.ln_gammas[l])
            beta = _get_param_array(self.ln_betas[l])

            # LayerNorm
            mean = x.mean(axis=-1, keepdims=True)
            var = x.var(axis=-1, keepdims=True)
            h = ((x - mean) / np.sqrt(var + 1e-5) * gamma + beta).astype(np.float32)

            # AdditionLinear up → sign → AdditionLinear down
            h_up = self.ffn_up[l].forward(h)
            h_sign = np.where(h_up > 0, 1.0, -1.0).astype(np.float32)
            h_ffn = self.ffn_down[l].forward(h_sign)

            x = x + h_ffn

        scale = 1.0 / np.sqrt(np.float32(d))
        return (x @ out_w.T * scale).astype(np.float32)

    # ── Backward ─────────────────────────────────────────────────────────

    def backward(self, input_ids: np.ndarray, grad_logits: np.ndarray) -> dict:
        """Run backward pass; returns gradient dict.

        GPU path returns one dict from C++.  Fallback stores grads on parameters.
        """
        ids = np.ascontiguousarray(input_ids, dtype=np.int32)
        gl = np.ascontiguousarray(grad_logits, dtype=np.float32)

        if self._handle is not None and self._bridge is not None:
            result = self._bridge.vsa_lm_backward(self._handle, ids, gl)
            if result is not None:
                self._scatter_grads(result)
                return result

        return self._backward_numpy(ids, gl)

    def _scatter_grads(self, grads: dict):
        """Store C++ gradient arrays onto parameter .grad attributes."""
        def _acc(param, g):
            if not hasattr(param, "grad") or param.grad is None:
                param.grad = g.copy()
            else:
                param.grad += g

        _acc(self.embed_w, grads["grad_embed"])
        _acc(self.pos_w, grads["grad_pos"])
        _acc(self.out_w, grads["grad_out_w"])

        for l in range(self.n_layers):
            _acc(self.ffn_up[l].weight, grads["grad_ffn_up_w"][l])
            if self.ffn_up[l].bias_param is not None:
                _acc(self.ffn_up[l].bias_param, grads["grad_ffn_up_b"][l])
            _acc(self.ffn_down[l].weight, grads["grad_ffn_down_w"][l])
            if self.ffn_down[l].bias_param is not None:
                _acc(self.ffn_down[l].bias_param, grads["grad_ffn_down_b"][l])
            _acc(self.ln_gammas[l], grads["grad_ln_gamma"][l])
            _acc(self.ln_betas[l], grads["grad_ln_beta"][l])

    def _backward_numpy(self, ids: np.ndarray, grad_logits: np.ndarray) -> dict:
        """Pure NumPy backward (slow but correct reference)."""
        S = ids.shape[0]
        d = self.d_model
        dF = self.d_ffn
        V = self.vocab
        L = self.n_layers

        embed = _get_param_array(self.embed_w)
        pos = _get_param_array(self.pos_w)
        out_w = _get_param_array(self.out_w)

        # Forward replay for activations
        acts = [None] * (L + 1)
        ln_outs = [None] * L
        ffn_ups = [None] * L
        sign_outs = [None] * L

        acts[0] = embed[ids.astype(np.int64)] + pos[:S]

        for l in range(L):
            gamma = _get_param_array(self.ln_gammas[l])
            beta = _get_param_array(self.ln_betas[l])
            uw = _get_param_array(self.ffn_up[l].weight)
            ub = _get_param_array(self.ffn_up[l].bias_param)
            dw = _get_param_array(self.ffn_down[l].weight)
            db = _get_param_array(self.ffn_down[l].bias_param)

            mean = acts[l].mean(axis=-1, keepdims=True)
            var = acts[l].var(axis=-1, keepdims=True)
            ln_outs[l] = ((acts[l] - mean) / np.sqrt(var + 1e-5) * gamma + beta).astype(np.float32)

            ffn_ups[l] = np.empty((S, dF), dtype=np.float32)
            for o in range(dF):
                ffn_ups[l][:, o] = -np.sum(np.abs(uw[o] - ln_outs[l]), axis=1) + ub[o]

            sign_outs[l] = np.where(ffn_ups[l] > 0, 1.0, -1.0).astype(np.float32)

            ffn_down_out = np.empty((S, d), dtype=np.float32)
            for o in range(d):
                ffn_down_out[:, o] = -np.sum(np.abs(dw[o] - sign_outs[l]), axis=1) + db[o]

            acts[l + 1] = acts[l] + ffn_down_out

        scale = 1.0 / np.sqrt(np.float32(d))
        gl_scaled = grad_logits * scale

        dx = gl_scaled @ out_w
        grad_out_w = gl_scaled.T @ acts[L]

        grads = {
            "grad_out_w": grad_out_w,
            "grad_ffn_up_w": [], "grad_ffn_up_b": [],
            "grad_ffn_down_w": [], "grad_ffn_down_b": [],
            "grad_ln_gamma": [], "grad_ln_beta": [],
        }

        for l in range(L - 1, -1, -1):
            uw = _get_param_array(self.ffn_up[l].weight)
            dw = _get_param_array(self.ffn_down[l].weight)

            # Addition-linear down backward
            g_dn_w = np.zeros_like(dw)
            g_dn_b = np.sum(dx, axis=0)
            grad_sign = np.zeros((S, dF), dtype=np.float32)
            for o in range(d):
                sgn = np.sign(dw[o][np.newaxis, :] - sign_outs[l])
                g_col = dx[:, o:o+1]
                g_dn_w[o] -= np.sum(g_col * sgn, axis=0)
                grad_sign -= g_col * sgn

            # STE through sign
            grad_ffn_up = grad_sign

            g_up_w = np.zeros_like(uw)
            g_up_b = np.sum(grad_ffn_up, axis=0)
            grad_ln = np.zeros((S, d), dtype=np.float32)
            for o in range(dF):
                sgn = np.sign(uw[o][np.newaxis, :] - ln_outs[l])
                g_col = grad_ffn_up[:, o:o+1]
                g_up_w[o] -= np.sum(g_col * sgn, axis=0)
                grad_ln -= g_col * sgn

            grads["grad_ffn_down_w"].insert(0, g_dn_w)
            grads["grad_ffn_down_b"].insert(0, g_dn_b)
            grads["grad_ffn_up_w"].insert(0, g_up_w)
            grads["grad_ffn_up_b"].insert(0, g_up_b)

            gamma = _get_param_array(self.ln_gammas[l])
            mean = acts[l].mean(axis=-1, keepdims=True)
            var = acts[l].var(axis=-1, keepdims=True)
            inv_std = 1.0 / np.sqrt(var + 1e-5)
            x_hat = (acts[l] - mean) * inv_std

            g_ln_gamma = np.sum(grad_ln * x_hat, axis=0)
            g_ln_beta = np.sum(grad_ln, axis=0)
            grads["grad_ln_gamma"].insert(0, g_ln_gamma)
            grads["grad_ln_beta"].insert(0, g_ln_beta)

            dl_dxhat = grad_ln * gamma
            sum1 = dl_dxhat.sum(axis=-1, keepdims=True)
            sum2 = (dl_dxhat * x_hat).sum(axis=-1, keepdims=True)
            grad_x_ln = inv_std * (dl_dxhat - sum1 / d - x_hat * sum2 / d)

            dx = dx + grad_x_ln

        grad_embed = np.zeros((V, d), dtype=np.float32)
        grad_pos = np.zeros((self.max_seq, d), dtype=np.float32)
        for s in range(S):
            tok = int(ids[s])
            if 0 <= tok < V:
                grad_embed[tok] += dx[s]
            grad_pos[s] = dx[s]

        grads["grad_embed"] = grad_embed
        grads["grad_pos"] = grad_pos

        self._scatter_grads(grads)
        return grads

    # ── Optimizer step ───────────────────────────────────────────────────

    def zero_grad(self):
        """Reset all parameter gradients to zero."""
        for p in self.parameters():
            if hasattr(p, "grad"):
                p.grad = None

    def step(self, grads: dict = None, lr: float = 1e-3):
        """Simple SGD step using stored .grad or explicit grads dict.

        After stepping, re-uploads weights to GPU.
        """
        if grads is not None:
            self._scatter_grads(grads)

        for p in self.parameters():
            arr = _get_param_array(p)
            if hasattr(p, "grad") and p.grad is not None:
                arr -= lr * p.grad

        self._sync_weights()

    def __repr__(self):
        return (
            f"VsaLmModel(vocab={self.vocab}, d_model={self.d_model}, "
            f"d_ffn={self.d_ffn}, max_seq={self.max_seq}, "
            f"n_layers={self.n_layers})"
        )
