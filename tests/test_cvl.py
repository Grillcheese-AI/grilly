"""Random Fourier Features + TD-free Contrastive Value Learning readout (backend/cvl.py)."""
import numpy as np

from backend.cvl import CVLValue, RandomFourierFeatures


def _unit(v):
    return v / np.linalg.norm(v)


def test_rff_self_kernel_is_e():
    # F(x)·F(x) ≈ e for a unit vector (Rahimi & Recht / CVL Lemma 1).
    rff = RandomFourierFeatures(32, 16384, seed=3)
    x = _unit(np.random.default_rng(0).standard_normal(32))
    assert abs(float(rff(x) @ rff(x)) - np.e) < 0.2


def test_rff_approximates_exp_inner_product():
    # E[F(x)·F(y)] ≈ e^{x·y} for unit x, y.
    rng = np.random.default_rng(1)
    rff = RandomFourierFeatures(32, 16384, seed=3)
    errs = []
    for _ in range(6):
        x, y = _unit(rng.standard_normal(32)), _unit(rng.standard_normal(32))
        true = float(np.exp(x @ y))
        errs.append(abs(float(rff(x) @ rff(y)) - true) / true)
    assert np.mean(errs) < 0.1


def test_cvl_ranks_by_reward_weighted_future():
    # TD-free: a state whose future occupancy carries +reward is worth more than one carrying -reward.
    rng = np.random.default_rng(2)
    rff = RandomFourierFeatures(32, 8192, seed=5)
    fa, fb = rff(_unit(rng.standard_normal(32))), rff(_unit(rng.standard_normal(32)))
    cvl = CVLValue(gamma=0.95, beta=0.9)
    cvl.observe(fa, +1.0)
    cvl.observe(fb, -1.0)
    assert cvl.value(fa) > cvl.value(fb)


def test_cvl_empty_is_zero():
    assert CVLValue().value(np.ones(8)) == 0.0
