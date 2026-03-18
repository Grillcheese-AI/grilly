import numpy as np
import sys

sys.path.insert(0, ".")


def test_q_similarity_identical_queries():
    """Identical consecutive queries should give q_sim approx 1.0"""
    from backend._bridge import q_similarity

    q = np.ones((2, 8, 64), dtype=np.float32)  # batch=2, seq=8, dim=64
    sims = q_similarity(q)
    np.testing.assert_allclose(sims, 1.0, atol=1e-5)


def test_q_similarity_random_queries():
    """Random queries should give q_sim between -1 and 1"""
    from backend._bridge import q_similarity

    np.random.seed(42)
    q = np.random.randn(4, 16, 32).astype(np.float32)
    sims = q_similarity(q)
    assert sims.shape == (4,)
    assert np.all(sims >= -1.0) and np.all(sims <= 1.0)


def test_q_similarity_4d():
    """Should work with (batch, heads, seq, dim) shape"""
    from backend._bridge import q_similarity

    np.random.seed(42)
    q = np.random.randn(2, 8, 16, 64).astype(np.float32)
    sims = q_similarity(q)
    assert sims.shape == (2, 8)


def test_q_similarity_orthogonal():
    """Orthogonal consecutive queries should give q_sim approx 0"""
    from backend._bridge import q_similarity

    q = np.zeros((1, 4, 4), dtype=np.float32)
    q[0, 0, 0] = 1.0  # [1,0,0,0]
    q[0, 1, 1] = 1.0  # [0,1,0,0]
    q[0, 2, 2] = 1.0  # [0,0,1,0]
    q[0, 3, 3] = 1.0  # [0,0,0,1]
    sims = q_similarity(q)
    np.testing.assert_allclose(sims, 0.0, atol=1e-5)
