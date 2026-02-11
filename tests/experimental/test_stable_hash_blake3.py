import numpy as np


def test_hash_to_bipolar_is_deterministic():
    try:
        from grilly.experimental.vsa.ops import BinaryOps
    except ModuleNotFoundError:
        from experimental.vsa.ops import BinaryOps

    v1 = BinaryOps.hash_to_bipolar("realm:technology", 128)
    v2 = BinaryOps.hash_to_bipolar("realm:technology", 128)
    assert v1.shape == (128,)
    assert np.array_equal(v1, v2)
    assert set(np.unique(v1)).issubset({-1.0, 1.0})


def test_stable_hash_seed_exists():
    try:
        from utils.stable_hash import stable_u32, stable_u64
    except ModuleNotFoundError:
        from grilly.utils.stable_hash import stable_u32, stable_u64  # type: ignore

    a = stable_u32("hello", domain="test")
    b = stable_u32("hello", domain="test")
    assert a == b
    assert isinstance(stable_u64("hello", domain="test"), int)
