"""Tests for utils.stable_hash module."""

import numpy as np
import pytest

try:
    from grilly.utils.stable_hash import (
        bipolar_from_key,
        digest,
        stable_bytes,
        stable_u32,
        stable_u64,
        using_blake3,
    )
except ImportError:
    pytest.skip("grilly not available", allow_module_level=True)


class TestStableHash:
    """Tests for stable hashing functions."""

    def test_using_blake3(self):
        """using_blake3 returns bool."""
        assert isinstance(using_blake3(), bool)

    def test_stable_u32_deterministic(self):
        """stable_u32 is deterministic for same inputs."""
        a = stable_u32("hello", "world", domain="test")
        b = stable_u32("hello", "world", domain="test")
        assert a == b
        assert 0 <= a < 2**32

    def test_stable_u32_different_inputs(self):
        """stable_u32 differs for different inputs."""
        a = stable_u32("hello", domain="test")
        b = stable_u32("world", domain="test")
        assert a != b

    def test_stable_u32_different_domains(self):
        """stable_u32 differs for different domains."""
        a = stable_u32("hello", domain="domain_a")
        b = stable_u32("hello", domain="domain_b")
        assert a != b

    def test_stable_u32_parts_types(self):
        """stable_u32 accepts str, int, float, bytes."""
        stable_u32("s")
        stable_u32(42)
        stable_u32(3.14)
        stable_u32(b"bytes")

    def test_stable_u64_deterministic(self):
        """stable_u64 is deterministic."""
        a = stable_u64("key", domain="test")
        b = stable_u64("key", domain="test")
        assert a == b
        assert 0 <= a < 2**64

    def test_stable_bytes(self):
        """stable_bytes returns bytes of requested length."""
        out = stable_bytes("hello", out_len=16, domain="test")
        assert isinstance(out, bytes)
        assert len(out) == 16

    def test_stable_bytes_long_output(self):
        """stable_bytes supports out_len > 32 (BLAKE2s chain)."""
        out = stable_bytes("key", out_len=64, domain="test")
        assert len(out) == 64

    def test_digest_basic(self):
        """digest returns bytes."""
        d = digest(("a", "b"), domain="test", out_len=32)
        assert isinstance(d, bytes)
        assert len(d) == 32

    def test_digest_long_output(self):
        """digest with out_len > 32 chains hashes."""
        d = digest(("x",), domain="test", out_len=48)
        assert len(d) == 48


class TestBipolarFromKey:
    """Tests for bipolar_from_key."""

    def test_bipolar_deterministic(self):
        """bipolar_from_key is deterministic."""
        v1 = bipolar_from_key("realm:test", 64)
        v2 = bipolar_from_key("realm:test", 64)
        assert np.array_equal(v1, v2)
        assert v1.shape == (64,)

    def test_bipolar_values(self):
        """bipolar_from_key produces only +1 and -1."""
        v = bipolar_from_key("key", 128)
        assert set(np.unique(v)).issubset({-1.0, 1.0})
        assert v.dtype == np.float32

    def test_bipolar_dim_zero(self):
        """bipolar_from_key with dim=0 returns empty array."""
        v = bipolar_from_key("key", 0)
        assert v.shape == (0,)
        assert v.dtype == np.float32

    def test_bipolar_dim_one(self):
        """bipolar_from_key with dim=1 works."""
        v = bipolar_from_key("key", 1)
        assert v.shape == (1,)
        assert v[0] in (-1.0, 1.0)

    def test_bipolar_different_keys(self):
        """Different keys produce different vectors."""
        v1 = bipolar_from_key("key1", 32)
        v2 = bipolar_from_key("key2", 32)
        assert not np.array_equal(v1, v2)

    def test_bipolar_custom_domain(self):
        """bipolar_from_key accepts custom domain."""
        v = bipolar_from_key("k", 16, domain="custom.bipolar")
        assert v.shape == (16,)
