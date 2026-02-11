"""
Shared fixtures for experimental tests.
"""

import numpy as np
import pytest

# =============================================================================
# Common Constants
# =============================================================================

DEFAULT_DIM = 1024
LARGE_DIM = 4096
SMALL_DIM = 256


# =============================================================================
# Random State Fixtures
# =============================================================================


@pytest.fixture
def random_seed() -> int:
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def rng(random_seed: int) -> np.random.Generator:
    """Numpy random generator with fixed seed."""
    return np.random.default_rng(random_seed)


# =============================================================================
# Dimension Fixtures
# =============================================================================


@pytest.fixture
def dim() -> int:
    """Default dimension for hypervectors."""
    return DEFAULT_DIM


@pytest.fixture
def large_dim() -> int:
    """Large dimension for stress tests."""
    return LARGE_DIM


@pytest.fixture
def small_dim() -> int:
    """Small dimension for quick tests."""
    return SMALL_DIM


# =============================================================================
# Vector Generation Fixtures
# =============================================================================


@pytest.fixture
def random_bipolar_vector(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random bipolar (+1/-1) vector."""
    return np.sign(rng.standard_normal(dim)).astype(np.float32)


@pytest.fixture
def random_unit_vector(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random unit vector."""
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def zero_vector(dim: int) -> np.ndarray:
    """Generate a zero vector."""
    return np.zeros(dim, dtype=np.float32)


@pytest.fixture
def ones_vector(dim: int) -> np.ndarray:
    """Generate a vector of all ones."""
    return np.ones(dim, dtype=np.float32)


# =============================================================================
# Vector Set Fixtures
# =============================================================================


@pytest.fixture
def bipolar_vector_pair(dim: int, rng: np.random.Generator) -> tuple:
    """Generate two independent random bipolar vectors."""
    a = np.sign(rng.standard_normal(dim)).astype(np.float32)
    b = np.sign(rng.standard_normal(dim)).astype(np.float32)
    return a, b


@pytest.fixture
def bipolar_vector_triple(dim: int, rng: np.random.Generator) -> tuple:
    """Generate three independent random bipolar vectors."""
    a = np.sign(rng.standard_normal(dim)).astype(np.float32)
    b = np.sign(rng.standard_normal(dim)).astype(np.float32)
    c = np.sign(rng.standard_normal(dim)).astype(np.float32)
    return a, b, c


@pytest.fixture
def unit_vector_pair(dim: int, rng: np.random.Generator) -> tuple:
    """Generate two independent random unit vectors."""
    a = rng.standard_normal(dim).astype(np.float32)
    a = a / np.linalg.norm(a)
    b = rng.standard_normal(dim).astype(np.float32)
    b = b / np.linalg.norm(b)
    return a, b


@pytest.fixture
def unit_vector_triple(dim: int, rng: np.random.Generator) -> tuple:
    """Generate three independent random unit vectors."""
    a = rng.standard_normal(dim).astype(np.float32)
    a = a / np.linalg.norm(a)
    b = rng.standard_normal(dim).astype(np.float32)
    b = b / np.linalg.norm(b)
    c = rng.standard_normal(dim).astype(np.float32)
    c = c / np.linalg.norm(c)
    return a, b, c


# =============================================================================
# Codebook Fixtures
# =============================================================================


@pytest.fixture
def small_codebook(small_dim: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a small codebook of bipolar vectors."""
    num_items = 10
    codebook = np.sign(rng.standard_normal((num_items, small_dim))).astype(np.float32)
    return codebook


@pytest.fixture
def medium_codebook(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a medium codebook of bipolar vectors."""
    num_items = 50
    codebook = np.sign(rng.standard_normal((num_items, dim))).astype(np.float32)
    return codebook


# =============================================================================
# Tolerance Fixtures
# =============================================================================


@pytest.fixture
def similarity_tolerance() -> float:
    """Tolerance for similarity comparisons."""
    return 0.1


@pytest.fixture
def strict_tolerance() -> float:
    """Strict tolerance for exact comparisons."""
    return 1e-5


@pytest.fixture
def orthogonality_threshold() -> float:
    """Threshold below which vectors are considered nearly orthogonal."""
    return 0.1


# =============================================================================
# GPU Availability
# =============================================================================


@pytest.fixture
def gpu_available() -> bool:
    """Check if GPU is available for testing."""
    try:
        from grilly.backend.experimental.vsa import VulkanVSA

        return True
    except ImportError:
        return False


# =============================================================================
# Markers
# =============================================================================


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
